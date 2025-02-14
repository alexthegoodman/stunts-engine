use js_sys::{Function, Object, Reflect};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    HtmlCanvasElement, VideoDecoder, VideoDecoderConfig, VideoDecoderInit, VideoFrame, Window,
};

use crate::mp4box::MP4Demuxer;

// Renderer trait and WebGPU implementation
trait Renderer {
    fn draw(&self, frame_data: &Rc<FrameData>);
}

struct WebGPURenderer {
    canvas: HtmlCanvasElement,
    // Add your wgpu-specific fields here
}

impl WebGPURenderer {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        // Initialize your wgpu pipeline here
        Ok(WebGPURenderer { canvas })
    }
}

impl Renderer for WebGPURenderer {
    fn draw(&self, frame_data: &Rc<FrameData>) { // Takes &Rc<FrameData>
                                                 // Use frame_data.data, frame_data.width, etc.
                                                 // ... your rendering logic ...
                                                 // Example:
                                                 // let data = &frame_data.data;
                                                 // let width = frame_data.width;
                                                 // ...
    }
}

//  Instead of storing VideoFrame directly, store something clonable
//  For example, if you can access the underlying data as a Uint8Array:
struct FrameData {
    data: Vec<u8>, // Or Box<[u8]> for potentially larger frames
    width: u32,
    height: u32,
    timestamp: f64, // Add other necessary metadata
}

impl FrameData {
    fn new(frame: &VideoFrame, data: Vec<u8>) -> Result<Self, JsValue> {
        // Example: Assuming you can access the data as a buffer
        let width = frame.coded_width();
        let height = frame.coded_height();
        let timestamp = frame.timestamp().expect("Couldn't get timestamp");

        // let mut data = vec![0; frame.coded_width() as usize * frame.coded_height() as usize * 4]; // Example RGBA
        // frame.copy_to(data.as_mut_slice()).unwrap(); // Or however you access frame data

        Ok(FrameData {
            data,
            width,
            height,
            timestamp,
        })
    }
}

struct FrameManager {
    renderer: Rc<RefCell<Box<dyn Renderer>>>,
    pending_frame: Option<Rc<FrameData>>, // Store Rc<FrameData>
}

impl FrameManager {
    fn new(renderer: Box<dyn Renderer>) -> Self {
        FrameManager {
            renderer: Rc::new(RefCell::new(renderer)),
            pending_frame: None,
        }
    }

    fn render_frame(&mut self, frame: VideoFrame) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();

        if self.pending_frame.is_none() {
            let renderer = self.renderer.clone();

            // Create FrameData from VideoFrame
            // TODO: send in real data from wgpu
            let frame_data = Rc::new(FrameData::new(&frame, Vec::new())?);

            let frame_data_clone = Rc::clone(&frame_data); // Clone for the closure

            let closure = Closure::wrap(Box::new(move || {
                renderer.borrow().draw(&frame_data_clone); // Pass a reference to FrameData
            }) as Box<dyn FnMut()>);

            window
                .request_animation_frame(closure.as_ref().unchecked_ref())
                .unwrap();
            closure.forget();

            self.pending_frame = Some(frame_data); // Store the Rc<FrameData>
        } else {
            // Close the current pending frame - no longer a VideoFrame
            // If FrameData holds resources that need explicit cleanup, handle that here.
            self.pending_frame.take(); // No close() needed, just drop the Rc
        }

        Ok(())
    }
}

#[wasm_bindgen]
pub struct VideoWorker {
    // status: Rc<RefCell<Status>>,
    frame_manager: Rc<RefCell<FrameManager>>,
    decoder: VideoDecoder,
}

#[wasm_bindgen]
impl VideoWorker {
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<VideoWorker, JsValue> {
        // let status = Rc::new(RefCell::new(Status::new()));
        let renderer = Box::new(WebGPURenderer::new(canvas).await?);
        let frame_manager = Rc::new(RefCell::new(FrameManager::new(renderer)));

        // Create the context object for the functions
        let context = Object::new();
        let status_js = Object::new();
        let frame_manager_js = Object::new();

        Reflect::set(
            &context,
            &"window".into(),
            &web_sys::window().unwrap().into(),
        )?;

        {
            let frame_manager = Rc::clone(&frame_manager); // Clone for use in closure

            let render_frame_fn = Closure::wrap(Box::new(move |frame: VideoFrame| {
                let mut frame_manager_borrow = frame_manager.borrow_mut(); // Borrow mutably
                frame_manager_borrow.render_frame(frame);
            }) as Box<dyn FnMut(VideoFrame)>);

            Reflect::set(
                &frame_manager_js,
                &"render_frame".into(),
                render_frame_fn.as_ref().unchecked_ref(),
            )?;
            render_frame_fn.forget();
        }

        // Create output handler function
        let output_fn = Function::new_with_args(
            "frame",
            r#"
            const status = this.status.borrow_mut();
            if (!status.start_time) {
                status.start_time = this.window.performance.now();
            } else {
                const elapsed = (this.window.performance.now() - status.start_time) / 1000;
                const fps = (++status.frame_count) / elapsed;
                status.set_status('render', `${Math.floor(fps)} fps`);
            }
            this.frameManager.borrow_mut().render_frame(frame);
            "#,
        );
        output_fn.bind(&context);

        // Create error handler function
        let error_fn =
            Function::new_with_args("e", "this.status.borrow_mut().set_status('decode', e);");
        error_fn.bind(&context);

        // Initialize VideoDecoder
        let decoder_init = VideoDecoderInit::new(&error_fn, &output_fn);
        let decoder = VideoDecoder::new(&decoder_init)?;

        Ok(VideoWorker {
            // status,
            frame_manager,
            decoder,
        })
    }

    pub fn start(&self, data_uri: &str) -> Result<(), JsValue> {
        let decoder = self.decoder.clone();

        // Create the config callback
        let on_config = Function::new_with_args(
            "config",
            "this.status.borrow_mut().set_status('decode', 
             `${config.codec} @ ${config.codedWidth}x${config.codedHeight}`);
             this.decoder.configure(config);",
        );

        // Create the chunk callback
        let on_chunk = Function::new_with_args("chunk", "this.decoder.decode(chunk);");

        // Bind this context for the callbacks
        let this_obj = Object::new();
        Reflect::set(&this_obj, &"decoder".into(), &self.decoder)?;

        on_config.bind(&this_obj);
        on_chunk.bind(&this_obj);

        let demuxer = MP4Demuxer::new(data_uri, on_config, on_chunk);

        Ok(())
    }
}
