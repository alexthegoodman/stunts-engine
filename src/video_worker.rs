use js_sys::{Function, Object, Reflect};
use std::rc::Rc;
use std::{cell::RefCell, io::BufReader};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::console::{error, log};
use web_sys::{
    Blob, HtmlCanvasElement, ImageBitmap, VideoDecoder, VideoDecoderConfig, VideoDecoderInit,
    VideoFrame, VideoFrameInit, Window,
};

// use crate::mp4box::MP4Demuxer;

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

use mp4::{Mp4Config, Mp4Reader, TrackType};
use std::io::Cursor;

struct Mp4Demuxer {
    mp4: Mp4Reader<BufReader<Cursor<Vec<u8>>>>,
}

impl Mp4Demuxer {
    pub async fn new(data: Vec<u8>) -> Result<Self, JsValue> {
        let size = data.len();
        let mp4_data = BufReader::new(Cursor::new(data));
        let mut mp4: Mp4Reader<BufReader<Cursor<Vec<u8>>>> =
            Mp4Reader::read_header(mp4_data, size as u64)
                .map_err(|e| JsValue::from_str(&e.to_string()))?; // Read once!

        Ok(Mp4Demuxer { mp4 })
    }

    pub async fn demux(
        &mut self,
        decoder: VideoDecoder,
        frame_manager: Rc<RefCell<FrameManager>>,
        window: Window, // Add window parameter
    ) -> Result<(), JsValue> {
        let track = self
            .mp4
            .tracks()
            .iter()
            .find(|t| t.1.track_type().expect("track type") == TrackType::Video)
            .ok_or_else(|| JsValue::from_str("No video track found in MP4"))?;

        let track_id = track.1.track_id();
        let timescale = track.1.timescale();

        let sample_count = self.mp4.sample_count(track_id).unwrap();

        for sample_idx in 0..sample_count {
            let sample_id = sample_idx + 1;
            let sample = self.mp4.read_sample(track_id, sample_id);

            if let Some(samp) = sample.unwrap() {
                let js_array = js_sys::Array::new();
                let byte_array = js_sys::Uint8Array::from(&samp.bytes[..]);
                js_array.push(&byte_array);

                let blob_parts = js_array.into(); // Convert to JsValue

                let blob = Blob::new_with_u8_array_sequence(&blob_parts)?; // Use correct Blob constructor

                let image_bitmap_promise = window.create_image_bitmap_with_blob(&blob)?;
                let image_bitmap_future = JsFuture::from(image_bitmap_promise);
                let image_bitmap_result = image_bitmap_future.await;

                match image_bitmap_result {
                    Ok(image_bitmap) => {
                        let image_bitmap: ImageBitmap = image_bitmap.dyn_into().unwrap();

                        let init = VideoFrameInit::new(); // Empty init
                        let video_frame = VideoFrame::new_with_image_bitmap_and_video_frame_init(
                            &image_bitmap,
                            &init,
                        )?;

                        let timestamp_ms = (samp.start_time as f64 / timescale as f64) * 1000.0;
                        Reflect::set(
                            &video_frame,
                            &"timestamp".into(),
                            &JsValue::from_f64(timestamp_ms),
                        )?;

                        frame_manager.borrow_mut().render_frame(video_frame)?;
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn read_sample(
        &mut self,
        track_id: u32,
        sample_id: u32,
    ) -> Result<Option<mp4::Mp4Sample>, mp4::Error> {
        self.mp4.read_sample(track_id, sample_id)
    }

    pub fn timescale(&self, track_id: u32) -> Option<u32> {
        self.mp4.tracks().get(&track_id).map(|t| t.timescale())
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

// #[wasm_bindgen]
// pub struct VideoWorker {
//     // status: Rc<RefCell<Status>>,
//     frame_manager: Rc<RefCell<FrameManager>>,
//     decoder: VideoDecoder,
// }

// #[wasm_bindgen]
// impl VideoWorker {
//     #[wasm_bindgen(constructor)]
//     pub async fn new(canvas: HtmlCanvasElement) -> Result<VideoWorker, JsValue> {
//         // let status = Rc::new(RefCell::new(Status::new()));
//         let renderer = Box::new(WebGPURenderer::new(canvas).await?);
//         let frame_manager = Rc::new(RefCell::new(FrameManager::new(renderer)));

//         // Create the context object for the functions
//         let context = Object::new();
//         let status_js = Object::new();
//         let frame_manager_js = Object::new();

//         Reflect::set(
//             &context,
//             &"window".into(),
//             &web_sys::window().unwrap().into(),
//         )?;

//         {
//             let frame_manager = Rc::clone(&frame_manager); // Clone for use in closure

//             let render_frame_fn = Closure::wrap(Box::new(move |frame: VideoFrame| {
//                 let mut frame_manager_borrow = frame_manager.borrow_mut(); // Borrow mutably
//                 frame_manager_borrow.render_frame(frame);
//             }) as Box<dyn FnMut(VideoFrame)>);

//             Reflect::set(
//                 &frame_manager_js,
//                 &"render_frame".into(),
//                 render_frame_fn.as_ref().unchecked_ref(),
//             )?;
//             render_frame_fn.forget();
//         }

//         // Create output handler function
//         let output_fn = Function::new_with_args(
//             "frame",
//             r#"
//             this.frameManager.borrow_mut().render_frame(frame);
//             "#,
//         );
//         output_fn.bind(&context);

//         // Create error handler function
//         let error_fn =
//             Function::new_with_args("e", "this.status.borrow_mut().set_status('decode', e);");
//         error_fn.bind(&context);

//         // Initialize VideoDecoder
//         let decoder_init = VideoDecoderInit::new(&error_fn, &output_fn);
//         let decoder = VideoDecoder::new(&decoder_init)?;

//         Ok(VideoWorker {
//             // status,
//             frame_manager,
//             decoder,
//         })
//     }

//     pub fn start(&self, data_uri: &str) -> Result<(), JsValue> {
//         let decoder = self.decoder.clone();

//         // Create the config callback
//         let on_config = Function::new_with_args(
//             "config",
//             "this.status.borrow_mut().set_status('decode',
//              `${config.codec} @ ${config.codedWidth}x${config.codedHeight}`);
//              this.decoder.configure(config);",
//         );

//         // Create the chunk callback
//         let on_chunk = Function::new_with_args("chunk", "this.decoder.decode(chunk);");

//         // Bind this context for the callbacks
//         let this_obj = Object::new();
//         Reflect::set(&this_obj, &"decoder".into(), &self.decoder)?;

//         on_config.bind(&this_obj);
//         on_chunk.bind(&this_obj);

//         let demuxer = MP4Demuxer::new(data_uri, on_config, on_chunk);

//         Ok(())
//     }
// }

struct FrameData {
    data: Vec<u8>,
    width: u32,
    height: u32,
    timestamp: f64,
}

impl FrameData {
    fn new(frame: &VideoFrame, data: Vec<u8>) -> Result<Self, JsValue> {
        let width = frame.coded_width();
        let height = frame.coded_height();
        let timestamp = frame.timestamp().expect("Couldn't get timestamp");

        Ok(FrameData {
            data,
            width,
            height,
            timestamp,
        })
    }
}

// ... (FrameManager remains the same)

// #[wasm_bindgen]
// pub struct VideoWorker {
//     frame_manager: Rc<RefCell<FrameManager>>,
//     decoder: VideoDecoder,
//     mp4_demuxer: Rc<RefCell<Mp4Demuxer>>, // Add MP4Demuxer
//     window: Window,                       // Store Window
// }

// #[wasm_bindgen]
// impl VideoWorker {
//     #[wasm_bindgen(constructor)]
//     pub async fn new(canvas: HtmlCanvasElement) -> Result<VideoWorker, JsValue> {
//         let renderer = Box::new(WebGPURenderer::new(canvas).await?);
//         let frame_manager = Rc::new(RefCell::new(FrameManager::new(renderer)));
//         let window = web_sys::window().unwrap(); // Get Window

//         let decoder_init = VideoDecoderInit::new(
//             &Closure::new(Box::new(|e: JsValue| {
//                 // Handle error (e.g., console.error)
//                 // error(&format!("Decoder error: {:?}", e));
//             }) as Box<dyn FnMut(JsValue)>)
//             .into_js_value()
//             .unchecked_into(),
//             &Closure::new(Box::new({
//                 let frame_manager = Rc::clone(&frame_manager);

//                 move |frame: VideoFrame| {
//                     // let frame_manager = Rc::clone(&frame_manager);

//                     // This is where you would now create FrameData and call render_frame
//                     let frame_data = FrameData::new(&frame, Vec::new()).unwrap(); // Assuming data is not needed.
//                     frame_manager.borrow_mut().render_frame(frame);
//                 }
//             }) as Box<dyn FnMut(VideoFrame)>)
//             .into_js_value()
//             .unchecked_into(),
//         );

//         let decoder = VideoDecoder::new(&decoder_init)?;
//         let mp4_demuxer = Rc::new(RefCell::new(Mp4Demuxer::new().await?)); // Initialize MP4Demuxer

//         Ok(VideoWorker {
//             frame_manager,
//             decoder,
//             mp4_demuxer,
//             window,
//         })
//     }

//     pub fn start(&self, data: &[u8]) -> Result<(), JsValue> {
//         let data_vec = data.to_vec(); // Convert &[u8] to Vec<u8>
//         let decoder = self.decoder.clone();
//         let frame_manager = self.frame_manager.clone();
//         let window = self.window.clone();
//         let mut demuxer = self.mp4_demuxer.borrow_mut();

//         wasm_bindgen_futures::spawn_local(async move {
//             if let Err(e) = demuxer
//                 .demux(data_vec, decoder, frame_manager, window)
//                 .await
//             {
//                 // error(&format!("Demuxing error: {:?}", e));
//             }
//         });

//         Ok(())
//     }
// }

#[wasm_bindgen]
pub struct VideoWorker {
    frame_manager: Rc<RefCell<FrameManager>>,
    decoder: VideoDecoder,
    mp4_demuxer: Rc<RefCell<Mp4Demuxer>>,
    window: Window,
    current_sample: Option<(u32, u32)>, // Track current sample and track ID
}

#[wasm_bindgen]
impl VideoWorker {
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement, data: Vec<u8>) -> Result<VideoWorker, JsValue> {
        let renderer = Box::new(WebGPURenderer::new(canvas).await?);
        let frame_manager = Rc::new(RefCell::new(FrameManager::new(renderer)));
        let window = web_sys::window().unwrap(); // Get Window

        let decoder_init = VideoDecoderInit::new(
            &Closure::new(Box::new(|e: JsValue| {
                // Handle error (e.g., console.error)
                // error(&format!("Decoder error: {:?}", e));
            }) as Box<dyn FnMut(JsValue)>)
            .into_js_value()
            .unchecked_into(),
            &Closure::new(Box::new({
                let frame_manager = Rc::clone(&frame_manager);

                move |frame: VideoFrame| {
                    // let frame_manager = Rc::clone(&frame_manager);

                    // This is where you would now create FrameData and call render_frame
                    let frame_data = FrameData::new(&frame, Vec::new()).unwrap(); // Assuming data is not needed.
                    frame_manager.borrow_mut().render_frame(frame);
                }
            }) as Box<dyn FnMut(VideoFrame)>)
            .into_js_value()
            .unchecked_into(),
        );

        let decoder = VideoDecoder::new(&decoder_init)?;
        let mp4_demuxer = Rc::new(RefCell::new(Mp4Demuxer::new(data).await?)); // Initialize MP4Demuxer

        Ok(VideoWorker {
            frame_manager,
            decoder,
            mp4_demuxer,
            window,
            current_sample: None, // Initialize current_sample
        })
    }

    pub fn load(&mut self, data: &[u8]) -> Result<(), JsValue> {
        // Renamed to load
        let data_vec = data.to_vec();
        let mut demuxer = self.mp4_demuxer.borrow_mut();

        // Parse MP4 header once during load
        let size = data_vec.len();
        let mp4_data = BufReader::new(Cursor::new(data_vec));
        let mp4: Mp4Reader<BufReader<Cursor<Vec<u8>>>> =
            Mp4Reader::read_header(mp4_data, size as u64)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let track = mp4
            .tracks()
            .iter()
            .find(|t| t.1.track_type().expect("track type") == TrackType::Video)
            .ok_or_else(|| JsValue::from_str("No video track found in MP4"))?;

        let track_id = track.1.track_id();
        let sample_count = mp4.sample_count(track_id).unwrap();

        // Store track ID and total sample count.  Start at sample 1.
        self.current_sample = Some((track_id, 1));
        Ok(())
    }

    pub fn draw_frame(&mut self) -> Result<(), JsValue> {
        let decoder = self.decoder.clone();
        let frame_manager = self.frame_manager.clone();
        let window = self.window.clone();
        let mut demuxer = self.mp4_demuxer.borrow_mut();

        if let Some((track_id, sample_id)) = self.current_sample {
            let sample = self
                .mp4_demuxer
                .borrow_mut()
                .read_sample(track_id, sample_id);
            if let Some(samp) = sample.unwrap() {
                let js_array = js_sys::Array::new();
                let byte_array = js_sys::Uint8Array::from(&samp.bytes[..]);
                js_array.push(&byte_array);

                let blob_parts = js_array.into();
                let blob = Blob::new_with_u8_array_sequence(&blob_parts)?;

                let image_bitmap_promise = window.create_image_bitmap_with_blob(&blob)?;
                let image_bitmap_future = JsFuture::from(image_bitmap_promise);

                wasm_bindgen_futures::spawn_local(async move {
                    let image_bitmap_result = image_bitmap_future.await;

                    match image_bitmap_result {
                        Ok(image_bitmap) => {
                            let image_bitmap: ImageBitmap = image_bitmap.dyn_into().unwrap();
                            let init = VideoFrameInit::new();
                            let video_frame =
                                VideoFrame::new_with_image_bitmap_and_video_frame_init(
                                    &image_bitmap,
                                    &init,
                                )
                                .expect("Couldn't create frame");

                            // let timestamp_ms = (samp.start_time as f64
                            //     / demuxer.timescale(track_id).unwrap() as f64)
                            //     * 1000.0; // Get timescale
                            // Reflect::set(
                            //     &video_frame,
                            //     &"timestamp".into(),
                            //     &JsValue::from_f64(timestamp_ms),
                            // )
                            // .expect("Couldn't set timestamp");

                            frame_manager
                                .borrow_mut()
                                .render_frame(video_frame)
                                .expect("Couldn't render frame");
                        }
                        Err(e) => {
                            // console_error(&format!("ImageBitmap error: {:?}", e));
                        }
                    }
                });

                // Increment sample ID for next frame
                self.current_sample = Some((track_id, sample_id + 1));
            } else {
                // Handle end of stream or error reading sample
                self.current_sample = None; // Or reset to the beginning if you want looping.
                                            // console_log("End of video stream or error reading sample");
            }
        }
        Ok(())
    }

    // ... (rest of the VideoWorker implementation)
}
