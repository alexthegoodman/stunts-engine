// NOTE: adapted from https://github.com/w3c/webcodecs/blob/main/samples/video-decode-display/demuxer_mp4.js

use js_sys::{ArrayBuffer, Function, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::VideoDecoderConfig;
use web_sys::{EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType};
use web_sys::{QueuingStrategy, Response, WritableStream};

#[wasm_bindgen]
extern "C" {
    type Mp4BoxFile;

    #[wasm_bindgen(js_namespace = MP4Box, js_name = createFile)]
    fn create_file() -> Mp4BoxFile;

    #[wasm_bindgen(method, js_name = appendBuffer)]
    fn append_buffer(this: &Mp4BoxFile, buffer: &ArrayBuffer);

    #[wasm_bindgen(method, js_name = flush)]
    fn flush(this: &Mp4BoxFile);

    #[wasm_bindgen(method, js_name = onError)]
    fn set_on_error(this: &Mp4BoxFile, callback: &Function);

    #[wasm_bindgen(method, js_name = onReady)]
    fn set_on_ready(this: &Mp4BoxFile, callback: &Function);

    #[wasm_bindgen(method, js_name = onSamples)]
    fn set_on_samples(this: &Mp4BoxFile, callback: &Function);

    #[wasm_bindgen(method, js_name = getTrackById)]
    fn get_track_by_id(this: &Mp4BoxFile, id: u32) -> JsValue;

    #[wasm_bindgen(method, js_name = setExtractionOptions)]
    fn set_extraction_options(this: &Mp4BoxFile, track_id: u32);

    #[wasm_bindgen(method, js_name = start)]
    fn start(this: &Mp4BoxFile);
}

use std::cell::RefCell;
use std::rc::Rc;

fn set_file_start(buffer: &ArrayBuffer, offset: u32) -> Result<(), JsValue> {
    // Convert the ArrayBuffer to a JsValue.
    let buffer_js = JsValue::from(buffer);

    // Set the `fileStart` property on the JsValue.
    Reflect::set(&buffer_js, &"fileStart".into(), &JsValue::from(offset))?;

    Ok(())
}

pub struct MP4FileSink {
    file: Rc<RefCell<Mp4BoxFile>>,
    offset: u32,
    set_status: Function,
}

impl MP4FileSink {
    pub fn new(file: Rc<RefCell<Mp4BoxFile>>, set_status: Function) -> Self {
        MP4FileSink {
            file,
            offset: 0,
            set_status,
        }
    }

    pub fn write(&mut self, chunk: Uint8Array) {
        let buffer = ArrayBuffer::new(chunk.byte_length());
        let uint8_array = Uint8Array::new(&buffer);
        uint8_array.set(&chunk, 0);

        // Set the `fileStart` property on the buffer.
        set_file_start(&buffer, self.offset).unwrap();

        self.offset += buffer.byte_length();

        let status = format!("fetch: {:.1} MiB", self.offset as f64 / (1024.0 * 1024.0));
        let _ = self
            .set_status
            .call1(&JsValue::NULL, &JsValue::from(status));

        self.file.borrow().append_buffer(&buffer);
    }

    pub fn close(&self) {
        let _ = self
            .set_status
            .call1(&JsValue::NULL, &JsValue::from("Done"));
        self.file.borrow().flush();
    }

    pub fn to_underlying_sink(self) -> Result<Object, JsValue> {
        let obj = Object::new();

        // Wrap self in Rc<RefCell> to allow shared ownership.
        let sink = Rc::new(RefCell::new(self));

        // Clone the Rc for the write closure.
        let sink_write = Rc::clone(&sink);
        let write_closure = Closure::wrap(Box::new(move |chunk: Uint8Array| {
            sink_write.borrow_mut().write(chunk);
        }) as Box<dyn FnMut(Uint8Array)>);

        // Clone the Rc for the close closure.
        let sink_close = Rc::clone(&sink);
        let close_closure = Closure::wrap(Box::new(move || {
            sink_close.borrow().close();
        }) as Box<dyn FnMut()>);

        // Attach the closures to the object.
        js_sys::Reflect::set(
            &obj,
            &"write".into(),
            &write_closure.as_ref().unchecked_ref(),
        )?;
        js_sys::Reflect::set(
            &obj,
            &"close".into(),
            &close_closure.as_ref().unchecked_ref(),
        )?;

        // Forget the closures to prevent them from being dropped.
        write_closure.forget();
        close_closure.forget();

        Ok(obj)
    }
}

pub struct MP4Demuxer {
    file: Rc<RefCell<Mp4BoxFile>>,
    // on_config: Function,
    // on_chunk: Function,
    // set_status: Function,
    // Add these fields to store the closures
    _ready_closure: Closure<dyn FnMut(JsValue)>,
    _samples_closure: Closure<dyn FnMut(u32, JsValue, js_sys::Array)>,
    _fetch_closure: Closure<dyn FnMut(JsValue)>,
}

impl MP4Demuxer {
    pub fn new(uri: &str, on_config: Function, on_chunk: Function, set_status: Function) -> Self {
        let file = create_file();
        let file = Rc::new(RefCell::new(file));
        file.borrow().set_on_error(&set_status);

        // Create file_sink
        let file_sink = MP4FileSink::new(file.clone(), set_status.clone());

        // Rest of the code using file.borrow() to access Mp4BoxFile methods
        let ready_closure = Closure::wrap(Box::new({
            let file = file.clone();
            move |info: JsValue| {
                Self::on_ready(&file.borrow(), &info, &on_config, &set_status).unwrap();
            }
        }) as Box<dyn FnMut(JsValue)>);

        // file.set_on_error(&set_status);

        // Create file_sink first
        // let file_sink = MP4FileSink::new(&file, set_status.clone());
        let underlying_sink = file_sink.to_underlying_sink().unwrap();
        let mut strategy = QueuingStrategy::new();
        strategy.set_high_water_mark(2.0);
        let writable_stream =
            WritableStream::new_with_underlying_sink_and_strategy(&underlying_sink, &strategy)
                .unwrap();

        // Then create closures
        // let ready_closure = Closure::wrap(Box::new(move |info: JsValue| {
        //     Self::on_ready(&file, &info, &on_config, &set_status).unwrap();
        // }) as Box<dyn FnMut(JsValue)>);

        let samples_closure = Closure::wrap(Box::new(
            move |track_id: u32, reff: JsValue, samples: js_sys::Array| {
                Self::on_samples(&on_chunk, track_id, &reff, &samples).unwrap();
            },
        )
            as Box<dyn FnMut(u32, JsValue, js_sys::Array)>);

        file.borrow()
            .set_on_ready(ready_closure.as_ref().unchecked_ref());
        file.borrow()
            .set_on_samples(samples_closure.as_ref().unchecked_ref());

        let fetch_closure = Closure::wrap(Box::new(move |response: JsValue| {
            let response: web_sys::Response = response.dyn_into().unwrap();
            let body = response.body().unwrap();
            let _ = body.pipe_to(&writable_stream);
        }) as Box<dyn FnMut(JsValue)>);

        let promise = web_sys::window()
            .unwrap()
            .fetch_with_str(uri)
            .then(&fetch_closure);

        MP4Demuxer {
            file,
            // on_config: on_config.clone(),
            // on_chunk: on_chunk.clone(),
            // set_status: set_status.clone(),
            _ready_closure: ready_closure,
            _samples_closure: samples_closure,
            _fetch_closure: fetch_closure,
        }
    }

    fn description(file: &Mp4BoxFile, track_id: u32) -> Result<Uint8Array, JsValue> {
        // Get the track by ID.
        let track = file.get_track_by_id(track_id);

        // Access the stsd entries.
        let stsd_entries = Reflect::get(&track, &"mdia.minf.stbl.stsd.entries".into())?
            .dyn_into::<js_sys::Array>()?;

        // Iterate through the entries to find the codec box.
        for entry in stsd_entries.iter() {
            let entry = entry.dyn_into::<Object>()?;
            let boxx = Reflect::get(&entry, &"avcC".into())
                .or_else(|_| Reflect::get(&entry, &"hvcC".into()))
                .or_else(|_| Reflect::get(&entry, &"vpcC".into()))
                .or_else(|_| Reflect::get(&entry, &"av1C".into()))?;

            if !boxx.is_undefined() {
                // Create a DataStream and write the box to it.
                let stream = js_sys::Function::new_no_args("DataStream")
                    .call1(&JsValue::NULL, &JsValue::UNDEFINED)?
                    .dyn_into::<Object>()?;

                // Convert boxx to Function before applying
                let boxx_fn = boxx.dyn_into::<Function>()?;
                Reflect::apply(&boxx_fn, &stream, &js_sys::Array::new())?;

                // Extract the description (skip the first 8 bytes).
                let buffer = Reflect::get(&stream, &"buffer".into())?.dyn_into::<ArrayBuffer>()?;
                return Ok(Uint8Array::new(&buffer).subarray(8, buffer.byte_length()));
            }
        }

        Err(JsValue::from_str("avcC, hvcC, vpcC, or av1C box not found"))
    }

    fn on_ready(
        file: &Mp4BoxFile,
        info: &JsValue,
        on_config: &Function,
        set_status: &Function,
    ) -> Result<(), JsValue> {
        set_status.call1(&JsValue::NULL, &JsValue::from_str("Ready"))?;

        // Get the first video track.
        let video_tracks =
            Reflect::get(info, &"videoTracks".into())?.dyn_into::<js_sys::Array>()?;
        let track = video_tracks.get(0).dyn_into::<Object>()?;

        // Extract track details.
        let codec = Reflect::get(&track, &"codec".into())?.as_string().unwrap();
        let coded_height = Reflect::get(&track, &"video.height".into())?
            .as_f64()
            .unwrap() as u32;
        let coded_width = Reflect::get(&track, &"video.width".into())?
            .as_f64()
            .unwrap() as u32;
        let track_id = Reflect::get(&track, &"id".into())?.as_f64().unwrap() as u32;

        // Generate the VideoDecoderConfig.
        let config = VideoDecoderConfig::new(if codec.starts_with("vp08") {
            "vp8"
        } else {
            &codec
        });

        config.set_coded_height(coded_height);
        config.set_coded_width(coded_width);

        // Handle the Result and convert Uint8Array to Object
        let description = Self::description(file, track_id)?;
        config.set_description(description.unchecked_ref());

        // Emit the config.
        on_config.call1(&JsValue::NULL, &config)?;

        // Start demuxing.
        file.set_extraction_options(track_id);
        file.start();

        Ok(())
    }

    fn on_samples(
        on_chunk: &Function,
        track_id: u32,
        reff: &JsValue,
        samples: &js_sys::Array,
    ) -> Result<(), JsValue> {
        for sample in samples.iter() {
            let sample = sample.dyn_into::<Object>()?;

            // Extract sample details.
            let is_sync = Reflect::get(&sample, &"is_sync".into())?.as_bool().unwrap();
            let cts = Reflect::get(&sample, &"cts".into())?.as_f64().unwrap();
            let timescale = Reflect::get(&sample, &"timescale".into())?
                .as_f64()
                .unwrap();
            let duration = Reflect::get(&sample, &"duration".into())?.as_f64().unwrap();
            let data = Reflect::get(&sample, &"data".into())?.dyn_into::<Uint8Array>()?;

            // Create an EncodedVideoChunk.
            let initializer = EncodedVideoChunkInit::new(
                &data,
                1e6 * cts / timescale,
                if is_sync {
                    EncodedVideoChunkType::Key
                } else {
                    EncodedVideoChunkType::Delta
                },
            );

            initializer.set_duration(1e6 * duration / timescale);

            let chunk = EncodedVideoChunk::new(
                &initializer, // if is_sync { "key" } else { "delta" },
                              // 1e6 * cts / timescale,
                              // 1e6 * duration / timescale,
                              // &data,
            )?;

            // Emit the chunk.
            on_chunk.call1(&JsValue::NULL, &chunk)?;
        }

        Ok(())
    }
}
