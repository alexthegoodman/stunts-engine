use crate::{animations::Sequence, editor::WindowSize, timelines::SavedTimelineStateConfig};

use super::{encode::VideoEncoder, frame_buffer::FrameCaptureBuffer, pipeline::ExportPipeline};

struct Exporter {
    video_encoder: VideoEncoder,
}

impl Exporter {
    pub fn new(output_path: &str) -> Self {
        let video_encoder = VideoEncoder::new(output_path).expect("Couldn't get video encoder");
        Exporter { video_encoder }
    }

    pub async fn run(
        &mut self,
        window_size: WindowSize,
        sequences: Vec<Sequence>,
        saved_timeline_state_config: SavedTimelineStateConfig,
        video_width: u32,
        video_height: u32,
        total_duration_s: f64,
    ) {
        let mut wgpu_pipeline = ExportPipeline::new();
        wgpu_pipeline
            .initialize(window_size, sequences, saved_timeline_state_config)
            .await;

        let frame_buffer = FrameCaptureBuffer::new(
            &wgpu_pipeline.device.as_ref().expect("Couldn't get device"),
            video_width,
            video_height,
        );
        wgpu_pipeline.frame_buffer = Some(frame_buffer);

        // Calculate total frames based on sequence duration
        const FPS: f64 = 60.0;
        // let total_duration = sequences.iter()
        //     .map(|seq| seq.duration)
        //     .sum::<f64>();
        let total_frames = (total_duration_s * FPS).ceil() as u32;

        // Frame loop
        for frame_index in 0..total_frames {
            // Calculate current time position
            let current_time = frame_index as f64 / FPS;

            // Render frame
            wgpu_pipeline.render_frame(current_time);

            // Get frame buffer and extract data
            let frame_buffer = wgpu_pipeline
                .frame_buffer
                .as_ref()
                .expect("Couldn't get frame buffer");

            let frame_bytes = frame_buffer
                .get_frame_data(&wgpu_pipeline.device.as_ref().expect("Couldn't get device"))
                .await;

            // Write frame to video
            self.video_encoder
                .write_frame(&frame_bytes)
                .expect("Couldn't write frame");

            // Optional: Add progress reporting
            if frame_index % 60 == 0 {
                println!(
                    "Export progress: {:.1}% ({}/{} frames)",
                    (frame_index as f32 / total_frames as f32) * 100.0,
                    frame_index,
                    total_frames
                );
            }
        }
    }
}
