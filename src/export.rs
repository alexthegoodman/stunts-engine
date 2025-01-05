// use mp4::*;
// use rav1e::*;
// use std::fs::File;
// use std::io::Write;

// struct Exporter {
//     // The rav1e encoder context
//     ctx: Context<u16>,
//     // The MP4 writer
//     mp4_writer: Mp4Writer<File>,
//     // The video track ID
//     video_track_id: TrackId,
// }

// impl Exporter {
//     pub fn export() {
//         let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//             backends: wgpu::Backends::all(),
//             dx12_shader_compiler: Default::default(),
//         });

//         let adapter = instance
//             .request_adapter(&wgpu::RequestAdapterOptions {
//                 power_preference: wgpu::PowerPreference::HighPerformance,
//                 force_fallback_adapter: false,
//                 compatible_surface: None, // No surface needed for offscreen
//             })
//             .await
//             .unwrap();

//         let (device, queue) = adapter
//             .request_device(
//                 &wgpu::DeviceDescriptor {
//                     features: wgpu::Features::empty(),
//                     limits: wgpu::Limits::default(),
//                     label: None,
//                 },
//                 None,
//             )
//             .await
//             .unwrap();

//         // Setup rav1e encoder
//         let enc = EncoderConfig {
//             width: video_width as usize,
//             height: video_height as usize,
//             speed_settings: SpeedSettings::from_preset(9),
//             time_base: Rational { num: 1, den: 30 }, // 30 FPS
//             ..Default::default()
//         };

//         let cfg = Config::new().with_encoder_config(enc.clone());
//         let mut ctx: Context<u16> = cfg.new_context().unwrap();

//         // Setup MP4 writer
//         let output_file = File::create("output.mp4")?;
//         let mut mp4_writer = mp4::Mp4Writer::write_start(
//             output_file,
//             &mp4::Mp4Config {
//                 major_brand: "mp42".parse()?,
//                 minor_version: 0,
//                 compatible_brands: vec!["mp42".parse()?, "iso5".parse()?],
//                 timescale: 30000, // This should match your video timing
//             },
//         )?;

//         // Create video track
//         let video_track_id = mp4_writer.add_track(&TrackConfig {
//             track_type: TrackType::Video,
//             timescale: 30000,
//             language: String::from("und"),
//             media_conf: MediaConfig::AV1(AV1Config {
//                 width: video_width as u16,
//                 height: video_height as u16,
//                 sequence_header: vec![], // We'll need to get this from rav1e
//                 profile: 0,
//                 level: 0,
//                 tier: 0,
//                 bit_depth: 8,
//                 monochrome: false,
//                 chroma_subsampling_x: 1,
//                 chroma_subsampling_y: 1,
//                 chroma_sample_position: 0,
//                 initial_presentation_delay: 0,
//             }),
//         })?;

//         // For each frame...
//         // For each frame you want to capture:
//         let buffer = device.create_buffer(&wgpu::BufferDescriptor {
//             usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
//             size: frame_size_in_bytes,
//             label: Some("frame_buffer"),
//         });

//         // Copy your rendered texture to buffer
//         encoder.copy_texture_to_buffer(/* ... */);

//         // Get the pixel data
//         let buffer_slice = buffer.slice(..);
//         buffer_slice.map_async(wgpu::MapMode::Read /* ... */);
//         device.poll(wgpu::Maintain::Wait);
//         // ... await the mapping ...

//         let data = buffer_slice.get_mapped_range();

//         // Create and send frame to rav1e
//         let mut frame = ctx.new_frame();
//         for p in &mut frame.planes {
//             let stride = (enc.width + p.cfg.xdec) >> p.cfg.xdec;
//             // You'll need to convert data from RGBA to the format rav1e expects
//             p.copy_from_raw_u8(&converted_data, stride, 1);
//         }

//         // Send frame to rav1e
//         ctx.send_frame(frame)?;

//         // Receive and write packets
//         while let Ok(packet) = ctx.receive_packet() {
//             mp4_writer.write_sample(
//                 video_track_id,
//                 &mp4::Sample {
//                     start_time: packet.input_frameno as u64 * 1000, // Convert to timescale units
//                     duration: 1000,                                 // Duration in timescale units
//                     rendering_offset: 0,
//                     is_sync: packet.frame_type.is_key(),
//                     bytes: packet.data,
//                 },
//             )?;
//         }

//         // Finalize the MP4 file
//         mp4_writer.write_end()?;

//         // Cleanup wgpu resources
//         drop(data);
//         buffer.unmap();
//         drop(buffer);

//         // When done with all frames:
//         ctx.flush();
//     }
// }
