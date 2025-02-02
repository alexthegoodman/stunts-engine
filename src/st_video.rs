use std::path::Path;

use cgmath::SquareMatrix;
use cgmath::{Matrix4, Vector2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};
use windows::Win32::Foundation::*;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::StructuredStorage::PropVariantToInt64;
use windows_core::PCWSTR;

use crate::camera::Camera;
use crate::editor::{Point, WindowSize};
use crate::polygon::{SavedPoint, INTERNAL_LAYER_SPACE};
use crate::transform::{create_empty_group_transform, matrix4_to_raw_array, Transform};
use crate::vertex::Vertex;

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedStVideoConfig {
    pub id: String,
    pub name: String,
    pub dimensions: (u32, u32),
    pub path: String,
    pub position: SavedPoint,
    pub layer: i32,
}

#[derive(Clone)]
pub struct StVideoConfig {
    pub id: String,
    pub name: String,
    pub dimensions: (u32, u32), // overrides actual image size
    pub position: Point,
    pub path: String,
    pub layer: i32,
}

pub struct StVideo {
    pub id: String,
    pub current_sequence_id: Uuid,
    pub name: String,
    pub path: String,
    pub source_duration: i64,
    pub source_dimensions: (u32, u32),
    pub source_frame_rate: f64,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub transform: Transform,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub dimensions: (u32, u32),
    pub bind_group: wgpu::BindGroup,
    pub vertices: [Vertex; 4],
    pub indices: [u32; 6],
    pub hidden: bool,
    pub layer: i32,
    pub group_bind_group: wgpu::BindGroup,
    #[cfg(target_os = "windows")]
    pub source_reader: IMFSourceReader,
    // #[cfg(target_arch = "wasm32")]
    // pub source_reader: WebCodecs
}

impl StVideo {
    pub fn new(
        device: &Device,
        queue: &Queue,
        path: &Path,
        video_config: StVideoConfig,
        window_size: &WindowSize,
        bind_group_layout: &wgpu::BindGroupLayout,
        group_bind_group_layout: &Arc<wgpu::BindGroupLayout>,
        z_index: f32,
        new_id: String,
        current_sequence_id: Uuid,
    ) -> Result<Self, windows::core::Error> {
        let (source_reader, duration, source_width, source_height, source_frame_rate) =
            Self::initialize_media_source(path)?;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Video Texture"),
            size: wgpu::Extent3d {
                // TODO: should be source video dimensions
                width: source_width,
                height: source_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // format: wgpu::TextureFormat::NV12,
            // use rgb for now
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb], // TODO: check if this is correct
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler with appropriate filtering
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let empty_buffer = Matrix4::<f32>::identity();
        let raw_matrix = matrix4_to_raw_array(&empty_buffer);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Video Uniform Buffer"),
            contents: bytemuck::cast_slice(&raw_matrix),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Rest of the bind group creation...
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Image Bind Group"),
        });

        let mut transform = Transform::new(
            Vector2::new(video_config.position.x, video_config.position.y),
            0.0,
            Vector2::new(
                video_config.dimensions.0 as f32,
                video_config.dimensions.1 as f32,
            ), // Apply scaling here instead of resizing image
            uniform_buffer,
            window_size,
        );

        // -10.0 to provide 10 spots for internal items on top of objects
        transform.layer = video_config.layer as f32 - INTERNAL_LAYER_SPACE as f32;
        transform.update_uniform_buffer(&queue, &window_size);

        let vertices = [
            Vertex {
                position: [-0.5, -0.5, 0.0],
                // tex_coords: [0.0, 1.0],
                tex_coords: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                // tex_coords: [1.0, 1.0],
                tex_coords: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                // tex_coords: [1.0, 0.0],
                tex_coords: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.0],
                // tex_coords: [0.0, 0.0],
                tex_coords: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (tmp_group_bind_group, tmp_group_transform) =
            create_empty_group_transform(device, group_bind_group_layout, window_size);

        Ok(Self {
            id: new_id,
            current_sequence_id,
            name: String::new(),
            path: path
                .to_str()
                .expect("Couldn't convert to string")
                .to_string(),
            source_duration: duration,
            source_dimensions: (source_width, source_height),
            source_frame_rate,
            texture,
            texture_view,
            transform,
            vertex_buffer,
            index_buffer,
            dimensions: video_config.dimensions,
            bind_group,
            vertices,
            indices,
            hidden: false,
            layer: video_config.layer - INTERNAL_LAYER_SPACE,
            source_reader,
            group_bind_group: tmp_group_bind_group,
        })
    }

    #[cfg(target_os = "windows")]
    fn initialize_media_source(
        path: &Path,
    ) -> Result<(IMFSourceReader, i64, u32, u32, f64), windows::core::Error> {
        // Intialize Media Foundation
        unsafe {
            MFStartup(MF_VERSION, MFSTARTUP_FULL)?;
        }

        let source_reader =
            StVideo::create_source_reader(&path.to_str().expect("Couldn't get path string"))
                .expect("Couldn't create source reader");

        // Get source duration
        let mut duration = 0;
        unsafe {
            let presentation_duration = source_reader
                .GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
                .expect("Couldn't get presentation duration");
            let ns_100_duration =
                PropVariantToInt64(&presentation_duration).expect("Couldn't get duration");
            duration = ns_100_duration / 10_000_000; // convert to seconds
        }

        // Get source dimensions
        let mut source_width = 0;
        let mut source_height = 0;
        unsafe {
            let mut media_type = source_reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)?;

            let mut size_attr = media_type.GetUINT64(&MF_MT_FRAME_SIZE)?;
            source_width = (size_attr >> 32) as u32;
            source_height = (size_attr & 0xFFFFFFFF) as u32;
        }

        // Get source frame rate
        let mut source_frame_rate = 0.0;
        unsafe {
            let mut media_type = source_reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)
                .expect("Failed to get media type");

            let mut frame_rate_attr = media_type
                .GetUINT64(&MF_MT_FRAME_RATE)
                .expect("Failed to get frame rate");

            let frame_rate = (frame_rate_attr >> 32) as f64; // Numerator
            let frame_rate_base = (frame_rate_attr & 0xFFFFFFFF) as f64; // Denominator

            source_frame_rate = frame_rate / frame_rate_base;
        }

        Ok((
            source_reader,
            duration,
            source_width,
            source_height,
            source_frame_rate,
        ))
    }

    // #[cfg(target_arch = "wasm32")]
    // fn initialize_media_source() {}

    fn create_source_reader(
        // &self,
        file_path: &str,
    ) -> Result<IMFSourceReader, windows::core::Error> {
        unsafe {
            let wide_path: Vec<u16> = file_path.encode_utf16().chain(Some(0)).collect();

            // Create the media source from the file path
            let mut attributes: &mut Option<IMFAttributes> = &mut None;
            MFCreateAttributes(attributes, 0).expect("Couldn't create video decoder attributes");

            let mut attributes = attributes
                .as_ref()
                .expect("Couldn't get video decoder attributes");
            attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)?; // not necessary, can use nv12 which is YUV
            attributes.SetUINT32(&MF_READWRITE_DISABLE_CONVERTERS, 0)?; // not necessary, can use nv12 which is YUV

            // the only source reader needed for video
            let mut source_reader =
                MFCreateSourceReaderFromURL(PCWSTR(wide_path.as_ptr()), *&attributes)?;

            // Set the output format to RGB32
            // let mut media_type: IMFMediaType = std::ptr::null_mut();
            let mut media_type = MFCreateMediaType()?;
            media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
            // media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
            source_reader.SetCurrentMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                None,
                &media_type,
            )?;

            Ok(source_reader)
        }
    }

    pub fn draw_video_frame(&self, device: &Device, queue: &Queue) -> windows::core::Result<()> {
        unsafe {
            // println!("Drawing video frame");
            let mut flags: u32 = 0;
            let mut timestamp: i64 = 0; // store timestamp for later use?
            let mut sample: Option<IMFSample> = None;
            let mut actual_stream_index: &mut u32 = &mut 0;

            // println!("Reading sample");
            self.source_reader.ReadSample(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                0,
                Some(actual_stream_index),
                Some(&mut flags),
                Some(&mut timestamp),
                Some(&mut sample),
            )?;

            // println!("Convert to buffer");
            let mut sample = sample.as_ref().expect("Couldn't get sample container");
            let mut buffer = sample.ConvertToContiguousBuffer()?;

            // println!("Lock buffer");
            let mut data_ptr: *mut u8 = std::ptr::null_mut();
            let mut data_len: u32 = 0;
            let mut max_length = 0;
            buffer.Lock(&mut data_ptr, Some(&mut max_length), Some(&mut data_len))?;

            // println!("Copy data");
            let mut frame_data = Vec::with_capacity(data_len as usize);
            std::ptr::copy_nonoverlapping(data_ptr, frame_data.as_mut_ptr(), data_len as usize);
            frame_data.set_len(data_len as usize);

            // println!("Unlock buffer");
            buffer.Unlock()?;

            // println!("Write texture {:?}", frame_data.len());
            // Write texture data
            // need to write nv12 / YUV data to texture with proper bytes per row
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &frame_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.source_dimensions.0),
                    rows_per_image: Some(self.source_dimensions.1),
                },
                wgpu::Extent3d {
                    width: self.source_dimensions.0,
                    height: self.source_dimensions.1,
                    depth_or_array_layers: 1,
                },
            );

            Ok(())
        }
    }

    pub fn update_data_from_dimensions(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        dimensions: (f32, f32),
        camera: &Camera,
    ) {
        self.dimensions = (dimensions.0 as u32, dimensions.1 as u32);
        self.transform.update_scale([dimensions.0, dimensions.1]);
        self.transform.update_uniform_buffer(&queue, &window_size);
    }

    pub fn update_opacity(&mut self, queue: &wgpu::Queue, opacity: f32) {
        let new_color = [1.0, 1.0, 1.0, opacity];

        self.vertices.iter_mut().for_each(|v| {
            v.color = new_color;
        });

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
    }
}

// TODO: add to Drop trait?
fn shutdown_media_foundation() -> Result<(), windows::core::Error> {
    unsafe {
        MFShutdown()?;
    }
    Ok(())
}

impl Drop for StVideo {
    fn drop(&mut self) {
        unsafe {
            shutdown_media_foundation().expect("Couldn't shut down media foundation");
        }
    }
}
