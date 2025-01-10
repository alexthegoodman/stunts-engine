use cgmath::SquareMatrix;
use cgmath::{Matrix4, Vector2, Vector3};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::path::Path;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, TextureView};

use crate::camera::Camera;
use crate::editor::Point;
use crate::polygon::SavedPoint;
use crate::transform::matrix4_to_raw_array;
use crate::{
    editor::WindowSize,
    transform::Transform,
    vertex::{get_z_layer, Vertex},
};

#[derive(Clone)]
pub struct StImageConfig {
    pub id: String,
    pub name: String,
    pub dimensions: (u32, u32), // overrides actual image size
    pub position: Point,
    pub path: String,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedStImageConfig {
    pub id: String,
    pub name: String,
    pub dimensions: (u32, u32),
    pub path: String,
    pub position: SavedPoint,
}

pub struct StImage {
    pub id: String,
    pub name: String,
    pub path: String,
    pub texture: wgpu::Texture,
    pub texture_view: TextureView,
    pub transform: Transform,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub dimensions: (u32, u32),
    pub bind_group: wgpu::BindGroup,
    pub vertices: [Vertex; 4],
    pub indices: [u32; 6],
}

impl StImage {
    pub fn new(
        device: &Device,
        queue: &Queue,
        path: &Path,
        image_config: StImageConfig,
        window_size: &WindowSize,
        bind_group_layout: &wgpu::BindGroupLayout,
        z_index: f32,
        new_id: String,
    ) -> StImage {
        // NOTE: may be best to move images into destination project folder before supplying a path to StImage

        // specify resizing strategy
        let feature = "low_quality_resize"; // faster
                                            // let feature = "high_quality_resize"; // slow

        // Load the image
        let img = image::open(path).expect("Couldn't open image");
        let original_dimensions = img.dimensions();
        let dimensions = image_config.dimensions;

        // Calculate scale factors to maintain aspect ratio
        let scale_x = dimensions.0 as f32 / original_dimensions.0 as f32;
        let scale_y = dimensions.1 as f32 / original_dimensions.1 as f32;

        // Option 1: Resize image data before creating texture
        let img = if (feature == "high_quality_resize") {
            img.resize_exact(
                dimensions.0,
                dimensions.1,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            img
        };

        // Create texture with original or resized dimensions
        let texture_size = wgpu::Extent3d {
            width: if (feature == "high_quality_resize") {
                dimensions.0
            } else {
                original_dimensions.0
            },
            height: if (feature == "high_quality_resize") {
                dimensions.1
            } else {
                original_dimensions.1
            },
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });

        // Convert image to RGBA
        // let rgba = img.to_rgba8();
        let rgba = img.to_rgba8().into_raw();

        // Write texture data
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_size.width),
                rows_per_image: Some(texture_size.height),
            },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler with appropriate filtering
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: if (feature == "high_quality_resize") {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Linear // You might want to use Nearest for pixel art
            },
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let empty_buffer = Matrix4::<f32>::identity();
        let raw_matrix = matrix4_to_raw_array(&empty_buffer);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Image Uniform Buffer"),
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

        let scale_x = original_dimensions.0 as f32 * scale_x;
        let scale_y = original_dimensions.1 as f32 * scale_y;

        println!("scales {} {}", scale_x, scale_y);

        // Option 2: Use scale in transform to adjust size
        let mut transform = if (feature != "high_quality_resize") {
            Transform::new(
                Vector2::new(image_config.position.x, image_config.position.y),
                0.0,
                Vector2::new(scale_x, scale_y), // Apply scaling here instead of resizing image
                uniform_buffer,
                window_size,
            )
        } else {
            Transform::new(
                Vector2::new(image_config.position.x, image_config.position.y),
                0.0,
                Vector2::new(1.0, 1.0),
                uniform_buffer,
                window_size,
            )
        };

        transform.layer = -1.0;

        // Rest of the implementation remains the same...
        // let z = get_z_layer(1.0);
        let vertices = [
            Vertex {
                position: [-0.5, -0.5, 0.0],
                tex_coords: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                tex_coords: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                tex_coords: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.0],
                tex_coords: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            id: new_id,
            name: image_config.name,
            path: path
                .to_str()
                .expect("Couldn't convert to string")
                .to_string(),
            texture,
            texture_view,
            transform,
            vertex_buffer,
            index_buffer,
            dimensions, // Store the target dimensions
            bind_group,
            vertices,
            indices: indices.clone(),
        }
    }

    pub fn update(&mut self, queue: &Queue, window_size: &WindowSize) {
        self.transform.update_uniform_buffer(queue, window_size);
    }

    pub fn get_dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn contains_point(&self, point: &Point, camera: &Camera) -> bool {
        let local_point = self.to_local_space(*point, camera);

        // Get the bounds of the rectangle based on dimensions
        // Since dimensions are (width, height), the rectangle extends from (0,0) to (width, height)
        let (width, height) = self.dimensions;

        // Check if the point is within the bounds
        local_point.x >= 0.0
            && local_point.x <= width as f32
            && local_point.y >= 0.0
            && local_point.y <= height as f32
    }

    pub fn to_local_space(&self, world_point: Point, camera: &Camera) -> Point {
        let untranslated = Point {
            x: world_point.x - (self.transform.position.x),
            y: world_point.y - self.transform.position.y,
        };

        let local_point = Point {
            x: untranslated.x / (self.dimensions.0 as f32),
            y: untranslated.y / (self.dimensions.1 as f32),
        };

        // println!("local_point {:?} {:?}", self.name, local_point);

        local_point
    }

    // will be integrated directly in render loop
    // pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
    //     render_pass.set_bind_group(0, &self.bind_group, &[]);
    //     render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    //     render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    //     render_pass.draw_indexed(0..6, 0, 0..1);
    // }
}
