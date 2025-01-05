use cgmath::{Vector2, Vector3};
use image::GenericImageView;
use std::path::Path;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, TextureView};

use crate::{
    editor::WindowSize,
    transform::Transform,
    vertex::{get_z_layer, Vertex},
};

pub struct StImage {
    pub texture: wgpu::Texture,
    pub texture_view: TextureView,
    pub transform: Transform,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub dimensions: (u32, u32),
    pub bind_group: wgpu::BindGroup,
}

impl StImage {
    pub fn new(
        device: &Device,
        queue: &Queue,
        path: &Path,
        position: Vector2<f32>,
        rotation: f32,
        scale: Vector2<f32>,
        window_size: &WindowSize,
        bind_group_layout: &wgpu::BindGroupLayout,
        z_index: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load the image
        let img = image::open(path)?;
        let dimensions = img.dimensions();

        // Create texture
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Convert image to RGBA
        let rgba = img.to_rgba8();

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
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Image Bind Group"),
        });

        // Create transform uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Uniform Buffer"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let transform = Transform::new(position, rotation, scale, uniform_buffer, window_size);

        // Create vertices for a quad
        let z = get_z_layer(z_index);
        let vertices = [
            Vertex::new(-0.5, -0.5, z, [1.0, 1.0, 1.0, 1.0]), // Bottom left
            Vertex::new(0.5, -0.5, z, [1.0, 1.0, 1.0, 1.0]),  // Bottom right
            Vertex::new(0.5, 0.5, z, [1.0, 1.0, 1.0, 1.0]),   // Top right
            Vertex::new(-0.5, 0.5, z, [1.0, 1.0, 1.0, 1.0]),  // Top left
        ];

        // Set texture coordinates
        let vertices = [
            Vertex {
                tex_coords: [0.0, 1.0],
                ..vertices[0]
            },
            Vertex {
                tex_coords: [1.0, 1.0],
                ..vertices[1]
            },
            Vertex {
                tex_coords: [1.0, 0.0],
                ..vertices[2]
            },
            Vertex {
                tex_coords: [0.0, 0.0],
                ..vertices[3]
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer
        let indices: &[u16] = &[0, 1, 2, 2, 3, 0];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            texture,
            texture_view,
            transform,
            vertex_buffer,
            index_buffer,
            dimensions,
            bind_group,
        })
    }

    pub fn update(&mut self, queue: &Queue, window_size: &WindowSize) {
        self.transform.update_uniform_buffer(queue, window_size);
    }

    pub fn get_dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    // pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
    //     render_pass.set_bind_group(0, &self.bind_group, &[]);
    //     render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    //     render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    //     render_pass.draw_indexed(0..6, 0, 0..1);
    // }
}
