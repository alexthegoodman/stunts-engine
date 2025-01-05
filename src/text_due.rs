use std::collections::HashMap;

use cgmath::{Matrix4, Vector2};
use fontdue::Font;
use wgpu::{BindGroup, Buffer, Device, Queue, RenderPipeline, TextureFormat};
// use allsorts::binary::read::ReadScope;
// use allsorts::font::read_cmap_subtable;
use bytemuck::{Pod, Zeroable};
use cgmath::SquareMatrix;
use wgpu::util::DeviceExt;

use crate::{
    editor::WindowSize,
    transform::{matrix4_to_raw_array, Transform},
    vertex::Vertex,
};

struct AtlasGlyph {
    uv_rect: [f32; 4], // x, y, width, height in UV coordinates
    metrics: [f32; 4], // width, height, xmin, ymin in pixels
}

pub struct TextRenderer {
    font: Font,
    transform: Transform,
    bind_group: BindGroup,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    atlas_texture: wgpu::Texture,
    atlas_size: (u32, u32),
    next_atlas_position: (u32, u32),
    current_row_height: u32,
    glyph_cache: HashMap<char, AtlasGlyph>,
}

impl TextRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        font_data: &[u8],
        window_size: &WindowSize,
    ) -> Self {
        // Load and initialize the font
        let font = Font::from_bytes(font_data, fontdue::FontSettings::default())
            .expect("Failed to load font");

        // Create texture atlas
        let atlas_size = (1024, 1024);
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Glyph Atlas Texture"),
            size: wgpu::Extent3d {
                width: atlas_size.0,
                height: atlas_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[wgpu::TextureFormat::R8Unorm],
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        // Initialize empty vertex and index buffers
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Text Vertex Buffer"),
            size: 1024, // Initial size, can be adjusted
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Text Index Buffer"),
            size: 1024, // Initial size, can be adjusted
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let empty_buffer = Matrix4::<f32>::identity();
        let raw_matrix = matrix4_to_raw_array(&empty_buffer);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Uniform Buffer"),
            contents: bytemuck::cast_slice(&raw_matrix),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        Self {
            font,
            transform: Transform::new(
                Vector2::new(100.0, 100.0),
                0.0,
                Vector2::new(1.0, 1.0),
                uniform_buffer,
                window_size,
            ),
            vertex_buffer,
            index_buffer,
            bind_group,
            atlas_texture,
            atlas_size,
            next_atlas_position: (0, 0),
            current_row_height: 0,
            glyph_cache: HashMap::new(),
        }
    }

    fn add_glyph_to_atlas(
        &mut self,
        device: &Device,
        queue: &Queue,
        c: char,
        size: f32,
    ) -> AtlasGlyph {
        let (metrics, bitmap) = self.font.rasterize(c, size);

        // Check if we need to move to the next row
        if self.next_atlas_position.0 + metrics.width as u32 > self.atlas_size.0 {
            self.next_atlas_position.0 = 0;
            self.next_atlas_position.1 += self.current_row_height;
            self.current_row_height = 0;
        }

        // Update current row height if this glyph is taller
        self.current_row_height = self.current_row_height.max(metrics.height as u32);

        // Calculate UV coordinates
        let uv_rect = [
            self.next_atlas_position.0 as f32 / self.atlas_size.0 as f32,
            self.next_atlas_position.1 as f32 / self.atlas_size.1 as f32,
            metrics.width as f32 / self.atlas_size.0 as f32,
            metrics.height as f32 / self.atlas_size.1 as f32,
        ];

        // Write glyph bitmap to atlas
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: self.next_atlas_position.0,
                    y: self.next_atlas_position.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(metrics.width as u32),
                rows_per_image: Some(metrics.height as u32),
            },
            wgpu::Extent3d {
                width: metrics.width as u32,
                height: metrics.height as u32,
                depth_or_array_layers: 1,
            },
        );

        // Update atlas position for next glyph
        self.next_atlas_position.0 += metrics.width as u32;

        AtlasGlyph {
            uv_rect,
            metrics: [
                metrics.width as f32,
                metrics.height as f32,
                metrics.xmin as f32,
                metrics.ymin as f32,
            ],
        }
    }

    pub fn render_text<'a>(
        &'a mut self,
        text: &str,
        transform: &Transform,
        device: &Device,
        queue: &Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut current_x = 0.0;

        // First, ensure all glyphs are in the atlas
        for c in text.chars() {
            if !self.glyph_cache.contains_key(&c) {
                let glyph = self.add_glyph_to_atlas(device, queue, c, 32.0);
                self.glyph_cache.insert(c, glyph);
            }
        }

        for c in text.chars() {
            let glyph = self.glyph_cache.get(&c).unwrap();

            let base_vertex = vertices.len() as u16;

            // Calculate vertex positions using metrics and UV coordinates
            let x0 = current_x + glyph.metrics[2];
            let x1 = x0 + glyph.metrics[0];
            let y0 = glyph.metrics[3];
            let y1 = y0 + glyph.metrics[1];

            // UV coordinates from atlas
            let u0 = glyph.uv_rect[0];
            let u1 = u0 + glyph.uv_rect[2];
            let v0 = glyph.uv_rect[1];
            let v1 = v0 + glyph.uv_rect[3];

            vertices.extend_from_slice(&[
                Vertex {
                    position: [x0, y0, 0.0],
                    tex_coords: [u0, v0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                Vertex {
                    position: [x1, y0, 0.0],
                    tex_coords: [u1, v0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                Vertex {
                    position: [x1, y1, 0.0],
                    tex_coords: [u1, v1],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                Vertex {
                    position: [x0, y1, 0.0],
                    tex_coords: [u0, v1],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ]);

            indices.extend_from_slice(&[
                base_vertex,
                base_vertex + 1,
                base_vertex + 2,
                base_vertex,
                base_vertex + 2,
                base_vertex + 3,
            ]);

            current_x += glyph.metrics[0];
        }

        // Update buffers and draw
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));

        // render_pass.set_pipeline(&self.pipeline);
        // render_pass.set_bind_group(0, &self.bind_group, &[]);
        // render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        // render_pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);
    }
}
