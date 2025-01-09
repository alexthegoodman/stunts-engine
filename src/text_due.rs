use std::{borrow::Borrow, collections::HashMap};

use cgmath::{Matrix4, Vector2};
use fontdue::Font;
use uuid::Uuid;
use wgpu::{BindGroup, Buffer, Device, Queue, RenderPipeline, TextureFormat};
// use allsorts::binary::read::ReadScope;
// use allsorts::font::read_cmap_subtable;
use bytemuck::{Pod, Zeroable};
use cgmath::SquareMatrix;
use serde::Deserialize;
use serde::Serialize;
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    editor::{Point, WindowSize},
    transform::{matrix4_to_raw_array, Transform},
    vertex::Vertex,
};

struct AtlasGlyph {
    uv_rect: [f32; 4], // x, y, width, height in UV coordinates
    metrics: [f32; 4], // width, height, xmin, ymin in pixels
}

#[derive(Clone)]
pub struct TextRendererConfig {
    pub id: Uuid,
    pub name: String,
    pub text: String,
    pub dimensions: (f32, f32),
    pub position: Point,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedTextRendererConfig {
    pub id: String,
    pub name: String,
    pub text: String,
    pub dimensions: (i32, i32),
    // position is determined by the keyframes (?)
}

pub struct TextRenderer {
    pub id: Uuid,
    pub intialized: bool,
    pub name: String,
    pub text: String,
    pub font: Font,
    pub transform: Transform,
    pub bind_group: BindGroup,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub dimensions: (f32, f32), // (width, height) in pixels
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub atlas_texture: wgpu::Texture,
    pub atlas_size: (u32, u32),
    pub next_atlas_position: (u32, u32),
    pub current_row_height: u32,
    pub glyph_cache: HashMap<char, AtlasGlyph>,
}

impl TextRenderer {
    pub fn new(
        device: &Device,
        // queue: &Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        font_data: &[u8],
        window_size: &WindowSize,
        text: String,
        text_config: TextRendererConfig,
        id: Uuid,
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

        let texture_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
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
            label: None,
        });

        Self {
            id,
            intialized: false,
            name: "New Text Item".to_string(),
            text,
            font,
            transform: Transform::new(
                Vector2::new(text_config.position.x, text_config.position.y),
                0.0,
                Vector2::new(1.0, 1.0),
                uniform_buffer,
                window_size,
            ),
            vertex_buffer,
            index_buffer,
            vertices: Vec::new(),
            indices: Vec::new(),
            dimensions: (text_config.dimensions.0, text_config.dimensions.1),
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

    pub fn update(&mut self, device: &Device, queue: &Queue, text: String, dimensions: (f32, f32)) {
        self.dimensions = dimensions;
        self.update_text(device, queue, text);

        self.intialized = true;
    }

    pub fn update_text(&mut self, device: &Device, queue: &Queue, text: String) {
        self.text = text;
        self.render_text(device, queue);
    }

    pub fn render_text<'a>(
        &'a mut self,
        device: &Device,
        queue: &Queue,
        // render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut current_x = 0.0;

        let text = self.text.clone();
        let chars = text.chars();

        // First, ensure all glyphs are in the atlas
        for c in chars.clone() {
            if !self.glyph_cache.contains_key(&c) {
                let glyph = self.add_glyph_to_atlas(device, queue, c, 32.0);
                self.glyph_cache.insert(c, glyph);
            }
        }

        for c in chars {
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

        // render_pass will be intergrated in render loop
        // render_pass.set_pipeline(&self.pipeline);
        // render_pass.set_bind_group(0, &self.bind_group, &[]);
        // render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        // render_pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);

        self.vertices = vertices;
        self.indices = indices;
    }

    // pub fn contains_point(&self, point: &Point, camera: &Camera) -> bool {
    //     let local_point = self.to_local_space(*point, camera);

    //     // Implement point-in-polygon test using the ray casting algorithm
    //     let mut inside = false;

    //     // TODO: simpler variation based on dimensions

    //     // let mut j = self.points.len() - 1;
    //     // for i in 0..self.points.len() {
    //     //     let pi = &self.points[i];
    //     //     let pj = &self.points[j];

    //     //     if ((pi.y > local_point.y) != (pj.y > local_point.y))
    //     //         && (local_point.x < (pj.x - pi.x) * (local_point.y - pi.y) / (pj.y - pi.y) + pi.x)
    //     //     {
    //     //         inside = !inside;
    //     //     }
    //     //     j = i;
    //     // }

    //     inside
    // }

    pub fn contains_point(&self, point: &Point, camera: &Camera) -> bool {
        let local_point = self.to_local_space(*point, camera);

        // Get the bounds of the rectangle based on dimensions
        // Since dimensions are (width, height), the rectangle extends from (0,0) to (width, height)
        let (width, height) = self.dimensions;

        // Check if the point is within the bounds
        local_point.x >= 0.0
            && local_point.x <= width
            && local_point.y >= 0.0
            && local_point.y <= height
    }

    pub fn to_local_space(&self, world_point: Point, camera: &Camera) -> Point {
        let untranslated = Point {
            x: world_point.x - (self.transform.position.x),
            y: world_point.y - self.transform.position.y,
        };

        let local_point = Point {
            x: untranslated.x / (self.dimensions.0),
            y: untranslated.y / (self.dimensions.1),
        };

        // println!("local_point {:?} {:?}", self.name, local_point);

        local_point
    }
}
