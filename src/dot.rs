use cgmath::SquareMatrix;
use cgmath::{Matrix4, Vector2};
use wgpu::util::DeviceExt;

use crate::{
    camera::{self, Camera},
    editor::{size_to_ndc, Point, WindowSize},
    transform::{matrix4_to_raw_array, Transform},
    vertex::{get_z_layer, Vertex},
};

#[derive(Clone, Copy, Debug)]
pub struct EdgePoint {
    pub point: Point,
    pub edge_index: usize,
}

pub fn closest_point_on_line_segment(start: Point, end: Point, point: Point) -> Point {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_squared = dx * dx + dy * dy;

    if length_squared == 0.0 {
        return start;
    }

    let t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared;
    let t = t.max(0.0).min(1.0);

    Point {
        x: start.x + t * dx,
        y: start.y + t * dy,
    }
}

// Helper function to get squared distance between two points
fn distance_squared(a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    dx * dx + dy * dy
}

// Helper function to get distance between two points
fn distance_sq(a: Point, b: Point) -> f32 {
    distance_squared(a, b).sqrt()
}

use cgmath::InnerSpace;

// Optional: Debug version that returns more information
#[derive(Debug)]
pub struct ClosestPointInfo {
    pub point: Point,
    pub distance: f32,
    pub normalized_t: f32, // Position along line segment (0 to 1)
    pub is_endpoint: bool, // Whether the closest point is at an endpoint
}

pub fn closest_point_on_line_segment_with_info(
    start: Point,
    end: Point,
    point: Point,
) -> ClosestPointInfo {
    let line_vec = Vector2::new(end.x - start.x, end.y - start.y);
    let point_vec = Vector2::new(point.x - start.x, point.y - start.y);

    let line_len_squared = line_vec.dot(line_vec);

    if line_len_squared == 0.0 {
        return ClosestPointInfo {
            point: start,
            distance: distance_sq(point, start),
            normalized_t: 0.0,
            is_endpoint: true,
        };
    }

    let t = (point_vec.dot(line_vec) / line_len_squared).clamp(0.0, 1.0);
    let is_endpoint = t == 0.0 || t == 1.0;

    let closest = Point {
        x: start.x + t * line_vec.x,
        y: start.y + t * line_vec.y,
    };

    ClosestPointInfo {
        point: closest,
        distance: distance_sq(point, closest),
        normalized_t: t,
        is_endpoint,
    }
}

pub fn distance(a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt()
}

pub struct RingDot {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub transform: Transform,
    pub bind_group: wgpu::BindGroup,
}

impl RingDot {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        window_size: &WindowSize,
        point: Point,
        color: [f32; 4],
        camera: &Camera,
    ) -> Self {
        let point = Point {
            x: 600.0 + point.x,
            y: 50.0 + point.y,
        };

        let (vertices, indices, vertex_buffer, index_buffer) =
            draw_dot(device, window_size, Point { x: 0.0, y: 0.0 }, color, camera);

        // Create a 1x1 white texture as a default
        let texture_size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default White Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create white pixel data
        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        // let blue_pixel: [u8; 4] = [10, 20, 255, 255]; // testing

        // Copy white pixel data to texture
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white_pixel,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create default sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let empty_buffer = Matrix4::<f32>::identity();
        let raw_matrix = matrix4_to_raw_array(&empty_buffer);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RingDot Uniform Buffer"),
            contents: bytemuck::cast_slice(&raw_matrix),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Now create your bind group with these defaults
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

        let mut transform = Transform::new(
            Vector2::new(point.x, point.y),
            0.0,
            Vector2::new(1.0, 1.0),
            uniform_buffer,
            window_size,
        );

        transform.layer = -2.0;

        return RingDot {
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
            transform,
            bind_group,
        };
    }
}

// draws a ring currently
// pub fn draw_ring(
//     device: &wgpu::Device,
//     window_size: &WindowSize,
//     point: Point,
//     color: [f32; 4],
//     camera: &Camera,
// ) -> (Vec<Vertex>, Vec<u32>, wgpu::Buffer, wgpu::Buffer) {
//     let x = point.x;
//     let y = point.y;

//     let scale_factor = camera.zoom;
//     let outer_radius = 0.01 * scale_factor;
//     let inner_radius = outer_radius * 0.7;

//     // println!("outer_radius {:?}", outer_radius);

//     let segments = 32 as u32; // Number of segments to approximate the circle

//     let dot_z = get_z_layer(2.0);

//     let mut vertices = Vec::with_capacity((segments * 2) as usize);
//     let mut indices: Vec<u32> = Vec::with_capacity((segments * 6) as usize);

//     // use indices to fill space between inner and outer vertices
//     for i in 0..segments {
//         let i = i as u32;
//         let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
//         let (sin, cos) = angle.sin_cos();

//         // Outer vertex
//         vertices.push(Vertex::new(
//             x + outer_radius * cos,
//             y + outer_radius * sin,
//             dot_z,
//             color,
//         ));

//         // Inner vertex
//         vertices.push(Vertex::new(
//             x + inner_radius * cos,
//             y + inner_radius * sin,
//             dot_z,
//             color,
//         ));

//         let base = i * 2;
//         let next_base = ((i + 1) % segments) * 2;

//         // Two triangles to form a quad
//         indices.extend_from_slice(&[
//             base,
//             base + 1,
//             next_base + 1,
//             base,
//             next_base + 1,
//             next_base,
//         ]);
//     }

//     // println!("dot vertices {:?}", vertices);

//     // Create a buffer for the vertices
//     let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("Ring Dot Vertex Buffer"),
//         contents: bytemuck::cast_slice(&vertices),
//         usage: wgpu::BufferUsages::VERTEX,
//     });

//     // Create a buffer for the indices
//     let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("Ring Dot Index Buffer"),
//         contents: bytemuck::cast_slice(&indices),
//         usage: wgpu::BufferUsages::INDEX,
//     });

//     (vertices, indices, vertex_buffer, index_buffer)
// }

// draws actual dot
pub fn draw_dot(
    device: &wgpu::Device,
    window_size: &WindowSize,
    point: Point,
    color: [f32; 4],
    camera: &Camera,
) -> (Vec<Vertex>, Vec<u32>, wgpu::Buffer, wgpu::Buffer) {
    let x = point.x;
    let y = point.y;

    let scale_factor = camera.zoom;
    let radius = 10.0 * scale_factor; // Just use a single radius for a solid dot

    let segments = 32 as u32; // Number of segments to approximate the circle
    let dot_z = get_z_layer(2.0);

    let mut vertices = Vec::with_capacity((segments + 1) as usize); // +1 for center vertex
    let mut indices: Vec<u32> = Vec::with_capacity((segments * 3) as usize); // 3 vertices per triangle

    // Add center vertex
    vertices.push(Vertex::new(x, y, dot_z, color));

    // Create outer vertices
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let (sin, cos) = angle.sin_cos();

        vertices.push(Vertex::new(
            x + radius * cos,
            y + radius * sin,
            dot_z,
            color,
        ));

        // Create triangles from center to outer edge
        let current_vertex = i + 1; // +1 because vertex 0 is the center
        let next_vertex = (i + 1) % segments + 1;

        indices.extend_from_slice(&[
            0, // Center vertex
            current_vertex,
            next_vertex,
        ]);
    }

    // Create buffers
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dot Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dot Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertices, indices, vertex_buffer, index_buffer)
}
