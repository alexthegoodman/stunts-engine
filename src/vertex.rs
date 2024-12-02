use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],   // x, y, z coordinates
    pub tex_coords: [f32; 2], // u, v coordinates
    // color: [f32; 3],      // RGB color
    // pub color: wgpu::Color, // RGBA color
    pub color: [f32; 4],
}

// Ensure Vertex is Pod and Zeroable
unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

pub fn get_z_layer(layer: f32) -> f32 {
    let z = -(layer as f32 / 1000.0);
    z
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32, color: [f32; 4]) -> Self {
        // the lower the layer, the higher in stack
        // let z = -(layer as f32 / 1000.0); // provide layer as 1, 2, etc but adjust z position minutely
        Vertex {
            position: [x, y, z],
            tex_coords: [0.0, 0.0], // Default UV coordinates
            color,
        }
    }
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, // Corresponds to layout(location = 0) in shader
                    format: wgpu::VertexFormat::Float32x3, // x3 for position
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1, // Corresponds to layout(location = 1) in shader
                    format: wgpu::VertexFormat::Float32x2, // x2 for uv
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2, // Corresponds to layout(location = 2) in shader
                    format: wgpu::VertexFormat::Float32x4, // x4 for color
                },
            ],
        }
    }
}
