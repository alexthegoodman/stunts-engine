use cgmath::{Matrix4, Point2, Point3, Vector2, Vector3, Vector4};

use crate::editor::{point_to_ndc, size_to_normal, Point, WindowSize};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vector2<f32>,
    pub zoom: f32,
    pub window_size: WindowSize,
    pub focus_point: Vector2<f32>, // Center point of the view
                                   // Implied constants:
                                   // direction: Always (0, 0, -1)    // Always looking into screen
                                   // up: Always (0, 1, 0)           // Y is always up
                                   // znear: -1.0                    // Simple z-range for 2D
                                   // zfar: 1.0                      // Simple z-range for 2D
}

impl Camera {
    pub fn new(window_size: WindowSize) -> Self {
        let focus_point = Vector2::new(
            window_size.width as f32 / 2.0,
            window_size.height as f32 / 2.0,
        );

        Self {
            position: Vector2::new(0.0, 0.0),
            zoom: 1.0,
            window_size,
            focus_point,
        }
    }

    // pub fn screen_to_world(&self, screen_pos: Point) -> Point {
    //     Point {
    //         x: (screen_pos.x + self.position.x),
    //         y: (screen_pos.y + self.position.y),
    //     }
    // }

    // pub fn world_to_screen(&self, world_pos: Point) -> Point {
    //     Point {
    //         x: (world_pos.x - self.position.x),
    //         y: (world_pos.y - self.position.y),
    //     }
    // }

    // pub fn ds_ndc_to_top_left(&self, ds_ndc_pos: Point) -> Point {
    //     let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;
    //     let pos_x = ds_ndc_pos.x / self.window_size.width as f32;
    //     let pos_y = ds_ndc_pos.y / self.window_size.height as f32;

    //     let (x, y) = self.ndc_to_normalized(pos_x, pos_y);

    //     let x = (x * self.window_size.width as f32) + 65.0;
    //     let y = y * self.window_size.height as f32;

    //     Point { x, y }
    // }

    // pub fn ndc_to_top_left(&self, ds_ndc_pos: Point) -> Point {
    //     let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;

    //     let (x, y) = self.ndc_to_normalized(ds_ndc_pos.x, ds_ndc_pos.y);

    //     let x = (x * self.window_size.width as f32);
    //     let y = y * self.window_size.height as f32;

    //     Point { x, y }
    // }

    // pub fn ndc_to_normalized(&self, ndc_x: f32, ndc_y: f32) -> (f32, f32) {
    //     // Convert from [-1, 1] to [0, 1]
    //     let norm_x = (ndc_x + 1.0) / 2.0;
    //     // Flip Y and convert from [-1, 1] to [0, 1]
    //     let norm_y = (-ndc_y + 1.0) / 2.0;

    //     (norm_x, norm_y)
    // }

    // And its inverse if you need it
    pub fn normalized_to_ndc(&self, norm_x: f32, norm_y: f32) -> (f32, f32) {
        // Convert from [0, 1] to [-1, 1]
        let ndc_x = (norm_x * 2.0) - 1.0;
        // Convert from [0, 1] to [-1, 1] and flip Y
        let ndc_y = -((norm_y * 2.0) - 1.0);

        (ndc_x, ndc_y)
    }

    pub fn get_view_projection_matrix(&self) -> Matrix4<f32> {
        let projection = self.get_projection();

        let view = self.get_view();

        let vp = projection * view;

        vp
    }

    pub fn get_projection(&self) -> Matrix4<f32> {
        let zoom_factor = self.zoom;
        let aspect_ratio = self.window_size.width as f32 / self.window_size.height as f32;

        cgmath::ortho(
            -(zoom_factor * aspect_ratio) / 2.0, // left
            (zoom_factor * aspect_ratio) / 2.0,  // right
            -zoom_factor,                        // bottom
            zoom_factor,                         // top
            -100.0,                              // near
            100.0,                               // far
        )
    }

    pub fn get_view(&self) -> Matrix4<f32> {
        let test_norm = size_to_normal(&self.window_size, self.position.x, self.position.y);
        let view = Matrix4::new(
            // self.zoom,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            // self.zoom,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -test_norm.0,
            -test_norm.1,
            0.0,
            1.0,
        );

        view
    }

    pub fn pan(&mut self, delta: Vector2<f32>) {
        // println!("delta {:?} {:?}", delta, self.position);
        self.position += delta;
    }

    pub fn zoom(&mut self, factor: f32, center: Point) {
        // let world_center = self.screen_to_world(center);
        let world_center = center;

        // For zoom in/out to be reversible, we need multiplicative inverses
        // e.g., zooming by 0.9 then by 1/0.9 should restore original state
        let zoom_factor = if factor > 0.0 {
            1.0 + factor
        } else {
            1.0 / (1.0 - factor)
        };

        println!("zoom {:?} {:?}", self.zoom, zoom_factor);

        let old_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_factor).clamp(-10.0, 10.0);

        // Keep the point under cursor stationary
        // let world_pos = Vector2::new(world_center.x, world_center.y);
        // self.position = world_pos + (self.position - world_pos) * (old_zoom / self.zoom);
    }
}

use bytemuck::{Pod, Zeroable};
use cgmath::SquareMatrix;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.get_view_projection_matrix().into();
    }
}

pub struct CameraBinding {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform: CameraUniform,
}

impl CameraBinding {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform = CameraUniform::new();

        // Create the uniform buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<CameraUniform>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            buffer,
            bind_group,
            bind_group_layout,
            uniform,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, camera: &Camera) {
        self.uniform.update_view_proj(camera);
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.uniform.view_proj]),
        );
    }
}
