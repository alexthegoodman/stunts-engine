// use crate::editor::Point;

// #[derive(Clone, Copy)]
// pub struct Transform {
//     pub position: Point,
//     // We could add scale and rotation here in the future if needed
// }

// impl Transform {
//     pub fn new(position: Point) -> Self {
//         Self { position }
//     }
// }

use std::f32::consts::PI;

use cgmath::{Deg, Matrix3, Matrix4, Rad, Vector2, Vector3};
use wgpu::util::DeviceExt;

use crate::editor::WindowSize;

pub struct Transform {
    pub position: Vector2<f32>,
    pub rotation: f32, // Rotation angle in radians
    pub scale: Vector2<f32>,
    pub uniform_buffer: wgpu::Buffer,
}

impl Transform {
    pub fn new(
        position: Vector2<f32>,
        rotation: f32, // Accepts angle in radians
        scale: Vector2<f32>,
        uniform_buffer: wgpu::Buffer,
        window_size: &WindowSize,
    ) -> Self {
        // let x = ((position[0]) / window_size.width as f32) * 2.0 - 1.0;
        // let y = 1.0 - ((position[1]) / window_size.height as f32) * 2.0;

        // println!("Transform x: {:?} y: {:?}", x, y);

        Self {
            // position: Vector2::new(x, y),
            position,
            rotation,
            scale,
            uniform_buffer,
        }
    }

    // pub fn update_transform(&self) -> Matrix3<f32> {
    //     // Create individual transformation matrices
    //     let translation = Matrix3::from_translation(self.position);
    //     let rotation = Matrix3::from_angle_z(Rad(self.rotation));
    //     let scale = Matrix3::from_nonuniform_scale(self.scale.x, self.scale.y);

    //     // Combine transformations: translation * rotation * scale
    //     translation * rotation * scale
    // }

    pub fn update_transform(&self, window_size: &WindowSize) -> Matrix4<f32> {
        // let x = self.position.x / window_size.width as f32 * 2.0 - 1.0;
        // let y = 1.0 - self.position.y / window_size.height as f32 * 2.0;
        let x = self.position.x;
        let y = self.position.y;

        // Create individual transformation matrices
        let translation = Matrix4::from_translation(Vector3::new(x, y, 0.0));
        let rotation = Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Rad(self.rotation));
        let scale = Matrix4::from_scale(self.scale.x);

        // Combine transformations: translation * rotation * scale
        translation * rotation * scale
    }

    pub fn update_uniform_buffer(&self, queue: &wgpu::Queue, window_size: &WindowSize) {
        // Convert Matrix3 to Matrix4 for shader compatibility
        // let transform_matrix = self.matrix3_to_matrix4(self.update_transform(window_size));
        let transform_matrix = self.update_transform(window_size);
        let raw_matrix = matrix4_to_raw_array(&transform_matrix);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&raw_matrix));
    }

    fn matrix3_to_matrix4(&self, mat3: Matrix3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            mat3.x.x, mat3.x.y, 0.0, mat3.x.z, // 1
            mat3.y.x, mat3.y.y, 0.0, mat3.y.z, // 2
            0.0, 0.0, 1.0, 0.0, // 3
            mat3.z.x, mat3.z.y, 0.0, mat3.z.z, // 4
        )
    }

    pub fn update_position(&mut self, position: [f32; 2], window_size: &WindowSize) {
        // let x = ((position[0]) / window_size.width as f32) * 2.0 - 1.0;
        // let y = 1.0 - ((position[1]) / window_size.height as f32) * 2.0;

        // self.position = Vector2::new(x, y);
        self.position = Vector2::new(position[0], position[1]);
    }

    pub fn update_rotation(&mut self, angle: f32) {
        self.rotation = angle;
    }

    pub fn update_rotation_degrees(&mut self, degrees: f32) {
        self.rotation = degrees * (PI / 180.0);
    }

    pub fn update_scale(&mut self, scale: [f32; 2]) {
        self.scale = Vector2::new(scale[0], scale[1]);
    }

    pub fn translate(&mut self, translation: Vector2<f32>) {
        self.position += translation;
    }

    pub fn rotate(&mut self, angle: f32) {
        self.rotation += angle;
    }

    pub fn rotate_degrees(&mut self, degrees: f32) {
        self.rotation += degrees * (PI / 180.0);
    }

    pub fn scale(&mut self, scale: Vector2<f32>) {
        self.scale.x *= scale.x;
        self.scale.y *= scale.y;
    }
}

pub fn matrix4_to_raw_array(matrix: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut array = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            array[i][j] = matrix[i][j];
        }
    }
    array
}
