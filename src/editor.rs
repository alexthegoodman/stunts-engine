use std::cell::RefCell;
use std::fmt::Display;
use std::sync::{Arc, Mutex, MutexGuard};

use cgmath::{Matrix4, Point3, Vector2, Vector3, Vector4};
use floem_renderer::gpu_resources::{self, GpuResources};
use floem_winit::keyboard::ModifiersState;
use floem_winit::window::Window;
use std::f32::consts::PI;
use uuid::Uuid;
use winit::window::CursorIcon;

// use crate::basic::{color_to_wgpu, string_to_f32, BoundingBox, Shape};
// use crate::brush::{BrushProperties, BrushStroke};
// use crate::camera::{self, Camera, CameraBinding};
// use crate::guideline::point_to_ndc;
// use crate::polygon::{PolygonConfig, Stroke};
// use crate::{
//     basic::Point,
//     basic::WindowSize,
//     dot::{distance, EdgePoint},
//     polygon::Polygon,
// };

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Debug, Clone, Copy)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

// Basic 2D point structure
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct BoundingBox {
    pub min: Point,
    pub max: Point,
}

// Basic shape traits
pub trait Shape {
    fn bounding_box(&self) -> BoundingBox;
    fn contains_point(&self, point: &Point, camera: &Camera) -> bool;
}

#[derive(Eq, PartialEq, Clone, Copy, EnumIter, Debug)]
pub enum ToolCategory {
    Shape,
    Brush,
}

#[derive(Eq, PartialEq, Clone, Copy, EnumIter, Debug)]
pub enum ControlMode {
    Point,
    Edge,
    Brush,
}

impl Display for ControlMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlMode::Point => f.write_str("Point"),
            ControlMode::Edge => f.write_str("Edge"),
            ControlMode::Brush => f.write_str("Brush"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Viewport {
    pub width: f32,
    pub height: f32,
}

impl Viewport {
    pub fn new(width: f32, height: f32) -> Self {
        Viewport { width, height }
    }

    pub fn to_ndc(&self, x: f32, y: f32) -> (f32, f32) {
        let ndc_x = (x / self.width) * 2.0 - 1.0;
        let ndc_y = -((y / self.height) * 2.0 - 1.0); // Flip Y-axis
        (ndc_x, ndc_y)
    }
}

pub fn size_to_ndc(window_size: &WindowSize, x: f32, y: f32) -> (f32, f32) {
    let ndc_x = x / window_size.width as f32;
    let ndc_y = y / window_size.height as f32;

    (ndc_x, ndc_y)
}

pub fn point_to_ndc(point: Point, window_size: &WindowSize) -> Point {
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;

    Point {
        x: ((point.x / window_size.width as f32) * 2.0 - 1.0),
        y: 1.0 - (point.y / window_size.height as f32) * 2.0,
    }
}

pub fn rgb_to_wgpu(r: u8, g: u8, b: u8, a: f32) -> [f32; 4] {
    [
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a.clamp(0.0, 1.0),
    ]
}

pub fn color_to_wgpu(c: f32) -> f32 {
    c / 255.0
}

pub fn wgpu_to_human(c: f32) -> f32 {
    c * 255.0
}

pub fn string_to_f32(s: &str) -> Result<f32, std::num::ParseFloatError> {
    let trimmed = s.trim();

    if trimmed.is_empty() {
        return Ok(0.0);
    }

    // Check if there's at least one digit in the string
    if !trimmed.chars().any(|c| c.is_ascii_digit()) {
        return Ok(0.0);
    }

    // At this point, we know there's at least one digit, so let's try to parse
    match trimmed.parse::<f32>() {
        Ok(num) => Ok(num),
        Err(e) => {
            // If parsing failed, check if it's because of a misplaced dash
            if trimmed.contains('-') && trimmed != "-" {
                // Remove all dashes and try parsing again
                let without_dashes = trimmed.replace('-', "");
                without_dashes.parse::<f32>().map(|num| -num.abs())
            } else {
                Err(e)
            }
        }
    }
}

// pub struct GuideLine {
//     pub start: Point,
//     pub end: Point,
// }

// Define all possible edit operations
#[derive(Debug)]
pub enum PolygonProperty {
    Width(f32),
}

#[derive(Debug)]
pub struct PolygonEditConfig {
    pub polygon_id: Uuid,
    pub field_name: String,
    pub old_value: PolygonProperty,
    pub new_value: PolygonProperty,
    // pub signal: RwSignal<String>,
}

pub type PolygonClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, PolygonConfig) + Send>> + Send + Sync;
// pub type LayersUpdateHandler =
//     dyn Fn() -> Option<Box<dyn FnMut(PolygonConfig) + Send>> + Send + Sync;

pub struct Editor {
    // polygons
    pub selected_polygon_id: Uuid,
    pub polygons: Vec<Polygon>,
    // pub layer_list: Vec<Uuid>,
    pub dragging_polygon: Option<usize>,

    // viewport
    pub viewport: Arc<Mutex<Viewport>>,
    pub handle_polygon_click: Option<Arc<PolygonClickHandler>>,
    pub gpu_resources: Option<Arc<GpuResources>>,
    // pub handle_layers_update: Option<Arc<LayersUpdateHandler>>,
    // pub control_mode: ControlMode,
    pub window: Option<Arc<Window>>,
    pub camera: Option<Camera>,
    // pub is_panning: bool,
    // pub is_brushing: bool,
    pub camera_binding: Option<CameraBinding>,

    // points
    pub last_mouse_pos: Option<Point>,
    pub drag_start: Option<Point>,
    pub last_screen: Point, // last mouse position from input event top-left origin
    pub last_world: Point,
    pub last_top_left: Point,   // for inside the editor zone
    pub global_top_left: Point, // for when recording mouse positions outside the editor zone
    pub ds_ndc_pos: Point,      // double-width sized ndc-style positioning (screen-oriented)
    pub ndc: Point,
}

use std::borrow::{Borrow, BorrowMut};

pub enum InputValue {
    Text(String),
    Number(f32),
    // Points(Vec<Point>),
}

impl Editor {
    pub fn new(viewport: Arc<Mutex<Viewport>>) -> Self {
        let viewport_unwrapped = viewport.lock().unwrap();
        let window_size = WindowSize {
            width: viewport_unwrapped.width as u32,
            height: viewport_unwrapped.height as u32,
        };
        Editor {
            selected_polygon_id: Uuid::nil(),
            polygons: Vec::new(),
            dragging_polygon: None,
            drag_start: None,
            viewport: viewport.clone(),
            handle_polygon_click: None,
            gpu_resources: None,
            // handle_layers_update: None,
            window: None,
            camera: None,
            camera_binding: None,
            last_mouse_pos: None,
            last_screen: Point { x: 0.0, y: 0.0 },
            last_world: Point { x: 0.0, y: 0.0 },
            ds_ndc_pos: Point { x: 0.0, y: 0.0 },
            last_top_left: Point { x: 0.0, y: 0.0 },
            global_top_left: Point { x: 0.0, y: 0.0 },
            ndc: Point { x: 0.0, y: 0.0 },
        }
    }

    pub fn update_camera_binding(&mut self, queue: &wgpu::Queue) {
        if (self.camera_binding.is_some()) {
            self.camera_binding
                .as_mut()
                .expect("Couldn't get camera binding")
                .update(queue, &self.camera.as_ref().expect("Couldn't get camera"));
        }
    }

    pub fn handle_wheel(&mut self, delta: f32, mouse_pos: Point, queue: &wgpu::Queue) {
        let camera = self.camera.as_mut().expect("Couldnt't get camera");

        let interactive_bounds = BoundingBox {
            min: Point { x: 550.0, y: 0.0 }, // account for aside width
            max: Point {
                x: camera.window_size.width as f32,
                y: camera.window_size.height as f32,
            },
        };

        if (mouse_pos.x < interactive_bounds.min.x
            || mouse_pos.x > interactive_bounds.max.x
            || mouse_pos.y < interactive_bounds.min.y
            || mouse_pos.y > interactive_bounds.max.y)
        {
            return;
        }

        // let zoom_factor = if delta > 0.0 { 1.1 } else { 0.9 };
        let zoom_factor = delta / 10.0;
        camera.zoom(zoom_factor, mouse_pos);
        self.update_camera_binding(queue);
    }

    pub fn add_polygon(&mut self, mut polygon: Polygon) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        // let world_position = camera.screen_to_world(polygon.transform.position);
        let world_position = polygon.transform.position;
        println!(
            "add polygon position {:?} {:?}",
            world_position, polygon.transform.position
        );
        polygon.transform.position = world_position;
        self.polygons.push(polygon);
        // self.run_layers_update();
    }

    pub fn update_polygon(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // let mut gpu_helper = cloned_helper.lock().unwrap();

        // First iteration: find the index of the selected polygon
        // let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        // if let Some(index) = polygon_index {
        //     println!("Found selected polygon with ID: {}", selected_id);

        //     let camera = self.camera.expect("Couldn't get camera");

        //     // Get the necessary data from editor
        //     let viewport_width = camera.window_size.width;
        //     let viewport_height = camera.window_size.height;
        //     let device = &self
        //         .gpu_resources
        //         .as_ref()
        //         .expect("Couldn't get gpu resources")
        //         .device;

        //     let window_size = WindowSize {
        //         width: viewport_width as u32,
        //         height: viewport_height as u32,
        //     };

        //     // Second iteration: update the selected polygon
        //     if let Some(selected_polygon) = self.polygons.get_mut(index) {
        //         match new_value {
        //             InputValue::Text(s) => match key {
        //                 _ => println!("No match on input"),
        //             },
        //             InputValue::Number(n) => match key {
        //                 "width" => selected_polygon.update_data_from_dimensions(
        //                     &window_size,
        //                     &device,
        //                     (n, selected_polygon.dimensions.1),
        //                     &camera,
        //                 ),
        //                 _ => println!("No match on input"),
        //             },
        //         }
        //     }
        // } else {
        //     println!("No polygon found with the selected ID: {}", selected_id);
        // }
    }

    pub fn get_polygon_width(&self, selected_id: Uuid) -> f32 {
        // let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        // if let Some(index) = polygon_index {
        //     if let Some(selected_polygon) = self.polygons.get(index) {
        //         return selected_polygon.dimensions.0;
        //     } else {
        //         return 0.0;
        //     }
        // }

        0.0
    }

    // pub fn update_date_from_window_resize(
    //     &mut self,
    //     window_size: &WindowSize,
    //     device: &wgpu::Device,
    // ) {
    //     let camera = self.camera.as_ref().expect("Couldn't get camera");
    //     for (poly_index, polygon) in self.polygons.iter_mut().enumerate() {
    //         polygon.update_data_from_window_size(window_size, device, &camera);
    //     }
    // }

    pub fn handle_mouse_down(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) -> Option<PolygonEditConfig> {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let x = self.ds_ndc_pos.x;
        let y = self.ds_ndc_pos.y;
        let mouse_pos = Point { x, y };
        // let world_pos = camera.screen_to_world(mouse_pos);

        // self.update_cursor();

        // match self.control_mode {
        //     ControlMode::Point => self.handle_mouse_down_point_mode(mouse_pos, window_size, device),
        //     ControlMode::Edge => self.handle_mouse_down_edge_mode(mouse_pos, window_size, device),
        //     ControlMode::Brush => self.handle_mouse_down_brush_mode(mouse_pos, window_size, device),
        // }

        // Check if we're clicking on a polygon to drag
        for (poly_index, polygon) in self.polygons.iter_mut().enumerate() {
            if polygon.contains_point(&self.last_top_left, &camera) {
                self.dragging_polygon = Some(poly_index);
                self.drag_start = Some(self.last_top_left);

                // TODO: make DRY with below
                if (self.handle_polygon_click.is_some()) {
                    let handler_creator = self
                        .handle_polygon_click
                        .as_ref()
                        .expect("Couldn't get handler");
                    let mut handle_click = handler_creator().expect("Couldn't get handler");
                    handle_click(
                        polygon.id,
                        PolygonConfig {
                            id: polygon.id,
                            name: polygon.name.clone(),
                            points: polygon.points.clone(),
                            dimensions: polygon.dimensions,
                            position: polygon.transform.position,
                            border_radius: polygon.border_radius,
                            fill: polygon.fill,
                            stroke: polygon.stroke,
                        },
                    );
                    self.selected_polygon_id = polygon.id;
                    polygon.old_points = Some(polygon.points.clone());
                }

                return None; // nothing to add to undo stack
            }
        }

        None
    }

    pub fn handle_mouse_move(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        x: f32,
        y: f32,
    ) {
        let camera = self.camera.as_mut().expect("Couldn't get camera");
        let mouse_pos = Point { x, y };
        let ds_ndc = visualize_ray_intersection(window_size, x, y, &camera);
        let ds_ndc_pos = ds_ndc.origin;
        let ds_ndc_pos = Point {
            x: ds_ndc_pos.x,
            y: ds_ndc_pos.y,
        };
        let top_left = ds_ndc.top_left;

        self.global_top_left = top_left;

        let interactive_bounds = BoundingBox {
            min: Point { x: 550.0, y: 0.0 }, // account for aside width
            max: Point {
                x: window_size.width as f32,
                y: window_size.height as f32,
            },
        };

        if (x < interactive_bounds.min.x
            || x > interactive_bounds.max.x
            || y < interactive_bounds.min.y
            || y > interactive_bounds.max.y)
        {
            return;
        }

        self.last_top_left = top_left;
        self.ds_ndc_pos = ds_ndc_pos;
        self.ndc = ds_ndc.ndc;

        self.last_screen = Point { x, y };
        self.last_world = camera.screen_to_world(mouse_pos);

        // // Handle panning
        // if self.is_panning {
        //     if let Some(last_pos) = self.last_mouse_pos {
        //         let delta = Vector2::new(ds_ndc_pos.x - last_pos.x, ds_ndc_pos.y - last_pos.y);

        //         // println!("is_panning A {:?}", delta);

        //         let adjusted_delta = Vector2::new(
        //             -delta.x, // Invert X
        //             -delta.y, // Keep Y as is
        //         );
        //         let delta = adjusted_delta / 2.0;

        //         // println!("is_panning B {:?}", delta);
        //         // adjusting the camera, so expect delta to be very small like 0-1
        //         camera.pan(delta);
        //         let mut camera_binding = self
        //             .camera_binding
        //             .as_mut()
        //             .expect("Couldn't get camera binging");
        //         let gpu_resources = self
        //             .gpu_resources
        //             .as_ref()
        //             .expect("Couldn't get gpu resources");
        //         camera_binding.update(&gpu_resources.queue, &camera);
        //     }
        //     self.last_mouse_pos = Some(ds_ndc_pos);
        //     return;
        // }

        // match self.control_mode {
        //     ControlMode::Point => {
        //         self.handle_mouse_move_point_mode(ds_ndc_pos, window_size, device)
        //     }
        //     ControlMode::Edge => self.handle_mouse_move_edge_mode(ds_ndc_pos, window_size, device),
        //     ControlMode::Brush => {
        //         self.handle_mouse_move_brush_mode(ds_ndc_pos, window_size, device)
        //     }
        // }

        // self.update_cursor();
    }

    pub fn handle_mouse_up(&mut self) -> Option<PolygonEditConfig> {
        let mut action_edit = None;

        // let polygon_index = self
        //     .polygons
        //     .iter()
        //     .position(|p| p.id == self.selected_polygon_id);

        // if let Some(index) = polygon_index {
        //     if let Some(selected_polygon) = self.polygons.get(index) {
        //         if (selected_polygon.old_points.is_some()) {
        //             if (self.dragging_polygon.is_some()) {
        //                 // return PolygonProperty::Position
        //             }
        //         }
        //     }
        // }

        // self.dragging_point = None;
        self.dragging_polygon = None;
        self.drag_start = None;
        // self.dragging_edge = None;
        // self.is_panning = false;
        // self.is_brushing = false;
        // self.guide_lines.clear();
        // self.update_cursor();

        action_edit
    }

    fn is_close(&self, a: f32, b: f32, threshold: f32) -> bool {
        (a - b).abs() < threshold
    }
}

use cgmath::InnerSpace;

#[derive(Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub ndc: Point,
    pub top_left: Point,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Ray {
            origin,
            direction: direction.normalize(),
            ndc: Point { x: 0.0, y: 0.0 },
            top_left: Point { x: 0.0, y: 0.0 },
        }
    }
}

use cgmath::SquareMatrix;
use cgmath::Transform;

use crate::camera::{Camera, CameraBinding};
use crate::polygon::{Polygon, PolygonConfig};

pub fn visualize_ray_intersection(
    // device: &wgpu::Device,
    window_size: &WindowSize,
    screen_x: f32,
    screen_y: f32,
    camera: &Camera,
) -> Ray {
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;

    let ndc_x = screen_x / camera.window_size.width as f32;
    let ndc_y = (screen_y / camera.window_size.height as f32);

    let view_pos = Vector3::new(0.0, 0.0, 0.0);
    let model_view = Matrix4::from_translation(view_pos);

    let scale_factor = camera.zoom;

    let plane_size_normal = Vector3::new(
        (1.0 * aspect_ratio * scale_factor) / 2.0,
        (1.0 * 2.0 * scale_factor) / 2.0,
        0.0,
    );

    // Transform NDC point to view space
    let view_point_normal = Point3::new(
        (ndc_x * plane_size_normal.x),
        (ndc_y * plane_size_normal.y),
        0.0,
    );
    let world_point_normal = model_view
        .invert()
        .unwrap()
        .transform_point(view_point_normal);

    // println!("normal {:?}", world_point_normal);

    // Create a plane in view space
    let plane_center = Point3::new(
        -(camera.window_size.width as f32) * scale_factor,
        -(camera.window_size.height as f32) * scale_factor,
        0.0,
    );

    let plane_size = Vector3::new(
        (camera.window_size.width as f32 * scale_factor) * aspect_ratio,
        (camera.window_size.height as f32 * scale_factor) * 2.0,
        0.0,
    );

    // Transform NDC point to view space, accounting for center offset
    let view_point = Point3::new(
        ndc_x * plane_size.x + plane_center.x,
        ndc_y * plane_size.y + plane_center.y,
        0.0,
    );

    // Transform to world space
    let world_point = model_view.invert().unwrap().transform_point(view_point);

    // Create ray from camera position to point (in 3D space)
    let camera_pos_3d = Point3::new(camera.position.x, camera.position.y, 0.0);
    let direction = (world_point - camera_pos_3d).normalize();

    let origin = Point3 {
        x: world_point.x + camera.position.x + 140.0,
        y: -(world_point.y) + camera.position.y,
        z: world_point.z,
    };

    let ndc = camera.normalized_to_ndc(world_point_normal.x, world_point_normal.y);

    let offset_x = (scale_factor - 1.0) * (400.0 * aspect_ratio);
    let offset_y = (scale_factor - 1.0) * 400.0;

    let top_left: Point = Point {
        x: (world_point_normal.x * window_size.width as f32) + (camera.position.x * 0.5) + 70.0
            - offset_x,
        y: (world_point_normal.y * window_size.height as f32)
            - (camera.position.y * 0.5)
            - offset_y,
    };

    Ray {
        direction,
        origin,
        ndc: Point { x: ndc.0, y: ndc.1 },
        top_left,
    }
}