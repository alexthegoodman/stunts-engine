use std::cell::RefCell;
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WindowSizeShader {
    pub width: f32,
    pub height: f32,
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
    Height(f32),
    Red(f32),
    Green(f32),
    Blue(f32),
    BorderRadius(f32),
    StrokeThickness(f32),
    StrokeRed(f32),
    StrokeGreen(f32),
    StrokeBlue(f32),
    // Points(Vec<Point>),
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

pub type TextItemClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, TextRendererConfig) + Send>> + Send + Sync;

pub type ImageItemClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, StImageConfig) + Send>> + Send + Sync;

pub type OnMouseUp =
    dyn Fn() -> Option<Box<dyn FnMut(usize, Point) -> Sequence + Send>> + Send + Sync;

pub struct Editor {
    // visual
    pub selected_polygon_id: Uuid,
    pub polygons: Vec<Polygon>,
    pub dragging_polygon: Option<usize>,
    pub static_polygons: Vec<Polygon>,
    pub project_selected: Option<Uuid>,
    pub text_items: Vec<TextRenderer>,
    pub dragging_text: Option<usize>,
    pub image_items: Vec<StImage>,
    pub dragging_image: Option<usize>,
    pub font_manager: FontManager,

    // viewport
    pub viewport: Arc<Mutex<Viewport>>,
    pub handle_polygon_click: Option<Arc<PolygonClickHandler>>,
    pub handle_text_click: Option<Arc<TextItemClickHandler>>,
    pub handle_image_click: Option<Arc<ImageItemClickHandler>>,
    pub gpu_resources: Option<Arc<GpuResources>>,
    pub window: Option<Arc<Window>>,
    pub camera: Option<Camera>,
    pub camera_binding: Option<CameraBinding>,
    pub model_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pub window_size_bind_group: Option<wgpu::BindGroup>,
    pub on_mouse_up: Option<Arc<OnMouseUp>>,
    pub current_view: String,

    // state
    pub is_playing: bool,
    pub current_sequence_data: Option<Sequence>,
    pub last_frame_time: Option<Instant>,
    pub start_playing_time: Option<Instant>,
    pub video_is_playing: bool,
    pub video_start_playing_time: Option<Instant>,
    pub video_current_sequence_timeline: Option<SavedTimelineStateConfig>,
    pub video_current_sequences_data: Option<Vec<Sequence>>,

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

        let font_manager = FontManager::new();

        Editor {
            font_manager,
            selected_polygon_id: Uuid::nil(),
            polygons: Vec::new(),
            dragging_polygon: None,
            drag_start: None,
            viewport: viewport.clone(),
            handle_polygon_click: None,
            handle_text_click: None,
            handle_image_click: None,
            gpu_resources: None,
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
            is_playing: false,
            current_sequence_data: None,
            last_frame_time: None,
            start_playing_time: None,
            model_bind_group_layout: None,
            window_size_bind_group: None,
            static_polygons: Vec::new(),
            on_mouse_up: None,
            current_view: "manage_projects".to_string(),
            project_selected: None,
            text_items: Vec::new(),
            dragging_text: None,
            image_items: Vec::new(),
            dragging_image: None,
            video_is_playing: false,
            video_start_playing_time: None,
            video_current_sequence_timeline: None,
            video_current_sequences_data: None,
        }
    }

    pub fn step_video_animations(&mut self, camera: &Camera) {
        if !self.video_is_playing || self.video_current_sequence_timeline.is_none() {
            return;
        }

        let now = std::time::Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            (now - last_time).as_secs_f32()
        } else {
            0.0
        };
        let total_dt = if let Some(video_start_playing_time) = self.video_start_playing_time {
            (now - video_start_playing_time).as_secs_f32()
        } else {
            0.0
        };
        // self.last_frame_time = Some(now);

        let sequence_timeline = self
            .video_current_sequence_timeline
            .as_ref()
            .expect("Couldn't get current sequence timeline");

        // Convert total_dt from seconds to milliseconds for comparison with timeline
        let current_time_ms = (total_dt * 1000.0) as i32;

        // Get the sequences data
        let video_current_sequences_data = match self.video_current_sequences_data.as_ref() {
            Some(data) => data,
            None => return,
        };

        // Iterate through timeline sequences in order
        for ts in &sequence_timeline.timeline_sequences {
            // Skip audio tracks as we're only handling video
            if ts.track_type != TrackType::Video {
                continue;
            }

            // Check if this sequence should be playing at the current time
            if current_time_ms >= ts.start_time_ms
                && current_time_ms < (ts.start_time_ms + ts.duration_ms)
            {
                // Find the corresponding sequence data
                if let Some(sequence) = video_current_sequences_data.iter().find(|s| s.id == ts.id)
                {
                    // Calculate local time within this sequence
                    let sequence_local_time = (current_time_ms - ts.start_time_ms) as f32 / 1000.0;
                    if let Some(current_sequence) = &self.current_sequence_data {
                        // TODO: need to somehow efficiently restore polygons for the sequence
                        // Check id to avoid unnecessary cloning
                        if sequence.id != current_sequence.id {
                            self.current_sequence_data = Some(sequence.clone());
                        }
                    } else {
                        self.current_sequence_data = Some(sequence.clone());
                    }
                }
            }
        }
    }

    pub fn step_motion_path_animations(&mut self, camera: &Camera) {
        if !self.is_playing || self.current_sequence_data.is_none() {
            return;
        }

        let now = std::time::Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            (now - last_time).as_secs_f32()
        } else {
            0.0
        };
        let total_dt = if let Some(start_playing_time) = self.start_playing_time {
            (now - start_playing_time).as_secs_f32()
        } else {
            0.0
        };
        self.last_frame_time = Some(now);

        self.step_animate_sequence(total_dt, camera);
    }

    /// Steps the currently selected sequence unless one is provided
    pub fn step_animate_sequence(
        &mut self,
        // chosen_sequence: Option<&Sequence>,
        total_dt: f32,
        camera: &Camera,
    ) {
        let sequence = self
            .current_sequence_data
            .as_ref()
            .expect("Couldn't get sequence");

        // Update each animation path
        for animation in &sequence.polygon_motion_paths {
            // Find the polygon to update
            let object_idx = match animation.object_type {
                ObjectType::Polygon => {
                    let polygon_idx = self
                        .polygons
                        .iter()
                        .position(|p| p.id.to_string() == animation.polygon_id);

                    polygon_idx
                }
                ObjectType::TextItem => {
                    let text_idx = self
                        .text_items
                        .iter()
                        .position(|t| t.id.to_string() == animation.polygon_id);

                    text_idx
                }
                ObjectType::ImageItem => {
                    let image_idx = self
                        .image_items
                        .iter()
                        .position(|i| i.id.to_string() == animation.polygon_id);

                    image_idx
                }
            };

            let Some(object_idx) = object_idx else {
                continue;
            };

            // Go through each property
            for property in &animation.properties {
                if property.keyframes.len() < 2 {
                    continue;
                }

                // Get current time within animation duration
                let current_time =
                    Duration::from_secs_f32((total_dt % animation.duration.as_secs_f32()));

                // println!("current_time {:?} {:?}", current_time, total_dt);

                // Find the surrounding keyframes
                let (start_frame, end_frame) =
                    self.get_surrounding_keyframes(&property.keyframes, current_time);
                let Some((start_frame, end_frame)) = start_frame.zip(end_frame) else {
                    continue;
                };

                // Calculate interpolation progress
                let duration = (end_frame.time - start_frame.time).as_secs_f32();
                let elapsed = (current_time - start_frame.time).as_secs_f32();
                let mut progress = elapsed / duration;

                // Apply easing (EaseInOut)
                progress = if progress < 0.5 {
                    2.0 * progress * progress
                } else {
                    1.0 - (-2.0 * progress + 2.0).powi(2) / 2.0
                };

                // println!(
                //     "Polygon Progress {:?} {:?} {:?}",
                //     duration, elapsed, progress
                // );

                // Apply interpolated value based on property type
                match (&start_frame.value, &end_frame.value) {
                    (KeyframeValue::Position(start), KeyframeValue::Position(end)) => {
                        let x = self.lerp(start[0], end[0], progress);
                        let y = self.lerp(start[1], end[1], progress);

                        let position = Point {
                            x: 600.0 + x,
                            y: 50.0 + y,
                        };

                        match animation.object_type {
                            ObjectType::Polygon => {
                                self.polygons[object_idx]
                                    .transform
                                    .update_position([position.x, position.y], &camera.window_size);
                            }
                            ObjectType::TextItem => {
                                self.text_items[object_idx]
                                    .transform
                                    .update_position([position.x, position.y], &camera.window_size);
                            }
                            ObjectType::ImageItem => {
                                self.image_items[object_idx]
                                    .transform
                                    .update_position([position.x, position.y], &camera.window_size);
                            }
                        }
                    }
                    (KeyframeValue::Rotation(start), KeyframeValue::Rotation(end)) => {
                        // self.polygons[polygon_idx].rotation = self.lerp(*start, *end, progress);
                    }
                    (KeyframeValue::Scale(start), KeyframeValue::Scale(end)) => {
                        // self.polygons[polygon_idx].scale =
                        //     self.lerp(*start, *end, progress) as f32 / 100.0;
                    }
                    (KeyframeValue::Opacity(start), KeyframeValue::Opacity(end)) => {
                        // self.polygons[polygon_idx].opacity =
                        //     self.lerp(*start, *end, progress) as f32 / 100.0;

                        match animation.object_type {
                            ObjectType::Polygon => {
                                self.polygons[object_idx].fill =
                                    [1.0, 1.0, 1.0, self.lerp(*start, *end, progress) / 100.0];
                            }
                            ObjectType::TextItem => {}
                            ObjectType::ImageItem => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn get_surrounding_keyframes<'a>(
        &self,
        keyframes: &'a [UIKeyframe],
        current_time: Duration,
    ) -> (Option<&'a UIKeyframe>, Option<&'a UIKeyframe>) {
        let mut prev_frame = None;
        let mut next_frame = None;

        for (i, frame) in keyframes.iter().enumerate() {
            if frame.time > current_time {
                next_frame = Some(frame);
                prev_frame = if i > 0 {
                    Some(&keyframes[i - 1])
                } else {
                    Some(&keyframes[keyframes.len() - 1])
                };
                break;
            }
        }

        // Handle wrap-around case
        if next_frame.is_none() {
            prev_frame = keyframes.last();
            next_frame = keyframes.first();
        }

        (prev_frame, next_frame)
    }

    pub fn lerp(&self, start: i32, end: i32, progress: f32) -> f32 {
        start as f32 + ((end - start) as f32 * progress)
    }

    /// Create motion path visualization for a polygon
    pub fn create_motion_path_visualization(&mut self, sequence: &Sequence, polygon_id: &str) {
        let animation_data = sequence
            .polygon_motion_paths
            .iter()
            .find(|anim| anim.polygon_id == polygon_id)
            .expect("Couldn't find animation data for polygon");

        // Find position property
        let position_property = animation_data
            .properties
            .iter()
            .find(|prop| prop.name.starts_with("Position"))
            .expect("Couldn't find position property");

        // Sort keyframes by time
        let mut keyframes = position_property.keyframes.clone();
        keyframes.sort_by_key(|k| k.time);

        // Create path segments between consecutive keyframes
        for window in keyframes.windows(2) {
            let start_kf = &window[0];
            let end_kf = &window[1];

            if let (KeyframeValue::Position(start_pos), KeyframeValue::Position(end_pos)) =
                (&start_kf.value, &end_kf.value)
            {
                let start_point = Point {
                    x: start_pos[0] as f32,
                    y: start_pos[1] as f32,
                };
                let end_point = Point {
                    x: end_pos[0] as f32,
                    y: end_pos[1] as f32,
                };

                // Create intermediate points for curved paths if using non-linear easing
                let num_segments = match start_kf.easing {
                    EasingType::Linear => 1,
                    _ => 10, // More segments for curved paths
                };

                for i in 0..num_segments {
                    let t1 = start_kf.time + (end_kf.time - start_kf.time) * i / num_segments;
                    let t2 = start_kf.time + (end_kf.time - start_kf.time) * (i + 1) / num_segments;

                    let pos1 = interpolate_position(start_kf, end_kf, t1);
                    let pos2 = interpolate_position(start_kf, end_kf, t2);

                    let camera = self.camera.expect("Couldn't get camera");

                    let segment = create_path_segment(
                        &camera.window_size,
                        &self
                            .gpu_resources
                            .as_ref()
                            .expect("No GPU resources")
                            .device,
                        &self
                            .model_bind_group_layout
                            .as_ref()
                            .expect("No bind group layout"),
                        &self.camera.expect("No camera"),
                        Point {
                            x: pos1[0] as f32,
                            y: pos1[1] as f32,
                        },
                        Point {
                            x: pos2[0] as f32,
                            y: pos2[1] as f32,
                        },
                        2.0, // thickness of the path
                    );

                    self.static_polygons.push(segment);
                }
            }
        }
    }

    /// Update the motion path visualization when keyframes change
    pub fn update_motion_paths(&mut self, sequence: &Sequence) {
        // Remove existing motion path segments
        self.static_polygons
            .retain(|p| p.name != "motion_path_segment");

        // Recreate motion paths for all polygons
        for polygon_config in &sequence.active_polygons {
            self.create_motion_path_visualization(sequence, &polygon_config.id);
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

    pub fn add_polygon(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        camera: &Camera,
        polygon_config: PolygonConfig,
        polygon_name: String,
        new_id: Uuid,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let mut polygon = Polygon::new(
            window_size,
            device,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            camera,
            polygon_config.points,
            polygon_config.dimensions,
            polygon_config.position,
            0.0,
            polygon_config.border_radius,
            polygon_config.fill,
            Stroke {
                thickness: 2.0,
                fill: rgb_to_wgpu(0, 0, 0, 1.0),
            },
            0.0,
            polygon_name,
            new_id,
        );
        // // let world_position = camera.screen_to_world(polygon.transform.position);
        // let world_position = polygon.transform.position;
        // println!(
        //     "add polygon position {:?} {:?}",
        //     world_position, polygon.transform.position
        // );
        // // polygon.transform.position = world_position;
        // polygon
        //     .transform
        //     .update_position([world_position.x, world_position.y]);
        self.polygons.push(polygon);
        // self.run_layers_update();

        // TODO: udpate motion paths when adding new polygon
        // self.update_motion_paths(sequence);
    }

    pub fn add_text_item(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        text_config: TextRendererConfig,
        text_content: String,
        new_id: Uuid,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");

        let default_font_family = self
            .font_manager
            .get_font_by_name("Aleo")
            .expect("Couldn't load default font family");

        let mut text_item = TextRenderer::new(
            device,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            default_font_family, // load font data ahead of time
            window_size,
            text_content.clone(),
            text_config,
            new_id,
        );

        self.text_items.push(text_item);
    }

    pub fn add_image_item(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image_config: StImageConfig,
        path: &Path,
        new_id: Uuid,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let mut image_item = StImage::new(
            device,
            queue,
            path,
            image_config, // load font data ahead of time
            window_size,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            0.0,
            new_id.to_string(),
        );

        self.image_items.push(image_item);
    }

    pub fn update_polygon(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // First iteration: find the index of the selected polygon
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            println!("Found selected polygon with ID: {}", selected_id);

            let camera = self.camera.expect("Couldn't get camera");

            // Get the necessary data from editor
            let viewport_width = camera.window_size.width;
            let viewport_height = camera.window_size.height;
            let device = &self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources")
                .device;

            let window_size = WindowSize {
                width: viewport_width as u32,
                height: viewport_height as u32,
            };

            // Second iteration: update the selected polygon
            if let Some(selected_polygon) = self.polygons.get_mut(index) {
                match new_value {
                    InputValue::Text(s) => match key {
                        _ => println!("No match on input"),
                    },
                    InputValue::Number(n) => match key {
                        "width" => selected_polygon.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (n, selected_polygon.dimensions.1),
                            &camera,
                        ),
                        "height" => selected_polygon.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (selected_polygon.dimensions.0, n),
                            &camera,
                        ),
                        "border_radius" => selected_polygon.update_data_from_border_radius(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            n,
                            &camera,
                        ),
                        "red" => selected_polygon.update_data_from_fill(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            [
                                color_to_wgpu(n),
                                selected_polygon.fill[1],
                                selected_polygon.fill[2],
                                selected_polygon.fill[3],
                            ],
                            &camera,
                        ),
                        "green" => selected_polygon.update_data_from_fill(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            [
                                selected_polygon.fill[0],
                                color_to_wgpu(n),
                                selected_polygon.fill[2],
                                selected_polygon.fill[3],
                            ],
                            &camera,
                        ),
                        "blue" => selected_polygon.update_data_from_fill(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            [
                                selected_polygon.fill[0],
                                selected_polygon.fill[1],
                                color_to_wgpu(n),
                                selected_polygon.fill[3],
                            ],
                            &camera,
                        ),
                        "stroke_thickness" => selected_polygon.update_data_from_stroke(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            Stroke {
                                thickness: n,
                                fill: selected_polygon.stroke.fill,
                            },
                            &camera,
                        ),
                        "stroke_red" => selected_polygon.update_data_from_stroke(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            Stroke {
                                thickness: selected_polygon.stroke.thickness,
                                fill: [
                                    color_to_wgpu(n),
                                    selected_polygon.stroke.fill[1],
                                    selected_polygon.stroke.fill[2],
                                    selected_polygon.stroke.fill[3],
                                ],
                            },
                            &camera,
                        ),
                        "stroke_green" => selected_polygon.update_data_from_stroke(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            Stroke {
                                thickness: selected_polygon.stroke.thickness,
                                fill: [
                                    selected_polygon.stroke.fill[0],
                                    color_to_wgpu(n),
                                    selected_polygon.stroke.fill[2],
                                    selected_polygon.stroke.fill[3],
                                ],
                            },
                            &camera,
                        ),
                        "stroke_blue" => selected_polygon.update_data_from_stroke(
                            &window_size,
                            &device,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            Stroke {
                                thickness: selected_polygon.stroke.thickness,
                                fill: [
                                    selected_polygon.stroke.fill[0],
                                    selected_polygon.stroke.fill[1],
                                    color_to_wgpu(n),
                                    selected_polygon.stroke.fill[3],
                                ],
                            },
                            &camera,
                        ),
                        _ => println!("No match on input"),
                    },
                }
            }
        } else {
            println!("No polygon found with the selected ID: {}", selected_id);
        }
    }

    pub fn get_polygon_width(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.dimensions.0;
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_height(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.dimensions.1;
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_red(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.fill[0];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_green(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.fill[1];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_blue(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.fill[2];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_border_radius(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.border_radius;
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_stroke_thickness(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.stroke.thickness;
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_stroke_red(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.stroke.fill[0];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_stroke_green(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.stroke.fill[1];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_polygon_stroke_blue(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.polygons.get(index) {
                return selected_polygon.stroke.fill[2];
            } else {
                return 0.0;
            }
        }

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

        let interactive_bounds = BoundingBox {
            min: Point { x: 550.0, y: 0.0 }, // account for aside width
            max: Point {
                x: window_size.width as f32,
                y: window_size.height as f32,
            },
        };

        if (self.last_screen.x < interactive_bounds.min.x
            || self.last_screen.x > interactive_bounds.max.x
            || self.last_screen.y < interactive_bounds.min.y
            || self.last_screen.y > interactive_bounds.max.y)
        {
            return None;
        }

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
                            position: Point {
                                x: polygon.transform.position.x,
                                y: polygon.transform.position.y,
                            },
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

        // Check if we're clicking on a text item to drag
        for (text_index, text_item) in self.text_items.iter_mut().enumerate() {
            if text_item.contains_point(&self.last_top_left, &camera) {
                self.dragging_text = Some(text_index);
                self.drag_start = Some(self.last_top_left);

                // TODO: make DRY with below
                if (self.handle_text_click.is_some()) {
                    let handler_creator = self
                        .handle_text_click
                        .as_ref()
                        .expect("Couldn't get handler");
                    let mut handle_click = handler_creator().expect("Couldn't get handler");
                    handle_click(
                        text_item.id,
                        TextRendererConfig {
                            id: text_item.id,
                            name: text_item.name.clone(),
                            text: text_item.text.clone(),
                            // points: polygon.points.clone(),
                            dimensions: text_item.dimensions,
                            position: Point {
                                x: text_item.transform.position.x,
                                y: text_item.transform.position.y,
                            },
                            // border_radius: polygon.border_radius,
                            // fill: polygon.fill,
                            // stroke: polygon.stroke,
                        },
                    );
                    self.selected_polygon_id = text_item.id; // TODO: separate property for each object type?
                                                             // polygon.old_points = Some(polygon.points.clone());
                }

                return None; // nothing to add to undo stack
            }
        }

        // Check if we're clicking on a image item to drag
        for (image_index, image_item) in self.image_items.iter_mut().enumerate() {
            if image_item.contains_point(&self.last_top_left, &camera) {
                self.dragging_image = Some(image_index);
                self.drag_start = Some(self.last_top_left);

                // TODO: make DRY with below
                if (self.handle_image_click.is_some()) {
                    let handler_creator = self
                        .handle_image_click
                        .as_ref()
                        .expect("Couldn't get handler");
                    let mut handle_click = handler_creator().expect("Couldn't get handler");
                    let uuid = Uuid::from_str(&image_item.id.clone())
                        .expect("Couldn't convert string to uuid");
                    handle_click(
                        uuid,
                        StImageConfig {
                            id: image_item.id.clone(),
                            name: image_item.name.clone(),
                            path: image_item.path.clone(),
                            // points: polygon.points.clone(),
                            dimensions: image_item.dimensions,
                            position: Point {
                                x: image_item.transform.position.x,
                                y: image_item.transform.position.y,
                            },
                            // border_radius: polygon.border_radius,
                            // fill: polygon.fill,
                            // stroke: polygon.stroke,
                        },
                    );
                    self.selected_polygon_id = uuid; // TODO: separate property for each object type?
                                                     // polygon.old_points = Some(polygon.points.clone());
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

        // self.update_cursor();

        // handle dragging to move objects (polygons, images, text, etc)
        if let Some(poly_index) = self.dragging_polygon {
            if let Some(start) = self.drag_start {
                self.move_polygon(self.last_top_left, start, poly_index, window_size, device);
            }
        }

        if let Some(text_index) = self.dragging_text {
            if let Some(start) = self.drag_start {
                self.move_text(self.last_top_left, start, text_index, window_size, device);
            }
        }

        if let Some(image_index) = self.dragging_image {
            if let Some(start) = self.drag_start {
                self.move_image(self.last_top_left, start, image_index, window_size, device);
            }
        }
    }

    pub fn handle_mouse_up(&mut self) -> Option<PolygonEditConfig> {
        let mut action_edit = None;

        let camera = self.camera.expect("Couldn't get camera");

        let interactive_bounds = BoundingBox {
            min: Point { x: 550.0, y: 0.0 }, // account for aside width
            max: Point {
                x: camera.window_size.width as f32,
                y: camera.window_size.height as f32,
            },
        };

        if (self.last_screen.x < interactive_bounds.min.x
            || self.last_screen.x > interactive_bounds.max.x
            || self.last_screen.y < interactive_bounds.min.y
            || self.last_screen.y > interactive_bounds.max.y)
        {
            return None;
        }

        if let Some(poly_index) = self.dragging_polygon {
            if let Some(on_mouse_up_creator) = &self.on_mouse_up {
                let mut on_up = on_mouse_up_creator().expect("Couldn't get on handler");
                let selected_sequence_data = on_up(
                    poly_index,
                    Point {
                        x: self.last_top_left.x - 600.0,
                        y: self.last_top_left.y - 50.0,
                    },
                );
                self.update_motion_paths(&selected_sequence_data);
                println!("Motion Paths updated!");
            }
        }

        self.dragging_polygon = None;
        self.dragging_text = None;
        self.dragging_image = None;
        self.drag_start = None;

        // self.dragging_edge = None;
        // self.is_panning = false;
        // self.is_brushing = false;
        // self.guide_lines.clear();
        // self.update_cursor();

        action_edit
    }

    pub fn move_polygon(
        &mut self,
        mouse_pos: Point,
        start: Point,
        poly_index: usize,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let polygon = &mut self.polygons[poly_index];
        let new_position = Point {
            x: polygon.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio?
            y: polygon.transform.position.y + dy,
        };

        println!("move_polygon {:?}", new_position);

        polygon.update_data_from_position(
            window_size,
            device,
            self.model_bind_group_layout
                .as_ref()
                .expect("Couldn't get bind group layout"),
            new_position,
            &camera,
        );

        self.drag_start = Some(mouse_pos);
        // self.update_guide_lines(poly_index, window_size);
    }

    pub fn move_text(
        &mut self,
        mouse_pos: Point,
        start: Point,
        text_index: usize,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let text_item = &mut self.text_items[text_index];
        let new_position = Point {
            x: text_item.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio?
            y: text_item.transform.position.y + dy,
        };

        println!("move_text {:?}", new_position);

        text_item
            .transform
            .update_position([new_position.x, new_position.y], window_size);

        self.drag_start = Some(mouse_pos);
        // self.update_guide_lines(poly_index, window_size);
    }

    pub fn move_image(
        &mut self,
        mouse_pos: Point,
        start: Point,
        image_index: usize,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let image_item = &mut self.image_items[image_index];
        let new_position = Point {
            x: image_item.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio?
            y: image_item.transform.position.y + dy,
        };

        println!("move_image {:?}", new_position);

        image_item
            .transform
            .update_position([new_position.x, new_position.y], window_size);

        self.drag_start = Some(mouse_pos);
        // self.update_guide_lines(poly_index, window_size);
    }

    fn is_close(&self, a: f32, b: f32, threshold: f32) -> bool {
        (a - b).abs() < threshold
    }
}

/// Creates a path segment using a rotated square
fn create_path_segment(
    window_size: &WindowSize,
    device: &wgpu::Device,
    model_bind_group_layout: &Arc<wgpu::BindGroupLayout>,
    camera: &Camera,
    start: Point,
    end: Point,
    thickness: f32,
) -> Polygon {
    // Calculate rotation angle from start to end point
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let rotation = dy.atan2(dx);

    // Calculate length of the segment
    let length = (dx * dx + dy * dy).sqrt();

    // Calculate segment midpoint for position
    let position = Point {
        x: (start.x + end.x) / 2.0,
        y: (start.y + end.y) / 2.0,
    };

    // Create polygon using default square points
    Polygon::new(
        window_size,
        device,
        model_bind_group_layout,
        camera,
        vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ],
        (length, thickness), // width = length of segment, height = thickness
        position,
        rotation,
        0.0,
        [0.5, 0.8, 1.0, 1.0], // light blue with some transparency
        Stroke {
            thickness: 0.0,
            fill: rgb_to_wgpu(0, 0, 0, 1.0),
        },
        -1.0,
        String::from("motion_path_segment"),
        Uuid::new_v4(),
    )
}

/// Get interpolated position at a specific time
fn interpolate_position(start: &UIKeyframe, end: &UIKeyframe, time: Duration) -> [i32; 2] {
    if let (KeyframeValue::Position(start_pos), KeyframeValue::Position(end_pos)) =
        (&start.value, &end.value)
    {
        let progress = match start.easing {
            EasingType::Linear => {
                let total_time = (end.time - start.time).as_secs_f32();
                let current_time = (time - start.time).as_secs_f32();
                current_time / total_time
            }
            // Add more sophisticated easing calculations here
            _ => {
                let total_time = (end.time - start.time).as_secs_f32();
                let current_time = (time - start.time).as_secs_f32();
                current_time / total_time
            }
        };

        [
            (start_pos[0] as f32 + (end_pos[0] - start_pos[0]) as f32 * progress) as i32,
            (start_pos[1] as f32 + (end_pos[1] - start_pos[1]) as f32 * progress) as i32,
        ]
    } else {
        panic!("Expected position keyframes")
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

use crate::animations::{
    AnimationData, EasingType, KeyframeValue, ObjectType, Sequence, UIKeyframe,
};
use crate::camera::{Camera, CameraBinding};
use crate::fonts::FontManager;
use crate::polygon::{Polygon, PolygonConfig, Stroke};
use crate::st_image::{StImage, StImageConfig};
use crate::text_due::{TextRenderer, TextRendererConfig};
use crate::timelines::{SavedTimelineStateConfig, TrackType};

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
