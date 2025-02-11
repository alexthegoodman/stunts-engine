use std::cell::RefCell;
use std::fmt::Display;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use cgmath::{Matrix4, Point3, Vector2, Vector3, Vector4};
use common_motion_2d_reg::inference::CommonMotionInference;
use common_motion_2d_reg::interface::load_common_motion_2d;
use common_motion_2d_reg::Wgpu;
use floem_renderer::gpu_resources::{self, GpuResources};
use floem_winit::keyboard::ModifiersState;
use floem_winit::window::Window;
use rand::Rng;
use serde::{Deserialize, Serialize};
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

const NUM_INFERENCE_FEATURES: usize = 7;
pub const CANVAS_HORIZ_OFFSET: f32 = 600.0;
pub const CANVAS_VERT_OFFSET: f32 = 50.0;

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

pub fn size_to_normal(window_size: &WindowSize, x: f32, y: f32) -> (f32, f32) {
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
        // a.clamp(0.0, 1.0),
        a / 255.0,
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

pub fn string_to_u32(s: &str) -> Result<u32, std::num::ParseIntError> {
    let trimmed = s.trim();

    if trimmed.is_empty() {
        return Ok(0);
    }

    // Check if there's at least one digit in the string
    if !trimmed.chars().any(|c| c.is_ascii_digit()) {
        return Ok(0);
    }

    // At this point, we know there's at least one digit, so let's try to parse
    match trimmed.parse::<u32>() {
        Ok(num) => Ok(num),
        Err(e) => Err(e),
    }
}

// pub struct GuideLine {
//     pub start: Point,
//     pub end: Point,
// }

// Define all possible edit operations
#[derive(Debug)]
pub enum ObjectProperty {
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
pub struct ObjectEditConfig {
    pub object_id: Uuid,
    pub object_type: ObjectType,
    pub field_name: String,
    pub old_value: ObjectProperty,
    pub new_value: ObjectProperty,
    // pub signal: RwSignal<String>,
}

pub type PolygonClickHandler = dyn Fn() -> Option<Box<dyn FnMut(Uuid, PolygonConfig)>>;

pub type TextItemClickHandler = dyn Fn() -> Option<Box<dyn FnMut(Uuid, TextRendererConfig)>>;

pub type ImageItemClickHandler = dyn Fn() -> Option<Box<dyn FnMut(Uuid, StImageConfig)>>;

pub type VideoItemClickHandler = dyn Fn() -> Option<Box<dyn FnMut(Uuid, StVideoConfig)>>;

pub type OnMouseUp = dyn Fn() -> Option<Box<dyn FnMut(Uuid, Point) -> (Sequence, Vec<UIKeyframe>)>>;

pub type OnHandleMouseUp =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, Uuid, Point) -> (Sequence, Vec<UIKeyframe>)>>;

pub type OnPathMouseUp =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, Point) -> (Sequence, Vec<UIKeyframe>)>>;

#[derive(Eq, PartialEq, Clone, Copy, EnumIter, Debug)]
pub enum ControlMode {
    Select,
    Pan,
}

pub struct Editor {
    // visual
    pub selected_polygon_id: Uuid,
    pub polygons: Vec<Polygon>,
    pub dragging_polygon: Option<Uuid>,
    pub static_polygons: Vec<Polygon>,
    pub project_selected: Option<Uuid>,
    pub text_items: Vec<TextRenderer>,
    pub dragging_text: Option<Uuid>,
    pub image_items: Vec<StImage>,
    pub dragging_image: Option<Uuid>,
    pub font_manager: FontManager,
    pub dragging_path: Option<Uuid>,
    pub dragging_path_handle: Option<Uuid>,
    pub dragging_path_object: Option<Uuid>,
    pub dragging_path_keyframe: Option<Uuid>,
    pub dragging_path_assoc_path: Option<Uuid>,
    pub cursor_dot: Option<RingDot>,
    pub video_items: Vec<StVideo>,
    pub dragging_video: Option<Uuid>,
    pub motion_paths: Vec<MotionPath>,

    // viewport
    pub viewport: Arc<Mutex<Viewport>>,
    pub handle_polygon_click: Option<Arc<PolygonClickHandler>>,
    pub handle_text_click: Option<Arc<TextItemClickHandler>>,
    pub handle_image_click: Option<Arc<ImageItemClickHandler>>,
    pub handle_video_click: Option<Arc<VideoItemClickHandler>>,
    pub gpu_resources: Option<Arc<GpuResources>>,
    pub window: Option<Arc<Window>>,
    pub camera: Option<Camera>,
    pub camera_binding: Option<CameraBinding>,
    pub model_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pub group_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pub window_size_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pub window_size_bind_group: Option<wgpu::BindGroup>,
    pub window_size_buffer: Option<Arc<wgpu::Buffer>>,
    pub on_mouse_up: Option<Arc<OnMouseUp>>,
    pub on_handle_mouse_up: Option<Arc<OnHandleMouseUp>>,
    pub on_path_mouse_up: Option<Arc<OnPathMouseUp>>,
    pub current_view: String,
    pub interactive_bounds: BoundingBox,

    // state
    pub is_playing: bool,
    pub current_sequence_data: Option<Sequence>,
    pub last_frame_time: Option<Instant>,
    pub start_playing_time: Option<Instant>,
    pub video_is_playing: bool,
    pub video_start_playing_time: Option<Instant>,
    pub video_current_sequence_timeline: Option<SavedTimelineStateConfig>,
    pub video_current_sequences_data: Option<Vec<Sequence>>,
    pub control_mode: ControlMode,
    pub is_panning: bool,

    // points
    pub last_mouse_pos: Option<Point>,
    pub drag_start: Option<Point>,
    pub last_screen: Point, // last mouse position from input event top-left origin
    pub last_world: Point,
    pub last_top_left: Point,   // for inside the editor zone
    pub global_top_left: Point, // for when recording mouse positions outside the editor zone
    pub ds_ndc_pos: Point,      // double-width sized ndc-style positioning (screen-oriented)
    pub ndc: Point,
    pub previous_top_left: Point,

    // ai
    pub inference: Option<CommonMotionInference<Wgpu>>,
    pub generation_count: u32,
    pub generation_curved: bool,
}

use std::borrow::{Borrow, BorrowMut};

#[cfg(target_os = "windows")]
pub fn init_editor_with_model(viewport: Arc<Mutex<Viewport>>) -> Editor {
    let inference = load_common_motion_2d();

    let editor = Editor::new(viewport, Some(inference));

    editor
}

#[cfg(target_arch = "wasm32")]
pub fn init_editor_with_model(viewport: Arc<Mutex<Viewport>>) -> Editor {
    let editor = Editor::new(viewport, None);

    editor
}

pub enum InputValue {
    Text(String),
    Number(f32),
    // Points(Vec<Point>),
}

impl Editor {
    pub fn new(
        viewport: Arc<Mutex<Viewport>>,
        inference: Option<CommonMotionInference<Wgpu>>,
    ) -> Self {
        let viewport_unwrapped = viewport.lock().unwrap();
        let window_size = WindowSize {
            width: viewport_unwrapped.width as u32,
            height: viewport_unwrapped.height as u32,
        };

        let font_manager = FontManager::new();

        Editor {
            font_manager,
            inference,
            selected_polygon_id: Uuid::nil(),
            polygons: Vec::new(),
            dragging_polygon: None,
            dragging_path_assoc_path: None,
            drag_start: None,
            viewport: viewport.clone(),
            handle_polygon_click: None,
            handle_text_click: None,
            handle_image_click: None,
            handle_video_click: None,
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
            previous_top_left: Point { x: 0.0, y: 0.0 },
            is_playing: false,
            current_sequence_data: None,
            last_frame_time: None,
            start_playing_time: None,
            model_bind_group_layout: None,
            group_bind_group_layout: None,
            window_size_bind_group_layout: None,
            window_size_bind_group: None,
            window_size_buffer: None,
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
            dragging_path: None,
            dragging_path_handle: None,
            on_handle_mouse_up: None,
            on_path_mouse_up: None,
            dragging_path_object: None,
            dragging_path_keyframe: None,
            cursor_dot: None,
            control_mode: ControlMode::Select,
            is_panning: false,
            video_items: Vec::new(),
            dragging_video: None,
            motion_paths: Vec::new(),
            generation_count: 4,
            generation_curved: false,
            // TODO: update interactive bounds on window resize?
            interactive_bounds: BoundingBox {
                min: Point { x: 550.0, y: 0.0 }, // account for aside width, allow for some off-canvas positioning
                max: Point {
                    x: window_size.width as f32,
                    // y: window_size.height as f32 - 350.0, // 350.0 for timeline space
                    y: 550.0, // allow for 50.0 padding below and above the canvas
                },
            },
        }
    }

    pub fn restore_sequence_objects(
        &mut self,
        saved_sequence: &Sequence,
        window_size: WindowSize,
        camera: &Camera,
        hidden: bool,
        // device: &wgpu::Device,
        // queue: &wgpu::Queue,
    ) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");
        let device = &gpu_resources.device;
        let queue = &gpu_resources.queue;

        saved_sequence.active_polygons.iter().for_each(|p| {
            // let gpu_resources = self
            //     .gpu_resources
            //     .as_ref()
            //     .expect("Couldn't get GPU Resources");

            // Generate a random number between 0 and 800
            // let random_number_800 = rng.gen_range(0..=800);

            // Generate a random number between 0 and 450
            // let random_number_450 = rng.gen_range(0..=450);

            let mut restored_polygon = Polygon::new(
                &window_size,
                &device,
                &queue,
                &self
                    .model_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get model bind group layout"),
                &self
                    .group_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get group bind group layout"),
                &camera,
                // TODO: restoring triangles or non rectangles?
                vec![
                    Point { x: 0.0, y: 0.0 },
                    Point { x: 1.0, y: 0.0 },
                    Point { x: 1.0, y: 1.0 },
                    Point { x: 0.0, y: 1.0 },
                ],
                (p.dimensions.0 as f32, p.dimensions.1 as f32),
                Point {
                    // x: random_number_800 as f32,
                    // y: random_number_450 as f32,
                    x: p.position.x as f32,
                    y: p.position.y as f32,
                },
                // TODO: restore rotation?
                0.0,
                p.border_radius as f32,
                [
                    p.fill[0] as f32,
                    p.fill[1] as f32,
                    p.fill[2] as f32,
                    p.fill[3] as f32,
                ],
                Stroke {
                    thickness: p.stroke.thickness as f32,
                    fill: [
                        p.stroke.fill[0] as f32,
                        p.stroke.fill[1] as f32,
                        p.stroke.fill[2] as f32,
                        p.stroke.fill[3] as f32,
                    ],
                },
                -2.0,
                p.layer.clone(),
                p.name.clone(),
                Uuid::from_str(&p.id).expect("Couldn't convert string to uuid"),
                Uuid::from_str(&saved_sequence.id.clone())
                    .expect("Couldn't convert string to uuid"),
            );

            restored_polygon.hidden = hidden;

            // editor.add_polygon(restored_polygon);
            self.polygons.push(restored_polygon);

            println!("Polygon restored...");
        });

        saved_sequence.active_text_items.iter().for_each(|t| {
            // let gpu_resources = self
            //     .gpu_resources
            //     .as_ref()
            //     .expect("Couldn't get GPU Resources");

            // TODO: save and restore chosen font

            let position = Point {
                x: 600.0 + t.position.x as f32,
                y: 50.0 + t.position.y as f32,
            };

            let mut restored_text = TextRenderer::new(
                &device,
                &queue,
                self.model_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get model bind group layout"),
                &self
                    .group_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get group bind group layout"),
                self.font_manager
                    .get_font_by_name(&t.font_family)
                    .expect("Couldn't get font family"),
                &window_size,
                t.text.clone(),
                TextRendererConfig {
                    id: Uuid::from_str(&t.id).expect("Couldn't convert uuid"),
                    name: t.name.clone(),
                    text: t.text.clone(),
                    font_family: t.font_family.clone(),
                    dimensions: (t.dimensions.0 as f32, t.dimensions.1 as f32),
                    position,
                    layer: t.layer.clone(),
                    color: t.color.clone(),
                    font_size: t.font_size.clone(),
                },
                Uuid::from_str(&t.id).expect("Couldn't convert string to uuid"),
                Uuid::from_str(&saved_sequence.id.clone())
                    .expect("Couldn't convert string to uuid"),
            );

            restored_text.hidden = hidden;

            restored_text.render_text(&device, &queue);

            // editor.add_polygon(restored_polygon);
            self.text_items.push(restored_text);

            println!("Text restored...");
        });

        saved_sequence.active_image_items.iter().for_each(|i| {
            // let gpu_resources = self
            //     .gpu_resources
            //     .as_ref()
            //     .expect("Couldn't get GPU Resources");

            let position = Point {
                x: 600.0 + i.position.x as f32,
                y: 50.0 + i.position.y as f32,
            };

            let image_config = StImageConfig {
                id: i.id.clone(),
                name: i.name.clone(),
                dimensions: i.dimensions.clone(),
                path: i.path.clone(),
                position,
                layer: i.layer.clone(),
            };

            let mut restored_image = StImage::new(
                &device,
                &queue,
                // string to Path
                Path::new(&i.path),
                image_config,
                &window_size,
                self.model_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get model bind group layout"),
                &self
                    .group_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get group bind group layout"),
                -2.0,
                i.id.clone(),
                Uuid::from_str(&saved_sequence.id.clone())
                    .expect("Couldn't convert string to uuid"),
            );

            restored_image.hidden = hidden;

            // editor.add_polygon(restored_polygon);
            self.image_items.push(restored_image);

            println!("Image restored...");
        });

        saved_sequence.active_video_items.iter().for_each(|i| {
            // let mut saved_mouse_path = None;
            let mut source_data_path = None;
            let mut stored_mouse_positions = None;
            if let Some(mouse_path) = &i.mouse_path {
                let mut mouse_pathbuf = Path::new(&mouse_path).to_path_buf();
                mouse_pathbuf.pop();
                source_data_path = Some(mouse_pathbuf.join("sourceData.json"));

                if let Ok(positions) = fs::read_to_string(mouse_path) {
                    if let Ok(mouse_positions) =
                        serde_json::from_str::<Vec<MousePosition>>(&positions)
                    {
                        // saved_mouse_path = Some(mouse_path);
                        stored_mouse_positions = Some(mouse_positions);
                    }
                }
            }

            let mut stored_source_data = None;
            if let Some(source_path) = &source_data_path {
                if let Ok(source_data) = fs::read_to_string(source_path) {
                    if let Ok(data) = serde_json::from_str::<SourceData>(&source_data) {
                        stored_source_data = Some(data);
                    }
                }
            }

            println!(
                "Restoring video source data... {:?} {:?}",
                source_data_path, stored_source_data
            );

            let position = Point {
                x: 600.0 + i.position.x as f32,
                y: 50.0 + i.position.y as f32,
            };

            let video_config = StVideoConfig {
                id: i.id.clone(),
                name: i.name.clone(),
                dimensions: i.dimensions.clone(),
                path: i.path.clone(),
                position,
                layer: i.layer.clone(),
                mouse_path: i.mouse_path.clone(),
            };

            let mut restored_video = StVideo::new(
                &device,
                &queue,
                // string to Path
                Path::new(&i.path),
                video_config,
                &window_size,
                self.model_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get model bind group layout"),
                &self
                    .group_bind_group_layout
                    .as_ref()
                    .expect("Couldn't get group bind group layout"),
                -2.0,
                i.id.clone(),
                Uuid::from_str(&saved_sequence.id.clone())
                    .expect("Couldn't convert string to uuid"),
            )
            .expect("Couldn't restore video");

            restored_video.hidden = hidden;

            // set window data from capture
            restored_video.source_data = stored_source_data;

            // set mouse positions
            restored_video.mouse_positions = stored_mouse_positions;

            // render 1 frame to provide preview image
            restored_video
                .draw_video_frame(device, queue)
                .expect("Couldn't draw video frame");

            // editor.add_polygon(restored_polygon);
            self.video_items.push(restored_video);

            println!("Video restored...");
        });
    }

    pub fn reset_sequence_objects(&mut self) {
        if let Some(current_sequence) = &self.current_sequence_data {
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get GPU Resources");
            let camera = self.camera.as_ref().expect("Couldn't get camera");

            // put all objects back in original positions
            current_sequence.active_polygons.iter().for_each(|p| {
                let polygon = self
                    .polygons
                    .iter_mut()
                    .find(|polygon| polygon.id.to_string() == p.id)
                    .expect("Couldn't find polygon");

                polygon.transform.position.x = p.position.x as f32 + CANVAS_HORIZ_OFFSET;
                polygon.transform.position.y = p.position.y as f32 + CANVAS_VERT_OFFSET;
                polygon.transform.rotation = 0.0;
                polygon.transform.update_scale([1.0, 1.0]);

                polygon
                    .transform
                    .update_uniform_buffer(&gpu_resources.queue, &camera.window_size);

                polygon.update_opacity(&gpu_resources.queue, 1.0);
            });

            current_sequence.active_text_items.iter().for_each(|t| {
                let text = self
                    .text_items
                    .iter_mut()
                    .find(|text| text.id.to_string() == t.id)
                    .expect("Couldn't find text");

                text.transform.position.x = t.position.x as f32 + CANVAS_HORIZ_OFFSET;
                text.transform.position.y = t.position.y as f32 + CANVAS_VERT_OFFSET;
                text.transform.rotation = 0.0;

                text.transform
                    .update_uniform_buffer(&gpu_resources.queue, &camera.window_size);

                text.update_opacity(&gpu_resources.queue, 1.0);

                // TODO: reset other properties once scale is figured out
            });

            current_sequence.active_image_items.iter().for_each(|i| {
                let image = self
                    .image_items
                    .iter_mut()
                    .find(|image| image.id == i.id)
                    .expect("Couldn't find image");

                image.transform.position.x = i.position.x as f32 + CANVAS_HORIZ_OFFSET;
                image.transform.position.y = i.position.y as f32 + CANVAS_VERT_OFFSET;

                image.transform.rotation = 0.0;

                image
                    .transform
                    .update_uniform_buffer(&gpu_resources.queue, &camera.window_size);

                image.update_opacity(&gpu_resources.queue, 1.0);

                // TODO: reset other properties once scale is figured out
            });

            current_sequence.active_video_items.iter().for_each(|i| {
                let video = self
                    .video_items
                    .iter_mut()
                    .find(|video| video.id == i.id)
                    .expect("Couldn't find image");

                video.transform.position.x = i.position.x as f32 + CANVAS_HORIZ_OFFSET;
                video.transform.position.y = i.position.y as f32 + CANVAS_VERT_OFFSET;

                video.transform.rotation = 0.0;

                video
                    .transform
                    .update_uniform_buffer(&gpu_resources.queue, &camera.window_size);

                video.update_opacity(&gpu_resources.queue, 1.0);

                video
                    .reset_playback()
                    .expect("Couldn't reset video playback");

                // TODO: reset other properties once scale is figured out
            });
        }
    }

    pub fn run_motion_inference(&self) -> Vec<AnimationData> {
        let mut prompt = "".to_string();
        let mut total = 0;
        for (i, polygon) in self.polygons.iter().enumerate() {
            if !polygon.hidden {
                let x = polygon.transform.position.x - 600.0;
                let x = (x / 800.0) * 100.0; // testing percentage based training
                let y = polygon.transform.position.y - 50.0;
                let y = (y / 450.0) * 100.0;

                prompt.push_str(&total.to_string());
                prompt.push_str(", ");
                prompt.push_str("5");
                prompt.push_str(", ");
                prompt.push_str(&polygon.dimensions.0.to_string());
                prompt.push_str(", ");
                prompt.push_str(&polygon.dimensions.1.to_string());
                prompt.push_str(", ");
                prompt.push_str(&(x.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str(&(y.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str("0.000"); // direction
                prompt.push_str(", ");
                prompt.push_str("\n");
                total = total + 1;
            }

            if (total > 6) {
                break;
            }
        }

        for (i, text) in self.text_items.iter().enumerate() {
            if !text.hidden {
                let x = text.transform.position.x - 600.0;
                let x = (x / 800.0) * 100.0; // testing percentage based training
                let y = text.transform.position.y - 50.0;
                let y = (y / 450.0) * 100.0;

                prompt.push_str(&total.to_string());
                prompt.push_str(", ");
                prompt.push_str("5");
                prompt.push_str(", ");
                prompt.push_str(&text.dimensions.0.to_string());
                prompt.push_str(", ");
                prompt.push_str(&text.dimensions.1.to_string());
                prompt.push_str(", ");
                prompt.push_str(&(x.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str(&(y.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str("0.000"); // direction
                prompt.push_str(", ");
                prompt.push_str("\n");
                total = total + 1;
            }
            if (total > 6) {
                break;
            }
        }

        for (i, image) in self.image_items.iter().enumerate() {
            if !image.hidden {
                let x = image.transform.position.x - 600.0;
                let x = (x / 800.0) * 100.0; // testing percentage based training
                let y = image.transform.position.y - 50.0;
                let y = (y / 450.0) * 100.0;

                prompt.push_str(&total.to_string());
                prompt.push_str(", ");
                prompt.push_str("5");
                prompt.push_str(", ");
                prompt.push_str(&image.dimensions.0.to_string());
                prompt.push_str(", ");
                prompt.push_str(&image.dimensions.1.to_string());
                prompt.push_str(", ");
                prompt.push_str(&(x.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str(&(y.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str("0.000"); // direction
                prompt.push_str(", ");
                prompt.push_str("\n");
                total = total + 1;
            }

            if (total > 6) {
                break;
            }
        }

        for (i, video) in self.video_items.iter().enumerate() {
            if !video.hidden {
                let x = video.transform.position.x - 600.0;
                let x = (x / 800.0) * 100.0; // testing percentage based training
                let y = video.transform.position.y - 50.0;
                let y = (y / 450.0) * 100.0;

                prompt.push_str(&total.to_string());
                prompt.push_str(", ");
                prompt.push_str("5");
                prompt.push_str(", ");
                prompt.push_str(&video.dimensions.0.to_string());
                prompt.push_str(", ");
                prompt.push_str(&video.dimensions.1.to_string());
                prompt.push_str(", ");
                prompt.push_str(&(x.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str(&(y.round() as i32).to_string());
                prompt.push_str(", ");
                prompt.push_str("0.000"); // direction
                prompt.push_str(", ");
                prompt.push_str("\n");
                total = total + 1;
            }

            if (total > 6) {
                break;
            }
        }

        println!("prompt {:?}", prompt);

        let inference = self.inference.as_ref().expect("Couldn't get inference");
        let predictions: Vec<f32> = inference
            // .infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
            .infer(prompt);

        // predictions are 6 rows per line in the prompt, with each row containing: `object_index, time, width, height, x, y`
        for (i, predicted) in predictions.clone().into_iter().enumerate() {
            if i % NUM_INFERENCE_FEATURES == 0 {
                println!();
            }
            print!("{}, ", predicted);
        }

        // create motion paths from predictions, each prediction must be rounded
        let motion_path_keyframes = self.create_motion_paths_from_predictions(predictions);

        motion_path_keyframes
    }

    pub fn create_motion_paths_from_predictions(
        &self,
        predictions: Vec<f32>,
    ) -> Vec<AnimationData> {
        let mut animation_data_vec = Vec::new();
        let values_per_prediction = NUM_INFERENCE_FEATURES; // object_index, time, width, height, x, y
        let keyframes_per_object = 6; // number of keyframes per object
        let timestamps = vec![0, 2500, 5000, 15000, 17500, 20000];

        // Calculate total number of objects from predictions
        let total_predictions = predictions.len();
        let num_objects = total_predictions / (values_per_prediction * keyframes_per_object);

        // Get the current positions of all objects
        let mut current_positions = Vec::new();
        let mut total = 0; // use controlled total as get_item_id function filters by hidden
        for (i, polygon) in self.polygons.iter().enumerate() {
            if !polygon.hidden {
                current_positions.push((
                    total,
                    polygon.transform.position.x - CANVAS_HORIZ_OFFSET,
                    polygon.transform.position.y - CANVAS_VERT_OFFSET,
                ));
                total = total + 1;
            }
        }
        for (i, text) in self.text_items.iter().enumerate() {
            if !text.hidden {
                current_positions.push((
                    total,
                    text.transform.position.x - CANVAS_HORIZ_OFFSET,
                    text.transform.position.y - CANVAS_VERT_OFFSET,
                ));
                total = total + 1;
            }
        }
        for (i, image) in self.image_items.iter().enumerate() {
            if !image.hidden {
                current_positions.push((
                    total,
                    image.transform.position.x - CANVAS_HORIZ_OFFSET,
                    image.transform.position.y - CANVAS_VERT_OFFSET,
                ));
                total = total + 1;
            }
        }
        for (i, video) in self.video_items.iter().enumerate() {
            if !video.hidden {
                current_positions.push((
                    total,
                    video.transform.position.x - CANVAS_HORIZ_OFFSET,
                    video.transform.position.y - CANVAS_VERT_OFFSET,
                ));
                total = total + 1;
            }
        }

        println!("current_positions length {:?}", current_positions.len());

        // Collect all 3rd keyframes (index 2) from predictions
        let mut third_keyframes = Vec::new();
        for object_idx in 0..num_objects {
            let base_idx = object_idx * (values_per_prediction * keyframes_per_object)
                + 2 * values_per_prediction; // 3rd keyframe (index 2)

            // Skip if out of bounds
            if base_idx + 5 >= predictions.len() {
                continue;
            }

            // percentage based predictions (800 is canvas width, 450 is canvas height)
            let predicted_x = ((predictions[base_idx + 4] * 0.01) * 800.0).round() as i32;
            let predicted_y = ((predictions[base_idx + 5] * 0.01) * 450.0).round() as i32;

            third_keyframes.push((object_idx, predicted_x, predicted_y));
        }

        println!("third_keyframes length {:?}", third_keyframes.len());

        // Create distance vector
        let mut distances = vec![vec![f64::MAX; third_keyframes.len()]; current_positions.len()];
        for (object_idx, (_, current_x, current_y)) in current_positions.iter().enumerate() {
            for (mp_object_idx, (_, predicted_x, predicted_y)) in third_keyframes.iter().enumerate()
            {
                let dx = *predicted_x as f32 - *current_x;
                let dy = *predicted_y as f32 - *current_y;
                let distance = (dx * dx + dy * dy).sqrt();
                distances[object_idx][mp_object_idx] = distance as f64;
            }
        }

        println!("distances length {:?}", distances.len());

        let motion_path_assignments = assign_motion_paths_to_objects(distances)
            .expect("Couldn't assign motion paths to objects");

        println!("motion_path_assignments {:?}", motion_path_assignments); // NOTE: for example, is [0,2,1] but should be [2,0,1]
                                                                           // println!("assigned_keyframes length {:?}", assigned_keyframes.len());

        // Create motion paths based on assignments
        for (object_idx, associated_object_idx) in motion_path_assignments.into_iter() {
            println!("object_idx {:?} {:?}", object_idx, associated_object_idx);

            let mut position_keyframes: Vec<UIKeyframe> = Vec::new();

            // Process keyframes for the assigned motion path
            for keyframe_time_idx in 0..keyframes_per_object {
                let base_idx = associated_object_idx
                    * (values_per_prediction * keyframes_per_object)
                    + keyframe_time_idx * values_per_prediction;

                // skip depending on chosen count
                if self.generation_count == 4 {
                    if keyframe_time_idx == 1 || keyframe_time_idx == 5 {
                        continue;
                    }
                }

                // Skip if out of bounds
                if base_idx + 5 >= predictions.len() {
                    continue;
                }

                // percentage based predictions (800 is canvas width, 450 is canvas height)
                let predicted_x = ((predictions[base_idx + 4] * 0.01) * 800.0).round() as i32;
                let predicted_y = ((predictions[base_idx + 5] * 0.01) * 450.0).round() as i32;

                let keyframe = UIKeyframe {
                    id: Uuid::new_v4().to_string(),
                    time: Duration::from_millis(timestamps[keyframe_time_idx] as u64),
                    value: KeyframeValue::Position([predicted_x, predicted_y]),
                    easing: EasingType::EaseInOut,
                    path_type: PathType::Linear,
                    // set the KeyType to Frame as default, with Range in place of 3rd and 4th keyframes next
                    key_type: KeyType::Frame,
                };

                // // Update path_type for previous keyframe if it exists
                // if let Some(prev_keyframe) = position_keyframes.last_mut() {
                //     prev_keyframe.path_type = prev_keyframe.calculate_default_curve(&keyframe);
                // }

                position_keyframes.push(keyframe);
            }

            // handle 6 keyframes
            if position_keyframes.len() == 6 {
                // set Range
                let forth_keyframe = &position_keyframes.clone()[3];
                let third_keyframe = &mut position_keyframes[2];

                third_keyframe.key_type = KeyType::Range(RangeData {
                    end_time: forth_keyframe.time,
                });

                position_keyframes.remove(3);
            }

            // handle 4 keyframes
            if position_keyframes.len() == 4 {
                // set Range
                let mid2_keyframe = &position_keyframes.clone()[2];
                let mid_keyframe = &mut position_keyframes[1];

                mid_keyframe.key_type = KeyType::Range(RangeData {
                    end_time: mid2_keyframe.time,
                });

                position_keyframes.remove(2);
            }

            let mut final_position_keyframes: Vec<UIKeyframe> = Vec::new();

            // create default curves between remaining keyframes
            if self.generation_curved {
                for (index, keyframe) in position_keyframes.clone().iter().enumerate() {
                    // // Update path_type for previous keyframe if it exists
                    if let Some(prev_keyframe) = final_position_keyframes.last_mut() {
                        prev_keyframe.path_type = prev_keyframe.calculate_default_curve(&keyframe);
                    }

                    final_position_keyframes.push(keyframe.clone());
                }
            } else {
                for (index, keyframe) in position_keyframes.clone().iter().enumerate() {
                    final_position_keyframes.push(keyframe.clone());
                }
            }

            // Get the item ID based on the object index
            let item_id = self.get_item_id(object_idx);
            let object_type = self.get_object_type(object_idx);

            println!("item_id {:?}", item_id);

            // Only create animation if we have valid keyframes and item ID
            if !final_position_keyframes.is_empty() && item_id.is_some() {
                let properties = vec![
                    // Position property with predicted values
                    AnimationProperty {
                        name: "Position".to_string(),
                        property_path: "position".to_string(),
                        children: Vec::new(),
                        keyframes: final_position_keyframes,
                        depth: 0,
                    },
                    // Default properties for rotation, scale, opacity
                    AnimationProperty {
                        name: "Rotation".to_string(),
                        property_path: "rotation".to_string(),
                        children: Vec::new(),
                        keyframes: timestamps
                            .iter()
                            .map(|&t| UIKeyframe {
                                id: Uuid::new_v4().to_string(),
                                time: Duration::from_millis(t as u64),
                                value: KeyframeValue::Rotation(0),
                                easing: EasingType::EaseInOut,
                                path_type: PathType::Linear,
                                // should be same as position? or safe to be independent?
                                key_type: KeyType::Frame,
                            })
                            .collect(),
                        depth: 0,
                    },
                    AnimationProperty {
                        name: "Scale".to_string(),
                        property_path: "scale".to_string(),
                        children: Vec::new(),
                        keyframes: timestamps
                            .iter()
                            .map(|&t| UIKeyframe {
                                id: Uuid::new_v4().to_string(),
                                time: Duration::from_millis(t as u64),
                                value: KeyframeValue::Scale(100),
                                easing: EasingType::EaseInOut,
                                path_type: PathType::Linear,
                                // should be same as position? or safe to be independent?
                                key_type: KeyType::Frame,
                            })
                            .collect(),
                        depth: 0,
                    },
                    AnimationProperty {
                        name: "Opacity".to_string(),
                        property_path: "opacity".to_string(),
                        children: Vec::new(),
                        keyframes: timestamps
                            .iter()
                            .map(|&t| UIKeyframe {
                                id: Uuid::new_v4().to_string(),
                                time: Duration::from_millis(t as u64),
                                value: KeyframeValue::Opacity(100),
                                easing: EasingType::EaseInOut,
                                path_type: PathType::Linear,
                                // should be same as position? or safe to be independent?
                                key_type: KeyType::Frame,
                            })
                            .collect(),
                        depth: 0,
                    },
                ];

                animation_data_vec.push(AnimationData {
                    id: Uuid::new_v4().to_string(),
                    object_type: object_type.unwrap_or(ObjectType::Polygon),
                    polygon_id: item_id.unwrap(),
                    duration: Duration::from_secs(20),
                    start_time_ms: 0,
                    position: [0, 0],
                    properties,
                });
            }
        }

        animation_data_vec
    }

    // Helper function to get item ID based on object index
    fn get_item_id(&self, object_idx: usize) -> Option<String> {
        // let polygon_count = self.polygons.len();
        // let text_count = self.text_items.len();
        let visible_polygons: Vec<&Polygon> = self.polygons.iter().filter(|p| !p.hidden).collect();
        let visible_texts: Vec<&TextRenderer> =
            self.text_items.iter().filter(|t| !t.hidden).collect();
        let visible_images: Vec<&StImage> = self.image_items.iter().filter(|i| !i.hidden).collect();
        let visible_videos: Vec<&StVideo> = self.video_items.iter().filter(|v| !v.hidden).collect();

        let polygon_count = self.polygons.iter().filter(|p| !p.hidden).count();
        let text_count = self.text_items.iter().filter(|t| !t.hidden).count();
        let image_count = self.image_items.iter().filter(|i| !i.hidden).count();

        match object_idx {
            idx if idx < polygon_count => Some(visible_polygons[idx].id.clone().to_string()),
            idx if idx < polygon_count + text_count => {
                Some(visible_texts[idx - polygon_count].id.clone().to_string())
            }
            idx if idx < polygon_count + text_count + visible_images.len() => Some(
                visible_images[idx - (polygon_count + text_count)]
                    .id
                    .clone(),
            ),
            idx if idx
                < polygon_count + text_count + visible_images.len() + visible_videos.len() =>
            {
                Some(
                    visible_videos[idx - (polygon_count + text_count + visible_images.len())]
                        .id
                        .clone(),
                )
            }
            _ => None,
        }
    }

    // Helper function to get object type based on object index
    fn get_object_type(&self, object_idx: usize) -> Option<ObjectType> {
        // let polygon_count = self.polygons.len();
        // let text_count = self.text_items.len();

        let polygon_count = self.polygons.iter().filter(|p| !p.hidden).count();
        let text_count = self.text_items.iter().filter(|t| !t.hidden).count();
        let image_count = self.image_items.iter().filter(|i| !i.hidden).count();
        let video_count = self.video_items.iter().filter(|i| !i.hidden).count();

        match object_idx {
            idx if idx < polygon_count => Some(ObjectType::Polygon),
            idx if idx < polygon_count + text_count => Some(ObjectType::TextItem),
            idx if idx < polygon_count + text_count + image_count => Some(ObjectType::ImageItem),
            idx if idx < polygon_count + text_count + image_count + video_count => {
                Some(ObjectType::VideoItem)
            }
            _ => None,
        }
    }

    pub fn step_video_animations(&mut self, camera: &Camera, provided_current_time_s: Option<f64>) {
        if !self.video_is_playing || self.video_current_sequence_timeline.is_none() {
            return;
        }

        let now = std::time::Instant::now();
        // let dt = if let Some(last_time) = self.last_frame_time {
        //     (now - last_time).as_secs_f32()
        // } else {
        //     0.0
        // };
        // let dt = if let Some(provided_dt) = provided_dt {
        //     provided_dt
        // } else {
        //     dt
        // };
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
        let current_time_ms = if let Some(provided_current_time_s) = provided_current_time_s {
            (provided_current_time_s * 1000.0) as i32
        } else {
            (total_dt * 1000.0) as i32
        };

        // Get the sequences data
        let video_current_sequences_data = match self.video_current_sequences_data.as_ref() {
            Some(data) => data,
            None => return,
        };

        // let mut elapsed = 0;
        // let mut current_found = false;

        let mut update_background = false;

        if total_dt <= 1.0 / 60.0 {
            println!("Update initial background...");
            update_background = true;
        }

        // Iterate through timeline sequences in order
        for ts in &sequence_timeline.timeline_sequences {
            // Skip audio tracks as we're only handling video
            if ts.track_type != TrackType::Video {
                continue;
            }

            // slow?
            let duration_ms = video_current_sequences_data
                .iter()
                .find(|s| s.id == ts.sequence_id)
                .map(|s| s.duration_ms)
                .unwrap_or(0);

            // dynamic start times
            // if let Some(current_sequence) = &self.current_sequence_data {
            //     if !current_found {
            //         elapsed = elapsed + ts.duration_ms;
            //     }

            //     if current_sequence.id == ts.sequence_id {
            //         current_found = true;
            //     }
            // } else {
            //     current_found = true;
            // }

            // if current_found {}
            // Check if this sequence should be playing at the current time
            if current_time_ms >= ts.start_time_ms
                && current_time_ms < (ts.start_time_ms + duration_ms)
            {
                // Find the corresponding sequence data
                if let Some(sequence) = video_current_sequences_data
                    .iter()
                    .find(|s| s.id == ts.sequence_id)
                {
                    // Calculate local time within this sequence
                    let sequence_local_time = (current_time_ms - ts.start_time_ms) as f32 / 1000.0;
                    if let Some(current_sequence) = &self.current_sequence_data {
                        // need to somehow efficiently restore polygons for the sequence
                        // Check id to avoid unnecessary cloning
                        // plan is to preload with a hidden attribute or similar
                        if sequence.id != current_sequence.id {
                            self.current_sequence_data = Some(sequence.clone());
                            // set hidden attribute on relevant objects
                            let current_sequence_id = sequence.id.clone();

                            for polygon in self.polygons.iter_mut() {
                                if polygon.current_sequence_id.to_string() == current_sequence_id {
                                    polygon.hidden = false;
                                } else {
                                    polygon.hidden = true;
                                }
                            }
                            for text in self.text_items.iter_mut() {
                                if text.current_sequence_id.to_string() == current_sequence_id {
                                    text.hidden = false;
                                } else {
                                    text.hidden = true;
                                }
                            }
                            for image in self.image_items.iter_mut() {
                                if image.current_sequence_id.to_string() == current_sequence_id {
                                    image.hidden = false;
                                } else {
                                    image.hidden = true;
                                }
                            }
                            for video in self.video_items.iter_mut() {
                                if video.current_sequence_id.to_string() == current_sequence_id {
                                    video.hidden = false;
                                } else {
                                    video.hidden = true;
                                }
                            }

                            update_background = true;
                        }
                    } else {
                        self.current_sequence_data = Some(sequence.clone());
                    }
                }
            }
        }

        {
            if update_background {
                if let Some(current_sequence) = &self.current_sequence_data {
                    match current_sequence
                        .background_fill
                        .as_ref()
                        .expect("Couldn't get default background fill")
                    {
                        BackgroundFill::Color(fill) => {
                            self.replace_background(
                                Uuid::from_str(&current_sequence.id)
                                    .expect("Couldn't convert string to uuid"),
                                rgb_to_wgpu(
                                    fill[0] as u8,
                                    fill[1] as u8,
                                    fill[2] as u8,
                                    fill[3] as f32,
                                ),
                            );
                        }
                        _ => {
                            println!("Not supported yet...");
                        }
                    }
                }
            }
        }
    }

    pub fn step_motion_path_animations(
        &mut self,
        camera: &Camera,
        provided_current_time_s: Option<f64>,
    ) {
        if !self.is_playing || self.current_sequence_data.is_none() {
            return;
        }

        // TODO: disable time based dt determination for export only
        let now = std::time::Instant::now();
        // let dt = if let Some(last_time) = self.last_frame_time {
        //     (now - last_time).as_secs_f32()
        // } else {
        //     0.0
        // };
        let total_dt = if let Some(start_playing_time) = self.start_playing_time {
            (now - start_playing_time).as_secs_f32()
        } else {
            0.0
        };
        let total_dt = if let Some(provided_current_time_s) = provided_current_time_s {
            provided_current_time_s
        } else {
            total_dt as f64
        };
        self.last_frame_time = Some(now);

        self.step_animate_sequence(total_dt as f32, camera);
    }

    /// Steps the currently selected sequence unless one is provided
    /// TODO: make more efficient
    pub fn step_animate_sequence(&mut self, total_dt: f32, camera: &Camera) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get GPU Resources");
        let sequence = self
            .current_sequence_data
            .as_ref()
            .expect("Couldn't get sequence");

        // Update each animation path
        for animation in &sequence.polygon_motion_paths {
            // Group transform position
            let path_group_position = animation.position;

            // Get current time within animation duration
            let current_time =
                Duration::from_secs_f32(total_dt % (sequence.duration_ms / 1000) as f32);
            let start_time = Duration::from_millis(animation.start_time_ms as u64);

            // Check if the current time is within the animation's active period
            if current_time < start_time || current_time > start_time + animation.duration {
                continue;
            }

            // Find the polygon to update
            let object_idx = match animation.object_type {
                ObjectType::Polygon => self
                    .polygons
                    .iter()
                    .position(|p| p.id.to_string() == animation.polygon_id),
                ObjectType::TextItem => self
                    .text_items
                    .iter()
                    .position(|t| t.id.to_string() == animation.polygon_id),
                ObjectType::ImageItem => self
                    .image_items
                    .iter()
                    .position(|i| i.id.to_string() == animation.polygon_id),
                ObjectType::VideoItem => self
                    .video_items
                    .iter()
                    .position(|i| i.id.to_string() == animation.polygon_id),
            };

            let Some(object_idx) = object_idx else {
                continue;
            };

            // Determine whether to draw the video frame based on the frame rate and current time
            // step rate is throttled to 60FPS
            // if video frame rate is 60FPS, then call draw on each frame
            // if video frame rate is 30FPS, then call draw on every other frame
            let mut animate_properties = false;

            if animation.object_type == ObjectType::VideoItem {
                let frame_rate = self.video_items[object_idx].source_frame_rate;
                let source_duration_ms = self.video_items[object_idx].source_duration_ms;
                let frame_interval = Duration::from_secs_f64(1.0 / frame_rate as f64);

                // Calculate the number of frames that should have been displayed by now
                let elapsed_time: Duration = current_time - start_time;
                let current_frame_time = self.video_items[object_idx].num_frames_drawn as f64
                    * frame_interval.as_secs_f64();
                // let last_frame_time = self.last_frame_time.expect("Couldn't get last frame time");

                // println!(
                //     "current times {:?} frame: {:?}",
                //     current_time.as_secs_f64(),
                //     current_frame_time
                // );

                // Only draw the frame if the current time is within the frame's display interval
                if (current_time.as_secs_f64() >= current_frame_time
                    && current_time.as_secs_f64()
                        < current_frame_time + frame_interval.as_secs_f64())
                {
                    if current_time.as_millis() + 1000 < source_duration_ms as u128 {
                        self.video_items[object_idx]
                            .draw_video_frame(&gpu_resources.device, &gpu_resources.queue)
                            .expect("Couldn't draw video frame");

                        animate_properties = true;
                        self.video_items[object_idx].num_frames_drawn += 1;
                    }
                } else {
                    // TODO: deteermine distance between current_time and current_frame_time to determine
                    // how many video frames to draw to catch up
                    let difference = current_time.as_secs_f64() - current_frame_time;
                    let catch_up_frames =
                        (difference / frame_interval.as_secs_f64()).floor() as u32;

                    // Only catch up if we're behind and within the video duration
                    if catch_up_frames > 0
                        && current_time.as_millis() + 1000 < source_duration_ms as u128
                    {
                        // Limit the maximum number of frames to catch up to avoid excessive CPU usage
                        let max_catch_up = 5;
                        let frames_to_draw = catch_up_frames.min(max_catch_up);

                        // println!("frames_to_draw {:?}", frames_to_draw);

                        for _ in 0..frames_to_draw {
                            self.video_items[object_idx]
                                .draw_video_frame(&gpu_resources.device, &gpu_resources.queue)
                                .expect("Couldn't draw catch-up video frame");

                            self.video_items[object_idx].num_frames_drawn += 1;
                        }

                        animate_properties = true;

                        // println!(
                        //     "Caught up {} frames out of {} needed",
                        //     frames_to_draw, catch_up_frames
                        // );
                    }
                }
            } else {
                animate_properties = true;
            }

            // let mut animate_properties = false;

            // Modified video drawing code
            // if animation.object_type == ObjectType::VideoItem {
            //     let frame_rate = self.video_items[object_idx].source_frame_rate;
            //     let source_duration_ms = self.video_items[object_idx].source_duration_ms;

            //     // Initialize frame timer if not exists
            //     if self.video_items[object_idx].frame_timer.is_none() {
            //         self.video_items[object_idx].frame_timer = Some(FrameTimer::new());
            //     }

            //     // Get number of frames to draw this step
            //     let frames_to_draw = self.video_items[object_idx]
            //         .frame_timer
            //         .as_mut()
            //         .expect("Couldn't get frame timer")
            //         .update_and_get_frames_to_draw(current_time, frame_rate as f32);

            //     // Draw the required number of frames
            //     if frames_to_draw > 0
            //         && current_time.as_millis() + 1000 < source_duration_ms as u128
            //     {
            //         println!("frames_to_draw {:?}", frames_to_draw);
            //         // Draw each frame
            //         for _ in 0..frames_to_draw {
            //             self.video_items[object_idx]
            //                 .draw_video_frame(&gpu_resources.device, &gpu_resources.queue)
            //                 .expect("Couldn't draw video frame");
            //         }

            //         animate_properties = true;
            //     }
            // }

            if !animate_properties {
                return;
            }

            // Go through each property
            for property in &animation.properties {
                if property.keyframes.len() < 2 {
                    continue;
                }

                if start_time > current_time {
                    continue;
                }

                // Find the surrounding keyframes
                let (start_frame, end_frame) = self.get_surrounding_keyframes(
                    &mut property.keyframes.clone(), // do not love clone in loop
                    current_time - start_time,
                );
                let Some((start_frame, end_frame)) = start_frame.zip(end_frame) else {
                    continue;
                };

                // Calculate interpolation progress
                let duration = (end_frame.time - start_frame.time).as_secs_f32(); // duration between keyframes
                let elapsed = (current_time - start_time - start_frame.time).as_secs_f32(); // elapsed since start keyframe
                let mut progress = elapsed / duration;

                // Apply easing (EaseInOut)
                progress = if progress < 0.5 {
                    2.0 * progress * progress
                } else {
                    1.0 - (-2.0 * progress + 2.0).powi(2) / 2.0
                };

                // do not update a property when start and end are the same
                // TODO: make this a setting for zooms so the center_point can continue its interpolation?
                // if start_frame.value == end_frame.value {
                //     continue;
                // }

                // Apply the interpolated value to the object's property
                match (&start_frame.value, &end_frame.value) {
                    (KeyframeValue::Position(start), KeyframeValue::Position(end)) => {
                        let x = self.lerp(start[0], end[0], progress);
                        let y = self.lerp(start[1], end[1], progress);

                        let position = Point {
                            x: 600.0 + x + path_group_position[0] as f32,
                            y: 50.0 + y + path_group_position[1] as f32,
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
                            ObjectType::VideoItem => {
                                self.video_items[object_idx]
                                    .transform
                                    .update_position([position.x, position.y], &camera.window_size);
                            }
                        }
                    }
                    (KeyframeValue::Rotation(start), KeyframeValue::Rotation(end)) => {
                        // rotation is stored as degrees
                        let new_rotation = self.lerp(*start, *end, progress);

                        let new_rotation_rad = new_rotation.to_radians();

                        match animation.object_type {
                            ObjectType::Polygon => {
                                self.polygons[object_idx]
                                    .transform
                                    .update_rotation(new_rotation_rad);
                            }
                            ObjectType::TextItem => {
                                self.text_items[object_idx]
                                    .transform
                                    .update_rotation(new_rotation_rad);
                            }
                            ObjectType::ImageItem => {
                                self.image_items[object_idx]
                                    .transform
                                    .update_rotation(new_rotation_rad);
                            }
                            ObjectType::VideoItem => {
                                self.video_items[object_idx]
                                    .transform
                                    .update_rotation(new_rotation_rad);
                            }
                        }
                    }
                    (KeyframeValue::Scale(start), KeyframeValue::Scale(end)) => {
                        // scale is stored out 100 (100 being standard size, ie. 100%)
                        let new_scale = self.lerp(*start, *end, progress) as f32 / 100.0;

                        // TODO: verify scale on all objects as some treat it differently as-is

                        match animation.object_type {
                            ObjectType::Polygon => {
                                self.polygons[object_idx]
                                    .transform
                                    .update_scale([new_scale, new_scale]);
                            }
                            ObjectType::TextItem => {
                                self.text_items[object_idx]
                                    .transform
                                    .update_scale([new_scale, new_scale]);
                            }
                            ObjectType::ImageItem => {
                                let original_scale = self.image_items[object_idx].dimensions;
                                self.image_items[object_idx].transform.update_scale([
                                    original_scale.0 as f32 * new_scale,
                                    original_scale.1 as f32 * new_scale,
                                ]);
                            }
                            ObjectType::VideoItem => {
                                let original_scale = self.video_items[object_idx].dimensions;
                                self.video_items[object_idx].transform.update_scale([
                                    original_scale.0 as f32 * new_scale,
                                    original_scale.1 as f32 * new_scale,
                                ]);
                            }
                        }
                    }
                    (KeyframeValue::Opacity(start), KeyframeValue::Opacity(end)) => {
                        // opacity is out 100 (100%)
                        let opacity = self.lerp(*start, *end, progress) / 100.0;

                        let gpu_resources = self
                            .gpu_resources
                            .as_ref()
                            .expect("Couldn't get gpu resources");

                        match animation.object_type {
                            ObjectType::Polygon => {
                                self.polygons[object_idx]
                                    .update_opacity(&gpu_resources.queue, opacity);
                            }
                            ObjectType::TextItem => {
                                self.text_items[object_idx]
                                    .update_opacity(&gpu_resources.queue, opacity);
                            }
                            ObjectType::ImageItem => {
                                self.image_items[object_idx]
                                    .update_opacity(&gpu_resources.queue, opacity);
                            }
                            ObjectType::VideoItem => {
                                self.video_items[object_idx]
                                    .update_opacity(&gpu_resources.queue, opacity);
                            }
                        }
                    }
                    (KeyframeValue::Zoom(start), KeyframeValue::Zoom(end)) => {
                        let zoom = self.lerp(*start, *end, progress) / 100.0;

                        let gpu_resources = self
                            .gpu_resources
                            .as_ref()
                            .expect("Couldn't get gpu resources");

                        match animation.object_type {
                            ObjectType::VideoItem => {
                                let video_item = &mut self.video_items[object_idx];
                                let elapsed_ms = current_time.as_millis() as u128;

                                let autofollow_delay = 150;

                                if let (Some(mouse_positions), Some(source_data)) = (
                                    video_item.mouse_positions.as_ref(),
                                    video_item.source_data.as_ref(),
                                ) {
                                    // Check if we need to update the shift points
                                    let should_update_shift = match video_item.last_shift_time {
                                        Some(last_shift_time) => {
                                            elapsed_ms - last_shift_time > autofollow_delay
                                        }
                                        None => {
                                            video_item.last_shift_time = Some(elapsed_ms);

                                            if let Some((start_point, end_point)) = mouse_positions
                                                .iter()
                                                .filter(|p| p.timestamp >= elapsed_ms)
                                                .zip(mouse_positions.iter().filter(|p| {
                                                    p.timestamp >= elapsed_ms + autofollow_delay
                                                }))
                                                .next()
                                                .map(|(start, end)| {
                                                    ((*start).clone(), (*end).clone())
                                                })
                                            {
                                                video_item.last_start_point = Some(start_point);
                                                video_item.last_end_point = Some(end_point);
                                            }

                                            false
                                        }
                                    };

                                    let delay_offset = 500; // Potential time offset for a consistent lag
                                    let min_distance = 100.0; // Distance to incur a shift
                                    let base_alpha = 0.01; // Your current default value
                                    let max_alpha = 0.1; // Maximum blending speed
                                    let scaling_factor = 0.01; // Controls how quickly alpha increases with distance

                                    // Update shift points if needed
                                    if should_update_shift {
                                        if let Some((start_point, end_point)) = mouse_positions
                                            .iter()
                                            .filter(|p| {
                                                p.timestamp
                                                    >= (elapsed_ms - autofollow_delay)
                                                        + delay_offset
                                                    && p.timestamp
                                                        < video_item.source_duration_ms as u128
                                            })
                                            .zip(mouse_positions.iter().filter(|p| {
                                                p.timestamp >= elapsed_ms + delay_offset
                                                    && p.timestamp
                                                        < video_item.source_duration_ms as u128
                                            }))
                                            .next()
                                            .map(|(start, end)| ((*start).clone(), (*end).clone()))
                                        {
                                            if let Some(last_start_point) =
                                                video_item.last_start_point
                                            {
                                                if let Some(last_end_point) =
                                                    video_item.last_end_point
                                                {
                                                    let dx = start_point.x - last_start_point.x;
                                                    let dy = start_point.y - last_start_point.y;
                                                    let distance = (dx * dx + dy * dy).sqrt(); // Euclidean distance

                                                    let dx2 = end_point.x - last_end_point.x;
                                                    let dy2 = end_point.y - last_end_point.y;
                                                    let distance2 = (dx2 * dx2 + dy2 * dy2).sqrt(); // Euclidean distance

                                                    if distance >= min_distance
                                                        || distance2 >= min_distance
                                                    {
                                                        video_item.last_shift_time =
                                                            Some(elapsed_ms);

                                                        video_item.last_start_point =
                                                            Some(start_point);
                                                        video_item.last_end_point = Some(end_point);

                                                        // Use the larger of the two distances
                                                        let max_distance = distance.max(distance2);

                                                        // Exponential smoothing that plateaus
                                                        let dynamic_alpha = base_alpha
                                                            + (max_alpha - base_alpha)
                                                                * (1.0
                                                                    - (-scaling_factor
                                                                        * max_distance)
                                                                        .exp());

                                                        video_item.dynamic_alpha = dynamic_alpha;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Always interpolate between the current shift points
                                    if let (Some(start), Some(end)) =
                                        (&video_item.last_start_point, &video_item.last_end_point)
                                    {
                                        let clamped_elapsed_ms =
                                            elapsed_ms.clamp(start.timestamp, end.timestamp);

                                        let time_progress = (clamped_elapsed_ms - start.timestamp)
                                            as f32
                                            / (end.timestamp - start.timestamp) as f32;

                                        let interpolated_x =
                                            start.x + (end.x - start.x) * time_progress;
                                        let interpolated_y =
                                            start.y + (end.y - start.y) * time_progress;

                                        let dimensions = video_item.dimensions;
                                        let source_dimensions = video_item.source_dimensions;

                                        let new_center_point = Point {
                                            x: ((interpolated_x - source_data.x as f32)
                                                / source_dimensions.0 as f32)
                                                * dimensions.0 as f32,
                                            y: ((interpolated_y - source_data.y as f32)
                                                / source_dimensions.1 as f32)
                                                * dimensions.1 as f32,
                                        };

                                        // Smooth transition with existing center point
                                        let blended_center_point = if let Some(last_center_point) =
                                            video_item.last_center_point
                                        {
                                            // need to calculate a dynamic alpha based on distance between start and and end point
                                            // let alpha = 0.01; // this was a close value, but not quite right depending on distance
                                            let alpha = video_item.dynamic_alpha;

                                            Point {
                                                x: last_center_point.x * (1.0 - alpha)
                                                    + new_center_point.x * alpha,
                                                y: last_center_point.y * (1.0 - alpha)
                                                    + new_center_point.y * alpha,
                                            }
                                        } else {
                                            new_center_point
                                        };

                                        video_item.update_zoom(
                                            &gpu_resources.queue,
                                            zoom,
                                            blended_center_point,
                                        );
                                        video_item.last_center_point = Some(blended_center_point);

                                        video_item.update_popout(
                                            &gpu_resources.queue,
                                            blended_center_point,
                                            1.5,
                                            (200.0, 200.0),
                                        );
                                    }
                                }
                            }
                            _ => {
                                println!("Zoom not supported here");
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // pub fn get_surrounding_keyframes<'a>(
    //     &self,
    //     keyframes: &'a [UIKeyframe],
    //     current_time: Duration,
    // ) -> (Option<&'a UIKeyframe>, Option<&'a UIKeyframe>) {
    //     let mut prev_frame = None;
    //     let mut next_frame = None;

    //     for (i, frame) in keyframes.iter().enumerate() {
    //         if frame.time > current_time {
    //             next_frame = Some(frame);
    //             prev_frame = if i > 0 {
    //                 Some(&keyframes[i - 1])
    //             } else {
    //                 Some(&keyframes[keyframes.len() - 1])
    //             };
    //             break;
    //         }
    //     }

    //     // Handle wrap-around case
    //     if next_frame.is_none() {
    //         prev_frame = keyframes.last();
    //         next_frame = keyframes.first();
    //     }

    //     (prev_frame, next_frame)
    // }

    /// Returns a "virtual" keyframe for the end keyframe in case of a Range type
    pub fn get_surrounding_keyframes(
        &self,
        keyframes: &mut [UIKeyframe],
        current_time: Duration,
    ) -> (Option<UIKeyframe>, Option<UIKeyframe>) {
        let mut prev_frame = None;
        let mut next_frame = None;

        // TODO: need to pick prev_frame based on timing not index
        // so just sort the keyframes here
        keyframes.sort_by_key(|k| k.time);

        for (i, frame) in keyframes.iter().enumerate() {
            if frame.time > current_time {
                // Check if the previous frame is a range
                if i > 0 {
                    if let KeyType::Range(range_data) = &keyframes[i - 1].key_type {
                        // Case 1: Current time is within the range
                        if current_time >= keyframes[i - 1].time
                            && current_time < range_data.end_time
                        {
                            // Current time is within a range
                            prev_frame = Some(keyframes[i - 1].clone());
                            next_frame = Some(UIKeyframe {
                                id: "virtual".to_string(),
                                time: range_data.end_time,
                                value: keyframes[i - 1].value.clone(),
                                easing: EasingType::Linear, // Doesn't matter for static ranges
                                path_type: PathType::Linear, // Doesn't matter for static ranges
                                key_type: KeyType::Frame, // Virtual keyframe is treated as a frame
                            });
                            return (prev_frame, next_frame);
                        }

                        // Case 2: Current time is after the range but before the next keyframe
                        if current_time >= range_data.end_time && current_time < frame.time {
                            prev_frame = Some(UIKeyframe {
                                id: "virtual".to_string(),
                                time: range_data.end_time, // End of the range
                                value: keyframes[i - 1].value.clone(), // Same value as start
                                easing: EasingType::Linear, // Doesn't matter for static ranges
                                path_type: PathType::Linear, // Doesn't matter for static ranges
                                key_type: KeyType::Frame,  // Virtual keyframe is treated as a frame
                            });
                            next_frame = Some(frame.clone()); // Next actual keyframe
                            return (prev_frame, next_frame);
                        }
                    }
                }

                // Regular keyframe logic

                next_frame = Some(frame.clone());
                prev_frame = if i > 0 {
                    Some(keyframes[i - 1].clone())
                } else {
                    Some(keyframes[keyframes.len() - 1].clone())
                };
                break;
            }
        }

        // Handle wrap-around case
        // can result in a duration subtraction error
        // if next_frame.is_none() {
        //     prev_frame = keyframes.last().cloned();
        //     next_frame = keyframes.first().cloned();
        // }

        (prev_frame, next_frame)
    }

    pub fn lerp(&self, start: i32, end: i32, progress: f32) -> f32 {
        start as f32 + ((end - start) as f32 * progress)
    }

    /// Create motion path visualization for a polygon
    /// // TODO: make for curves. already creates segments for the purpose
    pub fn create_motion_path_visualization(
        &mut self,
        sequence: &Sequence,
        polygon_id: &str,
        color_index: u32,
    ) {
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

        // let new_id = Uuid::new_v4();
        let new_id = Uuid::from_str(&animation_data.id).expect("Couldn't convert string to uuid");
        let initial_position = animation_data.position;
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get GPU Resources");

        // Create MotionPath
        let motion_path = MotionPath::new(
            &gpu_resources.device,
            &gpu_resources.queue,
            self.model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            self.group_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            new_id,
            &camera.window_size,
            keyframes,
            camera,
            sequence,
            // &mut self.static_polygons,
            color_index,
            polygon_id,
            initial_position,
        );

        self.motion_paths.push(motion_path);
    }

    /// Update the motion path visualization when keyframes change
    pub fn update_motion_paths(&mut self, sequence: &Sequence) {
        // Remove existing motion path segments
        // self.static_polygons.retain(|p| {
        //     p.name != "motion_path_segment"
        //         && p.name != "motion_path_handle"
        //         && p.name != "motion_path_arrow"
        // });

        // Remove existing motion paths
        self.motion_paths.clear();

        // Recreate motion paths for all polygons
        let mut color_index = 1;
        for polygon_config in &sequence.active_polygons {
            self.create_motion_path_visualization(sequence, &polygon_config.id, color_index);
            color_index = color_index + 1;
        }
        // Recreate motion paths for all texts
        for text_config in &sequence.active_text_items {
            self.create_motion_path_visualization(sequence, &text_config.id, color_index);
            color_index = color_index + 1;
        }
        // Recreate motion paths for all images
        for image_config in &sequence.active_image_items {
            self.create_motion_path_visualization(sequence, &image_config.id, color_index);
            color_index = color_index + 1;
        }
        // Recreate motion paths for all videos
        for video_config in &sequence.active_video_items {
            self.create_motion_path_visualization(sequence, &video_config.id, color_index);
            color_index = color_index + 1;
        }
    }

    pub fn update_camera_binding(&mut self) {
        if (self.camera_binding.is_some()) {
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");

            self.camera_binding
                .as_mut()
                .expect("Couldn't get camera binding")
                .update(
                    &gpu_resources.queue,
                    &self.camera.as_ref().expect("Couldn't get camera"),
                );
        }
    }

    pub fn handle_wheel(&mut self, delta: f32, mouse_pos: Point, queue: &wgpu::Queue) {
        let camera = self.camera.as_mut().expect("Couldnt't get camera");

        // let interactive_bounds = BoundingBox {
        //     min: Point { x: 550.0, y: 0.0 }, // account for aside width
        //     max: Point {
        //         x: camera.window_size.width as f32,
        //         y: camera.window_size.height as f32,
        //     },
        // };

        // if (mouse_pos.x < self.interactive_bounds.min.x
        //     || mouse_pos.x > self.interactive_bounds.max.x
        //     || mouse_pos.y < self.interactive_bounds.min.y
        //     || mouse_pos.y > self.interactive_bounds.max.y)
        // {
        //     return;
        // }

        if (self.last_screen.x < self.interactive_bounds.min.x
            || self.last_screen.x > self.interactive_bounds.max.x
            || self.last_screen.y < self.interactive_bounds.min.y
            || self.last_screen.y > self.interactive_bounds.max.y)
        {
            return;
        }

        // let zoom_factor = if delta > 0.0 { 1.1 } else { 0.9 };
        let zoom_factor = delta / 10.0;
        camera.zoom(zoom_factor, mouse_pos);
        self.update_camera_binding();
    }

    pub fn add_polygon(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        polygon_config: PolygonConfig,
        polygon_name: String,
        new_id: Uuid,
        selected_sequence_id: String,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let mut polygon = Polygon::new(
            window_size,
            device,
            queue,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            &self
                .group_bind_group_layout
                .as_ref()
                .expect("Couldn't get group bind group layout"),
            camera,
            polygon_config.points,
            polygon_config.dimensions,
            polygon_config.position,
            0.0,
            polygon_config.border_radius,
            polygon_config.fill,
            Stroke {
                thickness: 2.0,
                fill: rgb_to_wgpu(0, 0, 0, 255.0),
            },
            0.0,
            polygon_config.layer,
            polygon_name,
            new_id,
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
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
        queue: &wgpu::Queue,
        text_config: TextRendererConfig,
        text_content: String,
        new_id: Uuid,
        selected_sequence_id: String,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");

        let default_font_family = self
            .font_manager
            .get_font_by_name(&text_config.font_family)
            .expect("Couldn't load default font family");

        let mut text_item = TextRenderer::new(
            device,
            queue,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            &self
                .group_bind_group_layout
                .as_ref()
                .expect("Couldn't get group bind group layout"),
            default_font_family, // load font data ahead of time
            window_size,
            text_content.clone(),
            text_config,
            new_id,
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
        );

        text_item.render_text(&device, &queue);

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
        selected_sequence_id: String,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let mut image_item = StImage::new(
            device,
            queue,
            path,
            image_config,
            window_size,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            &self
                .group_bind_group_layout
                .as_ref()
                .expect("Couldn't get group bind group layout"),
            0.0,
            new_id.to_string(),
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
        );

        self.image_items.push(image_item);
    }

    pub fn add_video_item(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        video_config: StVideoConfig,
        path: &Path,
        new_id: Uuid,
        selected_sequence_id: String,
        mouse_positions: Option<Vec<MousePosition>>,
        stored_source_data: Option<SourceData>,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let mut video_item = StVideo::new(
            device,
            queue,
            path,
            video_config,
            window_size,
            &self
                .model_bind_group_layout
                .as_ref()
                .expect("Couldn't get model bind group layout"),
            &self
                .group_bind_group_layout
                .as_ref()
                .expect("Couldn't get group bind group layout"),
            0.0,
            new_id.to_string(),
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
        )
        .expect("Couldn't create video item");

        // set mouse capture source data if it exists
        video_item.source_data = stored_source_data;

        // set mouse positions for later use
        video_item.mouse_positions = mouse_positions;

        // render 1 frame to provide preview image
        video_item
            .draw_video_frame(device, queue)
            .expect("Couldn't draw video frame");

        self.video_items.push(video_item);
    }

    pub fn replace_background(&mut self, sequence_id: Uuid, fill: [f32; 4]) {
        println!("replace background {:?} {:?}", sequence_id, fill);

        let camera = self.camera.expect("Couldn't get camera");
        let window_size = camera.window_size;
        let model_bind_group_layout = self
            .model_bind_group_layout
            .as_ref()
            .expect("Couldn't get bind group layout");
        let group_bind_group_layout = self
            .group_bind_group_layout
            .as_ref()
            .expect("Couldn't get bind group layout");

        // Remove existing background
        self.static_polygons
            .retain(|p| p.name != "canvas_background");

        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");

        let canvas_polygon = Polygon::new(
            &window_size,
            &gpu_resources.device,
            &gpu_resources.queue,
            &model_bind_group_layout,
            &group_bind_group_layout,
            &camera,
            vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            (800.0 as f32, 450.0 as f32),
            Point { x: 400.0, y: 225.0 },
            0.0,
            0.0,
            fill,
            Stroke {
                thickness: 0.0,
                fill: rgb_to_wgpu(0, 0, 0, 255.0),
            },
            0.0,
            -89, // camera far is -100
            "canvas_background".to_string(),
            sequence_id,
            Uuid::nil(),
        );

        self.static_polygons.push(canvas_polygon);
    }

    pub fn update_background(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // First iteration: find the index of the selected polygon
        let polygon_index = self
            .static_polygons
            .iter()
            .position(|p| p.id == selected_id && p.name == "canvas_background".to_string());

        if let Some(index) = polygon_index {
            println!("Found selected static_polygon with ID: {}", selected_id);

            let camera = self.camera.expect("Couldn't get camera");

            // Get the necessary data from editor
            let viewport_width = camera.window_size.width;
            let viewport_height = camera.window_size.height;
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");
            let device = &gpu_resources.device;
            let queue = &gpu_resources.queue;

            let window_size = WindowSize {
                width: viewport_width as u32,
                height: viewport_height as u32,
            };

            // Second iteration: update the selected polygon
            if let Some(selected_polygon) = self.static_polygons.get_mut(index) {
                match new_value {
                    InputValue::Text(s) => match key {
                        _ => println!("No match on input"),
                    },
                    InputValue::Number(n) => match key {
                        "red" => selected_polygon.update_data_from_fill(
                            &window_size,
                            &device,
                            &queue,
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
                            &queue,
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
                            &queue,
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
                        _ => println!("No match on input"),
                    },
                }
            }
        } else {
            println!(
                "No static_polygon found with the selected ID: {}",
                selected_id
            );
        }
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
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");
            let device = &gpu_resources.device;
            let queue = &gpu_resources.queue;

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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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
                            &queue,
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

    pub fn update_text(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // First iteration: find the index of the selected polygon
        let text_index = self.text_items.iter().position(|p| p.id == selected_id);

        if let Some(index) = text_index {
            println!("Found selected text with ID: {}", selected_id);

            let camera = self.camera.expect("Couldn't get camera");

            // Get the necessary data from editor
            let viewport_width = camera.window_size.width;
            let viewport_height = camera.window_size.height;
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");
            let device = &gpu_resources.device;
            let queue = &gpu_resources.queue;

            let window_size = WindowSize {
                width: viewport_width as u32,
                height: viewport_height as u32,
            };

            // Second iteration: update the selected polygon
            if let Some(selected_text) = self.text_items.get_mut(index) {
                match new_value {
                    InputValue::Text(s) => match key {
                        _ => println!("No match on input"),
                    },
                    InputValue::Number(n) => match key {
                        "width" => selected_text.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (n, selected_text.dimensions.1),
                            &camera,
                        ),
                        "height" => selected_text.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (selected_text.dimensions.0, n),
                            &camera,
                        ),
                        _ => println!("No match on input"),
                    },
                }
            }
        } else {
            println!("No text found with the selected ID: {}", selected_id);
        }
    }

    pub fn update_image(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // First iteration: find the index of the selected polygon
        let image_index = self
            .image_items
            .iter()
            .position(|p| p.id == selected_id.to_string());

        if let Some(index) = image_index {
            println!("Found selected image with ID: {}", selected_id);

            let camera = self.camera.expect("Couldn't get camera");

            // Get the necessary data from editor
            let viewport_width = camera.window_size.width;
            let viewport_height = camera.window_size.height;
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");
            let device = &gpu_resources.device;
            let queue = &gpu_resources.queue;

            let window_size = WindowSize {
                width: viewport_width as u32,
                height: viewport_height as u32,
            };

            // Second iteration: update the selected polygon
            if let Some(selected_image) = self.image_items.get_mut(index) {
                match new_value {
                    InputValue::Text(s) => match key {
                        _ => println!("No match on input"),
                    },
                    InputValue::Number(n) => match key {
                        "width" => selected_image.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (n as f32, selected_image.dimensions.1 as f32),
                            &camera,
                        ),
                        "height" => selected_image.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (selected_image.dimensions.0 as f32, n as f32),
                            &camera,
                        ),
                        _ => println!("No match on input"),
                    },
                }
            }
        } else {
            println!("No image found with the selected ID: {}", selected_id);
        }
    }

    pub fn update_video(&mut self, selected_id: Uuid, key: &str, new_value: InputValue) {
        // First iteration: find the index of the selected polygon
        let video_index = self
            .video_items
            .iter()
            .position(|p| p.id == selected_id.to_string());

        if let Some(index) = video_index {
            println!("Found selected video with ID: {}", selected_id);

            let camera = self.camera.expect("Couldn't get camera");

            // Get the necessary data from editor
            let viewport_width = camera.window_size.width;
            let viewport_height = camera.window_size.height;
            let gpu_resources = self
                .gpu_resources
                .as_ref()
                .expect("Couldn't get gpu resources");
            let device = &gpu_resources.device;
            let queue = &gpu_resources.queue;

            let window_size = WindowSize {
                width: viewport_width as u32,
                height: viewport_height as u32,
            };

            // Second iteration: update the selected polygon
            if let Some(selected_video) = self.video_items.get_mut(index) {
                match new_value {
                    InputValue::Text(s) => match key {
                        _ => println!("No match on input"),
                    },
                    InputValue::Number(n) => match key {
                        "width" => selected_video.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (n as f32, selected_video.dimensions.1 as f32),
                            &camera,
                        ),
                        "height" => selected_video.update_data_from_dimensions(
                            &window_size,
                            &device,
                            &queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("Couldn't get model bind group layout"),
                            (selected_video.dimensions.0 as f32, n as f32),
                            &camera,
                        ),
                        _ => println!("No match on input"),
                    },
                }
            }
        } else {
            println!("No image found with the selected ID: {}", selected_id);
        }
    }

    pub fn get_object_width(&self, selected_id: Uuid, object_type: ObjectType) -> f32 {
        match object_type {
            ObjectType::Polygon => {
                let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.polygons.get(index) {
                        return selected_polygon.dimensions.0;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::TextItem => {
                let polygon_index = self.text_items.iter().position(|p| p.id == selected_id);

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.text_items.get(index) {
                        return selected_polygon.dimensions.0;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::ImageItem => {
                let polygon_index = self
                    .image_items
                    .iter()
                    .position(|p| p.id == selected_id.to_string());

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.image_items.get(index) {
                        return selected_polygon.dimensions.0 as f32;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::VideoItem => {
                let polygon_index = self
                    .video_items
                    .iter()
                    .position(|p| p.id == selected_id.to_string());

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.video_items.get(index) {
                        return selected_polygon.dimensions.0 as f32;
                    } else {
                        return 0.0;
                    }
                }
            }
        }

        0.0
    }

    pub fn get_object_height(&self, selected_id: Uuid, object_type: ObjectType) -> f32 {
        match object_type {
            ObjectType::Polygon => {
                let polygon_index = self.polygons.iter().position(|p| p.id == selected_id);

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.polygons.get(index) {
                        return selected_polygon.dimensions.1;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::TextItem => {
                let polygon_index = self.text_items.iter().position(|p| p.id == selected_id);

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.text_items.get(index) {
                        return selected_polygon.dimensions.1;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::ImageItem => {
                let polygon_index = self
                    .image_items
                    .iter()
                    .position(|p| p.id == selected_id.to_string());

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.image_items.get(index) {
                        return selected_polygon.dimensions.1 as f32;
                    } else {
                        return 0.0;
                    }
                }
            }
            ObjectType::VideoItem => {
                let polygon_index = self
                    .video_items
                    .iter()
                    .position(|p| p.id == selected_id.to_string());

                if let Some(index) = polygon_index {
                    if let Some(selected_polygon) = self.video_items.get(index) {
                        return selected_polygon.dimensions.1 as f32;
                    } else {
                        return 0.0;
                    }
                }
            }
        }

        0.0
    }

    pub fn get_background_red(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self
            .static_polygons
            .iter()
            .position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.static_polygons.get(index) {
                return selected_polygon.fill[0];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_background_green(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self
            .static_polygons
            .iter()
            .position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.static_polygons.get(index) {
                return selected_polygon.fill[1];
            } else {
                return 0.0;
            }
        }

        0.0
    }

    pub fn get_background_blue(&self, selected_id: Uuid) -> f32 {
        let polygon_index = self
            .static_polygons
            .iter()
            .position(|p| p.id == selected_id);

        if let Some(index) = polygon_index {
            if let Some(selected_polygon) = self.static_polygons.get(index) {
                return selected_polygon.fill[2];
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

    pub fn update_text_font_family(&mut self, font_id: String, selected_text_id: Uuid) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");

        let new_font_family = self
            .font_manager
            .get_font_by_name(&font_id)
            .expect("Couldn't load default font family");

        let text_item = self
            .text_items
            .iter_mut()
            .find(|t| t.id == selected_text_id)
            .expect("Couldn't find text item");

        text_item.font_family = font_id.clone();
        text_item.update_font_family(new_font_family);
        text_item.render_text(&gpu_resources.device, &gpu_resources.queue);
    }

    pub fn update_text_color(&mut self, selected_text_id: Uuid, color: [i32; 4]) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");

        let text_item = self
            .text_items
            .iter_mut()
            .find(|t| t.id == selected_text_id)
            .expect("Couldn't find text item");

        text_item.color = color;
        text_item.render_text(&gpu_resources.device, &gpu_resources.queue);
    }

    pub fn update_text_size(&mut self, selected_text_id: Uuid, size: i32) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");

        let text_item = self
            .text_items
            .iter_mut()
            .find(|t| t.id == selected_text_id)
            .expect("Couldn't find text item");

        text_item.font_size = size;
        text_item.render_text(&gpu_resources.device, &gpu_resources.queue);
    }

    pub fn update_text_content(&mut self, selected_text_id: Uuid, content: String) {
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");

        let text_item = self
            .text_items
            .iter_mut()
            .find(|t| t.id == selected_text_id)
            .expect("Couldn't find text item");

        text_item.text = content;
        text_item.render_text(&gpu_resources.device, &gpu_resources.queue);
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
    ) -> Option<ObjectEditConfig> {
        let camera = self.camera.as_ref().expect("Couldn't get camera");

        if (self.last_screen.x < self.interactive_bounds.min.x
            || self.last_screen.x > self.interactive_bounds.max.x
            || self.last_screen.y < self.interactive_bounds.min.y
            || self.last_screen.y > self.interactive_bounds.max.y)
        {
            return None;
        }

        // First, check if panning
        if self.control_mode == ControlMode::Pan {
            self.is_panning = true;
            self.drag_start = Some(self.last_top_left);

            return None;
        }

        // Next, check if we're clicking on a motion path handle to drag
        // for (poly_index, polygon) in self.static_polygons.iter_mut().enumerate() {
        //     if polygon.name != "motion_path_handle".to_string() {
        //         continue;
        //     }

        //     if polygon.contains_point(&self.last_top_left, &camera) {
        //         self.dragging_path_handle = Some(polygon.id);
        //         self.dragging_path_object = polygon.source_polygon_id;
        //         self.dragging_path_keyframe = polygon.source_keyframe_id;
        //         self.drag_start = Some(self.last_top_left);

        //         return None; // nothing to add to undo stack
        //     }
        // }

        for (path_index, path) in self.motion_paths.iter_mut().enumerate() {
            for (poly_index, polygon) in path.static_polygons.iter_mut().enumerate() {
                // check if we're clicking on a motion path handle to drag
                if polygon.name == "motion_path_handle".to_string() {
                    if polygon.contains_point(&self.last_top_left, &camera) {
                        self.dragging_path_handle = Some(polygon.id);
                        self.dragging_path_assoc_path = polygon.source_path_id;
                        self.dragging_path_object = polygon.source_polygon_id;
                        self.dragging_path_keyframe = polygon.source_keyframe_id;
                        self.drag_start = Some(self.last_top_left);

                        return None; // nothing to add to undo stack
                    }
                }
                if polygon.name == "motion_path_segment".to_string() {
                    if polygon.contains_point(&self.last_top_left, &camera) {
                        self.dragging_path = Some(path.id);
                        self.dragging_path_object = polygon.source_polygon_id;
                        self.drag_start = Some(self.last_top_left);

                        return None; // nothing to add to undo stack
                    }
                }
            }
        }

        // Finally, check for object interation
        let mut intersecting_objects: Vec<(i32, InteractionTarget)> = Vec::new();

        // Collect intersecting polygons
        for (poly_index, polygon) in self.polygons.iter_mut().enumerate() {
            if polygon.hidden {
                continue;
            }

            if polygon.contains_point(&self.last_top_left, &camera) {
                intersecting_objects.push((polygon.layer, InteractionTarget::Polygon(poly_index)));
            }
        }

        // Collect intersecting text items
        for (text_index, text_item) in self.text_items.iter_mut().enumerate() {
            if text_item.hidden {
                continue;
            }

            if text_item.contains_point(&self.last_top_left, &camera) {
                intersecting_objects.push((text_item.layer, InteractionTarget::Text(text_index)));
            }
        }

        // Collect intersecting image items
        for (image_index, image_item) in self.image_items.iter_mut().enumerate() {
            if image_item.hidden {
                continue;
            }

            if image_item.contains_point(&self.last_top_left, &camera) {
                intersecting_objects
                    .push((image_item.layer, InteractionTarget::Image(image_index)));
            }
        }

        // Collect intersecting image items
        for (video_index, video_item) in self.video_items.iter_mut().enumerate() {
            if video_item.hidden {
                continue;
            }

            println!("Checking video point");

            if video_item.contains_point(&self.last_top_left, &camera) {
                println!("Video contains point");
                intersecting_objects
                    .push((video_item.layer, InteractionTarget::Video(video_index)));
            }
        }

        // Sort intersecting objects by layer in descending order (highest layer first)
        // intersecting_objects.sort_by(|a, b| b.0.cmp(&a.0));

        // sort by lowest layer first, for this system
        intersecting_objects.sort_by(|a, b| a.0.cmp(&b.0));

        // Return the topmost intersecting object, if any
        let target = intersecting_objects
            .into_iter()
            .next()
            .map(|(_, target)| target);

        if let Some(target) = target {
            match target {
                InteractionTarget::Polygon(index) => {
                    let polygon = &mut self.polygons[index];

                    self.dragging_polygon = Some(polygon.id);
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
                                layer: polygon.layer,
                            },
                        );
                        self.selected_polygon_id = polygon.id;
                        polygon.old_points = Some(polygon.points.clone());
                    }

                    return None; // nothing to add to undo stack
                }
                InteractionTarget::Text(index) => {
                    let text_item = &mut self.text_items[index];

                    self.dragging_text = Some(text_item.id);
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
                                font_family: text_item.font_family.clone(),
                                // points: polygon.points.clone(),
                                dimensions: text_item.dimensions,
                                position: Point {
                                    x: text_item.transform.position.x,
                                    y: text_item.transform.position.y,
                                },
                                layer: text_item.layer,
                                color: text_item.color,
                                font_size: text_item.font_size, // border_radius: polygon.border_radius,
                                                                // fill: polygon.fill,
                                                                // stroke: polygon.stroke,
                            },
                        );
                        self.selected_polygon_id = text_item.id; // TODO: separate property for each object type?
                                                                 // polygon.old_points = Some(polygon.points.clone());
                    }

                    return None; // nothing to add to undo stack
                }
                InteractionTarget::Image(index) => {
                    let image_item = &mut self.image_items[index];

                    self.dragging_image =
                        Some(Uuid::from_str(&image_item.id).expect("Couldn't convert to uuid"));
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
                                layer: image_item.layer, // border_radius: polygon.border_radius,
                                                         // fill: polygon.fill,
                                                         // stroke: polygon.stroke,
                            },
                        );
                        self.selected_polygon_id = uuid; // TODO: separate property for each object type?
                                                         // polygon.old_points = Some(polygon.points.clone());
                    }

                    return None; // nothing to add to undo stack
                }
                InteractionTarget::Video(index) => {
                    let video_item = &mut self.video_items[index];

                    self.dragging_video =
                        Some(Uuid::from_str(&video_item.id).expect("Couldn't convert to uuid"));
                    self.drag_start = Some(self.last_top_left);

                    println!("Video interaction");

                    // TODO: make DRY with below
                    if (self.handle_video_click.is_some()) {
                        println!("Video click");

                        let handler_creator = self
                            .handle_video_click
                            .as_ref()
                            .expect("Couldn't get handler");
                        let mut handle_click = handler_creator().expect("Couldn't get handler");
                        let uuid = Uuid::from_str(&video_item.id.clone())
                            .expect("Couldn't convert string to uuid");
                        handle_click(
                            uuid,
                            StVideoConfig {
                                id: video_item.id.clone(),
                                name: video_item.name.clone(),
                                path: video_item.path.clone(),
                                // points: polygon.points.clone(),
                                dimensions: video_item.dimensions,
                                position: Point {
                                    x: video_item.transform.position.x,
                                    y: video_item.transform.position.y,
                                },
                                layer: video_item.layer,
                                mouse_path: video_item.mouse_path.clone(), // border_radius: polygon.border_radius,
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
        }

        None
    }

    pub fn handle_mouse_move(
        &mut self,
        window_size: &WindowSize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x: f32,
        y: f32,
    ) {
        let camera = self.camera.as_mut().expect("Couldn't get camera");
        let mouse_pos = Point { x, y };
        let ray = visualize_ray_intersection(window_size, x, y, &camera);
        let top_left = ray.top_left;
        // let top_left = camera.screen_to_world(x, y);
        // let top_left = mouse_pos;

        self.global_top_left = top_left;
        self.last_screen = Point { x, y };

        if (self.last_screen.x < self.interactive_bounds.min.x
            || self.last_screen.x > self.interactive_bounds.max.x
            || self.last_screen.y < self.interactive_bounds.min.y
            || self.last_screen.y > self.interactive_bounds.max.y)
        {
            // reset when out of bounds
            self.is_panning = false;
            return;
        }

        self.last_top_left = top_left;
        // self.ds_ndc_pos = ds_ndc_pos;
        // self.ndc = ds_ndc.ndc;

        // self.last_world = camera.screen_to_world(mouse_pos);

        // self.update_cursor();

        if let Some(dot) = &mut self.cursor_dot {
            // let ndc_position = point_to_ndc(self.last_top_left, &window_size);
            // println!("move dot {:?}", self.last_top_left);
            dot.transform
                .update_position([self.last_top_left.x, self.last_top_left.y], window_size);
        }

        // handle panning
        if self.control_mode == ControlMode::Pan && self.is_panning {
            let dx = (self.previous_top_left.x - self.last_top_left.x);
            let dy = (self.last_top_left.y - self.previous_top_left.y);
            let new_x = camera.position.x + dx;
            let new_y = camera.position.y + dy;

            camera.position = Vector2::new(new_x, new_y);
            // self.update_camera_binding(); // call in render loop, much more efficient
            // self.interactive_bounds = BoundingBox {
            //     max: Point {
            //         x: self.interactive_bounds.max.x + dx,
            //         y: self.interactive_bounds.max.y + dy,
            //     },
            //     min: Point {
            //         x: self.interactive_bounds.min.x + dx,
            //         y: self.interactive_bounds.min.y + dy,
            //     },
            // }
        }

        // handle dragging paths
        if let Some(path_id) = self.dragging_path {
            if let Some(start) = self.drag_start {
                self.move_path(self.last_top_left, start, path_id, window_size, device);
            }
        }

        // handle motion path handles
        if let Some(poly_id) = self.dragging_path_handle {
            if let Some(path_id) = self.dragging_path_assoc_path {
                if let Some(start) = self.drag_start {
                    // self.move_static_polygon(self.last_top_left, start, poly_id, window_size, device);
                    self.move_path_static_polygon(
                        self.last_top_left,
                        start,
                        poly_id,
                        path_id,
                        window_size,
                        device,
                    );
                }
            }
        }

        // handle dragging to move objects (polygons, images, text, etc)
        if let Some(poly_id) = self.dragging_polygon {
            if let Some(start) = self.drag_start {
                self.move_polygon(self.last_top_left, start, poly_id, window_size, device);
            }
        }

        if let Some(text_id) = self.dragging_text {
            if let Some(start) = self.drag_start {
                self.move_text(self.last_top_left, start, text_id, window_size, device);
            }
        }

        if let Some(image_id) = self.dragging_image {
            if let Some(start) = self.drag_start {
                self.move_image(self.last_top_left, start, image_id, window_size, device);
            }
        }

        if let Some(video_id) = self.dragging_video {
            if let Some(start) = self.drag_start {
                self.move_video(self.last_top_left, start, video_id, window_size, device);
            }
        }

        self.previous_top_left = self.last_top_left;
    }

    pub fn handle_mouse_up(&mut self) -> Option<ObjectEditConfig> {
        let mut action_edit = None;

        let camera = self.camera.expect("Couldn't get camera");

        // TODO: does another bounds cause this to get stuck?
        if (self.last_screen.x < self.interactive_bounds.min.x
            || self.last_screen.x > self.interactive_bounds.max.x
            || self.last_screen.y < self.interactive_bounds.min.y
            || self.last_screen.y > self.interactive_bounds.max.y)
        {
            return None;
        }

        // handle object on mouse up
        let mut object_id = Uuid::nil();
        let mut active_point = None;
        if let Some(poly_id) = self.dragging_polygon {
            object_id = poly_id;
            let active_polygon = self
                .polygons
                .iter()
                .find(|p| p.id == poly_id)
                .expect("Couldn't find polygon");
            active_point = Some(Point {
                x: active_polygon.transform.position.x,
                y: active_polygon.transform.position.y,
            });
        } else if let Some(image_id) = self.dragging_image {
            object_id = image_id;
            let active_image = self
                .image_items
                .iter()
                .find(|i| i.id == image_id.to_string())
                .expect("Couldn't find image");
            active_point = Some(Point {
                x: active_image.transform.position.x,
                y: active_image.transform.position.y,
            });
        } else if let Some(text_id) = self.dragging_text {
            object_id = text_id;
            let active_text = self
                .text_items
                .iter()
                .find(|t| t.id == text_id)
                .expect("Couldn't find text");
            active_point = Some(Point {
                x: active_text.transform.position.x,
                y: active_text.transform.position.y,
            });
        } else if let Some(video_id) = self.dragging_video {
            object_id = video_id;
            let active_video = self
                .video_items
                .iter()
                .find(|t| t.id == video_id.to_string())
                .expect("Couldn't find video");
            active_point = Some(Point {
                x: active_video.transform.position.x,
                y: active_video.transform.position.y,
            });
        }

        if object_id != Uuid::nil() && active_point.is_some() {
            if let Some(on_mouse_up_creator) = &self.on_mouse_up {
                let mut on_up = on_mouse_up_creator().expect("Couldn't get on handler");

                let active_point = active_point.expect("Couldn't get active point");
                let (selected_sequence_data, selected_keyframes) = on_up(
                    object_id,
                    Point {
                        x: active_point.x - 600.0,
                        y: active_point.y - 50.0,
                    },
                );

                // need some way of seeing if keyframe selected
                // perhaps need some way of opening keyframes explicitly
                // perhaps a toggle between keyframes and layout
                if selected_keyframes.len() > 0 {
                    self.update_motion_paths(&selected_sequence_data);
                    println!("Motion Paths updated!");
                }
            }
        }

        // handle handle on mouse up
        let handle_id = if let Some(poly_id) = self.dragging_path_handle {
            poly_id
        } else {
            Uuid::nil()
        };

        let mut handle_point = None;
        if handle_id != Uuid::nil() {
            let active_handle = self
                .motion_paths
                .iter()
                .flat_map(|m| &m.static_polygons)
                .find(|p| p.id == handle_id)
                .expect("Couldn't find handle");
            handle_point = Some(Point {
                x: active_handle.transform.position.x,
                y: active_handle.transform.position.y,
            })
        }

        // the object (polygon, text image, etc) related to this motion path handle
        let handle_object_id = if let Some(poly_id) = self.dragging_path_object {
            poly_id
        } else {
            Uuid::nil()
        };

        // the keyframe associated with this motion path handle
        let handle_keyframe_id = if let Some(kf_id) = self.dragging_path_keyframe {
            kf_id
        } else {
            Uuid::nil()
        };

        if handle_keyframe_id != Uuid::nil() && handle_point.is_some() {
            // need to update saved state and motion paths, handle polygon position already updated
            if let Some(on_mouse_up_creator) = &self.on_handle_mouse_up {
                let mut on_up = on_mouse_up_creator().expect("Couldn't get on handler");

                let handle_point = handle_point.expect("Couldn't get handle point");
                let (selected_sequence_data, selected_keyframes) = on_up(
                    handle_keyframe_id,
                    handle_object_id,
                    Point {
                        x: handle_point.x - 600.0,
                        y: handle_point.y - 50.0,
                    },
                );

                // always updated when handle is moved
                self.update_motion_paths(&selected_sequence_data);
                println!("Motion Paths updated!");
            }
        }

        // handle path mouse up
        if let Some(path_id) = self.dragging_path {
            let active_path = self
                .motion_paths
                .iter()
                .find(|p| p.id == path_id)
                .expect("Couldn't find path");
            let path_point = Point {
                x: active_path.transform.position.x,
                y: active_path.transform.position.y,
            };

            if let Some(on_mouse_up_creator) = &self.on_path_mouse_up {
                let mut on_up = on_mouse_up_creator().expect("Couldn't get on handler");

                let (selected_sequence_data, selected_keyframes) = on_up(
                    path_id,
                    // Point {
                    //     x: path_point.x - 600.0,
                    //     y: path_point.y - 50.0,
                    // },
                    // no offset needed because all relative?
                    Point {
                        x: path_point.x,
                        y: path_point.y,
                    },
                );

                // always updated when handle is moved
                // not necessary to update motion paths here? seems redundant
                // self.update_motion_paths(&selected_sequence_data);
                // println!("Motion Paths updated!");
            }
        }

        // reset variables
        self.dragging_polygon = None;
        self.dragging_text = None;
        self.dragging_image = None;
        self.dragging_video = None;
        self.drag_start = None;
        self.dragging_path = None;
        self.dragging_path_assoc_path = None;
        self.dragging_path_handle = None;
        self.dragging_path_object = None;
        self.dragging_path_keyframe = None;
        self.is_panning = false;

        // self.dragging_edge = None;
        // self.guide_lines.clear();
        // self.update_cursor();

        action_edit
    }

    pub fn reset_bounds(&mut self, window_size: &WindowSize) {
        let mut camera = self.camera.expect("Couldn't get camera");

        camera.position = Vector2::new(0.0, 0.0);
        camera.zoom = 1.0;
        self.update_camera_binding();
        self.interactive_bounds = BoundingBox {
            min: Point { x: 550.0, y: 0.0 }, // account for aside width, allow for some off-canvas positioning
            max: Point {
                x: window_size.width as f32,
                // y: window_size.height as f32 - 350.0, // 350.0 for timeline space
                y: 550.0, // allow for 50.0 padding below and above the canvas
            },
        };
    }

    pub fn move_polygon(
        &mut self,
        mouse_pos: Point,
        start: Point,
        poly_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let polygon = self
            .polygons
            .iter_mut()
            .find(|p| p.id == poly_id)
            .expect("Couldn't find polygon");

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

    pub fn move_static_polygon(
        &mut self,
        mouse_pos: Point,
        start: Point,
        poly_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let polygon = self
            .static_polygons
            .iter_mut()
            .find(|p| p.id == poly_id)
            .expect("Couldn't find polygon");

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

    pub fn move_path_static_polygon(
        &mut self,
        mouse_pos: Point,
        start: Point,
        poly_id: Uuid,
        path_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let path = self
            .motion_paths
            .iter_mut()
            .find(|p| p.id == path_id)
            .expect("Couldn't find polygon");
        let polygon = path
            .static_polygons
            .iter_mut()
            .find(|p| p.id == poly_id)
            .expect("Couldn't find polygon");

        let new_position = Point {
            x: polygon.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio?
            y: polygon.transform.position.y + dy,
        };

        println!("move path polygon {:?}", new_position);

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

    pub fn move_path(
        &mut self,
        mouse_pos: Point,
        start: Point,
        poly_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        let path = self
            .motion_paths
            .iter_mut()
            .find(|p| p.id == poly_id)
            .expect("Couldn't find path");

        let new_position = Point {
            x: path.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio? probably not needed now
            y: path.transform.position.y + dy,
        };

        println!("move_path {:?}", new_position);

        path.update_data_from_position(
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
        text_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        // let text_item = &mut self.text_items[text_index];
        let text_item = self
            .text_items
            .iter_mut()
            .find(|t| t.id == text_id)
            .expect("Couldn't find text item");
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
        image_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        // let image_item = &mut self.image_items[image_index];
        let image_item = self
            .image_items
            .iter_mut()
            .find(|i| i.id == image_id.to_string())
            .expect("Couldn't find image item");
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

    pub fn move_video(
        &mut self,
        mouse_pos: Point,
        start: Point,
        video_id: Uuid,
        window_size: &WindowSize,
        device: &wgpu::Device,
    ) {
        let camera = self.camera.as_ref().expect("Couldn't get camera");
        let aspect_ratio = camera.window_size.width as f32 / camera.window_size.height as f32;
        let dx = mouse_pos.x - start.x;
        let dy = mouse_pos.y - start.y;
        // let image_item = &mut self.image_items[image_index];
        let video_item = self
            .video_items
            .iter_mut()
            .find(|i| i.id == video_id.to_string())
            .expect("Couldn't find video item");
        let new_position = Point {
            x: video_item.transform.position.x + (dx * 0.9), // not sure relation with aspect_ratio?
            y: video_item.transform.position.y + dy,
        };

        println!("move_image {:?}", new_position);

        video_item
            .transform
            .update_position([new_position.x, new_position.y], window_size);

        self.drag_start = Some(mouse_pos);
        // self.update_guide_lines(poly_index, window_size);
    }

    fn is_close(&self, a: f32, b: f32, threshold: f32) -> bool {
        (a - b).abs() < threshold
    }

    pub fn hide_all_objects(&mut self) {
        // Remove objects
        self.polygons.iter_mut().for_each(|p| {
            p.hidden = true;
        });
        self.text_items.iter_mut().for_each(|t| {
            t.hidden = true;
        });
        self.image_items.iter_mut().for_each(|i| {
            i.hidden = true;
        });
        self.video_items.iter_mut().for_each(|v| {
            v.hidden = true;
        });

        // Remove existing motion path segments
        // self.static_polygons.retain(|p| {
        //     p.name != "motion_path_segment"
        //         && p.name != "motion_path_handle"
        //         && p.name != "motion_path_arrow"
        // });
        // Remove existing motion paths
        self.motion_paths.clear();
    }
}

// Helper function to create default properties with constant values
fn create_default_property(
    name: &str,
    path: &str,
    value: KeyframeValue,
    timestamps: &[i32],
) -> AnimationProperty {
    let keyframes = timestamps
        .iter()
        .map(|&time| UIKeyframe {
            id: Uuid::new_v4().to_string(),
            time: Duration::from_millis(time as u64),
            value: value.clone(),
            easing: EasingType::EaseInOut,
            path_type: PathType::Linear,
            key_type: KeyType::Frame,
        })
        .collect();

    AnimationProperty {
        name: name.to_string(),
        property_path: path.to_string(),
        children: Vec::new(),
        keyframes,
        depth: 0,
    }
}

// /// Get interpolated position at a specific time
// fn interpolate_position(start: &UIKeyframe, end: &UIKeyframe, time: Duration) -> [i32; 2] {
//     if let (KeyframeValue::Position(start_pos), KeyframeValue::Position(end_pos)) =
//         (&start.value, &end.value)
//     {
//         let progress = match start.easing {
//             EasingType::Linear => {
//                 let total_time = (end.time - start.time).as_secs_f32();
//                 let current_time = (time - start.time).as_secs_f32();
//                 current_time / total_time
//             }
//             // Add more sophisticated easing calculations here
//             _ => {
//                 let total_time = (end.time - start.time).as_secs_f32();
//                 let current_time = (time - start.time).as_secs_f32();
//                 current_time / total_time
//             }
//         };

//         [
//             (start_pos[0] as f32 + (end_pos[0] - start_pos[0]) as f32 * progress) as i32,
//             (start_pos[1] as f32 + (end_pos[1] - start_pos[1]) as f32 * progress) as i32,
//         ]
//     } else {
//         panic!("Expected position keyframes")
//     }
// }

// curves attempt
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ControlPoint {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct CurveData {
    pub control_point1: Option<ControlPoint>,
    pub control_point2: Option<ControlPoint>,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum PathType {
    Linear,
    Bezier(CurveData),
}

// impl Default for PathType {
//     fn default() -> Self {
//         PathType::Linear
//     }
// }

/// Creates curves in between keyframes, on the same path, rather than sharing a curve with another
/// but it's better this way, as using a keyframe as a middle point on a curve leads to various problems
pub fn interpolate_position(start: &UIKeyframe, end: &UIKeyframe, time: f32) -> [i32; 2] {
    if let (KeyframeValue::Position(start_pos), KeyframeValue::Position(end_pos)) =
        (&start.value, &end.value)
    {
        let progress = {
            let total_time = (end.time - start.time).as_secs_f32();
            let current_time = time - (start.time).as_secs_f32();
            let t = current_time / total_time;

            match start.easing {
                EasingType::Linear => t,
                EasingType::EaseIn => t * t,
                EasingType::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
                EasingType::EaseInOut => {
                    if t < 0.5 {
                        2.0 * t * t
                    } else {
                        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                    }
                }
            }
        };

        // Get curve data from the keyframe
        let path_type = start.path_type.clone();
        // let path_type = PathType::Bezier(CurveData {
        //     control_point1: None,
        //     control_point2: None,
        // });
        // let test_offset = 50.0;
        // let path_type = PathType::Bezier(CurveData {
        //     control_point1: Some(ControlPoint {
        //         x: (start_pos[0] as f32 + (end_pos[0] - start_pos[0]) as f32 * 0.2) + test_offset,
        //         y: (start_pos[1] as f32 + (end_pos[1] - start_pos[1]) as f32 * 0.2) + test_offset,
        //     }),
        //     control_point2: Some(ControlPoint {
        //         x: (start_pos[0] as f32 + (end_pos[0] - start_pos[0]) as f32 * 0.8) + test_offset,
        //         y: (start_pos[1] as f32 + (end_pos[1] - start_pos[1]) as f32 * 0.8) + test_offset,
        //     }),
        // });
        // let path_type = PathType::Bezier(CurveData {
        //     control_point1: Some(ControlPoint { x: 500.0, y: 300.0 }),
        //     control_point2: Some(ControlPoint { x: 700.0, y: 400.0 }),
        // });

        match path_type {
            PathType::Linear => [
                (start_pos[0] as f32 + (end_pos[0] - start_pos[0]) as f32 * progress) as i32,
                (start_pos[1] as f32 + (end_pos[1] - start_pos[1]) as f32 * progress) as i32,
            ],
            PathType::Bezier(curve_data) => {
                let p0 = (start_pos[0] as f32, start_pos[1] as f32);
                let p3 = (end_pos[0] as f32, end_pos[1] as f32);

                // Use control points if available, otherwise generate default ones
                let p1 = curve_data.control_point1.as_ref().map_or_else(
                    || (p0.0 + (p3.0 - p0.0) * 0.33, p0.1 + (p3.1 - p0.1) * 0.33),
                    |cp| (cp.x as f32, cp.y as f32),
                );

                let p2 = curve_data.control_point2.as_ref().map_or_else(
                    || (p0.0 + (p3.0 - p0.0) * 0.66, p0.1 + (p3.1 - p0.1) * 0.66),
                    |cp| (cp.x as f32, cp.y as f32),
                );

                // Cubic Bezier curve formula
                let t = progress;
                let t2 = t * t;
                let t3 = t2 * t;
                let mt = 1.0 - t;
                let mt2 = mt * mt;
                let mt3 = mt2 * mt;

                let x = p0.0 * mt3 + 3.0 * p1.0 * mt2 * t + 3.0 * p2.0 * mt * t2 + p3.0 * t3;
                let y = p0.1 * mt3 + 3.0 * p1.1 * mt2 * t + 3.0 * p2.1 * mt * t2 + p3.1 * t3;

                // println!(
                //     "Bezier {:?} and {:?} vs ({:?}, {:?}) at {:?} and {:?}",
                //     p0, p3, x, y, progress, time
                // );

                [x as i32, y as i32]
            }
        }
    } else {
        panic!("Expected position keyframes")
    }
}

use cgmath::InnerSpace;

#[derive(Debug)]
pub struct Ray {
    // pub origin: Point3<f32>,
    // pub direction: Vector3<f32>,
    // pub ndc: Point,
    pub top_left: Point,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Ray {
            // origin,
            // direction: direction.normalize(),
            // ndc: Point { x: 0.0, y: 0.0 },
            top_left: Point { x: 0.0, y: 0.0 },
        }
    }
}

use cgmath::SquareMatrix;
use cgmath::Transform;

use crate::animations::{
    AnimationData, AnimationProperty, BackgroundFill, EasingType, KeyType, KeyframeValue,
    ObjectType, RangeData, Sequence, UIKeyframe,
};
use crate::camera::{Camera, CameraBinding};
use crate::capture::{MousePosition, SourceData};
use crate::dot::RingDot;
use crate::fonts::FontManager;
use crate::motion_path::{MotionPath, MotionPathConfig};
use crate::polygon::{Polygon, PolygonConfig, Stroke};
use crate::st_image::{StImage, StImageConfig};
use crate::st_video::{FrameTimer, StVideo, StVideoConfig};
use crate::text_due::{TextRenderer, TextRendererConfig};
use crate::timelines::{SavedTimelineStateConfig, TrackType};
use crate::transform::{angle_between_points, degrees_between_points};

// old
// pub fn visualize_ray_intersection(
//     // device: &wgpu::Device,
//     window_size: &WindowSize,
//     screen_x: f32,
//     screen_y: f32,
//     camera: &Camera,
// ) -> Ray {
//     // only a small adjustment in aspect ratio when going full screen
//     // let aspect_ratio = window_size.width as f32 / window_size.height as f32; // ~1.5
//     // let aspect_ratio_rev = window_size.height as f32 / window_size.width as f32; // ~0.5

//     // // println!("Aspect Ratio: {:?} vs {:?}", aspect_ratio, aspect_ratio_rev);

//     // let norm_x = screen_x / camera.window_size.width as f32;
//     // let norm_y = screen_y / camera.window_size.height as f32;

//     // // put camera pos in view_pos instead?
//     // // let view_pos = Vector3::new(0.0, 0.0, 0.0);
//     // // let model_view = Matrix4::from_translation(view_pos);

//     // // defaults to 1.0
//     let scale_factor = camera.zoom;

//     // // the plane size, normalized
//     // let plane_size_normal = Vector3::new(
//     //     (1.0 * aspect_ratio * scale_factor) / 2.0,
//     //     (1.0 * 2.0 * scale_factor) / 2.0,
//     //     0.0,
//     // );

//     // // Transform norm point to view space
//     // let view_point_normal = Point3::new(
//     //     (norm_x * plane_size_normal.x),
//     //     (norm_y * plane_size_normal.y),
//     //     0.0,
//     // );
//     // // let world_point_normal = model_view
//     // //     .invert()
//     // //     .unwrap()
//     // //     .transform_point(view_point_normal);

//     // // NOTE: offset only applied if scale_factor (camera zoom) is adjusted from 1.0
//     // let offset_x = (scale_factor - 1.0) * (400.0 * aspect_ratio);
//     // let offset_y = (scale_factor - 1.0) * 400.0;

//     // // NOTE: camera position is 0,0 be default
//     // let top_left: Point = Point {
//     //     x: (view_point_normal.x * window_size.width as f32) + (camera.position.x * 0.5) + 70.0
//     //         - offset_x,
//     //     y: (view_point_normal.y * window_size.height as f32) - (camera.position.y * 0.5) - offset_y,
//     // };

//     let pan_offset_x = camera.position.x * 0.5;
//     let pan_offset_y = camera.position.y * 0.5;

//     // let zoom_offset_x = (scale_factor - 1.0) * (window_size.width as f32 / 2.0);
//     // let zoom_offset_y = (scale_factor - 1.0) * (window_size.height as f32 / 2.0);

//     let top_left: Point = Point {
//         x: screen_x + pan_offset_x,
//         y: screen_y - pan_offset_y,
//     };

//     Ray { top_left }
// }

// new
// pub fn visualize_ray_intersection(
//     window_size: &WindowSize,
//     screen_x: f32,
//     screen_y: f32,
//     camera: &Camera,
// ) -> Ray {
//     let aspect_ratio = window_size.width as f32 / window_size.height as f32; // ~1.5
//     let scale_factor = camera.zoom;
//     let pan_offset_x = camera.position.x * 0.5;
//     let pan_offset_y = camera.position.y * 0.5;

//     // let zoom_offset_x = (scale_factor - 1.0) * (400.0);
//     // let zoom_offset_y = (scale_factor - 1.0) * (400.0);

//     // Apply zoom to screen coordinates
//     let zoomed_screen_x = screen_x / scale_factor;
//     let zoomed_screen_y = screen_y / scale_factor;

//     let zoom_offset_x = (scale_factor - 1.0) * 500.0;
//     let zoom_offset_y = (scale_factor - 1.0) * 300.0;

//     let top_left: Point = Point {
//         x: zoomed_screen_x + zoom_offset_x + pan_offset_x,
//         y: zoomed_screen_y + zoom_offset_y - pan_offset_y,
//     };

//     Ray { top_left }
// }

pub fn visualize_ray_intersection(
    window_size: &WindowSize,
    screen_x: f32,
    screen_y: f32,
    camera: &Camera,
) -> Ray {
    let scale_factor = camera.zoom;
    let zoom_center_x = window_size.width as f32 / 2.0;
    let zoom_center_y = window_size.height as f32 / 2.0;

    // 1. Translate screen coordinates to zoom center
    let translated_screen_x = screen_x - zoom_center_x;
    let translated_screen_y = screen_y - zoom_center_y;

    // 2. Apply zoom
    let zoomed_screen_x = translated_screen_x / scale_factor;
    let zoomed_screen_y = translated_screen_y / scale_factor;

    // 3. Translate back to original screen space
    let scaled_screen_x = zoomed_screen_x + zoom_center_x;
    let scaled_screen_y = zoomed_screen_y + zoom_center_y;

    let pan_offset_x = camera.position.x * 0.5;
    let pan_offset_y = camera.position.y * 0.5;

    let top_left: Point = Point {
        x: scaled_screen_x + pan_offset_x,
        y: scaled_screen_y - pan_offset_y,
    };

    Ray { top_left }
}

// Define an enum to represent interaction targets
pub enum InteractionTarget {
    Polygon(usize),
    Text(usize),
    Image(usize),
    Video(usize),
}

pub fn get_color(color_index: u32) -> u32 {
    // Normalize the color_index to be within 0-29 range
    let normalized_index = color_index % 30;

    // Calculate which shade we're on (0-9)
    let shade_index = normalized_index / 3;

    // Calculate the shade intensity (0-255)
    // Using a range of 25-255 to avoid completely black colors
    155 + (shade_index * 10) // (255 - 25) / 10 ≈ 23 steps
}

// TODO: create an LayerColor struct for caching colors and reusing, rather than storing that color somewhere on the object?
pub fn get_full_color(index: u32) -> (u32, u32, u32) {
    // Normalize the index
    let normalized_index = index % 30;

    // Determine which color gets the intensity (0=red, 1=green, 2=blue)
    match normalized_index % 3 {
        0 => (get_color(index), 10, 10), // Red
        1 => (10, get_color(index), 10), // Green
        2 => (10, 10, get_color(index)), // Blue
        _ => unreachable!(),
    }
}

use munkres::{solve_assignment, Error, Position, WeightMatrix};

pub fn assign_motion_paths_to_objects(
    cost_matrix: Vec<Vec<f64>>,
) -> Result<Vec<(usize, usize)>, Error> {
    // Flatten the 2D cost matrix into a 1D vector
    let n = cost_matrix.len();
    let flat_matrix: Vec<f64> = cost_matrix.into_iter().flatten().collect();

    // Create a WeightMatrix from the flattened vector
    let mut weights = WeightMatrix::from_row_vec(n, flat_matrix);

    // Solve the assignment problem
    let result = solve_assignment(&mut weights)?;

    // Process the result into (object_index, path_index) pairs
    let assignments = result
        .into_iter()
        .map(|Position { row, column }| (row, column))
        .collect();

    Ok(assignments)
}
