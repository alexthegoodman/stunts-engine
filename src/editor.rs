use std::cell::RefCell;
use std::fmt::Display;
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

pub type PolygonClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, PolygonConfig) + Send>> + Send + Sync;

pub type TextItemClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, TextRendererConfig) + Send>> + Send + Sync;

pub type ImageItemClickHandler =
    dyn Fn() -> Option<Box<dyn FnMut(Uuid, StImageConfig) + Send>> + Send + Sync;

pub type OnMouseUp = dyn Fn() -> Option<Box<dyn FnMut(Uuid, Point) -> (Sequence, Vec<UIKeyframe>) + Send>>
    + Send
    + Sync;

pub type OnHandleMouseUp = dyn Fn() -> Option<Box<dyn FnMut(Uuid, Uuid, Point) -> (Sequence, Vec<UIKeyframe>) + Send>>
    + Send
    + Sync;

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
    pub dragging_path_handle: Option<Uuid>,
    pub dragging_path_object: Option<Uuid>,
    pub dragging_path_keyframe: Option<Uuid>,
    pub cursor_dot: Option<RingDot>,

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
    pub window_size_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pub window_size_bind_group: Option<wgpu::BindGroup>,
    pub window_size_buffer: Option<Arc<wgpu::Buffer>>,
    pub on_mouse_up: Option<Arc<OnMouseUp>>,
    pub on_handle_mouse_up: Option<Arc<OnHandleMouseUp>>,
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
}

use std::borrow::{Borrow, BorrowMut};

pub fn init_editor_with_model(viewport: Arc<Mutex<Viewport>>) -> Editor {
    let inference = load_common_motion_2d();

    let editor = Editor::new(viewport, Some(inference));

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
            previous_top_left: Point { x: 0.0, y: 0.0 },
            is_playing: false,
            current_sequence_data: None,
            last_frame_time: None,
            start_playing_time: None,
            model_bind_group_layout: None,
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
            dragging_path_handle: None,
            on_handle_mouse_up: None,
            dragging_path_object: None,
            dragging_path_keyframe: None,
            cursor_dot: None,
            control_mode: ControlMode::Select,
            is_panning: false,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
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

        for object_idx in 0..num_objects {
            let mut position_keyframes = Vec::new();

            // Get the item ID based on the object index
            let item_id = self.get_item_id(object_idx);
            let object_type = self.get_object_type(object_idx);

            // Process keyframes for this object
            for keyframe_idx in 0..keyframes_per_object {
                let base_idx = object_idx * (values_per_prediction * keyframes_per_object)
                    + keyframe_idx * values_per_prediction;

                // Skip if out of bounds
                if base_idx + 5 >= predictions.len() {
                    continue;
                }

                // let predicted_x = predictions[base_idx + 4].round() as i32;
                // let predicted_y = predictions[base_idx + 5].round() as i32;

                // testing percentage based training
                let predicted_x = ((predictions[base_idx + 4] * 0.01) * 800.0).round() as i32;
                let predicted_y = ((predictions[base_idx + 5] * 0.01) * 450.0).round() as i32;

                position_keyframes.push(UIKeyframe {
                    id: Uuid::new_v4().to_string(),
                    time: Duration::from_millis(timestamps[keyframe_idx] as u64),
                    value: KeyframeValue::Position([predicted_x, predicted_y]),
                    easing: EasingType::EaseInOut,
                    path_type: PathType::Linear,
                });
            }

            // Only create animation if we have valid keyframes and item ID
            if !position_keyframes.is_empty() && item_id.is_some() {
                let properties = vec![
                    // Position property with predicted values
                    AnimationProperty {
                        name: "Position".to_string(),
                        property_path: "position".to_string(),
                        children: Vec::new(),
                        keyframes: position_keyframes,
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

        let polygon_count = self.polygons.iter().filter(|p| !p.hidden).count();
        let text_count = self.text_items.iter().filter(|t| !t.hidden).count();

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

        match object_idx {
            idx if idx < polygon_count => Some(ObjectType::Polygon),
            idx if idx < polygon_count + text_count => Some(ObjectType::TextItem),
            idx if idx < polygon_count + text_count + image_count => Some(ObjectType::ImageItem),
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

        // Iterate through timeline sequences in order
        for ts in &sequence_timeline.timeline_sequences {
            // Skip audio tracks as we're only handling video
            if ts.track_type != TrackType::Video {
                continue;
            }

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
                && current_time_ms < (ts.start_time_ms + ts.duration_ms)
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
                        }
                    } else {
                        self.current_sequence_data = Some(sequence.clone());
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

        let (fill_r, fill_g, fill_b) = get_full_color(color_index);
        let path_fill = rgb_to_wgpu(fill_r as u8, fill_g as u8, fill_b as u8, 1.0);

        let polygon_id = Uuid::from_str(polygon_id).expect("Couldn't convert string to uuid");

        // Create path segments between consecutive keyframes
        let mut pairs_done = 0;
        for window in keyframes.windows(2) {
            let start_kf = &window[0];
            let end_kf = &window[1];

            let start_kf_id =
                Uuid::from_str(&start_kf.id).expect("Couldn't convert string to uuid");
            let end_kf_id = Uuid::from_str(&end_kf.id).expect("Couldn't convert string to uuid");

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
                    _ => 9, // More segments for smooth curves
                };

                let camera = self.camera.expect("Couldn't get camera");
                let gpu_resources = self.gpu_resources.as_ref().expect("No GPU resources");

                if pairs_done == 0 {
                    // handle for first keyframe in path
                    let mut handle = create_path_handle(
                        &camera.window_size,
                        &gpu_resources.device,
                        &gpu_resources.queue,
                        &self
                            .model_bind_group_layout
                            .as_ref()
                            .expect("No bind group layout"),
                        &self.camera.expect("No camera"),
                        start_point,
                        12.0, // width and height
                        sequence.id.clone(),
                        path_fill,
                    );

                    // // calculate angle so triangle handle points in correct direction
                    // // let angle = angle_between_points(start_point, end_point);
                    // // handle.transform.rotate(angle);
                    // let angle = degrees_between_points(start_point, end_point);
                    // handle.transform.rotate_degrees(angle);

                    handle.source_polygon_id = Some(polygon_id);
                    handle.source_keyframe_id = Some(start_kf_id);

                    self.static_polygons.push(handle);
                }

                // handles for remaining keyframes
                let mut handle = create_path_handle(
                    &camera.window_size,
                    &gpu_resources.device,
                    &gpu_resources.queue,
                    &self
                        .model_bind_group_layout
                        .as_ref()
                        .expect("No bind group layout"),
                    &self.camera.expect("No camera"),
                    end_point,
                    12.0, // width and height
                    sequence.id.clone(),
                    path_fill,
                );

                // // calculate angle so triangle handle points in correct direction
                // // let angle = angle_between_points(start_point, end_point);
                // // handle.transform.rotate(angle);
                // let angle = degrees_between_points(start_point, end_point);
                // handle.transform.rotate_degrees(angle);

                handle.source_polygon_id = Some(polygon_id);
                handle.source_keyframe_id = Some(end_kf_id);

                self.static_polygons.push(handle);

                let segment_duration =
                    (end_kf.time.as_secs_f32() - start_kf.time.as_secs_f32()) / num_segments as f32;

                let mut odd = false;
                for i in 0..num_segments {
                    // let t1 = start_kf.time + (end_kf.time - start_kf.time) * i / num_segments;
                    // let t2 = start_kf.time + (end_kf.time - start_kf.time) * (i + 1) / num_segments;

                    let t1 = start_kf.time.as_secs_f32() + segment_duration * i as f32;
                    let t2 = start_kf.time.as_secs_f32() + segment_duration * (i + 1) as f32;

                    // println!("pos1");
                    let pos1 = interpolate_position(start_kf, end_kf, t1);
                    // println!("pos2");
                    let pos2 = interpolate_position(start_kf, end_kf, t2);

                    let camera = self.camera.expect("Couldn't get camera");

                    let gpu_resources = self.gpu_resources.as_ref().expect("No GPU resources");

                    let path_start = Point {
                        x: pos1[0] as f32,
                        y: pos1[1] as f32,
                    };

                    let path_end = Point {
                        x: pos2[0] as f32,
                        y: pos2[1] as f32,
                    };

                    // Calculate rotation angle from start to end point
                    let dx = path_end.x - path_start.x;
                    let dy = path_end.y - path_start.y;
                    let rotation = dy.atan2(dx);

                    // Calculate length of the segment
                    let length = (dx * dx + dy * dy).sqrt();

                    // println!("length {:?}", length);

                    let mut segment = create_path_segment(
                        &camera.window_size,
                        &gpu_resources.device,
                        &gpu_resources.queue,
                        &self
                            .model_bind_group_layout
                            .as_ref()
                            .expect("No bind group layout"),
                        &self.camera.expect("No camera"),
                        path_start,
                        path_end,
                        2.0, // thickness of the path
                        sequence.id.clone(),
                        path_fill,
                        rotation,
                        length,
                    );

                    // segment.source_polygon_id = Some(polygon_id);
                    // segment.source_keyframe_id =
                    // Some(end_kf_id);

                    self.static_polygons.push(segment);

                    // arrow for indicating direction of motion
                    if odd {
                        let arrow_orientation_offset = -std::f32::consts::FRAC_PI_2; // for upward-facing arrow
                        let mut arrow = create_path_arrow(
                            &camera.window_size,
                            &gpu_resources.device,
                            &gpu_resources.queue,
                            &self
                                .model_bind_group_layout
                                .as_ref()
                                .expect("No bind group layout"),
                            &self.camera.expect("No camera"),
                            path_end,
                            15.0, // width and height
                            sequence.id.clone(),
                            path_fill,
                            rotation + arrow_orientation_offset,
                        );

                        self.static_polygons.push(arrow);
                    }

                    odd = !odd;
                }

                pairs_done = pairs_done + 1;
            }
        }
    }

    /// Update the motion path visualization when keyframes change
    pub fn update_motion_paths(&mut self, sequence: &Sequence) {
        // Remove existing motion path segments
        self.static_polygons.retain(|p| {
            p.name != "motion_path_segment"
                && p.name != "motion_path_handle"
                && p.name != "motion_path_arrow"
        });

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

        if (mouse_pos.x < self.interactive_bounds.min.x
            || mouse_pos.x > self.interactive_bounds.max.x
            || mouse_pos.y < self.interactive_bounds.min.y
            || mouse_pos.y > self.interactive_bounds.max.y)
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
            default_font_family, // load font data ahead of time
            window_size,
            text_content.clone(),
            text_config,
            new_id,
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
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
        selected_sequence_id: String,
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
            Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
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

        if self.control_mode == ControlMode::Pan {
            self.is_panning = true;
            self.drag_start = Some(self.last_top_left);

            return None;
        }

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
            }
        }

        // Now, with no objects in the way, check if we're clicking on a motion path handle to drag
        for (poly_index, polygon) in self.static_polygons.iter_mut().enumerate() {
            if polygon.name != "motion_path_handle".to_string() {
                continue;
            }

            if polygon.contains_point(&self.last_top_left, &camera) {
                self.dragging_path_handle = Some(polygon.id);
                self.dragging_path_object = polygon.source_polygon_id;
                self.dragging_path_keyframe = polygon.source_keyframe_id;
                self.drag_start = Some(self.last_top_left);

                return None; // nothing to add to undo stack
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

        // handle motion path handles
        if let Some(poly_id) = self.dragging_path_handle {
            if let Some(start) = self.drag_start {
                self.move_static_polygon(self.last_top_left, start, poly_id, window_size, device);
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
                .static_polygons
                .iter()
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

        // reset variables
        self.dragging_polygon = None;
        self.dragging_text = None;
        self.dragging_image = None;
        self.drag_start = None;
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

        // Remove existing motion path segments
        self.static_polygons.retain(|p| {
            p.name != "motion_path_segment"
                && p.name != "motion_path_handle"
                && p.name != "motion_path_arrow"
        });
    }
}

/// Creates a path segment using a rotated square
fn create_path_segment(
    window_size: &WindowSize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model_bind_group_layout: &Arc<wgpu::BindGroupLayout>,
    camera: &Camera,
    start: Point,
    end: Point,
    thickness: f32,
    selected_sequence_id: String,
    fill: [f32; 4],
    rotation: f32,
    length: f32,
) -> Polygon {
    // Calculate segment midpoint for position
    let position = Point {
        x: (start.x + end.x) / 2.0,
        y: (start.y + end.y) / 2.0,
    };

    // Create polygon using default square points
    Polygon::new(
        window_size,
        device,
        queue,
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
        // [0.5, 0.8, 1.0, 1.0], // light blue with some transparency
        fill,
        Stroke {
            thickness: 0.0,
            fill: rgb_to_wgpu(0, 0, 0, 1.0),
        },
        -1.0,
        -1,
        String::from("motion_path_segment"),
        Uuid::new_v4(),
        Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
    )
}

/// Creates a path handle for dragging and showing direction
fn create_path_handle(
    window_size: &WindowSize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model_bind_group_layout: &Arc<wgpu::BindGroupLayout>,
    camera: &Camera,
    end: Point,
    size: f32,
    selected_sequence_id: String,
    fill: [f32; 4],
) -> Polygon {
    Polygon::new(
        window_size,
        device,
        queue,
        model_bind_group_layout,
        camera,
        vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ],
        (size, size), // width = length of segment, height = thickness
        end,
        0.0,
        0.0,
        // [0.5, 0.8, 1.0, 1.0], // light blue with some transparency
        fill,
        Stroke {
            thickness: 0.0,
            fill: rgb_to_wgpu(0, 0, 0, 1.0),
        },
        -1.0,
        -1,
        String::from("motion_path_handle"),
        Uuid::new_v4(),
        Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
    )
}

/// Creates arrow for showing direction
fn create_path_arrow(
    window_size: &WindowSize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model_bind_group_layout: &Arc<wgpu::BindGroupLayout>,
    camera: &Camera,
    end: Point,
    size: f32,
    selected_sequence_id: String,
    fill: [f32; 4],
    rotation: f32,
) -> Polygon {
    Polygon::new(
        window_size,
        device,
        queue,
        model_bind_group_layout,
        camera,
        vec![
            // rightside up
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.5, y: 0.6 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.5, y: 1.0 },
            // upside down
            // Point { x: 1.0, y: 1.0 },
            // Point { x: 0.5, y: 0.4 },
            // Point { x: 0.0, y: 1.0 },
            // Point { x: 0.5, y: 0.0 },
        ],
        (size, size), // width = length of segment, height = thickness
        end,
        rotation,
        0.0,
        // [0.5, 0.8, 1.0, 1.0], // light blue with some transparency
        fill,
        Stroke {
            thickness: 0.0,
            fill: rgb_to_wgpu(0, 0, 0, 1.0),
        },
        -1.0,
        -1,
        String::from("motion_path_arrow"),
        Uuid::new_v4(),
        Uuid::from_str(&selected_sequence_id).expect("Couldn't convert string to uuid"),
    )
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
fn interpolate_position(start: &UIKeyframe, end: &UIKeyframe, time: f32) -> [i32; 2] {
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
    AnimationData, AnimationProperty, EasingType, KeyframeValue, ObjectType, Sequence, UIKeyframe,
};
use crate::camera::{Camera, CameraBinding};
use crate::dot::RingDot;
use crate::fonts::FontManager;
use crate::polygon::{Polygon, PolygonConfig, Stroke};
use crate::st_image::{StImage, StImageConfig};
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
}

fn get_color(color_index: u32) -> u32 {
    // Normalize the color_index to be within 0-29 range
    let normalized_index = color_index % 30;

    // Calculate which shade we're on (0-9)
    let shade_index = normalized_index / 3;

    // Calculate the shade intensity (0-255)
    // Using a range of 25-255 to avoid completely black colors
    155 + (shade_index * 10) // (255 - 25) / 10 ≈ 23 steps
}

fn get_full_color(index: u32) -> (u32, u32, u32) {
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
