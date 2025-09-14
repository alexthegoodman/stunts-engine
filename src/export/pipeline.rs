use crate::{
    animations::Sequence,
    camera::{Camera3D as Camera, CameraBinding},
    editor::{
        Editor, Viewport, WindowSize, WindowSizeShader, CANVAS_HORIZ_OFFSET, CANVAS_VERT_OFFSET,
    },
    timelines::{SavedTimelineStateConfig, TimelineSequence},
    vertex::Vertex,
};
use cgmath::{Vector2, Vector3};
use crate::gpu_resources::GpuResources;
use std::sync::{Arc, Mutex};
use wgpu::{util::DeviceExt, RenderPipeline};

use super::frame_buffer::FrameCaptureBuffer;

pub struct ExportPipeline {
    // pub device: Option<wgpu::Device>,
    // pub queue: Option<wgpu::Queue>,
    pub gpu_resources: Option<Arc<GpuResources>>,
    pub camera: Option<Camera>,
    pub camera_binding: Option<CameraBinding>,
    pub render_pipeline: Option<RenderPipeline>,
    pub texture: Option<Arc<wgpu::Texture>>,
    pub view: Option<Arc<wgpu::TextureView>>,
    pub depth_view: Option<wgpu::TextureView>,
    pub window_size_bind_group: Option<wgpu::BindGroup>,
    pub export_editor: Option<Editor>,
    pub frame_buffer: Option<FrameCaptureBuffer>,
}

impl ExportPipeline {
    pub fn new() -> Self {
        ExportPipeline {
            // device: None,
            // queue: None,
            gpu_resources: None,
            camera: None,
            camera_binding: None,
            render_pipeline: None,
            texture: None,
            view: None,
            depth_view: None,
            window_size_bind_group: None,
            export_editor: None,
            frame_buffer: None,
        }
    }

    pub async fn initialize(
        &mut self,
        window_size: WindowSize,
        sequences: Vec<Sequence>,
        video_current_sequence_timeline: SavedTimelineStateConfig,
        video_width: u32,
        video_height: u32,
        project_id: String,
    ) {
        let mut camera = Camera::new(
            //window_size
            WindowSize {
                width: video_width,
                height: video_height,
            },
        );

        camera.position = Vector3::new(85.0, -535.0, 0.0);
        // camera.zoom = 2.4;

        let viewport = Arc::new(Mutex::new(Viewport::new(
            // swap for video dimensions?
            // window_size.width as f32,
            // window_size.height as f32,
            video_width as f32,
            video_height as f32,
        )));

        // create a dedicated editor so it can be used in the async thread
        let mut export_editor = Editor::new(viewport, project_id.clone());

        // continue on with wgpu items
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None, // no surface desired for export
                force_fallback_adapter: false,
            })
            .await
            .expect("Couldn't get gpu adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Couldn't get gpu device");

        let mut camera_binding = CameraBinding::new(&device);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                // width: window_size.width.clone(),
                // height: window_size.height.clone(),
                width: video_width.clone(),
                height: video_height.clone(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // used in a multisampled environment
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Stunts Engine Export Depth Texture"),
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_stencil_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Existing uniform buffer binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Texture binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Stunts Engine Export Model Layout"),
            });

        let model_bind_group_layout = Arc::new(model_bind_group_layout);

        let group_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Existing uniform buffer binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("export group_bind_group_layout"),
            });

        let group_bind_group_layout = Arc::new(group_bind_group_layout);

        let window_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[WindowSizeShader {
                // swap for vidoe dimensions?
                // width: window_size.width as f32,
                // height: window_size.height as f32,
                width: video_width.clone() as f32,
                height: video_height.clone() as f32,
            }]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let window_size_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let window_size_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &window_size_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: window_size_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Define the layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Stunts Engine Export Pipeline Layout"),
            bind_group_layouts: &[
                &camera_binding.bind_group_layout,
                &model_bind_group_layout,
                &window_size_bind_group_layout,
                &group_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load the shaders
        let shader_module_vert_primary =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Stunts Engine Export Vert Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vert_primary.wgsl").into()),
            });

        let shader_module_frag_primary =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Stunts Engine Export Frag Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frag_primary.wgsl").into()),
            });

        // let swapchain_capabilities = gpu_resources
        //     .surface
        //     .get_capabilities(&gpu_resources.adapter);
        // let swapchain_format = swapchain_capabilities.formats[0]; // Choosing the first available format
        // let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb; // hardcode for now - may be able to change from the floem requirement
        let swapchain_format = wgpu::TextureFormat::Bgra8Unorm;
        // let swapchain_format = wgpu::TextureFormat::Rgba8Unorm;

        // Configure the render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Stunts Engine Export Render Pipeline"),
            layout: Some(&pipeline_layout),
            multiview: None,
            // cache: None,
            vertex: wgpu::VertexState {
                module: &shader_module_vert_primary,
                entry_point: "vs_main", // name of the entry point in your vertex shader
                buffers: &[Vertex::desc()], // Make sure your Vertex::desc() matches your vertex structure
                // compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module_frag_primary,
                entry_point: "fs_main", // name of the entry point in your fragment shader
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    // blend: Some(wgpu::BlendState::REPLACE),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                // compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            // primitive: wgpu::PrimitiveState::default(),
            // depth_stencil: None,
            // multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                conservative: false,
                topology: wgpu::PrimitiveTopology::TriangleList, // how vertices are assembled into geometric primitives
                // strip_index_format: Some(wgpu::IndexFormat::Uint32),
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // Counter-clockwise is considered the front face
                // none cull_mode
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                // Other properties such as conservative rasterization can be set here
                unclipped_depth: false,
            },
            depth_stencil: Some(depth_stencil_state), // Optional, only if you are using depth testing
            multisample: wgpu::MultisampleState {
                // count: 4, // effect performance
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                // width: window_size.width,
                // height: window_size.height,
                width: video_width.clone(),
                height: video_height.clone(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            // sample_count: 4,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: swapchain_format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("Export render texture"),
            view_formats: &[],
        });

        let texture = Arc::new(texture);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let view = Arc::new(view);

        camera_binding.update_3d(&queue, &camera);

        let gpu_resources = GpuResources::new(adapter, device, queue);

        let gpu_resources = Arc::new(gpu_resources);

        // set needed editor properties
        export_editor.model_bind_group_layout = Some(model_bind_group_layout);
        export_editor.group_bind_group_layout = Some(group_bind_group_layout);
        export_editor.gpu_resources = Some(gpu_resources.clone());

        // let gpu_resources = export_editor
        //     .gpu_resources
        //     .as_ref()
        //     .expect("Couldn't get gpu resources");

        // restore objects to the editor
        sequences.iter().enumerate().for_each(|(i, s)| {
            export_editor.restore_sequence_objects(
                &s,
                WindowSize {
                    // width: window_size.width as u32,
                    // height: window_size.height as u32,
                    width: video_width.clone(),
                    height: video_height.clone(),
                },
                &camera,
                if i == 0 { false } else { true },
                // &gpu_resources.device,
                // &gpu_resources.queue,
            );
        });

        // begin playback
        export_editor.camera = Some(camera);
        let now = std::time::Instant::now();
        export_editor.video_start_playing_time = Some(now.clone());

        export_editor.video_current_sequence_timeline = Some(video_current_sequence_timeline);
        export_editor.video_current_sequences_data = Some(sequences);

        export_editor.video_is_playing = true;

        // also set motion path playing
        export_editor.start_playing_time = Some(now);
        export_editor.is_playing = true;

        println!("Video exporting!");

        // self.device = Some(device);
        // self.queue = Some(queue);
        self.gpu_resources = export_editor.gpu_resources.clone();
        self.camera = Some(camera);
        self.camera_binding = Some(camera_binding);
        self.render_pipeline = Some(render_pipeline);
        self.texture = Some(texture);
        self.view = Some(view);
        self.depth_view = Some(depth_view);
        self.window_size_bind_group = Some(window_size_bind_group);
        self.export_editor = Some(export_editor);
    }

    pub fn render_frame(&mut self, current_time: f64) {
        let mut editor = self.export_editor.as_mut().expect("Couldn't get editor");
        let gpu_resources = self
            .gpu_resources
            .as_ref()
            .expect("Couldn't get gpu resources");
        let device = &gpu_resources.device;
        let queue = &gpu_resources.queue;
        // let device = self.device.as_ref().expect("Couldn't get device");
        // let queue = self.queue.as_ref().expect("Couldn't get queue");
        let view = self.view.as_ref().expect("Couldn't get texture view");
        let depth_view = self
            .depth_view
            .as_ref()
            .expect("Couldn't get depth texture view");
        let render_pipeline = self
            .render_pipeline
            .as_ref()
            .expect("Couldn't get render pipeline");
        let camera_binding = self
            .camera_binding
            .as_ref()
            .expect("Couldn't get camera binding");
        let window_size_bind_group = self
            .window_size_bind_group
            .as_ref()
            .expect("Couldn't get window size bind group");
        let camera = self.camera.as_ref().expect("Couldn't get camera"); // careful, we have a camera on editor and on self
        let texture = self.texture.as_ref().expect("Couldn't get texture");
        let frame_buffer = self
            .frame_buffer
            .as_ref()
            .expect("Couldn't get frame buffer");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    // resolve_target: Some(&resolve_view), // not sure how to add without surface
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view, // This is the depth texture view
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear to max depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None, // Set this if using stencil
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&render_pipeline);

            // actual rendering commands
            editor.step_video_animations(&camera, Some(current_time));
            editor.step_motion_path_animations(&camera, Some(current_time));

            render_pass.set_bind_group(0, &camera_binding.bind_group, &[]);
            render_pass.set_bind_group(2, window_size_bind_group, &[]);

            // draw static (internal) polygons
            for (poly_index, polygon) in editor.static_polygons.iter().enumerate() {
                polygon
                    .transform
                    .update_uniform_buffer(&queue, &camera.window_size);
                render_pass.set_bind_group(1, &polygon.bind_group, &[]);
                render_pass.set_bind_group(3, &polygon.group_bind_group, &[]);
                render_pass.set_vertex_buffer(0, polygon.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(polygon.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..polygon.indices.len() as u32, 0, 0..1);
            }

            // draw polygons
            for (poly_index, polygon) in editor.polygons.iter().enumerate() {
                if !polygon.hidden {
                    polygon
                        .transform
                        .update_uniform_buffer(&queue, &camera.window_size);
                    render_pass.set_bind_group(1, &polygon.bind_group, &[]);
                    render_pass.set_bind_group(3, &polygon.group_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, polygon.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        polygon.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..polygon.indices.len() as u32, 0, 0..1);
                }
            }

            // draw text items
            for (text_index, text_item) in editor.text_items.iter().enumerate() {
                if !text_item.hidden {
                    if !text_item.background_polygon.hidden {
                        text_item
                            .background_polygon
                            .transform
                            .update_uniform_buffer(&gpu_resources.queue, &camera.window_size);

                        render_pass.set_bind_group(
                            1,
                            &text_item.background_polygon.bind_group,
                            &[],
                        );
                        render_pass.set_bind_group(
                            3,
                            &text_item.background_polygon.group_bind_group,
                            &[],
                        );
                        render_pass.set_vertex_buffer(
                            0,
                            text_item.background_polygon.vertex_buffer.slice(..),
                        );
                        render_pass.set_index_buffer(
                            text_item.background_polygon.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(
                            0..text_item.background_polygon.indices.len() as u32,
                            0,
                            0..1,
                        );
                    }

                    text_item
                        .transform
                        .update_uniform_buffer(&queue, &camera.window_size);
                    render_pass.set_bind_group(1, &text_item.bind_group, &[]);
                    render_pass.set_bind_group(3, &text_item.group_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, text_item.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        text_item.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..text_item.indices.len() as u32, 0, 0..1);
                }
            }

            // draw image items
            for (image_index, st_image) in editor.image_items.iter().enumerate() {
                if !st_image.hidden {
                    st_image
                        .transform
                        .update_uniform_buffer(&queue, &camera.window_size);
                    render_pass.set_bind_group(1, &st_image.bind_group, &[]);
                    render_pass.set_bind_group(3, &st_image.group_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, st_image.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        st_image.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..st_image.indices.len() as u32, 0, 0..1);
                }
            }

            // draw video items
            for (video_index, st_video) in editor.video_items.iter().enumerate() {
                if !st_video.hidden {
                    st_video
                        .transform
                        .update_uniform_buffer(&queue, &camera.window_size);
                    render_pass.set_bind_group(1, &st_video.bind_group, &[]);
                    render_pass.set_bind_group(3, &st_video.group_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, st_video.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        st_video.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..st_video.indices.len() as u32, 0, 0..1);
                }
            }

            // Drop the render pass before doing texture copies
            drop(render_pass);

            frame_buffer.capture_frame(device, queue, texture, &mut encoder);

            let command_buffer = encoder.finish();
            queue.submit(std::iter::once(command_buffer));
        }
    }
}
