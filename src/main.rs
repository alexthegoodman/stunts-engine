mod animations;
mod camera;
mod dot;
mod editor;
mod fetchers;
mod fonts;
mod motion_path;
mod polygon;
mod st_image;
mod st_video;
mod text;
mod text_due;
mod timelines;
mod transform;
mod vertex;

#[cfg(target_os = "windows")]
mod export;

#[cfg(target_os = "windows")]
mod transcode;

#[cfg(target_os = "windows")]
mod capture;

#[cfg(target_arch = "wasm32")]
mod mp4box;

fn main() {
    println!("Hello, Stunts!");
}
