// use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// Represents metadata about a font file
struct FontMetadata {
    path: PathBuf,
    family_name: String,
    style: String,
}

// A font cache that supports both eager and lazy loading strategies
struct FontManager {
    // Maps font identifier to loaded font data
    font_cache: RwLock<HashMap<String, Arc<Vec<u8>>>>,
    // Maps font identifier to metadata
    font_registry: HashMap<String, FontMetadata>,
    fonts_dir: PathBuf,
    max_cache_size: usize,
}

impl FontManager {
    pub fn new(fonts_dir: PathBuf, max_cache_size: usize) -> Self {
        Self {
            font_cache: RwLock::new(HashMap::new()),
            font_registry: HashMap::new(),
            fonts_dir,
            max_cache_size,
        }
    }

    // Initialize by scanning font directory and building metadata registry
    pub fn initialize(&mut self) -> Result<(), std::io::Error> {
        for entry in fs::read_dir(&self.fonts_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .map_or(false, |ext| ext == "ttf" || ext == "otf")
            {
                // In a real implementation, you'd parse the font to get actual metadata
                let font_id = path.file_stem().unwrap().to_string_lossy().to_string();
                self.font_registry.insert(
                    font_id.clone(),
                    FontMetadata {
                        path,
                        family_name: font_id,
                        style: "Regular".to_string(),
                    },
                );
            }
        }
        Ok(())
    }

    // Lazy loading approach - load font only when requested
    pub fn get_font(&self, font_id: &str) -> Option<Arc<Vec<u8>>> {
        // Check if font is already cached
        if let Some(font_data) = self
            .font_cache
            .read()
            .expect("Couldn't read font cache")
            .get(font_id)
        {
            return Some(Arc::clone(font_data));
        }

        // If not in cache, load it
        if let Some(metadata) = self.font_registry.get(font_id) {
            if let Ok(font_data) = fs::read(&metadata.path) {
                let font_data = Arc::new(font_data);

                // Update cache with new font data
                let mut cache = self
                    .font_cache
                    .write()
                    .expect("Couldn't get font cache write guard");

                // If cache is full, remove least recently used entry
                if cache.len() >= self.max_cache_size {
                    if let Some(oldest_key) = cache.keys().next().cloned() {
                        cache.remove(&oldest_key);
                    }
                }

                cache.insert(font_id.to_string(), Arc::clone(&font_data));
                return Some(font_data);
            }
        }
        None
    }

    // // Eager loading approach - preload all fonts
    // pub fn preload_all_fonts(&self) -> Result<(), std::io::Error> {
    //     let mut cache = self
    //         .font_cache
    //         .write()
    //         .expect("Couldn't get font cache write guard");

    //     for (font_id, metadata) in &self.font_registry {
    //         let font_data = fs::read(&metadata.path)?;
    //         cache.insert(font_id.clone(), Arc::new(font_data));
    //     }
    //     Ok(())
    // }
}

// Example usage
// fn main() {
//     // Initialize with a 10-font cache size
//     let mut font_manager = FontManager::new(PathBuf::from("/usr/share/fonts"), 10);
//     font_manager.initialize().expect("Failed to initialize font manager");

//     // Lazy loading approach
//     if let Some(font_data) = font_manager.get_font("Arial") {
//         // Use font_data with fontdue
//         // let font = fontdue::Font::from_bytes(font_data.as_slice(), fontdue::FontSettings::default());
//     }

//     // Or preload all fonts at startup
//     font_manager.preload_all_fonts().expect("Failed to preload fonts");
// }
