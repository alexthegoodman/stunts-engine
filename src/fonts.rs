// use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// A font cache that supports both eager and lazy loading strategies
struct FontManager {
    // Maps font identifier to loaded font data
    // font_cache: RwLock<HashMap<String, Arc<FontMetadata>>>,
    font_data: [(String, Vec<u8>, String); 60],
}

impl FontManager {
    /// Loads about 60 fonts into memory
    pub fn new() -> Self {
        let font_data: [(String, Vec<u8>, String); 60] = [
            (
                "Actor".to_string(),
                include_bytes!("./fonts/actor/Actor-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Aladin".to_string(),
                include_bytes!("./fonts/aladin/Aladin-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Aleo".to_string(),
                include_bytes!("./fonts/aleo/Aleo[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Amiko".to_string(),
                include_bytes!("./fonts/amiko/Amiko-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Ballet".to_string(),
                include_bytes!("./fonts/ballet/Ballet[opsz].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Basic".to_string(),
                include_bytes!("./fonts/basic/Basic-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Bungee".to_string(),
                include_bytes!("./fonts/bungee/Bungee-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Caramel".to_string(),
                include_bytes!("./fonts/caramel/Caramel-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Cherish".to_string(),
                include_bytes!("./fonts/cherish/Cherish-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Coda".to_string(),
                include_bytes!("./fonts/coda/Coda-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "David Libre".to_string(),
                include_bytes!("./fonts/davidlibre/DavidLibre-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Dorsa".to_string(),
                include_bytes!("./fonts/dorsa/Dorsa-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Duru Sans".to_string(),
                include_bytes!("./fonts/durusans/DuruSans-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Dynalight".to_string(),
                include_bytes!("./fonts/dynalight/Dynalight-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Eater".to_string(),
                include_bytes!("./fonts/eater/Eater-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Epilogue".to_string(),
                include_bytes!("./fonts/epilogue/Epilogue[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Exo".to_string(),
                include_bytes!("./fonts/exo/Exo[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Explora".to_string(),
                include_bytes!("./fonts/explora/Explora-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Federo".to_string(),
                include_bytes!("./fonts/federo/Federo-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Figtree".to_string(),
                include_bytes!("./fonts/figtree/Figtree[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Flavors".to_string(),
                include_bytes!("./fonts/flavors/Flavors-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Galada".to_string(),
                include_bytes!("./fonts/galada/Galada-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Gantari".to_string(),
                include_bytes!("./fonts/gantari/Gantari[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Geo".to_string(),
                include_bytes!("./fonts/geo/Geo-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Glory".to_string(),
                include_bytes!("./fonts/glory/Glory[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "HappyMonkey".to_string(),
                include_bytes!("./fonts/happymonkey/HappyMonkey-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "HennyPenny".to_string(),
                include_bytes!("./fonts/hennypenny/HennyPenny-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Amiko".to_string(),
                include_bytes!("./fonts/iceberg/Iceberg-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Inika".to_string(),
                include_bytes!("./fonts/inika/Inika-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "InriaSans".to_string(),
                include_bytes!("./fonts/inriasans/InriaSans-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Jaro".to_string(),
                include_bytes!("./fonts/jaro/Jaro[opsz].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Kavoon".to_string(),
                include_bytes!("./fonts/kavoon/Kavoon-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Khula".to_string(),
                include_bytes!("./fonts/khula/Khula-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Kokoro".to_string(),
                include_bytes!("./fonts/kokoro/Kokoro-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Lemon".to_string(),
                include_bytes!("./fonts/lemon/Lemon-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Lexend".to_string(),
                include_bytes!("./fonts/lexend/Lexend[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Macondo".to_string(),
                include_bytes!("./fonts/macondo/Macondo-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Maitree".to_string(),
                include_bytes!("./fonts/maitree/Maitree-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Martel".to_string(),
                include_bytes!("./fonts/martel/Martel-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Maven Pro".to_string(),
                include_bytes!("./fonts/mavenpro/MavenPro[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Neuton".to_string(),
                include_bytes!("./fonts/neuton/Neuton-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "News Cycle".to_string(),
                include_bytes!("./fonts/newscycle/NewsCycle-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Nixie One".to_string(),
                include_bytes!("./fonts/nixieone/NixieOne-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Overlock".to_string(),
                include_bytes!("./fonts/overlock/Overlock-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Oxygen".to_string(),
                include_bytes!("./fonts/oxygen/Oxygen-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Play".to_string(),
                include_bytes!("./fonts/play/Play-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Quicksand".to_string(),
                include_bytes!("./fonts/quicksand/Quicksand[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Radley".to_string(),
                include_bytes!("./fonts/radley/Radley-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Rethink Sans".to_string(),
                include_bytes!("./fonts/rethinksans/RethinkSans[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Rosario".to_string(),
                include_bytes!("./fonts/rosario/Rosario[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Sacramento".to_string(),
                include_bytes!("./fonts/sacramento/Sacramento-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Salsa".to_string(),
                include_bytes!("./fonts/salsa/Salsa-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Scope One".to_string(),
                include_bytes!("./fonts/scopeone/ScopeOne-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Teachers".to_string(),
                include_bytes!("./fonts/teachers/Teachers[wght].ttf").to_vec(),
                "Variable".to_string(),
            ),
            (
                "Underdog".to_string(),
                include_bytes!("./fonts/underdog/Underdog-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Vibes".to_string(),
                include_bytes!("./fonts/vibes/Vibes-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Vina Sans".to_string(),
                include_bytes!("./fonts/vinasans/VinaSans-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Water Brush".to_string(),
                include_bytes!("./fonts/waterbrush/WaterBrush-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Wind Song".to_string(),
                include_bytes!("./fonts/windsong/WindSong-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
            (
                "Zain".to_string(),
                include_bytes!("./fonts/zain/Zain-Regular.ttf").to_vec(),
                "Regular".to_string(),
            ),
        ];

        Self { font_data }
    }
}