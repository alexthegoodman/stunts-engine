use serde::{Deserialize, Serialize};
use crate::{
    animations::Sequence, 
    polygon::SavedPolygonConfig, 
    timelines::SavedTimelineStateConfig,
};
use directories::{BaseDirs, UserDirs};
use std::{fs, path::PathBuf, sync::MutexGuard};

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedState {
    pub id: String,
    // pub name: String,
    pub sequences: Vec<Sequence>,
    pub timeline_state: SavedTimelineStateConfig,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ProjectData {
    pub project_id: String,
    pub project_name: String,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ProjectsDataFile {
    pub projects: Vec<ProjectData>,
}

pub fn get_ground_truth_dir() -> Option<PathBuf> {
    UserDirs::new().map(|user_dirs| {
        let common_os = user_dirs
            .document_dir()
            .expect("Couldn't find Documents directory")
            .join("Stunts");
        fs::create_dir_all(&common_os)
            .ok()
            .expect("Couldn't check or create Stunts directory");
        common_os
    })
}

pub fn save_saved_state(saved_state: MutexGuard<SavedState>) {
    let owned = saved_state.to_owned();
    save_saved_state_raw(owned);
}

pub fn save_saved_state_raw(saved_state: SavedState) {
    let json = serde_json::to_string_pretty(&saved_state).expect("Couldn't serialize saved state");
    let sync_dir = get_ground_truth_dir().expect("Couldn't get Stunts directory");
    let project_dir = sync_dir.join("projects").join(saved_state.id.clone());
    let save_path = project_dir.join("project_data.json");

    println!("Saving saved state... {}", save_path.display());

    // disabled for testing
    // fs::write(&save_path, json).expect("Couldn't write saved state");

    drop(saved_state);

    println!("Saved!");
}