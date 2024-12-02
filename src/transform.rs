use crate::editor::Point;

#[derive(Clone, Copy)]
pub struct Transform {
    pub position: Point,
    // We could add scale and rotation here in the future if needed
}

impl Transform {
    pub fn new(position: Point) -> Self {
        Self { position }
    }
}
