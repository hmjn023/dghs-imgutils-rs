pub mod dwpose;
pub mod format;
pub mod visual;

pub use dwpose::dwpose_estimate;
pub use format::OP18KeyPointSet;
pub use visual::op18_visualize;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PoseError {
    #[error("Inference error: {0}")]
    Inference(#[from] crate::inference::InferenceError),

    #[error("Image error: {0}")]
    Image(#[from] crate::image::ImageError),

    #[error("No persons detected")]
    NoPersonDetected,
}
