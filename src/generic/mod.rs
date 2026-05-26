pub mod clip;
pub mod enhance;
pub mod siglip;
pub mod timm;
pub mod yoloseg;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenericError {
    #[error("Image error: {0}")]
    Image(#[from] crate::image::ImageError),
    #[error("Inference error: {0}")]
    Inference(#[from] crate::inference::InferenceError),
    #[error("Hub error: {0}")]
    Hub(#[from] crate::hub::HubError),
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Shape error: {0}")]
    Shape(String),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
}

impl From<ndarray::ShapeError> for GenericError {
    fn from(e: ndarray::ShapeError) -> Self {
        GenericError::Shape(e.to_string())
    }
}
