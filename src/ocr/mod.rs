pub mod constants;
pub mod detect;
pub mod recognize;

pub use detect::detect_text_with_paddleocr;
pub use recognize::ocr;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("Inference error: {0}")]
    Inference(#[from] crate::inference::InferenceError),

    #[error("Image error: {0}")]
    Image(#[from] crate::image::ImageError),

    #[error("Hub error: {0}")]
    Hub(#[from] crate::hub::HubError),

    #[error("Character dictionary error: {0}")]
    CharDict(String),

    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
