//! 画像超解像（アップスケーリング）機能。
//!
//! CDC モデルによるアニメ画像の 4x 超解像を提供します。

pub mod cdc;

use thiserror::Error;

/// 画像超解像モジュールのエラー型。
#[derive(Debug, Error)]
pub enum UpscaleError {
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
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Shape error: {0}")]
    Shape(String),
}

impl From<ndarray::ShapeError> for UpscaleError {
    fn from(e: ndarray::ShapeError) -> Self {
        UpscaleError::Shape(e.to_string())
    }
}
