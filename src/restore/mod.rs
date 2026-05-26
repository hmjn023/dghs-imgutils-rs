//! 画像復元（ノイズ除去・劣化回復）機能。
//!
//! NafNet / SCUNet による ONNX ベースの復元と、
//! フィルタリングによる adversarial noise 除去を提供します。

pub mod adversarial;
pub mod nafnet;
pub mod scunet;

use thiserror::Error;

/// 画像復元モジュールのエラー型。
#[derive(Debug, Error)]
pub enum RestoreError {
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

impl From<ndarray::ShapeError> for RestoreError {
    fn from(e: ndarray::ShapeError) -> Self {
        RestoreError::Shape(e.to_string())
    }
}
