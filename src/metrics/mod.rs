//! 評価尺度（Metrics）および類似度比較に関する機能を提供します。

pub mod ccip;
pub mod dbaesthetic;
pub mod laplacian;
pub mod lpips;
pub mod psnr_;

pub use ccip::{
    ccip_batch_differences, ccip_batch_extract_features, ccip_batch_same, ccip_clustering,
    ccip_default_clustering_params, ccip_default_threshold, ccip_difference, ccip_extract_feature,
    ccip_merge, ccip_same,
};

pub use dbaesthetic::anime_dbaesthetic;
pub use laplacian::laplacian_score;
pub use lpips::{lpips_clustering, lpips_difference, lpips_extract_feature};
pub use psnr_::psnr;

use thiserror::Error;

/// CCIP 処理において発生しうるエラーの定義。
#[derive(Error, Debug)]
pub enum CcipError {
    /// 画像処理エラー
    #[error("Image error: {0}")]
    Image(#[from] crate::image::ImageError),

    /// ONNX 推論エラー
    #[error("Inference error: {0}")]
    Inference(#[from] crate::inference::InferenceError),

    /// Shape error
    #[error("Shape error: {0}")]
    Shape(String),

    /// ONNX Runtime のエラー
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    /// Hugging Face ハブのエラー
    #[error("Hub error: {0}")]
    Hub(#[from] crate::hub::HubError),

    /// JSON パースエラー
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O エラー
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// 無効な引数エラー
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<ndarray::ShapeError> for CcipError {
    fn from(e: ndarray::ShapeError) -> Self {
        CcipError::Shape(e.to_string())
    }
}
