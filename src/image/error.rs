//! 画像処理およびテンソル前処理におけるエラー型を定義します。

use thiserror::Error;

/// 画像前処理・操作で発生するエラー
#[derive(Debug, Error)]
pub enum ImageError {
    /// `image` クレートでの処理エラー
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    /// テンソル変換時の形状・次元エラー
    #[error("Shape error: {0}")]
    Shape(String),

    /// 引数や設定の異常エラー
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// I/O エラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP ダウンロードエラー
    #[error("HTTP error: {0}")]
    Http(String),

    /// Base64 デコードエラー
    #[error("Base64 decode error: {0}")]
    Base64(String),
}
