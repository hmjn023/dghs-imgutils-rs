//! ONNX Runtime 推論処理におけるエラー型を定義します。

use thiserror::Error;

/// ONNX 推論中に発生するエラー
#[derive(Debug, Error)]
pub enum InferenceError {
    /// `ort` クレート側の内部エラー
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// ファイル I/O エラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// テンソル入出力の次元や不整合のエラー
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    /// スレッド同期・セッション初期化エラー
    #[error("Initialization error: {0}")]
    Initialization(String),
}
