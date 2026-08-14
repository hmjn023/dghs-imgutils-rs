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

    /// ユーザーが指定したbackendまたはprecisionが不正な場合
    #[error("Invalid inference option: {0}")]
    InvalidInput(String),

    /// 指定されたprovider/runtimeがworkerに存在しない場合
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    /// モデルが指定providerの互換条件を満たさない場合
    #[error("Model unsupported by backend: {0}")]
    ModelUnsupported(String),

    /// providerのcompileまたはsession初期化に失敗した場合
    #[error("Backend compilation failed: {0}")]
    CompilationFailed(String),

    /// providerまたは実行時のメモリ不足
    #[error("Inference out of memory: {0}")]
    OutOfMemory(String),

    /// スレッド同期・セッション初期化エラー
    #[error("Initialization error: {0}")]
    Initialization(String),
}
