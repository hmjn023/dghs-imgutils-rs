//! HuggingFace Hub からモデルファイルをダウンロードするためのエラー型。

use thiserror::Error;

/// Hub 操作で発生しうるエラー
#[derive(Debug, Error)]
pub enum HubError {
    /// `hf-hub` クレートの API エラー
    #[error("HuggingFace Hub API error: {0}")]
    Api(#[from] hf_hub::api::sync::ApiError),

    /// ファイル I/O エラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
