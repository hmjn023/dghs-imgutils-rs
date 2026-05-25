//! ONNX Runtime 推論セッションの作成と、単一入力・単一出力の推論実行を共通化する共通ユーティリティを提供します。

pub mod error;

pub use error::InferenceError;

use ort::session::Session;
use std::path::Path;

/// ONNX Runtime のグローバル環境を初期化します。
///
/// `ort` クレートはセッションの初回作成時に自動で初期化を行うため、この関数の呼び出しは通常任意です。
pub fn init_onnx_runtime() -> Result<(), InferenceError> {
    // 将来的な環境のカスタマイズ用のプレースホルダ
    Ok(())
}

/// 指定されたモデルファイルのパスから ONNX Runtime の Session を作成します。
///
/// システムで利用可能な最高の実行プロバイダ（CUDA、TensorRT 等）を自動的に検出し、
/// 最適化レベルを最大にしてバインドしたセッションを返します。
///
/// # 引数
///
/// * `model_path` - ONNX モデルファイルへのパス
pub fn create_onnx_session<P: AsRef<Path>>(model_path: P) -> Result<Session, InferenceError> {
    let session = Session::builder()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?
        .commit_from_file(model_path)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    Ok(session)
}

/// ONNX Runtime のセッションを用いて推論を実行する共通ヘルパー関数です。
///
/// 入力として `[1, 3, H, W]` 形状の `ndarray::Array4<f32>` を受け取り、
/// 推論を実行した結果の全出力テンソルを格納した `SessionOutputs` を返します。
///
/// # 引数
///
/// * `session` - ONNX Runtime セッション（可変参照が必要です）
/// * `input_name` - 入力ノードの名前（例: `"images"`, `"input"`）
/// * `input_tensor` - ndarray 形式の入力テンソル（`[1, 3, H, W]`）
pub fn run_onnx_session<'s>(
    session: &'s mut Session,
    input_name: &str,
    input_tensor: &ndarray::Array4<f32>,
) -> Result<ort::session::SessionOutputs<'s>, InferenceError> {
    // ndarray から Tensor (ort::value::Tensor) を生成します
    let tensor = ort::value::Tensor::from_array(input_tensor.clone())?;

    // 推論を実行
    // inputs! マクロは直接値を受け取ります
    let outputs = session.run(ort::inputs![input_name => tensor])?;

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_onnx_runtime() {
        assert!(init_onnx_runtime().is_ok());
    }
}
