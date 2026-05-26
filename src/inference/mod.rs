//! ONNX Runtime 推論セッションの作成と、単一入力・単一出力の推論実行を共通化する共通ユーティリティを提供します。

pub mod error;
pub mod yolo;
pub mod classify;

pub use error::InferenceError;
pub use yolo::{
    DetectionResult, preprocess_image_yolo, xy_postprocess, yolo_nms, yolo_xywh2xyxy,
    postprocess_nms_yolo, postprocess_end2end_yolo, yolo_predict,
};

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
    use ort::ep::ExecutionProvider;
    use ort::execution_providers::{
        CUDAExecutionProvider, DirectMLExecutionProvider, OpenVINOExecutionProvider,
        TensorRTExecutionProvider,
    };

    let mut builder = Session::builder()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    let mut providers = Vec::new();

    // 1. TensorRT (NVIDIA 高性能 GPU)
    let trt = TensorRTExecutionProvider::default();
    match trt.is_available() {
        Ok(true) => {
            println!("[ort] TensorRT EP is available! Enabling TRT.");
            providers.push(trt.build());
        }
        Ok(false) => {}
        Err(e) => {
            println!("[ort] TensorRT EP check error: {:?}", e);
        }
    }

    // 2. CUDA (NVIDIA 標準 GPU)
    let cuda = CUDAExecutionProvider::default();
    match cuda.is_available() {
        Ok(true) => {
            println!("[ort] CUDA EP is available! Enabling NVIDIA GPU acceleration.");
            providers.push(cuda.build());
        }
        Ok(false) => {
            println!("[ort] CUDA EP is not available (returned false).");
        }
        Err(e) => {
            println!("[ort] CUDA EP check error: {:?}", e);
        }
    }

    // 3. DirectML (Windows NPU/GPU)
    let dml = DirectMLExecutionProvider::default();
    match dml.is_available() {
        Ok(true) => {
            println!("[ort] DirectML EP is available! Enabling DirectML.");
            providers.push(dml.build());
        }
        Ok(false) => {}
        Err(e) => {
            println!("[ort] DirectML EP check error: {:?}", e);
        }
    }

    // 4. OpenVINO (Intel CPU/GPU/NPU)
    let openvino = OpenVINOExecutionProvider::default();
    match openvino.is_available() {
        Ok(true) => {
            println!("[ort] OpenVINO EP is available! Enabling Intel NPU/GPU/CPU acceleration.");
            providers.push(openvino.build());
        }
        Ok(false) => {}
        Err(e) => {
            println!("[ort] OpenVINO EP check error: {:?}", e);
        }
    }

    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    }

    let session = builder
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
