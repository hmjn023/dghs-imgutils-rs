//! ONNX Runtime 推論セッションの作成と、単一入力・単一出力の推論実行を共通化する共通ユーティリティを提供します。

pub mod classify;
pub mod error;
pub mod yolo;

pub use error::InferenceError;
pub use yolo::{
    DetectionResult, postprocess_end2end_yolo, postprocess_nms_yolo, preprocess_image_yolo,
    xy_postprocess, yolo_nms, yolo_predict, yolo_xywh2xyxy,
};

use once_cell::sync::Lazy;
use ort::session::Session;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

const OPENVINO_DEVICE_ENV: &str = "DGHS_ORT_DEVICE";
const DEFAULT_OPENVINO_DEVICE: &str = "AUTO:NPU,GPU,CPU";

/// ONNX セッションのキャッシュ。
/// モデルファイルパスをキーとして、作成済みセッションを共有する。
/// `Session::run` が `&mut self` を必要とするため、`Mutex` で保護する。
static SESSION_CACHE: Lazy<Mutex<HashMap<String, Arc<Mutex<Session>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// ONNX Runtime のグローバル環境を初期化します。
///
/// `ort` クレートはセッションの初回作成時に自動で初期化を行うため、この関数の呼び出しは通常任意です。
pub fn init_onnx_runtime() -> Result<(), InferenceError> {
    // 将来的な環境のカスタマイズ用のプレースホルダ
    Ok(())
}

/// `DGHS_ORT_DEVICE` で指定された OpenVINO デバイスを優先して、
/// 指定されたモデルファイルのパスから ONNX Runtime の Session を作成します。
///
/// `DGHS_ORT_DEVICE` には `GPU`（Intel XPU）、`NPU`、`CPU`、または
/// `AUTO:GPU,NPU,CPU` のような OpenVINO のデバイス指定を設定できます。
/// 未設定時は `AUTO:NPU,GPU,CPU`（NPU 優先）を使用します。
///
/// # 引数
///
/// * `model_path` - ONNX モデルファイルへのパス
pub fn create_onnx_session<P: AsRef<Path>>(model_path: P) -> Result<Session, InferenceError> {
    use ort::ep::{CUDA, DirectML, ExecutionProvider, OpenVINO, TensorRT};

    let mut builder =
        Session::builder().map_err(|e| InferenceError::Initialization(e.to_string()))?;

    let mut providers = Vec::new();

    // 1. TensorRT (NVIDIA 高性能 GPU)
    let trt = TensorRT::default();
    match trt.is_available() {
        Ok(true) => {
            info!("[ort] TensorRT EP is available! Enabling TRT.");
            providers.push(trt.build());
        }
        Ok(false) => {}
        Err(e) => {
            warn!("[ort] TensorRT EP check error: {:?}", e);
        }
    }

    // 2. CUDA (NVIDIA 標準 GPU)
    let cuda = CUDA::default();
    match cuda.is_available() {
        Ok(true) => {
            info!("[ort] CUDA EP is available! Enabling NVIDIA GPU acceleration.");
            providers.push(cuda.build());
        }
        Ok(false) => {
            info!("[ort] CUDA EP is not available (returned false).");
        }
        Err(e) => {
            warn!("[ort] CUDA EP check error: {:?}", e);
        }
    }

    // 3. DirectML (Windows NPU/GPU)
    let dml = DirectML::default();
    match dml.is_available() {
        Ok(true) => {
            info!("[ort] DirectML EP is available! Enabling DirectML.");
            providers.push(dml.build());
        }
        Ok(false) => {}
        Err(e) => {
            warn!("[ort] DirectML EP check error: {:?}", e);
        }
    }

    // 4. OpenVINO (Intel CPU/GPU/NPU)
    let device_type = env::var(OPENVINO_DEVICE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_OPENVINO_DEVICE.to_owned());
    let openvino = OpenVINO::default().with_device_type(&device_type);
    match openvino.is_available() {
        Ok(true) => {
            info!("[ort] OpenVINO EP is available! Enabling device type {device_type}.");
            providers.push(openvino.build().error_on_failure());
        }
        Ok(false) => {}
        Err(e) => {
            warn!("[ort] OpenVINO EP check error: {:?}", e);
        }
    }

    if !providers.is_empty() {
        match builder.clone().with_execution_providers(providers) {
            Ok(b) => builder = b,
            Err(e) => {
                warn!(
                    "[ort] Failed to register execution providers, falling back to CPU: {:?}",
                    e
                );
            }
        }
    }

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    Ok(session)
}

/// キャッシュ付きで ONNX セッションを取得する。
///
/// モデルファイルパスが同じ場合はキャッシュからセッションを返す。
/// セッションは `Arc<Mutex<Session>>` で共有され、スレッドセーフに利用できる。
///
/// # 引数
///
/// * `model_path` - ONNX モデルファイルへのパス
pub fn get_or_create_session<P: AsRef<Path>>(
    model_path: P,
) -> Result<Arc<Mutex<Session>>, InferenceError> {
    let path_str = model_path.as_ref().to_string_lossy().to_string();

    let mut cache = SESSION_CACHE
        .lock()
        .map_err(|e| InferenceError::Initialization(format!("Session cache lock poisoned: {e}")))?;

    if let Some(session) = cache.get(&path_str) {
        return Ok(Arc::clone(session));
    }

    let session = create_onnx_session(&path_str)?;
    let session = Arc::new(Mutex::new(session));
    cache.insert(path_str, Arc::clone(&session));
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
