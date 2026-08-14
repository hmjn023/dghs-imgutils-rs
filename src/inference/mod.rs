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

/// Returns the effective OpenVINO device policy for the current process.
///
/// Empty values are treated the same as an unset `DGHS_ORT_DEVICE`.
pub fn openvino_device_type() -> String {
    env::var(OPENVINO_DEVICE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_OPENVINO_DEVICE.to_owned())
}

fn is_explicit_openvino_device(device_type: &str) -> bool {
    matches!(
        device_type.trim().to_ascii_uppercase().as_str(),
        "CPU" | "GPU" | "NPU"
    )
}

fn provider_registration_error<E: std::fmt::Display>(
    error: E,
    explicit_device: bool,
) -> Result<(), InferenceError> {
    if explicit_device {
        Err(InferenceError::Initialization(format!(
            "Failed to register execution providers: {error}"
        )))
    } else {
        Ok(())
    }
}

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

    let device_type = openvino_device_type();
    let explicit_openvino_device = is_explicit_openvino_device(&device_type);

    // Register an explicitly requested OpenVINO device first so that it has
    // priority over other available execution providers.
    let openvino = OpenVINO::default().with_device_type(&device_type);
    let mut openvino_provider = match openvino.is_available() {
        Ok(true) => {
            info!("[ort] OpenVINO EP is available! Enabling device type {device_type}.");
            Some(openvino.build().error_on_failure())
        }
        Ok(false) if explicit_openvino_device => {
            return Err(InferenceError::Initialization(format!(
                "OpenVINO EP is unavailable for requested device {device_type}"
            )));
        }
        Ok(false) => None,
        Err(e) if explicit_openvino_device => {
            return Err(InferenceError::Initialization(format!(
                "OpenVINO EP availability check failed for requested device {device_type}: {e}"
            )));
        }
        Err(e) => {
            warn!("[ort] OpenVINO EP check error: {:?}", e);
            None
        }
    };

    if explicit_openvino_device {
        if let Some(provider) = openvino_provider.take() {
            providers.push(provider);
        }
    }

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

    // 4. OpenVINO (Intel CPU/GPU/NPU), after the existing providers for the
    // automatic policy. Explicit CPU/GPU/NPU requests were added above.
    if !explicit_openvino_device {
        if let Some(provider) = openvino_provider {
            providers.push(provider);
        }
    }

    if !providers.is_empty() {
        match builder.clone().with_execution_providers(providers) {
            Ok(b) => builder = b,
            Err(e) => {
                if let Err(error) = provider_registration_error(&e, explicit_openvino_device) {
                    return Err(error);
                }
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
    use once_cell::sync::Lazy;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static DEVICE_ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct DeviceEnvGuard {
        previous: Option<OsString>,
    }

    impl DeviceEnvGuard {
        fn set(value: Option<&str>) -> Self {
            let previous = env::var_os(OPENVINO_DEVICE_ENV);
            unsafe {
                match value {
                    Some(value) => env::set_var(OPENVINO_DEVICE_ENV, value),
                    None => env::remove_var(OPENVINO_DEVICE_ENV),
                }
            }
            Self { previous }
        }
    }

    impl Drop for DeviceEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.previous.take() {
                    Some(value) => env::set_var(OPENVINO_DEVICE_ENV, value),
                    None => env::remove_var(OPENVINO_DEVICE_ENV),
                }
            }
        }
    }

    #[test]
    fn test_init_onnx_runtime() {
        assert!(init_onnx_runtime().is_ok());
    }

    #[test]
    fn test_openvino_device_defaults_when_unset() {
        let _lock = DEVICE_ENV_LOCK.lock().unwrap();
        let _env = DeviceEnvGuard::set(None);

        assert_eq!(openvino_device_type(), DEFAULT_OPENVINO_DEVICE);
        assert!(!is_explicit_openvino_device(&openvino_device_type()));
    }

    #[test]
    fn test_openvino_device_defaults_when_blank() {
        let _lock = DEVICE_ENV_LOCK.lock().unwrap();
        let _env = DeviceEnvGuard::set(Some("  "));

        assert_eq!(openvino_device_type(), DEFAULT_OPENVINO_DEVICE);
        assert!(!is_explicit_openvino_device(&openvino_device_type()));
    }

    #[test]
    fn test_openvino_device_accepts_explicit_npu() {
        let _lock = DEVICE_ENV_LOCK.lock().unwrap();
        let _env = DeviceEnvGuard::set(Some("NPU"));

        assert_eq!(openvino_device_type(), "NPU");
        assert!(is_explicit_openvino_device(&openvino_device_type()));
    }

    #[test]
    fn test_explicit_provider_registration_failure_is_propagated() {
        let error = provider_registration_error("provider unavailable", true).unwrap_err();

        assert!(matches!(
            error,
            InferenceError::Initialization(message) if message.contains("provider unavailable")
        ));
    }

    #[test]
    fn test_automatic_provider_registration_failure_allows_cpu_fallback() {
        assert!(provider_registration_error("provider unavailable", false).is_ok());
    }
}
