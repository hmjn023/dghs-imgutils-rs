//! ONNX Runtime 推論セッションの作成と、単一入力・単一出力の推論実行を共通化する共通ユーティリティを提供します。

pub mod backend;
pub mod classify;
pub mod error;
pub mod session;
pub mod yolo;

pub use backend::{
    Backend, BackendCapabilities, ModelManifest, ModelProfile, Precision, SessionOptions,
    choose_backend, probe_backend, probe_backends,
};
pub use error::InferenceError;
pub use session::{SessionKey, model_sha256, runtime_fingerprint};
pub use yolo::{
    DetectionResult, postprocess_end2end_yolo, postprocess_nms_yolo, preprocess_image_yolo,
    xy_postprocess, yolo_nms, yolo_predict, yolo_xywh2xyxy,
};

use once_cell::sync::Lazy;
use ort::ep::{ExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use tracing::{info, warn};

/// Shared, worker-local session handle used by all model modules.
pub type SharedSession = Arc<Mutex<Session>>;

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

/// ONNX セッションの worker-local cache.
///
/// The key includes model content, backend, precision, device, runtime, and
/// provider fingerprints, so replacing an ONNX file or runtime cannot reuse a
/// stale compiled session.
static SESSION_CACHE: Lazy<Mutex<HashMap<SessionKey, Arc<Mutex<Session>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Programmatic worker configuration used by the legacy high-level APIs.
///
/// It is intentionally process-wide: ONNX Runtime and its execution provider
/// libraries are loaded per process, so applications should configure this
/// once during startup before invoking a model API.
static PROGRAMMATIC_SESSION_OPTIONS: Lazy<RwLock<Option<SessionOptions>>> =
    Lazy::new(|| RwLock::new(None));

thread_local! {
    /// Per-call override used by API adapters such as the N-API bindings.
    ///
    /// A thread-local scope keeps concurrent UI requests independent while
    /// preserving the existing high-level Rust API signatures.
    static INFERENCE_OPTIONS_SCOPE: RefCell<Vec<SessionOptions>> = const { RefCell::new(Vec::new()) };
}

/// Prevents changing the process-wide worker policy after any session attempt.
///
/// This also covers callers that use the explicit, uncached session APIs, for
/// which the central session cache cannot tell that ONNX Runtime has already
/// been initialized.
static INFERENCE_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Configures the default inference worker for high-level model APIs.
///
/// This takes precedence over environment variables. Configure it before the
/// first inference session is created. Applications that need more than one
/// backend in one process can instead pass options directly to
/// [`create_onnx_session_with_options`] or
/// [`get_or_create_session_with_options`].
pub fn configure_inference(options: SessionOptions) -> Result<(), InferenceError> {
    if INFERENCE_INITIALIZED.load(Ordering::Acquire) {
        return Err(InferenceError::Initialization(
            "inference is already initialized; configure the worker before the first model call"
                .to_owned(),
        ));
    }

    *PROGRAMMATIC_SESSION_OPTIONS.write().map_err(|error| {
        InferenceError::Initialization(format!("Inference options lock poisoned: {error}"))
    })? = Some(options);
    Ok(())
}

/// Removes the programmatic default and restores environment-based selection.
///
/// Like [`configure_inference`], this is intended for startup/test setup and
/// refuses to change a worker after a session has been created.
pub fn reset_inference_configuration() -> Result<(), InferenceError> {
    if INFERENCE_INITIALIZED.load(Ordering::Acquire) {
        return Err(InferenceError::Initialization(
            "inference is already initialized; reset the worker before the first model call"
                .to_owned(),
        ));
    }

    *PROGRAMMATIC_SESSION_OPTIONS.write().map_err(|error| {
        InferenceError::Initialization(format!("Inference options lock poisoned: {error}"))
    })? = None;
    Ok(())
}

/// Returns the effective configuration for high-level APIs.
///
/// A programmatic configuration wins over environment variables; otherwise
/// the legacy environment-based configuration is parsed.
pub fn current_inference_options() -> Result<SessionOptions, InferenceError> {
    if let Some(options) = INFERENCE_OPTIONS_SCOPE.with(|scope| scope.borrow().last().cloned()) {
        return Ok(options);
    }

    if let Some(options) = PROGRAMMATIC_SESSION_OPTIONS
        .read()
        .map_err(|error| {
            InferenceError::Initialization(format!("Inference options lock poisoned: {error}"))
        })?
        .clone()
    {
        return Ok(options);
    }
    SessionOptions::from_env()
}

/// Runs one inference operation with a call-local worker policy.
///
/// The override is visible to all existing high-level model functions that
/// resolve sessions through [`get_or_create_session`]. It is removed even if
/// the operation panics, and it does not modify the process-wide startup
/// configuration.
pub fn with_inference_options<T>(options: SessionOptions, operation: impl FnOnce() -> T) -> T {
    struct ScopeGuard;

    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            INFERENCE_OPTIONS_SCOPE.with(|scope| {
                scope.borrow_mut().pop();
            });
        }
    }

    INFERENCE_OPTIONS_SCOPE.with(|scope| {
        scope.borrow_mut().push(options);
    });
    let _guard = ScopeGuard;
    operation()
}

/// ONNX Runtime のグローバル環境を初期化します。
///
/// `ort` クレートはセッションの初回作成時に自動で初期化を行うため、この関数の呼び出しは通常任意です。
pub fn init_onnx_runtime() -> Result<(), InferenceError> {
    // 将来的な環境のカスタマイズ用のプレースホルダ
    Ok(())
}

/// Creates a session using the configured worker policy.
///
/// A programmatic configuration installed by [`configure_inference`] wins;
/// otherwise `DGHS_*` environment variables are used. `DGHS_ORT_DEVICE`
/// continues to control the legacy automatic OpenVINO policy.
///
/// # 引数
///
/// * `model_path` - ONNX モデルファイルへのパス
pub fn create_onnx_session<P: AsRef<Path>>(model_path: P) -> Result<Session, InferenceError> {
    let options = current_inference_options()?;
    create_onnx_session_with_options(model_path, &options)
}

/// Creates a session with an explicit worker/backend policy.
pub fn create_onnx_session_with_options<P: AsRef<Path>>(
    model_path: P,
    options: &SessionOptions,
) -> Result<Session, InferenceError> {
    let key = SessionKey::from_path(model_path.as_ref(), options)?;
    create_onnx_session_from_key(&key, options)
}

fn create_onnx_session_from_key(
    key: &SessionKey,
    options: &SessionOptions,
) -> Result<Session, InferenceError> {
    INFERENCE_INITIALIZED.store(true, Ordering::Release);

    let mut builder =
        Session::builder().map_err(|e| InferenceError::Initialization(e.to_string()))?;

    match options.backend {
        Backend::Auto => configure_automatic_providers(&mut builder, options)?,
        Backend::Cpu => {}
        Backend::AmdGpu => configure_amd_gpu(&mut builder, options)?,
        Backend::AmdNpu => configure_amd_npu(&mut builder, options, key)?,
        Backend::Cuda => configure_cuda(&mut builder)?,
        Backend::TensorRt => configure_tensorrt(&mut builder)?,
        Backend::DirectMl => configure_directml(&mut builder)?,
        Backend::OpenVino => configure_openvino(&mut builder, options)?,
    }

    builder
        .commit_from_file(&key.model_path)
        .map_err(|error| classify_session_error(options.backend, error.to_string()))
}

/// Creates a session after validating model deployment metadata.
///
/// The manifest hash is optional for development, but when present it is a
/// hard guard against compiling a stale file under the same model path.  For
/// an AMD NPU manifest, `compiler_config` supplies the Vitis configuration
/// only when the worker options did not already specify one.
pub fn create_onnx_session_with_manifest<P: AsRef<Path>>(
    model_path: P,
    options: &SessionOptions,
    manifest: &ModelManifest,
) -> Result<Session, InferenceError> {
    let effective_options = validate_manifest(model_path.as_ref(), options, manifest)?;
    create_onnx_session_with_options(model_path, &effective_options)
}

fn validate_manifest(
    model_path: &Path,
    options: &SessionOptions,
    manifest: &ModelManifest,
) -> Result<SessionOptions, InferenceError> {
    let actual_hash = model_sha256(model_path)?;
    if let Some(expected_hash) = manifest.model_sha256.as_deref()
        && !expected_hash.eq_ignore_ascii_case(&actual_hash)
    {
        return Err(InferenceError::ModelUnsupported(format!(
            "model manifest hash mismatch for {}: expected {}, got {}",
            manifest.name, expected_hash, actual_hash
        )));
    }
    manifest.validate_precision(options.precision)?;

    let mut effective_options = options.clone();
    if effective_options.backend == Backend::AmdNpu && effective_options.vitis_config_file.is_none()
    {
        effective_options.vitis_config_file = manifest.compiler_config.clone();
    }
    Ok(effective_options)
}

fn configure_automatic_providers(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    use ort::ep::{CUDA, DirectML, OpenVINO, TensorRT};

    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

    let device_type = options
        .openvino_device_type
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(openvino_device_type);
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

    if explicit_openvino_device && let Some(provider) = openvino_provider.take() {
        providers.push(provider);
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
    if !explicit_openvino_device && let Some(provider) = openvino_provider {
        providers.push(provider);
    }

    if !providers.is_empty() {
        match builder.clone().with_execution_providers(providers) {
            Ok(b) => *builder = b,
            Err(e) => {
                if explicit_openvino_device {
                    provider_registration_error(&e, true)?;
                } else {
                    warn!(
                        "[ort] Failed to register execution providers, falling back to CPU: {:?}",
                        e
                    );
                }
            }
        }
    }

    Ok(())
}

fn configure_amd_gpu(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    #[cfg(feature = "amd-gpu")]
    {
        let mut provider = ort::ep::MIGraphX::default().with_device_id(options.device_id as i32);
        match options.precision {
            Precision::Auto | Precision::Fp32 => {}
            Precision::Fp16 => provider = provider.with_fp16(true),
            Precision::Bf16 => {
                return Err(InferenceError::ModelUnsupported(
                    "MIGraphX does not expose BF16 through the ort 2.0.0-rc.12 provider API"
                        .to_owned(),
                ));
            }
            Precision::Int8 => {
                let calibration = options
                    .migraphx_int8_calibration_table
                    .as_ref()
                    .ok_or_else(|| {
                        InferenceError::InvalidInput(
                            "DGHS_MIGRAPHX_INT8_CALIBRATION_TABLE is required for MIGraphX INT8"
                                .to_owned(),
                        )
                    })?;
                if !calibration.is_file() {
                    return Err(InferenceError::InvalidInput(format!(
                        "MIGraphX INT8 calibration table does not exist: {}",
                        calibration.display()
                    )));
                }
                provider = provider
                    .with_int8(true)
                    .with_int8_calibration_table(calibration.to_string_lossy(), false);
            }
        }
        provider = provider.with_exhaustive_tune(options.migraphx_exhaustive_tune);
        if !provider.supported_by_platform() {
            return Err(InferenceError::BackendUnavailable(
                "MIGraphX is unsupported on this platform".to_owned(),
            ));
        }
        if !provider
            .is_available()
            .map_err(|error| InferenceError::BackendUnavailable(error.to_string()))?
        {
            return Err(InferenceError::BackendUnavailable(
                "MIGraphXExecutionProvider is not available in the loaded ONNX Runtime".to_owned(),
            ));
        }
        *builder = builder
            .clone()
            .with_execution_providers([provider.build().error_on_failure()])
            .map_err(|error| InferenceError::BackendUnavailable(error.to_string()))?;
        Ok(())
    }
    #[cfg(not(feature = "amd-gpu"))]
    {
        let _ = (builder, options);
        Err(InferenceError::BackendUnavailable(
            "AMD GPU support is disabled; rebuild with --features amd-gpu".to_owned(),
        ))
    }
}

fn configure_amd_npu(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
    key: &SessionKey,
) -> Result<(), InferenceError> {
    #[cfg(feature = "amd-npu")]
    {
        let config = options.vitis_config_file.as_ref().ok_or_else(|| {
            InferenceError::InvalidInput(
                "DGHS_VITIS_CONFIG is required for the AMD NPU worker".to_owned(),
            )
        })?;
        if !config.is_file() {
            return Err(InferenceError::BackendUnavailable(format!(
                "Vitis AI config file does not exist: {}",
                config.display()
            )));
        }
        let cache_dir = options.ep_cache_dir(Backend::AmdNpu);
        std::fs::create_dir_all(&cache_dir)?;
        let provider = ort::ep::Vitis::default()
            .with_config_file(config.to_string_lossy())
            .with_cache_dir(cache_dir.to_string_lossy())
            .with_cache_key(key.vitis_cache_key());
        if !provider.supported_by_platform() {
            return Err(InferenceError::BackendUnavailable(
                "Vitis AI is unsupported on this platform".to_owned(),
            ));
        }
        if !provider
            .is_available()
            .map_err(|error| InferenceError::BackendUnavailable(error.to_string()))?
        {
            return Err(InferenceError::BackendUnavailable(
                "VitisAIExecutionProvider is not available in the loaded ONNX Runtime".to_owned(),
            ));
        }
        *builder = builder
            .clone()
            .with_execution_providers([provider.build().error_on_failure()])
            .map_err(|error| InferenceError::CompilationFailed(error.to_string()))?;
        Ok(())
    }
    #[cfg(not(feature = "amd-npu"))]
    {
        let _ = (builder, options, key);
        Err(InferenceError::BackendUnavailable(
            "AMD NPU support is disabled; rebuild with --features amd-npu".to_owned(),
        ))
    }
}

fn configure_cuda(
    builder: &mut ort::session::builder::SessionBuilder,
) -> Result<(), InferenceError> {
    #[cfg(feature = "cuda")]
    {
        register_strict_provider(builder, ort::ep::CUDA::default(), Backend::Cuda)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = builder;
        Err(InferenceError::BackendUnavailable(
            "CUDA support is disabled; rebuild with --features cuda".to_owned(),
        ))
    }
}

fn configure_tensorrt(
    builder: &mut ort::session::builder::SessionBuilder,
) -> Result<(), InferenceError> {
    #[cfg(feature = "tensorrt")]
    {
        register_strict_provider(builder, ort::ep::TensorRT::default(), Backend::TensorRt)
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        let _ = builder;
        Err(InferenceError::BackendUnavailable(
            "TensorRT support is disabled; rebuild with --features tensorrt".to_owned(),
        ))
    }
}

fn configure_directml(
    builder: &mut ort::session::builder::SessionBuilder,
) -> Result<(), InferenceError> {
    #[cfg(feature = "directml")]
    {
        register_strict_provider(builder, ort::ep::DirectML::default(), Backend::DirectMl)
    }
    #[cfg(not(feature = "directml"))]
    {
        let _ = builder;
        Err(InferenceError::BackendUnavailable(
            "DirectML support is disabled; rebuild with --features directml".to_owned(),
        ))
    }
}

fn configure_openvino(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    #[cfg(feature = "openvino")]
    {
        let device_type = options
            .openvino_device_type
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(openvino_device_type);
        let provider = ort::ep::OpenVINO::default().with_device_type(&device_type);
        register_strict_provider(builder, provider, Backend::OpenVino)
    }
    #[cfg(not(feature = "openvino"))]
    {
        let _ = (builder, options);
        Err(InferenceError::BackendUnavailable(
            "OpenVINO support is disabled; rebuild with --features openvino".to_owned(),
        ))
    }
}

#[allow(dead_code)]
fn register_strict_provider<E: ExecutionProvider + Into<ExecutionProviderDispatch> + 'static>(
    builder: &mut ort::session::builder::SessionBuilder,
    provider: E,
    backend: Backend,
) -> Result<(), InferenceError> {
    if !provider.supported_by_platform() {
        return Err(InferenceError::BackendUnavailable(format!(
            "{} is unsupported on this platform",
            backend.provider_name().unwrap_or("provider")
        )));
    }
    if !provider
        .is_available()
        .map_err(|error| InferenceError::BackendUnavailable(error.to_string()))?
    {
        return Err(InferenceError::BackendUnavailable(format!(
            "{} is not available in the loaded ONNX Runtime",
            backend.provider_name().unwrap_or("provider")
        )));
    }
    *builder = builder
        .clone()
        .with_execution_providers([provider.into().error_on_failure()])
        .map_err(|error| InferenceError::BackendUnavailable(error.to_string()))?;
    Ok(())
}

fn classify_session_error(backend: Backend, message: String) -> InferenceError {
    let lower = message.to_ascii_lowercase();
    if lower.contains("out of memory") || lower.contains("cuda error 2") {
        InferenceError::OutOfMemory(message)
    } else if lower.contains("unsupported")
        || lower.contains("not implemented")
        || lower.contains("no op")
    {
        InferenceError::ModelUnsupported(message)
    } else if matches!(backend, Backend::AmdGpu | Backend::AmdNpu) {
        InferenceError::CompilationFailed(message)
    } else {
        InferenceError::Initialization(message)
    }
}

/// キャッシュ付きで ONNX セッションを取得する。
///
/// モデルとworker/runtime identityが同じ場合はキャッシュからセッションを返す。
/// セッションは `Arc<Mutex<Session>>` で共有され、スレッドセーフに利用できる。
///
/// # 引数
///
/// * `model_path` - ONNX モデルファイルへのパス
pub fn get_or_create_session<P: AsRef<Path>>(
    model_path: P,
) -> Result<Arc<Mutex<Session>>, InferenceError> {
    let options = current_inference_options()?;
    get_or_create_session_with_options(model_path, &options)
}

/// Returns a cached session using an explicit worker/backend policy.
pub fn get_or_create_session_with_options<P: AsRef<Path>>(
    model_path: P,
    options: &SessionOptions,
) -> Result<Arc<Mutex<Session>>, InferenceError> {
    let key = SessionKey::from_path(model_path.as_ref(), options)?;

    let mut cache = SESSION_CACHE
        .lock()
        .map_err(|e| InferenceError::Initialization(format!("Session cache lock poisoned: {e}")))?;

    if let Some(session) = cache.get(&key) {
        return Ok(Arc::clone(session));
    }

    // A model path can be replaced in place by a downloader or deployment
    // tool. Drop stale entries for that path after the content hash changes;
    // callers holding an old Arc may finish using it, but new lookups cannot
    // retain an obsolete compiled graph indefinitely.
    if cache.keys().any(|cached| {
        cached.model_path == key.model_path && cached.model_sha256 != key.model_sha256
    }) {
        cache.retain(|cached, _| {
            cached.model_path != key.model_path || cached.model_sha256 == key.model_sha256
        });
    }

    let session = create_onnx_session_from_key(&key, options)?;
    let session = Arc::new(Mutex::new(session));
    cache.insert(key, Arc::clone(&session));
    Ok(session)
}

/// Returns a cached session after validating model deployment metadata.
pub fn get_or_create_session_with_manifest<P: AsRef<Path>>(
    model_path: P,
    options: &SessionOptions,
    manifest: &ModelManifest,
) -> Result<SharedSession, InferenceError> {
    let effective_options = validate_manifest(model_path.as_ref(), options, manifest)?;
    get_or_create_session_with_options(model_path, &effective_options)
}

/// Clears all worker-local cached sessions.
pub fn clear_session_cache() -> Result<(), InferenceError> {
    SESSION_CACHE
        .lock()
        .map_err(|e| InferenceError::Initialization(format!("Session cache lock poisoned: {e}")))?
        .clear();
    Ok(())
}

/// Locks a shared session and normalizes poisoned-lock errors.
pub fn lock_session(
    session: &SharedSession,
) -> Result<std::sync::MutexGuard<'_, Session>, InferenceError> {
    session
        .lock()
        .map_err(|error| InferenceError::Initialization(format!("Session lock poisoned: {error}")))
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
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::NamedTempFile;

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

    #[test]
    fn manifest_hash_mismatch_is_rejected_before_session_creation() {
        let mut model = NamedTempFile::new().unwrap();
        model.write_all(b"model").unwrap();
        let manifest = ModelManifest {
            name: "test".to_owned(),
            model_sha256: Some("wrong".to_owned()),
            opset: Some(17),
            preferred_backends: vec![Backend::Cpu],
            default_precision: Precision::Fp32,
            supported_precisions: vec![Precision::Fp32],
            static_shape: true,
            compiler_config: None,
        };

        let error =
            validate_manifest(model.path(), &SessionOptions::default(), &manifest).unwrap_err();
        assert!(
            matches!(error, InferenceError::ModelUnsupported(message) if message.contains("hash mismatch"))
        );
    }

    #[test]
    fn programmatic_configuration_is_used_by_high_level_session_apis() {
        reset_inference_configuration().unwrap();
        let options = SessionOptions::for_backend(Backend::AmdGpu)
            .with_precision(Precision::Fp16)
            .with_device_id(3);
        configure_inference(options).unwrap();

        let effective = current_inference_options().unwrap();
        assert_eq!(effective.backend, Backend::AmdGpu);
        assert_eq!(effective.precision, Precision::Fp16);
        assert_eq!(effective.device_id, 3);

        reset_inference_configuration().unwrap();
    }

    #[test]
    fn call_local_configuration_is_scoped_and_nestable() {
        let outer = SessionOptions::for_backend(Backend::Cpu)
            .with_precision(Precision::Fp32)
            .with_device_id(1);
        let inner = SessionOptions::for_backend(Backend::AmdGpu)
            .with_precision(Precision::Fp16)
            .with_device_id(2);

        with_inference_options(outer.clone(), || {
            assert_eq!(current_inference_options().unwrap(), outer);

            with_inference_options(inner.clone(), || {
                assert_eq!(current_inference_options().unwrap(), inner);
            });

            assert_eq!(current_inference_options().unwrap(), outer);
        });
    }
}
