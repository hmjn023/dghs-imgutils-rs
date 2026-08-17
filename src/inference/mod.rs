//! ONNX Runtime 推論セッションの作成と、単一入力・単一出力の推論実行を共通化する共通ユーティリティを提供します。

pub mod backend;
pub mod classify;
pub mod error;
mod runtime;
pub mod session;
pub mod yolo;

pub use backend::{
    Backend, BackendCapabilities, DeviceProvider, DeviceSelection, ModelManifest, ModelProfile,
    Precision, SessionOptions, choose_backend, probe_backend, probe_backend_with_options,
    probe_backends, probe_backends_with_options,
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
use tracing::{debug, info};

/// Shared, worker-local session handle used by all model modules.
pub type SharedSession = Arc<Mutex<Session>>;

const OPENVINO_DEVICE_ENV: &str = "DGHS_ORT_DEVICE";

/// Returns the effective OpenVINO device policy for the current process.
///
/// Empty values are treated the same as an unset `DGHS_ORT_DEVICE`.
pub fn openvino_device_type() -> String {
    env::var(OPENVINO_DEVICE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(runtime::default_openvino_device)
}

#[cfg(any(feature = "openvino", test))]
fn is_explicit_openvino_device(device_type: &str) -> bool {
    let device_type = device_type.trim().to_ascii_uppercase();
    matches!(device_type.as_str(), "CPU" | "GPU" | "NPU")
        || device_type.starts_with("CPU.")
        || device_type.starts_with("GPU.")
        || device_type.starts_with("NPU.")
        || device_type.starts_with("HETERO:")
        || device_type.starts_with("MULTI:")
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
    runtime::prepare();
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
    init_onnx_runtime()?;
    let key = SessionKey::from_path(model_path.as_ref(), options)?;
    create_onnx_session_from_key(&key, options)
}

fn create_onnx_session_from_key(
    key: &SessionKey,
    options: &SessionOptions,
) -> Result<Session, InferenceError> {
    init_onnx_runtime()?;
    INFERENCE_INITIALIZED.store(true, Ordering::Release);

    match options.backend {
        Backend::Auto => {
            validate_provider_precision(Backend::Auto, options.precision)?;
            create_automatic_session(key, options)
        }
        _ => create_session_for_backend(key, options),
    }
}

fn create_automatic_session(
    key: &SessionKey,
    options: &SessionOptions,
) -> Result<Session, InferenceError> {
    let backends = automatic_backend_order(options);
    let mut last_error = None;
    let mut first_non_cpu_error = None;

    for backend in backends {
        let candidate_options = options.clone().with_backend(backend);
        // Keep the public cache identity tied to the caller's `auto` request,
        // but give provider-specific compiler/cache setup the concrete
        // candidate identity.
        let candidate_key = SessionKey {
            model_path: key.model_path.clone(),
            model_sha256: key.model_sha256.clone(),
            backend,
            precision: candidate_options.precision,
            device_id: candidate_options.device_id,
            runtime_fingerprint: key.runtime_fingerprint.clone(),
            provider_fingerprint: candidate_options.provider_fingerprint(),
        };
        match create_session_for_backend(&candidate_key, &candidate_options) {
            Ok(session) => {
                info!(backend = %backend, "[ort] automatic backend selected");
                return Ok(session);
            }
            Err(error) => {
                debug!(
                    backend = %backend,
                    error = %error,
                    "[ort] automatic backend unavailable"
                );
                if backend != Backend::Cpu && first_non_cpu_error.is_none() {
                    first_non_cpu_error = Some(error);
                } else {
                    last_error = Some(error);
                }
            }
        }
    }

    Err(first_non_cpu_error.or(last_error).unwrap_or_else(|| {
        InferenceError::Initialization("no automatic inference backend is available".to_owned())
    }))
}

fn automatic_backend_order(_options: &SessionOptions) -> Vec<Backend> {
    #[cfg(feature = "openvino")]
    let device_type = _options
        .openvino_device_type
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(openvino_device_type);

    // An explicit OpenVINO device is a strict request even when the public
    // backend remains `auto`, preserving the DGHS_ORT_DEVICE contract.
    #[cfg(feature = "openvino")]
    if is_explicit_openvino_device(&device_type) {
        return vec![Backend::OpenVino];
    }

    let mut backends = Vec::with_capacity(6);
    #[cfg(feature = "tensorrt")]
    backends.push(Backend::TensorRt);
    #[cfg(feature = "cuda")]
    backends.push(Backend::Cuda);
    #[cfg(feature = "directml")]
    backends.push(Backend::DirectMl);
    #[cfg(feature = "openvino")]
    backends.push(Backend::OpenVino);
    #[cfg(feature = "amd-gpu")]
    backends.push(Backend::AmdGpu);
    #[cfg(feature = "amd-npu")]
    if _options.vitis_config_file.is_some() {
        backends.push(Backend::AmdNpu);
    }
    backends.push(Backend::Cpu);
    backends
}

fn create_session_for_backend(
    key: &SessionKey,
    options: &SessionOptions,
) -> Result<Session, InferenceError> {
    let mut builder =
        Session::builder().map_err(|e| InferenceError::Initialization(e.to_string()))?;
    configure_session_builder(&mut builder, key, options)?;

    builder
        .commit_from_file(&key.model_path)
        .map_err(|error| classify_session_error(options.backend, error.to_string()))
}

fn configure_session_builder(
    builder: &mut ort::session::builder::SessionBuilder,
    key: &SessionKey,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    match options.backend {
        Backend::Auto => Err(InferenceError::InvalidInput(
            "automatic backend must be resolved before configuring a session".to_owned(),
        )),
        Backend::Cpu => validate_provider_precision(Backend::Cpu, options.precision),
        Backend::AmdGpu => configure_amd_gpu(builder, options),
        Backend::AmdNpu => configure_amd_npu(builder, options, key),
        Backend::Cuda => configure_cuda(builder, options),
        Backend::TensorRt => configure_tensorrt(builder, options),
        Backend::DirectMl => configure_directml(builder, options),
        Backend::OpenVino => configure_openvino(builder, options),
    }
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

fn validate_provider_precision(
    backend: Backend,
    precision: Precision,
) -> Result<(), InferenceError> {
    if matches!(precision, Precision::Auto | Precision::Fp32) {
        return Ok(());
    }

    Err(InferenceError::ModelUnsupported(format!(
        "{} does not accept an explicit {} precision request; use a model exported for that precision or Precision::Auto",
        backend, precision
    )))
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
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    #[cfg(feature = "cuda")]
    {
        validate_provider_precision(Backend::Cuda, options.precision)?;
        let provider = ort::ep::CUDA::default().with_device_id(options.device_id as i32);
        register_strict_provider(builder, provider, Backend::Cuda)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (builder, options);
        Err(InferenceError::BackendUnavailable(
            "CUDA support is disabled; rebuild with --features cuda".to_owned(),
        ))
    }
}

fn configure_tensorrt(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    #[cfg(feature = "tensorrt")]
    {
        validate_provider_precision(Backend::TensorRt, options.precision)?;
        let provider = ort::ep::TensorRT::default().with_device_id(options.device_id as i32);
        register_strict_provider(builder, provider, Backend::TensorRt)
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        let _ = (builder, options);
        Err(InferenceError::BackendUnavailable(
            "TensorRT support is disabled; rebuild with --features tensorrt".to_owned(),
        ))
    }
}

fn configure_directml(
    builder: &mut ort::session::builder::SessionBuilder,
    options: &SessionOptions,
) -> Result<(), InferenceError> {
    #[cfg(feature = "directml")]
    {
        validate_provider_precision(Backend::DirectMl, options.precision)?;
        let provider = ort::ep::DirectML::default().with_device_id(options.device_id as i32);
        register_strict_provider(builder, provider, Backend::DirectMl)
    }
    #[cfg(not(feature = "directml"))]
    {
        let _ = (builder, options);
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
        validate_provider_precision(Backend::OpenVino, options.precision)?;
        let device_type = options
            .openvino_device_type
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(openvino_device_type);
        runtime::validate_openvino_device_policy(&device_type).map_err(|reason| {
            InferenceError::BackendUnavailable(format!(
                "OpenVINO device policy {device_type} is unavailable: {reason}"
            ))
        })?;
        runtime::prepare_openvino_device(&device_type);
        let provider = ort::ep::OpenVINO::default().with_device_type(&device_type);
        let detected_devices = runtime::detected_intel_device_summary();
        register_strict_provider(builder, provider, Backend::OpenVino).map_err(|error| {
            InferenceError::BackendUnavailable(format!(
                "OpenVINO device policy {device_type} is unavailable (detected Intel devices: {detected_devices}): {error}"
            ))
        })
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
    init_onnx_runtime()?;
    let key = SessionKey::from_path(model_path.as_ref(), options)?;

    {
        let mut cache = SESSION_CACHE.lock().map_err(|e| {
            InferenceError::Initialization(format!("Session cache lock poisoned: {e}"))
        })?;

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
    }

    // Compilation can take seconds or minutes for provider-backed sessions.
    // Do not hold the global cache mutex while ONNX Runtime compiles the graph.
    let created_session = Arc::new(Mutex::new(create_onnx_session_from_key(&key, options)?));

    let mut cache = SESSION_CACHE
        .lock()
        .map_err(|e| InferenceError::Initialization(format!("Session cache lock poisoned: {e}")))?;
    if let Some(session) = cache.get(&key) {
        // Another thread may have compiled and inserted the same key while this
        // thread was compiling. Reuse the canonical entry in that case.
        return Ok(Arc::clone(session));
    }
    cache.insert(key, Arc::clone(&created_session));
    Ok(created_session)
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
            // Tests serialize process-wide environment access with
            // DEVICE_ENV_LOCK, making this scoped mutation safe here.
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
            // Restore the value while DEVICE_ENV_LOCK is still held by the
            // owning test, so no concurrent test observes a partial change.
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

        assert_eq!(openvino_device_type(), runtime::default_openvino_device());
        assert!(!is_explicit_openvino_device(&openvino_device_type()));
    }

    #[test]
    fn test_openvino_device_defaults_when_blank() {
        let _lock = DEVICE_ENV_LOCK.lock().unwrap();
        let _env = DeviceEnvGuard::set(Some("  "));

        assert_eq!(openvino_device_type(), runtime::default_openvino_device());
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
    fn non_migraphx_backends_reject_unapplied_precision_requests() {
        assert!(validate_provider_precision(Backend::Cuda, Precision::Fp32).is_ok());
        assert!(validate_provider_precision(Backend::Cpu, Precision::Auto).is_ok());
        assert!(matches!(
            validate_provider_precision(Backend::Cuda, Precision::Fp16),
            Err(InferenceError::ModelUnsupported(message)) if message.contains("fp16")
        ));
    }

    #[test]
    fn automatic_backend_order_ends_with_cpu() {
        let _lock = DEVICE_ENV_LOCK.lock().unwrap();
        let _env = DeviceEnvGuard::set(None);

        let order = automatic_backend_order(&SessionOptions::default());

        assert_eq!(order.last(), Some(&Backend::Cpu));
        assert!(!order.is_empty());
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn explicit_openvino_device_keeps_auto_selection_strict() {
        let options = SessionOptions::default().with_openvino_device_type("GPU.0");

        assert_eq!(automatic_backend_order(&options), vec![Backend::OpenVino]);
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
