//! Execution-provider selection and model execution policy.

use crate::inference::InferenceError;
use crate::utils::storage::get_storage_dir;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

const BACKEND_ENV: &str = "DGHS_BACKEND";
const PRECISION_ENV: &str = "DGHS_PRECISION";
const DEVICE_ID_ENV: &str = "DGHS_DEVICE_ID";
const OPENVINO_DEVICE_ENV: &str = "DGHS_ORT_DEVICE";
const VITIS_CONFIG_ENV: &str = "DGHS_VITIS_CONFIG";
const EP_CACHE_DIR_ENV: &str = "DGHS_EP_CACHE_DIR";
const MIGRAPHX_INT8_CALIBRATION_ENV: &str = "DGHS_MIGRAPHX_INT8_CALIBRATION_TABLE";
const MIGRAPHX_EXHAUSTIVE_TUNE_ENV: &str = "DGHS_MIGRAPHX_EXHAUSTIVE_TUNE";

/// A logical execution backend selected for an inference worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// Preserve the legacy automatic provider policy.
    Auto,
    Cpu,
    AmdGpu,
    AmdNpu,
    Cuda,
    TensorRt,
    DirectMl,
    OpenVino,
}

impl Backend {
    /// Returns the ONNX Runtime execution-provider name, when one exists.
    pub const fn provider_name(self) -> Option<&'static str> {
        match self {
            Self::Auto | Self::Cpu => None,
            Self::AmdGpu => Some("MIGraphXExecutionProvider"),
            Self::AmdNpu => Some("VitisAIExecutionProvider"),
            Self::Cuda => Some("CUDAExecutionProvider"),
            Self::TensorRt => Some("TensorrtExecutionProvider"),
            Self::DirectMl => Some("DmlExecutionProvider"),
            Self::OpenVino => Some("OpenVINOExecutionProvider"),
        }
    }

    /// Returns the stable environment/configuration spelling for this backend.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::AmdGpu => "amd-gpu",
            Self::AmdNpu => "amd-npu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
            Self::DirectMl => "directml",
            Self::OpenVino => "openvino",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "amd-gpu" | "amd_gpu" | "migraphx" => Ok(Self::AmdGpu),
            "amd-npu" | "amd_npu" | "vitis" | "vitis-ai" => Ok(Self::AmdNpu),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" | "tensor-rt" => Ok(Self::TensorRt),
            "directml" | "direct-ml" => Ok(Self::DirectMl),
            "openvino" | "open-vino" => Ok(Self::OpenVino),
            _ => Err(format!("unsupported backend: {value}")),
        }
    }
}

/// A provider precision request. Backend-specific validation happens while a
/// session is being built; this enum is not a promise that every model can use
/// the requested representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    #[default]
    Auto,
    Fp32,
    Fp16,
    Bf16,
    Int8,
}

impl Precision {
    /// Returns the stable configuration string used in manifests and cache keys.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Precision {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fp32" | "float32" => Ok(Self::Fp32),
            "fp16" | "float16" => Ok(Self::Fp16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "int8" => Ok(Self::Int8),
            _ => Err(format!("unsupported precision: {value}")),
        }
    }
}

/// Model deployment metadata used by a worker or scheduler.
///
/// This is deliberately separate from the ONNX graph itself.  A model that
/// was exported with a different opset, quantization recipe, or compiler
/// configuration must not silently reuse a session compiled for another
/// deployment.  Store this structure next to the model and validate it before
/// creating a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    #[serde(default)]
    pub model_sha256: Option<String>,
    #[serde(default)]
    pub opset: Option<u32>,
    #[serde(default)]
    pub preferred_backends: Vec<Backend>,
    #[serde(default)]
    pub default_precision: Precision,
    #[serde(default)]
    pub supported_precisions: Vec<Precision>,
    #[serde(default)]
    pub static_shape: bool,
    #[serde(default)]
    pub compiler_config: Option<PathBuf>,
}

impl ModelManifest {
    /// Creates the scheduler-facing profile represented by this manifest.
    pub fn profile(&self) -> ModelProfile {
        ModelProfile {
            name: self.name.clone(),
            preferred_backends: self.preferred_backends.clone(),
            default_precision: self.default_precision,
            static_shape: self.static_shape,
        }
    }

    /// Validates a requested precision against the model's deployment matrix.
    pub fn validate_precision(&self, requested: Precision) -> Result<(), InferenceError> {
        if requested != Precision::Auto
            && !self.supported_precisions.is_empty()
            && !self.supported_precisions.contains(&requested)
        {
            return Err(InferenceError::ModelUnsupported(format!(
                "model {} has no {} deployment",
                self.name, requested
            )));
        }
        Ok(())
    }
}

/// Per-session options that are safe to pass to a worker-local runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionOptions {
    pub backend: Backend,
    pub precision: Precision,
    pub device_id: usize,
    pub openvino_device_type: Option<String>,
    pub vitis_config_file: Option<PathBuf>,
    pub ep_cache_dir: Option<PathBuf>,
    pub migraphx_int8_calibration_table: Option<PathBuf>,
    pub migraphx_exhaustive_tune: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            precision: Precision::Auto,
            device_id: 0,
            openvino_device_type: None,
            vitis_config_file: None,
            ep_cache_dir: None,
            migraphx_int8_calibration_table: None,
            migraphx_exhaustive_tune: false,
        }
    }
}

impl SessionOptions {
    /// Creates a programmatic configuration for one worker/backend.
    ///
    /// The returned value does not inspect the process environment. Use
    /// [`SessionOptions::from_env`] when environment compatibility is desired.
    pub fn for_backend(backend: Backend) -> Self {
        Self {
            backend,
            ..Self::default()
        }
    }

    /// Changes the backend in a programmatic configuration.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Selects the execution precision for a programmatic configuration.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Selects the device ordinal used by providers that expose one.
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Selects an OpenVINO device string such as `CPU`, `GPU`, or `NPU`.
    pub fn with_openvino_device_type(mut self, device_type: impl Into<String>) -> Self {
        self.openvino_device_type = Some(device_type.into());
        self
    }

    /// Sets the Vitis-AI provider configuration file.
    pub fn with_vitis_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.vitis_config_file = Some(path.into());
        self
    }

    /// Sets the worker-local execution-provider cache directory.
    pub fn with_ep_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.ep_cache_dir = Some(path.into());
        self
    }

    /// Sets the MIGraphX INT8 calibration table.
    pub fn with_migraphx_int8_calibration_table(mut self, path: impl Into<PathBuf>) -> Self {
        self.migraphx_int8_calibration_table = Some(path.into());
        self
    }

    /// Enables or disables MIGraphX exhaustive tuning.
    pub fn with_migraphx_exhaustive_tune(mut self, enabled: bool) -> Self {
        self.migraphx_exhaustive_tune = enabled;
        self
    }

    /// Reads worker configuration from environment variables.
    pub fn from_env() -> Result<Self, InferenceError> {
        let backend = optional_env(BACKEND_ENV)
            .map(|value| value.parse::<Backend>())
            .transpose()
            .map_err(InferenceError::InvalidInput)?
            .unwrap_or(Backend::Auto);
        let precision = optional_env(PRECISION_ENV)
            .map(|value| value.parse::<Precision>())
            .transpose()
            .map_err(InferenceError::InvalidInput)?
            .unwrap_or(Precision::Auto);
        let device_id = optional_env(DEVICE_ID_ENV)
            .map(|value| {
                value.parse::<usize>().map_err(|error| {
                    InferenceError::InvalidInput(format!(
                        "{DEVICE_ID_ENV} must be an integer: {error}"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(0);

        Ok(Self {
            backend,
            precision,
            device_id,
            openvino_device_type: optional_env(OPENVINO_DEVICE_ENV),
            vitis_config_file: optional_env(VITIS_CONFIG_ENV).map(PathBuf::from),
            ep_cache_dir: optional_env(EP_CACHE_DIR_ENV).map(PathBuf::from),
            migraphx_int8_calibration_table: optional_env(MIGRAPHX_INT8_CALIBRATION_ENV)
                .map(PathBuf::from),
            migraphx_exhaustive_tune: optional_env(MIGRAPHX_EXHAUSTIVE_TUNE_ENV)
                .map(parse_bool)
                .transpose()?
                .unwrap_or(false),
        })
    }

    /// Returns the worker-local EP cache directory.
    pub fn ep_cache_dir(&self, backend: Backend) -> PathBuf {
        self.ep_cache_dir.clone().unwrap_or_else(|| {
            get_storage_dir()
                .join("execution-providers")
                .join(backend.as_str())
        })
    }

    /// Returns a stable provider-options component for session identity.
    pub fn provider_fingerprint(&self) -> String {
        format!(
            "backend={};precision={};device={};openvino_device={};vitis_config={};ep_cache={};migraphx_calibration={};provider_library_path={};tune={}",
            self.backend,
            self.precision,
            self.device_id,
            self.openvino_device_type.as_deref().unwrap_or_default(),
            file_fingerprint(self.vitis_config_file.as_deref()),
            path_identity(self.ep_cache_dir.as_deref()),
            file_fingerprint(self.migraphx_int8_calibration_table.as_deref()),
            optional_env("ORT_PROVIDER_LIBRARY_PATH").unwrap_or_default(),
            self.migraphx_exhaustive_tune,
        )
    }
}

fn path_identity(path: Option<&Path>) -> String {
    let Some(path) = path else {
        return String::new();
    };
    path.to_string_lossy().into_owned()
}

fn file_fingerprint(path: Option<&Path>) -> String {
    let Some(path) = path else {
        return String::new();
    };
    let metadata = fs::metadata(path)
        .ok()
        .map(|metadata| {
            format!(
                "size={};modified={:?}",
                metadata.len(),
                metadata.modified().ok()
            )
        })
        .unwrap_or_else(|| "metadata=unavailable".to_owned());
    format!("{};{metadata}", path.to_string_lossy())
}

/// Known runtime/provider availability for one worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: Backend,
    pub provider_name: Option<String>,
    pub available: bool,
    pub supports_fp16: Option<bool>,
    pub supports_bf16: Option<bool>,
    pub supports_int8: Option<bool>,
    pub supports_dynamic_shapes: Option<bool>,
    pub reason: Option<String>,
}

impl BackendCapabilities {
    /// Returns whether this probe can accept the requested precision.
    ///
    /// `None` means that the provider did not publish a static capability in
    /// this build, so the answer remains optimistic and must be confirmed by
    /// model compilation.
    pub fn supports_precision(&self, precision: Precision) -> bool {
        match precision {
            Precision::Auto => true,
            Precision::Fp16 => self.supports_fp16.unwrap_or(true),
            Precision::Bf16 => self.supports_bf16.unwrap_or(true),
            Precision::Int8 => self.supports_int8.unwrap_or(true),
            Precision::Fp32 => true,
        }
    }

    /// Returns whether a model profile can be routed to this worker.
    pub fn supports_profile(&self, profile: &ModelProfile) -> bool {
        self.available
            && self.supports_precision(profile.default_precision)
            && (profile.static_shape || self.supports_dynamic_shapes != Some(false))
    }
}

/// A model-level preference used by a scheduler or application integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub name: String,
    pub preferred_backends: Vec<Backend>,
    pub default_precision: Precision,
    pub static_shape: bool,
}

impl ModelProfile {
    /// Creates a scheduler-facing profile with backend and precision preferences.
    pub fn new(
        name: impl Into<String>,
        preferred_backends: impl Into<Vec<Backend>>,
        default_precision: Precision,
        static_shape: bool,
    ) -> Self {
        Self {
            name: name.into(),
            preferred_backends: preferred_backends.into(),
            default_precision,
            static_shape,
        }
    }
}

/// Chooses the first compatible backend from a model profile.
pub fn choose_backend(
    profile: &ModelProfile,
    capabilities: &[BackendCapabilities],
) -> Option<Backend> {
    profile.preferred_backends.iter().copied().find(|backend| {
        capabilities.iter().any(|capability| {
            capability.backend == *backend && capability.supports_profile(profile)
        })
    })
}

/// Probes whether the worker's loaded ONNX Runtime exposes one backend.
///
/// `available` means that the provider is present in the loaded runtime and
/// can be registered. It does not guarantee that every model or operator is
/// supported; that still requires a session-creation smoke test.
pub fn probe_backend(backend: Backend) -> BackendCapabilities {
    match backend {
        Backend::Auto => BackendCapabilities {
            backend,
            provider_name: None,
            available: true,
            supports_fp16: None,
            supports_bf16: None,
            supports_int8: None,
            supports_dynamic_shapes: None,
            reason: Some("automatic policy; probe concrete providers instead".to_owned()),
        },
        Backend::Cpu => BackendCapabilities {
            backend,
            provider_name: Some("CPUExecutionProvider".to_owned()),
            available: true,
            supports_fp16: Some(true),
            supports_bf16: Some(true),
            supports_int8: Some(true),
            supports_dynamic_shapes: Some(true),
            reason: None,
        },
        Backend::AmdGpu => probe_amd_gpu(),
        Backend::AmdNpu => probe_amd_npu(),
        Backend::Cuda => probe_cuda(),
        Backend::TensorRt => probe_tensorrt(),
        Backend::DirectMl => probe_directml(),
        Backend::OpenVino => probe_openvino(),
    }
}

/// Probes all concrete providers known to this build.
pub fn probe_backends() -> Vec<BackendCapabilities> {
    [
        Backend::Cpu,
        Backend::AmdGpu,
        Backend::AmdNpu,
        Backend::Cuda,
        Backend::TensorRt,
        Backend::DirectMl,
        Backend::OpenVino,
    ]
    .into_iter()
    .map(probe_backend)
    .collect()
}

fn unavailable(backend: Backend, reason: impl Into<String>) -> BackendCapabilities {
    BackendCapabilities {
        backend,
        provider_name: backend.provider_name().map(str::to_owned),
        available: false,
        supports_fp16: None,
        supports_bf16: None,
        supports_int8: None,
        supports_dynamic_shapes: None,
        reason: Some(reason.into()),
    }
}

#[cfg(feature = "amd-gpu")]
fn probe_amd_gpu() -> BackendCapabilities {
    use ort::ep::ExecutionProvider;

    let provider = ort::ep::MIGraphX::default();
    match provider.is_available() {
        Ok(available) if available && provider.supported_by_platform() => BackendCapabilities {
            backend: Backend::AmdGpu,
            provider_name: Some(provider.name().to_owned()),
            available: true,
            supports_fp16: Some(true),
            supports_bf16: None,
            supports_int8: Some(true),
            supports_dynamic_shapes: None,
            reason: None,
        },
        Ok(false) => unavailable(Backend::AmdGpu, "MIGraphX provider is not available"),
        Ok(true) => unavailable(Backend::AmdGpu, "MIGraphX is unsupported on this platform"),
        Err(error) => unavailable(Backend::AmdGpu, error.to_string()),
    }
}

#[cfg(not(feature = "amd-gpu"))]
fn probe_amd_gpu() -> BackendCapabilities {
    unavailable(Backend::AmdGpu, "build without the amd-gpu feature")
}

#[cfg(feature = "amd-npu")]
fn probe_amd_npu() -> BackendCapabilities {
    use ort::ep::ExecutionProvider;

    let provider = ort::ep::Vitis::default();
    match provider.is_available() {
        Ok(available) if available && provider.supported_by_platform() => BackendCapabilities {
            backend: Backend::AmdNpu,
            provider_name: Some(provider.name().to_owned()),
            available: true,
            supports_fp16: None,
            supports_bf16: Some(true),
            supports_int8: Some(true),
            supports_dynamic_shapes: Some(false),
            reason: None,
        },
        Ok(false) => unavailable(Backend::AmdNpu, "Vitis AI provider is not available"),
        Ok(true) => unavailable(Backend::AmdNpu, "Vitis AI is unsupported on this platform"),
        Err(error) => unavailable(Backend::AmdNpu, error.to_string()),
    }
}

#[cfg(not(feature = "amd-npu"))]
fn probe_amd_npu() -> BackendCapabilities {
    unavailable(Backend::AmdNpu, "build without the amd-npu feature")
}

fn probe_cuda() -> BackendCapabilities {
    #[cfg(feature = "cuda")]
    {
        return probe_ort_provider(Backend::Cuda, ort::ep::CUDA::default());
    }
    #[cfg(not(feature = "cuda"))]
    unavailable(Backend::Cuda, "build without the cuda feature")
}

fn probe_tensorrt() -> BackendCapabilities {
    #[cfg(feature = "tensorrt")]
    {
        return probe_ort_provider(Backend::TensorRt, ort::ep::TensorRT::default());
    }
    #[cfg(not(feature = "tensorrt"))]
    unavailable(Backend::TensorRt, "build without the tensorrt feature")
}

fn probe_directml() -> BackendCapabilities {
    #[cfg(feature = "directml")]
    {
        return probe_ort_provider(Backend::DirectMl, ort::ep::DirectML::default());
    }
    #[cfg(not(feature = "directml"))]
    unavailable(Backend::DirectMl, "build without the directml feature")
}

fn probe_openvino() -> BackendCapabilities {
    #[cfg(feature = "openvino")]
    {
        return probe_ort_provider(Backend::OpenVino, ort::ep::OpenVINO::default());
    }
    #[cfg(not(feature = "openvino"))]
    unavailable(Backend::OpenVino, "build without the openvino feature")
}

#[allow(dead_code)]
fn probe_ort_provider<E: ort::ep::ExecutionProvider>(
    backend: Backend,
    provider: E,
) -> BackendCapabilities {
    match provider.is_available() {
        Ok(available) if available && provider.supported_by_platform() => BackendCapabilities {
            backend,
            provider_name: Some(provider.name().to_owned()),
            available: true,
            supports_fp16: None,
            supports_bf16: None,
            supports_int8: None,
            supports_dynamic_shapes: None,
            reason: None,
        },
        Ok(false) => unavailable(backend, "provider is not available"),
        Ok(true) => unavailable(backend, "provider is unsupported on this platform"),
        Err(error) => unavailable(backend, error.to_string()),
    }
}

fn optional_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn parse_bool(value: String) -> Result<bool, InferenceError> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(InferenceError::InvalidInput(format!(
            "expected a boolean value, got {value}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct EnvGuard {
        previous: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            let names = [
                BACKEND_ENV,
                PRECISION_ENV,
                DEVICE_ID_ENV,
                OPENVINO_DEVICE_ENV,
                VITIS_CONFIG_ENV,
                EP_CACHE_DIR_ENV,
                MIGRAPHX_INT8_CALIBRATION_ENV,
                MIGRAPHX_EXHAUSTIVE_TUNE_ENV,
            ];
            Self {
                previous: names
                    .into_iter()
                    .map(|name| (name, std::env::var_os(name)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                // Tests serialize environment access with ENV_LOCK, so these
                // process-wide mutations cannot race another test in this module.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    #[test]
    fn backend_aliases_are_stable() {
        assert_eq!("migraphx".parse::<Backend>(), Ok(Backend::AmdGpu));
        assert_eq!("vitis-ai".parse::<Backend>(), Ok(Backend::AmdNpu));
        assert_eq!(Backend::AmdGpu.to_string(), "amd-gpu");
    }

    #[test]
    fn precision_aliases_are_stable() {
        assert_eq!("float16".parse::<Precision>(), Ok(Precision::Fp16));
        assert_eq!("bfloat16".parse::<Precision>(), Ok(Precision::Bf16));
    }

    #[test]
    fn session_options_read_worker_environment() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::capture();
        // Tests serialize environment access with ENV_LOCK before mutating the
        // process-wide variables used by SessionOptions::from_env.
        unsafe {
            std::env::set_var(BACKEND_ENV, "amd-gpu");
            std::env::set_var(PRECISION_ENV, "fp16");
            std::env::set_var(DEVICE_ID_ENV, "2");
            std::env::set_var(VITIS_CONFIG_ENV, "/opt/vitis/config.json");
            std::env::set_var(EP_CACHE_DIR_ENV, "/var/cache/dghs");
            std::env::set_var(MIGRAPHX_EXHAUSTIVE_TUNE_ENV, "yes");
        }

        let options = SessionOptions::from_env().unwrap();
        assert_eq!(options.backend, Backend::AmdGpu);
        assert_eq!(options.precision, Precision::Fp16);
        assert_eq!(options.device_id, 2);
        assert_eq!(options.ep_cache_dir, Some(PathBuf::from("/var/cache/dghs")));
        assert!(options.migraphx_exhaustive_tune);
    }

    #[test]
    fn profile_chooses_only_available_backend() {
        let profile = ModelProfile::new(
            "pixai",
            vec![Backend::AmdNpu, Backend::AmdGpu, Backend::Cpu],
            Precision::Auto,
            true,
        );
        let capabilities = vec![
            BackendCapabilities {
                backend: Backend::AmdNpu,
                provider_name: Some("VitisAIExecutionProvider".to_owned()),
                available: false,
                supports_fp16: None,
                supports_bf16: Some(true),
                supports_int8: Some(true),
                supports_dynamic_shapes: Some(false),
                reason: Some("not installed".to_owned()),
            },
            BackendCapabilities {
                backend: Backend::AmdGpu,
                provider_name: Some("MIGraphXExecutionProvider".to_owned()),
                available: true,
                supports_fp16: Some(true),
                supports_bf16: None,
                supports_int8: Some(true),
                supports_dynamic_shapes: None,
                reason: None,
            },
        ];

        assert_eq!(
            choose_backend(&profile, &capabilities),
            Some(Backend::AmdGpu)
        );
    }

    #[test]
    fn manifest_round_trips_and_rejects_unknown_precision() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{
                "name": "wd14",
                "model_sha256": "abc",
                "opset": 17,
                "preferred_backends": ["amd-npu", "cpu"],
                "default_precision": "bf16",
                "supported_precisions": ["bf16", "int8"],
                "static_shape": true,
                "compiler_config": "/opt/vitis/wd14.json"
            }"#,
        )
        .unwrap();

        assert_eq!(manifest.profile().name, "wd14");
        assert_eq!(manifest.opset, Some(17));
        assert!(manifest.validate_precision(Precision::Int8).is_ok());
        assert!(matches!(
            manifest.validate_precision(Precision::Fp32),
            Err(InferenceError::ModelUnsupported(_))
        ));
        assert_eq!(
            serde_json::to_value(&manifest).unwrap()["default_precision"],
            "bf16"
        );
    }
}
