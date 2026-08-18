//! Programmatic inference-worker configuration for Node.js/TypeScript.

use crate::inference::{
    Backend, BackendCapabilities, DeviceProvider, DeviceSelection, InferenceError, Precision,
    SessionOptions, configure_inference as configure_core_inference,
    current_inference_options as current_core_inference_options,
    probe_backends_with_options as probe_core_backends,
    reset_inference_configuration as reset_core_inference,
};
use napi_derive::napi;
use std::path::PathBuf;

/// Provider names accepted by the programmatic `provider`/`device` selector.
#[napi(string_enum)]
pub enum NapiInferenceProvider {
    #[napi(value = "auto")]
    Auto,
    #[napi(value = "cpu")]
    Cpu,
    #[napi(value = "cuda")]
    Cuda,
    #[napi(value = "tensorrt")]
    TensorRt,
    #[napi(value = "directml")]
    DirectMl,
    #[napi(value = "intel_gpu")]
    IntelGpu,
    #[napi(value = "intel_npu")]
    IntelNpu,
    #[napi(value = "amd_gpu")]
    AmdGpu,
    #[napi(value = "amd_npu")]
    AmdNpu,
    #[napi(value = "openvino")]
    OpenVino,
}

/// Converts the JavaScript-facing provider enum to the Rust selector.
impl From<NapiInferenceProvider> for DeviceProvider {
    fn from(provider: NapiInferenceProvider) -> Self {
        match provider {
            NapiInferenceProvider::Auto => Self::Auto,
            NapiInferenceProvider::Cpu => Self::Cpu,
            NapiInferenceProvider::Cuda => Self::Cuda,
            NapiInferenceProvider::TensorRt => Self::TensorRt,
            NapiInferenceProvider::DirectMl => Self::DirectMl,
            NapiInferenceProvider::IntelGpu => Self::IntelGpu,
            NapiInferenceProvider::IntelNpu => Self::IntelNpu,
            NapiInferenceProvider::AmdGpu => Self::AmdGpu,
            NapiInferenceProvider::AmdNpu => Self::AmdNpu,
            NapiInferenceProvider::OpenVino => Self::OpenVino,
        }
    }
}

/// Partial worker options accepted by the JavaScript API.
///
/// The backend and precision strings use the same values as the Rust
/// [`Backend`] and [`Precision`] types, for example `amd-gpu` and `fp16`.
///
/// Omitted fields retain their current value (environment-derived unless a
/// programmatic configuration has already been installed).
#[napi(object)]
pub struct NapiInferenceOptions {
    /// Preferred provider-oriented selector, for example `{ provider: 'cuda', device: '1' }`.
    pub provider: Option<NapiInferenceProvider>,
    /// Provider-specific device string. It is optional; omitted means provider default.
    pub device: Option<String>,
    /// Legacy backend spelling retained for compatibility.
    pub backend: Option<String>,
    pub precision: Option<String>,
    pub device_id: Option<i64>,
    pub openvino_device_type: Option<String>,
    pub vitis_config_file: Option<String>,
    pub ep_cache_dir: Option<String>,
    pub migraphx_int8_calibration_table: Option<String>,
    pub migraphx_exhaustive_tune: Option<bool>,
}

/// Effective worker configuration returned to JavaScript.
#[napi(object)]
pub struct NapiInferenceConfiguration {
    pub provider: Option<NapiInferenceProvider>,
    pub device: Option<String>,
    pub backend: String,
    pub precision: String,
    pub device_id: i64,
    pub openvino_device_type: Option<String>,
    pub vitis_config_file: Option<String>,
    pub ep_cache_dir: Option<String>,
    pub migraphx_int8_calibration_table: Option<String>,
    pub migraphx_exhaustive_tune: bool,
}

/// Provider capability information returned by the runtime probe.
#[napi(object)]
pub struct NapiBackendCapabilities {
    pub backend: String,
    pub provider_name: Option<String>,
    pub available: bool,
    pub supports_fp16: Option<bool>,
    pub supports_bf16: Option<bool>,
    pub supports_int8: Option<bool>,
    pub supports_dynamic_shapes: Option<bool>,
    pub reason: Option<String>,
}

fn to_napi_error(error: InferenceError) -> napi::Error {
    let status = if matches!(&error, InferenceError::InvalidInput(_)) {
        napi::Status::InvalidArg
    } else {
        napi::Status::GenericFailure
    };
    napi::Error::new(status, error.to_string())
}

fn path_option(path: Option<String>) -> Option<PathBuf> {
    path.map(PathBuf::from)
}

fn merge_options(input: NapiInferenceOptions) -> Result<SessionOptions, InferenceError> {
    let options = current_core_inference_options()?;
    merge_options_from(options, input)
}

fn merge_options_from(
    mut options: SessionOptions,
    input: NapiInferenceOptions,
) -> Result<SessionOptions, InferenceError> {
    let NapiInferenceOptions {
        provider,
        device,
        backend,
        precision,
        device_id,
        openvino_device_type,
        vitis_config_file,
        ep_cache_dir,
        migraphx_int8_calibration_table,
        migraphx_exhaustive_tune,
    } = input;

    if provider.is_some() && backend.is_some() {
        return Err(InferenceError::InvalidInput(
            "provider and backend cannot be specified together".to_owned(),
        ));
    }
    if (provider.is_some() || device.is_some())
        && (device_id.is_some() || openvino_device_type.is_some())
    {
        return Err(InferenceError::InvalidInput(
            "provider/device cannot be combined with legacy deviceId/openvinoDeviceType".to_owned(),
        ));
    }

    if provider.is_some() || device.is_some() {
        let provider = provider
            .map(DeviceProvider::from)
            .or_else(|| {
                backend
                    .as_deref()
                    .and_then(|value| value.parse::<DeviceProvider>().ok())
            })
            .ok_or_else(|| {
                InferenceError::InvalidInput(
                    "device requires a provider such as cuda, intel_gpu, or intel_npu".to_owned(),
                )
            })?;
        DeviceSelection { provider, device }.apply_to(&mut options)?;
    } else if let Some(backend) = backend {
        options.backend = backend
            .parse::<Backend>()
            .map_err(InferenceError::InvalidInput)?;
    }
    if let Some(precision) = precision {
        options.precision = precision
            .parse::<Precision>()
            .map_err(InferenceError::InvalidInput)?;
    }
    if let Some(device_id) = device_id {
        options.device_id = usize::try_from(device_id).map_err(|_| {
            InferenceError::InvalidInput("deviceId must be a non-negative integer".to_owned())
        })?;
    }
    if openvino_device_type.is_some() {
        options.openvino_device_type = openvino_device_type;
    }
    if vitis_config_file.is_some() {
        options.vitis_config_file = path_option(vitis_config_file);
    }
    if ep_cache_dir.is_some() {
        options.ep_cache_dir = path_option(ep_cache_dir);
    }
    if migraphx_int8_calibration_table.is_some() {
        options.migraphx_int8_calibration_table = path_option(migraphx_int8_calibration_table);
    }
    if let Some(enabled) = migraphx_exhaustive_tune {
        options.migraphx_exhaustive_tune = enabled;
    }

    Ok(options)
}

/// Resolves an optional call-local policy and runs one model operation inside
/// that policy's thread-local scope.
///
/// `None` preserves the legacy environment/process configuration. A partial
/// object is merged with the current configuration, so a caller can change
/// only `provider`, `device`, `precision`, or the legacy backend fields for one request.
pub(crate) fn run_with_inference_options<T>(
    options: Option<NapiInferenceOptions>,
    operation: impl FnOnce() -> napi::Result<T>,
) -> napi::Result<T> {
    let Some(options) = options else {
        return operation();
    };

    let options = merge_options(options).map_err(to_napi_error)?;
    crate::inference::with_inference_options(options, operation)
}

fn configuration_from(options: SessionOptions) -> NapiInferenceConfiguration {
    let (provider, device) = provider_configuration(&options);
    NapiInferenceConfiguration {
        provider,
        device,
        backend: options.backend.to_string(),
        precision: options.precision.to_string(),
        device_id: options.device_id as i64,
        openvino_device_type: options.openvino_device_type,
        vitis_config_file: options
            .vitis_config_file
            .map(|path| path.to_string_lossy().into_owned()),
        ep_cache_dir: options
            .ep_cache_dir
            .map(|path| path.to_string_lossy().into_owned()),
        migraphx_int8_calibration_table: options
            .migraphx_int8_calibration_table
            .map(|path| path.to_string_lossy().into_owned()),
        migraphx_exhaustive_tune: options.migraphx_exhaustive_tune,
    }
}

/// Converts the effective Rust options back to the public provider/device
/// representation returned by the N-API configuration query.
fn provider_configuration(
    options: &SessionOptions,
) -> (Option<NapiInferenceProvider>, Option<String>) {
    match options.backend {
        Backend::Auto => (None, None),
        Backend::Cpu => (Some(NapiInferenceProvider::Cpu), None),
        Backend::Cuda => (Some(NapiInferenceProvider::Cuda), ordinal_device(options)),
        Backend::TensorRt => (
            Some(NapiInferenceProvider::TensorRt),
            ordinal_device(options),
        ),
        Backend::DirectMl => (
            Some(NapiInferenceProvider::DirectMl),
            ordinal_device(options),
        ),
        Backend::AmdGpu => (Some(NapiInferenceProvider::AmdGpu), ordinal_device(options)),
        Backend::AmdNpu => (Some(NapiInferenceProvider::AmdNpu), None),
        Backend::OpenVino => {
            let device_type = options.openvino_device_type.as_deref().unwrap_or("AUTO");
            let normalized = device_type.to_ascii_uppercase();
            if let Some(device) = normalized.strip_prefix("GPU") {
                return (
                    Some(NapiInferenceProvider::IntelGpu),
                    openvino_ordinal(device),
                );
            }
            if let Some(device) = normalized.strip_prefix("NPU") {
                return (
                    Some(NapiInferenceProvider::IntelNpu),
                    openvino_ordinal(device),
                );
            }
            (
                Some(NapiInferenceProvider::OpenVino),
                Some(device_type.to_owned()),
            )
        }
    }
}

/// Returns the explicit ordinal when the effective provider uses one.
fn ordinal_device(options: &SessionOptions) -> Option<String> {
    (options.device_id != 0).then(|| options.device_id.to_string())
}

/// Extracts the ordinal portion from an OpenVINO `GPU.N`/`NPU.N` policy.
fn openvino_ordinal(suffix: &str) -> Option<String> {
    suffix.strip_prefix('.').map(str::to_owned)
}

fn capabilities_from(capability: BackendCapabilities) -> NapiBackendCapabilities {
    NapiBackendCapabilities {
        backend: capability.backend.to_string(),
        provider_name: capability.provider_name,
        available: capability.available,
        supports_fp16: capability.supports_fp16,
        supports_bf16: capability.supports_bf16,
        supports_int8: capability.supports_int8,
        supports_dynamic_shapes: capability.supports_dynamic_shapes,
        reason: capability.reason,
    }
}

/// Configures the worker used by existing high-level inference APIs.
///
/// Call this once during application startup, before the first model API. The
/// configuration takes precedence over `DGHS_*` environment variables.
#[napi]
pub fn configure_inference(options: NapiInferenceOptions) -> napi::Result<()> {
    let options = merge_options(options).map_err(to_napi_error)?;
    configure_core_inference(options).map_err(to_napi_error)
}

/// Restores environment-based selection for a worker that has not started.
#[napi]
pub fn reset_inference_configuration() -> napi::Result<()> {
    reset_core_inference().map_err(to_napi_error)
}

/// Returns the effective backend/device configuration.
#[napi]
pub fn get_inference_configuration() -> napi::Result<NapiInferenceConfiguration> {
    current_core_inference_options()
        .map(configuration_from)
        .map_err(to_napi_error)
}

/// Probes providers available in the loaded ONNX Runtime.
#[napi]
pub fn probe_inference_backends(
    options: Option<NapiInferenceOptions>,
) -> napi::Result<Vec<NapiBackendCapabilities>> {
    let options = match options {
        Some(options) => merge_options(options).map_err(to_napi_error)?,
        None => current_core_inference_options().map_err(to_napi_error)?,
    };
    Ok(probe_core_backends(&options)
        .into_iter()
        .map(capabilities_from)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{Backend, Precision};

    fn empty_options() -> NapiInferenceOptions {
        NapiInferenceOptions {
            provider: None,
            device: None,
            backend: None,
            precision: None,
            device_id: None,
            openvino_device_type: None,
            vitis_config_file: None,
            ep_cache_dir: None,
            migraphx_int8_calibration_table: None,
            migraphx_exhaustive_tune: None,
        }
    }

    #[test]
    fn merge_options_parses_valid_backend_and_precision() {
        let mut input = empty_options();
        input.backend = Some("amd-gpu".to_owned());
        input.precision = Some("fp16".to_owned());

        let options = merge_options_from(SessionOptions::default(), input).unwrap();

        assert_eq!(options.backend, Backend::AmdGpu);
        assert_eq!(options.precision, Precision::Fp16);
    }

    #[test]
    fn merge_options_accepts_provider_without_a_device() {
        let mut input = empty_options();
        input.provider = Some(NapiInferenceProvider::Cuda);

        let options = merge_options_from(SessionOptions::default(), input).unwrap();

        assert_eq!(options.backend, Backend::Cuda);
        assert_eq!(options.device_id, 0);
    }

    #[test]
    fn merge_options_maps_intel_provider_device() {
        let mut input = empty_options();
        input.provider = Some(NapiInferenceProvider::IntelGpu);
        input.device = Some("1".to_owned());

        let options = merge_options_from(SessionOptions::default(), input).unwrap();

        assert_eq!(options.backend, Backend::OpenVino);
        assert_eq!(options.openvino_device_type.as_deref(), Some("GPU.1"));
    }

    #[test]
    fn merge_options_rejects_invalid_backend_and_precision() {
        let mut invalid_backend = empty_options();
        invalid_backend.backend = Some("not-a-backend".to_owned());
        assert!(matches!(
            merge_options_from(SessionOptions::default(), invalid_backend),
            Err(InferenceError::InvalidInput(message)) if message.contains("unsupported backend")
        ));

        let mut invalid_precision = empty_options();
        invalid_precision.precision = Some("fp64".to_owned());
        assert!(matches!(
            merge_options_from(SessionOptions::default(), invalid_precision),
            Err(InferenceError::InvalidInput(message)) if message.contains("unsupported precision")
        ));
    }

    #[test]
    fn merge_options_rejects_negative_device_id() {
        let mut input = empty_options();
        input.device_id = Some(-1);

        assert!(matches!(
            merge_options_from(SessionOptions::default(), input),
            Err(InferenceError::InvalidInput(message)) if message.contains("non-negative")
        ));
    }

    #[test]
    fn merge_options_preserves_omitted_fields() {
        let base = SessionOptions::for_backend(Backend::OpenVino)
            .with_precision(Precision::Fp32)
            .with_device_id(4)
            .with_openvino_device_type("GPU")
            .with_vitis_config_file("/opt/vitis/config.json")
            .with_ep_cache_dir("/var/cache/ep")
            .with_migraphx_int8_calibration_table("/opt/models/calibration.table")
            .with_migraphx_exhaustive_tune(true);
        let mut input = empty_options();
        input.backend = Some("cpu".to_owned());

        let options = merge_options_from(base.clone(), input).unwrap();

        assert_eq!(options.backend, Backend::Cpu);
        assert_eq!(options.precision, base.precision);
        assert_eq!(options.device_id, base.device_id);
        assert_eq!(options.openvino_device_type, base.openvino_device_type);
        assert_eq!(options.vitis_config_file, base.vitis_config_file);
        assert_eq!(options.ep_cache_dir, base.ep_cache_dir);
        assert_eq!(
            options.migraphx_int8_calibration_table,
            base.migraphx_int8_calibration_table
        );
        assert_eq!(
            options.migraphx_exhaustive_tune,
            base.migraphx_exhaustive_tune
        );
    }
}
