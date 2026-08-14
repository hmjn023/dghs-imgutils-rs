//! Programmatic inference-worker configuration for Node.js/TypeScript.

use crate::inference::{
    Backend, BackendCapabilities, InferenceError, Precision, SessionOptions,
    configure_inference as configure_core_inference,
    current_inference_options as current_core_inference_options,
    probe_backends as probe_core_backends, reset_inference_configuration as reset_core_inference,
};
use napi_derive::napi;
use std::path::PathBuf;

/// Partial worker options accepted by the JavaScript API.
///
/// The backend and precision strings use the same values as the Rust
/// [`Backend`] and [`Precision`] types, for example `amd-gpu` and `fp16`.
///
/// Omitted fields retain their current value (environment-derived unless a
/// programmatic configuration has already been installed).
#[napi(object)]
pub struct NapiInferenceOptions {
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
    let mut options = current_core_inference_options()?;

    if let Some(backend) = input.backend {
        options.backend = backend
            .parse::<Backend>()
            .map_err(InferenceError::InvalidInput)?;
    }
    if let Some(precision) = input.precision {
        options.precision = precision
            .parse::<Precision>()
            .map_err(InferenceError::InvalidInput)?;
    }
    if let Some(device_id) = input.device_id {
        options.device_id = usize::try_from(device_id).map_err(|_| {
            InferenceError::InvalidInput("deviceId must be a non-negative integer".to_owned())
        })?;
    }
    if input.openvino_device_type.is_some() {
        options.openvino_device_type = input.openvino_device_type;
    }
    if input.vitis_config_file.is_some() {
        options.vitis_config_file = path_option(input.vitis_config_file);
    }
    if input.ep_cache_dir.is_some() {
        options.ep_cache_dir = path_option(input.ep_cache_dir);
    }
    if input.migraphx_int8_calibration_table.is_some() {
        options.migraphx_int8_calibration_table =
            path_option(input.migraphx_int8_calibration_table);
    }
    if let Some(enabled) = input.migraphx_exhaustive_tune {
        options.migraphx_exhaustive_tune = enabled;
    }

    Ok(options)
}

/// Resolves an optional call-local policy and runs one model operation inside
/// that policy's thread-local scope.
///
/// `None` preserves the legacy environment/process configuration. A partial
/// object is merged with the current configuration, so a caller can change
/// only `backend`, `precision`, or `deviceId` for one request.
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
    NapiInferenceConfiguration {
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
pub fn probe_inference_backends() -> Vec<NapiBackendCapabilities> {
    probe_core_backends()
        .into_iter()
        .map(capabilities_from)
        .collect()
}
