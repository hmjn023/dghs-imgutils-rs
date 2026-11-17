//! Node.js adapter for SmolVLM2 image-conditioned text generation.

use crate::generic::smolvlm2::{
    SmolVlm2Error, SmolVlm2Options, smolvlm2_generate as core_generate,
};
use crate::napi::inference::{NapiInferenceOptions, run_with_inference_options};
use napi_derive::napi;

/// SmolVLM2-256M model location and generation limit.
#[napi(object)]
pub struct NapiSmolVlm2Options {
    /// Local export root containing tokenizer.json and onnx/. Defaults to HF download.
    pub model_dir: Option<String>,
    /// Maximum generated tokens including EOS, default 128. Output may end at this limit.
    pub max_new_tokens: Option<u32>,
}

/// Generate English text from a local image and a user prompt using SmolVLM2-256M.
/// Uses FP32 vision and INT8 text graphs, one 512×512 image, and greedy decoding.
/// Sessions are cached; each invocation starts a fresh conversation. Synchronous.
#[napi]
pub fn smolvlm2_generate(
    path: String,
    prompt: String,
    model_options: Option<NapiSmolVlm2Options>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<String> {
    let mut options = SmolVlm2Options::default();
    if let Some(value) = model_options {
        options.model_dir = value.model_dir.map(Into::into);
        if let Some(limit) = value.max_new_tokens {
            options.max_new_tokens = limit as usize;
        }
    }
    let image = crate::napi::open_image_robust(&path)
        .map_err(|error| napi::Error::new(napi::Status::InvalidArg, error.to_string()))?;
    run_with_inference_options(inference_options, || {
        core_generate(&image, &prompt, &options).map_err(|error| {
            let status = match &error {
                SmolVlm2Error::InvalidInput(_) => napi::Status::InvalidArg,
                _ => napi::Status::GenericFailure,
            };
            napi::Error::new(status, error.to_string())
        })
    })
}
