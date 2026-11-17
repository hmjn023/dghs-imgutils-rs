//! Node.js adapter for PLaMo 3 Translate 0.8B English-to-Japanese translation.

use crate::generic::plamo3::{Plamo3Error, Plamo3Options, plamo3_translate as core_translate};
use crate::napi::inference::{NapiInferenceOptions, run_with_inference_options};
use napi_derive::napi;

/// PLaMo 3 Translate 0.8B Q4 model location and generation budget.
#[napi(object)]
pub struct NapiPlamo3Options {
    /// Local root containing tokenizer.json and onnx/model_q4.onnx. Defaults to HF download.
    pub model_dir: Option<String>,
    /// Maximum generated tokens including EOS, default 512. Throws if translation is incomplete.
    pub max_new_tokens: Option<u32>,
}

/// Translate English text to Japanese with PLaMo 3 Translate 0.8B.
/// Uses Q4 weights, FP32 activations, and cached sessions. Synchronous.
/// Throws if source prompt plus generation budget exceeds 32768 tokens,
/// or generation reaches maxNewTokens before EOS.
#[napi]
pub fn plamo3_translate(
    text: String,
    model_options: Option<NapiPlamo3Options>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<String> {
    let mut options = Plamo3Options::default();
    if let Some(value) = model_options {
        options.model_dir = value.model_dir.map(Into::into);
        if let Some(limit) = value.max_new_tokens {
            options.max_new_tokens = limit as usize;
        }
    }
    run_with_inference_options(inference_options, || {
        core_translate(&text, &options).map_err(|error| {
            let status = match &error {
                Plamo3Error::InvalidInput(_) => napi::Status::InvalidArg,
                _ => napi::Status::GenericFailure,
            };
            napi::Error::new(status, error.to_string())
        })
    })
}
