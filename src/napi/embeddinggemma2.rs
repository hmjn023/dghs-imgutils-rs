//! Node.js adapters for EmbeddingGemma 2 image/text embeddings.

use crate::generic::embeddinggemma2::{self, EmbeddingGemma2Options, EmbeddingGemma2TextTask};
use crate::napi::inference::{NapiInferenceOptions, run_with_inference_options};
use napi_derive::napi;

/// EmbeddingGemma 2 model and output options.
#[napi(object)]
pub struct NapiEmbeddingGemma2Options {
    /// Local export root containing tokenizer.json and onnx/. Defaults to HF download.
    pub model_dir: Option<String>,
    /// Weight format: q8 (default), q4, or fp32.
    pub dtype: Option<String>,
    /// Output dimensions: 768 (default), 512, 256, or 128. Always L2-normalized.
    pub dimensions: Option<u32>,
    /// Vision token budget: 280 (default), 70, 140, 560, or 1120.
    pub max_soft_tokens: Option<u32>,
}

fn options(value: Option<NapiEmbeddingGemma2Options>) -> napi::Result<EmbeddingGemma2Options> {
    let mut options = EmbeddingGemma2Options::default();
    if let Some(value) = value {
        options.model_dir = value.model_dir.map(Into::into);
        if let Some(dtype) = value.dtype {
            options.dtype = dtype.parse().map_err(|e: crate::generic::GenericError| {
                napi::Error::new(napi::Status::InvalidArg, e.to_string())
            })?;
        }
        if let Some(dimensions) = value.dimensions {
            options.dimensions = dimensions as usize;
        }
        if let Some(budget) = value.max_soft_tokens {
            options.max_soft_tokens = budget as usize;
        }
    }
    Ok(options)
}

fn convert(embeddings: Vec<Vec<f32>>) -> Vec<Vec<f64>> {
    embeddings
        .into_iter()
        .map(|v| v.into_iter().map(f64::from).collect())
        .collect()
}

/// Encode text into EmbeddingGemma 2 vectors comparable with image embeddings.
/// task: SearchQuery (default), Document, SentenceSimilarity, Clustering,
/// Classification, or Raw (already formatted). Rejects inputs over 8192 tokens.
#[napi]
pub fn embeddinggemma2_text_encode(
    texts: Vec<String>,
    task: Option<String>,
    model_options: Option<NapiEmbeddingGemma2Options>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<Vec<Vec<f64>>> {
    let task: EmbeddingGemma2TextTask = task.as_deref().unwrap_or("SearchQuery").parse().map_err(
        |e: crate::generic::GenericError| napi::Error::new(napi::Status::InvalidArg, e.to_string()),
    )?;
    let options = options(model_options)?;
    run_with_inference_options(inference_options, || {
        embeddinggemma2::embeddinggemma2_text_encode(&texts, task, &options)
            .map(convert)
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    })
}

/// Encode local image files into separate EmbeddingGemma 2 vectors.
/// Returns one normalized vector per image, in input order.
#[napi]
pub fn embeddinggemma2_image_encode(
    paths: Vec<String>,
    model_options: Option<NapiEmbeddingGemma2Options>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<Vec<Vec<f64>>> {
    let options = options(model_options)?;
    let images = paths
        .iter()
        .map(|path| {
            crate::napi::open_image_robust(path)
                .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))
        })
        .collect::<napi::Result<Vec<_>>>()?;
    run_with_inference_options(inference_options, || {
        embeddinggemma2::embeddinggemma2_image_encode(&images, &options)
            .map(convert)
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    })
}
