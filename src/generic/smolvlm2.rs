//! Single-image English text generation with SmolVLM2-256M-Video-Instruct.
//!
//! Uses the official FP32 vision encoder and INT8 text graphs. Images are
//! resized to one 512×512 RGB frame without splitting. Generation is greedy,
//! stops at EOS or the requested token limit, and keeps KV caches per call.

use std::collections::HashMap;
use std::path::PathBuf;

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ort::value::{DynValue, Tensor};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::hub::hf_hub_download;
use crate::inference::{get_or_create_session, lock_session};

/// Official model repository containing the ONNX graphs and tokenizer.
pub const REPO_ID: &str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct";
const IMAGE_SIZE: usize = 512;
const IMAGE_TOKENS: usize = 64;
const HIDDEN_SIZE: usize = 576;
const VOCAB_SIZE: usize = 49280;
const IMAGE_TOKEN_ID: i64 = 49190;
const EOS_TOKEN_ID: u32 = 49279;
const CONTEXT_LENGTH: usize = 8192;
const CACHE_LAYERS: usize = 30;

/// Errors from image validation, model loading, and caption generation.
#[derive(Debug, Error)]
pub enum SmolVlm2Error {
    /// Invalid image, prompt, or generation limit.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Incompatible model tensors or non-finite output.
    #[error("Invalid SmolVLM2 model output: {0}")]
    ModelOutput(String),
    /// Tokenizer loading, encoding, or decoding failure.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// Model download failure.
    #[error(transparent)]
    Hub(#[from] crate::hub::HubError),
    /// Session initialization or locking failure.
    #[error(transparent)]
    Inference(#[from] crate::inference::InferenceError),
    /// ONNX Runtime tensor or inference failure.
    #[error(transparent)]
    Ort(#[from] ort::Error),
}

/// Model location and greedy generation limit.
#[derive(Debug, Clone)]
pub struct SmolVlm2Options {
    /// Local root containing tokenizer.json and onnx/ with the official
    /// vision_encoder.onnx, embed_tokens_int8.onnx and decoder_model_merged_int8.onnx.
    /// None downloads these assets into the standard Hugging Face cache.
    pub model_dir: Option<PathBuf>,
    /// Maximum generated tokens, including EOS. Defaults to 128.
    /// Prompt plus generation budget must fit the 8192-token context.
    pub max_new_tokens: usize,
}

impl Default for SmolVlm2Options {
    fn default() -> Self {
        Self {
            model_dir: None,
            max_new_tokens: 128,
        }
    }
}

impl SmolVlm2Options {
    fn asset(&self, name: &str) -> Result<PathBuf, SmolVlm2Error> {
        if let Some(root) = &self.model_dir {
            let path = root.join(name);
            if !path.is_file() {
                return Err(SmolVlm2Error::InvalidInput(format!(
                    "missing model asset: {}",
                    path.display()
                )));
            }
            Ok(path)
        } else {
            Ok(hf_hub_download(REPO_ID, name, None, None)?)
        }
    }

    fn validate(&self, prompt: &str) -> Result<(), SmolVlm2Error> {
        if prompt.trim().is_empty() {
            return Err(SmolVlm2Error::InvalidInput(
                "prompt must contain text".into(),
            ));
        }
        if !(1..CONTEXT_LENGTH).contains(&self.max_new_tokens) {
            return Err(SmolVlm2Error::InvalidInput(
                "max_new_tokens must be between 1 and 8191".into(),
            ));
        }
        Ok(())
    }
}

fn tokenize(tokenizer: &Tokenizer, prompt: &str, budget: usize) -> Result<Vec<i64>, SmolVlm2Error> {
    let formatted = format!(
        "<|im_start|>User:<fake_token_around_image><global-img>{}<fake_token_around_image>{prompt}<end_of_utterance>\nAssistant:",
        "<image>".repeat(IMAGE_TOKENS)
    );
    let encoding = tokenizer
        .encode(formatted, false)
        .map_err(|error| SmolVlm2Error::Tokenizer(error.to_string()))?;
    if encoding.len() > CONTEXT_LENGTH - budget {
        return Err(SmolVlm2Error::InvalidInput(
            "prompt plus generation budget exceeds 8192 tokens".into(),
        ));
    }
    let ids: Vec<_> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    if ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count() != IMAGE_TOKENS {
        return Err(SmolVlm2Error::InvalidInput(
            "prompt must not contain image placeholder tokens".into(),
        ));
    }
    Ok(ids)
}

fn preprocess(image: &DynamicImage) -> Vec<f32> {
    let rgb = crate::image::force_image_background(image, [255, 255, 255])
        .resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::Lanczos3)
        .to_rgb8();
    let mut pixels = vec![0.0; 3 * IMAGE_SIZE * IMAGE_SIZE];
    for (x, y, rgb) in rgb.enumerate_pixels() {
        for channel in 0..3 {
            pixels[channel * IMAGE_SIZE * IMAGE_SIZE + y as usize * IMAGE_SIZE + x as usize] =
                (rgb[channel] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    pixels
}

fn merge_features(ids: &[i64], embeds: &mut [f32], features: &[f32]) -> Result<(), SmolVlm2Error> {
    if embeds.len() != ids.len() * HIDDEN_SIZE || features.len() != IMAGE_TOKENS * HIDDEN_SIZE {
        return Err(SmolVlm2Error::ModelOutput(
            "unexpected embedding dimensions".into(),
        ));
    }
    let positions: Vec<_> = ids
        .iter()
        .enumerate()
        .filter(|&(_, &id)| id == IMAGE_TOKEN_ID)
        .map(|(index, _)| index)
        .collect();
    if positions.len() != IMAGE_TOKENS {
        return Err(SmolVlm2Error::ModelOutput(
            "expected 64 image placeholders".into(),
        ));
    }
    for (source, destination) in positions.into_iter().enumerate() {
        embeds[destination * HIDDEN_SIZE..(destination + 1) * HIDDEN_SIZE]
            .copy_from_slice(&features[source * HIDDEN_SIZE..(source + 1) * HIDDEN_SIZE]);
    }
    Ok(())
}

fn next_token(logits: &[f32]) -> Result<u32, SmolVlm2Error> {
    if logits.len() < VOCAB_SIZE || !logits.len().is_multiple_of(VOCAB_SIZE) {
        return Err(SmolVlm2Error::ModelOutput(
            "unexpected logits dimensions".into(),
        ));
    }
    let last = &logits[logits.len() - VOCAB_SIZE..];
    if !last.iter().all(|value| value.is_finite()) {
        return Err(SmolVlm2Error::ModelOutput("non-finite logits".into()));
    }
    // Preserve the first index on ties, matching greedy argmax.
    Ok(last.iter().enumerate().fold(
        0,
        |best, (index, value)| {
            if *value > last[best] { index } else { best }
        },
    ) as u32)
}

/// Generates text from one image and a user prompt using the English-focused model.
///
/// Sessions follow the existing inference configuration and session cache.
/// Each call starts a fresh conversation and KV cache. EOS is omitted from the
/// returned string. Reaching max_new_tokens returns the generated prefix, which
/// may end mid-sentence. Japanese output is unreliable for this model.
pub fn smolvlm2_generate(
    image: &DynamicImage,
    prompt: &str,
    options: &SmolVlm2Options,
) -> Result<String, SmolVlm2Error> {
    options.validate(prompt)?;
    if image.dimensions().0 == 0 || image.dimensions().1 == 0 {
        return Err(SmolVlm2Error::InvalidInput(
            "image dimensions must be positive".into(),
        ));
    }
    let mut tokenizer = Tokenizer::from_file(options.asset("tokenizer.json")?)
        .map_err(|error| SmolVlm2Error::Tokenizer(error.to_string()))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| SmolVlm2Error::Tokenizer(error.to_string()))?;
    let mut ids = tokenize(&tokenizer, prompt, options.max_new_tokens)?;
    let prompt_len = ids.len();

    let vision = get_or_create_session(options.asset("onnx/vision_encoder.onnx")?)?;
    let embedding = get_or_create_session(options.asset("onnx/embed_tokens_int8.onnx")?)?;
    let decoder = get_or_create_session(options.asset("onnx/decoder_model_merged_int8.onnx")?)?;
    let features = {
        let mut session = lock_session(&vision)?;
        let outputs = session.run(ort::inputs![
            "pixel_values" => Tensor::from_array(([1, 1, 3, IMAGE_SIZE, IMAGE_SIZE], preprocess(image)))?,
            "pixel_attention_mask" => Tensor::from_array(([1, 1, IMAGE_SIZE, IMAGE_SIZE], vec![true; IMAGE_SIZE * IMAGE_SIZE]))?,
        ])?;
        outputs
            .get("image_features")
            .ok_or_else(|| SmolVlm2Error::ModelOutput("missing image_features".into()))?
            .try_extract_tensor::<f32>()?
            .1
            .to_vec()
    };
    let mut cache: HashMap<String, DynValue> = HashMap::new();
    {
        let session = lock_session(&decoder)?;
        for layer in 0..CACHE_LAYERS {
            for kind in ["key", "value"] {
                // ort rc.12 requires the allocator API for zero-length tensors.
                cache.insert(
                    format!("past_key_values.{layer}.{kind}"),
                    Tensor::<f32>::new(session.allocator(), [1_usize, 3, 0, 64])?.into_dyn(),
                );
            }
        }
    }
    let mut generated = Vec::new();
    for step in 0..options.max_new_tokens {
        let seq_len = ids.len();
        let mut embeds = {
            let mut session = lock_session(&embedding)?;
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array(([1, seq_len], ids.clone()))?,
            ])?;
            outputs
                .get("inputs_embeds")
                .ok_or_else(|| SmolVlm2Error::ModelOutput("missing inputs_embeds".into()))?
                .try_extract_tensor::<f32>()?
                .1
                .to_vec()
        };
        if step == 0 {
            merge_features(&ids, &mut embeds, &features)?;
        }
        let total_len = prompt_len + step;
        let positions: Vec<i64> = ((total_len - seq_len)..total_len)
            .map(|n| n as i64)
            .collect();
        let mut inputs = cache;
        inputs.insert(
            "inputs_embeds".into(),
            Tensor::from_array(([1, seq_len, HIDDEN_SIZE], embeds))?.into_dyn(),
        );
        inputs.insert(
            "attention_mask".into(),
            Tensor::from_array(([1, total_len], vec![1_i64; total_len]))?.into_dyn(),
        );
        inputs.insert(
            "position_ids".into(),
            Tensor::from_array(([1, seq_len], positions))?.into_dyn(),
        );
        let mut session = lock_session(&decoder)?;
        let outputs = session.run(inputs)?;
        let logits = outputs
            .get("logits")
            .ok_or_else(|| SmolVlm2Error::ModelOutput("missing logits".into()))?;
        let next = next_token(logits.try_extract_tensor::<f32>()?.1)?;
        if next == EOS_TOKEN_ID {
            break;
        }
        generated.push(next);
        ids = vec![i64::from(next)];
        cache = outputs
            .into_iter()
            .filter_map(|(name, value)| {
                name.strip_prefix("present.")
                    .map(|suffix| (format!("past_key_values.{suffix}"), value))
            })
            .collect();
        if cache.len() != CACHE_LAYERS * 2 {
            return Err(SmolVlm2Error::ModelOutput("incomplete KV cache".into()));
        }
    }
    tokenizer
        .decode(&generated, true)
        .map(|text| text.trim().to_owned())
        .map_err(|error| SmolVlm2Error::Tokenizer(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inputs_before_loading_assets() {
        let image = DynamicImage::new_rgb8(1, 1);
        let options = SmolVlm2Options::default();
        assert!(matches!(
            smolvlm2_generate(&image, "  ", &options),
            Err(SmolVlm2Error::InvalidInput(_))
        ));
        for limit in [0, CONTEXT_LENGTH, usize::MAX] {
            let options = SmolVlm2Options {
                max_new_tokens: limit,
                ..Default::default()
            };
            assert!(matches!(
                smolvlm2_generate(&image, "Describe", &options),
                Err(SmolVlm2Error::InvalidInput(_))
            ));
        }
        assert!(matches!(
            smolvlm2_generate(&DynamicImage::new_rgb8(0, 1), "Describe", &options),
            Err(SmolVlm2Error::InvalidInput(_))
        ));
    }

    #[test]
    fn merges_only_image_tokens_and_rejects_wrong_shapes() {
        let mut ids = vec![IMAGE_TOKEN_ID; IMAGE_TOKENS];
        ids.insert(0, 12);
        let mut embeds = vec![2.0; ids.len() * HIDDEN_SIZE];
        let features = vec![3.0; IMAGE_TOKENS * HIDDEN_SIZE];
        merge_features(&ids, &mut embeds, &features).unwrap();
        assert!(embeds[..HIDDEN_SIZE].iter().all(|&v| v == 2.0));
        assert!(embeds[HIDDEN_SIZE..].iter().all(|&v| v == 3.0));
        assert!(merge_features(&ids, &mut embeds, &[]).is_err());
        ids[1] = 4;
        assert!(merge_features(&ids, &mut embeds, &features).is_err());
    }

    #[test]
    fn greedy_argmax_uses_last_position_and_first_tie() {
        let mut logits = vec![0.0; VOCAB_SIZE * 2];
        logits[1] = 100.0;
        logits[VOCAB_SIZE + 7] = 3.0;
        logits[VOCAB_SIZE + 8] = 3.0;
        assert_eq!(next_token(&logits).unwrap(), 7);
        logits[VOCAB_SIZE] = f32::NAN;
        assert!(next_token(&logits).is_err());
        assert!(next_token(&[]).is_err());
    }

    #[test]
    fn transparent_pixels_use_white_background() {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([0, 0, 0, 0]),
        ));
        assert!(preprocess(&image).iter().all(|&value| value == 1.0));
    }
}
