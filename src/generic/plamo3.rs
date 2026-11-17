//! English-to-Japanese translation with PLaMo 3 Translate 0.8B.
//!
//! Uses the TeamFlaxia Q4 ONNX conversion with FP32 activations and KV caches.
//! PFML3 prompts and per-request caches are built in Rust. Sessions follow
//! the existing inference configuration and shared session cache.

use std::{collections::HashMap, path::PathBuf};

use ort::value::{DynValue, Tensor};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::hub::hf_hub_download;
use crate::inference::{get_or_create_session, lock_session};

/// Community ONNX conversion of pfnet/plamo-3-translate-0.8b.
pub const REPO_ID: &str = "TeamFlaxia/plamo-3-translate-0.8b-onnx";
const CONTEXT_LENGTH: usize = 32768;
const CACHE_LAYERS: usize = 16;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 107520;
const EOS_TOKEN_IDS: [u32; 2] = [1, 16];

/// Translation input, model, tokenizer, and generation failures.
#[derive(Debug, Error)]
pub enum Plamo3Error {
    /// Invalid source text, model path, or generation limit.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Incompatible or non-finite model output.
    #[error("Invalid PLaMo 3 model output: {0}")]
    ModelOutput(String),
    /// Failure to load or use the tokenizer.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// The translation did not reach EOS within the generation budget.
    #[error(
        "Translation exceeded max_new_tokens ({0}); increase the limit or split the source text"
    )]
    GenerationLimit(usize),
    /// Failure to download a model asset.
    #[error(transparent)]
    Hub(#[from] crate::hub::HubError),
    /// Session initialization or locking failure.
    #[error(transparent)]
    Inference(#[from] crate::inference::InferenceError),
    /// ONNX Runtime tensor or inference failure.
    #[error(transparent)]
    Ort(#[from] ort::Error),
}

/// Location of the PLaMo 3 Translate 0.8B Q4 export and generation budget.
#[derive(Debug, Clone)]
pub struct Plamo3Options {
    /// Local root containing tokenizer.json and onnx/model_q4.onnx.
    /// None downloads both assets to the standard Hugging Face cache.
    pub model_dir: Option<PathBuf>,
    /// Maximum generated tokens including EOS, default 512.
    /// Source prompt plus this budget must fit the 32768-token trained context.
    pub max_new_tokens: usize,
}

impl Default for Plamo3Options {
    fn default() -> Self {
        Self {
            model_dir: None,
            max_new_tokens: 512,
        }
    }
}

impl Plamo3Options {
    fn validate(&self, text: &str) -> Result<(), Plamo3Error> {
        if text.trim().is_empty() {
            return Err(Plamo3Error::InvalidInput(
                "text must contain non-whitespace characters".into(),
            ));
        }
        // The official chat template reserves these tokens for PFML structure.
        if text.contains("<|plamo:") {
            return Err(Plamo3Error::InvalidInput(
                "text must not contain reserved PFML tokens (<|plamo:...)".into(),
            ));
        }
        if !(1..CONTEXT_LENGTH).contains(&self.max_new_tokens) {
            return Err(Plamo3Error::InvalidInput(
                "max_new_tokens must be between 1 and 32767".into(),
            ));
        }
        Ok(())
    }

    fn asset(&self, name: &str) -> Result<PathBuf, Plamo3Error> {
        if let Some(root) = &self.model_dir {
            let path = root.join(name);
            if !path.is_file() {
                return Err(Plamo3Error::InvalidInput(format!(
                    "missing model asset: {}",
                    path.display()
                )));
            }
            Ok(path)
        } else {
            Ok(hf_hub_download(REPO_ID, name, None, None)?)
        }
    }
}

fn translation_prompt(text: &str) -> String {
    format!(
        concat!(
            "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation",
            "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>{}",
            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
        ),
        text
    )
}

fn tokenize(tokenizer: &Tokenizer, text: &str, budget: usize) -> Result<Vec<i64>, Plamo3Error> {
    // The prompt already contains BOS. Disable automatic special-token insertion.
    let encoding = tokenizer
        .encode(translation_prompt(text), false)
        .map_err(|error| Plamo3Error::Tokenizer(error.to_string()))?;
    if encoding.len() > CONTEXT_LENGTH - budget {
        return Err(Plamo3Error::InvalidInput(
            "source prompt plus generation budget exceeds 32768 tokens".into(),
        ));
    }
    Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
}

fn next_token(logits: &[f32]) -> Result<u32, Plamo3Error> {
    if logits.len() < VOCAB_SIZE || !logits.len().is_multiple_of(VOCAB_SIZE) {
        return Err(Plamo3Error::ModelOutput(
            "unexpected logits dimensions".into(),
        ));
    }
    let last = &logits[logits.len() - VOCAB_SIZE..];
    if !last.iter().all(|value| value.is_finite()) {
        return Err(Plamo3Error::ModelOutput("non-finite logits".into()));
    }
    // Preserve the first token when logits are tied, matching greedy argmax.
    Ok(last.iter().enumerate().fold(
        0,
        |best, (index, value)| {
            if *value > last[best] { index } else { best }
        },
    ) as u32)
}

/// Translates English source text to Japanese with greedy decoding.
///
/// Uses the Q4 weights and FP32 activations of the 0.8B ONNX conversion.
/// Each call starts a fresh PFML3 request and KV cache. Returns only the
/// translated text, without the prompt or EOS. Reaching the generation budget
/// before EOS returns GenerationLimit, allowing callers to retry with a larger
/// budget. Input whitespace is preserved in the prompt.
pub fn plamo3_translate(text: &str, options: &Plamo3Options) -> Result<String, Plamo3Error> {
    options.validate(text)?;
    let mut tokenizer = Tokenizer::from_file(options.asset("tokenizer.json")?)
        .map_err(|error| Plamo3Error::Tokenizer(error.to_string()))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| Plamo3Error::Tokenizer(error.to_string()))?;
    let mut ids = tokenize(&tokenizer, text, options.max_new_tokens)?;
    let prompt_len = ids.len();

    let shared = get_or_create_session(options.asset("onnx/model_q4.onnx")?)?;
    let mut session = lock_session(&shared)?;
    let mut cache: HashMap<String, DynValue> = HashMap::new();
    for layer in 0..CACHE_LAYERS {
        for kind in ["key", "value"] {
            // ort rc.12 requires the allocator API for zero-length KV tensors.
            cache.insert(
                format!("past_key_values.{layer}.{kind}"),
                Tensor::<f32>::new(session.allocator(), [1, KV_HEADS, 0, HEAD_DIM])?.into_dyn(),
            );
        }
    }
    let mut generated = Vec::new();
    for step in 0..options.max_new_tokens {
        let seq_len = ids.len();
        let total_len = prompt_len + step;
        let mut inputs = cache;
        inputs.insert(
            "input_ids".into(),
            Tensor::from_array(([1, seq_len], ids.clone()))?.into_dyn(),
        );
        inputs.insert(
            "attention_mask".into(),
            Tensor::from_array(([1, total_len], vec![1_i64; total_len]))?.into_dyn(),
        );
        let outputs = session.run(inputs)?;
        let logits = outputs
            .get("logits")
            .ok_or_else(|| Plamo3Error::ModelOutput("missing logits".into()))?;
        let (shape, data) = logits.try_extract_tensor::<f32>()?;
        if shape.last().copied() != Some(VOCAB_SIZE as i64) {
            return Err(Plamo3Error::ModelOutput(
                "unexpected logits vocabulary size".into(),
            ));
        }
        let next = next_token(data)?;
        if EOS_TOKEN_IDS.contains(&next) {
            let result = tokenizer
                .decode(&generated, true)
                .map_err(|error| Plamo3Error::Tokenizer(error.to_string()))?;
            if result.trim().is_empty() {
                return Err(Plamo3Error::ModelOutput("empty translation".into()));
            }
            return Ok(result.trim().to_owned());
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
            return Err(Plamo3Error::ModelOutput("incomplete KV cache".into()));
        }
    }
    Err(Plamo3Error::GenerationLimit(options.max_new_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_inputs_fail_before_model_download() {
        let options = Plamo3Options::default();
        for text in ["", " \n\t", "<|plamo:tag|>output", "text <|plamo:bos|>"] {
            assert!(matches!(
                plamo3_translate(text, &options),
                Err(Plamo3Error::InvalidInput(_))
            ));
        }
        for limit in [0, CONTEXT_LENGTH, usize::MAX] {
            let options = Plamo3Options {
                max_new_tokens: limit,
                ..Default::default()
            };
            assert!(matches!(
                plamo3_translate("Hello", &options),
                Err(Plamo3Error::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn prompt_matches_pfml3_without_extra_whitespace() {
        assert_eq!(
            translation_prompt(" A cat.\n"),
            concat!(
                "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation",
                "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|> A cat.\n",
                "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
            )
        );
    }

    #[test]
    fn argmax_uses_last_position_and_first_tie() {
        let mut logits = vec![0.0; VOCAB_SIZE * 2];
        logits[5] = 99.0;
        logits[VOCAB_SIZE + 3] = 4.0;
        logits[VOCAB_SIZE + 4] = 4.0;
        assert_eq!(next_token(&logits).unwrap(), 3);
        logits[VOCAB_SIZE] = f32::NAN;
        assert!(next_token(&logits).is_err());
        assert!(next_token(&[0.0]).is_err());
    }

    #[test]
    fn local_assets_require_files() {
        let root = tempfile::tempdir().unwrap();
        let options = Plamo3Options {
            model_dir: Some(root.path().to_owned()),
            ..Default::default()
        };
        std::fs::create_dir(root.path().join("tokenizer.json")).unwrap();
        assert!(options.asset("tokenizer.json").is_err());
        std::fs::write(root.path().join("model.onnx"), b"test").unwrap();
        assert_eq!(
            options.asset("model.onnx").unwrap(),
            root.path().join("model.onnx")
        );
    }
}
