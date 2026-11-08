//! Use the exported tokenizer, including its special tokens and padding policy.
use super::GenericError;
use ndarray::Array2;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokenizers::{Tokenizer, TruncationParams};

type TokenizerCache = HashMap<(String, String, usize), Arc<Tokenizer>>;
static TOKENIZERS: Lazy<Mutex<TokenizerCache>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Lazily loads a text tokenizer once per repository, model and context limit.
pub(crate) fn encode(
    repo: &str,
    model: &str,
    texts: &[String],
    max_length: usize,
) -> Result<(Array2<i64>, Array2<i64>), GenericError> {
    let key = (repo.to_owned(), model.to_owned(), max_length);
    let tokenizer = {
        let mut cache = TOKENIZERS
            .lock()
            .map_err(|e| GenericError::InvalidArgument(e.to_string()))?;
        if let Some(tokenizer) = cache.get(&key) {
            Arc::clone(tokenizer)
        } else {
            let path =
                crate::hub::hf_hub_download(repo, &format!("{model}/tokenizer.json"), None, None)?;
            let mut tokenizer = Tokenizer::from_file(path)
                .map_err(|e| GenericError::InvalidArgument(format!("invalid tokenizer: {e}")))?;
            configure_context(&mut tokenizer, max_length)?;
            let tokenizer = Arc::new(tokenizer);
            cache.insert(key, Arc::clone(&tokenizer));
            tokenizer
        }
    };
    encode_with(&tokenizer, texts)
}

/// Limits tokens including special tokens without changing exported padding.
fn configure_context(tokenizer: &mut Tokenizer, max_length: usize) -> Result<(), GenericError> {
    if max_length == 0 {
        return Err(GenericError::InvalidArgument(
            "text context must be positive".into(),
        ));
    }
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| GenericError::InvalidArgument(format!("invalid text context: {e}")))?;
    Ok(())
}

/// Encodes padded IDs and attention masks with the configured tokenizer.
fn encode_with(
    tokenizer: &Tokenizer,
    texts: &[String],
) -> Result<(Array2<i64>, Array2<i64>), GenericError> {
    let encoded = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| GenericError::InvalidArgument(format!("tokenization failed: {e}")))?;
    let width = encoded.first().map_or(0, |v| v.len());
    if encoded.iter().any(|v| v.len() != width) {
        return Err(GenericError::Shape(
            "tokenizer must pad each batch to a common length".into(),
        ));
    }
    let ids = encoded
        .iter()
        .flat_map(|v| v.get_ids().iter().map(|&x| i64::from(x)))
        .collect();
    let mask = encoded
        .iter()
        .flat_map(|v| v.get_attention_mask().iter().map(|&x| i64::from(x)))
        .collect();
    Ok((
        Array2::from_shape_vec((texts.len(), width), ids)?,
        Array2::from_shape_vec((texts.len(), width), mask)?,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":{"strategy":{"Fixed":64},"direction":"Right","pad_to_multiple_of":null,"pad_id":1,"pad_type_id":0,"pad_token":"[PAD]"},"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"[PAD]":1,"hello":2},"unk_token":"[UNK]"}}"#;
    #[test]
    fn respects_siglip_length_and_vocabulary() {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER).unwrap();
        let (ids, mask) = encode_with(&tokenizer, &["hello 世界".into()]).unwrap();
        assert_eq!(ids.shape(), [1, 64]);
        assert_eq!(ids[[0, 0]], 2);
        assert_eq!(ids[[0, 1]], 0);
        assert_eq!(ids[[0, 2]], 1);
        assert_eq!(mask[[0, 1]], 1);
        assert_eq!(mask[[0, 2]], 0);
    }
    #[test]
    fn respects_batch_longest_padding() {
        let tokenizer =
            Tokenizer::from_bytes(TOKENIZER.replace(r#"{"Fixed":64}"#, r#""BatchLongest""#))
                .unwrap();
        let (ids, mask) = encode_with(&tokenizer, &["hello".into(), "hello hello".into()]).unwrap();
        assert_eq!(ids.shape(), [2, 2]);
        assert_eq!(mask[[0, 1]], 0);
        assert_eq!(mask[[1, 1]], 1);
    }
    #[test]
    fn long_text_respects_context_and_padding() {
        for (length, json) in [
            (64, TOKENIZER.to_owned()),
            (
                77,
                TOKENIZER.replace(r#"{"Fixed":64}"#, r#""BatchLongest""#),
            ),
        ] {
            let mut tokenizer = Tokenizer::from_bytes(json).unwrap();
            // Include EOS in the limit, as real CLIP/SigLIP tokenizers do.
            tokenizer.with_post_processor(Some(
                tokenizers::processors::template::TemplateProcessing::builder()
                    .try_single("$A [EOS]")
                    .unwrap()
                    .special_tokens(vec![("[EOS]", 3)])
                    .build()
                    .unwrap(),
            ));
            let padding = tokenizer.get_padding().cloned();
            configure_context(&mut tokenizer, length).unwrap();
            assert_eq!(
                serde_json::to_value(tokenizer.get_padding()).unwrap(),
                serde_json::to_value(&padding).unwrap()
            );
            let (ids, mask) =
                encode_with(&tokenizer, &["hello ".repeat(200), "hello".into()]).unwrap();
            assert_eq!(ids.shape(), [2, length]);
            assert_eq!(ids[[0, length - 1]], 3);
            assert_eq!(ids[[1, 1]], 3);
            assert_eq!(mask[[1, 2]], 0);
        }
    }
    #[test]
    fn zero_context_is_rejected() {
        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER).unwrap();
        assert!(configure_context(&mut tokenizer, 0).is_err());
    }
}
