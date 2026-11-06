//! Use the exported tokenizer, including its special tokens and padding policy.
use super::GenericError;
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

/// Encodes a batch without substituting CLIP's token IDs or sequence length.
pub(crate) fn encode(
    path: &Path,
    texts: &[String],
) -> Result<(Array2<i64>, Array2<i64>), GenericError> {
    let tokenizer = Tokenizer::from_file(path)
        .map_err(|e| GenericError::InvalidArgument(format!("invalid tokenizer: {e}")))?;
    encode_with(&tokenizer, texts)
}
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
    fn invalid_tokenizer_fails() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert!(encode(f.path(), &["hello".into()]).is_err());
    }
}
