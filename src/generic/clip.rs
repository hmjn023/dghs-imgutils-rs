use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::{Array2, Array4};
use once_cell::sync::Lazy;
use ort::session::Session;

use crate::generic::GenericError;
use crate::hub::hf_hub_download;
use crate::image::{force_image_background, to_ndarray_chw};
use crate::inference::create_onnx_session;

const DEFAULT_REPO_ID: &str = "deepghs/clip_onnx";

struct ClipModelEntry {
    image_encoder: Session,
    text_encoder: Session,
    logit_scale: f32,
    input_size: u32,
}

static CLIP_CACHE: Lazy<Mutex<HashMap<String, ClipModelEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn ensure_clip_model(repo_id: &str, model_name: &str) -> Result<(), GenericError> {
    let key = format!("{}/{}", repo_id, model_name);
    {
        let cache = CLIP_CACHE.lock().unwrap();
        if cache.contains_key(&key) {
            return Ok(());
        }
    }

    let image_encoder_path = hf_hub_download(
        repo_id,
        &format!("{}/image_encode.onnx", model_name),
        None,
        None,
    )?;
    let text_encoder_path = hf_hub_download(
        repo_id,
        &format!("{}/text_encode.onnx", model_name),
        None,
        None,
    )?;
    let meta_path = hf_hub_download(repo_id, &format!("{}/meta.json", model_name), None, None)?;

    let meta_str = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
    let logit_scale = meta["logit_scale"].as_f64().unwrap_or(4.605170185988091) as f32;
    let input_size = meta["input_size"].as_u64().unwrap_or(224) as u32;

    let image_encoder = create_onnx_session(&image_encoder_path)?;
    let text_encoder = create_onnx_session(&text_encoder_path)?;

    let entry = ClipModelEntry {
        image_encoder,
        text_encoder,
        logit_scale,
        input_size,
    };

    CLIP_CACHE.lock().unwrap().insert(key, entry);
    Ok(())
}

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

/// Preprocess image for CLIP: white background → resize → normalize → [1, 3, H, W]
fn preprocess_clip_image(image: &DynamicImage, size: u32) -> Result<Array4<f32>, GenericError> {
    let rgb = force_image_background(image, [255, 255, 255]);
    let resized = rgb.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let tensor = to_ndarray_chw(&resized, &CLIP_MEAN, &CLIP_STD)?;
    Ok(tensor)
}

/// Encode image(s) into CLIP embeddings.
/// Returns (embeddings, encodings) where embeddings is L2-normalized.
pub fn clip_image_encode(
    images: &[DynamicImage],
    repo_id: &str,
    model_name: &str,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    ensure_clip_model(repo, model_name)?;

    let key = format!("{}/{}", repo, model_name);
    let mut cache = CLIP_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let n = images.len();
    let size = entry.input_size;
    let mut batch = Array4::<f32>::zeros((n, 3, size as usize, size as usize));
    for (i, img) in images.iter().enumerate() {
        let tensor = preprocess_clip_image(img, size)?;
        batch
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&tensor.slice(ndarray::s![0, .., .., ..]));
    }

    let outputs = entry
        .image_encoder
        .run(ort::inputs!["pixel_values" => ort::value::Tensor::from_array(batch)?])?;

    let extract = |name: &str| -> Result<Vec<Vec<f32>>, GenericError> {
        let val = outputs
            .get(name)
            .ok_or_else(|| GenericError::InvalidArgument(format!("Missing output: {}", name)))?;
        let (shape, data) = val.try_extract_tensor::<f32>()?;
        let dim = shape[1] as usize;
        let raw: Vec<f32> = data.to_vec();
        let chunks: Vec<Vec<f32>> = raw.chunks(dim).map(|c| c.to_vec()).collect();
        Ok(chunks)
    };

    let embeddings = extract("embeddings")?;
    let encodings = extract("encodings")?;
    Ok((embeddings, encodings))
}

/// A simplified text tokenizer for CLIP-like models.
/// In production, use a proper BPE tokenizer; this handles basic ASCII words via vocab lookup
/// with a byte-level fallback.
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
    pub max_len: usize,
}

impl SimpleTokenizer {
    pub fn from_tokenizer_json(path: &PathBuf) -> Result<Self, GenericError> {
        let content = std::fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&content)?;
        let vocab_obj = &root["model"]["vocab"];
        let mut vocab = HashMap::new();
        if let Some(obj) = vocab_obj.as_object() {
            for (token, id_val) in obj {
                let id = id_val.as_u64().unwrap_or(0) as u32;
                vocab.insert(token.clone(), id);
            }
        }
        let bos_id = root["add_prefix_space"].as_bool().map_or(49406, |_| 49406);
        let eos_id = root["added_tokens"]
            .as_array()
            .and_then(|arr| arr.iter().find(|t| t["content"] == "</s>"))
            .and_then(|t| t["id"].as_u64())
            .unwrap_or(49407) as u32;
        let pad_id = 0;
        Ok(Self {
            vocab,
            bos_id,
            eos_id,
            pad_id,
            max_len: 77,
        })
    }

    pub fn encode(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(self.bos_id as i64);
        for w in words {
            if ids.len() >= self.max_len - 1 {
                break;
            }
            let w_lower = w.to_lowercase();
            if let Some(&id) = self.vocab.get(&w_lower) {
                ids.push(id as i64);
            } else if let Some(&id) = self.vocab.get(w) {
                ids.push(id as i64);
            } else {
                for &b in w.as_bytes() {
                    if ids.len() >= self.max_len - 1 {
                        break;
                    }
                    let byte_token = format!("<{}>", b);
                    if let Some(&id) = self.vocab.get(&byte_token) {
                        ids.push(id as i64);
                    } else {
                        ids.push(b as i64 + 3);
                    }
                }
            }
        }
        ids.push(self.eos_id as i64);
        while ids.len() < self.max_len {
            ids.push(self.pad_id as i64);
        }
        let attention_mask: Vec<i64> = ids
            .iter()
            .map(|&id| if id == self.pad_id as i64 { 0 } else { 1 })
            .collect();
        (ids, attention_mask)
    }

    pub fn encode_batch(&self, texts: &[String]) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();
        for t in texts {
            let (ids, mask) = self.encode(t);
            all_ids.push(ids);
            all_masks.push(mask);
        }
        (all_ids, all_masks)
    }
}

fn get_tokenizer(repo_id: &str, model_name: &str) -> Result<SimpleTokenizer, GenericError> {
    let path = hf_hub_download(
        repo_id,
        &format!("{}/tokenizer.json", model_name),
        None,
        None,
    )?;
    SimpleTokenizer::from_tokenizer_json(&path)
}

/// Encode text(s) into CLIP embeddings.
pub fn clip_text_encode(
    texts: &[String],
    repo_id: &str,
    model_name: &str,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    ensure_clip_model(repo, model_name)?;

    let key = format!("{}/{}", repo, model_name);
    let mut cache = CLIP_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let tokenizer = get_tokenizer(repo, model_name)?;
    let (all_ids, all_masks) = tokenizer.encode_batch(texts);
    let n = texts.len();

    let mut input_ids = Array2::<i64>::zeros((n, tokenizer.max_len));
    let mut attention_mask = Array2::<i64>::zeros((n, tokenizer.max_len));
    for i in 0..n {
        for j in 0..tokenizer.max_len {
            input_ids[[i, j]] = all_ids[i][j];
            attention_mask[[i, j]] = all_masks[i][j];
        }
    }

    let outputs = entry.text_encoder.run(ort::inputs![
        "input_ids" => ort::value::Tensor::from_array(input_ids)?,
        "attention_mask" => ort::value::Tensor::from_array(attention_mask)?,
    ])?;

    let extract = |name: &str| -> Result<Vec<Vec<f32>>, GenericError> {
        let val = outputs
            .get(name)
            .ok_or_else(|| GenericError::InvalidArgument(format!("Missing output: {}", name)))?;
        let (shape, data) = val.try_extract_tensor::<f32>()?;
        let dim = shape[1] as usize;
        let raw: Vec<f32> = data.to_vec();
        Ok(raw.chunks(dim).map(|c| c.to_vec()).collect())
    };

    let embeddings = extract("embeddings")?;
    let encodings = extract("encodings")?;
    Ok((embeddings, encodings))
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

/// Compute cosine similarity matrix between image and text embeddings.
/// Returns (similarities, logits, predictions) where predictions = softmax(logits).
pub fn clip_predict(
    image_embeds: &[Vec<f32>],
    text_embeds: &[Vec<f32>],
    logit_scale: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_img = image_embeds.len();
    let n_txt = text_embeds.len();
    let mut similarities = vec![vec![0.0f32; n_txt]; n_img];
    let img_norm: Vec<Vec<f32>> = image_embeds.iter().map(|e| l2_normalize(e)).collect();
    let txt_norm: Vec<Vec<f32>> = text_embeds.iter().map(|e| l2_normalize(e)).collect();
    for i in 0..n_img {
        for j in 0..n_txt {
            let sim: f32 = img_norm[i]
                .iter()
                .zip(txt_norm[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            similarities[i][j] = sim;
        }
    }
    let scale = logit_scale.exp();
    let mut logits = vec![vec![0.0f32; n_txt]; n_img];
    let mut predictions = vec![vec![0.0f32; n_txt]; n_img];
    for i in 0..n_img {
        for j in 0..n_txt {
            logits[i][j] = similarities[i][j] * scale;
        }
        let max_logit = logits[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits[i].iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for j in 0..n_txt {
            predictions[i][j] = exps[j] / sum;
        }
    }
    (similarities, logits, predictions)
}

/// Get the logit_scale from model metadata.
pub fn clip_logit_scale(repo_id: &str, model_name: &str) -> Result<f32, GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    let key = format!("{}/{}", repo, model_name);
    {
        let cache = CLIP_CACHE.lock().unwrap();
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.logit_scale);
        }
    }
    let meta_path = hf_hub_download(repo, &format!("{}/meta.json", model_name), None, None)?;
    let meta_str = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
    Ok(meta["logit_scale"].as_f64().unwrap_or(4.605170185988091) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let data = r#"{
            "model": {
                "vocab": {"hello": 100, "world": 200, "</s>": 2},
                "merges": []
            },
            "add_prefix_space": false,
            "added_tokens": [{"content": "</s>", "id": 2}]
        }"#;
        let dir = std::env::temp_dir().join("clip_test_tokenizer.json");
        std::fs::write(&dir, data).unwrap();
        let tok = SimpleTokenizer::from_tokenizer_json(&dir).unwrap();
        let (ids, mask) = tok.encode("hello world");
        assert_eq!(ids.len(), 77);
        assert_eq!(mask.len(), 77);
        assert!(ids[0] > 0);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }
}
