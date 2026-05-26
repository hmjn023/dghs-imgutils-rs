use std::collections::HashMap;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::{Array2, Array4};
use once_cell::sync::Lazy;
use ort::session::Session;

use crate::generic::GenericError;
use crate::generic::clip::SimpleTokenizer;
use crate::hub::hf_hub_download;
use crate::image::{force_image_background, to_ndarray_chw};
use crate::inference::create_onnx_session;

const DEFAULT_REPO_ID: &str = "deepghs/siglip_onnx";

struct SigLIPModelEntry {
    image_encoder: Session,
    text_encoder: Session,
    logit_scale: f32,
    logit_bias: f32,
    input_size: u32,
}

static SIGLIP_CACHE: Lazy<Mutex<HashMap<String, SigLIPModelEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn ensure_siglip_model(repo_id: &str, model_name: &str) -> Result<(), GenericError> {
    let key = format!("{}/{}", repo_id, model_name);
    {
        let cache = SIGLIP_CACHE.lock().unwrap();
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
    let logit_bias = meta["logit_bias"].as_f64().unwrap_or(0.0) as f32;
    let input_size = meta["input_size"].as_u64().unwrap_or(224) as u32;

    let image_encoder = create_onnx_session(&image_encoder_path)?;
    let text_encoder = create_onnx_session(&text_encoder_path)?;

    let entry = SigLIPModelEntry {
        image_encoder,
        text_encoder,
        logit_scale,
        logit_bias,
        input_size,
    };

    SIGLIP_CACHE.lock().unwrap().insert(key, entry);
    Ok(())
}

const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

fn preprocess_siglip_image(image: &DynamicImage, size: u32) -> Result<Array4<f32>, GenericError> {
    let rgb = force_image_background(image, [255, 255, 255]);
    let resized = rgb.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let tensor = to_ndarray_chw(&resized, &SIGLIP_MEAN, &SIGLIP_STD)?;
    Ok(tensor)
}

/// Encode image(s) into SigLIP embeddings.
pub fn siglip_image_encode(
    images: &[DynamicImage],
    repo_id: &str,
    model_name: &str,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    ensure_siglip_model(repo, model_name)?;

    let key = format!("{}/{}", repo, model_name);
    let mut cache = SIGLIP_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let n = images.len();
    let size = entry.input_size;
    let mut batch = Array4::<f32>::zeros((n, 3, size as usize, size as usize));
    for (i, img) in images.iter().enumerate() {
        let tensor = preprocess_siglip_image(img, size)?;
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
        Ok(raw.chunks(dim).map(|c| c.to_vec()).collect())
    };

    let embeddings = extract("embeddings")?;
    let encodings = extract("encodings")?;
    Ok((embeddings, encodings))
}

/// Encode text(s) into SigLIP embeddings.
pub fn siglip_text_encode(
    texts: &[String],
    repo_id: &str,
    model_name: &str,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    ensure_siglip_model(repo, model_name)?;

    let key = format!("{}/{}", repo, model_name);
    let mut cache = SIGLIP_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let tok_path = hf_hub_download(repo, &format!("{}/tokenizer.json", model_name), None, None)?;
    let tokenizer = SimpleTokenizer::from_tokenizer_json(&tok_path)?;
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

/// Compute SigLIP similarity, logits, and predictions between image and text embeddings.
/// Predictions use sigmoid(logits) instead of softmax (distinct from CLIP).
pub fn siglip_predict(
    image_embeds: &[Vec<f32>],
    text_embeds: &[Vec<f32>],
    logit_scale: f32,
    logit_bias: f32,
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
            logits[i][j] = similarities[i][j] * scale + logit_bias;
            predictions[i][j] = sigmoid(logits[i][j]);
        }
    }
    (similarities, logits, predictions)
}

/// Get logit scale and bias from model metadata.
pub fn siglip_logit_params(repo_id: &str, model_name: &str) -> Result<(f32, f32), GenericError> {
    let repo = if repo_id.is_empty() {
        DEFAULT_REPO_ID
    } else {
        repo_id
    };
    let key = format!("{}/{}", repo, model_name);
    {
        let cache = SIGLIP_CACHE.lock().unwrap();
        if let Some(entry) = cache.get(&key) {
            return Ok((entry.logit_scale, entry.logit_bias));
        }
    }
    let meta_path = hf_hub_download(repo, &format!("{}/meta.json", model_name), None, None)?;
    let meta_str = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
    let logit_scale = meta["logit_scale"].as_f64().unwrap_or(4.605170185988091) as f32;
    let logit_bias = meta["logit_bias"].as_f64().unwrap_or(0.0) as f32;
    Ok((logit_scale, logit_bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
    }
}
