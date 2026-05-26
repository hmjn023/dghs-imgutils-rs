use std::collections::HashMap;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use serde::Deserialize;

use crate::generic::GenericError;
use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{InferenceError, create_onnx_session};

/// A cached TIMM model entry shared between classify and multilabel variants.
struct TIMMEntry {
    session: Session,
    labels: Vec<String>,
    preprocess: PreprocessConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct PreprocessConfig {
    #[serde(default)]
    test: Vec<PreprocessStage>,
    #[serde(default)]
    val: Vec<PreprocessStage>,
}

#[derive(Debug, Clone, Deserialize)]
struct PreprocessStage {
    #[serde(rename = "type")]
    stage_type: String,
    #[serde(default)]
    size: Option<u32>,
    #[serde(default)]
    mean: Option<Vec<f32>>,
    #[serde(default)]
    std: Option<Vec<f32>>,
}

static TIMM_CACHE: Lazy<Mutex<HashMap<String, TIMMEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn ensure_timm_model(repo_id: &str) -> Result<(), GenericError> {
    let cache = TIMM_CACHE.lock().unwrap();
    if cache.contains_key(repo_id) {
        return Ok(());
    }
    drop(cache);

    let model_path = hf_hub_download(repo_id, "model.onnx", None, None)?;
    let tags_path = hf_hub_download(repo_id, "selected_tags.csv", None, None)?;

    // Parse labels from CSV (first column = tag name)
    let mut labels = Vec::new();
    let mut reader = csv::Reader::from_path(&tags_path)?;
    for result in reader.records() {
        let record = result?;
        if let Some(name) = record.get(0) {
            labels.push(name.to_string());
        }
    }

    let session = create_onnx_session(&model_path)?;

    // Try to load preprocess.json
    let preprocess = match hf_hub_download(repo_id, "preprocess.json", None, None) {
        Ok(p) => {
            let content = std::fs::read_to_string(p)?;
            serde_json::from_str(&content).unwrap_or(PreprocessConfig {
                test: vec![],
                val: vec![],
            })
        }
        Err(_) => PreprocessConfig {
            test: vec![],
            val: vec![],
        },
    };

    let entry = TIMMEntry {
        session,
        labels,
        preprocess,
    };

    TIMM_CACHE
        .lock()
        .unwrap()
        .insert(repo_id.to_string(), entry);
    Ok(())
}

/// Apply basic preprocessing: resize to target size, normalize with mean/std.
fn apply_preprocess(
    image: &DynamicImage,
    stages: &[PreprocessStage],
) -> Result<Array4<f32>, GenericError> {
    let rgb = force_image_background(image, [255, 255, 255]);
    let img = rgb;

    let mut target_size = 224u32;
    let mut mean = vec![0.5f32, 0.5f32, 0.5f32];
    let mut std_dev = vec![0.5f32, 0.5f32, 0.5f32];

    for stage in stages {
        match stage.stage_type.as_str() {
            "resize" | "Resize" => {
                if let Some(sz) = stage.size {
                    target_size = sz;
                }
            }
            _ => {}
        }
    }

    // Extract mean/std from first stage that has them, or from "normalize" stage
    for stage in stages {
        if stage.stage_type == "normalize" || stage.stage_type == "Normalize" {
            if let Some(ref m) = stage.mean {
                mean = m.clone();
            }
            if let Some(ref s) = stage.std {
                std_dev = s.clone();
            }
        }
    }

    let resized = img.resize_exact(
        target_size,
        target_size,
        image::imageops::FilterType::Triangle,
    );
    let rgb8 = resized.to_rgb8();
    let (w, h) = rgb8.dimensions();

    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    let m0 = *mean.first().unwrap_or(&0.5);
    let m1 = *mean.get(1).unwrap_or(&0.5);
    let m2 = *mean.get(2).unwrap_or(&0.5);
    let s0 = *std_dev.first().unwrap_or(&0.5);
    let s1 = *std_dev.get(1).unwrap_or(&0.5);
    let s2 = *std_dev.get(2).unwrap_or(&0.5);

    for (x, y, pixel) in rgb8.enumerate_pixels() {
        tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - m0) / s0;
        tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - m1) / s1;
        tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - m2) / s2;
    }

    Ok(tensor)
}

/// Classification result: label → confidence
pub type ClassifyResult = HashMap<String, f32>;

/// Run a single-image classification prediction using a TIMM model.
pub fn classify_timm_predict(
    image: &DynamicImage,
    repo_id: &str,
    topk: Option<usize>,
) -> Result<ClassifyResult, GenericError> {
    ensure_timm_model(repo_id)?;

    let mut cache = TIMM_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(repo_id)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let tensor = apply_preprocess(image, &entry.preprocess.test)?;

    let outputs = entry.session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(tensor)?
    ])?;

    // Try "prediction" output, fall back to "output"
    let pred_val = outputs
        .get("prediction")
        .or_else(|| outputs.get("output"))
        .ok_or_else(|| InferenceError::InvalidShape("No prediction output found".to_string()))?;

    let (_shape, data) = pred_val.try_extract_tensor::<f32>()?;
    let scores: Vec<f32> = data.to_vec();

    let k = topk.unwrap_or(scores.len());
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = HashMap::new();
    for &idx in indices.iter().take(k.min(scores.len())) {
        if idx < entry.labels.len() {
            result.insert(entry.labels[idx].clone(), scores[idx]);
        }
    }

    Ok(result)
}

/// Category-based multi-label classification result.
#[derive(Debug, Clone)]
pub struct MultiLabelResult {
    /// Per-category results
    pub categories: HashMap<String, HashMap<String, f32>>,
    /// Flattened all tags across categories
    pub tags: HashMap<String, f32>,
}

/// Run a multi-label TIMM prediction.
pub fn multilabel_timm_predict(
    image: &DynamicImage,
    repo_id: &str,
    thresholds: Option<HashMap<String, f32>>,
) -> Result<MultiLabelResult, GenericError> {
    ensure_timm_model(repo_id)?;

    let mut cache = TIMM_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(repo_id)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let tensor = apply_preprocess(image, &entry.preprocess.test)?;

    let outputs = entry.session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(tensor)?
    ])?;

    let pred_val = outputs
        .get("prediction")
        .ok_or_else(|| InferenceError::InvalidShape("No prediction output found".to_string()))?;

    let (_shape, data) = pred_val.try_extract_tensor::<f32>()?;
    let scores: Vec<f32> = data.to_vec();

    let default_threshold = 0.4;
    let mut categories: HashMap<String, HashMap<String, f32>> = HashMap::new();
    let mut tags: HashMap<String, f32> = HashMap::new();

    for (i, label) in entry.labels.iter().enumerate() {
        if i >= scores.len() {
            break;
        }
        let score = scores[i];
        let threshold = thresholds
            .as_ref()
            .and_then(|t| t.get(label))
            .copied()
            .unwrap_or(default_threshold);

        if score >= threshold {
            tags.insert(label.clone(), score);
            // If label has a category prefix (e.g., "general:1girl"), split it
            if let Some(pos) = label.find(':') {
                let cat = label[..pos].to_string();
                let name = label[pos + 1..].to_string();
                categories.entry(cat).or_default().insert(name, score);
            } else {
                categories
                    .entry("general".to_string())
                    .or_default()
                    .insert(label.clone(), score);
            }
        }
    }

    Ok(MultiLabelResult { categories, tags })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn make_test_image() -> DynamicImage {
        let mut img = RgbImage::new(224, 224);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 128, 128]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_classify_timm_no_model() {
        let img = make_test_image();
        let result = classify_timm_predict(&img, "nonexistent/repo", Some(5));
        assert!(result.is_err());
    }
}
