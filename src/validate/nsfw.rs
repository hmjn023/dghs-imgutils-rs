//! NSFW 判定モデル（特殊実装）。
//!
//! Python 版の `nsfw.py` は他の validate モジュールと異なる特殊な構造を持ちます:
//! - 入力形式: `float32[1, H, W, 3]`（HWC、他モジュールは CHW）
//! - 入力名: `input_1`（他は `input`）
//! - 出力名: `dense_3`（他は `output`）
//! - リサイズ: NEAREST（他は Bilinear）
//! - モデル: `deepghs/imgutils-models` の `nsfw/nsfwjs.onnx`（224px）または `nsfw/inception_v3.onnx`（299px）

use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{InferenceError, create_onnx_session};
use image::imageops::FilterType;
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

const NSFW_REPO: &str = "deepghs/imgutils-models";
const NSFW_LABELS: &[&str] = &["drawings", "hentai", "neutral", "porn", "sexy"];

struct NsfwModel {
    session: Session,
    size: u32,
}

static NSFW_CACHE: Lazy<Mutex<HashMap<String, NsfwModel>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn ensure_nsfw_model(model_name: &str) -> Result<(), InferenceError> {
    if NSFW_CACHE.lock().unwrap().contains_key(model_name) {
        return Ok(());
    }

    let (filename, size) = match model_name {
        "nsfwjs" | "" => ("nsfw/nsfwjs.onnx", 224u32),
        "inception_v3" => ("nsfw/inception_v3.onnx", 299u32),
        _ => ("nsfw/nsfwjs.onnx", 224u32),
    };

    let model_path = hf_hub_download(NSFW_REPO, filename, None, None)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let session = create_onnx_session(&model_path)?;

    NSFW_CACHE
        .lock()
        .unwrap()
        .insert(model_name.to_string(), NsfwModel { session, size });
    Ok(())
}

/// NSFW スコアマップを返します。
///
/// ラベル: `drawings`, `hentai`, `neutral`, `porn`, `sexy`
pub fn nsfw_pred_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let model = model_name.unwrap_or("nsfwjs");
    ensure_nsfw_model(model)?;

    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);

    let mut cache = NSFW_CACHE.lock().unwrap();
    let entry = cache.get_mut(model).unwrap();
    let size = entry.size;

    // NEAREST リサイズ（nsfw.py と同一）
    let resized = rgb.resize_exact(size, size, FilterType::Nearest);
    let rgb8 = resized.to_rgb8();

    // HWC テンソル: float32[1, H, W, 3]（他モジュールと異なる！）
    let mut tensor = Array4::<f32>::zeros((1, size as usize, size as usize, 3));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        tensor[[0, y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
        tensor[[0, y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
        tensor[[0, y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
    }

    let outputs = entry
        .session
        .run(ort::inputs!["input_1" => ort::value::Tensor::from_array(tensor.clone())?])
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;

    let (_, output_data) = outputs["dense_3"]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let probs: Vec<f32> = output_data.to_vec();

    let result: HashMap<String, f32> = NSFW_LABELS
        .iter()
        .zip(probs.iter())
        .map(|(&label, &prob)| (label.to_string(), prob))
        .collect();

    Ok(result)
}

/// NSFW 判定の最上位クラスと全スコアを返します。
pub fn nsfw_pred(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let scores = nsfw_pred_score(path, model_name)?;
    let top = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| k.clone())
        .ok_or_else(|| InferenceError::InvalidShape("Empty NSFW output".to_string()))?;
    Ok((top, scores))
}
