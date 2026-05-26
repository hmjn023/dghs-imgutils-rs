//! セーフ（不適切画像）判定モデル。
//!
//! Python の `safe.py` に相当します。
//! モデル: `mf666/shit-checker` → `mobilenet.xs.v2.onnx`
//!
//! 特殊な実装（ClassifyModel 非対応）:
//! - 入力: `float32[1, 3, 384, 384]`、正規化 `(x - 0.5) / 0.5`
//! - 出力ラベル: `polluted`, `safe`

use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{InferenceError, create_onnx_session};
use image::imageops::FilterType;
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

const SAFE_REPO: &str = "mf666/shit-checker";
const SAFE_MODEL: &str = "mobilenet.xs.v2";
const SAFE_SIZE: u32 = 384;
const SAFE_LABELS: &[&str] = &["polluted", "safe"];

struct SafeModel {
    session: Session,
}

static SAFE_CACHE: Lazy<Mutex<Option<SafeModel>>> = Lazy::new(|| Mutex::new(None));

fn ensure_safe_model(model_name: &str) -> Result<(), InferenceError> {
    let mut cache = SAFE_CACHE.lock().unwrap();
    if cache.is_some() {
        return Ok(());
    }

    let filename = format!("{}.onnx", model_name);
    let model_path = hf_hub_download(SAFE_REPO, &filename, None, None)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let session = create_onnx_session(&model_path)?;

    *cache = Some(SafeModel { session });
    Ok(())
}

fn preprocess_safe_image(path: &str) -> Result<Array4<f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let resized = rgb.resize_exact(SAFE_SIZE, SAFE_SIZE, FilterType::Triangle);
    let rgb8 = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, SAFE_SIZE as usize, SAFE_SIZE as usize));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        // normalize: (x - 0.5) / 0.5 = x * 2.0 - 1.0
        tensor[[0, 0, y as usize, x as usize]] = r * 2.0 - 1.0;
        tensor[[0, 1, y as usize, x as usize]] = g * 2.0 - 1.0;
        tensor[[0, 2, y as usize, x as usize]] = b * 2.0 - 1.0;
    }
    Ok(tensor)
}

/// セーフスコアマップを返します。
///
/// ラベル: `polluted`, `safe`
pub fn safe_check_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let model = model_name.unwrap_or(SAFE_MODEL);
    ensure_safe_model(model)?;

    let tensor = preprocess_safe_image(path)?;
    let mut cache = SAFE_CACHE.lock().unwrap();
    let entry = cache.as_mut().unwrap();

    let outputs = entry
        .session
        .run(ort::inputs!["input" => ort::value::Tensor::from_array(tensor)?])
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;

    let (_, output_slice) = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let probs: Vec<f32> = output_slice.to_vec();

    let result: HashMap<String, f32> = SAFE_LABELS
        .iter()
        .zip(probs.iter())
        .map(|(&label, &prob)| (label.to_string(), prob))
        .collect();

    Ok(result)
}

/// セーフ判定の最上位クラスと全スコアを返します。
pub fn safe_check(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let scores = safe_check_score(path, model_name)?;
    let top = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| k.clone())
        .ok_or_else(|| InferenceError::InvalidShape("Empty safe output".to_string()))?;
    Ok((top, scores))
}
