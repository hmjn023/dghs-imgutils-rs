//! ティーン（青少年）レーティング判定モデル。
//!
//! Python の `teen.py` に相当します。
//! ClassifyModel パターンを使用します。
//! モデル: `deepghs/anime_teen`

use crate::image::force_image_background;
use crate::inference::InferenceError;
use crate::inference::classify::{classify_predict, classify_top1};
use std::collections::HashMap;

const TEEN_REPO: &str = "deepghs/anime_teen";
const TEEN_MODEL: &str = "mobilenetv3_v0_dist";

/// ティーンレーティングスコアマップを返します。
///
/// ラベル: `contentious`, `safe_teen`, `non_teen`
pub fn anime_teen_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, TEEN_REPO, model_name.unwrap_or(TEEN_MODEL))
}

/// ティーンレーティング判定の最上位クラスと全スコアを返します。
pub fn anime_teen(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, TEEN_REPO, model_name.unwrap_or(TEEN_MODEL))
}
