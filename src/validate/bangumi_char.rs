//! バングミキャラクター種類分類モデル。
//!
//! Python の `bangumi_char.py` に相当します。
//! ClassifyModel パターンを使用します。
//! モデル: `deepghs/bangumi_char_type`

use crate::image::force_image_background;
use crate::inference::InferenceError;
use crate::inference::classify::{classify_predict, classify_top1};
use std::collections::HashMap;

const BANGUMI_CHAR_REPO: &str = "deepghs/bangumi_char_type";
const BANGUMI_CHAR_MODEL: &str = "mobilenetv3_v0_dist";

/// バングミキャラクター種類スコアマップを返します。
///
/// ラベル: `vision`, `imagery`, `halfbody`, `face`
pub fn anime_bangumi_char_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(
        &rgb,
        BANGUMI_CHAR_REPO,
        model_name.unwrap_or(BANGUMI_CHAR_MODEL),
    )
}

/// バングミキャラクター種類判定の最上位クラスと全スコアを返します。
pub fn anime_bangumi_char(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(
        &rgb,
        BANGUMI_CHAR_REPO,
        model_name.unwrap_or(BANGUMI_CHAR_MODEL),
    )
}
