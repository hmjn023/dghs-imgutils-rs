//! Danbooru レーティング判定モデル。
//!
//! Python の `dbrating.py` に相当します。
//! ClassifyModel パターンを使用します。
//! モデル: `deepghs/anime_dbrating`

use crate::image::force_image_background;
use crate::inference::InferenceError;
use crate::inference::classify::{classify_predict, classify_top1};
use std::collections::HashMap;

const DBRATING_REPO: &str = "deepghs/anime_dbrating";
const DBRATING_MODEL: &str = "mobilenetv3_large_100_v0_ls0.2";

/// Danbooru レーティングスコアマップを返します。
///
/// ラベル: `general`, `sensitive`, `questionable`, `explicit`
pub fn anime_dbrating_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, DBRATING_REPO, model_name.unwrap_or(DBRATING_MODEL))
}

/// Danbooru レーティング判定の最上位クラスと全スコアを返します。
pub fn anime_dbrating(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, DBRATING_REPO, model_name.unwrap_or(DBRATING_MODEL))
}
