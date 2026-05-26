//! 汎用 ClassifyModel を使った各種画像分類器。
//!
//! Python の `validate/*.py` の大半は `generic/classify.py` の `ClassifyModel` を
//! 通じて `{repo_id}/{model_name}/model.onnx` + `meta.json` を使います。
//! ここでは `inference::classify::classify_predict` / `classify_top1` を利用します。

use crate::image::force_image_background;
use crate::inference::classify::{classify_predict, classify_top1};
use crate::inference::InferenceError;
use std::collections::HashMap;

// ========================
//  AI 生成判定
// ========================

const AI_CHECK_REPO: &str = "deepghs/anime_ai_check";
const AI_CHECK_MODEL: &str = "mobilenetv3_sce_dist";

/// AI 生成確信度スコア（0.0〜1.0）を返します。
pub fn get_ai_created_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<f32, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let model = model_name.unwrap_or(AI_CHECK_MODEL);
    let scores = classify_predict(&rgb, AI_CHECK_REPO, model)?;
    Ok(*scores.get("ai").unwrap_or(
        scores.get("AI").unwrap_or(&0.0),
    ))
}

/// 画像が AI 生成かどうかを判定します（しきい値: 0.5）。
pub fn is_ai_created(
    path: &str,
    model_name: Option<&str>,
    threshold: f32,
) -> Result<bool, InferenceError> {
    Ok(get_ai_created_score(path, model_name)? >= threshold)
}

// ========================
//  モノクロ判定
// ========================

const MONO_REPO: &str = "deepghs/monochrome_detect";
const MONO_MODEL: &str = "mobilenetv3_large_100_dist_safe2";

/// モノクロ確信度スコア（0.0〜1.0）を返します。
pub fn get_monochrome_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<f32, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let model = model_name.unwrap_or(MONO_MODEL);
    let scores = classify_predict(&rgb, MONO_REPO, model)?;
    // ラベルは ["monochrome", "colored"] など
    Ok(*scores.get("monochrome").unwrap_or(&0.0))
}

/// 画像がモノクロかどうかを判定します（しきい値: 0.5）。
pub fn is_monochrome(
    path: &str,
    model_name: Option<&str>,
    threshold: f32,
) -> Result<bool, InferenceError> {
    Ok(get_monochrome_score(path, model_name)? >= threshold)
}

// ========================
//  アニメ分類（classify）
// ========================

const CLASSIFY_REPO: &str = "deepghs/anime_classification";
const CLASSIFY_MODEL: &str = "mobilenetv3_v1.5_dist";

pub fn anime_classify_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, CLASSIFY_REPO, model_name.unwrap_or(CLASSIFY_MODEL))
}

pub fn anime_classify(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, CLASSIFY_REPO, model_name.unwrap_or(CLASSIFY_MODEL))
}

// ========================
//  実写 / アニメ判定
// ========================

const REAL_REPO: &str = "deepghs/anime_real_cls";
const REAL_MODEL: &str = "mobilenetv3_v1.4_dist";

pub fn anime_real_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, REAL_REPO, model_name.unwrap_or(REAL_MODEL))
}

pub fn anime_real(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, REAL_REPO, model_name.unwrap_or(REAL_MODEL))
}

// ========================
//  レーティング（safe/r15/r18 など）
// ========================

const RATING_REPO: &str = "deepghs/anime_rating";
const RATING_MODEL: &str = "mobilenetv3_v1_pruned_ls0.1";

pub fn anime_rating_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, RATING_REPO, model_name.unwrap_or(RATING_MODEL))
}

pub fn anime_rating(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, RATING_REPO, model_name.unwrap_or(RATING_MODEL))
}

// ========================
//  完成度判定
// ========================

const COMPLETENESS_REPO: &str = "deepghs/anime_completeness";
const COMPLETENESS_MODEL: &str = "mobilenetv3_v2.2_dist";

pub fn anime_completeness_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, COMPLETENESS_REPO, model_name.unwrap_or(COMPLETENESS_MODEL))
}

pub fn anime_completeness(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, COMPLETENESS_REPO, model_name.unwrap_or(COMPLETENESS_MODEL))
}

// ========================
//  ポートレート判定
// ========================

const PORTRAIT_REPO: &str = "deepghs/anime_portrait";
const PORTRAIT_MODEL: &str = "mobilenetv3_v0_dist";

pub fn anime_portrait_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, PORTRAIT_REPO, model_name.unwrap_or(PORTRAIT_MODEL))
}

pub fn anime_portrait(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, PORTRAIT_REPO, model_name.unwrap_or(PORTRAIT_MODEL))
}

// ========================
//  ケモノ（Furry）判定
// ========================

const FURRY_REPO: &str = "deepghs/anime_furry";
const FURRY_MODEL: &str = "mobilenetv3_v0.1_dist";

pub fn anime_furry_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, FURRY_REPO, model_name.unwrap_or(FURRY_MODEL))
}

pub fn anime_furry(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, FURRY_REPO, model_name.unwrap_or(FURRY_MODEL))
}

// ========================
//  画風・年代判定
// ========================

const STYLE_AGE_REPO: &str = "deepghs/anime_style_ages";
const STYLE_AGE_MODEL: &str = "mobilenetv3_v0_dist";

pub fn anime_style_age_score(
    path: &str,
    model_name: Option<&str>,
) -> Result<HashMap<String, f32>, InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_predict(&rgb, STYLE_AGE_REPO, model_name.unwrap_or(STYLE_AGE_MODEL))
}

pub fn anime_style_age(
    path: &str,
    model_name: Option<&str>,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let img = image::open(path).map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    classify_top1(&rgb, STYLE_AGE_REPO, model_name.unwrap_or(STYLE_AGE_MODEL))
}
