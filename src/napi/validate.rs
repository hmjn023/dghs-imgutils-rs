//! 画像検証 (validate) モジュールの napi-rs バインディング。

use crate::validate::bangumi_char::{
    anime_bangumi_char as core_anime_bangumi_char,
    anime_bangumi_char_score as core_anime_bangumi_char_score,
};
use crate::validate::classify::{
    anime_classify as core_anime_classify, anime_classify_score as core_anime_classify_score,
    anime_completeness as core_anime_completeness,
    anime_completeness_score as core_anime_completeness_score, anime_furry as core_anime_furry,
    anime_furry_score as core_anime_furry_score, anime_portrait as core_anime_portrait,
    anime_portrait_score as core_anime_portrait_score, anime_rating as core_anime_rating,
    anime_rating_score as core_anime_rating_score, anime_real as core_anime_real,
    anime_real_score as core_anime_real_score, anime_style_age as core_anime_style_age,
    anime_style_age_score as core_anime_style_age_score,
    get_ai_created_score as core_get_ai_created_score,
    get_monochrome_score as core_get_monochrome_score, is_ai_created as core_is_ai_created,
    is_monochrome as core_is_monochrome,
};
use crate::validate::color::is_greyscale as core_is_greyscale;
use crate::validate::dbrating::{
    anime_dbrating as core_anime_dbrating, anime_dbrating_score as core_anime_dbrating_score,
};
use crate::validate::nsfw::{nsfw_pred as core_nsfw_pred, nsfw_pred_score as core_nsfw_pred_score};
use crate::validate::safe::{
    safe_check as core_safe_check, safe_check_score as core_safe_check_score,
};
use crate::validate::teen::{
    anime_teen as core_anime_teen, anime_teen_score as core_anime_teen_score,
};
use crate::validate::truncate::is_truncated_file as core_is_truncated_file;
use napi_derive::napi;
use std::collections::HashMap;

/// 分類結果（トップクラス + 全スコア）を JS/TS 側で表現する構造体
#[napi(object)]
pub struct ClassifyTopResult {
    pub label: String,
    pub scores: HashMap<String, f64>,
}

// ─── truncation / greyscale ───

/// ファイルが破損（truncated）しているかを判定します。
#[napi]
pub fn is_truncated_file(path: String) -> bool {
    core_is_truncated_file(&path)
}

/// 画像がグレースケール（モノクロ）かどうかを判定します。
#[napi]
pub fn is_greyscale(path: String) -> bool {
    core_is_greyscale(&path)
}

// ─── AI生成判定 ───

/// AI生成画像のスコアを返します（0.0〜1.0）。
#[napi]
pub fn get_ai_created_score(path: String, model_name: Option<String>) -> napi::Result<f64> {
    let model = model_name.as_deref();
    let score = core_get_ai_created_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("AI created score failed: {}", e),
        )
    })?;
    Ok(score as f64)
}

/// AI生成画像かどうかを判定します。
#[napi]
pub fn is_ai_created(
    path: String,
    model_name: Option<String>,
    threshold: Option<f64>,
) -> napi::Result<bool> {
    let model = model_name.as_deref();
    let th = threshold.unwrap_or(0.5) as f32;
    let result = core_is_ai_created(&path, model, th).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("AI created check failed: {}", e),
        )
    })?;
    Ok(result)
}

// ─── モノクロ判定 ───

/// モノクロ画像のスコアを返します（0.0〜1.0）。
#[napi]
pub fn get_monochrome_score(path: String, model_name: Option<String>) -> napi::Result<f64> {
    let model = model_name.as_deref();
    let score = core_get_monochrome_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Monochrome score failed: {}", e),
        )
    })?;
    Ok(score as f64)
}

/// モノクロ画像かどうかを判定します。
#[napi]
pub fn is_monochrome(
    path: String,
    model_name: Option<String>,
    threshold: Option<f64>,
) -> napi::Result<bool> {
    let model = model_name.as_deref();
    let th = threshold.unwrap_or(0.5) as f32;
    let result = core_is_monochrome(&path, model, th).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Monochrome check failed: {}", e),
        )
    })?;
    Ok(result)
}

// ─── アニメ分類（汎用 classify 系） ───

/// アニメ画像を分類し、全クラスのスコアを返します。
#[napi]
pub fn anime_classify_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_classify_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime classify score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// アニメ画像を分類し、トップクラスとスコアを返します。
#[napi]
pub fn anime_classify(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_classify(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime classify failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// 実写/アニメ判定スコアを返します。
#[napi]
pub fn anime_real_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_real_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime real score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 実写/アニメを判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_real(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_real(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime real failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// アニメレーティングスコアを返します。
#[napi]
pub fn anime_rating_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_rating_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime rating score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// アニメレーティングを判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_rating(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_rating(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime rating failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// 完成度スコアを返します。
#[napi]
pub fn anime_completeness_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_completeness_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime completeness score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 完成度を判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_completeness(
    path: String,
    model_name: Option<String>,
) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_completeness(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime completeness failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// ポートレートスコアを返します。
#[napi]
pub fn anime_portrait_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_portrait_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime portrait score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// ポートレートを判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_portrait(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_portrait(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime portrait failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// ファリー（ケモノ）スコアを返します。
#[napi]
pub fn anime_furry_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_furry_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime furry score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// ファリー（ケモノ）を判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_furry(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_furry(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime furry failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// 画風年代スコアを返します。
#[napi]
pub fn anime_style_age_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_style_age_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime style age score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 画風年代を判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_style_age(
    path: String,
    model_name: Option<String>,
) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_style_age(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Anime style age failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

// ─── NSFW ───

/// NSFW 予測スコアを返します。
#[napi]
pub fn nsfw_pred_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_nsfw_pred_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NSFW prediction failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// NSFW 予測を行い、トップクラスとスコアを返します。
#[napi]
pub fn nsfw_pred(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_nsfw_pred(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NSFW prediction failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

// ─── Safe check ───

/// 指定した画像ファイルのセーフ（不適切画像）スコアを予測します。
#[napi]
pub fn safe_check_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_safe_check_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Safe check failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// セーフチェックを行い、トップクラスとスコアを返します。
#[napi]
pub fn safe_check(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_safe_check(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Safe check failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

// ─── Danbooru rating ───

/// Danbooru レーティングスコアを返します。
#[napi]
pub fn anime_dbrating_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_dbrating_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("DB rating score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// Danbooru レーティングを判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_dbrating(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_dbrating(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("DB rating failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

// ─── Bangumi char ───

/// バングミキャラクター種類スコアを返します。
#[napi]
pub fn anime_bangumi_char_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_bangumi_char_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Bangumi char score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// バングミキャラクター種類を判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_bangumi_char(
    path: String,
    model_name: Option<String>,
) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_bangumi_char(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Bangumi char failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

// ─── Teen ───

/// ティーンレーティングスコアを返します。
#[napi]
pub fn anime_teen_score(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_teen_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Teen score failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// ティーンレーティングを判定し、トップクラスとスコアを返します。
#[napi]
pub fn anime_teen(path: String, model_name: Option<String>) -> napi::Result<ClassifyTopResult> {
    let model = model_name.as_deref();
    let (label, scores) = core_anime_teen(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Teen check failed: {}", e),
        )
    })?;
    Ok(ClassifyTopResult {
        label,
        scores: scores.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}
