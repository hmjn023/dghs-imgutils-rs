//! 画像検証 (validate) モジュールの napi-rs バインディング。

use crate::validate::bangumi_char::anime_bangumi_char_score as core_anime_bangumi_char_score;
use crate::validate::dbrating::anime_dbrating_score as core_anime_dbrating_score;
use crate::validate::safe::safe_check_score as core_safe_check_score;
use crate::validate::teen::anime_teen_score as core_anime_teen_score;
use napi_derive::napi;
use std::collections::HashMap;

/// 指定した画像ファイルのセーフ（不適切画像）スコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデル名（オプション。省略時は `"mobilenet.xs.v2"`）
#[napi]
pub fn check_safe(path: String, model_name: Option<String>) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_safe_check_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Safe check failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 指定した画像ファイルの Danbooru レーティングスコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデル名（オプション。省略時は `"mobilenetv3_large_100_v0_ls0.2"`）
#[napi]
pub fn check_dbrating(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_dbrating_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("DB rating check failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 指定した画像ファイルのバングミキャラクター種類スコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデル名（オプション。省略時は `"mobilenetv3_v0_dist"`）
#[napi]
pub fn check_bangumi_char(
    path: String,
    model_name: Option<String>,
) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_bangumi_char_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Bangumi char check failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}

/// 指定した画像ファイルのティーンレーティングスコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデル名（オプション。省略時は `"mobilenetv3_v0_dist"`）
#[napi]
pub fn check_teen(path: String, model_name: Option<String>) -> napi::Result<HashMap<String, f64>> {
    let model = model_name.as_deref();
    let scores = core_anime_teen_score(&path, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Teen check failed: {}", e),
        )
    })?;
    Ok(scores.into_iter().map(|(k, v)| (k, v as f64)).collect())
}
