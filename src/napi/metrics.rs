//! メトリクス (metrics) モジュールの napi-rs バインディング。

use crate::metrics::ccip::{
    ccip_batch_extract_features as core_ccip_batch_extract_features,
    ccip_default_clustering_params as core_ccip_default_clustering_params,
    ccip_default_threshold as core_ccip_default_threshold,
};
use crate::metrics::dbaesthetic::{
    anime_dbaesthetic as core_anime_dbaesthetic,
    anime_dbaesthetic_full as core_anime_dbaesthetic_full,
};
use crate::metrics::laplacian::laplacian_score as core_laplacian_score;
use crate::metrics::lpips::lpips_clustering as core_lpips_clustering;
use crate::metrics::psnr_::psnr as core_psnr;
use crate::napi::inference::{NapiInferenceOptions, run_with_inference_options};
use napi_derive::napi;
use std::collections::HashMap;

#[napi]
pub fn laplacian_score(path: String) -> napi::Result<f64> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path, e),
        )
    })?;
    let score = core_laplacian_score(&image);
    Ok(score as f64)
}

#[napi]
pub fn psnr(path1: String, path2: String) -> napi::Result<f64> {
    let img1 = image::open(&path1).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path1, e),
        )
    })?;
    let img2 = image::open(&path2).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path2, e),
        )
    })?;
    let result = core_psnr(&img1, &img2);
    Ok(result)
}

/// 複数画像から CCIP 特徴ベクトルを一括抽出します。
///
/// * `paths`: 画像ファイルパスの配列
/// * `model_name`: CCIPモデル名（オプション）
#[napi]
pub fn ccip_batch_extract_features(
    paths: Vec<String>,
    model_name: Option<String>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<Vec<Vec<f64>>> {
    let images: Vec<image::DynamicImage> = paths
        .iter()
        .map(|p| image::open(p))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            napi::Error::new(
                napi::Status::InvalidArg,
                format!("Failed to open image: {}", e),
            )
        })?;
    let model = model_name.as_deref().unwrap_or("");
    let features = run_with_inference_options(inference_options, || {
        core_ccip_batch_extract_features(&images, 384, model).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("CCIP batch extract failed: {}", e),
            )
        })
    })?;
    Ok(features
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// CCIP モデルのデフォルト同一判定しきい値を取得します。
#[napi]
pub fn ccip_default_threshold(model_name: Option<String>) -> napi::Result<f64> {
    let model = model_name.as_deref().unwrap_or("");
    let threshold = core_ccip_default_threshold(model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to get default threshold: {}", e),
        )
    })?;
    Ok(threshold as f64)
}

/// CCIP モデルのデフォルトクラスタリングパラメータを取得します。
/// 戻り値は `[eps, min_samples]` の配列です。
#[napi]
pub fn ccip_default_clustering_params(
    model_name: Option<String>,
    method: Option<String>,
) -> napi::Result<Vec<f64>> {
    let model = model_name.as_deref().unwrap_or("");
    let m = method.as_deref().unwrap_or("optics");
    let (eps, min_samples) = core_ccip_default_clustering_params(model, m).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to get clustering params: {}", e),
        )
    })?;
    Ok(vec![eps as f64, min_samples as f64])
}

/// アニメ画像の美的スコアを返します。
#[napi]
pub fn anime_dbaesthetic(
    path: String,
    model_name: Option<String>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<f64> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path, e),
        )
    })?;
    let model = model_name.as_deref();
    let score = run_with_inference_options(inference_options, || {
        core_anime_dbaesthetic(&image, model).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Aesthetic scoring failed: {}", e),
            )
        })
    })?;
    Ok(score as f64)
}

/// アニメ画像の美的スコアとカテゴリ別詳細スコアを返します。
/// 戻り値は `{ score, details }` 形式です。
#[napi(object)]
pub struct DbaestheticFullResult {
    pub score: f64,
    pub details: HashMap<String, f64>,
}

#[napi]
pub fn anime_dbaesthetic_full(
    path: String,
    model_name: Option<String>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<DbaestheticFullResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path, e),
        )
    })?;
    let model = model_name.as_deref();
    let (score, details) = run_with_inference_options(inference_options, || {
        core_anime_dbaesthetic_full(&image, model).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Aesthetic scoring failed: {}", e),
            )
        })
    })?;
    Ok(DbaestheticFullResult {
        score: score as f64,
        details: details.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}

/// LPIPS ベースのクラスタリングを実行します。
/// 画像のパスの配列を受け取り、クラスタラベルの配列を返します。
#[napi]
pub fn lpips_clustering(
    paths: Vec<String>,
    threshold: Option<f64>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<Vec<i32>> {
    let images: Vec<image::DynamicImage> = paths
        .iter()
        .map(|p| image::open(p))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            napi::Error::new(
                napi::Status::InvalidArg,
                format!("Failed to open image: {}", e),
            )
        })?;
    let th = threshold.map(|v| v as f32);
    let clusters = run_with_inference_options(inference_options, || {
        core_lpips_clustering(&images, th).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("LPIPS clustering failed: {}", e),
            )
        })
    })?;
    Ok(clusters.into_iter().map(|v| v as i32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{Backend, Precision, current_inference_options};

    fn empty_options() -> NapiInferenceOptions {
        NapiInferenceOptions {
            backend: None,
            precision: None,
            device_id: None,
            openvino_device_type: None,
            vitis_config_file: None,
            ep_cache_dir: None,
            migraphx_int8_calibration_table: None,
            migraphx_exhaustive_tune: None,
        }
    }

    #[test]
    fn default_inference_options_leave_operation_result_unchanged() {
        let result = run_with_inference_options(None, || Ok::<_, napi::Error>(42_u32)).unwrap();

        assert_eq!(result, 42);
    }

    #[test]
    fn partial_inference_options_are_visible_inside_the_operation() {
        let mut input = empty_options();
        input.backend = Some("cpu".to_owned());
        input.precision = Some("fp32".to_owned());
        input.device_id = Some(7);

        let options = run_with_inference_options(Some(input), || {
            current_inference_options()
                .map_err(|error| napi::Error::new(napi::Status::GenericFailure, error.to_string()))
        })
        .unwrap();

        assert_eq!(options.backend, Backend::Cpu);
        assert_eq!(options.precision, Precision::Fp32);
        assert_eq!(options.device_id, 7);
    }

    #[test]
    fn operation_errors_are_propagated_without_swallowing_them() {
        let error = run_with_inference_options(None, || {
            Err::<(), _>(napi::Error::new(
                napi::Status::GenericFailure,
                "sentinel inference failure",
            ))
        })
        .unwrap_err();

        assert_eq!(error.status, napi::Status::GenericFailure);
        assert!(error.reason.contains("sentinel inference failure"));
    }
}
