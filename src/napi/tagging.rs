use crate::napi::inference::{NapiInferenceOptions, run_with_inference_options};
use crate::tagging::camie::get_camie_tags as core_camie;
use crate::tagging::deepdanbooru::get_deepdanbooru_tags as core_deepdanbooru;
use crate::tagging::deepgelbooru::get_deepgelbooru_tags as core_deepgelbooru;
use crate::tagging::mldanbooru::get_mldanbooru_tags as core_mldanbooru;
use crate::tagging::oppaioracle::get_oppaioracle_tags as core_oppaioracle;
use crate::tagging::wd14::get_wd14_tags as core_wd14;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct TagResult {
    pub general: HashMap<String, f64>,
    pub character: HashMap<String, f64>,
    pub rating: HashMap<String, f64>,
    pub tag: HashMap<String, f64>,
}

fn vec_to_map(tags: Vec<(String, f32)>) -> HashMap<String, f64> {
    tags.into_iter().map(|(k, v)| (k, v as f64)).collect()
}

#[napi]
pub fn get_deepdanbooru_tags(
    path: String,
    general_threshold: Option<f64>,
    character_threshold: Option<f64>,
    use_real_name: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let gt = general_threshold.unwrap_or(0.5) as f32;
    let ct = character_threshold.unwrap_or(0.5) as f32;
    let ur = use_real_name.unwrap_or(false);
    let result = run_with_inference_options(inference_options, || {
        core_deepdanbooru(&image, gt, ct, ur).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("DeepDanbooru failed: {}", e),
            )
        })
    })?;
    let rating = result.rest.get("rating").cloned().unwrap_or_default();
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: vec_to_map(result.character),
        rating: vec_to_map(rating),
        tag: vec_to_map(result.tag),
    })
}

#[napi]
pub fn get_mldanbooru_tags(
    path: String,
    threshold: Option<f64>,
    size: Option<i32>,
    keep_ratio: Option<bool>,
    use_real_name: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let th = threshold.unwrap_or(0.7) as f32;
    let sz = size.unwrap_or(448) as u32;
    let kr = keep_ratio.unwrap_or(false);
    let ur = use_real_name.unwrap_or(false);
    let result = run_with_inference_options(inference_options, || {
        core_mldanbooru(&image, th, sz, kr, ur).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("ML-Danbooru failed: {}", e),
            )
        })
    })?;
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: vec_to_map(result.character),
        rating: HashMap::new(),
        tag: vec_to_map(result.tag),
    })
}

#[napi]
pub fn get_deepgelbooru_tags(
    path: String,
    general_threshold: Option<f64>,
    character_threshold: Option<f64>,
    drop_overlap: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let gt = general_threshold.unwrap_or(0.3) as f32;
    let ct = character_threshold.unwrap_or(0.3) as f32;
    let result = run_with_inference_options(inference_options, || {
        core_deepgelbooru(&image, gt, ct, drop_overlap.unwrap_or(false)).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("DeepGelbooru failed: {}", e),
            )
        })
    })?;
    let rating = result.rest.get("rating").cloned().unwrap_or_default();
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: vec_to_map(result.character),
        rating: vec_to_map(rating),
        tag: vec_to_map(result.tag),
    })
}

#[napi]
pub fn get_wd14_tags(
    path: String,
    model_name: Option<String>,
    general_threshold: Option<f64>,
    general_mcut_enabled: Option<bool>,
    character_threshold: Option<f64>,
    character_mcut_enabled: Option<bool>,
    no_underline: Option<bool>,
    drop_overlap: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let model = model_name.unwrap_or_else(|| "SwinV2_v3".to_string());
    let gt = general_threshold.unwrap_or(0.35) as f32;
    let gm = general_mcut_enabled.unwrap_or(false);
    let ct = character_threshold.unwrap_or(0.85) as f32;
    let cm = character_mcut_enabled.unwrap_or(false);
    let nu = no_underline.unwrap_or(false);
    let result = run_with_inference_options(inference_options, || {
        core_wd14(
            &image,
            &model,
            gt,
            gm,
            ct,
            cm,
            nu,
            drop_overlap.unwrap_or(false),
        )
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("WD14 failed: {}", e)))
    })?;
    let rating = result.rest.get("rating").cloned().unwrap_or_default();
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: vec_to_map(result.character),
        rating: vec_to_map(rating),
        tag: vec_to_map(result.tag),
    })
}

#[napi]
pub fn get_camie_tags(
    path: String,
    model_name: Option<String>,
    mode: Option<String>,
    thresholds: Option<HashMap<String, f64>>,
    no_underline: Option<bool>,
    drop_overlap: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let model = model_name.unwrap_or_else(|| "initial".to_string());
    let mode_str = mode.unwrap_or_else(|| "balanced".to_string());
    let nu = no_underline.unwrap_or(false);
    let thresh_map = thresholds.map(|m| m.into_iter().map(|(k, v)| (k, v as f32)).collect());
    let result = run_with_inference_options(inference_options, || {
        core_camie(
            &image,
            &model,
            &mode_str,
            thresh_map,
            nu,
            drop_overlap.unwrap_or(false),
        )
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("Camie failed: {}", e)))
    })?;
    let rating = result.rest.get("rating").cloned().unwrap_or_default();
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: vec_to_map(result.character),
        rating: vec_to_map(rating),
        tag: vec_to_map(result.tag),
    })
}

/// 指定した画像ファイルパスから OppaiOracle モデルを用いてアニメ調イラストのタグとその確信度スコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデルバリアント (`"v1"` または `"v1.1"`, デフォルト `"v1.1"`)
/// * `threshold`: タグ採用の確率閾値 (0.0〜1.0)。省略時はモデル同梱の
///   `pr_thresholds.json` にある macro P=R breakeven 閾値を使用
/// * `no_underline`: `true` の場合、タグ名のアンダースコアをスペースに置換 (デフォルト `false`)
///
/// 戻り値の `TagResult` には `general` / `tag` に全タグ (確率降順)、
/// `rating` にレーティングスコアが格納されます (このモデルの語彙は一般タグのみのため
/// `character` は空です)。
#[napi]
pub async fn get_oppaioracle_tags(
    path: String,
    model_name: Option<String>,
    threshold: Option<f64>,
    no_underline: Option<bool>,
    inference_options: Option<NapiInferenceOptions>,
) -> napi::Result<TagResult> {
    let image = crate::napi::open_image_robust(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let model = model_name.unwrap_or_else(|| "v1.1".to_string());
    let result = run_with_inference_options(inference_options, || {
        core_oppaioracle(
            &image,
            &model,
            threshold.map(|t| t as f32),
            no_underline.unwrap_or(false),
        )
        .map_err(|e| match e {
            crate::tagging::TaggingError::InvalidArgument(msg) => napi::Error::new(
                napi::Status::InvalidArg,
                format!("OppaiOracle failed: {}", msg),
            ),
            e => napi::Error::new(
                napi::Status::GenericFailure,
                format!("OppaiOracle failed: {}", e),
            ),
        })
    })?;
    let rating = result.rest.get("rating").cloned().unwrap_or_default();
    Ok(TagResult {
        general: vec_to_map(result.general),
        character: HashMap::new(),
        rating: vec_to_map(rating),
        tag: vec_to_map(result.tag),
    })
}

#[napi]
pub fn is_blacklisted(tag: String) -> bool {
    crate::tagging::blacklist::is_blacklisted(&tag)
}

#[napi]
pub fn is_basic_character_tag(tag: String) -> bool {
    crate::tagging::character::is_basic_character_tag(&tag)
}

#[napi]
pub fn drop_blacklisted_tags(
    tags: HashMap<String, f64>,
    use_presets: Option<bool>,
) -> HashMap<String, f64> {
    let vec: Vec<(String, f32)> = tags.into_iter().map(|(k, v)| (k, v as f32)).collect();
    let up = use_presets.unwrap_or(true);
    crate::tagging::blacklist::drop_blacklisted_tags(&vec, up)
        .into_iter()
        .map(|(k, v)| (k, v as f64))
        .collect()
}

#[napi]
pub fn drop_overlap_tags(tags: HashMap<String, f64>) -> HashMap<String, f64> {
    let vec: Vec<(String, f32)> = tags.into_iter().map(|(k, v)| (k, v as f32)).collect();
    crate::tagging::overlap::drop_overlap_tags(&vec)
        .into_iter()
        .map(|(k, v)| (k, v as f64))
        .collect()
}

#[napi]
pub fn drop_basic_character_tags(tags: HashMap<String, f64>) -> HashMap<String, f64> {
    let vec: Vec<(String, f32)> = tags.into_iter().map(|(k, v)| (k, v as f32)).collect();
    crate::tagging::character::drop_basic_character_tags(&vec)
        .into_iter()
        .map(|(k, v)| (k, v as f64))
        .collect()
}

#[napi]
pub fn tag_match_suffix(text: String, suffix: String) -> bool {
    crate::tagging::tag_match::tag_match_suffix(&text, &suffix)
}

#[napi]
pub fn tag_match_prefix(text: String, prefix: String) -> bool {
    crate::tagging::tag_match::tag_match_prefix(&text, &prefix)
}

#[napi]
pub fn tag_match_full(t1: String, t2: String) -> bool {
    crate::tagging::tag_match::tag_match_full(&t1, &t2)
}

#[napi]
pub fn add_underline(tag: String) -> String {
    crate::tagging::format::add_underline(&tag)
}

#[napi]
pub fn remove_underline(tag: String) -> String {
    crate::tagging::format::remove_underline(&tag)
}

#[napi]
pub fn tags_to_text(
    tags: HashMap<String, f64>,
    use_spaces: Option<bool>,
    use_escape: Option<bool>,
    include_score: Option<bool>,
    score_descend: Option<bool>,
) -> String {
    let map: HashMap<String, f32> = tags.into_iter().map(|(k, v)| (k, v as f32)).collect();
    crate::tagging::format::tags_to_text(
        &map,
        use_spaces.unwrap_or(false),
        use_escape.unwrap_or(true),
        include_score.unwrap_or(false),
        score_descend.unwrap_or(true),
    )
}

#[napi]
pub fn sort_tags(tags: HashMap<String, f64>, mode: Option<String>) -> Vec<String> {
    let map: HashMap<String, f32> = tags.into_iter().map(|(k, v)| (k, v as f32)).collect();
    let m = mode.as_deref().unwrap_or("score");
    crate::tagging::order::sort_tags(&map, m)
}
