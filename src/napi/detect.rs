//! 物体検出 (detect) モジュールの napi-rs バインディング。

use crate::detect::base::BBox as CoreBBox;
use crate::detect::base::Detection as CoreDetection;
use crate::detect::booru::detect_with_booru_yolo as core_detect_with_booru_yolo;
use crate::detect::censor::detect_censors as core_detect_censors;
use crate::detect::eye::detect_eyes as core_detect_eyes;
use crate::detect::face::detect_faces as core_detect_faces;
use crate::detect::halfbody::detect_halfbody as core_detect_halfbody;
use crate::detect::hand::detect_hands as core_detect_hands;
use crate::detect::head::detect_heads as core_detect_heads;
use crate::detect::nudenet::detect_with_nudenet as core_detect_with_nudenet;
use crate::detect::person::detect_person as core_detect_person;
use crate::detect::similarity::{
    bboxes_similarity as core_bboxes_similarity, calculate_mask_iou as core_calculate_mask_iou,
    detection_similarity as core_detection_similarity,
    detection_with_mask_similarity as core_detection_with_mask_similarity,
    masks_similarity as core_masks_similarity,
};
use crate::detect::text::detect_text as core_detect_text;
use crate::detect::visual::detection_visualize as core_detection_visualize;
use napi_derive::napi;
use ndarray::Array2;

/// 境界ボックスを JS/TS 側で表現する構造体
#[napi(object)]
#[derive(Debug, Clone, Copy)]
pub struct NapiBBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl From<CoreBBox> for NapiBBox {
    fn from(bbox: CoreBBox) -> Self {
        Self {
            x1: bbox.0 as i32,
            y1: bbox.1 as i32,
            x2: bbox.2 as i32,
            y2: bbox.3 as i32,
        }
    }
}

impl From<NapiBBox> for CoreBBox {
    fn from(bbox: NapiBBox) -> Self {
        (
            bbox.x1.max(0) as u32,
            bbox.y1.max(0) as u32,
            bbox.x2.max(0) as u32,
            bbox.y2.max(0) as u32,
        )
    }
}

/// 検出結果を JS/TS 側で表現する構造体
#[napi(object)]
#[derive(Debug, Clone)]
pub struct NapiDetection {
    pub bbox: NapiBBox,
    pub label: String,
    pub score: f64,
}

impl From<CoreDetection> for NapiDetection {
    fn from(d: CoreDetection) -> Self {
        Self {
            bbox: NapiBBox::from(d.bbox),
            label: d.label,
            score: d.score as f64,
        }
    }
}

impl From<NapiDetection> for CoreDetection {
    fn from(d: NapiDetection) -> Self {
        Self {
            bbox: CoreBBox::from(d.bbox),
            label: d.label,
            score: d.score as f32,
            mask: None,
        }
    }
}

/// 指定した画像ファイルパスから顔を検出します。
#[napi]
pub fn detect_faces(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v1.4".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_faces(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Face detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから頭部を検出します。
#[napi]
pub fn detect_heads(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v2.0".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_heads(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Head detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから人物全身・半身を検出します。
#[napi]
pub fn detect_person(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "m".to_string());
    let version = version.unwrap_or_else(|| "v1.1".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_person(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Person detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスからモザイクなどの検閲部位を検出します。
#[napi]
pub fn detect_censors(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v1.0".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_censors(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Censor detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから Booru-YOLO モデルを用いてオブジェクト検出を実行します。
#[napi]
pub fn detect_with_booru_yolo(
    path: String,
    model_name: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let model = model_name.unwrap_or_else(|| "yolov8s_aa11".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_with_booru_yolo(&image, &model, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Booru YOLO detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから NudeNet を用いて露出部位を検出します。
#[napi]
pub fn detect_with_nudenet(
    path: String,
    topk: Option<i32>,
    iou_threshold: Option<f64>,
    score_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let k = topk.unwrap_or(100);
    let iou = iou_threshold.unwrap_or(0.45) as f32;
    let score = score_threshold.unwrap_or(0.25) as f32;

    let results = core_detect_with_nudenet(&image, k, iou, score).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NudeNet detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから DBNet++ を用いてテキストを検出します。
#[napi]
pub fn detect_text(
    path: String,
    model: Option<String>,
    threshold: Option<f64>,
    max_area_size: Option<i32>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let model = model.unwrap_or_else(|| "dbnetpp_resnet50_fpnc_1200e_icdar2015".to_string());
    let conf = threshold.unwrap_or(0.05) as f32;
    let max_size = max_area_size.map(|v| v.max(0) as u32);

    let results = core_detect_text(&image, &model, conf, max_size).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Text detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから目を検出します。
#[napi]
pub fn detect_eyes(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v1.0".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_eyes(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Eye detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから上半身を検出します。
#[napi]
pub fn detect_halfbody(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v1.0".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_halfbody(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Halfbody detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 指定した画像ファイルパスから手を検出します。
#[napi]
pub fn detect_hands(
    path: String,
    level: Option<String>,
    version: Option<String>,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<NapiDetection>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let level = level.unwrap_or_else(|| "s".to_string());
    let version = version.unwrap_or_else(|| "v1.0".to_string());
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;

    let results = core_detect_hands(&image, &level, &version, conf, iou).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Hand detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiDetection::from).collect())
}

/// 二つの境界ボックスリストの間のペア IoU 類似度を計算します。
#[napi]
pub fn bboxes_similarity(boxes1: Vec<NapiBBox>, boxes2: Vec<NapiBBox>) -> napi::Result<Vec<f64>> {
    let core_boxes1: Vec<CoreBBox> = boxes1.into_iter().map(CoreBBox::from).collect();
    let core_boxes2: Vec<CoreBBox> = boxes2.into_iter().map(CoreBBox::from).collect();

    let sims = core_bboxes_similarity(&core_boxes1, &core_boxes2);
    Ok(sims.into_iter().map(|v| v as f64).collect())
}

/// 二つの検出結果リストの間のペア類似度を計算します。
#[napi]
pub fn detection_similarity(
    detect1: Vec<NapiDetection>,
    detect2: Vec<NapiDetection>,
) -> napi::Result<Vec<f64>> {
    let core_d1: Vec<CoreDetection> = detect1.into_iter().map(CoreDetection::from).collect();
    let core_d2: Vec<CoreDetection> = detect2.into_iter().map(CoreDetection::from).collect();

    let sims = core_detection_similarity(&core_d1, &core_d2);
    Ok(sims.into_iter().map(|v| v as f64).collect())
}

/// 境界ボックスと信頼度スコアを元の画像上に描画した画像を生成し、PNG バッファとして返します。
#[napi]
pub fn detection_visualize(
    path: String,
    detections: Vec<NapiDetection>,
    text_padding: Option<i32>,
    fontsize: Option<f64>,
    max_short_edge_size: Option<i32>,
    mask_alpha: Option<f64>,
    no_label: Option<bool>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let core_detections: Vec<CoreDetection> =
        detections.into_iter().map(CoreDetection::from).collect();
    let padding = text_padding.unwrap_or(6) as u32;
    let size = fontsize.unwrap_or(12.0) as f32;
    let limit = max_short_edge_size.map(|v| v.max(0) as u32);
    let alpha = mask_alpha.unwrap_or(0.5) as f32;
    let no_lbl = no_label.unwrap_or(false);

    let visual_img = core_detection_visualize(
        &image,
        &core_detections,
        padding,
        size,
        limit,
        alpha,
        no_lbl,
    );

    let mut buf = std::io::Cursor::new(Vec::new());
    visual_img
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode visualized image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}

/// 2つのマスク間の IoU (Intersection over Union) を計算します。
///
/// * `mask1`: 1つ目のマスク（2次元のf64配列）
/// * `mask2`: 2つ目のマスク（2次元のf64配列）
/// * `threshold`: 二値化のしきい値（デフォルト 0.5）
#[napi]
pub fn calculate_mask_iou(
    mask1: Vec<Vec<f64>>,
    mask2: Vec<Vec<f64>>,
    threshold: Option<f64>,
) -> napi::Result<f64> {
    let to_array2 = |mask: Vec<Vec<f64>>| -> napi::Result<Array2<f32>> {
        let rows = mask.len();
        if rows == 0 {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "Mask must not be empty".to_string(),
            ));
        }
        let cols = mask[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in &mask {
            if row.len() != cols {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    "All rows must have the same length".to_string(),
                ));
            }
            for &v in row {
                data.push(v as f32);
            }
        }
        Array2::from_shape_vec((rows, cols), data).map_err(|e| {
            napi::Error::new(
                napi::Status::InvalidArg,
                format!("Invalid mask shape: {}", e),
            )
        })
    };

    let a1 = to_array2(mask1)?;
    let a2 = to_array2(mask2)?;
    let th = threshold.unwrap_or(0.5) as f32;
    let iou = core_calculate_mask_iou(&a1, &a2, th);
    Ok(iou as f64)
}

/// 複数のマスク間のペア類似度を計算します。
#[napi]
pub fn masks_similarity(
    masks1: Vec<Vec<Vec<f64>>>,
    masks2: Vec<Vec<Vec<f64>>>,
    threshold: Option<f64>,
) -> napi::Result<Vec<f64>> {
    let to_array2 = |mask: Vec<Vec<f64>>| -> napi::Result<Array2<f32>> {
        let rows = mask.len();
        let cols = mask[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in &mask {
            for &v in row {
                data.push(v as f32);
            }
        }
        Array2::from_shape_vec((rows, cols), data).map_err(|e| {
            napi::Error::new(
                napi::Status::InvalidArg,
                format!("Invalid mask shape: {}", e),
            )
        })
    };

    let a1: Vec<Array2<f32>> = masks1
        .into_iter()
        .map(to_array2)
        .collect::<napi::Result<Vec<_>>>()?;
    let a2: Vec<Array2<f32>> = masks2
        .into_iter()
        .map(to_array2)
        .collect::<napi::Result<Vec<_>>>()?;
    let th = threshold.unwrap_or(0.5) as f32;

    let sims = core_masks_similarity(&a1, &a2, th);
    Ok(sims.into_iter().map(|v| v as f64).collect())
}

/// 検出結果（bbox + mask）のペア類似度を計算します。
#[napi]
pub fn detection_with_mask_similarity(
    detect1: Vec<NapiDetection>,
    detect2: Vec<NapiDetection>,
    threshold: Option<f64>,
) -> napi::Result<Vec<f64>> {
    let core_d1: Vec<CoreDetection> = detect1.into_iter().map(CoreDetection::from).collect();
    let core_d2: Vec<CoreDetection> = detect2.into_iter().map(CoreDetection::from).collect();
    let th = threshold.unwrap_or(0.5) as f32;

    let sims = core_detection_with_mask_similarity(&core_d1, &core_d2, th);
    Ok(sims.into_iter().map(|v| v as f64).collect())
}
