use crate::generic::clip;
use crate::generic::siglip;
use crate::generic::yoloseg;
use napi_derive::napi;

/// Encode an image into CLIP embeddings.
/// Returns [embeddings, encodings] as flat f64 arrays.
#[napi]
pub fn clip_image_encode(
    path: String,
    repo_id: Option<String>,
    model_name: String,
) -> napi::Result<Vec<Vec<f64>>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let repo = repo_id.unwrap_or_else(|| "deepghs/clip_onnx".to_string());
    let (embeddings, _encodings) = clip::clip_image_encode(&[image], &repo, &model_name)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
    Ok(embeddings
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// Encode text into CLIP embeddings.
#[napi]
pub fn clip_text_encode(
    texts: Vec<String>,
    repo_id: Option<String>,
    model_name: String,
) -> napi::Result<Vec<Vec<f64>>> {
    let repo = repo_id.unwrap_or_else(|| "deepghs/clip_onnx".to_string());
    let (embeddings, _encodings) = clip::clip_text_encode(&texts, &repo, &model_name)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
    Ok(embeddings
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// Compute CLIP similarity matrix between image and text embeddings.
/// Returns (similarities, logits, predictions).
#[napi]
pub fn clip_similarity(
    image_embeds: Vec<Vec<f64>>,
    text_embeds: Vec<Vec<f64>>,
    logit_scale: Option<f64>,
) -> napi::Result<Vec<Vec<f64>>> {
    let img: Vec<Vec<f32>> = image_embeds
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .collect();
    let txt: Vec<Vec<f32>> = text_embeds
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .collect();
    let ls = logit_scale.unwrap_or(4.605170185988091) as f32;
    let (_similarities, _logits, predictions) = clip::clip_predict(&img, &txt, ls);
    Ok(predictions
        .into_iter()
        .map(|row| row.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// Encode an image into SigLIP embeddings.
#[napi]
pub fn siglip_image_encode(
    path: String,
    repo_id: Option<String>,
    model_name: String,
) -> napi::Result<Vec<Vec<f64>>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let repo = repo_id.unwrap_or_else(|| "deepghs/siglip_onnx".to_string());
    let (embeddings, _encodings) = siglip::siglip_image_encode(&[image], &repo, &model_name)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
    Ok(embeddings
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// Run SigLIP prediction (sigmoid-based scores).
#[napi]
pub fn siglip_predict(
    image_embeds: Vec<Vec<f64>>,
    text_embeds: Vec<Vec<f64>>,
    logit_scale: Option<f64>,
    logit_bias: Option<f64>,
) -> napi::Result<Vec<Vec<f64>>> {
    let img: Vec<Vec<f32>> = image_embeds
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .collect();
    let txt: Vec<Vec<f32>> = text_embeds
        .into_iter()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .collect();
    let ls = logit_scale.unwrap_or(4.605170185988091) as f32;
    let lb = logit_bias.unwrap_or(0.0) as f32;
    let (_similarities, _logits, predictions) = siglip::siglip_predict(&img, &txt, ls, lb);
    Ok(predictions
        .into_iter()
        .map(|row| row.into_iter().map(|x| x as f64).collect())
        .collect())
}

/// A segmentation detection result for napi.
#[napi(object)]
pub struct SegDetectionResult {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub label: String,
    pub score: f64,
    /// Flattened mask (H * W) as f64 values.
    pub mask: Vec<f64>,
    pub mask_height: i32,
    pub mask_width: i32,
}

/// Run YOLO segmentation on an image.
#[napi]
pub fn yolo_segment(
    path: String,
    repo_id: String,
    model_name: String,
    conf_threshold: Option<f64>,
    iou_threshold: Option<f64>,
) -> napi::Result<Vec<SegDetectionResult>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let conf = conf_threshold.unwrap_or(0.25) as f32;
    let iou = iou_threshold.unwrap_or(0.7) as f32;
    let detections = yoloseg::yolo_seg_predict(&image, &repo_id, &model_name, conf, iou)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;

    Ok(detections
        .into_iter()
        .map(|d| {
            let (mh, mw) = d.mask.dim();
            SegDetectionResult {
                x1: d.bbox.0 as i32,
                y1: d.bbox.1 as i32,
                x2: d.bbox.2 as i32,
                y2: d.bbox.3 as i32,
                label: d.label,
                score: d.score as f64,
                mask: d
                    .mask
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as f64)
                    .collect(),
                mask_height: mh as i32,
                mask_width: mw as i32,
            }
        })
        .collect())
}
