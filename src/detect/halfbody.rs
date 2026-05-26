//! アニメ画像の上半身（ハーフボディ）検出処理を提供します。

use crate::detect::base::Detection;
use crate::inference::yolo_predict;
use crate::inference::InferenceError;
use image::DynamicImage;

/// アニメ画像内の上半身を検出します。
pub fn detect_halfbody(
    image: &DynamicImage,
    level: &str,
    version: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/anime_halfbody_detection";
    let model_name = format!("halfbody_detect_{}_{}", version, level);
    let labels = vec!["halfbody".to_string()];

    let results = yolo_predict(
        image,
        repo_id,
        &model_name,
        (640, 640),
        conf_threshold,
        iou_threshold,
        &labels,
    )?;

    let detections = results
        .into_iter()
        .map(|r| Detection {
            bbox: r.bbox,
            label: r.label,
            score: r.score,
            mask: None,
        })
        .collect();

    Ok(detections)
}
