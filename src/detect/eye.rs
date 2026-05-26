//! アニメ画像の目検出処理を提供します。

use crate::detect::base::Detection;
use crate::inference::InferenceError;
use crate::inference::yolo_predict;
use image::DynamicImage;

/// アニメ画像内の目を検出します。
pub fn detect_eyes(
    image: &DynamicImage,
    level: &str,
    version: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/anime_eye_detection";
    let model_name = format!("eye_detect_{}_{}", version, level);
    let labels = vec!["eye".to_string()];

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
