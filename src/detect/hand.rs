//! アニメ画像の手検出処理を提供します。

use crate::detect::base::Detection;
use crate::inference::InferenceError;
use crate::inference::yolo_predict;
use image::DynamicImage;

/// アニメ画像内の手を検出します。
pub fn detect_hands(
    image: &DynamicImage,
    level: &str,
    version: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/anime_hand_detection";
    let model_name = format!("hand_detect_{}_{}", version, level);
    let labels = vec!["hand".to_string()];

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
