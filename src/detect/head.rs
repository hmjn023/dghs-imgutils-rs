//! アニメ画像の頭部検出処理を提供します。

use crate::detect::base::Detection;
use crate::inference::InferenceError;
use crate::inference::yolo_predict;
use image::DynamicImage;

/// アニメ画像内の頭部を検出します。
///
/// # 引数
///
/// * `image` - 入力画像
/// * `level` - モデルのレベル（"s" または "n"。デフォルト: "s"）
/// * `version` - モデルバージョン（"v2.0" 等。デフォルト: "v2.0"）
/// * `conf_threshold` - 信頼度しきい値（デフォルト: 0.25）
/// * `iou_threshold` - NMS の IoU しきい値（デフォルト: 0.7）
pub fn detect_heads(
    image: &DynamicImage,
    level: &str,
    version: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/anime_head_detection";
    let model_name = format!("head_detect_{}_{}", version, level);
    let labels = vec!["head".to_string()];

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
