//! Booru-YOLO モデルを用いた26クラスに及ぶ詳細なイラスト特徴・露出部位検出処理を提供します。

use crate::detect::base::Detection;
use crate::inference::InferenceError;
use crate::inference::yolo_predict;
use image::DynamicImage;

/// Booru-YOLO モデルを用いてオブジェクト検出を実行します。
///
/// # 引数
///
/// * `image` - 入力画像
/// * `model_name` - 使用するモデル名（例: "yolov8s_aa11"（デフォルト）, "yolov8s_pp12", "yolov8m_pp13" 等）
/// * `conf_threshold` - 信頼度しきい値（デフォルト: 0.25）
/// * `iou_threshold` - NMS の IoU しきい値（デフォルト: 0.7）
pub fn detect_with_booru_yolo(
    image: &DynamicImage,
    model_name: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/booru_yolo";

    // 26クラスの完全定義
    let labels = vec![
        "head".to_string(),
        "bust".to_string(),
        "boob".to_string(),
        "shld".to_string(),
        "sideb".to_string(),
        "belly".to_string(),
        "nopan".to_string(),
        "butt".to_string(),
        "ass".to_string(),
        "split".to_string(),
        "sprd".to_string(),
        "vsplt".to_string(),
        "vsprd".to_string(),
        "hip".to_string(),
        "wing".to_string(),
        "feral".to_string(),
        "hdrago".to_string(),
        "hpony".to_string(),
        "hfox".to_string(),
        "hrabb".to_string(),
        "hcat".to_string(),
        "hbear".to_string(),
        "jacko".to_string(),
        "jackx".to_string(),
        "hhorse".to_string(),
        "hbird".to_string(),
    ];

    let results = yolo_predict(
        image,
        repo_id,
        model_name,
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
