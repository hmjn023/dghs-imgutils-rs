//! NudeNet モデルを用いた露出部位検出処理を提供します。
//! YOLOv8 推論と専用の NMS (Non-Maximum Suppression) モデルの2段階 ONNX 処理を実行します。

use crate::detect::base::Detection;
use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{InferenceError, create_onnx_session};
use image::DynamicImage;
use ndarray::{Array1, Array4};
use once_cell::sync::Lazy;
use std::sync::Mutex;

static NUDENET_YOLO_SESSION: Lazy<Mutex<Option<ort::session::Session>>> =
    Lazy::new(|| Mutex::new(None));
static NUDENET_NMS_SESSION: Lazy<Mutex<Option<ort::session::Session>>> =
    Lazy::new(|| Mutex::new(None));

/// NudeNet の 18 クラスラベルの定義
pub const NUDENET_LABELS: &[&str] = &[
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
];

/// NudeNet 用の前処理。
/// 画像を白背景に貼り付け、最大辺サイズにパディングした上で 320x320 にリサイズします。
pub fn preprocess_nudenet(
    image: &DynamicImage,
    model_size: u32,
) -> Result<(Array4<f32>, f32), InferenceError> {
    // 半透明 PNG を白背景の上にアルファブレンド
    let rgb_img = crate::image::force_image_background(image, [255, 255, 255]).to_rgb8();
    let w = rgb_img.width();
    let h = rgb_img.height();
    let max_size = w.max(h);

    // 最大辺の正方形キャンバスを作成（白背景）
    let mut canvas =
        image::ImageBuffer::from_pixel(max_size, max_size, image::Rgb([255, 255, 255]));
    image::imageops::overlay(&mut canvas, &rgb_img, 0, 0);

    // ビリニア補間でリサイズを実行 (Triangle フィルタ)
    let resized = image::DynamicImage::ImageRgb8(canvas).resize_exact(
        model_size,
        model_size,
        image::imageops::FilterType::Triangle,
    );

    // テンソル変換
    let tensor = to_ndarray_chw(&resized, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0])
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    let global_ratio = max_size as f32 / model_size as f32;
    Ok((tensor, global_ratio))
}

/// NudeNet モデルを用いて露出部位を検出します。
pub fn detect_with_nudenet(
    image: &DynamicImage,
    topk: i32,
    iou_threshold: f32,
    score_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    let repo_id = "deepghs/nudenet_onnx";

    // 1. YOLOv8 検出モデルの初期化・取得
    let mut yolo_lock = NUDENET_YOLO_SESSION
        .lock()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    if yolo_lock.is_none() {
        let path = hf_hub_download(repo_id, "320n.onnx", None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
        *yolo_lock = Some(create_onnx_session(path)?);
    }
    let yolo_session = yolo_lock.as_mut().unwrap();

    // 2. NMS 処理モデルの初期化・取得
    let mut nms_lock = NUDENET_NMS_SESSION
        .lock()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    if nms_lock.is_none() {
        let path = hf_hub_download(repo_id, "nms-yolov8.onnx", None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
        *nms_lock = Some(create_onnx_session(path)?);
    }
    let nms_session = nms_lock.as_mut().unwrap();

    // 3. 画像の前処理
    let (input_tensor, global_ratio) = preprocess_nudenet(image, 320)?;

    // 4. YOLOv8 推論の実行
    // 入力は "images"
    let yolo_inputs = ort::inputs!["images" => ort::value::Tensor::from_array(input_tensor)?];
    let yolo_outputs = yolo_session.run(yolo_inputs)?;
    let output0 = yolo_outputs
        .get("output0")
        .ok_or_else(|| InferenceError::InvalidShape("Missing output0 from YOLOv8".to_string()))?;

    // 5. NMS 推論の実行
    // NMS の設定配列: [topk, iou_threshold, score_threshold]
    let config_array = Array1::from_vec(vec![topk as f32, iou_threshold, score_threshold]);
    let config_tensor = ort::value::Tensor::from_array(config_array)?;

    // output0 は YOLOv8 側の出力をそのまま渡す (clone可能なので clone する)
    let nms_inputs = ort::inputs![
        "detection" => output0,
        "config" => config_tensor
    ];

    let nms_outputs = nms_session.run(nms_inputs)?;
    let selected = nms_outputs
        .get("selected")
        .ok_or_else(|| InferenceError::InvalidShape("Missing selected from NMS".to_string()))?;

    let (shape, slice) = selected
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let selected_view = ndarray::ArrayView::from_shape(shape_usize, slice)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;

    // selected の形状: [1, box_cnt, 4 + 18]
    // バッチ次元 0 を取り出す
    let selected_array = selected_view.index_axis(ndarray::Axis(0), 0);

    let num_boxes = selected_array.shape()[0];
    let mut detections = Vec::new();

    for idx in 0..num_boxes {
        // 各行は [cx, cy, w, h, class0_score, class1_score, ...]
        let cx = selected_array[[idx, 0]] * global_ratio;
        let cy = selected_array[[idx, 1]] * global_ratio;
        let w = selected_array[[idx, 2]] * global_ratio;
        let h = selected_array[[idx, 3]] * global_ratio;

        let x = cx - 0.5 * w;
        let y = cy - 0.5 * h;

        // クラススコアの最大値を取得
        let mut max_score = -1.0f32;
        let mut max_class_id = 0usize;
        for c in 0..18 {
            let score = selected_array[[idx, 4 + c]];
            if score > max_score {
                max_score = score;
                max_class_id = c;
            }
        }

        let x0 = x.round().max(0.0) as u32;
        let y0 = y.round().max(0.0) as u32;
        let x1 = (x + w).round().min(image.width() as f32) as u32;
        let y1 = (y + h).round().min(image.height() as f32) as u32;

        detections.push(Detection {
            bbox: (x0, y0, x1, y1),
            label: NUDENET_LABELS[max_class_id].to_string(),
            score: max_score,
            mask: None,
        });
    }

    Ok(detections)
}
