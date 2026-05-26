//! YOLOv8 (NMSベース)、YOLOv10 (End-to-End)、RT-DETR に対応した汎用 ONNX 推論および NMS (Non-Maximum Suppression) 処理を提供します。

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{create_onnx_session, run_onnx_session, InferenceError};
use image::DynamicImage;
use ndarray::Array4;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

/// YOLO 検出結果
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// 境界ボックス `(x1, y1, x2, y2)`
    pub bbox: (u32, u32, u32, u32),
    /// クラスラベル名
    pub label: String,
    /// 信頼度スコア
    pub score: f32,
    /// オプションのインスタンスセグメンテーションマスク (H, W)
    pub mask: Option<ndarray::Array2<f32>>,
}

/// YOLO 用の画像前処理。
///
/// アスペクト比を維持してリサイズし、`allow_dynamic` が有効な場合は `align` の倍数に調整します。
/// ピクセル値は `0.0..1.0` にスケーリングされた `[1, 3, H, W]` テンソルを返します。
pub fn preprocess_image_yolo(
    image: &DynamicImage,
    max_infer_size: (u32, u32),
    allow_dynamic: bool,
    align: u32,
) -> Result<(Array4<f32>, (u32, u32), (u32, u32)), InferenceError> {
    let old_w = image.width();
    let old_h = image.height();
    let (mut new_w, mut new_h) = (old_w, old_h);

    if allow_dynamic {
        let scale_w = max_infer_size.0 as f32 / old_w as f32;
        let scale_h = max_infer_size.1 as f32 / old_h as f32;
        let r = scale_w.min(scale_h);
        if r < 1.0 {
            new_w = (old_w as f32 * r) as u32;
            new_h = (old_h as f32 * r) as u32;
        }
        // align の倍数に切り上げ
        new_w = ((new_w as f32 / align as f32).ceil() * align as f32) as u32;
        new_h = ((new_h as f32 / align as f32).ceil() * align as f32) as u32;
    } else {
        new_w = max_infer_size.0;
        new_h = max_infer_size.1;
    }

    // Bilinear リサイズを実行 (Triangle フィルタを採用)
    let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);

    // 正規化なし (mean = 0, std = 1) で [1, 3, H, W] テンソルへ変換
    let tensor = to_ndarray_chw(&resized, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0])
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    Ok((tensor, (old_w, old_h), (new_w, new_h)))
}

/// 検出された座標を、前処理後の画像サイズからオリジナル画像サイズへ逆変換します。
pub fn xy_postprocess(
    x: f32,
    y: f32,
    old_size: (u32, u32),
    new_size: (u32, u32),
) -> (u32, u32) {
    let old_w = old_size.0 as f32;
    let old_h = old_size.1 as f32;
    let new_w = new_size.0 as f32;
    let new_h = new_size.1 as f32;

    let x = (x / new_w) * old_w;
    let y = (y / new_h) * old_h;

    let x_clip = x.clamp(0.0, old_w).round() as u32;
    let y_clip = y.clamp(0.0, old_h).round() as u32;
    (x_clip, y_clip)
}

/// `[cx, cy, w, h]` (中心座標と幅高さ) 形式を `[x1, y1, x2, y2]` 形式へ変換します。
pub fn yolo_xywh2xyxy(bbox: &[f32; 4]) -> [f32; 4] {
    let cx = bbox[0];
    let cy = bbox[1];
    let w = bbox[2];
    let h = bbox[3];
    [
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ]
}

/// BBox に対する Non-Maximum Suppression (非極大値抑制) の等価 Rust 実装。
///
/// Python 版の `_yolo_nms` の数式と完全に同一のピクセル補正係数 `+ 1.0` を含みます。
pub fn yolo_nms(
    boxes: &[[f32; 4]],
    scores: &[f32],
    iou_threshold: f32,
) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let n = boxes.len();
    let mut areas = vec![0.0; n];
    for i in 0..n {
        let w = boxes[i][2] - boxes[i][0] + 1.0;
        let h = boxes[i][3] - boxes[i][1] + 1.0;
        areas[i] = w.max(0.0) * h.max(0.0);
    }

    // スコアの降順にソート
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        if order.len() == 1 {
            break;
        }

        let mut next_order = Vec::new();
        for &j in &order[1..] {
            let xx1 = boxes[i][0].max(boxes[j][0]);
            let yy1 = boxes[i][1].max(boxes[j][1]);
            let xx2 = boxes[i][2].min(boxes[j][2]);
            let yy2 = boxes[i][3].min(boxes[j][3]);

            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;

            let iou = inter / (areas[i] + areas[j] - inter);
            if iou <= iou_threshold {
                next_order.push(j);
            }
        }
        order = next_order;
    }
    keep
}

/// NMSベースモデル (YOLOv8など) の出力後処理。
///
/// `output` テンソルは `[4 + class_count, box_cnt]` の形状を持ちます。
pub fn postprocess_nms_yolo(
    output: &ndarray::Array2<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
    old_size: (u32, u32),
    new_size: (u32, u32),
    labels: &[String],
) -> Result<Vec<DetectionResult>, InferenceError> {
    let class_cnt = labels.len();
    if output.shape()[0] != 4 + class_cnt {
        return Err(InferenceError::InvalidShape(format!(
            "Expected output row count {} (4 + classes), found {}",
            4 + class_cnt,
            output.shape()[0]
        )));
    }

    let box_cnt = output.shape()[1];
    let mut candidate_boxes = Vec::new();
    let mut candidate_scores = Vec::new();
    let mut candidate_class_ids = Vec::new();

    for col in 0..box_cnt {
        // 各クラススコアを取得
        let mut max_score = -1.0f32;
        let mut max_class_id = 0usize;
        for c in 0..class_cnt {
            let score = output[[4 + c, col]];
            if score > max_score {
                max_score = score;
                max_class_id = c;
            }
        }

        if max_score > conf_threshold {
            let cx = output[[0, col]];
            let cy = output[[1, col]];
            let w = output[[2, col]];
            let h = output[[3, col]];
            let xyxy = yolo_xywh2xyxy(&[cx, cy, w, h]);

            candidate_boxes.push(xyxy);
            candidate_scores.push(max_score);
            candidate_class_ids.push(max_class_id);
        }
    }

    if candidate_boxes.is_empty() {
        return Ok(Vec::new());
    }

    // NMS の実行
    let selected_idx = yolo_nms(&candidate_boxes, &candidate_scores, iou_threshold);

    let mut results = Vec::new();
    for idx in selected_idx {
        let box_coords = candidate_boxes[idx];
        let score = candidate_scores[idx];
        let class_id = candidate_class_ids[idx];

        let (x0, y0) = xy_postprocess(box_coords[0], box_coords[1], old_size, new_size);
        let (x1, y1) = xy_postprocess(box_coords[2], box_coords[3], old_size, new_size);

        results.push(DetectionResult {
            bbox: (x0, y0, x1, y1),
            label: labels[class_id].clone(),
            score,
            mask: None,
        });
    }

    Ok(results)
}

/// End-to-Endモデル (YOLOv10など) の出力後処理。
///
/// `output` テンソルは `[box_cnt, 6]` の形状を持ち、
/// 各行は `[x1, y1, x2, y2, score, class_id]` です。
pub fn postprocess_end2end_yolo(
    output: &ndarray::Array2<f32>,
    conf_threshold: f32,
    iou_threshold: f32,
    old_size: (u32, u32),
    new_size: (u32, u32),
    labels: &[String],
) -> Result<Vec<DetectionResult>, InferenceError> {
    if output.shape()[1] != 6 {
        return Err(InferenceError::InvalidShape(format!(
            "Expected end-to-end output dimension of 6, found {}",
            output.shape()[1]
        )));
    }

    let box_cnt = output.shape()[0];
    let mut candidate_boxes = Vec::new();
    let mut candidate_scores = Vec::new();
    let mut candidate_class_ids = Vec::new();

    for row in 0..box_cnt {
        let score = output[[row, 4]];
        if score > conf_threshold {
            let x1 = output[[row, 0]];
            let y1 = output[[row, 1]];
            let x2 = output[[row, 2]];
            let y2 = output[[row, 3]];
            let class_id = output[[row, 5]] as usize;

            candidate_boxes.push([x1, y1, x2, y2]);
            candidate_scores.push(score);
            candidate_class_ids.push(class_id);
        }
    }

    if candidate_boxes.is_empty() {
        return Ok(Vec::new());
    }

    // End-to-EndモデルでもさらにNMSをかける (Python側 _end2end_postprocess と等価)
    let selected_idx = yolo_nms(&candidate_boxes, &candidate_scores, iou_threshold);

    let mut results = Vec::new();
    for idx in selected_idx {
        let box_coords = candidate_boxes[idx];
        let score = candidate_scores[idx];
        let class_id = candidate_class_ids[idx];

        let (x0, y0) = xy_postprocess(box_coords[0], box_coords[1], old_size, new_size);
        let (x1, y1) = xy_postprocess(box_coords[2], box_coords[3], old_size, new_size);

        results.push(DetectionResult {
            bbox: (x0, y0, x1, y1),
            label: labels[class_id].clone(),
            score,
            mask: None,
        });
    }

    Ok(results)
}

/// RT-DETR モデルの出力後処理。
///
/// `output` テンソルは `[box_cnt, 4 + class_count]` の形状を持ちます。
/// 出力座標は `[0.0, 1.0]` に正規化されています。
pub fn postprocess_rtdetr(
    output: &ndarray::Array2<f32>,
    _conf_threshold: f32,
    _iou_threshold: f32,
    _old_size: (u32, u32),
) -> Result<ndarray::Array2<f32>, InferenceError> {
    // Python側 _rtdetr_postprocess と等価な NMS 後処理の準備として、転置を行って postprocess_nms_yolo に渡せる形式にします
    // 形状: [4 + class_count, box_cnt]
    let transposed = output.t().to_owned();
    Ok(transposed)
}

// スレッドセーフな ONNX セッションのキャッシュ
static SESSION_CACHE: Lazy<Mutex<HashMap<(String, String), ort::session::Session>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// 汎用 YOLOv8 物体検出を実行します。
pub fn yolo_predict(
    image: &DynamicImage,
    repo_id: &str,
    model_name: &str,
    max_infer_size: (u32, u32),
    conf_threshold: f32,
    iou_threshold: f32,
    labels: &[String],
) -> Result<Vec<DetectionResult>, InferenceError> {
    let mut cache = SESSION_CACHE.lock().map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let key = (repo_id.to_string(), model_name.to_string());

    if !cache.contains_key(&key) {
        // ONNX モデルファイルをダウンロード
        let model_filename = format!("{}/model.onnx", model_name);
        let model_path = hf_hub_download(repo_id, &model_filename, None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;

        let session = create_onnx_session(model_path)?;
        cache.insert(key.clone(), session);
    }

    let session = cache.get_mut(&key).unwrap();

    // 前処理
    let (input_tensor, old_size, new_size) = preprocess_image_yolo(image, max_infer_size, false, 32)?;

    // 推論の実行
    let outputs = run_onnx_session(session, "images", &input_tensor)?;
    
    let output0 = outputs.get("output0")
        .ok_or_else(|| InferenceError::InvalidShape("Missing output0".to_string()))?;

    let (shape, slice) = output0.try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let output0_view = ndarray::ArrayView::from_shape(shape_usize, slice)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;

    // バッチ次元 0 を取り出す
    let output_array = output0_view.index_axis(ndarray::Axis(0), 0).to_owned();

    // NMS後処理
    let results = postprocess_nms_yolo(&output_array, conf_threshold, iou_threshold, old_size, new_size, labels)?;
    Ok(results)
}
