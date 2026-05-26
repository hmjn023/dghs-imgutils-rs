//! DBNet++ モデルを用いた画像内のテキスト領域検出処理を提供します。
//! OpenCV を使用せず、純粋な Rust (imageproc) による等価な輪郭抽出および領域平均スコアフィルタリングを実装しています。

use crate::detect::base::Detection;
use crate::hub::hf_hub_download;
use crate::inference::{InferenceError, create_onnx_session};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, Array4};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

static TEXT_SESSION_CACHE: Lazy<Mutex<HashMap<String, ort::session::Session>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// テキスト検出用のヒートマップ正規化
fn normalize_text_input(
    img: &DynamicImage,
    padded_w: u32,
    padded_h: u32,
    orig_w: u32,
    orig_h: u32,
) -> Array4<f32> {
    let mut array = Array4::zeros((1, 3, padded_h as usize, padded_w as usize));
    let rgb_img = img.to_rgb8();

    let mean = [0.48145466f32, 0.4578275f32, 0.40821073f32];
    let std = [0.26862954f32, 0.26130258f32, 0.27577711f32];

    for y in 0..orig_h {
        for x in 0..orig_w {
            let pixel = rgb_img.get_pixel(x, y);
            let r = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            let g = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            let b = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];

            array[[0, 0, y as usize, x as usize]] = r;
            array[[0, 1, y as usize, x as usize]] = g;
            array[[0, 2, y as usize, x as usize]] = b;
        }
    }

    // パディング部分（右側と下側）はすでに 0 で初期化されているためそのまま

    array
}

/// 画像内のテキスト領域を検出します。
///
/// # 引数
///
/// * `image` - 入力画像
/// * `model` - 使用する DBNet++ モデル名（デフォルト: "dbnetpp_resnet50_fpnc_1200e_icdar2015"）
/// * `threshold` - 信頼度しきい値（デフォルト: 0.05）
/// * `max_area_size` - 推論時の最大面積制限（最短辺がこれより大きい場合縮小。デフォルト: 640）
pub fn detect_text(
    image: &DynamicImage,
    model: &str,
    threshold: f32,
    max_area_size: Option<u32>,
) -> Result<Vec<Detection>, InferenceError> {
    // 1. 必要に応じてアスペクト比を維持しつつ面積制限リサイズ
    let (resized_image, r) = if let Some(limit) = max_area_size {
        let area = image.width() * image.height();
        let limit_area = limit * limit;
        if area > limit_area {
            let ratio = (area as f32 / limit_area as f32).sqrt();
            let nw = (image.width() as f32 / ratio) as u32;
            let nh = (image.height() as f32 / ratio) as u32;
            let resized = image.resize_exact(nw, nh, image::imageops::FilterType::Triangle);
            (resized, ratio)
        } else {
            (image.clone(), 1.0f32)
        }
    } else {
        (image.clone(), 1.0f32)
    };

    let orig_w = resized_image.width();
    let orig_h = resized_image.height();

    // 2. 幅と高さを 32 の倍数に切り上げパディング
    let align = 32u32;
    let padded_w = if orig_w % align != 0 {
        orig_w + (align - orig_w % align)
    } else {
        orig_w
    };
    let padded_h = if orig_h % align != 0 {
        orig_h + (align - orig_h % align)
    } else {
        orig_h
    };

    // 3. 前処理
    let input_tensor = normalize_text_input(&resized_image, padded_w, padded_h, orig_w, orig_h);

    // 4. ONNX 推論の実行
    let mut cache = TEXT_SESSION_CACHE
        .lock()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    if !cache.contains_key(model) {
        let model_filename = format!("{}/end2end.onnx", model);
        let path = hf_hub_download("deepghs/text_detection", &model_filename, None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
        let session = create_onnx_session(path)?;
        cache.insert(model.to_string(), session);
    }
    let session = cache.get_mut(model).unwrap();

    let inputs = ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor)?];
    let outputs = session.run(inputs)?;
    let output = outputs
        .get("output")
        .ok_or_else(|| InferenceError::InvalidShape("Missing output from DBNet++".to_string()))?;

    let (shape, slice) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let output_view = ndarray::ArrayView::from_shape(shape_usize, slice)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    // 出力形状: [1, 1, padded_h, padded_w] または [1, padded_h, padded_w]
    // 形状不整合を防ぐため、最後の二次元から元のリサイズサイズ (orig_h, orig_w) 部分をスライスしてヒートマップ Array2 を作成
    let mut heatmap = Array2::zeros((orig_h as usize, orig_w as usize));
    let view_shape = output_view.shape();

    for y in 0..orig_h as usize {
        for x in 0..orig_w as usize {
            // パディングを除く元の有効領域のみを抽出
            let val = match view_shape.len() {
                4 => output_view[[0, 0, y, x]],
                3 => output_view[[0, y, x]],
                _ => {
                    return Err(InferenceError::InvalidShape(
                        "Unexpected DBNet++ output dimensions".to_string(),
                    ));
                }
            };
            heatmap[[y, x]] = val;
        }
    }

    // 5. 輪郭抽出のための二値化画像 (GrayImage) の作成
    let mut binary_img: GrayImage = ImageBuffer::new(orig_w, orig_h);
    for y in 0..orig_h {
        for x in 0..orig_w {
            let val = heatmap[[y as usize, x as usize]];
            // Python 版は (heatmap * 255.0).astype(np.uint8) であり、
            // findContours は 0 より大きいピクセルを全て前景として処理するため、
            // 0.0 より大きい値を 255 とした二値マスクを作成します。
            let pixel_val = if val > 0.0 { 255 } else { 0 };
            binary_img.put_pixel(x, y, Luma([pixel_val]));
        }
    }

    // imageproc による輪郭検出
    // find_contours は全ての輪郭（Outer / Hole）を抽出します
    let contours = imageproc::contours::find_contours::<u32>(&binary_img);

    let mut bboxes = Vec::new();
    for contour in contours {
        // 最外郭（Outer）の輪郭のみを対象とする（OpenCV の RETR_EXTERNAL と等価）
        if contour.border_type == imageproc::contours::BorderType::Outer {
            let points = &contour.points;
            if points.is_empty() {
                continue;
            }

            let mut min_x = u32::MAX;
            let mut min_y = u32::MAX;
            let mut max_x = u32::MIN;
            let mut max_y = u32::MIN;

            for pt in points {
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
            }

            // バウンディングボックスの切り出し
            let x0 = min_x;
            let y0 = min_y;
            let x1 = max_x.min(orig_w - 1);
            let y1 = max_y.min(orig_h - 1);

            if x1 <= x0 || y1 <= y0 {
                continue;
            }

            // 領域内のヒートマップ値の平均（スコア）を計算
            let mut sum_score = 0.0f32;
            let mut count = 0f32;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    sum_score += heatmap[[y as usize, x as usize]];
                    count += 1.0;
                }
            }
            let score = sum_score / (count + 1e-6);

            if score >= threshold {
                bboxes.push(((x0, y0, x1, y1), score));
            }
        }
    }

    // 6. 縮小比率を用いてオリジナル画像サイズ座標に復元
    let mut detections = Vec::new();
    for ((x0, y0, x1, y1), score) in bboxes {
        let x0_orig = (x0 as f32 * r).round() as u32;
        let y0_orig = (y0 as f32 * r).round() as u32;
        let x1_orig = (x1 as f32 * r).round() as u32;
        let y1_orig = (y1 as f32 * r).round() as u32;

        detections.push(Detection {
            bbox: (x0_orig, y0_orig, x1_orig, y1_orig),
            label: "text".to_string(),
            score,
            mask: None,
        });
    }

    // スコア降順でソート
    detections.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(detections)
}
