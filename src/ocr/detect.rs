use std::collections::HashMap;
use std::sync::Mutex;

use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use imageproc::contours::{BorderType, find_contours};
use ndarray::{Array2, Array4};
use once_cell::sync::Lazy;

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{InferenceError, create_onnx_session};
use crate::ocr::OcrError;
use crate::ocr::constants::*;

static DET_SESSION: Lazy<Mutex<Option<ort::session::Session>>> = Lazy::new(|| Mutex::new(None));

pub type BBox = (i32, i32, i32, i32);

pub fn detect_text_with_paddleocr(image: &DynamicImage) -> Result<Vec<(BBox, f32)>, OcrError> {
    let mut lock = DET_SESSION
        .lock()
        .map_err(|e| OcrError::Inference(InferenceError::Initialization(e.to_string())))?;
    if lock.is_none() {
        let path = hf_hub_download(DET_REPO_ID, DET_MODEL_PATH, None, None)?;
        *lock = Some(create_onnx_session(path)?);
    }
    let session = lock.as_mut().unwrap();

    let (input_tensor, orig_w, orig_h, ratio_w, ratio_h) = preprocess_det(image)?;

    let inputs = ort::inputs![DET_INPUT_NAME => ort::value::Tensor::from_array(input_tensor)?];
    let outputs = session.run(inputs)?;

    let prob_map = extract_prob_map(&outputs, orig_h, orig_w)?;

    let bboxes = db_postprocess(&prob_map, orig_w, orig_h, ratio_w, ratio_h)?;

    Ok(bboxes)
}

fn preprocess_det(image: &DynamicImage) -> Result<(Array4<f32>, u32, u32, f32, f32), OcrError> {
    let (orig_w, orig_h) = image.dimensions();

    let max_side = orig_w.max(orig_h);
    let ratio = if max_side > DET_LIMIT_SIDE_LEN {
        DET_LIMIT_SIDE_LEN as f32 / max_side as f32
    } else {
        1.0
    };

    let new_w = (orig_w as f32 * ratio).round() as u32;
    let new_h = (orig_h as f32 * ratio).round() as u32;

    let resized = image.resize_exact(
        new_w.max(1),
        new_h.max(1),
        image::imageops::FilterType::Triangle,
    );

    let align = 32u32;
    let padded_w = ((new_w + align - 1) / align) * align;
    let padded_h = ((new_h + align - 1) / align) * align;

    let mut rgb = resized.to_rgb8();
    if padded_w != new_w || padded_h != new_h {
        let mut canvas = ImageBuffer::from_pixel(padded_w, padded_h, image::Rgb([0, 0, 0]));
        image::imageops::overlay(&mut canvas, &rgb, 0, 0);
        rgb = canvas;
    }

    let padded_img = DynamicImage::ImageRgb8(rgb);
    let tensor = to_ndarray_chw(&padded_img, &DET_IMG_NORMALIZE_MEAN, &DET_IMG_NORMALIZE_STD)?;

    Ok((tensor, orig_w, orig_h, 1.0 / ratio, 1.0 / ratio))
}

fn extract_prob_map(
    outputs: &ort::session::SessionOutputs,
    h: u32,
    w: u32,
) -> Result<Array2<f32>, OcrError> {
    let output_val = outputs
        .iter()
        .next()
        .ok_or_else(|| {
            OcrError::Inference(InferenceError::InvalidShape(
                "Missing output from PaddleOCR detection".to_string(),
            ))
        })?
        .1;

    let (shape, slice) = output_val
        .try_extract_tensor::<f32>()
        .map_err(|e| OcrError::Inference(InferenceError::Initialization(e.to_string())))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    let output_h = shape_usize.get(shape_usize.len() - 2).copied().unwrap_or(0);
    let output_w = shape_usize.get(shape_usize.len() - 1).copied().unwrap_or(0);

    let stride_h = h as usize;
    let stride_w = w as usize;

    let mut prob_map = Array2::zeros((stride_h, stride_w));
    for y in 0..stride_h.min(output_h) {
        for x in 0..stride_w.min(output_w) {
            let val = match shape_usize.len() {
                4 => slice[y * output_w + x],
                _ => slice[y * output_w + x],
            };
            prob_map[[y, x]] = val;
        }
    }

    Ok(prob_map)
}

fn db_postprocess(
    prob_map: &Array2<f32>,
    orig_w: u32,
    orig_h: u32,
    ratio_w: f32,
    ratio_h: f32,
) -> Result<Vec<(BBox, f32)>, OcrError> {
    let h = prob_map.shape()[0];
    let w = prob_map.shape()[1];

    let mut binary: GrayImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let val = if prob_map[[y, x]] > DET_THRESHOLD {
                255u8
            } else {
                0u8
            };
            binary.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    let contours = find_contours::<u32>(&binary);

    let mut results = Vec::new();
    for contour in contours {
        if contour.border_type != BorderType::Outer {
            continue;
        }
        let points = &contour.points;
        if points.len() < 4 {
            continue;
        }

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        for pt in points {
            min_x = min_x.min(pt.x as i32);
            min_y = min_y.min(pt.y as i32);
            max_x = max_x.max(pt.x as i32);
            max_y = max_y.max(pt.y as i32);
        }

        if max_x <= min_x || max_y <= min_y {
            continue;
        }

        let mut sum_score = 0.0f32;
        let mut count = 0u32;
        for y in min_y as u32..=max_y as u32 {
            for x in min_x as u32..=max_x as u32 {
                if y < h as u32 && x < w as u32 {
                    sum_score += prob_map[[y as usize, x as usize]];
                    count += 1;
                }
            }
        }
        let score = if count > 0 {
            sum_score / count as f32
        } else {
            0.0
        };

        if score < DET_BOX_THRESHOLD {
            continue;
        }

        let x0 = (min_x as f32 * ratio_w).round() as i32;
        let y0 = (min_y as f32 * ratio_h).round() as i32;
        let x1 = (max_x as f32 * ratio_w).round() as i32;
        let y1 = (max_y as f32 * ratio_h).round() as i32;

        let x0 = x0.max(0).min(orig_w as i32 - 1);
        let y0 = y0.max(0).min(orig_h as i32 - 1);
        let x1 = x1.max(0).min(orig_w as i32 - 1);
        let y1 = y1.max(0).min(orig_h as i32 - 1);

        if x1 <= x0 || y1 <= y0 {
            continue;
        }

        results.push(((x0, y0, x1, y1), score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if results.len() > DET_MAX_CANDIDATES {
        results.truncate(DET_MAX_CANDIDATES);
    }

    Ok(results)
}
