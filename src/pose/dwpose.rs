use std::sync::Mutex;

use image::{DynamicImage, GenericImageView, Rgb};
use ndarray::{Array4, IxDyn};
use once_cell::sync::Lazy;

use crate::detect::person::detect_person;
use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{InferenceError, create_onnx_session};

use super::PoseError;
use super::format::{OP18KeyPoint, OP18KeyPointSet};

static DWPOSE_SESSION: Lazy<Mutex<Option<ort::session::Session>>> = Lazy::new(|| Mutex::new(None));

fn ensure_model() -> Result<(), InferenceError> {
    let mut lock = DWPOSE_SESSION
        .lock()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    if lock.is_none() {
        let path = hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
        *lock = Some(create_onnx_session(path)?);
    }
    Ok(())
}

fn rot_rad(angle_deg: f32) -> f32 {
    angle_deg * std::f32::consts::PI / 180.0
}

fn rotate_point(pt: [f32; 2], angle_rad: f32) -> [f32; 2] {
    let sn = angle_rad.sin();
    let cs = angle_rad.cos();
    [cs * pt[0] - sn * pt[1], sn * pt[0] + cs * pt[1]]
}

fn get_3rd_point(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let direction = [a[0] - b[0], a[1] - b[1]];
    [b[0] - direction[1], b[1] + direction[0]]
}

fn get_warp_matrix(
    center: [f32; 2],
    scale: [f32; 2],
    rot: f32,
    output_size: (u32, u32),
    shift: [f32; 2],
    inv: bool,
) -> [[f32; 3]; 2] {
    let src_w = scale[0];
    let dst_w = output_size.0 as f32;
    let dst_h = output_size.1 as f32;

    let rot_rad = rot_rad(rot);

    let src_dir = rotate_point([0.0, src_w * -0.5], rot_rad);
    let dst_dir = [0.0, dst_w * -0.5];

    let mut src = [[0.0f32; 2]; 3];
    src[0] = [
        center[0] + scale[0] * shift[0],
        center[1] + scale[1] * shift[1],
    ];
    src[1] = [
        center[0] + src_dir[0] + scale[0] * shift[0],
        center[1] + src_dir[1] + scale[1] * shift[1],
    ];
    src[2] = get_3rd_point(src[0], src[1]);

    let mut dst = [[0.0f32; 2]; 3];
    dst[0] = [dst_w * 0.5, dst_h * 0.5];
    dst[1] = [dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]];
    dst[2] = get_3rd_point(dst[0], dst[1]);

    if inv {
        std::mem::swap(&mut src, &mut dst);
    }

    get_affine_transform(&src, &dst)
}

fn get_affine_transform(src: &[[f32; 2]; 3], dst: &[[f32; 2]; 3]) -> [[f32; 3]; 2] {
    let x1 = src[0][0];
    let y1 = src[0][1];
    let x2 = src[1][0];
    let y2 = src[1][1];
    let x3 = src[2][0];
    let y3 = src[2][1];

    let u1 = dst[0][0];
    let v1 = dst[0][1];
    let u2 = dst[1][0];
    let v2 = dst[1][1];
    let u3 = dst[2][0];
    let v3 = dst[2][1];

    let det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    let det = if det == 0.0 { 1e-10 } else { det };

    let a = ((u2 - u1) * (y3 - y1) - (u3 - u1) * (y2 - y1)) / det;
    let b = ((x2 - x1) * (u3 - u1) - (x3 - x1) * (u2 - u1)) / det;
    let c = ((x2 * y3 - x3 * y2) * u1 + (x3 * y1 - x1 * y3) * u2 + (x1 * y2 - x2 * y1) * u3) / det;

    let d = ((v2 - v1) * (y3 - y1) - (v3 - v1) * (y2 - y1)) / det;
    let e = ((x2 - x1) * (v3 - v1) - (x3 - x1) * (v2 - v1)) / det;
    let f = ((x2 * y3 - x3 * y2) * v1 + (x3 * y1 - x1 * y3) * v2 + (x1 * y2 - x2 * y1) * v3) / det;

    [[a, b, c], [d, e, f]]
}

fn bbox_xyxy2cs(bbox: &[f32; 4], padding: f32) -> ([f32; 2], [f32; 2]) {
    let x1 = bbox[0];
    let y1 = bbox[1];
    let x2 = bbox[2];
    let y2 = bbox[3];

    let center = [(x1 + x2) * 0.5, (y1 + y2) * 0.5];
    let scale = [(x2 - x1) * padding, (y2 - y1) * padding];

    (center, scale)
}

fn fix_aspect_ratio(bbox_scale: [f32; 2], aspect_ratio: f32) -> [f32; 2] {
    let w = bbox_scale[0];
    let h = bbox_scale[1];
    if w > h * aspect_ratio {
        [w, w / aspect_ratio]
    } else {
        [h * aspect_ratio, h]
    }
}

fn warp_affine_image(
    img: &DynamicImage,
    matrix: &[[f32; 3]; 2],
    dest_w: u32,
    dest_h: u32,
) -> DynamicImage {
    let rgb = force_image_background(img, [255, 255, 255]);
    let (src_w, src_h) = rgb.dimensions();
    let src = rgb.to_rgb8();

    let a = matrix[0][0] as f64;
    let b = matrix[0][1] as f64;
    let c = matrix[0][2] as f64;
    let d = matrix[1][0] as f64;
    let e = matrix[1][1] as f64;
    let f = matrix[1][2] as f64;

    let det = a * e - b * d;
    let det = if det == 0.0 { 1e-10 } else { det };

    let inv_a = e / det;
    let inv_b = -b / det;
    let inv_c = (b * f - e * c) / det;
    let inv_d = -d / det;
    let inv_e = a / det;
    let inv_f = (d * c - a * f) / det;

    let mut output = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(dest_w, dest_h);

    for y_out in 0..dest_h {
        for x_out in 0..dest_w {
            let x_src = inv_a * x_out as f64 + inv_b * y_out as f64 + inv_c;
            let y_src = inv_d * x_out as f64 + inv_e * y_out as f64 + inv_f;

            if x_src >= 0.0
                && y_src >= 0.0
                && x_src < src_w as f64 - 1.0
                && y_src < src_h as f64 - 1.0
            {
                let x0 = x_src.floor() as u32;
                let y0 = y_src.floor() as u32;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);

                let dx = x_src - x0 as f64;
                let dy = y_src - y0 as f64;

                let p00 = src.get_pixel(x0, y0);
                let p10 = src.get_pixel(x1, y0);
                let p01 = src.get_pixel(x0, y1);
                let p11 = src.get_pixel(x1, y1);

                let r = ((1.0 - dx) * (1.0 - dy) * p00[0] as f64
                    + dx * (1.0 - dy) * p10[0] as f64
                    + (1.0 - dx) * dy * p01[0] as f64
                    + dx * dy * p11[0] as f64)
                    .round() as u8;
                let g = ((1.0 - dx) * (1.0 - dy) * p00[1] as f64
                    + dx * (1.0 - dy) * p10[1] as f64
                    + (1.0 - dx) * dy * p01[1] as f64
                    + dx * dy * p11[1] as f64)
                    .round() as u8;
                let b = ((1.0 - dx) * (1.0 - dy) * p00[2] as f64
                    + dx * (1.0 - dy) * p10[2] as f64
                    + (1.0 - dx) * dy * p01[2] as f64
                    + dx * dy * p11[2] as f64)
                    .round() as u8;

                output.put_pixel(x_out, y_out, Rgb([r, g, b]));
            }
        }
    }

    DynamicImage::ImageRgb8(output)
}

fn top_down_affine(
    img: &DynamicImage,
    bbox_center: [f32; 2],
    bbox_scale: [f32; 2],
    input_size: (u32, u32),
) -> (DynamicImage, [f32; 2]) {
    let (w, h) = input_size;
    let aspect_ratio = w as f32 / h as f32;

    let bbox_scale = fix_aspect_ratio(bbox_scale, aspect_ratio);

    let warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0.0, (w, h), [0.0, 0.0], false);
    let warped = warp_affine_image(img, &warp_mat, w, h);

    (warped, bbox_scale)
}

fn preprocess_dwpose(
    img: &DynamicImage,
    out_bboxes: &[[f32; 4]],
    input_size: (u32, u32),
) -> (Vec<Array4<f32>>, Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let mut out_img = Vec::new();
    let mut out_center = Vec::new();
    let mut out_scale = Vec::new();

    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];

    for bbox in out_bboxes {
        let (center, scale) = bbox_xyxy2cs(bbox, 1.25);
        let (resized, fixed_scale) = top_down_affine(img, center, scale, input_size);
        let (w, h) = resized.dimensions();

        let rgb = resized.to_rgb8();
        let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            tensor[[0, 0, y as usize, x as usize]] = (r - mean[0]) / std[0];
            tensor[[0, 1, y as usize, x as usize]] = (g - mean[1]) / std[1];
            tensor[[0, 2, y as usize, x as usize]] = (b - mean[2]) / std[2];
        }

        out_img.push(tensor);
        out_center.push(center);
        out_scale.push(fixed_scale);
    }

    (out_img, out_center, out_scale)
}

fn get_simcc_maximum(
    simcc_x: &ndarray::ArrayViewD<f32>,
    simcc_y: &ndarray::ArrayViewD<f32>,
) -> (ndarray::Array2<f32>, ndarray::Array1<f32>) {
    let shape_x = simcc_x.shape();
    let k = shape_x[0];
    let wx = shape_x[1];

    let simcc_x_flat = simcc_x.as_slice().unwrap();
    let simcc_y_flat = simcc_y.as_slice().unwrap();

    let mut locs = ndarray::Array2::<f32>::zeros((k, 2));
    let mut vals = ndarray::Array1::<f32>::zeros(k);

    for i in 0..k {
        let x_start = i * wx;

        let (mut max_x_idx, mut max_x_val) = (0usize, -1.0f32);
        let (mut max_y_idx, mut max_y_val) = (0usize, -1.0f32);

        for j in 0..wx {
            let vx = simcc_x_flat[x_start + j];
            let vy = simcc_y_flat[x_start + j];
            if vx > max_x_val {
                max_x_val = vx;
                max_x_idx = j;
            }
            if vy > max_y_val {
                max_y_val = vy;
                max_y_idx = j;
            }
        }

        let final_val = if max_x_val > max_y_val {
            max_y_val
        } else {
            max_x_val
        };

        if final_val <= 0.0 {
            locs[[i, 0]] = -1.0;
            locs[[i, 1]] = -1.0;
        } else {
            locs[[i, 0]] = max_x_idx as f32;
            locs[[i, 1]] = max_y_idx as f32;
        }
        vals[i] = final_val;
    }

    (locs, vals)
}

fn output_decode(
    simcc_x: &ndarray::ArrayViewD<f32>,
    simcc_y: &ndarray::ArrayViewD<f32>,
    simcc_split_ratio: f32,
) -> (ndarray::Array2<f32>, ndarray::Array1<f32>) {
    let (mut keypoints, scores) = get_simcc_maximum(simcc_x, simcc_y);
    keypoints.mapv_inplace(|v| v / simcc_split_ratio);
    (keypoints, scores)
}

fn dwpose_postprocess(
    outputs: &[Vec<ndarray::Array<f32, IxDyn>>],
    model_input_size: (u32, u32),
    center: &[[f32; 2]],
    scale: &[[f32; 2]],
    simcc_split_ratio: f32,
) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    let w = model_input_size.0 as f32;
    let h = model_input_size.1 as f32;

    let mut all_rescaled = Vec::new();
    let mut all_scores = Vec::new();

    for (i, output_pair) in outputs.iter().enumerate() {
        let simcc_x = output_pair[0].view().into_dyn();
        let simcc_y = output_pair[1].view().into_dyn();

        let (keypoints, scores) = output_decode(&simcc_x, &simcc_y, simcc_split_ratio);

        let ct = center[i];
        let sc = scale[i];

        let n = keypoints.shape()[0];
        let mut rescaled = ndarray::Array2::<f32>::zeros((n, 2));
        for j in 0..n {
            rescaled[[j, 0]] = keypoints[[j, 0]] / w * sc[0] + ct[0] - sc[0] / 2.0;
            rescaled[[j, 1]] = keypoints[[j, 1]] / h * sc[1] + ct[1] - sc[1] / 2.0;
        }

        all_rescaled.push(rescaled);
        all_scores.push(scores);
    }

    let n_people = all_rescaled.len();
    let n_kps = all_rescaled[0].shape()[0];
    let mut flat_key = ndarray::Array2::<f32>::zeros((n_people, n_kps * 2));
    let mut flat_score = ndarray::Array2::<f32>::zeros((n_people, n_kps));
    for i in 0..n_people {
        for j in 0..n_kps {
            flat_key[[i, j * 2]] = all_rescaled[i][[j, 0]];
            flat_key[[i, j * 2 + 1]] = all_rescaled[i][[j, 1]];
            flat_score[[i, j]] = all_scores[i][j];
        }
    }

    (flat_key, flat_score)
}

fn dwpose_reorder_body_points(
    keypoints: &ndarray::Array2<f32>,
    scores: &ndarray::Array2<f32>,
) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    let shape = keypoints.shape();
    let n = shape[0];

    let n_orig = 133;
    let n_new = 134;

    let mut new_kps = ndarray::Array2::<f32>::zeros((n, n_new * 2));
    let mut new_scores = ndarray::Array2::<f32>::zeros((n, n_new));

    for i in 0..n {
        for j in 0..n_orig {
            new_kps[[i, j * 2]] = keypoints[[i, j * 2]];
            new_kps[[i, j * 2 + 1]] = keypoints[[i, j * 2 + 1]];
            new_scores[[i, j]] = scores[[i, j]];
        }

        let neck_x = (keypoints[[i, 5 * 2]] + keypoints[[i, 6 * 2]]) * 0.5;
        let neck_y = (keypoints[[i, 5 * 2 + 1]] + keypoints[[i, 6 * 2 + 1]]) * 0.5;
        new_kps[[i, 17 * 2]] = neck_x;
        new_kps[[i, 17 * 2 + 1]] = neck_y;
        let sh5_ok = scores[[i, 5]] > 0.3;
        let sh6_ok = scores[[i, 6]] > 0.3;
        let neck_score = if sh5_ok && sh6_ok {
            (scores[[i, 5]] + scores[[i, 6]]) * 0.5
        } else {
            0.0
        };
        new_scores[[i, 17]] = neck_score;

        let mmpose_idx: [usize; 15] = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3];
        let openpose_idx: [usize; 15] = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17];

        for (&from, &to) in mmpose_idx.iter().zip(openpose_idx.iter()) {
            new_kps[[i, to * 2]] = new_kps[[i, from * 2]];
            new_kps[[i, to * 2 + 1]] = new_kps[[i, from * 2 + 1]];
            new_scores[[i, to]] = new_scores[[i, from]];
        }
    }

    (new_kps, new_scores)
}

fn split_data(
    keypoints: &ndarray::Array2<f32>,
    scores: &ndarray::Array2<f32>,
    bboxes: &[[f32; 4]],
) -> Vec<OP18KeyPointSet> {
    let shape = keypoints.shape();
    let n = shape[0];
    let n_total = shape[1];

    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let mut kps = Vec::with_capacity(n_total / 2);
        for j in 0..(n_total / 2) {
            kps.push(OP18KeyPoint {
                x: keypoints[[i, j * 2]],
                y: keypoints[[i, j * 2 + 1]],
                score: scores[[i, j]],
            });
        }
        results.push(OP18KeyPointSet {
            keypoints: kps,
            bbox: bboxes[i],
        });
    }

    results
}

/// Estimate pose keypoints for all people in an image using DWPose (RTMPose-x).
///
/// This is a 2-stage pipeline:
/// 1. Detect people in the image (using `detect_person`)
/// 2. For each person, crop, affine-transform, run RTMPose model, and project keypoints back
///
/// # Arguments
/// * `image` - Input image
/// * `auto_detect` - If true, use person detection; if false, use the provided bboxes or full image
/// * `out_bboxes` - Optional pre-detected bounding boxes
/// * `person_detect_cfgs` - Detection config overrides (level, version, conf_threshold, iou_threshold)
///
/// # Returns
/// A vector of `OP18KeyPointSet`, one per detected person
pub fn dwpose_estimate(
    image: &DynamicImage,
    auto_detect: bool,
    out_bboxes: Option<Vec<[f32; 4]>>,
    person_detect_cfgs: Option<serde_json::Value>,
) -> Result<Vec<OP18KeyPointSet>, PoseError> {
    ensure_model()?;

    let rgb = force_image_background(image, [255, 255, 255]);

    let bboxes: Vec<[f32; 4]> = if let Some(bboxes) = out_bboxes {
        bboxes
    } else if auto_detect {
        let level = person_detect_cfgs
            .as_ref()
            .and_then(|c| c.get("level"))
            .and_then(|v| v.as_str())
            .unwrap_or("m");
        let version = person_detect_cfgs
            .as_ref()
            .and_then(|c| c.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("v1.1");
        let conf_threshold = person_detect_cfgs
            .as_ref()
            .and_then(|c| c.get("conf_threshold"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25) as f32;
        let iou_threshold = person_detect_cfgs
            .as_ref()
            .and_then(|c| c.get("iou_threshold"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let detections = detect_person(&rgb, level, version, conf_threshold, iou_threshold)
            .map_err(PoseError::Inference)?;

        detections
            .into_iter()
            .map(|d| {
                [
                    d.bbox.0 as f32,
                    d.bbox.1 as f32,
                    d.bbox.2 as f32,
                    d.bbox.3 as f32,
                ]
            })
            .collect()
    } else {
        vec![[0.0, 0.0, rgb.width() as f32, rgb.height() as f32]]
    };

    if bboxes.is_empty() {
        return Ok(Vec::new());
    }

    let model_h = 384;
    let model_w = 288;
    let model_input_size = (model_w, model_h);

    let (tensors, centers, scales) = preprocess_dwpose(&rgb, &bboxes, model_input_size);

    let mut lock = DWPOSE_SESSION
        .lock()
        .map_err(|e| PoseError::Inference(InferenceError::Initialization(e.to_string())))?;
    let session = lock.as_mut().unwrap();

    let mut all_outputs = Vec::new();
    for tensor in &tensors {
        let tensor_val = ort::value::Tensor::from_array(tensor.clone())
            .map_err(|e| PoseError::Inference(InferenceError::Initialization(e.to_string())))?;
        let outputs = session
            .run(ort::inputs![tensor_val])
            .map_err(|e| PoseError::Inference(InferenceError::Initialization(e.to_string())))?;

        let mut person_outputs = Vec::new();
        for output_val in outputs.iter() {
            let (shape, slice) = output_val
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| PoseError::Inference(InferenceError::Initialization(e.to_string())))?;
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let arr_view = ndarray::ArrayView::from_shape(shape_usize, slice)
                .map_err(|e| PoseError::Inference(InferenceError::Initialization(e.to_string())))?;
            let arr = arr_view.into_owned();
            let squeezed = arr.index_axis(ndarray::Axis(0), 0).to_owned();
            person_outputs.push(squeezed);
        }
        all_outputs.push(person_outputs);
    }

    let (keypoints, scores) =
        dwpose_postprocess(&all_outputs, model_input_size, &centers, &scales, 2.0);

    if keypoints.shape()[0] > 0 {
        let (keypoints, scores) = dwpose_reorder_body_points(&keypoints, &scores);
        Ok(split_data(&keypoints, &scores, &bboxes))
    } else {
        Ok(Vec::new())
    }
}
