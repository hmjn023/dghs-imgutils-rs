use std::collections::HashMap;
use std::sync::Mutex;

use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array2, Array4};
use once_cell::sync::Lazy;
use ort::session::Session;

use crate::generic::GenericError;
use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{create_onnx_session, yolo_nms, yolo_xywh2xyxy};

/// A segmentation detection result.
#[derive(Debug, Clone)]
pub struct SegDetection {
    /// Bounding box (x1, y1, x2, y2) in original image coordinates.
    pub bbox: (u32, u32, u32, u32),
    /// Class label string.
    pub label: String,
    /// Confidence score.
    pub score: f32,
    /// Binary mask (H, W) in original image size.
    pub mask: ndarray::Array2<f32>,
}

struct YOLOSegEntry {
    session: Session,
    max_infer_size: u32,
    labels: Vec<String>,
}

static YOLOSEG_CACHE: Lazy<Mutex<HashMap<String, YOLOSegEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn ensure_yoloseg_model(repo_id: &str, model_name: &str) -> Result<(), GenericError> {
    let key = format!("{}/{}", repo_id, model_name);
    {
        let cache = YOLOSEG_CACHE.lock().unwrap();
        if cache.contains_key(&key) {
            return Ok(());
        }
    }

    let model_path = hf_hub_download(repo_id, &format!("{}/model.onnx", model_name), None, None)?;

    let session = create_onnx_session(&model_path)?;

    // Extract metadata from the model (in a block to drop the borrow on session)
    let (max_infer_size, labels) = {
        let metadata = session.metadata()?;
        let imgsz_str = metadata
            .custom("imgsz")
            .unwrap_or_else(|| "640".to_string());
        let max_infer_size: u32 = serde_json::from_str::<u32>(&imgsz_str).unwrap_or(640);

        let names_str = metadata
            .custom("names")
            .unwrap_or_else(|| r#"{"0": "object"}"#.to_string());
        let names_map: HashMap<usize, String> =
            serde_json::from_str(&names_str).unwrap_or_else(|_| {
                let mut m = HashMap::new();
                m.insert(0, "object".to_string());
                m
            });
        let max_idx = names_map.keys().max().copied().unwrap_or(0);
        let labels: Vec<String> = (0..=max_idx)
            .map(|i| {
                names_map
                    .get(&i)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", i))
            })
            .collect();
        (max_infer_size, labels)
    };

    let entry = YOLOSegEntry {
        session,
        max_infer_size,
        labels,
    };

    YOLOSEG_CACHE.lock().unwrap().insert(key, entry);
    Ok(())
}

fn preprocess_yoloseg(
    image: &DynamicImage,
    max_infer_size: u32,
) -> Result<(Array4<f32>, (u32, u32), (u32, u32)), GenericError> {
    let old_w = image.width();
    let old_h = image.height();
    let new_w = max_infer_size;
    let new_h = max_infer_size;

    let rgb = image.to_rgb8();
    let resized = image::imageops::resize(&rgb, new_w, new_h, FilterType::Triangle);
    let dyn_img = DynamicImage::ImageRgb8(resized);
    let tensor = to_ndarray_chw(&dyn_img, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0])?;
    Ok((tensor, (old_w, old_h), (new_w, new_h)))
}

fn xy_postprocess(x: f32, y: f32, old_size: (u32, u32), new_size: (u32, u32)) -> (u32, u32) {
    let ow = old_size.0 as f32;
    let oh = old_size.1 as f32;
    let nw = new_size.0 as f32;
    let nh = new_size.1 as f32;
    let x = (x / nw * ow).clamp(0.0, ow).round() as u32;
    let y = (y / nh * oh).clamp(0.0, oh).round() as u32;
    (x, y)
}

fn scale_masks(
    mask: &ndarray::Array2<f32>,
    target_h: usize,
    target_w: usize,
) -> ndarray::Array2<f32> {
    let mh = mask.shape()[0];
    let mw = mask.shape()[1];
    let mut img_buf = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::new(mw as u32, mh as u32);
    for (x, y, p) in img_buf.enumerate_pixels_mut() {
        p[0] = (mask[[y as usize, x as usize]].clamp(0.0, 1.0) * 255.0) as u8;
    }
    let dyn_img = DynamicImage::ImageLuma8(img_buf);
    let resized = dyn_img.resize_exact(
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let resized_luma = resized.to_luma8();
    let mut result = ndarray::Array2::<f32>::zeros((target_h, target_w));
    for (x, y, p) in resized_luma.enumerate_pixels() {
        result[[y as usize, x as usize]] = p[0] as f32 / 255.0;
    }
    result
}

fn crop_mask(mask: &ndarray::Array2<f32>, bbox: (u32, u32, u32, u32)) -> ndarray::Array2<f32> {
    let (h, w) = mask.dim();
    let (x1, y1, x2, y2) = bbox;
    let mut cropped = mask.clone();
    for y in 0..h {
        for x in 0..w {
            if x < x1 as usize || x >= x2 as usize || y < y1 as usize || y >= y2 as usize {
                cropped[[y, x]] = 0.0;
            }
        }
    }
    cropped
}

/// Perform YOLO segmentation prediction on an image.
pub fn yolo_seg_predict(
    image: &DynamicImage,
    repo_id: &str,
    model_name: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<SegDetection>, GenericError> {
    ensure_yoloseg_model(repo_id, model_name)?;

    let key = format!("{}/{}", repo_id, model_name);
    let mut cache = YOLOSEG_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| GenericError::InvalidArgument("Model not found in cache".to_string()))?;

    let (input_tensor, old_size, new_size) = preprocess_yoloseg(image, entry.max_infer_size)?;

    let outputs = entry.session.run(ort::inputs![
        "images" => ort::value::Tensor::from_array(input_tensor)?
    ])?;

    let output0 = outputs
        .get("output0")
        .ok_or_else(|| GenericError::InvalidArgument("Missing output0".to_string()))?;
    let (_shape0, slice0) = output0.try_extract_tensor::<f32>()?;

    let output1 = outputs
        .get("output1")
        .ok_or_else(|| GenericError::InvalidArgument("Missing output1".to_string()))?;
    let (shape1, slice1) = output1.try_extract_tensor::<f32>()?;

    // Parse output0 shape: [1, 4 + num_classes + proto_dim, num_boxes]
    let shape0_vec: Vec<usize> = _shape0.iter().map(|&d| d as usize).collect();
    if shape0_vec.len() != 3 {
        return Err(GenericError::Shape(format!(
            "Expected output0 to be 3D, got {:?}",
            shape0_vec
        )));
    }
    let num_classes = entry.labels.len();
    let proto_dim = shape0_vec[1] - 4 - num_classes;
    let num_boxes = shape0_vec[2];

    // Reshape output0 to [4 + cls + pe, box_cnt]
    let output0_arr = ndarray::ArrayView2::from_shape((shape0_vec[1], shape0_vec[2]), slice0)?;
    let output0_owned = output0_arr.to_owned();

    // Parse output1 shape: proto masks [1, pe, ph, pw]
    let shape1_vec: Vec<usize> = shape1.iter().map(|&d| d as usize).collect();
    if shape1_vec.len() != 4 {
        return Err(GenericError::Shape(format!(
            "Expected output1 to be 4D, got {:?}",
            shape1_vec
        )));
    }
    let pe = shape1_vec[1];
    let ph = shape1_vec[2];
    let pw = shape1_vec[3];

    let protos_arr = ndarray::ArrayView4::from_shape(
        (shape1_vec[0], shape1_vec[1], shape1_vec[2], shape1_vec[3]),
        slice1,
    )?;
    // Take first batch: [pe, ph, pw]
    let protos = protos_arr.index_axis(ndarray::Axis(0), 0).to_owned();

    // --- NMS post-processing ---
    // max_scores: per box, max over classes
    let max_scores: Vec<f32> = (0..num_boxes)
        .map(|col| {
            (0..num_classes)
                .map(|c| output0_owned[[4 + c, col]])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();

    // Filter by conf_threshold
    let keep_cols: Vec<usize> = (0..num_boxes)
        .filter(|&col| max_scores[col] > conf_threshold)
        .collect();

    if keep_cols.is_empty() {
        return Ok(Vec::new());
    }

    // Extract boxes, scores, mask_embeddings for filtered columns
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut mask_embs = Vec::new();
    let mut class_ids = Vec::new();

    for &col in &keep_cols {
        let cx = output0_owned[[0, col]];
        let cy = output0_owned[[1, col]];
        let w = output0_owned[[2, col]];
        let h = output0_owned[[3, col]];
        let xyxy = yolo_xywh2xyxy(&[cx, cy, w, h]);
        boxes.push(xyxy);

        // Best class score
        let mut best_score = -1.0f32;
        let mut best_cid = 0usize;
        for c in 0..num_classes {
            let sc = output0_owned[[4 + c, col]];
            if sc > best_score {
                best_score = sc;
                best_cid = c;
            }
        }
        scores.push(best_score);
        class_ids.push(best_cid);

        let mut emb = Vec::with_capacity(proto_dim);
        for p in 0..proto_dim {
            emb.push(output0_owned[[4 + num_classes + p, col]]);
        }
        mask_embs.push(emb);
    }

    // NMS
    let selected = yolo_nms(&boxes, &scores, iou_threshold);

    let mut detections = Vec::new();
    for &idx in &selected {
        let box_coords = boxes[idx];
        let (x0, y0) = xy_postprocess(box_coords[0], box_coords[1], old_size, new_size);
        let (x1, y1) = xy_postprocess(box_coords[2], box_coords[3], old_size, new_size);

        // Decode mask: mask_embedding @ protos.reshape(pe, ph*pw) -> reshape(ph, pw)
        let mask_emb = ndarray::ArrayView1::from(&mask_embs[idx]);
        let protos_flat = protos.view().into_shape((pe, ph * pw))?;
        let mask_flat = mask_emb.dot(&protos_flat);
        let mut mask = mask_flat.into_shape_with_order((ph, pw))?;

        // Scale mask to original image size
        mask = scale_masks(&mask, old_size.1 as usize, old_size.0 as usize);
        // Crop to bbox
        mask = crop_mask(&mask, (x0, y0, x1, y1));
        // Binarize
        mask.mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });

        detections.push(SegDetection {
            bbox: (x0, y0, x1, y1),
            label: entry.labels[class_ids[idx]].clone(),
            score: scores[idx],
            mask,
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xy_postprocess() {
        let (x, y) = xy_postprocess(100.0, 50.0, (800, 600), (640, 480));
        assert_eq!(x, 125);
        assert_eq!(y, 63);
    }
}
