use std::sync::Mutex;

use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use once_cell::sync::Lazy;

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{InferenceError, get_or_create_session, lock_session};
use crate::ocr::OcrError;
use crate::ocr::constants::*;
use crate::ocr::detect::{BBox, detect_text_with_paddleocr};

static CHAR_DICT: Lazy<Mutex<Option<Vec<String>>>> = Lazy::new(|| Mutex::new(None));

pub fn ocr(image: &DynamicImage) -> Result<Vec<(BBox, String, f32)>, OcrError> {
    let _ = load_char_dict()?;

    let detections = detect_text_with_paddleocr(image)?;

    let mut results = Vec::new();
    for (bbox, det_score) in &detections {
        let cropped = crop_region(image, bbox);
        let text = recognize_text(&cropped)?;
        if !text.is_empty() {
            results.push((*bbox, text, *det_score));
        }
    }

    Ok(results)
}

fn crop_region(image: &DynamicImage, bbox: &BBox) -> DynamicImage {
    let (x0, y0, x1, y1) = *bbox;
    let x0 = x0.max(0) as u32;
    let y0 = y0.max(0) as u32;
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;

    if x1 <= x0 || y1 <= y0 {
        return DynamicImage::new_rgb8(1, 1);
    }

    image.crop_imm(x0, y0, x1 - x0, y1 - y0)
}

fn recognize_text(region: &DynamicImage) -> Result<String, OcrError> {
    let path = hf_hub_download(REC_REPO_ID, REC_MODEL_PATH, None, None)?;
    let session = get_or_create_session(path)?;
    let mut session = lock_session(&session)
        .map_err(|e| OcrError::Inference(InferenceError::Initialization(e.to_string())))?;

    let input_tensor = preprocess_rec(region)?;

    let inputs = ort::inputs![REC_INPUT_NAME => ort::value::Tensor::from_array(input_tensor)?];
    let outputs = session.run(inputs)?;

    let output_val = outputs
        .iter()
        .next()
        .ok_or_else(|| {
            OcrError::Inference(InferenceError::InvalidShape(
                "Missing output from PaddleOCR recognition".to_string(),
            ))
        })?
        .1;

    let (shape, slice) = output_val
        .try_extract_tensor::<f32>()
        .map_err(|e| OcrError::Inference(InferenceError::Initialization(e.to_string())))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    let chars = load_char_dict()?;

    let text = ctc_decode(&shape_usize, slice, &chars)?;

    Ok(text)
}

fn preprocess_rec(region: &DynamicImage) -> Result<Array4<f32>, OcrError> {
    let (w, h) = region.dimensions();

    let aspect_ratio = w as f32 / h as f32;
    let target_w = (REC_HEIGHT as f32 * aspect_ratio).round() as u32;
    let target_w = target_w.clamp(4, REC_WIDTH);

    let resized = region.resize_exact(target_w, REC_HEIGHT, image::imageops::FilterType::Triangle);

    let mut rgb = resized.to_rgb8();
    if target_w < REC_WIDTH {
        let mut canvas =
            image::ImageBuffer::from_pixel(REC_WIDTH, REC_HEIGHT, image::Rgb([0, 0, 0]));
        image::imageops::overlay(&mut canvas, &rgb, 0, 0);
        rgb = canvas;
    }

    let padded_img = DynamicImage::ImageRgb8(rgb);
    let tensor = to_ndarray_chw(&padded_img, &REC_IMG_NORMALIZE_MEAN, &REC_IMG_NORMALIZE_STD)?;

    Ok(tensor)
}

fn ctc_decode(shape: &[usize], data: &[f32], chars: &[String]) -> Result<String, OcrError> {
    let blank_idx = 0usize;

    let seq_len = match shape.len() {
        3 => shape[1],                  // [1, seq_len, num_classes]
        4 if shape[2] == 1 => shape[3], // [1, num_classes, 1, seq_len]
        _ => {
            return Err(OcrError::CharDict(format!(
                "Unexpected recognition output shape: {:?}",
                shape
            )));
        }
    };

    let num_classes = match shape.len() {
        3 => shape[2],
        4 => shape[1],
        _ => unreachable!(),
    };

    if shape[0] != 1
        || num_classes != chars.len()
        || seq_len.checked_mul(num_classes) != Some(data.len())
    {
        return Err(OcrError::CharDict(format!(
            "Recognition output {shape:?} does not match dictionary ({} classes) or data ({} values)",
            chars.len(),
            data.len()
        )));
    }

    let mut raw_indices = Vec::new();
    for t in 0..seq_len {
        let start = match shape.len() {
            3 => t * num_classes,
            4 => t,
            _ => unreachable!(),
        };

        let mut best_idx = blank_idx;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = match shape.len() {
                3 => data[start + c],
                4 => data[c * seq_len + t],
                _ => unreachable!(),
            };
            if val > best_val {
                best_val = val;
                best_idx = c;
            }
        }
        raw_indices.push(best_idx);
    }

    let mut text = String::new();
    let mut prev = blank_idx;
    for &idx in &raw_indices {
        if idx != blank_idx
            && idx != prev
            && let Some(ch) = chars.get(idx)
        {
            text.push_str(ch);
        }
        prev = idx;
    }

    Ok(text)
}

fn load_char_dict() -> Result<Vec<String>, OcrError> {
    let mut lock = CHAR_DICT
        .lock()
        .map_err(|e| OcrError::CharDict(e.to_string()))?;
    if let Some(ref dict) = *lock {
        return Ok(dict.clone());
    }

    let path = hf_hub_download(REC_REPO_ID, CHAR_DICT_PATH, None, None)?;
    let content = std::fs::read_to_string(path)?;

    let chars = parse_char_dict(&content)?;

    *lock = Some(chars.clone());
    Ok(chars)
}

// PaddleOCR exports omit class zero (CTC blank) and the final space class.
fn parse_char_dict(content: &str) -> Result<Vec<String>, OcrError> {
    if content.is_empty() {
        return Err(OcrError::CharDict("Character dictionary is empty".into()));
    }
    let mut chars = vec!["<blank>".to_string()];
    chars.extend(content.lines().map(|line| line.trim().to_string()));
    chars.push(" ".to_string());
    Ok(chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exported_dictionary_includes_blank_and_space_classes() {
        let chars = parse_char_dict("a\nb\n").unwrap();
        assert_eq!(chars, ["<blank>", "a", "b", " "]);
        let data = [
            0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
            1., 0.,
        ];
        assert_eq!(ctc_decode(&[1, 6, 4], &data, &chars).unwrap(), "aa b");
    }
    #[test]
    fn channels_first_recognition_uses_last_dimension_for_time() {
        let chars = parse_char_dict("a\nb").unwrap();
        let data = [0., 0., 1., 0., 0., 1., 0., 0.];
        assert_eq!(ctc_decode(&[1, 4, 1, 2], &data, &chars).unwrap(), "ab");
    }
    #[test]
    fn rejects_mismatched_dictionary_and_truncated_outputs() {
        let chars = parse_char_dict("a").unwrap();
        assert!(ctc_decode(&[1, 1, 2], &[0., 1.], &chars).is_err());
        assert!(ctc_decode(&[1, 2, 3], &[0., 1., 0.], &chars).is_err());
        assert!(parse_char_dict("").is_err());
    }
}
