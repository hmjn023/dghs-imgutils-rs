use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use crate::tagging::overlap::drop_overlap_tags;
use crate::tagging::{TagResult, TaggingError};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use serde::Deserialize;
use std::collections::HashMap;

const REPO_ID: &str = "deepghs/deepgelbooru_onnx";

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    stages: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct TagRecord {
    #[serde(rename = "tag_id")]
    tag_id: usize,
    name: String,
    category: usize,
}

fn apply_preprocessing(image: &DynamicImage, stages: &[serde_json::Value]) -> Array4<f32> {
    let mut img = image.clone();

    for stage in stages {
        let ttype = stage.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "resize" => {
                let size = stage
                    .get("size")
                    .and_then(|v| {
                        if let Some(n) = v.as_u64() {
                            Some(n as u32)
                        } else if let Some(arr) = v.as_array() {
                            arr.first().and_then(|x| x.as_u64()).map(|x| x as u32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(448);
                let (w, h) = img.dimensions();
                let (nw, nh) = if w < h {
                    (size, (size as f64 * h as f64 / w as f64) as u32)
                } else {
                    ((size as f64 * w as f64 / h as f64) as u32, size)
                };
                img = img.resize_exact(
                    nw.max(1),
                    nh.max(1),
                    image::imageops::FilterType::CatmullRom,
                );
            }
            "center_crop" => {
                let size = stage
                    .get("size")
                    .and_then(|v| {
                        if let Some(n) = v.as_u64() {
                            Some(n as u32)
                        } else if let Some(arr) = v.as_array() {
                            arr.first().and_then(|x| x.as_u64()).map(|x| x as u32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(448);
                let (w, h) = img.dimensions();
                let x = (w.saturating_sub(size)) / 2;
                let y = (h.saturating_sub(size)) / 2;
                img = img.crop_imm(x, y, size.min(w - x), size.min(h - y));
            }
            "pad_to_size" => {
                let size_val = stage
                    .get("size")
                    .and_then(|v| {
                        if let Some(arr) = v.as_array() {
                            if arr.len() >= 2 {
                                Some((
                                    arr[0].as_u64().unwrap_or(448) as u32,
                                    arr[1].as_u64().unwrap_or(448) as u32,
                                ))
                            } else {
                                let s = arr[0].as_u64().unwrap_or(448) as u32;
                                Some((s, s))
                            }
                        } else if let Some(n) = v.as_u64() {
                            Some((n as u32, n as u32))
                        } else {
                            None
                        }
                    })
                    .unwrap_or((448, 448));
                let bg = stage
                    .get("background_color")
                    .and_then(|v| {
                        if let Some(s) = v.as_str() {
                            if s == "white" {
                                Some([255u8, 255, 255])
                            } else {
                                Some([0u8, 0, 0])
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or([255, 255, 255]);
                img = crate::image::pad_image_to_size(&img, size_val.0, size_val.1, bg);
            }
            "to_tensor" | "maybe_to_tensor" => {}
            "rescale" => {}
            "normalize" => {}
            "convert_rgb" => {
                img = force_image_background(&img, [255, 255, 255]);
            }
            _ => {}
        }
    }

    let (w, h) = img.dimensions();
    let mut array = Array4::<f32>::zeros((1, h as usize, w as usize, 3));
    let rgb8 = img.to_rgb8();
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        array[[0, y as usize, x as usize, 0]] = pixel[0] as f32;
        array[[0, y as usize, x as usize, 1]] = pixel[1] as f32;
        array[[0, y as usize, x as usize, 2]] = pixel[2] as f32;
    }

    for stage in stages {
        let ttype = stage.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "rescale" => {
                let factor = stage
                    .get("rescale_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0 / 255.0) as f32;
                array.mapv_inplace(|v| v * factor);
            }
            "normalize" => {
                let mean = stage
                    .get("mean")
                    .and_then(|v| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64())
                                .map(|x| x as f32)
                                .collect::<Vec<_>>()
                        })
                    })
                    .unwrap_or_else(|| vec![0.0; 3]);
                let std = stage
                    .get("std")
                    .and_then(|v| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64())
                                .map(|x| x as f32)
                                .collect::<Vec<_>>()
                        })
                    })
                    .unwrap_or_else(|| vec![1.0; 3]);
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        for c in 0..3 {
                            let m = mean.get(c).copied().unwrap_or(0.0);
                            let s = std.get(c).copied().unwrap_or(1.0);
                            array[[0, y, x, c]] = (array[[0, y, x, c]] - m) / s;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    array
}

pub fn get_deepgelbooru_tags(
    image: &DynamicImage,
    general_threshold: f32,
    character_threshold: f32,
    drop_overlap: bool,
) -> Result<TagResult, TaggingError> {
    let model_path = hf_hub_download(REPO_ID, "model.onnx", None, None)?;
    let tags_path = hf_hub_download(REPO_ID, "tags.csv", None, None)?;
    let preproc_path = hf_hub_download(REPO_ID, "preprocessor.json", None, None)?;

    let preproc_content = std::fs::read_to_string(preproc_path)?;
    let preproc: PreprocessorConfig = serde_json::from_str(&preproc_content)?;

    let input_tensor = apply_preprocessing(image, &preproc.stages);

    let mut session = create_onnx_session(&model_path)?;
    let output_name = session.outputs()[0].name().to_string();
    let outputs = session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(input_tensor.clone())?
    ])?;
    let output_value = outputs.get(output_name.as_str()).ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No output tensor found".to_string())
    })?;
    let (_shape, prediction) = output_value.try_extract_tensor::<f32>()?;

    let mut rdr = csv::Reader::from_path(&tags_path)?;
    let mut tags_def: Vec<TagRecord> = Vec::new();
    for result in rdr.deserialize() {
        let record: TagRecord = result?;
        tags_def.push(record);
    }

    if prediction.len() != tags_def.len() {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            prediction.len(),
            tags_def.len()
        )));
    }

    let mut rating = Vec::new();
    let mut general = Vec::new();
    let mut character = Vec::new();
    let mut all_tags = Vec::new();

    for (i, record) in tags_def.iter().enumerate() {
        let prob = prediction[i];
        match record.category {
            9 => rating.push((record.name.clone(), prob)),
            0 => {
                if prob >= general_threshold {
                    general.push((record.name.clone(), prob));
                    all_tags.push((record.name.clone(), prob));
                }
            }
            4 => {
                if prob >= character_threshold {
                    character.push((record.name.clone(), prob));
                    all_tags.push((record.name.clone(), prob));
                }
            }
            _ => {}
        }
    }

    let sort_fn = |a: &(String, f32), b: &(String, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };
    general.sort_by(sort_fn);
    character.sort_by(sort_fn);
    all_tags.sort_by(sort_fn);

    if drop_overlap {
        general = drop_overlap_tags(&general);
        all_tags = general.iter().chain(character.iter()).cloned().collect();
        all_tags.sort_by(sort_fn);
    }

    let mut rest = HashMap::new();
    rest.insert("rating".to_string(), rating);

    Ok(TagResult {
        general,
        character,
        rest,
        tag: all_tags,
        ips: Vec::new(),
    })
}

pub fn get_deepgelbooru_tags_simple(image: &DynamicImage) -> Result<TagResult, TaggingError> {
    get_deepgelbooru_tags(image, 0.3, 0.3, false)
}
