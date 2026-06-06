use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use crate::tagging::overlap::drop_overlap_tags;
use crate::tagging::{TagResult, TaggingError};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use serde::Deserialize;
use std::collections::HashMap;

const REPO_ID: &str = "deepghs/camie_tagger_onnx";

#[derive(Debug, Deserialize)]
struct PreprocessConfig {
    stages: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ThresholdConfig {
    #[serde(flatten)]
    inner: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct TagRecord {
    name: String,
    category: usize,
}

fn apply_preprocess_to_chw(image: &DynamicImage, stages: &[serde_json::Value]) -> Array4<f32> {
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
                    .unwrap_or(384);
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
                    .unwrap_or(384);
                let (w, h) = img.dimensions();
                let x = (w.saturating_sub(size)) / 2;
                let y = (h.saturating_sub(size)) / 2;
                img = img.crop_imm(x, y, size.min(w - x), size.min(h - y));
            }
            "pad_to_size" => {
                let s = 384u32;
                img = crate::image::pad_image_to_size(&img, s, s, [255, 255, 255]);
            }
            "convert_rgb" => {
                img = force_image_background(&img, [255, 255, 255]);
            }
            "to_tensor" | "maybe_to_tensor" => {}
            "rescale" => {}
            "normalize" => {}
            _ => {}
        }
    }

    let (w, h) = img.dimensions();
    let mut array = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    let rgb8 = img.to_rgb8();

    for (x, y, pixel) in rgb8.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32;
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32;
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32;
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
                let mean: Vec<f32> = stage
                    .get("mean")
                    .and_then(|v| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64())
                                .map(|x| x as f32)
                                .collect()
                        })
                    })
                    .unwrap_or_else(|| vec![0.0; 3]);
                let std: Vec<f32> = stage
                    .get("std")
                    .and_then(|v| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64())
                                .map(|x| x as f32)
                                .collect()
                        })
                    })
                    .unwrap_or_else(|| vec![1.0; 3]);
                for c in 0..3 {
                    let m = mean.get(c).copied().unwrap_or(0.0);
                    let s = std.get(c).copied().unwrap_or(1.0);
                    let mut slice = array.slice_mut(ndarray::s![0, c, .., ..]);
                    slice.mapv_inplace(|v| (v - m) / s);
                }
            }
            _ => {}
        }
    }

    array
}

fn get_threshold_value(config: &serde_json::Value, cate_name: &str, mode: &str) -> f32 {
    if let Some(obj) = config.get(cate_name) {
        if let Some(mode_obj) = obj.get(mode) {
            if let Some(th) = mode_obj.get("threshold").and_then(|v| v.as_f64()) {
                return th as f32;
            }
        }
    }
    match cate_name {
        "rating" => 0.0,
        "general" => 0.35,
        "character" => 0.5,
        "artist" => 0.5,
        "copyright" => 0.5,
        "meta" => 0.5,
        "year" => 0.5,
        _ => 0.5,
    }
}

const CATEGORY_MAP: &[(&str, usize)] = &[
    ("rating", 9),
    ("general", 0),
    ("character", 4),
    ("artist", 1),
    ("copyright", 3),
    ("meta", 5),
    ("year", 10),
];

pub fn get_camie_tags(
    image: &DynamicImage,
    model_name: &str,
    mode: &str,
    thresholds: Option<HashMap<String, f32>>,
    no_underline: bool,
    drop_overlap: bool,
) -> Result<TagResult, TaggingError> {
    let preproc_path = hf_hub_download(
        REPO_ID,
        &format!("{}/preprocess.json", model_name),
        None,
        None,
    )?;
    let tags_path = hf_hub_download(
        REPO_ID,
        &format!("{}/selected_tags.csv", model_name),
        None,
        None,
    )?;
    let thresh_path = hf_hub_download(
        REPO_ID,
        &format!("{}/threshold.json", model_name),
        None,
        None,
    )?;
    let model_path = hf_hub_download(REPO_ID, &format!("{}/model.onnx", model_name), None, None)?;

    let preproc_content = std::fs::read_to_string(preproc_path)?;
    let preproc: PreprocessConfig = serde_json::from_str(&preproc_content)?;

    let thresh_content = std::fs::read_to_string(thresh_path)?;
    let thresh_config: serde_json::Value = serde_json::from_str(&thresh_content)?;

    let rgb = force_image_background(image, [255, 255, 255]);
    let input_tensor = apply_preprocess_to_chw(&rgb, &preproc.stages);

    let mut session = create_onnx_session(&model_path)?;
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    let has_refined = output_names
        .iter()
        .any(|n| n == "embedding" || n == "refined/embedding");

    let mut rdr = csv::Reader::from_path(&tags_path)?;
    let mut tag_defs = Vec::new();
    for result in rdr.deserialize() {
        let mut record: TagRecord = result?;
        if no_underline {
            record.name = record.name.replace('_', " ");
        }
        tag_defs.push(record);
    }

    let (pred, _embedding) = if has_refined {
        let outputs = session.run(ort::inputs![
            "input" => ort::value::Tensor::from_array(input_tensor.clone())?
        ])?;

        let refined_pred = outputs
            .get("output")
            .or_else(|| outputs.get("refined/output"))
            .ok_or_else(|| TaggingError::InvalidArgument("No refined output found".to_string()))?;
        let (_shape, refined_pred_data) = refined_pred.try_extract_tensor::<f32>()?;

        let refined_emb = outputs
            .get("embedding")
            .or_else(|| outputs.get("refined/embedding"))
            .ok_or_else(|| {
                TaggingError::InvalidArgument("No refined embedding found".to_string())
            })?;
        let (_emb_shape, refined_emb_data) = refined_emb.try_extract_tensor::<f32>()?;

        (refined_pred_data.to_vec(), Some(refined_emb_data.to_vec()))
    } else {
        let outputs = session.run(ort::inputs![
            "input" => ort::value::Tensor::from_array(input_tensor.clone())?
        ])?;

        let init_pred = outputs
            .get("output")
            .or_else(|| outputs.get("initial/output"))
            .ok_or_else(|| TaggingError::InvalidArgument("No initial output found".to_string()))?;
        let (_shape, init_pred_data) = init_pred.try_extract_tensor::<f32>()?;

        (init_pred_data.to_vec(), None)
    };

    let n = tag_defs.len();
    if pred.len() != n {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            pred.len(),
            n
        )));
    }

    let mut category_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (cate_name, cate_id) in CATEGORY_MAP {
        let indices: Vec<usize> = tag_defs
            .iter()
            .enumerate()
            .filter(|(_, t)| t.category == *cate_id)
            .map(|(i, _)| i)
            .collect();
        category_indices.insert(cate_name, indices);
    }

    let mut rating = Vec::new();
    let mut general = Vec::new();
    let mut character = Vec::new();
    let mut all_tags = Vec::new();

    for (cate_name, indices) in &category_indices {
        let default_th = get_threshold_value(&thresh_config, cate_name, mode);
        let threshold = thresholds
            .as_ref()
            .and_then(|t| t.get(*cate_name))
            .copied()
            .unwrap_or(default_th);

        if *cate_name == "rating" {
            for &i in indices {
                rating.push((tag_defs[i].name.clone(), pred[i]));
            }
        } else {
            for &i in indices {
                let prob = pred[i];
                if prob >= threshold {
                    let entry = (tag_defs[i].name.clone(), prob);
                    match *cate_name {
                        "general" => {
                            general.push(entry.clone());
                            all_tags.push(entry);
                        }
                        "character" => {
                            character.push(entry.clone());
                            all_tags.push(entry);
                        }
                        _ => {}
                    }
                }
            }
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
        ips_mapping: HashMap::new(),
    })
}

pub fn get_camie_tags_simple(image: &DynamicImage) -> Result<TagResult, TaggingError> {
    get_camie_tags(image, "initial", "balanced", None, false, false)
}
