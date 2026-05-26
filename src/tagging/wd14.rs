use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use crate::tagging::{TagResult, TaggingError};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use std::collections::HashMap;

const EMBEDDING_REPO: &str = "deepghs/wd14_tagger_with_embeddings";

const MODEL_NAMES: &[(&str, &str, u32)] = &[
    ("SwinV2_v3", "SmilingWolf/wd-swinv2-tagger-v3", 224),
    ("SwinV2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", 224),
    ("ConvNext", "SmilingWolf/wd-v1-4-convnext-tagger-v2", 224),
    ("ConvNext_v3", "SmilingWolf/wd-convnext-tagger-v3", 256),
    ("ViT", "SmilingWolf/wd-v1-4-vit-tagger-v2", 224),
    ("ViT_v3", "SmilingWolf/wd-vit-tagger-v3", 384),
    ("ViT_Large", "SmilingWolf/wd-vit-large-tagger-v3", 384),
    ("EVA02_Large", "SmilingWolf/wd-eva02-large-tagger-v3", 448),
    ("MOAT", "SmilingWolf/wd-v1-4-moat-tagger-v2", 224),
    ("ConvNextV2", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2", 224),
];

const DEFAULT_MODEL: &str = "SwinV2_v3";

fn lookup_model(model_name: &str) -> (&'static str, u32) {
    for (name, repo, size) in MODEL_NAMES {
        if *name == model_name {
            return (*repo, *size);
        }
    }
    let (_, repo, size) = MODEL_NAMES[0];
    (repo, size)
}

#[derive(Debug, serde::Deserialize)]
struct Wd14TagRecord {
    name: String,
    category: usize,
}

fn prepare_image_for_tagging(image: &DynamicImage, target_size: u32) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);

    let padded = if w == max_dim && h == max_dim {
        image.clone()
    } else {
        let pad_left = (max_dim - w) / 2;
        let pad_top = (max_dim - h) / 2;
        let mut canvas = image::ImageBuffer::from_pixel(max_dim, max_dim, image::Rgb([255, 255, 255]));
        let rgb8 = force_image_background(image, [255, 255, 255]).to_rgb8();
        image::imageops::overlay(&mut canvas, &rgb8, pad_left as i64, pad_top as i64);
        DynamicImage::ImageRgb8(canvas)
    };

    let resized = if max_dim != target_size {
        padded.resize_exact(target_size, target_size, image::imageops::FilterType::CatmullRom)
    } else {
        padded
    };

    let mut array = Array4::<f32>::zeros((1, target_size as usize, target_size as usize, 3));
    let rgb8 = resized.to_rgb8();
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        // BGR format
        array[[0, y as usize, x as usize, 0]] = pixel[2] as f32; // B
        array[[0, y as usize, x as usize, 1]] = pixel[1] as f32; // G
        array[[0, y as usize, x as usize, 2]] = pixel[0] as f32; // R
    }

    array
}

fn mcut_threshold(probs: &[f32]) -> f32 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() < 2 {
        return sorted.first().copied().unwrap_or(0.5);
    }
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..sorted.len() - 1 {
        let diff = sorted[i] - sorted[i + 1];
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    (sorted[max_idx] + sorted[max_idx + 1]) / 2.0
}

pub fn get_wd14_tags(
    image: &DynamicImage,
    model_name: &str,
    general_threshold: f32,
    general_mcut_enabled: bool,
    character_threshold: f32,
    character_mcut_enabled: bool,
    no_underline: bool,
) -> Result<TagResult, TaggingError> {
    let (repo_path, target_size) = lookup_model(model_name);

    let model_path = hf_hub_download(
        EMBEDDING_REPO,
        &format!("{}/model.onnx", repo_path),
        None,
        None,
    )?;

    let tags_path = hf_hub_download(repo_path, "selected_tags.csv", None, None)?;

    let input_tensor = prepare_image_for_tagging(image, target_size);

    let mut session = create_onnx_session(&model_path)?;
    let out0_name = session.outputs()[0].name().to_string();
    let out1_name = session.outputs()[1].name().to_string();
    let outputs = session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(input_tensor.clone())?
    ])?;

    let pred_value = outputs.get(out0_name.as_str()).ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No prediction output".to_string())
    })?;
    let (_, pred_raw) = pred_value.try_extract_tensor::<f32>()?;

    let _emb_value = outputs.get(out1_name.as_str()).ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No embedding output".to_string())
    })?;
    let (_emb_shape, _embedding) = _emb_value.try_extract_tensor::<f32>()?;

    let mut rdr = csv::Reader::from_path(&tags_path)?;
    let mut tag_defs = Vec::new();
    for result in rdr.deserialize() {
        let mut record: Wd14TagRecord = result?;
        if no_underline {
            record.name = record.name.replace('_', " ");
        }
        tag_defs.push(record);
    }

    let n = tag_defs.len();
    let prediction: Vec<f32> = pred_raw.iter().copied().collect();

    if prediction.len() != n {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            prediction.len(),
            n
        )));
    }

    let labels: Vec<(String, f32)> = tag_defs.iter().zip(prediction.iter()).map(|(t, &p)| (t.name.clone(), p)).collect();

    let rating: Vec<(String, f32)> = labels.iter().enumerate().filter(|(i, _)| tag_defs[*i].category == 9).map(|(_, v)| v.clone()).collect();

    let general_indices: Vec<usize> = tag_defs.iter().enumerate().filter(|(_, t)| t.category == 0).map(|(i, _)| i).collect();
    let character_indices: Vec<usize> = tag_defs.iter().enumerate().filter(|(_, t)| t.category == 4).map(|(i, _)| i).collect();

    let mut general_th = general_threshold;
    if general_mcut_enabled {
        let general_probs: Vec<f32> = general_indices.iter().map(|&i| prediction[i]).collect();
        general_th = mcut_threshold(&general_probs);
    }

    let mut char_th = character_threshold;
    if character_mcut_enabled {
        let char_probs: Vec<f32> = character_indices.iter().map(|&i| prediction[i]).collect();
        char_th = mcut_threshold(&char_probs).max(0.15);
    }

    let mut general = Vec::new();
    let mut character = Vec::new();
    let mut all_tags = Vec::new();

    for (i, tag) in tag_defs.iter().enumerate() {
        let prob = prediction[i];
        match tag.category {
            0 => {
                if prob > general_th {
                    general.push((tag.name.clone(), prob));
                    all_tags.push((tag.name.clone(), prob));
                }
            }
            4 => {
                if prob > char_th {
                    character.push((tag.name.clone(), prob));
                    all_tags.push((tag.name.clone(), prob));
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

pub fn get_wd14_tags_simple(image: &DynamicImage) -> Result<TagResult, TaggingError> {
    get_wd14_tags(image, DEFAULT_MODEL, 0.35, false, 0.85, false, false)
}
