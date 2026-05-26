use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use crate::tagging::{TagResult, TaggingError};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
struct DeepDanbooruTagRecord {
    name: String,
    real_name: String,
    category: usize,
}

pub fn get_deepdanbooru_tags(
    image: &DynamicImage,
    general_threshold: f32,
    character_threshold: f32,
    use_real_name: bool,
) -> Result<TagResult, TaggingError> {
    let model_path = hf_hub_download("deepghs/imgutils-models", "deepdanbooru/deepdanbooru.onnx", None, None)?;
    let tags_path = hf_hub_download("deepghs/imgutils-models", "deepdanbooru/deepdanbooru_tags.csv", None, None)?;

    let rgb = force_image_background(image, [0, 0, 0]);
    let (w, h) = rgb.dimensions();
    let scale = 512.0 / w.max(h) as f32;
    let fw = (w as f32 * scale).round() as u32;
    let fh = (h as f32 * scale).round() as u32;
    let resized = rgb.resize_exact(fw.max(1), fh.max(1), image::imageops::FilterType::Triangle);

    let pad_left = (512i32 - fw as i32) / 2;
    let pad_top = (512i32 - fh as i32) / 2;

    let mut array = Array4::<f32>::zeros((1, 512usize, 512usize, 3));
    let rgb8 = resized.to_rgb8();
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        let ay = (y as i32 + pad_top).max(0).min(511) as usize;
        let ax = (x as i32 + pad_left).max(0).min(511) as usize;
        array[[0, ay, ax, 0]] = pixel[0] as f32 / 255.0;
        array[[0, ay, ax, 1]] = pixel[1] as f32 / 255.0;
        array[[0, ay, ax, 2]] = pixel[2] as f32 / 255.0;
    }

    let mut session = create_onnx_session(&model_path)?;
    let output_name = session.outputs()[0].name().to_string();
    let outputs = session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(array.clone())?
    ])?;
    let output_value = outputs.get(output_name.as_str()).ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No output tensor found".to_string())
    })?;
    let (_shape, prediction) = output_value.try_extract_tensor::<f32>()?;

    let mut rdr = csv::Reader::from_path(&tags_path)?;
    let mut tags_definition = Vec::new();
    for result in rdr.deserialize() {
        let record: DeepDanbooruTagRecord = result?;
        tags_definition.push(record);
    }

    if prediction.len() != tags_definition.len() {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            prediction.len(),
            tags_definition.len()
        )));
    }

    let mut rating = Vec::new();
    let mut general = Vec::new();
    let mut character = Vec::new();
    let mut all_tags = Vec::new();

    for (i, record) in tags_definition.iter().enumerate() {
        let prob = prediction[i];
        let tag_name = if use_real_name {
            record.real_name.clone()
        } else {
            record.name.clone()
        };

        match record.category {
            9 => rating.push((tag_name, prob)),
            0 => {
                if prob >= general_threshold {
                    general.push((tag_name.clone(), prob));
                    all_tags.push((tag_name, prob));
                }
            }
            4 => {
                if prob >= character_threshold {
                    character.push((tag_name.clone(), prob));
                    all_tags.push((tag_name, prob));
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

pub fn get_deepdanbooru_tags_simple(
    image: &DynamicImage,
) -> Result<TagResult, TaggingError> {
    get_deepdanbooru_tags(image, 0.5, 0.5, false)
}
