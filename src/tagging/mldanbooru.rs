use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use crate::tagging::{TagResult, TaggingError};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
struct MlDanbooruTagRecord {
    name: String,
    real_name: Option<String>,
}

fn resize_align(image: &DynamicImage, size: u32, keep_ratio: bool, align: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    let (tw, th) = if !keep_ratio {
        (size, size)
    } else {
        let min_edge = w.min(h).max(1);
        (
            (w as f64 / min_edge as f64 * size as f64) as u32,
            (h as f64 / min_edge as f64 * size as f64) as u32,
        )
    };
    let tw = (tw / align) * align;
    let th = (th / align) * align;
    image.resize_exact(tw.max(1), th.max(1), image::imageops::FilterType::Triangle)
}

pub fn get_mldanbooru_tags(
    image: &DynamicImage,
    threshold: f32,
    size: u32,
    keep_ratio: bool,
    use_real_name: bool,
) -> Result<TagResult, TaggingError> {
    let model_path = hf_hub_download(
        "deepghs/ml-danbooru-onnx",
        "ml_caformer_m36_dec-5-97527.onnx",
        None,
        None,
    )?;
    let tags_path = hf_hub_download(
        "deepghs/imgutils-models",
        "mldanbooru/mldanbooru_tags.csv",
        None,
        None,
    )?;

    let rgb = force_image_background(image, [255, 255, 255]);
    let resized = resize_align(&rgb, size, keep_ratio, 4);
    let (rw, rh) = resized.dimensions();

    let mut array = Array4::<f32>::zeros((1, 3, rh as usize, rw as usize));
    let rgb8 = resized.to_rgb8();
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }

    let mut session = create_onnx_session(&model_path)?;
    let output_name = session.outputs()[0].name().to_string();
    let outputs = session.run(ort::inputs![
        "input" => ort::value::Tensor::from_array(array.clone())?
    ])?;
    let output_value = outputs.get(output_name.as_str()).ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No output tensor found".to_string())
    })?;
    let (_shape, logits) = output_value.try_extract_tensor::<f32>()?;

    let mut rdr = csv::Reader::from_path(&tags_path)?;
    let mut tag_names = Vec::new();
    for result in rdr.deserialize() {
        let record: MlDanbooruTagRecord = result?;
        let name = if use_real_name {
            record.real_name.unwrap_or(record.name)
        } else {
            record.name
        };
        tag_names.push(name);
    }

    let n = tag_names.len();
    if logits.len() != n {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            logits.len(),
            n
        )));
    }

    let mut pairs: Vec<(String, f32)> = tag_names
        .iter()
        .zip(logits.iter())
        .map(|(name, &logit)| {
            let prob = 1.0 / (1.0 + (-logit).exp());
            (name.clone(), prob)
        })
        .collect();

    pairs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let general: Vec<(String, f32)> = pairs.into_iter().filter(|(_, p)| *p >= threshold).collect();

    Ok(TagResult {
        general: general.clone(),
        character: Vec::new(),
        rest: HashMap::new(),
        tag: general,
        ips: Vec::new(),
        ips_mapping: HashMap::new(),
    })
}

pub fn get_mldanbooru_tags_simple(image: &DynamicImage) -> Result<TagResult, TaggingError> {
    get_mldanbooru_tags(image, 0.7, 448, false, false)
}
