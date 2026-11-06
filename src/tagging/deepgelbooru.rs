use crate::hub::hf_hub_download;
use crate::inference::get_or_create_session;
use crate::tagging::overlap::drop_overlap_tags;
use crate::tagging::{TagResult, TaggingError};
use image::DynamicImage;
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
    _tag_id: usize,
    name: String,
    category: usize,
}

fn apply_preprocessing(
    image: &DynamicImage,
    stages: &[serde_json::Value],
) -> Result<Array4<f32>, TaggingError> {
    let chw = crate::image::preprocess::apply(image, stages)?;
    Ok(chw
        .permuted_axes([0, 2, 3, 1])
        .as_standard_layout()
        .to_owned())
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

    let input_tensor = apply_preprocessing(image, &preproc.stages)?;

    let prediction = {
        let session_arc = get_or_create_session(&model_path)?;
        let mut session = session_arc.lock().map_err(|e| {
            crate::inference::InferenceError::Initialization(format!("Session lock poisoned: {e}"))
        })?;
        let output_name = session.outputs()[0].name().to_string();
        let outputs = session.run(ort::inputs![
            "input" => ort::value::Tensor::from_array(input_tensor.clone())?
        ])?;
        let output_value = outputs.get(output_name.as_str()).ok_or_else(|| {
            crate::inference::InferenceError::InvalidShape("No output tensor found".to_string())
        })?;
        let (_shape, prediction) = output_value.try_extract_tensor::<f32>()?;
        prediction.to_vec()
    };

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
            4 if prob >= character_threshold => {
                character.push((record.name.clone(), prob));
                all_tags.push((record.name.clone(), prob));
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
        ips_mapping: HashMap::new(),
    })
}

pub fn get_deepgelbooru_tags_simple(image: &DynamicImage) -> Result<TagResult, TaggingError> {
    get_deepgelbooru_tags(image, 0.3, 0.3, false)
}
