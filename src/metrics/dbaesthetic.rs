use crate::inference::classify::classify_predict;

use image::DynamicImage;

const REPO_ID: &str = "deepghs/anime_aesthetic";
const DEFAULT_MODEL_NAME: &str = "swinv2pv3_v0_448_ls0.2_x";

const LABELS: &[&str] = &[
    "worst",
    "low",
    "normal",
    "good",
    "great",
    "best",
    "masterpiece",
];

/// Compute Danbooru aesthetic score (0.0 to 1.0) for an anime image.
/// This uses a weighted sum of softmax probabilities over 7 aesthetic labels.
pub fn anime_dbaesthetic(
    image: &DynamicImage,
    model_name: Option<&str>,
) -> Result<f32, crate::inference::InferenceError> {
    let model = model_name.unwrap_or(DEFAULT_MODEL_NAME);
    let scores = classify_predict(image, REPO_ID, model)?;
    let total: f32 = LABELS
        .iter()
        .enumerate()
        .map(|(i, label)| scores.get(*label).copied().unwrap_or(0.0) * i as f32)
        .sum();
    let normalized = total / (LABELS.len() - 1) as f32;
    Ok(normalized)
}

/// Return both score and full confidence map.
pub fn anime_dbaesthetic_full(
    image: &DynamicImage,
    model_name: Option<&str>,
) -> Result<(f32, std::collections::HashMap<String, f32>), crate::inference::InferenceError> {
    let model = model_name.unwrap_or(DEFAULT_MODEL_NAME);
    let scores = classify_predict(image, REPO_ID, model)?;
    let total: f32 = LABELS
        .iter()
        .enumerate()
        .map(|(i, label)| scores.get(*label).copied().unwrap_or(0.0) * i as f32)
        .sum();
    let normalized = total / (LABELS.len() - 1) as f32;
    Ok((normalized, scores))
}
