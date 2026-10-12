//! Image and text embeddings from `onnx-community/embeddinggemma-2-ONNX`.
//!
//! The vision encoder produces soft tokens for the shared text model. Both
//! paths return comparable, L2-normalized vectors. Audio/video are not exposed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use image::DynamicImage;
use ndarray::{Array2, Array3};
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

use crate::generic::GenericError;
use crate::hub::hf_hub_download;
use crate::inference::{get_or_create_session, lock_session, model_sha256};

/// Public ONNX export, including tokenizer and external weight files.
pub const REPO_ID: &str = "onnx-community/embeddinggemma-2-ONNX";
const HIDDEN_SIZE: usize = 512;
const EMBEDDING_SIZE: usize = 768;
const MAX_TOKENS: usize = 8192;
const IMAGE_TOKEN: u32 = 258880;
const PATCH_SIZE: usize = 16;
const POOL_SIZE: usize = 3;

/// Weight formats that keep activations in FP32.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingGemma2Dtype {
    /// Full precision; text and vision weights total about 1.76 GB.
    Fp32,
    /// Eight-bit weights; text and vision weights total about 510 MB.
    #[default]
    Q8,
    /// Four-bit weights; text and vision weights total about 283 MB.
    Q4,
}

impl FromStr for EmbeddingGemma2Dtype {
    type Err = GenericError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "q8" => Ok(Self::Q8),
            "q4" => Ok(Self::Q4),
            _ => Err(invalid("dtype must be fp32, q8, or q4")),
        }
    }
}

impl EmbeddingGemma2Dtype {
    fn suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "",
            Self::Q8 => "_quantized",
            Self::Q4 => "_q4",
        }
    }
}

/// Task instruction prepended to text before tokenization.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingGemma2TextTask {
    /// Search queries, including queries used to retrieve images.
    #[default]
    SearchQuery,
    /// Corpus documents without a title.
    Document,
    /// Symmetric text similarity.
    SentenceSimilarity,
    /// Symmetric text clustering.
    Clustering,
    /// Symmetric text classification.
    Classification,
    /// Already formatted text, e.g. `title: My title | text: My document`.
    Raw,
}

impl FromStr for EmbeddingGemma2TextTask {
    type Err = GenericError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "SearchQuery" => Ok(Self::SearchQuery),
            "Document" => Ok(Self::Document),
            "SentenceSimilarity" => Ok(Self::SentenceSimilarity),
            "Clustering" => Ok(Self::Clustering),
            "Classification" => Ok(Self::Classification),
            "Raw" => Ok(Self::Raw),
            _ => Err(invalid("unknown EmbeddingGemma 2 text task")),
        }
    }
}

impl EmbeddingGemma2TextTask {
    fn format(self, text: &str) -> String {
        match self {
            Self::SearchQuery => format!("task: search result | query: {text}"),
            Self::Document => format!("title: none | text: {text}"),
            Self::SentenceSimilarity => format!("task: sentence similarity | query: {text}"),
            Self::Clustering => format!("task: clustering | query: {text}"),
            Self::Classification => format!("task: classification | query: {text}"),
            Self::Raw => text.to_owned(),
        }
    }
}

/// Options shared by image and text encoding.
#[derive(Debug, Clone)]
pub struct EmbeddingGemma2Options {
    /// Local export root containing `tokenizer.json` and `onnx/`. When absent,
    /// required files are downloaded from [`REPO_ID`] into the HF cache.
    pub model_dir: Option<PathBuf>,
    /// ONNX weight format, independent of the execution provider's precision.
    pub dtype: EmbeddingGemma2Dtype,
    /// Matryoshka output dimension: 128, 256, 512, or 768. Truncation is followed
    /// by L2 normalization. Use at least 256 for image retrieval.
    pub dimensions: usize,
    /// Maximum vision soft tokens: 70, 140, 280, 560, or 1120.
    pub max_soft_tokens: usize,
}

impl Default for EmbeddingGemma2Options {
    fn default() -> Self {
        Self {
            model_dir: None,
            dtype: EmbeddingGemma2Dtype::Q8,
            dimensions: EMBEDDING_SIZE,
            max_soft_tokens: 280,
        }
    }
}

impl EmbeddingGemma2Options {
    fn validate(&self) -> Result<(), GenericError> {
        if ![128, 256, 512, 768].contains(&self.dimensions) {
            return Err(invalid("dimensions must be 128, 256, 512, or 768"));
        }
        if ![70, 140, 280, 560, 1120].contains(&self.max_soft_tokens) {
            return Err(invalid(
                "max_soft_tokens must be 70, 140, 280, 560, or 1120",
            ));
        }
        Ok(())
    }

    fn asset(&self, name: &str) -> Result<PathBuf, GenericError> {
        if let Some(root) = &self.model_dir {
            let path = root.join(name);
            if !path.is_file() {
                return Err(invalid(format!("missing model asset: {}", path.display())));
            }
            Ok(path)
        } else {
            Ok(hf_hub_download(REPO_ID, name, None, None)?)
        }
    }

    fn model(&self, component: &str) -> Result<PathBuf, GenericError> {
        let name = format!("onnx/{component}{}.onnx", self.dtype.suffix());
        let path = self.asset(&name)?;
        // All published graphs store their weights in an adjacent external file.
        let weights = self.asset(&format!("{name}_data"))?;
        if weights.symlink_metadata()?.file_type().is_symlink() {
            // ORT 1.24 rejects external-data symlinks escaping the model directory.
            // Stage an immutable export with ordinary files/hard links instead.
            stage_model(
                &path,
                &weights,
                &crate::utils::storage::get_storage_dir().join("embeddinggemma2"),
            )
        } else {
            Ok(path)
        }
    }
}

fn stage_model(graph: &Path, weights: &Path, cache: &Path) -> Result<PathBuf, GenericError> {
    let identity = format!("{}-{}", model_sha256(graph)?, model_sha256(weights)?);
    let destination = cache.join(identity);
    let graph_name = graph
        .file_name()
        .ok_or_else(|| invalid("model path has no filename"))?;
    let weights_name = weights
        .file_name()
        .ok_or_else(|| invalid("weights path has no filename"))?;
    let result = destination.join(graph_name);
    if result.is_file() && destination.join(weights_name).is_file() {
        return Ok(result);
    }
    std::fs::create_dir_all(cache)?;
    let temporary = tempfile::tempdir_in(cache)?;
    std::fs::copy(graph, temporary.path().join(graph_name))?;
    let staged_weights = temporary.path().join(weights_name);
    if std::fs::hard_link(weights.canonicalize()?, &staged_weights).is_err() {
        // HF_HOME and IU_HOME can be on different filesystems.
        std::fs::copy(weights, staged_weights)?;
    }
    if let Err(error) = std::fs::rename(temporary.path(), &destination) {
        // Another caller may have installed the same immutable export meanwhile.
        if !result.is_file() || !destination.join(weights_name).is_file() {
            return Err(error.into());
        }
    }
    Ok(result)
}

static TOKENIZERS: Lazy<Mutex<HashMap<PathBuf, Arc<Tokenizer>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn tokenizer(options: &EmbeddingGemma2Options) -> Result<Arc<Tokenizer>, GenericError> {
    let path = options.asset("tokenizer.json")?.canonicalize()?;
    let mut cache = TOKENIZERS
        .lock()
        .map_err(|e| invalid(format!("tokenizer cache lock poisoned: {e}")))?;
    if let Some(tokenizer) = cache.get(&path) {
        return Ok(Arc::clone(tokenizer));
    }
    let mut tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| invalid(format!("failed to load tokenizer: {e}")))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|e| invalid(e.to_string()))?;
    let tokenizer = Arc::new(tokenizer);
    cache.insert(path, Arc::clone(&tokenizer));
    Ok(tokenizer)
}

fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, GenericError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| invalid(e.to_string()))?;
    if encoding.is_empty() || encoding.len() > MAX_TOKENS {
        return Err(invalid(
            "input must contain 1..=8192 tokens including special tokens",
        ));
    }
    Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
}

/// Encode texts into normalized embeddings, preserving input order.
///
/// Items are evaluated individually to bound attention memory. Empty batches
/// return immediately. Overlong text is rejected, never silently truncated.
pub fn embeddinggemma2_text_encode(
    texts: &[String],
    task: EmbeddingGemma2TextTask,
    options: &EmbeddingGemma2Options,
) -> Result<Vec<Vec<f32>>, GenericError> {
    options.validate()?;
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let tokenizer = tokenizer(options)?;
    let encoded = texts
        .iter()
        .map(|text| {
            let ids = tokenize(&tokenizer, &task.format(text))?;
            if ids.iter().any(|&id| [258880, 258881, 258884].contains(&id)) {
                return Err(invalid(
                    "media placeholders are not supported in text-only input",
                ));
            }
            Ok(ids)
        })
        .collect::<Result<Vec<_>, GenericError>>()?;
    let model = options.model("model")?;
    encoded
        .into_iter()
        .map(|ids| {
            encode(
                &model,
                ids,
                Array2::zeros((0, HIDDEN_SIZE)),
                options.dimensions,
            )
        })
        .collect()
}

/// Encode each image into a separate normalized embedding in the text model's
/// vector space. RGB conversion drops alpha, matching the upstream processor.
/// Images use no task prefix; a 280-token vision budget is the default.
pub fn embeddinggemma2_image_encode(
    images: &[DynamicImage],
    options: &EmbeddingGemma2Options,
) -> Result<Vec<Vec<f32>>, GenericError> {
    options.validate()?;
    if images.is_empty() {
        return Ok(Vec::new());
    }
    for image in images {
        target_size(image.width(), image.height(), options.max_soft_tokens)?;
    }
    let tokenizer = tokenizer(options)?;
    let model = options.model("model")?;
    let vision = options.model("vision_encoder")?;
    images
        .iter()
        .map(|image| {
            let (pixels, positions, count) = preprocess_image(image, options.max_soft_tokens)?;
            let session = get_or_create_session(&vision)?;
            let mut session = lock_session(&session)?;
            let outputs = session.run(ort::inputs![
                "pixel_values" => ort::value::Tensor::from_array(pixels)?,
                "pixel_position_ids" => ort::value::Tensor::from_array(positions)?,
            ])?;
            let output = outputs
                .get("image_features")
                .ok_or_else(|| invalid("missing image_features"))?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            if shape.as_ref() != [count as i64, HIDDEN_SIZE as i64] {
                return Err(invalid(format!(
                    "unexpected image feature shape: {shape:?}"
                )));
            }
            let features = Array2::from_shape_vec((count, HIDDEN_SIZE), data.to_vec())?;
            drop(outputs);
            drop(session);
            let text = format!("<|image>{}<image|>", "<|image|>".repeat(count));
            let ids = tokenize(&tokenizer, &text)?;
            if ids
                .iter()
                .filter(|&&id| id == i64::from(IMAGE_TOKEN))
                .count()
                != count
            {
                return Err(invalid(
                    "tokenizer does not match EmbeddingGemma 2 image tokens",
                ));
            }
            encode(&model, ids, features, options.dimensions)
        })
        .collect()
}

fn encode(
    model: &Path,
    ids: Vec<i64>,
    image_features: Array2<f32>,
    dimensions: usize,
) -> Result<Vec<f32>, GenericError> {
    let mask = Array2::<i64>::ones((1, ids.len()));
    let input = Array2::from_shape_vec((1, ids.len()), ids)?;
    let session = get_or_create_session(model)?;
    let mut session = lock_session(&session)?;
    let outputs = session.run(ort::inputs![
        "input_ids" => ort::value::Tensor::from_array(input)?,
        "attention_mask" => ort::value::Tensor::from_array(mask)?,
        "image_features" => ort::value::Tensor::from_array(image_features)?,
        "video_features" => ort::value::Tensor::from_array(Array2::<f32>::zeros((0, HIDDEN_SIZE)))?,
        "audio_features" => ort::value::Tensor::from_array(Array2::<f32>::zeros((0, HIDDEN_SIZE)))?,
    ])?;
    let output = outputs
        .get("sentence_embedding")
        .ok_or_else(|| invalid("missing sentence_embedding"))?;
    let (shape, data) = output.try_extract_tensor::<f32>()?;
    if shape.as_ref() != [1, EMBEDDING_SIZE as i64] {
        return Err(invalid(format!("unexpected embedding shape: {shape:?}")));
    }
    normalize(data, dimensions)
}

fn normalize(data: &[f32], dimensions: usize) -> Result<Vec<f32>, GenericError> {
    if data.len() != EMBEDDING_SIZE || data.iter().any(|x| !x.is_finite()) {
        return Err(invalid("model returned an invalid embedding"));
    }
    let mut output = data[..dimensions].to_vec();
    let norm = output
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm <= 0.0 {
        return Err(invalid("model returned a zero embedding"));
    }
    for value in &mut output {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(output)
}

fn target_size(width: u32, height: u32, budget: usize) -> Result<(usize, usize), GenericError> {
    if width == 0 || height == 0 {
        return Err(invalid("image dimensions must be nonzero"));
    }
    let side = PATCH_SIZE * POOL_SIZE;
    let factor = ((budget * side * side) as f64 / (f64::from(width) * f64::from(height))).sqrt();
    let mut w = (factor * f64::from(width) / side as f64).floor() as usize * side;
    let mut h = (factor * f64::from(height) / side as f64).floor() as usize * side;
    if h == 0 {
        h = side;
        w = ((width / height) as usize).min(budget) * side;
    } else if w == 0 {
        w = side;
        h = ((height / width) as usize).min(budget) * side;
    }
    Ok((w, h))
}

fn preprocess_image(
    image: &DynamicImage,
    budget: usize,
) -> Result<(Array3<f32>, Array3<i64>, usize), GenericError> {
    let (w, h) = target_size(image.width(), image.height(), budget)?;
    let rgb = image.to_rgb8();
    let rgb = image::imageops::resize(
        &rgb,
        w as u32,
        h as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let max_patches = budget * POOL_SIZE * POOL_SIZE;
    let mut pixels = Array3::zeros((1, max_patches, PATCH_SIZE * PATCH_SIZE * 3));
    let mut positions = Array3::from_elem((1, max_patches, 2), -1);
    let cols = w / PATCH_SIZE;
    let rows = h / PATCH_SIZE;
    for row in 0..rows {
        for col in 0..cols {
            let patch = row * cols + col;
            positions[[0, patch, 0]] = col as i64;
            positions[[0, patch, 1]] = row as i64;
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let pixel = rgb.get_pixel(
                        (col * PATCH_SIZE + dx) as u32,
                        (row * PATCH_SIZE + dy) as u32,
                    );
                    for c in 0..3 {
                        pixels[[0, patch, (dy * PATCH_SIZE + dx) * 3 + c]] =
                            f32::from(pixel[c]) / 255.0;
                    }
                }
            }
        }
    }
    Ok((pixels, positions, rows * cols / (POOL_SIZE * POOL_SIZE)))
}

fn invalid(message: impl Into<String>) -> GenericError {
    GenericError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_weight_staging_is_reusable_and_content_addressed() {
        let root = tempfile::tempdir().unwrap();
        let graph = root.path().join("model.onnx");
        let weights = root.path().join("model.onnx_data");
        std::fs::write(&graph, b"graph").unwrap();
        std::fs::write(&weights, b"weights").unwrap();
        let cache = root.path().join("cache");
        let first = stage_model(&graph, &weights, &cache).unwrap();
        assert_eq!(std::fs::read(&first).unwrap(), b"graph");
        assert_eq!(
            std::fs::read(first.with_file_name("model.onnx_data")).unwrap(),
            b"weights"
        );
        assert_eq!(stage_model(&graph, &weights, &cache).unwrap(), first);
        // Replace, rather than mutate, the source inode (as HF immutable blobs do).
        std::fs::remove_file(&weights).unwrap();
        std::fs::write(&weights, b"new weights").unwrap();
        assert_ne!(stage_model(&graph, &weights, &cache).unwrap(), first);
        assert_eq!(
            std::fs::read(first.with_file_name("model.onnx_data")).unwrap(),
            b"weights"
        );
    }

    #[test]
    fn options_and_empty_batches_do_not_download() {
        let mut options = EmbeddingGemma2Options::default();
        assert!(
            embeddinggemma2_text_encode(&[], EmbeddingGemma2TextTask::Raw, &options)
                .unwrap()
                .is_empty()
        );
        assert!(
            embeddinggemma2_image_encode(&[], &options)
                .unwrap()
                .is_empty()
        );
        options.dimensions = 64;
        assert!(embeddinggemma2_image_encode(&[], &options).is_err());
        options.dimensions = 768;
        options.max_soft_tokens = 71;
        assert!(options.validate().is_err());
        assert!("fp16".parse::<EmbeddingGemma2Dtype>().is_err());
    }

    #[test]
    fn task_prefixes_match_model_card() {
        assert_eq!(
            EmbeddingGemma2TextTask::SearchQuery.format("猫"),
            "task: search result | query: 猫"
        );
        assert_eq!(
            EmbeddingGemma2TextTask::Document.format("猫"),
            "title: none | text: 猫"
        );
        assert_eq!(EmbeddingGemma2TextTask::Raw.format("猫"), "猫");
    }

    #[test]
    fn resizing_preserves_budget_including_extreme_aspect_ratios() {
        assert_eq!(target_size(640, 480, 280).unwrap(), (912, 672));
        assert_eq!(target_size(100000, 1, 280).unwrap(), (13440, 48));
        assert_eq!(target_size(1, 100000, 280).unwrap(), (48, 13440));
        assert!(target_size(0, 1, 280).is_err());
        for budget in [70, 140, 280, 560, 1120] {
            for (width, height) in [(1, 1), (640, 480), (100000, 1), (1, 100000)] {
                let (w, h) = target_size(width, height, budget).unwrap();
                assert_eq!((w % 48, h % 48), (0, 0));
                assert!(w * h / (48 * 48) <= budget);
            }
        }
    }

    #[test]
    fn patch_layout_positions_and_padding_match_processor() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(384, 384, |x, y| {
            image::Rgb([x as u8, y as u8, 255])
        }));
        let (pixels, positions, count) = preprocess_image(&image, 70).unwrap();
        assert_eq!(count, 64);
        assert_eq!(pixels.dim(), (1, 630, 768));
        assert_eq!(positions[[0, 1, 0]], 1);
        assert_eq!(positions[[0, 24, 1]], 1);
        assert_eq!(pixels[[0, 1, 0]], 16.0 / 255.0);
        assert_eq!(pixels[[0, 24, 1]], 16.0 / 255.0);
        assert_eq!(pixels[[0, 0, 2]], 1.0);
        assert_eq!(positions[[0, 576, 0]], -1);
        assert_eq!(pixels[[0, 576, 0]], 0.0);
    }

    #[test]
    fn truncation_is_renormalized_and_invalid_outputs_are_rejected() {
        let mut vector = vec![1.0; EMBEDDING_SIZE];
        let output = normalize(&vector, 256).unwrap();
        assert_eq!(output.len(), 256);
        assert!((output.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        vector[767] = f32::NAN;
        assert!(normalize(&vector, 256).is_err());
        assert!(normalize(&[0.0; EMBEDDING_SIZE], 256).is_err());
    }
}
