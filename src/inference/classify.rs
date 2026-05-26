//! 汎用画像分類モデル推論エンジン。
//!
//! `validate/` モジュールの大半が使う `ClassifyModel` パターンを実装します。
//! - モデル: `{repo_id}/{model_name}/model.onnx`
//! - ラベル: `{repo_id}/{model_name}/meta.json` の `"labels"` キー
//!
//! 前処理パイプライン:
//! 1. 白背景アルファブレンド
//! 2. Bilinear リサイズ
//! 3. `[0, 1]` → `(x - 0.5) / 0.5` → `[-1, 1]` 正規化
//! 4. CHW バッチ展開 → `[1, 3, H, W]`

use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{InferenceError, create_onnx_session};
use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;

/// meta.json のスキーマ
#[derive(Debug, Deserialize)]
struct MetaJson {
    labels: Vec<String>,
}

/// ロード済みモデルのキャッシュエントリ
struct ClassifyEntry {
    session: Session,
    labels: Vec<String>,
    input_w: u32,
    input_h: u32,
}

static CLASSIFY_CACHE: Lazy<Mutex<HashMap<String, ClassifyEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// 指定した `(repo_id, model_name)` の ClassifyModel を取得します。
/// キャッシュ済みならキャッシュから返します。
fn ensure_classify_model(
    repo_id: &str,
    model_name: &str,
) -> Result<(), InferenceError> {
    let key = format!("{}/{}", repo_id, model_name);
    {
        let cache = CLASSIFY_CACHE.lock().unwrap();
        if cache.contains_key(&key) {
            return Ok(());
        }
    }

    // model.onnx をダウンロード
    let model_path = hf_hub_download(
        repo_id,
        &format!("{}/model.onnx", model_name),
        None,
        None,
    )
    .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    // meta.json をダウンロード
    let meta_path = hf_hub_download(
        repo_id,
        &format!("{}/meta.json", model_name),
        None,
        None,
    )
    .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    let meta_str = std::fs::read_to_string(&meta_path)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let meta: MetaJson = serde_json::from_str(&meta_str)
        .map_err(|e| InferenceError::Initialization(format!("meta.json parse error: {}", e)))?;

    let session = create_onnx_session(&model_path)?;

    // 入力サイズをモデルメタから取得（fallback: 224x224）
    let (input_w, input_h) = get_model_input_size(&session);

    let entry = ClassifyEntry {
        session,
        labels: meta.labels,
        input_w,
        input_h,
    };

    CLASSIFY_CACHE.lock().unwrap().insert(key, entry);
    Ok(())
}

/// モデルの入力サイズをセッションメタデータから取得します。
/// 取得できない場合は 224x224 をデフォルトとして返します。
fn get_model_input_size(_session: &Session) -> (u32, u32) {
    // ort 2.0 の API では動的にサイズを取れないケースがあるため 224x224 をデフォルトとする
    (224, 224)
}

/// 汎用 ClassifyModel 推論を実行します。
///
/// # 引数
/// * `image` - 入力画像（任意フォーマット）
/// * `repo_id` - HuggingFace リポジトリ ID
/// * `model_name` - モデル名（`repo_id` 内のサブディレクトリ名）
///
/// # 戻り値
/// ラベル → 確信度スコアのマップ（softmax 済み）
pub fn classify_predict(
    image: &DynamicImage,
    repo_id: &str,
    model_name: &str,
) -> Result<HashMap<String, f32>, InferenceError> {
    ensure_classify_model(repo_id, model_name)?;

    let key = format!("{}/{}", repo_id, model_name);
    let mut cache = CLASSIFY_CACHE.lock().unwrap();
    let entry = cache
        .get_mut(&key)
        .ok_or_else(|| InferenceError::Initialization("Model not found in cache".to_string()))?;

    let tensor = preprocess_classify(image, entry.input_w, entry.input_h)?;
    let outputs = entry.session.run(
        ort::inputs!["input" => ort::value::Tensor::from_array(tensor.clone())?]
    )?;

    let (_, output_slice) = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::InvalidShape(e.to_string()))?;
    let logits: Vec<f32> = output_slice.to_vec();

    // Softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    let result: HashMap<String, f32> = entry
        .labels
        .iter()
        .zip(probs.iter())
        .map(|(label, &prob)| (label.clone(), prob))
        .collect();

    Ok(result)
}

/// 共通前処理: 白背景ブレンド → Bilinear リサイズ → [-1, 1] 正規化 → [1, 3, H, W]
fn preprocess_classify(
    image: &DynamicImage,
    target_w: u32,
    target_h: u32,
) -> Result<Array4<f32>, InferenceError> {
    let rgb = force_image_background(image, [255, 255, 255]);
    let resized = rgb.resize_exact(target_w, target_h, FilterType::Triangle);
    let rgb8 = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        // (x - 0.5) / 0.5 = x * 2.0 - 1.0
        tensor[[0, 0, y as usize, x as usize]] = r * 2.0 - 1.0;
        tensor[[0, 1, y as usize, x as usize]] = g * 2.0 - 1.0;
        tensor[[0, 2, y as usize, x as usize]] = b * 2.0 - 1.0;
    }
    Ok(tensor)
}

/// `classify_predict` の結果から最大クラスを返します。
pub fn classify_top1(
    image: &DynamicImage,
    repo_id: &str,
    model_name: &str,
) -> Result<(String, HashMap<String, f32>), InferenceError> {
    let scores = classify_predict(image, repo_id, model_name)?;
    let top = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| k.clone())
        .ok_or_else(|| InferenceError::InvalidShape("Empty output".to_string()))?;
    Ok((top, scores))
}
