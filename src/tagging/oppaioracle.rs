//! OppaiOracle モデルを用いた自動タグ付け処理の実装。
//!
//! HuggingFace 上の [`Grio43/OppaiOracle`](https://huggingface.co/Grio43/OppaiOracle)
//! で公開されている Vision Transformer ベースのマルチラベルアニメタガーです。
//! V1 (320x320, スクラッチ学習) と V1.1 (448x448, V1 のファインチューン) の
//! 2 つのチェックポイントが提供されています。

use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::get_or_create_session;
use crate::tagging::{TagResult, TaggingError};

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array3, Array4};
use serde::Deserialize;
use std::collections::HashMap;

/// モデルがホストされている HuggingFace リポジトリ ID
const REPO_ID: &str = "Grio43/OppaiOracle";
/// レターボックスのパディング色 (公式前処理と同じ RGB 114)
const PAD_COLOR: [u8; 3] = [114, 114, 114];
/// 正規化パラメータ (公式 preprocessing.json と同じ mean/std 0.5)
const NORMALIZE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const NORMALIZE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// モデルバリアントの定義
struct Variant {
    /// リポジトリ内の ONNX ディレクトリ名
    dir: &'static str,
    /// ネイティブ入力解像度
    image_size: u32,
    /// pr_thresholds.json のパースに失敗した場合のフォールバック閾値
    /// (macro single-threshold P=R breakeven)
    fallback_threshold: f32,
}

/// モデル名からバリアントを解決します。
///
/// `"v1"` / `"v1.1"` (大小文字不問) を受け付けます。
fn resolve_variant(model_name: &str) -> Result<Variant, TaggingError> {
    match model_name.to_ascii_lowercase().as_str() {
        "v1" => Ok(Variant {
            dir: "V1_onnx",
            image_size: 320,
            fallback_threshold: 0.614,
        }),
        "v1.1" => Ok(Variant {
            dir: "V1.1_onnx",
            image_size: 448,
            fallback_threshold: 0.760,
        }),
        _ => Err(TaggingError::InvalidArgument(format!(
            "Unknown OppaiOracle model variant: {model_name} (expected \"v1\" or \"v1.1\")"
        ))),
    }
}

/// vocabulary.json のデシリアライズ用構造体
#[derive(Debug, Deserialize)]
struct Vocabulary {
    tag_to_index: HashMap<String, usize>,
}

/// pr_thresholds.json のデシリアライズ用構造体 (必要なフィールドのみ)
#[derive(Debug, Deserialize)]
struct PrThresholds {
    macro_single_threshold: Option<MacroSingleThreshold>,
}

#[derive(Debug, Deserialize)]
struct MacroSingleThreshold {
    support_ge_0: Option<PrBreakEven>,
}

#[derive(Debug, Deserialize)]
struct PrBreakEven {
    pr_breakeven: Option<PrBreakEvenValue>,
}

#[derive(Debug, Deserialize)]
struct PrBreakEvenValue {
    threshold: Option<f32>,
}

/// レターボックス前処理を行い、正規化済み BCHW テンソルとパディングマスクを返します。
///
/// 公式前処理 (preprocessing.json / 公式デモ Space) と同様に:
/// 1. アスペクト比を維持したまま縮小のみ行う (拡大はしない)
/// 2. `(114, 114, 114)` で中央パディング
/// 3. `(x / 255 - 0.5) / 0.5` で正規化
///
/// パディングマスクは `True = パディング画素, False = 有効画素` の規約に従います。
fn letterbox_to_tensor(image: &DynamicImage, size: u32) -> (Array4<f32>, Array3<bool>) {
    let rgb = force_image_background(image, PAD_COLOR);
    let (w, h) = rgb.dimensions();

    // 縮小のみ (拡大しない): scale = min(size/w, size/h, 1.0)
    let scale = (size as f64 / w as f64)
        .min(size as f64 / h as f64)
        .min(1.0);
    let new_w = ((w as f64 * scale).round() as u32).max(1).min(size);
    let new_h = ((h as f64 * scale).round() as u32).max(1).min(size);

    let resized = if (new_w, new_h) != (w, h) {
        rgb.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
    } else {
        rgb
    };

    let mut canvas = ImageBuffer::from_pixel(size, size, Rgb(PAD_COLOR));
    let left = (size - new_w) / 2;
    let top = (size - new_h) / 2;
    image::imageops::overlay(&mut canvas, &resized.to_rgb8(), left as i64, top as i64);

    let size_us = size as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, size_us, size_us));
    // マスク形状は (B, H, W)。初期値は True (= パディング画素)
    let mut mask = Array3::<bool>::from_elem((1, size_us, size_us), true);

    for y in 0..size {
        for x in 0..size {
            let pixel = canvas.get_pixel(x, y);
            for c in 0..3 {
                tensor[[0, c, y as usize, x as usize]] =
                    (pixel[c] as f32 / 255.0 - NORMALIZE_MEAN[c]) / NORMALIZE_STD[c];
            }
        }
    }

    // 有効領域 (リサイズ後の画像が貼られた領域) は False
    for y in top..(top + new_h) {
        for x in left..(left + new_w) {
            mask[[0, y as usize, x as usize]] = false;
        }
    }

    (tensor, mask)
}

/// pr_thresholds.json から macro single-threshold の P=R breakeven 閾値を読み取ります。
fn load_default_threshold(variant: &Variant) -> f32 {
    let parsed = hf_hub_download(
        REPO_ID,
        &format!("{}/pr_thresholds.json", variant.dir),
        None,
        None,
    )
    .ok()
    .and_then(|p| std::fs::read_to_string(p).ok())
    .and_then(|content| serde_json::from_str::<PrThresholds>(&content).ok());

    parsed
        .and_then(|p| p.macro_single_threshold)
        .and_then(|m| m.support_ge_0)
        .and_then(|s| s.pr_breakeven)
        .and_then(|b| b.threshold)
        .unwrap_or(variant.fallback_threshold)
}

/// OppaiOracle モデルを用いて画像から Danbooru スタイルのタグを抽出します。
///
/// このモデルの語彙はすべて一般 (general) カテゴリのタグで構成されているため、
/// 結果は `TagResult.general` および `TagResult.tag` に格納されます。
///
/// # 引数
///
/// * `image` - 解析対象の画像オブジェクト
/// * `model_name` - モデルバリアント (`"v1"` または `"v1.1"`, デフォルト `"v1.1"`)
/// * `threshold` - タグ採用の確率閾値。`None` の場合はモデル同梱の
///   `pr_thresholds.json` にある macro P=R breakeven 閾値を使用
/// * `no_underline` - `true` の場合、タグ名のアンダースコアをスペースに置換
pub fn get_oppaioracle_tags(
    image: &DynamicImage,
    model_name: &str,
    threshold: Option<f32>,
    no_underline: bool,
) -> Result<TagResult, TaggingError> {
    let variant = resolve_variant(model_name)?;

    // 1. モデルと語彙ファイルのロード
    let model_path = hf_hub_download(REPO_ID, &format!("{}/model.onnx", variant.dir), None, None)?;
    let vocab_path = hf_hub_download(
        REPO_ID,
        &format!("{}/vocabulary.json", variant.dir),
        None,
        None,
    )?;

    let vocab_content = std::fs::read_to_string(vocab_path)?;
    let vocab: Vocabulary = serde_json::from_str(&vocab_content)?;
    let pad_idx =
        vocab.tag_to_index.get("<PAD>").copied().ok_or_else(|| {
            TaggingError::InvalidArgument("Vocabulary is missing <PAD>".to_string())
        })?;
    let unk_idx =
        vocab.tag_to_index.get("<UNK>").copied().ok_or_else(|| {
            TaggingError::InvalidArgument("Vocabulary is missing <UNK>".to_string())
        })?;

    // index -> tag の逆引きテーブル (インデックスの一意性・範囲・連続性を検証)
    let mut index_to_tag = vec![None; vocab.tag_to_index.len()];
    for (tag, idx) in &vocab.tag_to_index {
        let slot = index_to_tag.get_mut(*idx).ok_or_else(|| {
            TaggingError::InvalidArgument(format!("Vocabulary index {idx} is out of range"))
        })?;
        if slot.replace(tag.clone()).is_some() {
            return Err(TaggingError::InvalidArgument(format!(
                "Duplicate vocabulary index: {idx}"
            )));
        }
    }
    let index_to_tag: Vec<String> =
        index_to_tag
            .into_iter()
            .collect::<Option<_>>()
            .ok_or_else(|| {
                TaggingError::InvalidArgument("Vocabulary indices are not contiguous".to_string())
            })?;

    let threshold = threshold.unwrap_or_else(|| load_default_threshold(&variant));
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(TaggingError::InvalidArgument(
            "Threshold must be a finite value between 0 and 1".to_string(),
        ));
    }

    // 2. 前処理 (レターボックス + 正規化 + パディングマスク)
    let (input_tensor, padding_mask) = letterbox_to_tensor(image, variant.image_size);

    // 3. 推論 (sigmoid は ONNX グラフ内で適用済みのため、出力は確率)
    let probs = {
        let session_arc = get_or_create_session(&model_path)?;
        let mut session = session_arc.lock().map_err(|e| {
            crate::inference::InferenceError::Initialization(format!("Session lock poisoned: {e}"))
        })?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let pixel_input = input_names
            .iter()
            .find(|n| n.as_str() == "pixel_values")
            .cloned()
            .or_else(|| input_names.first().cloned())
            .ok_or_else(|| TaggingError::InvalidArgument("Model has no input".to_string()))?;
        let has_mask_input = input_names.iter().any(|n| n == "padding_mask");
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let outputs = if has_mask_input {
            session.run(ort::inputs![
                pixel_input.as_str() => ort::value::Tensor::from_array(input_tensor)?,
                "padding_mask" => ort::value::Tensor::from_array(padding_mask)?
            ])?
        } else {
            session.run(ort::inputs![
                pixel_input.as_str() => ort::value::Tensor::from_array(input_tensor)?
            ])?
        };

        let output_name = output_names
            .iter()
            .find(|n| n.as_str() == "tag_logits")
            .cloned()
            .or_else(|| output_names.first().cloned())
            .ok_or_else(|| TaggingError::InvalidArgument("Model has no output".to_string()))?;
        let output_value = outputs.get(output_name.as_str()).ok_or_else(|| {
            TaggingError::InvalidArgument(format!("No output tensor found: {output_name}"))
        })?;
        let (_shape, probs) = output_value.try_extract_tensor::<f32>()?;
        probs.to_vec()
    };

    if probs.len() != index_to_tag.len() {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match vocabulary size ({})",
            probs.len(),
            index_to_tag.len()
        )));
    }

    // 4. 後処理 (PAD/UNK 除外、rating 分離、閾値フィルタ、ソート)
    let sort_fn = |a: &(String, f32), b: &(String, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };

    let mut rating: Vec<(String, f32)> = Vec::new();
    let mut general: Vec<(String, f32)> = Vec::new();
    for (idx, &prob) in probs.iter().enumerate() {
        if idx == pad_idx || idx == unk_idx {
            continue;
        }
        let name = &index_to_tag[idx];
        if let Some(stripped) = name.strip_prefix("rating:") {
            // レーティングタグは閾値に関係なくすべて返す (他のタガーと同様の扱い)
            rating.push((stripped.to_string(), prob));
            continue;
        }
        if prob >= threshold {
            let mut name = name.clone();
            if no_underline {
                name = name.replace('_', " ");
            }
            general.push((name, prob));
        }
    }
    rating.sort_by(sort_fn);
    general.sort_by(sort_fn);

    let mut rest = HashMap::new();
    rest.insert("rating".to_string(), rating);

    Ok(TagResult {
        general: general.clone(),
        character: Vec::new(),
        rest,
        tag: general,
        ips: Vec::new(),
        ips_mapping: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_variant() {
        assert_eq!(resolve_variant("v1").unwrap().image_size, 320);
        assert_eq!(resolve_variant("V1").unwrap().image_size, 320);
        assert_eq!(resolve_variant("v1.1").unwrap().image_size, 448);
        assert_eq!(resolve_variant("V1.1").unwrap().image_size, 448);
        assert!(resolve_variant("v2").is_err());
    }

    #[test]
    fn test_letterbox_square() {
        // 正方形画像: そのままのサイズで貼り付け、マスクは全域 False
        let img = DynamicImage::new_rgb8(448, 448);
        let (tensor, mask) = letterbox_to_tensor(&img, 448);
        assert_eq!(tensor.shape(), &[1, 3, 448, 448]);
        assert!(!mask.iter().any(|&m| m));
    }

    #[test]
    fn test_letterbox_downscale_only() {
        // ターゲットより小さい画像は拡大されず、パディングされる
        let img = DynamicImage::new_rgb8(100, 50);
        let (tensor, mask) = letterbox_to_tensor(&img, 448);
        assert_eq!(tensor.shape(), &[1, 3, 448, 448]);

        // 有効領域: left=(448-100)/2=174, top=(448-50)/2=199
        assert!(!mask[[0, 199, 174]]);
        assert!(!mask[[0, 248, 273]]);
        // パディング領域
        assert!(mask[[0, 0, 0]]);
        assert!(mask[[0, 198, 174]]); // top の1行上
        assert!(mask[[0, 199, 173]]); // left の1列左
        assert!(mask[[0, 447, 447]]);
    }

    #[test]
    fn test_letterbox_keep_ratio() {
        // 横長画像: 幅がターゲットに合わせられ、高さは縮小される
        let img = DynamicImage::new_rgb8(896, 448);
        let (tensor, mask) = letterbox_to_tensor(&img, 448);
        assert_eq!(tensor.shape(), &[1, 3, 448, 448]);

        // new_w=448, new_h=224, top=(448-224)/2=112
        assert!(mask[[0, 0, 0]]);
        assert!(mask[[0, 111, 224]]);
        assert!(!mask[[0, 112, 0]]);
        assert!(!mask[[0, 335, 447]]);
        assert!(mask[[0, 336, 224]]);
    }

    /// 実モデルを用いたスモークテスト。
    /// 約1GBのモデルダウンロードが必要なため `#[ignore]` 付き。
    /// 実行例:
    /// `OPPAIORACLE_TEST_IMAGE=/path/to/image.jpg cargo test --no-default-features --lib tagging::oppaioracle -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn test_oppaioracle_smoke() {
        let path = std::env::var("OPPAIORACLE_TEST_IMAGE")
            .expect("OPPAIORACLE_TEST_IMAGE must be set to a test image path");
        let image = image::open(&path).expect("Failed to open test image");

        let result = get_oppaioracle_tags(&image, "v1.1", Some(0.65), false)
            .expect("OppaiOracle inference failed");

        println!("general tags ({}):", result.general.len());
        for (name, score) in result.general.iter().take(20) {
            println!("  {name}: {score:.4}");
        }

        assert!(
            !result.general.is_empty(),
            "Expected at least some general tags"
        );
        assert_eq!(result.general, result.tag);
        // 確率降順にソートされていること
        for w in result.general.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_letterbox_normalization() {
        // パディング色 (114) の正規化値を確認: (114/255 - 0.5) / 0.5
        let expected = (114.0f32 / 255.0 - 0.5) / 0.5;

        let small = DynamicImage::new_rgb8(10, 10);
        let (tensor, _) = letterbox_to_tensor(&small, 448);
        let v = tensor[[0, 0, 0, 0]];
        assert!((v - expected).abs() < 1e-5);
    }
}
