//! PixAI Tagger モデルを用いた自動タグ付け処理の実装。

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{create_onnx_session, run_onnx_session};
use crate::tagging::{TagResult, TaggingError};

use image::DynamicImage;
use std::collections::HashMap;

/// リポジトリ名または簡易モデル名から HuggingFace の repo_id を解決します。
fn resolve_repo_id(model_name: &str) -> String {
    if model_name.contains('/') {
        model_name.to_string()
    } else {
        format!("deepghs/pixai-tagger-{}-onnx", model_name)
    }
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct CsvTagRecord {
    id: usize,
    tag_id: i64,
    name: String,
    category: usize,
    count: i64,
    ips: String, // JSON形式の文字列（例: `["genshin_impact"]`）
}

/// CSV レコード定義: thresholds.csv
#[derive(Debug, serde::Deserialize)]
struct CsvThresholdRecord {
    category: usize,
    name: String,
    threshold: f32,
}

/// PixAI Tagger を用いて画像から Danbooru スタイルのタグおよび原作 IP 情報を抽出します。
///
/// # 引数
///
/// * `image` - 解析対象の画像オブジェクト
/// * `model_name` - 使用するモデルの名称（デフォルト `"v0.9"`）
/// * `custom_thresholds` - カテゴリ名または ID に対するカスタム閾値の設定（None の場合はデフォルト閾値）
pub fn get_pixai_tags(
    image: &DynamicImage,
    model_name: &str,
    custom_thresholds: Option<HashMap<String, f32>>,
) -> Result<TagResult, TaggingError> {
    let repo_id = resolve_repo_id(model_name);

    // 1. 各種メタデータとモデルファイルのロード
    let model_path = hf_hub_download(&repo_id, "model.onnx", Some("model"), None)?;
    let tags_csv_path = hf_hub_download(&repo_id, "selected_tags.csv", Some("model"), None)?;
    let thresholds_csv_path = hf_hub_download(&repo_id, "thresholds.csv", Some("model"), None);

    // 2. 前処理 (リサイズとテンソル化)
    // アルファチャンネルがある場合、白背景 [255, 255, 255] とブレンドして RGB に強制変換
    let rgb_img = crate::image::force_image_background(image, [255, 255, 255]);
    // 448x448 に歪みリサイズ（Bilinear 補間相当の Triangle フィルタ）
    let resized = rgb_img.resize_exact(448, 448, image::imageops::FilterType::Triangle);
    // mean: 0.5, std: 0.5 で [-1.0, 1.0] に正規化して CHW テンソルを構築
    let input_tensor = to_ndarray_chw(&resized, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])?;

    let mut session = create_onnx_session(&model_path)?;
    let outputs = run_onnx_session(&mut session, "input", &input_tensor)?;

    // "prediction" 出力テンソルを名前で明示的に解決して取り出します
    let output_value = outputs.get("prediction").ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape(
            "No prediction output tensor found".to_string(),
        )
    })?;

    let (_shape, prediction) = output_value.try_extract_tensor::<f32>()?;

    // 4. selected_tags.csv のロードとパース
    let mut rdr = csv::Reader::from_path(&tags_csv_path)?;
    let mut tags_definition = Vec::new();
    let mut character_to_ips = HashMap::new();

    for result in rdr.deserialize() {
        let record: CsvTagRecord = result?;

        // ips カラムから原作 IP リストを JSON パース
        let ips: Vec<String> = serde_json::from_str(&record.ips).unwrap_or_default();
        if !ips.is_empty() {
            character_to_ips.insert(record.name.clone(), ips);
        }

        tags_definition.push(record);
    }

    if prediction.len() != tags_definition.len() {
        return Err(TaggingError::InvalidArgument(format!(
            "Model prediction size ({}) does not match tag list size ({})",
            prediction.len(),
            tags_definition.len()
        )));
    }

    // 5. 閾値情報のパース
    let mut default_thresholds = HashMap::new();
    let mut category_names = HashMap::new();

    if let Ok(path) = thresholds_csv_path {
        let rdr_res = csv::Reader::from_path(path);
        if let Ok(mut rdr) = rdr_res {
            for record in rdr.deserialize().flatten() {
                let record: CsvThresholdRecord = record;
                default_thresholds.insert(record.category, record.threshold);
                category_names.insert(record.category, record.name);
            }
        }
    }

    // カテゴリ名の解決用のデフォルトフォールバック
    category_names
        .entry(0)
        .or_insert_with(|| "general".to_string());
    category_names
        .entry(4)
        .or_insert_with(|| "character".to_string());

    // 6. タグ判定としきい値適用
    let mut general_tags = Vec::new();
    let mut character_tags = Vec::new();
    let mut rest_tags: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    let mut all_tags = Vec::new();

    for record in tags_definition {
        let prob = prediction[record.id];
        let category = record.category;
        let category_name = category_names
            .get(&category)
            .cloned()
            .unwrap_or_else(|| category.to_string());

        // 閾値の決定
        let threshold = if let Some(ref custom) = custom_thresholds {
            // カスタム閾値の検索 (名前優先、IDフォールバック)
            custom
                .get(&category_name)
                .or_else(|| custom.get(&category.to_string()))
                .cloned()
                .unwrap_or_else(|| default_thresholds.get(&category).cloned().unwrap_or(0.4))
        } else {
            default_thresholds.get(&category).cloned().unwrap_or(0.4)
        };

        if prob >= threshold {
            let tag_entry = (record.name, prob);
            all_tags.push(tag_entry.clone());

            match category {
                0 => general_tags.push(tag_entry),
                4 => character_tags.push(tag_entry),
                _ => {
                    rest_tags.entry(category_name).or_default().push(tag_entry);
                }
            }
        }
    }

    // 7. ソート処理 (確率降順、タグ名昇順)
    let sort_tags_fn = |a: &(String, f32), b: &(String, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };

    general_tags.sort_by(sort_tags_fn);
    character_tags.sort_by(sort_tags_fn);
    all_tags.sort_by(sort_tags_fn);

    for list in rest_tags.values_mut() {
        list.sort_by(sort_tags_fn);
    }

    // 8. 原作 IP (著作権) マッピング
    let mut ips_counts = HashMap::new();
    for (tag_name, _) in &all_tags {
        if let Some(ips_list) = character_to_ips.get(tag_name) {
            for ip in ips_list {
                *ips_counts.entry(ip.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut sorted_ips: Vec<String> = ips_counts.keys().cloned().collect();
    sorted_ips.sort_by(|a, b| {
        let count_a = ips_counts.get(a).unwrap_or(&0);
        let count_b = ips_counts.get(b).unwrap_or(&0);
        count_b.cmp(count_a).then_with(|| a.cmp(b))
    });

    Ok(TagResult {
        general: general_tags,
        character: character_tags,
        rest: rest_tags,
        tag: all_tags,
        ips: sorted_ips,
    })
}
