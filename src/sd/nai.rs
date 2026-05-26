//! NovelAI (NAI) メタデータのパース。

use std::collections::HashMap;

use image::DynamicImage;

use crate::metadata::lsb;

/// NovelAI メタデータを表現する構造体。
#[derive(Debug, Clone)]
pub struct NAIMetaData {
    /// 生成に使用されたソフトウェア名
    pub software: String,
    /// 画像のソース
    pub source: String,
    /// 生成パラメーター (JSON オブジェクト)
    pub parameters: HashMap<String, serde_json::Value>,
    /// タイトル (任意)
    pub title: Option<String>,
    /// 生成時間 (秒, 任意)
    pub generation_time: Option<f64>,
    /// 説明 (任意)
    pub description: Option<String>,
}

/// NAI メタデータのバリデーションを行います。
fn validate_nai_data(data: &HashMap<String, String>) -> bool {
    if !data.contains_key("Software")
        || !data.contains_key("Source")
        || !data.contains_key("Comment")
    {
        return false;
    }
    // Comment が有効な JSON かチェック
    if let Some(comment) = data.get("Comment") {
        if serde_json::from_str::<serde_json::Value>(comment).is_err() {
            return false;
        }
    } else {
        return false;
    }
    // Generation time があれば float としてパース可能か
    if let Some(gt) = data.get("Generation time") {
        if gt.parse::<f64>().is_err() {
            return false;
        }
    }
    true
}

/// NAI メタデータを画像の LSB から読み取ります。
fn read_naimeta_lsb(image: &DynamicImage) -> Option<HashMap<String, String>> {
    let meta = lsb::read_lsb_metadata(image);
    if validate_nai_data(&meta) {
        Some(meta)
    } else {
        None
    }
}

#[allow(dead_code)]
fn read_naimeta_info(_image: &DynamicImage) -> Option<HashMap<String, String>> {
    None
}

/// 画像から NAI メタデータを抽出します。
///
/// 以下の順で試行します:
/// 1. LSB ステガノグラフィ
/// 2. PNG テキストチャンク (将来拡張)
///
/// # 引数
///
/// * `image` — 入力画像
///
/// # 戻り値
///
/// 抽出された `NAIMetaData`、見つからなければ `None`。
pub fn get_naimeta_from_image(image: &DynamicImage) -> Option<NAIMetaData> {
    // 1. LSB を試行
    if let Some(data) = read_naimeta_lsb(image) {
        return Some(naimeta_from_map(&data));
    }

    None
}

fn naimeta_from_map(data: &HashMap<String, String>) -> NAIMetaData {
    let software = data.get("Software").cloned().unwrap_or_default();
    let source = data.get("Source").cloned().unwrap_or_default();
    let parameters = data
        .get("Comment")
        .and_then(|c| serde_json::from_str(c).ok())
        .unwrap_or_default();
    let title = data.get("Title").cloned();
    let generation_time = data
        .get("Generation time")
        .and_then(|v| v.parse::<f64>().ok());
    let description = data.get("Description").cloned();

    NAIMetaData {
        software,
        source,
        parameters,
        title,
        generation_time,
        description,
    }
}

/// NAI メタデータを画像に LSB で書き込みます。
///
/// # 引数
///
/// * `image` — 入力画像
/// * `metadata` — 書き込む NAI メタデータ
///
/// # 戻り値
///
/// LSB データが埋め込まれた新しい画像。
pub fn add_naimeta_to_image(image: &DynamicImage, metadata: &NAIMetaData) -> DynamicImage {
    let mut map = HashMap::new();
    map.insert("Software".to_string(), metadata.software.clone());
    map.insert("Source".to_string(), metadata.source.clone());
    map.insert(
        "Comment".to_string(),
        serde_json::to_string(&metadata.parameters).unwrap_or_default(),
    );
    if let Some(ref title) = metadata.title {
        map.insert("Title".to_string(), title.clone());
    }
    if let Some(gt) = metadata.generation_time {
        map.insert("Generation time".to_string(), gt.to_string());
    }
    if let Some(ref desc) = metadata.description {
        map.insert("Description".to_string(), desc.clone());
    }

    lsb::write_lsb_metadata(image, &map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    fn create_test_rgba() -> DynamicImage {
        let mut img = RgbaImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.put_pixel(x, y, image::Rgba([128, 128, 128, 255]));
            }
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_nai_write_read_lsb() {
        let img = create_test_rgba();
        let mut params = HashMap::new();
        params.insert(
            "prompt".to_string(),
            serde_json::Value::String("a cat".to_string()),
        );
        params.insert(
            "steps".to_string(),
            serde_json::Value::Number(serde_json::Number::from(20)),
        );

        let nai = NAIMetaData {
            software: "NovelAI".to_string(),
            source: "user".to_string(),
            parameters: params,
            title: Some("Test".to_string()),
            generation_time: Some(1.5),
            description: None,
        };

        let encoded = add_naimeta_to_image(&img, &nai);
        let decoded = get_naimeta_from_image(&encoded).unwrap();

        assert_eq!(decoded.software, "NovelAI");
        assert_eq!(decoded.source, "user");
        assert_eq!(decoded.title, Some("Test".to_string()));
        assert!((decoded.generation_time.unwrap() - 1.5).abs() < 1e-6);
        assert_eq!(
            decoded.parameters.get("prompt").and_then(|v| v.as_str()),
            Some("a cat")
        );
    }

    #[test]
    fn test_get_naimeta_from_image_none() {
        let img = create_test_rgba();
        let result = get_naimeta_from_image(&img);
        assert!(result.is_none());
    }
}
