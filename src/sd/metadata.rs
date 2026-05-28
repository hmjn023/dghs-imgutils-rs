//! A1111 (Automatic1111) 形式の Stable Diffusion メタデータのパース。

use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;

/// パラメーター行のキー／値ペアにマッチする正規表現。
static PARAM_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"\s*(?P<key>[\w ]+):\s*(?P<value>"(?:\\.|[^\\"])+"|[^,]*)(?:,|$)"#).unwrap()
});

/// "NxM" 形式のサイズ文字列にマッチする正規表現。
static SIZE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?P<size1>-?\d+)\s*x\s*(?P<size2>-?\d+)$").unwrap());

/// Negative prompt 行にマッチする正規表現。
static NEG_LINE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^Negative prompt:\s*(?P<content>[\S\s]*?)$").unwrap());

/// A1111 形式のパラメーターキー (先頭グループ)。
const PRE_KEYS: &[&str] = &[
    "Steps",
    "Sampler",
    "CFG scale",
    "Image CFG scale",
    "Seed",
    "Face restoration",
    "Size",
    "Model hash",
    "Model",
    "VAE hash",
    "VAE",
    "Variation seed",
    "Variation seed strength",
    "Seed resize from",
    "Denoising strength",
    "Conditional mask weight",
    "Clip skip",
    "ENSD",
    "Token merging ratio",
    "Token merging ratio hr",
    "Init image hash",
    "RNG",
    "NGMS",
    "Tiling",
];

/// A1111 形式のパラメーターキー (末尾グループ)。
const POST_KEYS: &[&str] = &["Version", "User"];

/// A1111 WebUI の生成メタデータを表現する構造体。
#[derive(Debug, Clone)]
pub struct SDMetaData {
    /// メインプロンプト
    pub prompt: String,
    /// ネガティブプロンプト (空文字列の可能性あり)
    pub neg_prompt: String,
    /// パラメーター (Steps, Sampler, Seed など)
    pub parameters: HashMap<String, String>,
}

impl SDMetaData {
    /// メタデータを A1111 形式の文字列として出力します。
    pub fn to_geninfo_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(self.prompt.clone());
        if !self.neg_prompt.is_empty() {
            lines.push(format!("Negative prompt: {}", self.neg_prompt));
        }

        let ext_keys: Vec<&String> = self
            .parameters
            .keys()
            .filter(|k| !PRE_KEYS.contains(&k.as_str()) && !POST_KEYS.contains(&k.as_str()))
            .collect();

        let mut param_parts = Vec::new();
        for key in PRE_KEYS {
            if let Some(v) = self.parameters.get(*key) {
                param_parts.push(format!("{}: {}", key, quote_value(key, v)));
            }
        }
        for key in &ext_keys {
            if let Some(v) = self.parameters.get(*key) {
                param_parts.push(format!("{}: {}", key, quote_value(key, v)));
            }
        }
        for key in POST_KEYS {
            if let Some(v) = self.parameters.get(*key) {
                param_parts.push(format!("{}: {}", key, quote_value(key, v)));
            }
        }

        if !param_parts.is_empty() {
            lines.push(param_parts.join(", "));
        }

        lines.join("\n")
    }
}

fn quote_value(key: &str, value: &str) -> String {
    if key == "Size" {
        return value.to_string();
    }
    if value.contains(',') || value.contains('\n') || value.contains(':') {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    } else {
        value.to_string()
    }
}

/// パラメーター文字列をパースしてキー／値マップを返します。
fn parse_parameters(param_text: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    for cap in PARAM_PATTERN.captures_iter(param_text) {
        let key = cap["key"].trim().to_string();
        let mut value: String = cap["value"].trim().to_string();

        // JSON 引用符を外す
        if value.starts_with('"') && value.ends_with('"') {
            if let Ok(decoded) = serde_json::from_str::<String>(&value) {
                value = decoded;
            }
        }

        // Size パターンマッチ
        if let Some(size_cap) = SIZE_PATTERN.captures(&value) {
            let s1 = size_cap["size1"].to_string();
            let s2 = size_cap["size2"].to_string();
            params.insert(key, format!("{}x{}", s1, s2));
        } else {
            params.insert(key, value);
        }
    }
    params
}

/// テキストから SD メタデータをパースします。
///
/// # 引数
///
/// * `x` — A1111 形式の geninfo 文字列 (プロンプト、Negative prompt、パラメーター行)
///
/// # 戻り値
///
/// パースされた `SDMetaData`。
///
/// # フォーマット例
///
/// ```text
/// 1girl, solo, blue eyes,
/// Negative prompt: ugly, lowres,
/// Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 12345, Size: 512x768
/// ```
pub fn parse_sdmeta_from_text(x: &str) -> SDMetaData {
    let x = x.trim();
    let all_lines: Vec<&str> = x.lines().collect();

    let (prompt_lines, argument_line) = if all_lines.is_empty() {
        (vec![], "")
    } else {
        let last = all_lines[all_lines.len() - 1];
        // パラメーターパターンが 3 つ以上マッチすれば最終行はパラメーター行と判定
        let param_count = PARAM_PATTERN.find_iter(last).count();
        if param_count >= 3 {
            (all_lines[..all_lines.len() - 1].to_vec(), last)
        } else {
            (all_lines, "")
        }
    };

    // 状態機械: 0x1 = prompt, 0x2 = neg_prompt
    let mut status: u8 = 0x1;
    let mut prompt_parts = Vec::new();
    let mut neg_prompt_parts = Vec::new();

    for line in &prompt_lines {
        let line = line.trim();
        if let Some(neg_cap) = NEG_LINE_PATTERN.captures(line) {
            let content = neg_cap["content"].trim().to_string();
            neg_prompt_parts.push(content);
            status = 0x2;
        } else if status == 0x1 {
            prompt_parts.push(line.to_string());
        } else if status == 0x2 {
            neg_prompt_parts.push(line.to_string());
        }
    }

    let prompt = prompt_parts.join("\n").trim().to_string();
    let neg_prompt = neg_prompt_parts.join("\n").trim().to_string();
    let parameters = parse_parameters(argument_line);

    SDMetaData {
        prompt,
        neg_prompt,
        parameters,
    }
}

// ─── Image-level functions ───

use crate::metadata::geninfo::{
    read_geninfo_exif, read_geninfo_gif, read_geninfo_parameters, write_geninfo_exif,
    write_geninfo_gif, write_geninfo_parameters,
};

/// 画像ファイルから SD メタデータを読み取ります。
///
/// PNG の場合は `parameters` テキストチャンク、
/// JPEG/WebP の場合は EXIF UserComment、
/// GIF の場合はコメント拡張ブロックから取得します。
///
/// * `path` — 画像ファイルのパス
pub fn get_sdmeta_from_image(path: &str) -> Option<SDMetaData> {
    // 優先順位: PNG parameters → EXIF UserComment → GIF comment
    let text = read_geninfo_parameters(path)
        .or_else(|| read_geninfo_exif(path))
        .or_else(|| read_geninfo_gif(path))?;

    // 空テキストは無効
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    // NAI メタデータでないことを簡易確認
    if text.contains("tEXtSoftware") && text.contains("NovelAI") {
        return None;
    }

    Some(parse_sdmeta_from_text(text))
}

/// 画像ファイルに SD メタデータを保存します。
///
/// 保存先のファイル形式に応じて適切な書き込み方法を選択します:
/// - PNG: `parameters` テキストチャンク
/// - JPEG/WebP: EXIF UserComment
/// - GIF: コメント拡張ブロック
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `metadata` — 書き込む SD メタデータ
pub fn save_image_with_sdmeta(
    src_path: &str,
    dst_path: &str,
    metadata: &SDMetaData,
) -> std::io::Result<()> {
    let geninfo_text = metadata.to_geninfo_text();
    let ext = std::path::Path::new(dst_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "png" => write_geninfo_parameters(src_path, dst_path, &geninfo_text),
        "jpg" | "jpeg" | "webp" => write_geninfo_exif(src_path, dst_path, &geninfo_text),
        "gif" => write_geninfo_gif(src_path, dst_path, &geninfo_text),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unsupported image format: .{}", ext),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_metadata() {
        let text = "1girl, solo, blue eyes,\nNegative prompt: ugly, lowres,\nSteps: 20, Sampler: DDIM, CFG scale: 7, Seed: 12345, Size: 512x768";
        let meta = parse_sdmeta_from_text(text);
        assert!(meta.prompt.contains("1girl"));
        assert_eq!(meta.neg_prompt, "ugly, lowres,");
        assert_eq!(meta.parameters.get("Steps").map(|s| s.as_str()), Some("20"));
        assert_eq!(
            meta.parameters.get("Sampler").map(|s| s.as_str()),
            Some("DDIM")
        );
        assert_eq!(
            meta.parameters.get("Size").map(|s| s.as_str()),
            Some("512x768")
        );
    }

    #[test]
    fn test_parse_metadata_no_neg_prompt() {
        let text = "a cat\nSteps: 10, Sampler: Euler, Seed: 999";
        let meta = parse_sdmeta_from_text(text);
        assert_eq!(meta.prompt, "a cat");
        assert!(meta.neg_prompt.is_empty());
        assert_eq!(meta.parameters.get("Steps").map(|s| s.as_str()), Some("10"));
    }

    #[test]
    fn test_parse_metadata_multiline_prompt() {
        let text = "line1,\nline2,\nline3\nSteps: 1, Sampler: A, CFG scale: 1, Seed: 1";
        let meta = parse_sdmeta_from_text(text);
        assert_eq!(meta.prompt, "line1,\nline2,\nline3");
    }

    #[test]
    fn test_roundtrip_geninfo_text() {
        let text = "1girl, solo,\nNegative prompt: ugly,\nSteps: 20, Sampler: DDIM, CFG scale: 7, Seed: 12345, Size: 512x768, Model: test";
        let meta = parse_sdmeta_from_text(text);
        let regenerated = meta.to_geninfo_text();
        assert!(regenerated.contains("1girl, solo,"));
        assert!(regenerated.contains("Negative prompt: ugly,"));
        assert!(regenerated.contains("Steps: 20"));
        assert!(regenerated.contains("Size: 512x768"));
    }

    #[test]
    fn test_parse_parameters_simple() {
        let params = parse_parameters("Steps: 20, Sampler: DDIM, CFG scale: 7");
        assert_eq!(params.get("Steps").map(|s| s.as_str()), Some("20"));
        assert_eq!(params.get("Sampler").map(|s| s.as_str()), Some("DDIM"));
        assert_eq!(params.get("CFG scale").map(|s| s.as_str()), Some("7"));
    }

    #[test]
    fn test_parse_parameters_size() {
        let params = parse_parameters("Size: 512x768, Seed: 42");
        assert_eq!(params.get("Size").map(|s| s.as_str()), Some("512x768"));
    }

    #[test]
    fn test_parse_complex_metadata() {
        let text = "1girl, solo,\nNegative prompt: ugly,\nSteps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 2647703743, Size: 768x768, Model hash: 72bd94132e, Model: CuteMix, Denoising strength: 0.7, Version: v1.5.1";
        let meta = parse_sdmeta_from_text(text);
        assert_eq!(
            meta.parameters.get("Sampler").map(|s| s.as_str()),
            Some("DPM++ 2M SDE Karras")
        );
        assert_eq!(
            meta.parameters.get("Version").map(|s| s.as_str()),
            Some("v1.5.1")
        );
    }
}
