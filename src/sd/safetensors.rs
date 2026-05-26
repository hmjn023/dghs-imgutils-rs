//! Safetensors 形式のモデルファイルからメタデータを読み取ります。
//!
//! Safetensors ファイルフォーマット:
//! - 8 bytes: ヘッダーサイズ (u64, little-endian)
//! - N bytes: JSON ヘッダー (UTF-8)
//! - 残り: テンサーデータ

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

/// Safetensors ファイルからメタデータ辞書を読み取ります。
///
/// # 引数
///
/// * `path` — `.safetensors` ファイルのパス
///
/// # 戻り値
///
/// メタデータのキー／値マップ。メタデータがない場合は空のマップ。
///
/// # エラー
///
/// ファイルが開けない場合や JSON のパースに失敗した場合は `Err` を返します。
pub fn read_safetensors_metadata(path: &str) -> Result<HashMap<String, String>, String> {
    let path = Path::new(path);
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;

    // ヘッダーサイズ (u64 LE) を読み取り
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)
        .map_err(|e| format!("Failed to read header size: {}", e))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    // ヘッダー JSON を読み取り
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)
        .map_err(|e| format!("Failed to read header: {}", e))?;

    let header_str =
        String::from_utf8(header_buf).map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

    let header: Value =
        serde_json::from_str(&header_str).map_err(|e| format!("Invalid JSON in header: {}", e))?;

    // __metadata__ フィールドを抽出
    let metadata = header
        .get("__metadata__")
        .and_then(|m| m.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                .collect::<HashMap<String, String>>()
        })
        .unwrap_or_default();

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_safetensors(metadata: Option<&HashMap<String, String>>) -> Vec<u8> {
        let mut json_map = serde_json::Map::new();

        if let Some(meta) = metadata {
            let mut meta_map = serde_json::Map::new();
            for (k, v) in meta {
                meta_map.insert(k.clone(), Value::String(v.clone()));
            }
            json_map.insert("__metadata__".to_string(), Value::Object(meta_map));
        }

        // ダミーテンソル
        json_map.insert(
            "test_tensor".to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [1, 3, 64, 64],
                "data_offsets": [0, 49152]
            }),
        );

        let header_json = serde_json::to_string(&json_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut result = Vec::new();
        result.extend_from_slice(&header_size.to_le_bytes());
        result.extend_from_slice(header_bytes);
        // パディング (テンサーデータの代わり)
        result.resize(result.len() + 49152, 0);

        result
    }

    #[test]
    fn test_read_safetensors_metadata() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "test".to_string());
        meta.insert("version".to_string(), "1.0".to_string());

        let data = create_test_safetensors(Some(&meta));
        let dir = std::env::temp_dir();
        let path = dir.join("test_model.safetensors");
        std::fs::write(&path, &data).unwrap();

        let result = read_safetensors_metadata(path.to_str().unwrap()).unwrap();
        assert_eq!(result.get("author").map(|s| s.as_str()), Some("test"));
        assert_eq!(result.get("version").map(|s| s.as_str()), Some("1.0"));

        std::fs::remove_file(&path).unwrap_or(());
    }

    #[test]
    fn test_read_safetensors_metadata_empty() {
        let data = create_test_safetensors(None);
        let dir = std::env::temp_dir();
        let path = dir.join("test_model_no_meta.safetensors");
        std::fs::write(&path, &data).unwrap();

        let result = read_safetensors_metadata(path.to_str().unwrap()).unwrap();
        assert!(result.is_empty());

        std::fs::remove_file(&path).unwrap_or(());
    }

    #[test]
    fn test_read_safetensors_file_not_found() {
        let result = read_safetensors_metadata("/nonexistent/file.safetensors");
        assert!(result.is_err());
    }
}
