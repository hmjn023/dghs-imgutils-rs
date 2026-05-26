//! PNG / EXIF / GIF 画像から生成情報 (geninfo) を読み取る機能を提供します。

use std::collections::HashMap;
use std::path::Path;

/// PNG ファイルを生バイト列から解析し、テキストチャンク (tEXt / iTXt) を
/// 「キーワード→本文」のマップとして返します。
fn parse_png_text_chunks(data: &[u8]) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if data.len() < 8 {
        return result;
    }
    // シグネチャチェック: PNG
    let sig = &data[..8];
    if sig != [137, 80, 78, 71, 13, 10, 26, 10] {
        return result;
    }
    let mut pos = 8usize;
    while pos + 8 <= data.len() {
        let len_bytes = &data[pos..pos + 4];
        let chunk_len =
            u32::from_be_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        if chunk_type == b"tEXt" {
            if pos + 8 + chunk_len + 4 <= data.len() {
                let chunk_data = &data[pos + 8..pos + 8 + chunk_len];
                if let Some(null_pos) = chunk_data.iter().position(|&b| b == 0) {
                    let keyword = String::from_utf8_lossy(&chunk_data[..null_pos]).to_string();
                    let text = String::from_utf8_lossy(&chunk_data[null_pos + 1..]).to_string();
                    result.insert(keyword, text);
                }
            }
        } else if chunk_type == b"iTXt" {
            if pos + 8 + chunk_len + 4 <= data.len() {
                let chunk_data = &data[pos + 8..pos + 8 + chunk_len];
                let mut offset = 0usize;
                // keyword: null-terminated
                if let Some(kw_end) = chunk_data[offset..].iter().position(|&b| b == 0) {
                    let keyword =
                        String::from_utf8_lossy(&chunk_data[offset..offset + kw_end]).to_string();
                    offset += kw_end + 1;
                    // compression flag (1 byte) + compression method (1 byte)
                    if offset + 2 <= chunk_data.len() {
                        offset += 2;
                        // language tag: null-terminated
                        if let Some(lang_end) = chunk_data[offset..].iter().position(|&b| b == 0) {
                            offset += lang_end + 1;
                            // translated keyword: null-terminated
                            if let Some(tk_end) = chunk_data[offset..].iter().position(|&b| b == 0)
                            {
                                offset += tk_end + 1;
                                // text: remaining
                                let text =
                                    String::from_utf8_lossy(&chunk_data[offset..]).to_string();
                                result.insert(keyword, text);
                            }
                        }
                    }
                }
            }
        } else if chunk_type == b"IEND" {
            break;
        }
        pos += 4 + 4 + chunk_len + 4; // length + type + data + crc
    }
    result
}

/// PNG 画像の iTXt/tEXt チャンクから `parameters` キーに対応する生成情報を読み取ります。
///
/// # 引数
///
/// * `path` — 画像ファイルのパス (PNG 形式推奨)
///
/// # 戻り値
///
/// 生成情報文字列。見つからなければ `None`。
pub fn read_geninfo_parameters(path: &str) -> Option<String> {
    read_geninfo_raw(path).and_then(|map| map.get("parameters").cloned())
}

/// PNG 画像からすべてのテキストチャンクを読み取り、キー／値のマップとして返します。
pub fn read_geninfo_raw(path: &str) -> Option<HashMap<String, String>> {
    let path = Path::new(path);
    let data = std::fs::read(path).ok()?;
    Some(parse_png_text_chunks(&data))
}

/// EXIF データから生成情報を読み取ります (UserComment フィールド)。
pub fn read_geninfo_exif(path: &str) -> Option<String> {
    let path = Path::new(path);
    let data = std::fs::read(path).ok()?;
    // PNG の場合は tEXt/iTXt の中に "exif" チャンクがある可能性 → 簡易スキップ
    // 実際の EXIF は JPEG/WebP で利用。ここでは簡易的にバイト列から "UserComment" を探索
    if let Some(comment) = extract_exif_user_comment(&data) {
        return Some(comment);
    }
    None
}

/// GIF コメントから生成情報を読み取ります。
pub fn read_geninfo_gif(path: &str) -> Option<String> {
    let path = Path::new(path);
    let data = std::fs::read(path).ok()?;
    extract_gif_comment(&data)
}

/// バイト列から EXIF UserComment を簡易抽出します。
fn extract_exif_user_comment(data: &[u8]) -> Option<String> {
    // UserComment の EXIF IFD タグは 0x9286
    // 簡易的な実装: バイト列中から "UserComment" またはタグ 0x9286 を探索
    // 本格的な実装には `kamadak-exif` や `rexif` クレート推奨
    // ここでは将来の拡張のためのプレースホルダーとして簡単なテキスト探索を行う
    if let Ok(s) = std::str::from_utf8(data) {
        // COM marker などに "parameters" が含まれる場合に対応
        if let Some(pos) = s.find("parameters") {
            let end = s[pos..].find('\0').map(|e| pos + e).unwrap_or(s.len());
            let value = s[pos..end].trim();
            if let Some(eq) = value.find(':') {
                return Some(value[eq + 1..].trim().to_string());
            }
        }
    }
    None
}

/// GIF ファイルからコメント拡張ブロックを抽出します。
fn extract_gif_comment(data: &[u8]) -> Option<String> {
    // GIF シグネチャ: GIF87a または GIF89a
    if data.len() < 6 {
        return None;
    }
    let sig = &data[..6];
    if sig != b"GIF87a" && sig != b"GIF89a" {
        return None;
    }
    // コメント拡張ブロック: 0x21 0xFE [size] [data...] 0x00
    let mut pos = 6usize;
    while pos + 1 < data.len() {
        if data[pos] == 0x21 && pos + 1 < data.len() {
            if data[pos + 1] == 0xFE {
                // コメント拡張ブロック
                pos += 2;
                let mut comment = Vec::new();
                while pos < data.len() && data[pos] != 0x00 {
                    let block_size = data[pos] as usize;
                    pos += 1;
                    if pos + block_size <= data.len() {
                        comment.extend_from_slice(&data[pos..pos + block_size]);
                        pos += block_size;
                    } else {
                        break;
                    }
                }
                if !comment.is_empty() {
                    return Some(String::from_utf8_lossy(&comment).to_string());
                }
            }
        }
        pos += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_png_with_text(key: &str, value: &str) -> Vec<u8> {
        // Minimal valid PNG with one tEXt chunk
        let mut png = Vec::new();
        // Signature
        png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
        // IHDR chunk (13 bytes data)
        let ihdr_data = &[
            0, 0, 0, 1, // width=1
            0, 0, 0, 1, // height=1
            8, // bit depth
            2, // color type RGB
            0, 0, 0, // compression, filter, interlace
        ];
        let ihdr_len = 13u32;
        png.extend_from_slice(&ihdr_len.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(ihdr_data);
        // CRC (simplified, just placeholder)
        png.extend_from_slice(&[0u8; 4]);

        // tEXt chunk
        let text_data = [key.as_bytes(), &[0], value.as_bytes()].concat();
        let text_len = text_data.len() as u32;
        png.extend_from_slice(&text_len.to_be_bytes());
        png.extend_from_slice(b"tEXt");
        png.extend_from_slice(&text_data);
        png.extend_from_slice(&[0u8; 4]);

        // IEND chunk
        png.extend_from_slice(&0u32.to_be_bytes());
        png.extend_from_slice(b"IEND");
        png.extend_from_slice(&[0u8; 4]);

        png
    }

    #[test]
    fn test_parse_png_text_chunks() {
        let png = create_test_png_with_text("parameters", "test prompt");
        let map = parse_png_text_chunks(&png);
        assert_eq!(
            map.get("parameters").map(|s| s.as_str()),
            Some("test prompt")
        );
    }

    #[test]
    fn test_read_geninfo_parameters() {
        let png = create_test_png_with_text("parameters", "Steps: 20, Sampler: DDIM");
        let dir = std::env::temp_dir();
        let path = dir.join("test_geninfo.png");
        std::fs::write(&path, &png).unwrap();
        let result = read_geninfo_parameters(path.to_str().unwrap());
        assert_eq!(result, Some("Steps: 20, Sampler: DDIM".to_string()));
        std::fs::remove_file(&path).unwrap_or(());
    }

    #[test]
    fn test_read_geninfo_parameters_not_found() {
        let png = create_test_png_with_text("other", "value");
        let dir = std::env::temp_dir();
        let path = dir.join("test_geninfo_none.png");
        std::fs::write(&path, &png).unwrap();
        let result = read_geninfo_parameters(path.to_str().unwrap());
        assert!(result.is_none());
        std::fs::remove_file(&path).unwrap_or(());
    }
}
