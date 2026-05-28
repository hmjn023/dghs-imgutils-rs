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

// ─── Write functions ───

/// PNG ファイルの IEND チャンク直前に tEXt チャンクを挿入して保存します。
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `key` — テキストチャンクのキーワード (例: `"parameters"`)
/// * `value` — 書き込むテキスト
pub fn write_geninfo_png(
    src_path: &str,
    dst_path: &str,
    key: &str,
    value: &str,
) -> std::io::Result<()> {
    let data = std::fs::read(src_path)?;
    let new_data = inject_png_text_chunk(&data, key, value)?;
    std::fs::write(dst_path, &new_data)
}

/// PNG ファイルに `parameters` キーで生成情報を書き込みます。
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `geninfo` — 書き込む生成情報文字列
pub fn write_geninfo_parameters(
    src_path: &str,
    dst_path: &str,
    geninfo: &str,
) -> std::io::Result<()> {
    write_geninfo_png(src_path, dst_path, "parameters", geninfo)
}

/// JPEG/WebP ファイルに EXIF UserComment として生成情報を書き込みます。
///
/// EXIF UserComment タグ (0x9286) に Unicode エンコーディングで保存します。
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `geninfo` — 書き込む生成情報文字列
pub fn write_geninfo_exif(src_path: &str, dst_path: &str, geninfo: &str) -> std::io::Result<()> {
    let data = std::fs::read(src_path)?;
    let new_data = inject_exif_user_comment(&data, geninfo)?;
    std::fs::write(dst_path, &new_data)
}

/// GIF ファイルにコメント拡張ブロックとして生成情報を書き込みます。
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `geninfo` — 書き込む生成情報文字列
pub fn write_geninfo_gif(src_path: &str, dst_path: &str, geninfo: &str) -> std::io::Result<()> {
    let data = std::fs::read(src_path)?;
    let new_data = inject_gif_comment(&data, geninfo)?;
    std::fs::write(dst_path, &new_data)
}

/// PNG バイト列の IEND チャンク直前に tEXt チャンクを挿入します。
fn inject_png_text_chunk(data: &[u8], key: &str, value: &str) -> std::io::Result<Vec<u8>> {
    if data.len() < 8 || &data[..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid PNG file",
        ));
    }

    // IEND チャンクの位置を探す
    let mut iend_pos = None;
    let mut pos = 8usize;
    while pos + 8 <= data.len() {
        let chunk_len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        if chunk_type == b"IEND" {
            iend_pos = Some(pos);
            break;
        }
        pos += 4 + 4 + chunk_len + 4; // length + type + data + crc
    }

    let iend_pos = iend_pos.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "IEND chunk not found")
    })?;

    // tEXt チャンクを構築: keyword + null separator + text
    let mut chunk_data = Vec::new();
    chunk_data.extend_from_slice(key.as_bytes());
    chunk_data.push(0); // null separator
    chunk_data.extend_from_slice(value.as_bytes());

    let chunk_len = chunk_data.len() as u32;
    let mut text_chunk = Vec::new();
    text_chunk.extend_from_slice(&chunk_len.to_be_bytes());
    text_chunk.extend_from_slice(b"tEXt");
    text_chunk.extend_from_slice(&chunk_data);
    // CRC32 for tEXt chunk (type + data)
    let crc = crc32_ieee(&[b"tEXt".as_slice(), &chunk_data].concat());
    text_chunk.extend_from_slice(&crc.to_be_bytes());

    // IEND の前に挿入
    let mut result = Vec::with_capacity(data.len() + text_chunk.len());
    result.extend_from_slice(&data[..iend_pos]);
    result.extend_from_slice(&text_chunk);
    result.extend_from_slice(&data[iend_pos..]);

    Ok(result)
}

/// JPEG バイト列に EXIF UserComment を挿入します。
/// 簡易実装: APP1 セグメントを新規作成して挿入します。
fn inject_exif_user_comment(data: &[u8], comment: &str) -> std::io::Result<Vec<u8>> {
    // JPEG シグネチャ確認
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid JPEG file",
        ));
    }

    // UserComment を Unicode (UTF-16LE) でエンコード
    // EXIF UserComment: "UNICODE\0\0" プレフィックス + UTF-16LE バイト列
    let mut comment_bytes = Vec::new();
    comment_bytes.extend_from_slice(b"UNICODE\0\0");
    for ch in comment.encode_utf16() {
        comment_bytes.extend_from_slice(&ch.to_le_bytes());
    }

    // 簡易 EXIF APP1 セグメントを構築
    let exif_data = build_exif_app1_with_user_comment(&comment_bytes)?;

    // SOI (FF D8) の直後に APP1 を挿入
    let mut result = Vec::with_capacity(data.len() + exif_data.len());
    result.extend_from_slice(&data[..2]); // SOI
    result.extend_from_slice(&exif_data);
    // 既存の APP1 があればスキップ
    let mut pos = 2;
    while pos + 4 <= data.len() && data[pos] == 0xFF {
        let marker = data[pos + 1];
        if marker == 0xE1 {
            // 既存 APP1 をスキップ
            if pos + 4 <= data.len() {
                let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + seg_len;
                continue;
            }
        }
        break;
    }
    result.extend_from_slice(&data[pos..]);

    Ok(result)
}

/// 簡易 EXIF APP1 セグメントを構築します。
fn build_exif_app1_with_user_comment(comment_bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    // 最小限の EXIF APP1 セグメント:
    // APP1 marker (FF E1) + length + "Exif\0\0" + TIFF header + IFD0 + IFD(EXIF IFD with UserComment)
    let mut exif_inner = Vec::new();

    // "Exif\0\0" ヘッダー
    exif_inner.extend_from_slice(b"Exif\0\0");

    // TIFF ヘッダー (little-endian)
    let tiff_start = exif_inner.len();
    exif_inner.extend_from_slice(b"II"); // Little-endian
    exif_inner.extend_from_slice(&42u16.to_le_bytes()); // Magic
    exif_inner.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset

    // IFD0: 1 エントリ (ExifTag IFD ポインタ)
    let ifd0_offset = exif_inner.len() - tiff_start;
    exif_inner.extend_from_slice(&1u16.to_le_bytes()); // entry count
    // ExifTag (0x8769) pointing to Exif IFD
    exif_inner.extend_from_slice(&0x8769u16.to_le_bytes()); // tag
    exif_inner.extend_from_slice(&4u16.to_le_bytes()); // type: LONG
    exif_inner.extend_from_slice(&1u32.to_le_bytes()); // count
    let exif_ifd_offset = ifd0_offset as u32 + 2 + 12 * 1 + 4 + 40; // after IFD0 + next_ifd + padding
    exif_inner.extend_from_slice(&exif_ifd_offset.to_le_bytes()); // value
    exif_inner.extend_from_slice(&0u32.to_le_bytes()); // next IFD offset (none)

    // パディング
    exif_inner.extend_from_slice(&[0u8; 40]);

    // Exif IFD: 1 エントリ (UserComment)
    exif_inner.extend_from_slice(&1u16.to_le_bytes()); // entry count
    let user_comment_tag: u16 = 0x9286;
    let user_comment_type: u16 = 7; // UNDEFINED
    let user_comment_count = comment_bytes.len() as u32;
    let user_comment_value_offset = exif_inner.len() as u32 - tiff_start as u32 + 2 + 12 + 4 + 20;

    exif_inner.extend_from_slice(&user_comment_tag.to_le_bytes());
    exif_inner.extend_from_slice(&user_comment_type.to_le_bytes());
    exif_inner.extend_from_slice(&user_comment_count.to_le_bytes());

    if comment_bytes.len() <= 4 {
        // 4 バイト以下なら直接値フィールドに格納
        let mut val = [0u8; 4];
        val[..comment_bytes.len()].copy_from_slice(comment_bytes);
        exif_inner.extend_from_slice(&val);
        exif_inner.extend_from_slice(&0u32.to_le_bytes()); // next IFD
    } else {
        exif_inner.extend_from_slice(&user_comment_value_offset.to_le_bytes());
        exif_inner.extend_from_slice(&0u32.to_le_bytes()); // next IFD
        // パディング
        exif_inner.extend_from_slice(&[0u8; 20]);
        // UserComment データ
        exif_inner.extend_from_slice(comment_bytes);
    }

    // APP1 セグメント全体を構築
    let app1_len = exif_inner.len() + 2; // +2 for length field itself
    let mut app1 = Vec::new();
    app1.extend_from_slice(&[0xFF, 0xE1]); // APP1 marker
    app1.extend_from_slice(&(app1_len as u16).to_be_bytes());
    app1.extend_from_slice(&exif_inner);

    Ok(app1)
}

/// GIF バイト列にコメント拡張ブロックを挿入します。
fn inject_gif_comment(data: &[u8], comment: &str) -> std::io::Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid GIF file",
        ));
    }
    let sig = &data[..6];
    if sig != b"GIF87a" && sig != b"GIF89a" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid GIF file",
        ));
    }

    // 既存のコメント拡張ブロックを除去
    let cleaned = remove_gif_comments(data);

    // コメント拡張ブロックを構築
    let mut comment_block = Vec::new();
    comment_block.push(0x21); // Extension introducer
    comment_block.push(0xFE); // Comment extension label

    // データサブブロック (最大 255 バイトずつ)
    let comment_bytes = comment.as_bytes();
    for chunk in comment_bytes.chunks(255) {
        comment_block.push(chunk.len() as u8);
        comment_block.extend_from_slice(chunk);
    }
    comment_block.push(0x00); // ブロック終端

    // Logical Screen Descriptor の後に挿入
    // LSD は 6 (シグネチャ) + 7 (LSD 固定) = 13 バイト目以降
    // オプションの GCT (Global Color Table) をスキップ
    let mut insert_pos = 6;
    if data.len() >= 13 {
        let packed = data[10];
        let has_gct = packed & 0x80 != 0;
        if has_gct {
            let gct_size = 3 * (1 << ((packed & 0x07) + 1) as usize);
            insert_pos = 13 + gct_size;
        } else {
            insert_pos = 13;
        }
    }

    let insert_pos = insert_pos.min(cleaned.len());
    let mut result = Vec::with_capacity(cleaned.len() + comment_block.len());
    result.extend_from_slice(&cleaned[..insert_pos]);
    result.extend_from_slice(&comment_block);
    result.extend_from_slice(&cleaned[insert_pos..]);

    Ok(result)
}

/// GIF バイト列から既存のコメント拡張ブロックを除去します。
fn remove_gif_comments(data: &[u8]) -> Vec<u8> {
    if data.len() < 6 {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(data.len());
    result.extend_from_slice(&data[..6]); // signature

    let mut pos = 6;
    while pos < data.len() {
        if data[pos] == 0x21 && pos + 1 < data.len() && data[pos + 1] == 0xFE {
            // コメント拡張ブロックをスキップ
            pos += 2;
            while pos < data.len() && data[pos] != 0x00 {
                let block_size = data[pos] as usize;
                pos += 1 + block_size;
            }
            if pos < data.len() {
                pos += 1; // 終端 0x00 をスキップ
            }
        } else if data[pos] == 0x21 && pos + 1 < data.len() {
            // その他の拡張ブロック
            let start = pos;
            pos += 2;
            while pos < data.len() && data[pos] != 0x00 {
                let block_size = data[pos] as usize;
                pos += 1 + block_size;
            }
            if pos < data.len() {
                pos += 1;
            }
            result.extend_from_slice(&data[start..pos]);
        } else if data[pos] == 0x2C {
            // Image Descriptor + 画像データ - 残り全部
            result.extend_from_slice(&data[pos..]);
            return result;
        } else {
            // その他のブロック ( Trailer 等 )
            result.push(data[pos]);
            pos += 1;
        }
    }
    result
}

/// CRC32 (IEEE) を計算します。
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
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
