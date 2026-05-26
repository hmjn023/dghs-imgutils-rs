//! LSB (Least Significant Bit) ステガノグラフィによるメタデータの埋め込みと抽出。
//!
//! 画像の各ピクセルのアルファチャンネルの最下位ビット (LSB) にデータを格納します。
//! フォーマット (Python 版 `dghs-imgutils` との互換):
//!
//! 1. マジックバイト: `stealth_pngcomp` (14 バイト)
//! 2. データ長 (ビット数, big-endian u32, 4 バイト)
//! 3. データ本体 (gzip 圧縮された JSON)
//!
//! ピクセル走査順: カラムメジャー (y が先に変動 → Python 版と互換)

use std::collections::HashMap;
use std::io::{Read, Write};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use image::{DynamicImage, ImageBuffer, Rgba};

const MAGIC: &[u8] = b"stealth_pngcomp";

/// 与えられたバイト列から全ビット (MSB first) をフラットなベクタとして構築します。
fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    bits
}

/// フラットなビット列を (MSB first) バイト列に復元します。
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for &b in chunk {
            byte = (byte << 1) | b;
        }
        bytes.push(byte);
    }
    bytes
}

/// 画像のアルファチャンネルの LSB にビット列を書き込みます。
/// 走査順: y (row) が先に変動 → Python 版と互換
fn write_bits_to_alpha(raw: &mut [u8], width: u32, height: u32, bits: &[u8]) {
    let mut idx = 0;
    for x in 0..width {
        for y in 0..height {
            if idx >= bits.len() {
                return;
            }
            let pixel_offset = ((y * width + x) * 4 + 3) as usize;
            if pixel_offset < raw.len() {
                raw[pixel_offset] = (raw[pixel_offset] & 0xFE) | bits[idx];
            }
            idx += 1;
        }
    }
}

/// 画像のアルファチャンネルの LSB からビット列を読み取ります。
/// 走査順: y (row) が先に変動 → Python 版と互換
fn read_bits_from_alpha(raw: &[u8], width: u32, height: u32, num_bits: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(num_bits);
    'outer: for x in 0..width {
        for y in 0..height {
            if bits.len() >= num_bits {
                break 'outer;
            }
            let pixel_offset = ((y * width + x) * 4 + 3) as usize;
            let bit = if pixel_offset < raw.len() {
                raw[pixel_offset] & 1
            } else {
                0
            };
            bits.push(bit);
        }
    }
    bits
}

/// メタデータマップを LSB ステガノグラフィで画像に埋め込みます。
///
/// # 引数
///
/// * `image` — 入力画像 (RGBA に変換して処理)
/// * `metadata` — 埋め込むキー／値マップ
///
/// # 戻り値
///
/// LSB データが埋め込まれた新しい `DynamicImage`。
pub fn write_lsb_metadata(
    image: &DynamicImage,
    metadata: &HashMap<String, String>,
) -> DynamicImage {
    let json_str = serde_json::to_string(metadata).unwrap_or_default();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(json_str.as_bytes()).unwrap_or(());
    let compressed = encoder.finish().unwrap_or_default();
    write_lsb_raw_bytes(image, &compressed)
}

/// 生バイト列を LSB ステガノグラフィで画像に埋め込みます。
pub fn write_lsb_raw_bytes(image: &DynamicImage, data: &[u8]) -> DynamicImage {
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut raw = rgba.into_raw();

    // 全ビット列を構築: MAGIC + 長さ(32bit) + データ
    let mut all_bits = Vec::new();
    all_bits.extend_from_slice(&bytes_to_bits(MAGIC));
    all_bits.extend_from_slice(&bytes_to_bits(&(data.len() as u32 * 8).to_be_bytes()));
    all_bits.extend_from_slice(&bytes_to_bits(data));

    write_bits_to_alpha(&mut raw, width, height, &all_bits);

    let buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, raw)
        .unwrap_or_else(|| ImageBuffer::new(width, height));
    DynamicImage::ImageRgba8(buffer)
}

/// 画像から LSB ステガノグラフィで埋め込まれたメタデータを読み取り、デコードします。
///
/// # 引数
///
/// * `image` — LSB データを含む画像
///
/// # 戻り値
///
/// デコードされたキー／値マップ。データが見つからないか破損している場合は空のマップ。
pub fn read_lsb_metadata(image: &DynamicImage) -> HashMap<String, String> {
    match read_lsb_raw_bytes(image) {
        Ok(compressed) => {
            let mut decoder = GzDecoder::new(&compressed[..]);
            let mut json_str = String::new();
            if decoder.read_to_string(&mut json_str).is_ok() {
                serde_json::from_str(&json_str).unwrap_or_default()
            } else {
                HashMap::new()
            }
        }
        Err(_) => HashMap::new(),
    }
}

/// 画像から LSB ステガノグラフィで埋め込まれた生バイト列を読み取ります。
pub fn read_lsb_raw_bytes(image: &DynamicImage) -> Result<Vec<u8>, String> {
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let raw = rgba.into_raw();

    // マジックバイト読み取り
    let magic_bits = read_bits_from_alpha(&raw, width, height, MAGIC.len() * 8);
    let magic_bytes = bits_to_bytes(&magic_bits);
    if magic_bytes != MAGIC {
        return Err("Magic number mismatch: not a stealth_pngcomp image".to_string());
    }

    // データ長 (ビット数) 読み取り
    let len_bits = read_bits_from_alpha(&raw, width, height, MAGIC.len() * 8 + 32);
    let len_bytes = bits_to_bytes(&len_bits[MAGIC.len() * 8..]);
    if len_bytes.len() < 4 {
        return Err("Failed to read data length".to_string());
    }
    let bit_len = u32::from_be_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]);
    let byte_len = (bit_len / 8) as usize;

    // データ本体読み取り
    let header_bits = (MAGIC.len() + 4) * 8;
    let data_bits = read_bits_from_alpha(&raw, width, height, header_bits + byte_len * 8);
    let data_bytes = bits_to_bytes(&data_bits[header_bits..]);

    if data_bytes.len() < byte_len {
        return Err("Unexpected end of data".to_string());
    }

    Ok(data_bytes[..byte_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    fn create_test_rgba(w: u32, h: u32) -> DynamicImage {
        let mut img = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, Rgba([128, 128, 128, 255]));
            }
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_bits_roundtrip() {
        let data = b"Hello, LSB!";
        let bits = bytes_to_bits(data);
        let restored = bits_to_bytes(&bits);
        assert_eq!(&restored[..data.len()], data);
    }

    #[test]
    fn test_write_read_lsb_raw_bytes_roundtrip() {
        let img = create_test_rgba(16, 16);
        let data = b"test data";
        let encoded = write_lsb_raw_bytes(&img, data);
        let read_back = read_lsb_raw_bytes(&encoded).unwrap();
        assert_eq!(&read_back, data);
    }

    #[test]
    fn test_write_read_lsb_metadata_roundtrip() {
        let img = create_test_rgba(64, 64);
        let mut metadata = HashMap::new();
        metadata.insert("prompt".to_string(), "a cat".to_string());
        metadata.insert("steps".to_string(), "20".to_string());

        let encoded = write_lsb_metadata(&img, &metadata);
        let decoded = read_lsb_metadata(&encoded);

        assert_eq!(decoded.get("prompt").map(|s| s.as_str()), Some("a cat"));
        assert_eq!(decoded.get("steps").map(|s| s.as_str()), Some("20"));
    }

    #[test]
    fn test_read_lsb_metadata_none() {
        let img = create_test_rgba(64, 64);
        let decoded = read_lsb_metadata(&img);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_read_lsb_raw_bytes_invalid() {
        let img = create_test_rgba(64, 64);
        let result = read_lsb_raw_bytes(&img);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_read_lsb_large_data() {
        let img = create_test_rgba(128, 128);
        let mut metadata = HashMap::new();
        for i in 0..50 {
            metadata.insert(format!("key_{}", i), format!("value_{}", i));
        }
        let encoded = write_lsb_metadata(&img, &metadata);
        let decoded = read_lsb_metadata(&encoded);
        assert_eq!(decoded.len(), metadata.len());
        for (k, v) in &metadata {
            assert_eq!(decoded.get(k), Some(v));
        }
    }
}
