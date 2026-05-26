//! 領域分割 (segment) モジュールの napi-rs バインディング。

use crate::segment::isnetis::get_isnetis_mask as core_get_isnetis_mask;
use crate::segment::isnetis::segment_rgba_with_isnetis as core_segment_rgba_with_isnetis;
use crate::segment::isnetis::segment_with_isnetis as core_segment_with_isnetis;
use napi_derive::napi;

/// ISNetIS 前景セグメンテーション結果のマスク値を 2次元配列（f64）で返します。
#[napi]
pub fn get_isnetis_mask(path: String, scale: Option<i32>) -> napi::Result<Vec<Vec<f64>>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let s = scale.unwrap_or(1024) as u32;

    let mask = core_get_isnetis_mask(&image, s).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("ISNetIS mask extraction failed: {}", e),
        )
    })?;

    let mut result = Vec::with_capacity(mask.nrows());
    for r in 0..mask.nrows() {
        let row_vec = mask.row(r).iter().map(|&v| v as f64).collect();
        result.push(row_vec);
    }

    Ok(result)
}

/// セグメンテーション領域を切り抜き、指定された背景色とブレンドした画像を生成して PNG バッファで返します。
///
/// * `path`: 元画像のファイルパス
/// * `bg_color`: [r, g, b] 形式の背景色配列（例: `[0, 255, 0]`）
/// * `scale`: 推論時のサイズ（オプション）
#[napi]
pub fn segment_with_isnetis(
    path: String,
    bg_color: Vec<i32>,
    scale: Option<i32>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    if bg_color.len() != 3 {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            "bg_color must be an array of exactly 3 integers [r, g, b]".to_string(),
        ));
    }

    let bg_color_u8 = [
        bg_color[0].clamp(0, 255) as u8,
        bg_color[1].clamp(0, 255) as u8,
        bg_color[2].clamp(0, 255) as u8,
    ];

    let s = scale.unwrap_or(1024) as u32;

    let (_, segmented_img) = core_segment_with_isnetis(&image, bg_color_u8, s).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("ISNetIS color segmentation failed: {}", e),
        )
    })?;

    let mut buf = std::io::Cursor::new(Vec::new());
    segmented_img
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode segmented image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}

/// セグメンテーション領域を切り抜き、背景を透過させた RGBA 画像を生成して PNG バッファで返します。
///
/// * `path`: 元画像のファイルパス
/// * `scale`: 推論時のサイズ（オプション）
#[napi]
pub fn segment_rgba_with_isnetis(path: String, scale: Option<i32>) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let s = scale.unwrap_or(1024) as u32;

    let (_, segmented_img) = core_segment_rgba_with_isnetis(&image, s).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("ISNetIS RGBA segmentation failed: {}", e),
        )
    })?;

    let mut buf = std::io::Cursor::new(Vec::new());
    segmented_img
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode RGBA segmented image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}
