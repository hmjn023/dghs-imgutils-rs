//! 画像I/O (image) モジュールの napi-rs バインディング。

use napi_derive::napi;

/// 画像を指定サイズにパディングします。
///
/// * `path`: 画像ファイルのパス
/// * `target_w`: 目標幅
/// * `target_h`: 目標高さ
/// * `bg_color`: 背景色 [r, g, b]（省略時は白）
/// 戻り値は PNG エンコードされた画像バッファです。
#[napi]
pub fn pad_image_to_size(
    path: String,
    target_w: i32,
    target_h: i32,
    bg_color: Option<Vec<i32>>,
) -> napi::Result<Vec<u8>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let bg = parse_color(bg_color, [255, 255, 255]);
    let result =
        crate::image::pad_image_to_size(&img, target_w.max(1) as u32, target_h.max(1) as u32, bg);
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}

/// RGBA 画像の透過部分を指定色で塗りつぶします。
///
/// * `path`: 画像ファイルのパス
/// * `bg_color`: 背景色 [r, g, b]（省略時は白）
/// 戻り値は PNG エンコードされた画像バッファです。
#[napi]
pub fn force_image_background(path: String, bg_color: Option<Vec<i32>>) -> napi::Result<Vec<u8>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let bg = parse_color(bg_color, [255, 255, 255]);
    let result = crate::image::force_image_background(&img, bg);
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}

/// 画像を CHW テンソル（Array4）に変換します。
///
/// * `path`: 画像ファイルのパス
/// * `mean`: 正規化の平均値 [r, g, b]（省略時は [0.0, 0.0, 0.0]）
/// * `std`: 正規化の標準偏差 [r, g, b]（省略時は [1.0, 1.0, 1.0]）
/// 戻り値はフラットな f64 配列です。
#[napi]
pub fn to_ndarray_chw(
    path: String,
    mean: Option<Vec<f64>>,
    std: Option<Vec<f64>>,
) -> napi::Result<Vec<f64>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {}", e),
        )
    })?;
    let m = parse_f32_array(mean, [0.0, 0.0, 0.0]);
    let s = parse_f32_array(std, [1.0, 1.0, 1.0]);
    let arr = crate::image::to_ndarray_chw(&img, &m, &s).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("to_ndarray_chw failed: {}", e),
        )
    })?;
    Ok(arr.iter().map(|&v| v as f64).collect())
}

fn parse_color(v: Option<Vec<i32>>, default: [u8; 3]) -> [u8; 3] {
    match v {
        Some(c) if c.len() == 3 => [
            c[0].clamp(0, 255) as u8,
            c[1].clamp(0, 255) as u8,
            c[2].clamp(0, 255) as u8,
        ],
        _ => default,
    }
}

fn parse_f32_array(v: Option<Vec<f64>>, default: [f32; 3]) -> [f32; 3] {
    match v {
        Some(c) if c.len() == 3 => [c[0] as f32, c[1] as f32, c[2] as f32],
        _ => default,
    }
}
