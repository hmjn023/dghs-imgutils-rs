//! エッジ検出 (edge) モジュールの napi-rs バインディング。

use napi_derive::napi;

/// Canny エッジ検出でエッジマスク（f64 の2次元配列）を返します。
#[napi]
pub fn get_edge_by_canny(
    path: String,
    low_threshold: Option<f64>,
    high_threshold: Option<f64>,
) -> napi::Result<Vec<Vec<f64>>> {
    let lt = low_threshold.unwrap_or(100.0) as f32;
    let ht = high_threshold.unwrap_or(200.0) as f32;
    let result = crate::edge::canny::get_edge_by_canny(&path, lt, ht).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Canny edge detection failed: {}", e),
        )
    })?;
    Ok(result
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f64).collect())
        .collect())
}

/// Canny エッジ検出結果を画像として返します（PNG バッファ）。
#[napi]
pub fn edge_image_with_canny(
    path: String,
    low_threshold: Option<f64>,
    high_threshold: Option<f64>,
    backcolor: Option<Vec<i32>>,
    forecolor: Option<Vec<i32>>,
) -> napi::Result<Vec<u8>> {
    let lt = low_threshold.unwrap_or(100.0) as f32;
    let ht = high_threshold.unwrap_or(200.0) as f32;
    let bg = parse_color(backcolor, [255, 255, 255]);
    let fg = forecolor.map(|v| parse_color(Some(v), [0, 0, 0]));
    let result = crate::edge::canny::edge_image_with_canny(&path, lt, ht, bg, fg).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Canny edge image failed: {}", e),
        )
    })?;
    Ok(result)
}

/// Lineart エッジ検出でエッジマスクを返します。
#[napi]
pub fn get_edge_by_lineart(
    path: String,
    coarse: Option<bool>,
    detect_resolution: Option<i32>,
) -> napi::Result<Vec<Vec<f64>>> {
    let c = coarse.unwrap_or(false);
    let res = detect_resolution.unwrap_or(512) as u32;
    let result = crate::edge::lineart::get_edge_by_lineart(&path, c, res).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Lineart edge detection failed: {}", e),
        )
    })?;
    Ok(result
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f64).collect())
        .collect())
}

/// Lineart エッジ検出結果を画像として返します（PNG バッファ）。
#[napi]
pub fn edge_image_with_lineart(
    path: String,
    coarse: Option<bool>,
    detect_resolution: Option<i32>,
    backcolor: Option<Vec<i32>>,
    forecolor: Option<Vec<i32>>,
) -> napi::Result<Vec<u8>> {
    let c = coarse.unwrap_or(false);
    let res = detect_resolution.unwrap_or(512) as u32;
    let bg = parse_color(backcolor, [255, 255, 255]);
    let fg = forecolor.map(|v| parse_color(Some(v), [0, 0, 0]));
    let result =
        crate::edge::lineart::edge_image_with_lineart(&path, c, res, bg, fg).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Lineart edge image failed: {}", e),
            )
        })?;
    Ok(result)
}

/// LineartAnime エッジ検出でエッジマスクを返します。
#[napi]
pub fn get_edge_by_lineart_anime(
    path: String,
    detect_resolution: Option<i32>,
) -> napi::Result<Vec<Vec<f64>>> {
    let res = detect_resolution.unwrap_or(512) as u32;
    let result =
        crate::edge::lineart_anime::get_edge_by_lineart_anime(&path, res).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("LineartAnime edge detection failed: {}", e),
            )
        })?;
    Ok(result
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f64).collect())
        .collect())
}

/// LineartAnime エッジ検出結果を画像として返します（PNG バッファ）。
#[napi]
pub fn edge_image_with_lineart_anime(
    path: String,
    detect_resolution: Option<i32>,
    backcolor: Option<Vec<i32>>,
    forecolor: Option<Vec<i32>>,
) -> napi::Result<Vec<u8>> {
    let res = detect_resolution.unwrap_or(512) as u32;
    let bg = parse_color(backcolor, [255, 255, 255]);
    let fg = forecolor.map(|v| parse_color(Some(v), [0, 0, 0]));
    let result = crate::edge::lineart_anime::edge_image_with_lineart_anime(&path, res, bg, fg)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("LineartAnime edge image failed: {}", e),
            )
        })?;
    Ok(result)
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
