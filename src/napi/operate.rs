//! 画像操作 (operate) モジュールの napi-rs バインディング。
//!
//! - `censor_nsfw_image`: NSFW 検出 → 画像（絵文字/スタンプ）検閲
//! - `censor_area_image`: 指定領域に画像検閲
//! - `censor_areas_image`: 複数 BBox 領域に画像検閲

use crate::detect::base::BBox;
use crate::operate::censor::CensorMethod;
use crate::operate::censor::censor_nsfw_image as core_censor_nsfw_image;
use crate::operate::imgcensor::{CensorImageFit, censor_areas_image as core_censor_areas_image};
use napi_derive::napi;

fn parse_fit(s: Option<String>) -> CensorImageFit {
    match s.as_deref() {
        Some("contain") => CensorImageFit::Contain,
        _ => CensorImageFit::Cover,
    }
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct NapiBBoxArea {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// 画像パスと絵文字/スタンプ画像バッファを受け取り、NSFW 検出 + 画像検閲を実行します。
///
/// * `path` - 元画像のファイルパス
/// * `censor_image_buffer` - 検閲用絵文字/スタンプ画像の PNG/JPEG バッファ（省略時はデフォルトの赤い塗りつぶし）
/// * `fit` - フィット方法: `"contain"` または `"cover"`（デフォルト `"cover"`）
/// * `nipple_f` - 女性の乳首を検閲するか（デフォルト `false`）
/// * `penis` - ペニスを検閲するか（デフォルト `true`）
/// * `pussy` - プッシーを検閲するか（デフォルト `true`）
///
/// 戻り値は PNG エンコードされた画像バッファです。
#[napi]
pub fn censor_nsfw_image(
    path: String,
    censor_image_buffer: Option<Vec<u8>>,
    fit: Option<String>,
    nipple_f: Option<bool>,
    penis: Option<bool>,
    pussy: Option<bool>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let censor_img = match censor_image_buffer {
        Some(buf) => image::load_from_memory(&buf).map_err(|e| {
            napi::Error::new(
                napi::Status::InvalidArg,
                format!("Failed to decode censor image buffer: {}", e),
            )
        })?,
        None => {
            // Default: solid red stamp (256x256)
            let mut img = image::RgbaImage::new(256, 256);
            for y in 0..256 {
                for x in 0..256 {
                    img.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
                }
            }
            image::DynamicImage::ImageRgba8(img)
        }
    };

    let censor_fit = parse_fit(fit);
    let method = CensorMethod::Image {
        image: censor_img,
        fit: censor_fit,
    };

    let result = core_censor_nsfw_image(
        &image,
        &method,
        nipple_f.unwrap_or(false),
        penis.unwrap_or(true),
        pussy.unwrap_or(true),
    )
    .map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NSFW image censor failed: {}", e),
        )
    })?;

    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode output image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}

/// 画像パスと BBox 領域リストを受け取り、指定領域を画像（絵文字/スタンプ）で検閲します。
///
/// * `path` - 元画像のファイルパス
/// * `areas` - 検閲領域のリスト（各要素は `{ x1, y1, x2, y2 }`）
/// * `censor_image_buffer` - 検閲用画像の PNG/JPEG バッファ
/// * `fit` - フィット方法: `"contain"` または `"cover"`（デフォルト `"cover"`）
///
/// 戻り値は PNG エンコードされた画像バッファです。
#[napi]
pub fn censor_areas_with_image(
    path: String,
    areas: Vec<NapiBBoxArea>,
    censor_image_buffer: Vec<u8>,
    fit: Option<String>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let censor_img = image::load_from_memory(&censor_image_buffer).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to decode censor image buffer: {}", e),
        )
    })?;

    let censor_fit = parse_fit(fit);

    let bboxes: Vec<BBox> = areas
        .into_iter()
        .map(|a| {
            (
                a.x1.max(0) as u32,
                a.y1.max(0) as u32,
                a.x2.max(0) as u32,
                a.y2.max(0) as u32,
            )
        })
        .collect();

    let mut rgba = image.to_rgba8();
    core_censor_areas_image(&mut rgba, &bboxes, &censor_img, censor_fit);

    let result = image::DynamicImage::ImageRgba8(rgba);
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode output image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}
