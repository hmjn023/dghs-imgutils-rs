//! 画像操作 (operate) モジュールの napi-rs バインディング。

use crate::detect::base::BBox;
use crate::operate::align::align_maxsize as core_align_maxsize;
use crate::operate::censor::CensorMethod;
use crate::operate::censor::censor_nsfw as core_censor_nsfw;
use crate::operate::censor::censor_nsfw_image as core_censor_nsfw_image;
use crate::operate::imgcensor::{CensorImageFit, censor_areas_image as core_censor_areas_image};
use crate::operate::squeeze::{
    squeeze as core_squeeze, squeeze_with_transparency as core_squeeze_with_transparency,
};
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

/// 画像の長辺を max_size に合わせてリサイズします。
#[napi]
pub fn align_maxsize(path: String, max_size: i32) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let result = core_align_maxsize(&image, max_size.max(1) as u32);
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

/// マスクに基づいて画像をトリミング（squeeze）します。
#[napi]
pub fn squeeze(path: String, mask: Vec<Vec<bool>>) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let result = core_squeeze(&image, &mask).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Squeeze failed: {}", e),
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

/// 透過領域に基づいて画像を自動トリミングします。
#[napi]
pub fn squeeze_with_transparency(
    path: String,
    threshold: Option<f64>,
    median_size: Option<i32>,
) -> napi::Result<Vec<u8>> {
    let th = threshold.unwrap_or(0.7) as f32;
    let ms = median_size.unwrap_or(5).max(1) as usize;
    let result = core_squeeze_with_transparency(&path, th, ms).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Squeeze with transparency failed: {}", e),
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

/// NSFW 検出 + 基本検閲（モザイク/ぼかし/色塗り）を実行します。
#[napi]
pub fn censor_nsfw_basic(
    path: String,
    method: Option<String>,
    radius: Option<f64>,
    nipple_f: Option<bool>,
    penis: Option<bool>,
    pussy: Option<bool>,
) -> napi::Result<Vec<u8>> {
    let r = radius.unwrap_or(4.0) as f32;
    let censor_method = match method.as_deref() {
        Some("blur") => CensorMethod::Blur { radius: r },
        Some("color") => CensorMethod::Color { color: [255, 0, 0] },
        _ => CensorMethod::Pixelate { radius: r as u32 },
    };
    let result = core_censor_nsfw(
        &path,
        &censor_method,
        nipple_f.unwrap_or(false),
        penis.unwrap_or(true),
        pussy.unwrap_or(true),
    )
    .map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NSFW censor failed: {}", e),
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
