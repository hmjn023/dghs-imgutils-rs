use crate::ocr::detect::{BBox, detect_text_with_paddleocr as core_detect_text};
use crate::ocr::recognize::ocr as core_ocr;
use napi_derive::napi;

#[napi(object)]
#[derive(Debug, Clone)]
pub struct NapiBBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct NapiOcrResult {
    pub bbox: NapiBBox,
    pub text: String,
    pub score: f64,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct NapiBBoxResult {
    pub bbox: NapiBBox,
    pub score: f64,
}

impl From<(BBox, f32)> for NapiBBoxResult {
    fn from((bbox, score): (BBox, f32)) -> Self {
        Self {
            bbox: NapiBBox {
                x1: bbox.0,
                y1: bbox.1,
                x2: bbox.2,
                y2: bbox.3,
            },
            score: score as f64,
        }
    }
}

impl From<(BBox, String, f32)> for NapiOcrResult {
    fn from((bbox, text, score): (BBox, String, f32)) -> Self {
        Self {
            bbox: NapiBBox {
                x1: bbox.0,
                y1: bbox.1,
                x2: bbox.2,
                y2: bbox.3,
            },
            text,
            score: score as f64,
        }
    }
}

#[napi]
pub fn ocr(path: String) -> napi::Result<Vec<NapiOcrResult>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let results = core_ocr(&image).map_err(|e| {
        napi::Error::new(napi::Status::GenericFailure, format!("OCR failed: {}", e))
    })?;

    Ok(results.into_iter().map(NapiOcrResult::from).collect())
}

#[napi]
pub fn detect_text(path: String) -> napi::Result<Vec<NapiBBoxResult>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let results = core_detect_text(&image).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Text detection failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiBBoxResult::from).collect())
}
