//! 自動検閲（モザイク・ぼかし・塗りつぶし）。
//!
//! Python 版の `PixelateCensor`, `BlurCensor`, `ColorCensor` に対応します。
//! `censor_nsfw` は detect モジュールと連携します。

use crate::detect::base::Detection;
use crate::detect::censor::detect_censors as core_detect_censors;
use crate::inference::InferenceError;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use imageproc::filter::gaussian_blur_f32;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CensorError {
    #[error("Failed to open image: {0}")]
    ImageOpen(#[from] image::ImageError),
    #[error("Detection failed: {0}")]
    Detection(#[from] InferenceError),
    #[error("Processing error: {0}")]
    Processing(String),
}

/// 検閲方法
#[derive(Debug, Clone)]
pub enum CensorMethod {
    /// ピクセレート（モザイク）。`radius` はダウンサンプル係数（デフォルト 4）
    Pixelate { radius: u32 },
    /// ガウシアンぼかし。`radius` はぼかし半径（デフォルト 4）
    Blur { radius: f32 },
    /// 単色塗りつぶし。`color` は RGB
    Color { color: [u8; 3] },
}

impl Default for CensorMethod {
    fn default() -> Self {
        CensorMethod::Pixelate { radius: 4 }
    }
}

/// 画像の特定 BBox 領域に検閲を適用します。
///
/// # 引数
/// * `img` - 入力画像（可変）
/// * `detection` - 検出結果（bbox は `(x1, y1, x2, y2)` の tuple）
/// * `method` - 検閲方法
pub fn censor_area(
    img: &DynamicImage,
    detection: &Detection,
    method: &CensorMethod,
) -> DynamicImage {
    let (iw, ih) = img.dimensions();
    let (bx1, by1, bx2, by2) = detection.bbox;
    let x0 = bx1.min(iw);
    let y0 = by1.min(ih);
    let x1 = bx2.min(iw);
    let y1 = by2.min(ih);
    if x0 >= x1 || y0 >= y1 {
        return img.clone();
    }
    let bw = x1 - x0;
    let bh = y1 - y0;

    let region = img.crop_imm(x0, y0, bw, bh);
    let censored = match method {
        CensorMethod::Pixelate { radius } => {
            let r = (*radius).max(1);
            let small_w = (bw / r).max(1);
            let small_h = (bh / r).max(1);
            let small = region.resize_exact(small_w, small_h, FilterType::Nearest);
            small.resize_exact(bw, bh, FilterType::Nearest)
        }
        CensorMethod::Blur { radius } => {
            let rgb8 = region.to_rgb8();
            let blurred = gaussian_blur_f32(&rgb8, *radius);
            DynamicImage::ImageRgb8(blurred)
        }
        CensorMethod::Color { color } => {
            let canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(bw, bh, Rgb(*color));
            DynamicImage::ImageRgb8(canvas)
        }
    };

    // 元画像にペースト
    let mut result = img.clone().into_rgb8();
    let censored_rgb = censored.to_rgb8();
    for y in 0..bh {
        for x in 0..bw {
            result.put_pixel(x0 + x, y0 + y, *censored_rgb.get_pixel(x, y));
        }
    }
    DynamicImage::ImageRgb8(result)
}

/// 複数の Detection 結果に検閲を適用します。
pub fn censor_areas(
    path: &str,
    detections: &[Detection],
    method: &CensorMethod,
) -> Result<DynamicImage, CensorError> {
    let img = image::open(path).map_err(CensorError::ImageOpen)?;
    let mut result = img;
    for det in detections {
        result = censor_area(&result, det, method);
    }
    Ok(result)
}

/// 検出された NSFW 部位（censors）に自動で検閲を適用します。
///
/// # 引数
/// * `path` - 入力画像のパス
/// * `method` - 検閲方法
/// * `nipple_f` - 女性の乳首を検閲するか（デフォルト false）
/// * `penis` - ペニスを検閲するか（デフォルト true）
/// * `pussy` - プッシーを検閲するか（デフォルト true）
pub fn censor_nsfw(
    path: &str,
    method: &CensorMethod,
    nipple_f: bool,
    penis: bool,
    pussy: bool,
) -> Result<DynamicImage, CensorError> {
    let img = image::open(path).map_err(CensorError::ImageOpen)?;
    let detections = core_detect_censors(&img, "s", "v1.0", 0.3, 0.5)?;

    let target_labels: Vec<&str> = {
        let mut labels = Vec::new();
        if nipple_f {
            labels.push("nipple_f");
        }
        if penis {
            labels.push("penis");
        }
        if pussy {
            labels.push("pussy");
        }
        labels
    };

    let filtered: Vec<Detection> = detections
        .into_iter()
        .filter(|d| target_labels.contains(&d.label.as_str()))
        .collect();

    let mut result = img;
    for det in &filtered {
        result = censor_area(&result, det, method);
    }
    Ok(result)
}
