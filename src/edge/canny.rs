//! Canny エッジ検出（モデル不要）。
//!
//! Python 版では `cv2.Canny` を使用しますが、Rust では `imageproc::edges::canny` で等価実装します。

use crate::image::force_image_background;
use image::imageops::colorops::grayscale;
use image::{DynamicImage, GrayImage, ImageBuffer, Rgb};
use imageproc::edges::canny;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EdgeError {
    #[error("Failed to open image: {0}")]
    ImageOpen(#[from] image::ImageError),
    #[error("Edge detection failed: {0}")]
    Processing(String),
}

/// Canny エッジ検出を実行し、エッジマスク（float32 [H, W]）を返します。
///
/// * `path` - 入力画像のパス
/// * `low_threshold` - Canny 下限しきい値（デフォルト 100.0）
/// * `high_threshold` - Canny 上限しきい値（デフォルト 200.0）
pub fn get_edge_by_canny(
    path: &str,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<Vec<f32>>, EdgeError> {
    let img = image::open(path)?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let gray: GrayImage = grayscale(&rgb.to_rgb8());

    // imageproc::edges::canny は u8 グレースケール画像を受け取り、u8 エッジ画像を返します
    let edges: GrayImage = canny(&gray, low_threshold, high_threshold);

    let (w, h) = edges.dimensions();
    let mut result = vec![vec![0.0f32; w as usize]; h as usize];
    for y in 0..h {
        for x in 0..w {
            result[y as usize][x as usize] = edges.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }
    Ok(result)
}

/// Canny エッジ画像を生成し、PNG バイト列として返します。
///
/// * `path` - 入力画像のパス
/// * `low_threshold` - Canny 下限しきい値（デフォルト 100.0）
/// * `high_threshold` - Canny 上限しきい値（デフォルト 200.0）
/// * `backcolor` - 背景色 RGB（エッジ以外の領域の色）
/// * `forecolor` - 線の色 RGB（`None` の場合は元画像の色を使用）
pub fn edge_image_with_canny(
    path: &str,
    low_threshold: f32,
    high_threshold: f32,
    backcolor: [u8; 3],
    forecolor: Option<[u8; 3]>,
) -> Result<Vec<u8>, EdgeError> {
    let img = image::open(path)?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let rgb8 = rgb.to_rgb8();
    let gray: GrayImage = grayscale(&rgb8);
    let edges: GrayImage = canny(&gray, low_threshold, high_threshold);

    let (w, h) = edges.dimensions();
    let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(w, h, Rgb(backcolor));

    let fore = forecolor;
    for y in 0..h {
        for x in 0..w {
            let edge_val = edges.get_pixel(x, y)[0] as f32 / 255.0;
            if edge_val > 0.0 {
                let fore_pix = if let Some(fc) = fore {
                    Rgb(fc)
                } else {
                    *rgb8.get_pixel(x, y)
                };
                // エッジ強度で線形補間
                let bg = canvas.get_pixel(x, y).0;
                let blended = [
                    (bg[0] as f32 * (1.0 - edge_val) + fore_pix[0] as f32 * edge_val).round() as u8,
                    (bg[1] as f32 * (1.0 - edge_val) + fore_pix[1] as f32 * edge_val).round() as u8,
                    (bg[2] as f32 * (1.0 - edge_val) + fore_pix[2] as f32 * edge_val).round() as u8,
                ];
                canvas.put_pixel(x, y, Rgb(blended));
            }
        }
    }

    let mut buf = Vec::new();
    DynamicImage::ImageRgb8(canvas)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    Ok(buf)
}
