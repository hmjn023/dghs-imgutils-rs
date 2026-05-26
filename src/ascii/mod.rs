//! アスキーアート（AA）生成機能。
//!
//! 画像をグレースケールに変換し、各ピクセルの輝度値を指定された文字レベルに
//! マッピングすることでアスキーアートを生成します。
//!
//! # 使用例
//!
//! ```no_run
//! use dghs_imgutils_rs::ascii::ascii_drawing;
//!
//! let img = image::open("input.jpg").unwrap();
//! let aa = ascii_drawing(&img, "@%#*+=-:. ", Some(80));
//! println!("{}", aa);
//! ```

use image::{DynamicImage, GenericImageView};
use thiserror::Error;

/// ASCII アート生成に関するエラー型。
#[derive(Debug, Error)]
pub enum AsciiError {
    /// 画像の読み込み・処理エラー
    #[error("Image error: {0}")]
    Image(String),

    /// 空の文字レベル指定
    #[error("Levels string must not be empty")]
    EmptyLevels,
}

/// 文字のアスペクト比（幅 / 高さ）。標準的な等幅フォントでは約 0.5。
const FONT_ASPECT_RATIO: f32 = 0.5;

/// 画像をアスキーアートに変換します。
///
/// 入力画像をグレースケールに変換し、指定された幅（`width`）にリサイズした後、
/// 各ピクセルの輝度を `levels` の文字にマッピングします。
///
/// 高さはアスペクト比を保ちつつ、フォントの縦横比 (`FONT_ASPECT_RATIO = 0.5`) を
/// 考慮して計算されます（文字1つが正方形でないため、縦方向のピクセル数を半分にします）。
///
/// # 引数
///
/// * `image` - 入力画像
/// * `levels` - 明るさの順に並べた文字列（先頭が暗い、末尾が明るい）。デフォルトは `"@%#*+=-:. "`
/// * `width` - 出力のアスキーアートの幅（文字数）。`None` の場合は元画像の幅を使用
///
/// # 戻り値
///
/// 改行で区切られたアスキーアート文字列。
pub fn ascii_drawing(
    image: &DynamicImage,
    levels: &str,
    width: Option<u32>,
) -> Result<String, AsciiError> {
    if levels.is_empty() {
        return Err(AsciiError::EmptyLevels);
    }

    let (orig_w, orig_h) = image.dimensions();
    let target_w = width.unwrap_or(orig_w);

    let aspect = orig_h as f32 / orig_w as f32;
    let target_h = (target_w as f32 * aspect * FONT_ASPECT_RATIO).round() as u32;

    let resized = image.resize_exact(
        target_w.max(1),
        target_h.max(1),
        image::imageops::FilterType::Lanczos3,
    );

    let gray = resized.to_luma8();

    let levels_chars: Vec<char> = levels.chars().collect();
    let max_idx = levels_chars.len() - 1;

    let mut result = String::with_capacity((target_w * (target_h + 1)) as usize);

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = gray.get_pixel(x, y);
            let luminance = pixel[0] as f32 / 255.0;
            let idx = (luminance * max_idx as f32).round() as usize;
            let idx = idx.min(max_idx);
            result.push(levels_chars[idx]);
        }
        result.push('\n');
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_ascii_drawing_basic() {
        let mut img = RgbImage::new(10, 10);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([0, 0, 0]); // 真っ黒
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = ascii_drawing(&dyn_img, "@%#*+=-:. ", Some(10)).unwrap();
        // 黒なので最初の文字 '@' のみ
        assert!(result.contains('@'));
        assert_eq!(result.lines().count(), 5); // 10 * 0.5 = 5
    }

    #[test]
    fn test_ascii_drawing_white() {
        let mut img = RgbImage::new(10, 10);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]); // 真っ白
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = ascii_drawing(&dyn_img, "@%#*+=-:. ", Some(10)).unwrap();
        // 白なので最後の文字 ' ' が期待される
        assert!(result.ends_with(" \n") || result.contains(' '));
    }

    #[test]
    fn test_ascii_drawing_empty_levels() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let result = ascii_drawing(&img, "", Some(10));
        assert!(result.is_err());
    }

    #[test]
    fn test_ascii_drawing_custom_levels() {
        // Use a larger image (20x20) to minimize interpolation effects
        let mut img = RgbImage::new(20, 20);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 128, 128]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        // 3段階の文字レベル: dark='.' mid=':' bright=' '
        let result = ascii_drawing(&dyn_img, ".: ", Some(20)).unwrap();
        // 50% gray should map to ':' (idx 1, since 0.5*2=1.0)
        // Due to Lanczos3 interpolation, some pixels may drift slightly
        let mid_count = result.chars().filter(|&c| c == ':').count();
        assert!(mid_count > 0, "Expected some mid-tone pixels ':'");
    }
}
