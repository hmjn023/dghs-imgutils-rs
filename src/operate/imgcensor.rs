//! 画像ベース検閲（スタンプ・絵文字オーバーレイ）。
//!
//! Python 版の `ImageBasedCensor` の単純化版を移植します。
//! 絵文字画像を対象領域にスケーリングしてアルファブレンドで合成します。

use image::{DynamicImage, RgbaImage, imageops};

use crate::detect::base::BBox;

/// 検閲画像のフィット方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CensorImageFit {
    /// 対象領域内に収める（アスペクト比維持、はみ出さない）
    Contain,
    /// 対象領域全体を覆う（アスペクト比維持、領域内に収まるよう拡大縮小）
    Cover,
}

/// 指定された領域に検閲画像をアルファブレンドでオーバーレイします。
///
/// # 引数
/// * `target` - 対象画像（可変参照、RGBA）
/// * `area` - 検閲領域 `[x1, y1, x2, y2]`
/// * `censor_image` - 検閲に使う画像（RGBA 推奨、アルファチャンネルで合成）
/// * `fit` - フィット方法
pub fn censor_area_image(
    target: &mut RgbaImage,
    area: &[f32; 4],
    censor_image: &DynamicImage,
    fit: CensorImageFit,
) {
    let (iw, ih) = target.dimensions();
    let x1 = (area[0].max(0.0) as u32).min(iw);
    let y1 = (area[1].max(0.0) as u32).min(ih);
    let x2 = (area[2].max(0.0) as u32).min(iw);
    let y2 = (area[3].max(0.0) as u32).min(ih);

    if x1 >= x2 || y1 >= y2 {
        return;
    }

    let area_w = x2 - x1;
    let area_h = y2 - y1;
    let (cw, ch) = (censor_image.width(), censor_image.height());

    if cw == 0 || ch == 0 {
        return;
    }

    let scale = match fit {
        CensorImageFit::Contain => (area_w as f32 / cw as f32).min(area_h as f32 / ch as f32),
        CensorImageFit::Cover => (area_w as f32 / cw as f32).max(area_h as f32 / ch as f32),
    };

    let new_w = ((cw as f32 * scale).round() as u32).max(1);
    let new_h = ((ch as f32 * scale).round() as u32).max(1);

    let resized = censor_image.resize_exact(new_w, new_h, imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();

    let offset_x = x1 + (area_w.saturating_sub(new_w)) / 2;
    let offset_y = y1 + (area_h.saturating_sub(new_h)) / 2;

    for y in 0..new_h {
        for x in 0..new_w {
            let px = offset_x + x;
            let py = offset_y + y;
            if px >= iw || py >= ih {
                continue;
            }
            let src = rgba.get_pixel(x, y);
            let dst = target.get_pixel_mut(px, py);
            let a = src[3] as f32 / 255.0;
            if a >= 1.0 {
                *dst = *src;
            } else if a > 0.0 {
                let inv_a = 1.0 - a;
                dst[0] = (src[0] as f32 * a + dst[0] as f32 * inv_a).round() as u8;
                dst[1] = (src[1] as f32 * a + dst[1] as f32 * inv_a).round() as u8;
                dst[2] = (src[2] as f32 * a + dst[2] as f32 * inv_a).round() as u8;
                dst[3] = dst[3].max(src[3]);
            }
        }
    }
}

/// 複数の BBox 領域に検閲画像をオーバーレイします。
pub fn censor_areas_image(
    target: &mut RgbaImage,
    areas: &[BBox],
    censor_image: &DynamicImage,
    fit: CensorImageFit,
) {
    for area in areas {
        let area_f32 = [area.0 as f32, area.1 as f32, area.2 as f32, area.3 as f32];
        censor_area_image(target, &area_f32, censor_image, fit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn make_solid_rgba(w: u32, h: u32, color: Rgba<u8>) -> DynamicImage {
        let mut buf = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                buf.put_pixel(x, y, color);
            }
        }
        DynamicImage::ImageRgba8(buf)
    }

    #[test]
    fn test_censor_area_image_contain() {
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let stamp = make_solid_rgba(20, 20, Rgba([255, 0, 0, 255]));
        let area = [10.0, 10.0, 50.0, 50.0];
        censor_area_image(&mut img, &area, &stamp, CensorImageFit::Contain);
        // area: 40x40, stamp: 20x20, scale = 2.0, new_w=40, new_h=40
        // offset = (10, 10), stamp covers (10..50, 10..50)
        // Interior pixels should be red
        assert_eq!(img.get_pixel(12, 12), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(48, 48), &Rgba([255, 0, 0, 255]));
        // Outside area should remain white
        assert_eq!(img.get_pixel(5, 5), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_censor_area_image_cover() {
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let stamp = make_solid_rgba(10, 10, Rgba([0, 255, 0, 255]));
        let area = [0.0, 0.0, 40.0, 20.0];
        censor_area_image(&mut img, &area, &stamp, CensorImageFit::Cover);
        // Cover: scale_x = 4, scale_y = 2 -> max = 4
        // new_w = 40, new_h = 40
        // offset = (0 + (40-40)/2, 0 + (20-40)/2) = (0, -10) -> clamp
        // stamp covers x 0..40, y 0..20 (clipped)
        assert_eq!(img.get_pixel(0, 0), &Rgba([0, 255, 0, 255]));
        assert_eq!(img.get_pixel(39, 19), &Rgba([0, 255, 0, 255]));
    }

    #[test]
    fn test_censor_areas_image() {
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        let stamp = make_solid_rgba(5, 5, Rgba([0, 0, 255, 255]));
        let areas = [(0, 0, 10, 10), (50, 50, 60, 60)];
        censor_areas_image(&mut img, &areas, &stamp, CensorImageFit::Contain);
        assert_eq!(img.get_pixel(3, 3), &Rgba([0, 0, 255, 255]));
        assert_eq!(img.get_pixel(53, 53), &Rgba([0, 0, 255, 255]));
        // middle pixel should be white
        assert_eq!(img.get_pixel(30, 30), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn test_censor_area_alpha_blend() {
        let mut img = RgbaImage::from_pixel(50, 50, Rgba([255, 0, 0, 255]));
        let stamp = make_solid_rgba(10, 10, Rgba([0, 255, 0, 128]));
        let area = [0.0, 0.0, 10.0, 10.0];
        censor_area_image(&mut img, &area, &stamp, CensorImageFit::Contain);
        // alpha=128/255≈0.502, inv=0.498
        // R = 0*0.502 + 255*0.498 = 126.99 ≈ 127
        // G = 255*0.502 + 0*0.498 = 128.01 ≈ 128
        let px = img.get_pixel(5, 5);
        assert_eq!(px[0], 127);
        assert_eq!(px[1], 128);
        assert_eq!(px[2], 0);
    }
}
