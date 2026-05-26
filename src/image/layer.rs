use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use ndarray::Array2;

use super::error::ImageError;

pub enum AlphaTyping {
    Global(f32),
    PerPixel(Array2<f32>),
}

pub struct LayerItem {
    pub image: LayerSource,
    pub alpha: AlphaTyping,
}

pub enum LayerSource {
    Image(DynamicImage),
    ColorRgb([u8; 3]),
    ColorRgba([u8; 4]),
}

pub struct LayerItemBuilder {
    pub image: DynamicImage,
    pub alpha: f32,
}

impl LayerItemBuilder {
    pub fn from_image(img: DynamicImage) -> Self {
        Self {
            image: img,
            alpha: 1.0,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }
}

fn apply_alpha_to_image(img: &DynamicImage, alpha: f32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    let mut result = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = (p[3] as f32 * alpha).clamp(0.0, 255.0).round() as u8;
            result.put_pixel(x, y, Rgba([p[0], p[1], p[2], a]));
        }
    }
    result
}

pub fn istack(
    layers: Vec<LayerItemBuilder>,
    size: Option<(u32, u32)>,
) -> Result<RgbaImage, ImageError> {
    let (width, height) = if let Some(s) = size {
        s
    } else {
        let mut found = None;
        for layer in &layers {
            if !matches!(
                layer.image.color(),
                image::ColorType::L8 | image::ColorType::La8
            ) {
                let dims = layer.image.dimensions();
                if dims.0 > 0 && dims.1 > 0 {
                    found = Some(dims);
                    break;
                }
            }
        }
        found.ok_or_else(|| {
            ImageError::InvalidArgument(
                "Unable to determine image size; provide at least one valid image".into(),
            )
        })?
    };

    let mut canvas = RgbaImage::new(width, height);

    for layer in &layers {
        let rgba_img = {
            let source_img = &layer.image;
            let (iw, ih) = source_img.dimensions();
            if iw == width && ih == height {
                source_img.to_rgba8()
            } else {
                let resized =
                    source_img.resize_exact(width, height, image::imageops::FilterType::Triangle);
                resized.to_rgba8()
            }
        };

        let composited = apply_alpha_to_image(&DynamicImage::ImageRgba8(rgba_img), layer.alpha);

        for y in 0..height {
            for x in 0..width {
                let src = composited.get_pixel(x, y);
                let dst = canvas.get_pixel_mut(x, y);
                let sa = src[3] as f32 / 255.0;
                let da = dst[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                if out_a > 0.0 {
                    dst[0] = ((src[0] as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) / out_a)
                        .round() as u8;
                    dst[1] = ((src[1] as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) / out_a)
                        .round() as u8;
                    dst[2] = ((src[2] as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) / out_a)
                        .round() as u8;
                    dst[3] = (out_a * 255.0).round() as u8;
                }
            }
        }
    }

    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_istack_single_image() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0])));
        let layers = vec![LayerItemBuilder::from_image(img)];
        let result = istack(layers, Some((10, 10))).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
        let p = result.get_pixel(0, 0);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
    }

    #[test]
    fn test_istack_two_images() {
        let red = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0])));
        let blue = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, image::Rgb([0, 0, 255])));
        let layers = vec![
            LayerItemBuilder::from_image(red).with_alpha(0.5),
            LayerItemBuilder::from_image(blue).with_alpha(0.5),
        ];
        let result = istack(layers, Some((10, 10))).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_istack_with_alpha() {
        let red = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0])));
        let layers = vec![LayerItemBuilder::from_image(red).with_alpha(0.0)];
        let result = istack(layers, Some((10, 10))).unwrap();
        let p = result.get_pixel(0, 0);
        assert_eq!(p[3], 0);
    }

    #[test]
    fn test_istack_no_size() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(20, 30, image::Rgb([0, 255, 0])));
        let layers = vec![LayerItemBuilder::from_image(img)];
        let result = istack(layers, None).unwrap();
        assert_eq!(result.dimensions(), (20, 30));
    }
}
