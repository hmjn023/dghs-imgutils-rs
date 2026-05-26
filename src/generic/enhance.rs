use image::{DynamicImage, ImageBuffer, Rgba};
use ndarray::{Array2, Array3};

use crate::generic::GenericError;

/// Abstract base class for tile-based image enhancement models.
///
/// Processes RGB and RGBA images by running the RGB channels through `process_rgb`
/// and the alpha channel (if present) through a separate model pass.
pub trait ImageEnhancer {
    /// Process the RGB channels of an image.
    /// Input shape: [3, H, W] float32, values in [0, 1].
    /// Output shape: [3, H, W] float32, values in [0, 1].
    fn process_rgb(&self, rgb_array: &Array3<f32>) -> Result<Array3<f32>, GenericError>;

    /// Process the alpha channel using the same model (stack alpha 3 times as RGB, then average).
    fn process_alpha(&self, alpha_array: &Array2<f32>) -> Result<Array2<f32>, GenericError> {
        assert_eq!(
            alpha_array.ndim(),
            2,
            "Alpha array should be 2-dim, got {:?}",
            alpha_array.shape()
        );
        let h = alpha_array.shape()[0];
        let w = alpha_array.shape()[1];
        // Stack alpha as 3-channel RGB
        let mut stacked = Array3::<f32>::zeros((3, h, w));
        for c in 0..3 {
            stacked
                .slice_mut(ndarray::s![c, .., ..])
                .assign(alpha_array);
        }
        let enhanced = self.process_rgb(&stacked)?;
        // Average across channels
        let mut result = Array2::<f32>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mean = (enhanced[[0, y, x]] + enhanced[[1, y, x]] + enhanced[[2, y, x]]) / 3.0;
                result[[y, x]] = mean;
            }
        }
        Ok(result)
    }

    /// Process an RGBA image (4 channels).
    fn process_rgba(&self, rgba_array: &Array3<f32>) -> Result<Array3<f32>, GenericError> {
        assert_eq!(
            rgba_array.shape()[0],
            4,
            "RGBA array should have 4 channels, got shape {:?}",
            rgba_array.shape()
        );
        let h = rgba_array.shape()[1];
        let w = rgba_array.shape()[2];

        let rgb = rgba_array.slice(ndarray::s![0..3, .., ..]).to_owned();
        let alpha = rgba_array.slice(ndarray::s![3, .., ..]).to_owned();

        let enhanced_rgb = self.process_rgb(&rgb)?;
        let enhanced_alpha = self.process_alpha(&alpha)?;

        let mut result = Array3::<f32>::zeros((4, h, w));
        result
            .slice_mut(ndarray::s![0..3, .., ..])
            .assign(&enhanced_rgb);
        result
            .slice_mut(ndarray::s![3, .., ..])
            .assign(&enhanced_alpha);
        Ok(result)
    }

    /// Enhance an image. Handles both RGB and RGBA images.
    fn enhance(&self, image: &DynamicImage) -> Result<DynamicImage, GenericError> {
        let has_alpha = image.color().has_alpha();

        let img_rgba = if has_alpha {
            image.to_rgba8()
        } else {
            let rgb = image.to_rgb8();
            let mut rgba = ImageBuffer::new(rgb.width(), rgb.height());
            for (x, y, p) in rgb.enumerate_pixels() {
                rgba.put_pixel(x, y, Rgba([p[0], p[1], p[2], 255]));
            }
            rgba
        };

        let (w, h) = img_rgba.dimensions();
        let mut input_array = Array3::<f32>::zeros((4, h as usize, w as usize));

        for (x, y, pixel) in img_rgba.enumerate_pixels() {
            input_array[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input_array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input_array[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            input_array[[3, y as usize, x as usize]] = pixel[3] as f32 / 255.0;
        }

        let output_array = if has_alpha {
            self.process_rgba(&input_array)?
        } else {
            let rgb = input_array.slice(ndarray::s![0..3, .., ..]).to_owned();
            let enhanced_rgb = self.process_rgb(&rgb)?;
            let mut result = Array3::<f32>::zeros((4, h as usize, w as usize));
            result
                .slice_mut(ndarray::s![0..3, .., ..])
                .assign(&enhanced_rgb);
            result.slice_mut(ndarray::s![3, .., ..]).fill(1.0);
            result
        };

        // Convert back to image
        let mut out_rgba = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = (output_array[[0, y as usize, x as usize]].clamp(0.0, 1.0) * 255.0).round()
                    as u8;
                let g = (output_array[[1, y as usize, x as usize]].clamp(0.0, 1.0) * 255.0).round()
                    as u8;
                let b = (output_array[[2, y as usize, x as usize]].clamp(0.0, 1.0) * 255.0).round()
                    as u8;
                let a = if has_alpha {
                    (output_array[[3, y as usize, x as usize]].clamp(0.0, 1.0) * 255.0).round()
                        as u8
                } else {
                    255
                };
                out_rgba.put_pixel(x, y, Rgba([r, g, b, a]));
            }
        }

        if has_alpha {
            Ok(DynamicImage::ImageRgba8(out_rgba))
        } else {
            Ok(DynamicImage::ImageRgb8(
                image::ImageBuffer::from_raw(w, h, {
                    let rgba_raw = out_rgba.into_raw();
                    rgba_raw
                        .chunks(4)
                        .flat_map(|c| c[0..3].to_vec())
                        .collect::<Vec<u8>>()
                })
                .ok_or_else(|| {
                    GenericError::InvalidArgument("Failed to create RGB buffer".into())
                })?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    struct DummyEnhancer;

    impl ImageEnhancer for DummyEnhancer {
        fn process_rgb(&self, rgb_array: &Array3<f32>) -> Result<Array3<f32>, GenericError> {
            Ok(rgb_array.clone())
        }
    }

    #[test]
    fn test_dummy_enhancer_rgb() {
        let enhancer = DummyEnhancer;
        let mut img = ImageBuffer::new(4, 4);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgba([100, 150, 200, 255]);
        }
        let dyn_img = DynamicImage::ImageRgba8(img);
        let result = enhancer.enhance(&dyn_img).unwrap();
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
    }
}
