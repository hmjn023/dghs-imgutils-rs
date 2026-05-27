//! Adversarial noise removal using bilateral and guided filtering.
//!
//! Pure image processing — no ONNX models required.
//! Inspired by <https://huggingface.co/spaces/mf666/mist-fucker>

use image::DynamicImage;
use ndarray::{Array2, Array3, Axis};
use rand::Rng;

use super::RestoreError;

/// adversarial noise removal のパラメータ。
///
/// Python 版 `remove_adversarial_noise` の全パラメータを網羅する。
#[derive(Debug, Clone)]
pub struct AdversarialNoiseParams {
    /// Bilateral filtering の反復回数（デフォルト: 64）。
    pub b_iters: usize,
    /// Guided filtering の反復回数（デフォルト: 8）。
    pub g_iters: usize,
    /// Bilateral filter の直径の最小値（デフォルト: 4）。
    pub diameter_min: usize,
    /// Bilateral filter の直径の最大値（デフォルト: 6）。
    pub diameter_max: usize,
    /// Bilateral filter の sigma_color の最小値（デフォルト: 6.0）。
    pub sigma_color_min: f32,
    /// Bilateral filter の sigma_color の最大値（デフォルト: 10.0）。
    pub sigma_color_max: f32,
    /// Bilateral filter の sigma_space の最小値（デフォルト: 6.0）。
    pub sigma_space_min: f32,
    /// Bilateral filter の sigma_space の最大値（デフォルト: 10.0）。
    pub sigma_space_max: f32,
    /// Guided filter の radius の最小値（デフォルト: 3）。
    pub radius_min: usize,
    /// Guided filter の radius の最大値（デフォルト: 6）。
    pub radius_max: usize,
    /// Guided filter の eps の最小値（デフォルト: 16.0）。
    pub eps_min: f32,
    /// Guided filter の eps の最大値（デフォルト: 24.0）。
    pub eps_max: f32,
}

impl Default for AdversarialNoiseParams {
    fn default() -> Self {
        Self {
            b_iters: 64,
            g_iters: 8,
            diameter_min: 4,
            diameter_max: 6,
            sigma_color_min: 6.0,
            sigma_color_max: 10.0,
            sigma_space_min: 6.0,
            sigma_space_max: 10.0,
            radius_min: 3,
            radius_max: 6,
            eps_min: 16.0,
            eps_max: 24.0,
        }
    }
}

/// Apply a box (mean) filter to a 2D array.
/// Uses direct O(HW*R^2) summation (R ≤ 6, so performance is acceptable).
fn box_filter_2d(arr: &Array2<f32>, radius: usize) -> Array2<f32> {
    let (h, w) = arr.dim();
    let mut result = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let y_start = if y >= radius { y - radius } else { 0 };
            let y_end = (y + radius + 1).min(h);
            let x_start = if x >= radius { x - radius } else { 0 };
            let x_end = (x + radius + 1).min(w);
            let mut sum = 0.0;
            let mut count = 0.0;
            for ky in y_start..y_end {
                for kx in x_start..x_end {
                    sum += arr[[ky, kx]];
                    count += 1.0;
                }
            }
            result[[y, x]] = if count > 0.0 { sum / count } else { 0.0 };
        }
    }
    result
}

/// Guided filter applied to each channel independently.
///
/// Reference: K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010.
fn guided_filter_channel(guide: &Array2<f32>, input: &Array2<f32>, radius: usize, eps: f32) -> Array2<f32> {
    let mean_i = box_filter_2d(guide, radius);
    let mean_p = box_filter_2d(input, radius);
    let mean_ii = box_filter_2d(&(guide * guide), radius);
    let mean_ip = box_filter_2d(&(guide * input), radius);

    let var_i = &mean_ii - &mean_i * &mean_i;
    let cov_ip = &mean_ip - &mean_i * &mean_p;

    let a = &cov_ip / (&var_i + eps);
    let b = &mean_p - &a * &mean_i;

    let mean_a = box_filter_2d(&a, radius);
    let mean_b = box_filter_2d(&b, radius);

    mean_a * guide + mean_b
}

/// Guided filter for multi-channel image [C, H, W].
fn guided_filter_color(guide: &Array3<f32>, input: &Array3<f32>, radius: usize, eps: f32) -> Array3<f32> {
    let channels = guide.shape()[0];
    let h = guide.shape()[1];
    let w = guide.shape()[2];
    let mut result = Array3::<f32>::zeros((channels, h, w));
    for c in 0..channels {
        let g = guide.index_axis(Axis(0), c).to_owned();
        let p = input.index_axis(Axis(0), c).to_owned();
        let q = guided_filter_channel(&g, &p, radius, eps);
        result.slice_mut(ndarray::s![c, .., ..]).assign(&q);
    }
    result
}

/// Simple bilateral filter for an RGB image array [3, H, W].
fn bilateral_filter_color(img: &Array3<f32>, diameter: usize, sigma_color: f32, sigma_space: f32) -> Array3<f32> {
    let (h, w) = (img.shape()[1], img.shape()[2]);
    let half = (diameter / 2) as isize;
    let mut result = img.to_owned();

    let spatial_weights: Vec<Vec<f32>> = (0..diameter)
        .map(|dy| {
            (0..diameter)
                .map(|dx| {
                    let dy = dy as isize - half;
                    let dx = dx as isize - half;
                    (-(dy * dy + dx * dx) as f32 / (2.0 * sigma_space * sigma_space)).exp()
                })
                .collect()
        })
        .collect();

    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            let mut total_weight = 0.0f32;

            let r0 = img[[0, y, x]];
            let g0 = img[[1, y, x]];
            let b0 = img[[2, y, x]];

            for dy in 0..diameter {
                for dx in 0..diameter {
                    let ny = y as isize + dy as isize - half;
                    let nx = x as isize + dx as isize - half;
                    if ny < 0 || ny >= h as isize || nx < 0 || nx >= w as isize {
                        continue;
                    }
                    let ny = ny as usize;
                    let nx = nx as usize;

                    let dr = r0 - img[[0, ny, nx]];
                    let dg = g0 - img[[1, ny, nx]];
                    let db = b0 - img[[2, ny, nx]];
                    let range_dist = dr * dr + dg * dg + db * db;
                    let range_weight = (-range_dist / (2.0 * sigma_color * sigma_color)).exp();
                    let weight = spatial_weights[dy][dx] * range_weight;

                    sum[0] += img[[0, ny, nx]] * weight;
                    sum[1] += img[[1, ny, nx]] * weight;
                    sum[2] += img[[2, ny, nx]] * weight;
                    total_weight += weight;
                }
            }

            if total_weight > 1e-8 {
                result[[0, y, x]] = sum[0] / total_weight;
                result[[1, y, x]] = sum[1] / total_weight;
                result[[2, y, x]] = sum[2] / total_weight;
            }
        }
    }

    result
}

/// Remove adversarial noise from an image using randomized bilateral and guided filtering.
///
/// This is a pure filter-based approach (no ONNX model required).
///
/// # Arguments
///
/// * `image` - Input image
/// * `b_iters` - Number of bilateral filtering iterations (default: 64)
/// * `g_iters` - Number of guided filtering iterations (default: 8)
pub fn remove_adversarial_noise(
    image: &DynamicImage,
    b_iters: usize,
    g_iters: usize,
) -> Result<DynamicImage, RestoreError> {
    let params = AdversarialNoiseParams {
        b_iters,
        g_iters,
        ..Default::default()
    };
    remove_adversarial_noise_with_params(image, &params)
}

/// Remove adversarial noise from an image using randomized bilateral and guided filtering.
///
/// Python 版と同一の全パラメータを指定可能。
///
/// # Arguments
///
/// * `image` - Input image
/// * `params` - Filtering parameters
pub fn remove_adversarial_noise_with_params(
    image: &DynamicImage,
    params: &AdversarialNoiseParams,
) -> Result<DynamicImage, RestoreError> {
    let img_rgb = image.to_rgb8();
    let (w, h) = img_rgb.dimensions();
    let h = h as usize;
    let w = w as usize;

    let mut img_arr = Array3::<f32>::zeros((3, h, w));
    for (x, y, pixel) in img_rgb.enumerate_pixels() {
        img_arr[[0, y as usize, x as usize]] = pixel[0] as f32;
        img_arr[[1, y as usize, x as usize]] = pixel[1] as f32;
        img_arr[[2, y as usize, x as usize]] = pixel[2] as f32;
    }

    let original = img_arr.clone();
    let mut y = img_arr;

    let mut rng = rand::thread_rng();

    // Bilateral filtering iterations
    for _ in 0..params.b_iters {
        let diameter = rng.gen_range(params.diameter_min..=params.diameter_max);
        let sigma_color = rng.gen_range(params.sigma_color_min..=params.sigma_color_max);
        let sigma_space = rng.gen_range(params.sigma_space_min..=params.sigma_space_max);
        y = bilateral_filter_color(&y, diameter, sigma_color, sigma_space);
    }

    // Guided filtering iterations
    for _ in 0..params.g_iters {
        let radius = rng.gen_range(params.radius_min..=params.radius_max);
        let eps = rng.gen_range(params.eps_min..=params.eps_max);
        y = guided_filter_color(&original, &y, radius, eps);
    }

    // Clip and convert back
    let mut out_rgb = image::ImageBuffer::new(w as u32, h as u32);
    for y_idx in 0..h {
        for x_idx in 0..w {
            let r = (y[[0, y_idx, x_idx]].clamp(0.0, 255.0)).round() as u8;
            let g = (y[[1, y_idx, x_idx]].clamp(0.0, 255.0)).round() as u8;
            let b = (y[[2, y_idx, x_idx]].clamp(0.0, 255.0)).round() as u8;
            out_rgb.put_pixel(x_idx as u32, y_idx as u32, image::Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(out_rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_filter_2d() {
        let arr = Array2::<f32>::ones((10, 10));
        let result = box_filter_2d(&arr, 1);
        // All values should be 1.0 after box filter of constant array
        for v in result.iter() {
            assert!((*v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_remove_adversarial_noise_basic() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(32, 32));
        let result = remove_adversarial_noise(&img, 2, 1);
        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.width(), 32);
        assert_eq!(out.height(), 32);
    }
}
