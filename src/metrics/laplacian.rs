use image::DynamicImage;
use imageproc::filter::laplacian_filter;

/// Compute Laplacian variance score (sharpness metric).
/// Higher values indicate sharper images. Threshold ~100 is typical for blur detection.
pub fn laplacian_score(image: &DynamicImage) -> f32 {
    let gray = image.to_luma8();
    let lap = laplacian_filter(&gray);
    let pixels: Vec<f32> = lap.iter().map(|&p| p as f32).collect();
    let n = pixels.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean = pixels.iter().sum::<f32>() / n;
    let variance = pixels.iter().map(|&p| (p - mean).powi(2)).sum::<f32>() / n;
    variance
}
