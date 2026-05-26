use image::DynamicImage;

/// Compute PSNR (Peak Signal-to-Noise Ratio) between two images in dB.
/// Returns `f64::INFINITY` if both images are identical (MSE = 0).
pub fn psnr(img1: &DynamicImage, img2: &DynamicImage) -> f64 {
    let rgb1 = img1.to_rgb8();
    let rgb2 = img2.to_rgb8();

    let (w1, h1) = rgb1.dimensions();
    let (w2, h2) = rgb2.dimensions();
    let w = w1.min(w2);
    let h = h1.min(h2);

    let mut mse = 0.0f64;
    let n = (w * h * 3) as f64;
    for y in 0..h {
        for x in 0..w {
            let p1 = rgb1.get_pixel(x, y);
            let p2 = rgb2.get_pixel(x, y);
            for c in 0..3 {
                let d = p1[c] as f64 - p2[c] as f64;
                mse += d * d;
            }
        }
    }
    mse /= n;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64.powi(2) / mse).log10()
    }
}
