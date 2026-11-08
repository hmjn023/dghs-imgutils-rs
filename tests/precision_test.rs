//! Compare Rust's filter to actual Python/OpenCV output, never to the input.
//! Generate fixtures with tests/precision/generate_baseline.py, then set
//! DGHS_PRECISION_BASELINE_DIR and run this test with --ignored.
use dghs_imgutils_rs::restore::adversarial::{
    AdversarialNoiseParams, remove_adversarial_noise_with_params,
};
use image::{DynamicImage, RgbImage};
use std::path::PathBuf;

fn psnr(actual: &RgbImage, expected: &RgbImage) -> f64 {
    region_psnr(actual, expected, None)
}

/// Separates the four-pixel border affected by two radius-two filter passes.
fn region_psnr(actual: &RgbImage, expected: &RgbImage, border: Option<bool>) -> f64 {
    assert_eq!(
        actual.dimensions(),
        expected.dimensions(),
        "reference shape mismatch"
    );
    assert!(!actual.is_empty(), "empty reference image");
    let mut squared_error = 0.0;
    let mut count = 0;
    for (x, y, pixel) in actual.enumerate_pixels() {
        let is_border = x < 4 || y < 4 || x + 4 >= actual.width() || y + 4 >= actual.height();
        if border.is_some_and(|wanted| wanted != is_border) {
            continue;
        }
        for (&a, &b) in pixel.0.iter().zip(expected.get_pixel(x, y).0.iter()) {
            squared_error += (f64::from(a) - f64::from(b)).powi(2);
            count += 1;
        }
    }
    assert!(count > 0, "empty comparison region");
    let mse = squared_error / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64.powi(2) / mse).log10()
    }
}
fn params() -> AdversarialNoiseParams {
    AdversarialNoiseParams {
        b_iters: 2,
        g_iters: 0,
        diameter_min: 5,
        diameter_max: 5,
        sigma_color_min: 20.0,
        sigma_color_max: 20.0,
        sigma_space_min: 8.0,
        sigma_space_max: 8.0,
        ..Default::default()
    }
}
#[test]
#[ignore = "requires generated Python/OpenCV input and expected PNG fixtures"]
fn test_adversarial_noise_removal_precision() {
    let dir = std::env::var_os("DGHS_PRECISION_BASELINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/precision"));
    let input = image::open(dir.join("baseline_input.png"))
        .expect("missing input: run tests/precision/generate_baseline.py");
    let expected = image::open(dir.join("baseline_adversarial.png"))
        .expect("missing Python expected output: regenerate the baseline")
        .to_rgb8();
    assert_ne!(
        input.to_rgb8(),
        expected,
        "reference must exercise filtering, not identity"
    );
    let actual = remove_adversarial_noise_with_params(&input, &params())
        .expect("Rust filtering failed")
        .to_rgb8();
    let score = psnr(&actual, &expected);
    println!("Rust versus Python/OpenCV PSNR: {score:.2} dB");
    assert!(score > 35.0, "Python parity PSNR {score:.2} dB <= 35 dB");
    for (name, border) in [("border", true), ("interior", false)] {
        let score = region_psnr(&actual, &expected, Some(border));
        println!("{name} PSNR: {score:.2} dB");
        assert!(score > 35.0, "{name} PSNR {score:.2} dB <= 35 dB");
    }
    let identity_score = psnr(&input.to_rgb8(), &expected);
    assert!(
        score > identity_score + 3.0,
        "filter must improve over unchanged input ({identity_score:.2} dB)"
    );
}
#[test]
fn test_bilateral_filter_basic() {
    let input = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, image::Rgb([128; 3])));
    let actual = remove_adversarial_noise_with_params(&input, &params())
        .unwrap()
        .to_rgb8();
    assert_eq!(actual, input.to_rgb8());
}
#[test]
fn psnr_rejects_a_wrong_reference() {
    let a = RgbImage::from_pixel(16, 16, image::Rgb([0; 3]));
    let b = RgbImage::from_pixel(16, 16, image::Rgb([255; 3]));
    assert_eq!(psnr(&a, &b), 0.0);
}

#[test]
fn nonconstant_filter_has_known_weighted_output() {
    let input = DynamicImage::ImageRgb8(RgbImage::from_fn(2, 1, |x, _| {
        image::Rgb([100 + x as u8 * 10; 3])
    }));
    let mut settings = params();
    settings.b_iters = 1;
    settings.diameter_min = 3;
    settings.diameter_max = 3;
    // Neighbor weight exp(-1/(2*8²) - 3*10²/(2*20²)); normalized means round to 104,106.
    let actual = remove_adversarial_noise_with_params(&input, &settings)
        .unwrap()
        .to_rgb8();
    assert_eq!(actual.get_pixel(0, 0).0, [104; 3]);
    assert_eq!(actual.get_pixel(1, 0).0, [106; 3]);
}
