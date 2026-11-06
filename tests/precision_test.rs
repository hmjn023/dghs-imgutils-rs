//! Compare Rust's filter to actual Python/OpenCV output, never to the input.
//! Generate fixtures with tests/precision/generate_baseline.py, then set
//! DGHS_PRECISION_BASELINE_DIR and run this test with --ignored.
use dghs_imgutils_rs::restore::adversarial::{
    AdversarialNoiseParams, remove_adversarial_noise_with_params,
};
use image::{DynamicImage, RgbImage};
use std::path::PathBuf;

fn psnr(actual: &RgbImage, expected: &RgbImage) -> f64 {
    assert_eq!(
        actual.dimensions(),
        expected.dimensions(),
        "reference shape mismatch"
    );
    assert!(!actual.is_empty(), "empty reference image");
    let mse = actual
        .as_raw()
        .iter()
        .zip(expected.as_raw())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / actual.as_raw().len() as f64;
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
