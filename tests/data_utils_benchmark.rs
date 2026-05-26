//! Data utils precision benchmark comparing Rust vs Python for image loading, encoding, and compositing.

use image::{DynamicImage, RgbImage, RgbaImage};
use ndarray::Array3;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonDataOutput {
    success: bool,
    error: Option<String>,
    r#type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    channels: Option<usize>,
    shape: Option<Vec<usize>>,
    values: Option<Vec<Vec<f64>>>,
}

fn run_python_bench(script: &str, image_path: &str) -> std::process::Output {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir(format!("{manifest_dir}/imgutils"))
        .arg("run")
        .arg("python")
        .arg(format!("{manifest_dir}/benchmark/{script}"))
        .arg(image_path);
    cmd.output().expect("Failed to execute python")
}

#[test]
#[ignore]
fn test_load_image_precision() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/nude_girl.png",
        "imgutils/test/testfile/hutao.jpg",
    ];

    println!("============================================================");
    println!("🧪 Data Utils: load_image precision benchmark");
    println!("============================================================");

    for rel_path in &test_images {
        let image_path = PathBuf::from(manifest_dir).join(rel_path);
        assert!(image_path.exists(), "Test image not found: {image_path:?}");

        println!("\n📸 Testing: {rel_path}");
        println!("------------------------------------------------------------");

        let py_output = run_python_bench("run_python_data.py", image_path.to_str().unwrap());
        let py_stdout = String::from_utf8_lossy(&py_output.stdout);
        let py_result: PythonDataOutput =
            serde_json::from_str(&py_stdout).expect("Failed to parse Python output");

        assert!(
            py_result.success,
            "Python data test failed: {:?}",
            py_result.error
        );

        let rust_start = Instant::now();
        let rust_img = image::open(&image_path).expect("Failed to open image in Rust");
        let rust_duration = rust_start.elapsed();

        let (rust_w, rust_h) = rust_img.dimensions();
        println!(
            "  Python: {}x{} ({} ch)",
            py_result.width.unwrap_or(0),
            py_result.height.unwrap_or(0),
            py_result.channels.unwrap_or(0)
        );
        println!("  Rust:   {rust_w}x{rust_h}");
        println!("  Rust load duration: {rust_duration:?}");

        assert_eq!(
            rust_w,
            py_result.width.unwrap_or(0),
            "Width mismatch for {rel_path}"
        );
        assert_eq!(
            rust_h,
            py_result.height.unwrap_or(0),
            "Height mismatch for {rel_path}"
        );
    }

    println!("\n✅ All load_image precision checks passed!");
}

#[test]
#[ignore]
fn test_rgb_encode_precision() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/nude_girl.png",
    ];

    println!("\n🧪 rgb_encode precision benchmark");
    println!("============================================================");

    for rel_path in &test_images {
        let image_path = PathBuf::from(manifest_dir).join(rel_path);
        assert!(image_path.exists());

        println!("\n📸 Testing: {rel_path}");

        let py_output = run_python_bench("run_python_data.py", image_path.to_str().unwrap());
        let py_stdout = String::from_utf8_lossy(&py_output.stdout);
        let py_result: PythonDataOutput =
            serde_json::from_str(&py_stdout).expect("Failed to parse Python output");

        assert!(
            py_result.success,
            "Python encode failed: {:?}",
            py_result.error
        );

        let rust_start = Instant::now();
        let img = image::open(&image_path).expect("Failed to open image in Rust");
        let rust_encoded =
            dghs_imgutils_rs::image::rgb_encode(&img, "CHW", true).expect("Rust rgb_encode failed");
        let rust_duration = rust_start.elapsed();

        println!("  Python shape: {:?}", py_result.shape);
        println!("  Rust shape:   {:?}", rust_encoded.shape());
        println!("  Rust encode duration: {rust_duration:?}");

        assert_eq!(
            rust_encoded.shape(),
            &[
                3,
                py_result.height.unwrap_or(0) as usize,
                py_result.width.unwrap_or(0) as usize
            ],
            "Shape mismatch for {rel_path}"
        );

        if let Some(values) = &py_result.values {
            let max_diff = values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let rust_val = rust_encoded[[
                        i % 3,
                        (i / 3) % py_result.height.unwrap_or(0) as usize,
                        (i / 3) / py_result.height.unwrap_or(0) as usize,
                    ]];
                    (v[0] - rust_val as f64).abs()
                })
                .fold(0.0f64, f64::max);

            println!("  Max pixel diff: {max_diff:.8}");
            assert!(
                max_diff < 1e-5,
                "Max pixel difference too large: {max_diff}"
            );
        }
    }

    println!("\n✅ All rgb_encode precision checks passed!");
}

#[test]
#[ignore]
fn test_rgb_decode_roundtrip() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_img_path = PathBuf::from(manifest_dir).join("imgutils/test/testfile/skadi.jpg");

    let img = image::open(&test_img_path).expect("Failed to open image");
    let encoded =
        dghs_imgutils_rs::image::rgb_encode(&img, "CHW", true).expect("rgb_encode failed");
    let decoded =
        dghs_imgutils_rs::image::rgb_decode(&encoded.view(), "CHW").expect("rgb_decode failed");

    let (dw, dh) = decoded.dimensions();
    let (iw, ih) = img.dimensions();
    assert_eq!(dw, iw);
    assert_eq!(dh, ih);

    let orig_rgb = img.to_rgb8();
    let dec_rgb = decoded.to_rgb8();
    let mut total_diff: u64 = 0;
    for y in 0..ih {
        for x in 0..iw {
            let op = orig_rgb.get_pixel(x, y);
            let dp = dec_rgb.get_pixel(x, y);
            let d = (op[0] as i32 - dp[0] as i32).unsigned_abs()
                + (op[1] as i32 - dp[1] as i32).unsigned_abs()
                + (op[2] as i32 - dp[2] as i32).unsigned_abs();
            total_diff += d as u64;
        }
    }
    let total_pixels = iw as u64 * ih as u64;
    let avg_diff = total_diff as f64 / (total_pixels * 3) as f64;
    println!("Average per-channel difference in roundtrip: {avg_diff:.4}");
    assert!(
        avg_diff < 1.0,
        "Roundtrip error too large: avg_diff={avg_diff}"
    );
}

#[test]
#[ignore]
fn test_istack_compositing() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let red_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, image::Rgb([255, 0, 0])));
    let blue_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, image::Rgb([0, 0, 255])));

    let layers = vec![
        dghs_imgutils_rs::image::LayerItemBuilder::from_image(red_img.clone()).with_alpha(0.5),
        dghs_imgutils_rs::image::LayerItemBuilder::from_image(blue_img.clone()).with_alpha(0.5),
    ];

    let start = Instant::now();
    let result = dghs_imgutils_rs::image::istack(layers, Some((100, 100))).expect("istack failed");
    let duration = start.elapsed();

    assert_eq!(result.dimensions(), (100, 100));
    println!("istack duration: {duration:?}");

    let py_output = run_python_bench("run_python_data.py", manifest_dir);
    let py_stdout = String::from_utf8_lossy(&py_output.stdout);
    println!("Python output: {py_stdout}");
}
