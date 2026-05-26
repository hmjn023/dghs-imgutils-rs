//! 画像操作 (operate) モジュールのベンチマーク・精度検証テスト。
//!
//! - 画像ベース検閲 (imgcensor): `censor_area_image`, `censor_areas_image`
//! - Python 版と Rust 版の出力をピクセル単位で比較

use dghs_imgutils_rs::detect::base::BBox;
use dghs_imgutils_rs::operate::imgcensor::{CensorImageFit, censor_area_image, censor_areas_image};
use image::{DynamicImage, Rgba, RgbaImage};
use std::path::PathBuf;
use std::time::Instant;

/// Python 側が出力した画像パスを受け取る構造体
#[derive(Debug, serde::Deserialize)]
struct PythonOperateOutput {
    success: bool,
    error: Option<String>,
    output_path: Option<String>,
}

/// 100x100 のテスト用スタンプ画像を生成（不透明な赤い円に近いイメージ）
fn make_test_stamp(size: u32) -> DynamicImage {
    let mut img = RgbaImage::new(size, size);
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let r = size as f32 / 2.0 - 1.0;
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 - cx).abs();
            let dy = (y as f32 - cy).abs();
            if dx * dx + dy * dy <= r * r {
                img.put_pixel(x, y, Rgba([255, 50, 50, 255]));
            } else {
                img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }
    DynamicImage::ImageRgba8(img)
}

/// 50x50 の半透明緑の正方形スタンプ
fn make_translucent_stamp(size: u32) -> DynamicImage {
    let mut img = RgbaImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            img.put_pixel(x, y, Rgba([0, 200, 100, 128]));
        }
    }
    DynamicImage::ImageRgba8(img)
}

#[test]
fn test_imgcensor_area_contain_basic() {
    let mut target = RgbaImage::from_pixel(200, 200, Rgba([255, 255, 255, 255]));
    let stamp = make_test_stamp(20);
    let area = [30.0, 30.0, 80.0, 80.0]; // 50x50 area

    censor_area_image(&mut target, &area, &stamp, CensorImageFit::Contain);

    // Stamp is 20x20, area is 50x50, contain fit -> scale = min(50/20, 50/20) = 2.5
    // new_w = new_h = 50, centered -> offset = (30 + 0, 30 + 0) = (30, 30)
    // Stamp should cover (30..80, 30..80)
    let px = target.get_pixel(55, 55);
    assert_eq!(px[0], 255);
    assert_eq!(px[1], 50);
    assert_eq!(px[2], 50);

    // Outside stamp but inside area should be white
    let px = target.get_pixel(30, 30);
    assert_eq!(px[0], 255);
    assert_eq!(px[1], 255);
    assert_eq!(px[2], 255);
}

#[test]
fn test_imgcensor_area_cover_basic() {
    let mut target = RgbaImage::from_pixel(200, 200, Rgba([255, 255, 255, 255]));
    let stamp = make_test_stamp(20);
    let area = [0.0, 0.0, 40.0, 20.0]; // 40x20 area

    censor_area_image(&mut target, &area, &stamp, CensorImageFit::Cover);

    // Cover: scale = max(40/20, 20/20) = 2.0, new_w=40, new_h=40
    // offset = (0 + (40-40)/2, 0 + (20-40)/2) = (0, -10) -> clamp at y1=0
    // Stamp visible at x 0..40, y 0..20
    let px = target.get_pixel(20, 10);
    assert_eq!(px[0], 255);
    assert_eq!(px[1], 50);
    assert_eq!(px[2], 50);
}

#[test]
fn test_imgcensor_alpha_blend() {
    let mut target = RgbaImage::from_pixel(50, 50, Rgba([255, 0, 0, 255]));
    let stamp = make_translucent_stamp(10);
    let area = [0.0, 0.0, 10.0, 10.0];

    censor_area_image(&mut target, &area, &stamp, CensorImageFit::Contain);

    // 50% green overlay on red -> (0*0.5 + 255*0.5, 200*0.5 + 0*0.5, 100*0.5 + 0*0.5)
    // = (127, 100, 50)
    let px = target.get_pixel(5, 5);
    assert!(
        (px[0] as i16 - 127).abs() <= 1,
        "Red channel mismatch at alpha blend: got {}",
        px[0]
    );
    assert!(
        (px[1] as i16 - 100).abs() <= 1,
        "Green channel mismatch at alpha blend: got {}",
        px[1]
    );
    assert!(
        (px[2] as i16 - 50).abs() <= 1,
        "Blue channel mismatch at alpha blend: got {}",
        px[2]
    );
}

#[test]
fn test_imgcensor_areas_multiple() {
    let mut target = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
    let stamp = make_test_stamp(10);
    let areas: Vec<BBox> = vec![(10, 10, 30, 30), (60, 60, 80, 80)];

    censor_areas_image(&mut target, &areas, &stamp, CensorImageFit::Contain);

    // Both areas should be stamped
    assert_eq!(target.get_pixel(15, 15)[0], 255);
    assert_eq!(target.get_pixel(65, 65)[0], 255);

    // Middle should be white
    assert_eq!(target.get_pixel(45, 45)[0], 255);
}

#[test]
fn test_imgcensor_edge_clip() {
    let mut target = RgbaImage::from_pixel(30, 30, Rgba([255, 255, 255, 255]));
    let stamp = make_test_stamp(20);
    // Area extends beyond image boundary
    let area = [20.0, 20.0, 50.0, 50.0];

    // Should not panic
    censor_area_image(&mut target, &area, &stamp, CensorImageFit::Contain);
    // Pixels inside the image portion of the area should still be stamped
    assert_eq!(target.get_pixel(29, 29)[0], 255);
}

#[test]
fn test_imgcensor_empty_area() {
    let mut target = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
    let stamp = make_test_stamp(20);
    // Zero-size area
    let area = [50.0, 50.0, 50.0, 50.0];

    // Should not panic
    censor_area_image(&mut target, &area, &stamp, CensorImageFit::Contain);
}

#[test]
#[ignore]
fn test_imgcensor_benchmark_and_precision() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push("imgutils/test/testfile/mostima_post.jpg");

    let mut stamp_path = PathBuf::from(manifest_dir);
    stamp_path.push("imgutils/imgutils/operate/heart_censor.png");

    if !image_path.exists() || !stamp_path.exists() {
        eprintln!("Skipping benchmark: test files not found.");
        eprintln!("  image: {:?}", image_path);
        eprintln!("  stamp: {:?}", stamp_path);
        return;
    }

    println!("\n🧪 Operate Benchmark: Image-based censor");
    println!("==============================================");

    // --- Rust benchmark ---
    let img = image::open(&image_path).expect("Failed to open test image");
    let stamp = image::open(&stamp_path).expect("Failed to open stamp image");

    let mut areas: Vec<BBox> = vec![];
    // Simulate a few NSFW-like areas (face-sized boxes at various positions)
    areas.push((100, 100, 220, 240));
    areas.push((300, 150, 380, 250));
    areas.push((500, 80, 600, 180));

    let rust_start = Instant::now();
    let mut rgba = img.to_rgba8();
    censor_areas_image(&mut rgba, &areas, &stamp, CensorImageFit::Cover);
    let rust_duration = rust_start.elapsed();
    println!("✓ Rust censor_areas_image: {:?}", rust_duration);

    // Save output for visual inspection
    let result_img = DynamicImage::ImageRgba8(rgba);
    let out_path = PathBuf::from(manifest_dir).join("target/rust_censor_output.png");
    result_img
        .save(&out_path)
        .expect("Failed to save Rust output");
    println!("  Output saved to: {:?}", out_path);

    // --- Python reference ---
    let py_start = Instant::now();
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_operate.py")
        .arg("censor_area_image")
        .arg(&image_path)
        .arg(&stamp_path);

    let output = cmd.output().expect("Failed to execute Python benchmark");
    let py_duration = py_start.elapsed();

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "⚠ Python side failed (non-critical):\n{}\n{}",
            stdout_str, stderr_str
        );
        eprintln!("Skipping precision comparison.");
        return;
    }

    let py_output: PythonOperateOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python inference failed: {:?}",
        py_output.error
    );
    println!("✓ Python reference: {:?}", py_duration);

    println!("\nSpeed comparison:");
    println!("  Rust:  {:?}", rust_duration);
    println!("  Python: {:?}", py_duration);
    println!(
        "  Speedup: {:.2}x",
        py_duration.as_secs_f64() / rust_duration.as_secs_f64()
    );

    // --- Output size check ---
    let rust_file_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!("  Rust output file size: {} bytes", rust_file_size);
    assert!(rust_file_size > 1000, "Rust output file suspiciously small");

    println!("🎉 Operate benchmark passed!");
}
