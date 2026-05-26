use dghs_imgutils_rs::ocr::{detect_text_with_paddleocr, ocr};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonOcrOutput {
    success: bool,
    error: Option<String>,
    detections: Vec<PythonDetection>,
    results: Vec<PythonOcrItem>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonDetection {
    bbox: [i32; 4],
    score: f32,
}

#[derive(Debug, serde::Deserialize)]
struct PythonOcrItem {
    bbox: [i32; 4],
    text: String,
    score: f32,
}

#[test]
#[ignore]
fn test_ocr_precision_and_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/nude_girl.png",
        "imgutils/test/testfile/hutao.jpg",
    ];

    println!("============================================================");
    println!(" OCR (PaddleOCR) Precision Validation & Benchmark");
    println!("============================================================");

    let mut image_paths = Vec::new();
    for rel_path in &test_images {
        let mut path = PathBuf::from(manifest_dir);
        path.push(rel_path);
        assert!(path.exists(), "Test image not found: {:?}", path);
        image_paths.push(path);
    }

    println!("Running Python OCR inference...");
    let py_start = Instant::now();

    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_ocr.py");
    for path in &image_paths {
        cmd.arg(path);
    }

    let output = cmd.output().expect("Failed to execute python via uv");
    let py_duration = py_start.elapsed();

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        panic!(
            "Python error:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }

    let py_output: PythonOcrOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");

    assert!(
        py_output.success,
        "Python OCR failed: {:?}",
        py_output.error
    );
    println!("Python OCR done ({:?})", py_duration);

    println!("Running Rust OCR inference...");
    let mut images = Vec::new();
    for path in &image_paths {
        let img = image::open(path).expect("Failed to open test image in Rust");
        images.push(img);
    }

    let rust_start = Instant::now();
    let mut all_results = Vec::new();
    for img in &images {
        let results = ocr(img).expect("Rust OCR failed");
        all_results.push(results);
    }
    let rust_duration = rust_start.elapsed();
    println!("Rust OCR done ({:?})", rust_duration);

    assert_eq!(
        all_results.len(),
        py_output.results.len(),
        "Number of images processed differs"
    );

    for (img_idx, rust_items) in all_results.iter().enumerate() {
        let py_items = &py_output.results;
        println!(
            "Image {}: Rust got {} results, Python got {} results",
            img_idx,
            rust_items.len(),
            py_items.len()
        );
    }

    let speedup = py_duration.as_secs_f64() / rust_duration.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);
    println!("============================================================");
}
