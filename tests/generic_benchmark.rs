//! Benchmark comparing Rust generic module inference with Python (uv) reference.
//!
//! Run with: cargo test --release -- --ignored --nocapture generic_benchmark
//!
//! Requires:
//!   - Python environment with `imgutils` installed (e.g., via `uv`)
//!   - Network access to HuggingFace Hub for model downloads

use std::process::Command;
use std::time::Instant;

/// Path to Python interpreter via uv
const PYTHON_CMD: &str = "uv";
const PYTHON_ARGS: &[&str] = &["run", "python"];

fn run_python_script(script: &str) -> Result<String, String> {
    let output = Command::new(PYTHON_CMD)
        .args(PYTHON_ARGS)
        .arg("-c")
        .arg(script)
        .output()
        .map_err(|e| format!("Failed to run Python: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python error: {}", stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[test]
#[ignore]
fn generic_benchmark_clip_image_encode() {
    // Python reference timing
    let py_script = r#"
import time, numpy as np
from imgutils.generic import clip_image_encode

# Create a test image
from PIL import Image
img = Image.new('RGB', (224, 224), color=(128, 128, 128))

# Warm up
clip_image_encode([img], repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')

# Timed runs
start = time.perf_counter()
n_runs = 10
for _ in range(n_runs):
    clip_image_encode([img], repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark:\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }

    // Rust timing
    let image = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
        224,
        224,
        image::Rgb([128u8, 128, 128]),
    ));

    // Warm up
    let _ = dghs_imgutils_rs::generic::clip::clip_image_encode(
        &[image.clone()],
        "deepghs/clip_onnx",
        "clip-vit-base-patch32",
    );

    let n_runs = 10;
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = dghs_imgutils_rs::generic::clip::clip_image_encode(
            &[image.clone()],
            "deepghs/clip_onnx",
            "clip-vit-base-patch32",
        );
    }
    let elapsed = start.elapsed();
    println!(
        "Rust avg: {:.2}ms",
        elapsed.as_secs_f64() / n_runs as f64 * 1000.0
    );
}

#[test]
#[ignore]
fn generic_benchmark_clip_text_encode() {
    let py_script = r#"
import time, numpy as np
from imgutils.generic import clip_text_encode

texts = ["a cat sitting on a mat", "a dog playing in the park", "anime style illustration", "beautiful landscape"]

clip_text_encode(texts, repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')

start = time.perf_counter()
n_runs = 10
for _ in range(n_runs):
    clip_text_encode(texts, repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark:\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }

    let texts: Vec<String> = vec![
        "a cat sitting on a mat".into(),
        "a dog playing in the park".into(),
        "anime style illustration".into(),
        "beautiful landscape".into(),
    ];

    let _ = dghs_imgutils_rs::generic::clip::clip_text_encode(
        &texts,
        "deepghs/clip_onnx",
        "clip-vit-base-patch32",
    );

    let n_runs = 10;
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = dghs_imgutils_rs::generic::clip::clip_text_encode(
            &texts,
            "deepghs/clip_onnx",
            "clip-vit-base-patch32",
        );
    }
    let elapsed = start.elapsed();
    println!(
        "Rust avg: {:.2}ms",
        elapsed.as_secs_f64() / n_runs as f64 * 1000.0
    );
}

#[test]
#[ignore]
fn generic_benchmark_clip_predict() {
    let py_script = r#"
import time, numpy as np
from imgutils.generic import clip_predict
from PIL import Image

img = Image.new('RGB', (224, 224), color=(128, 128, 128))
texts = ["cat", "dog", "anime", "landscape"]

clip_predict([img], texts, repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')

start = time.perf_counter()
n_runs = 10
for _ in range(n_runs):
    clip_predict([img], texts, repo_id='deepghs/clip_onnx', model_name='clip-vit-base-patch32')
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark:\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }
}

#[test]
#[ignore]
fn generic_benchmark_yoloseg() {
    println!("YOLO-Seg benchmark requires a model + image. Skipping automated run.");
    println!("Run manually with: cargo test --release -- --ignored --nocapture yoloseg");
}
