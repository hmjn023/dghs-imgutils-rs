//! Benchmark comparing Rust restore/upscale with Python (uv) reference.
//!
//! Run with: `cargo test --release -- --ignored --nocapture restore_upscale_benchmark`
//!
//! Requires:
//!   - Python environment with `imgutils` installed (e.g., via `uv`)
//!   - Network access to HuggingFace Hub for model downloads
//!   - Test images in `tests/` directory

use std::process::Command;
use std::time::Instant;

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
fn restore_benchmark_adversarial() {
    let py_script = r#"
import time
from PIL import Image
from imgutils.restore import remove_adversarial_noise

img = Image.new('RGB', (256, 256), color=(128, 128, 128))
for _ in range(2):
    remove_adversarial_noise(img, b_iters=4, g_iters=1)

start = time.perf_counter()
n_runs = 3
for _ in range(n_runs):
    remove_adversarial_noise(img, b_iters=4, g_iters=1)
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark (adversarial):\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }

    let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
        256,
        256,
        image::Rgb([128u8, 128, 128]),
    ));

    let _ = dghs_imgutils_rs::restore::adversarial::remove_adversarial_noise(&img, 4, 1);

    let n_runs = 3;
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = dghs_imgutils_rs::restore::adversarial::remove_adversarial_noise(&img, 4, 1);
    }
    let elapsed = start.elapsed();
    println!(
        "Rust avg: {:.2}ms",
        elapsed.as_secs_f64() / n_runs as f64 * 1000.0
    );
}

#[test]
#[ignore]
fn restore_benchmark_nafnet() {
    let py_script = r#"
import time
from PIL import Image
from imgutils.restore import restore_with_nafnet

img = Image.new('RGB', (256, 256), color=(128, 128, 128))
restore_with_nafnet(img, model='REDS', tile_size=128, tile_overlap=16, batch_size=2)

start = time.perf_counter()
n_runs = 2
for _ in range(n_runs):
    restore_with_nafnet(img, model='REDS', tile_size=128, tile_overlap=16, batch_size=2)
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark (NafNet):\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }
    println!("Rust NafNet benchmark requires model download. Skipping automated run.");
    println!("Run manually after downloading: cargo test --release -- --ignored --nocapture nafnet");
}

#[test]
#[ignore]
fn restore_benchmark_scunet() {
    let py_script = r#"
import time
from PIL import Image
from imgutils.restore import restore_with_scunet

img = Image.new('RGB', (256, 256), color=(128, 128, 128))
restore_with_scunet(img, model='GAN', tile_size=128, tile_overlap=16, batch_size=2)

start = time.perf_counter()
n_runs = 2
for _ in range(n_runs):
    restore_with_scunet(img, model='GAN', tile_size=128, tile_overlap=16, batch_size=2)
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark (SCUNet):\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }
    println!("Rust SCUNet benchmark requires model download. Skipping automated run.");
}

#[test]
#[ignore]
fn upscale_benchmark_cdc() {
    let py_script = r#"
import time
from PIL import Image
from imgutils.upscale import upscale_with_cdc

img = Image.new('RGB', (256, 256), color=(128, 128, 128))
upscale_with_cdc(img, model='HGSR-MHR-anime-aug_X4_320', tile_size=128, tile_overlap=32, batch_size=1)

start = time.perf_counter()
n_runs = 1
for _ in range(n_runs):
    upscale_with_cdc(img, model='HGSR-MHR-anime-aug_X4_320', tile_size=128, tile_overlap=32, batch_size=1)
elapsed = time.perf_counter() - start
print(f"Python avg: {elapsed/n_runs*1000:.2f}ms")
"#;
    match run_python_script(py_script) {
        Ok(out) => println!("Python benchmark (CDC):\n{}", out),
        Err(e) => println!("Python benchmark skipped: {}", e),
    }
    println!("Rust CDC benchmark requires model download. Skipping automated run.");
}
