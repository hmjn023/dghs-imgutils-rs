#![allow(dead_code)]
//! Shared assertions for opt-in model tests; failures must reach libtest.
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use std::{path::PathBuf, time::Instant};

pub fn image() -> DynamicImage {
    if let Some(path) = std::env::var_os("DGHS_TEST_IMAGE") {
        return image::open(path).expect("DGHS_TEST_IMAGE must be a readable image");
    }
    DynamicImage::ImageRgb8(RgbImage::from_fn(128, 128, |x, y| {
        Rgb([(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8])
    }))
}
pub fn fixture(relative: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    assert!(
        path.is_file(),
        "required test fixture missing: {} (initialize the imgutils submodule)",
        path.display()
    );
    path
}
pub fn benchmark<T>(name: &str, runs: usize, mut run: impl FnMut() -> T) -> T {
    assert!(runs > 0);
    let _warmup = run();
    let start = Instant::now();
    let mut result = run();
    for _ in 1..runs {
        result = run();
    }
    println!(
        "{name}: {:.2} ms/run ({runs} runs, after warmup)",
        start.elapsed().as_secs_f64() * 1000.0 / runs as f64
    );
    result
}
pub fn embeddings(v: &[Vec<f32>], batch: usize) {
    assert_eq!(v.len(), batch, "embedding batch size");
    for row in v {
        assert!(!row.is_empty());
        assert!(row.iter().all(|x| x.is_finite()));
        assert!(row.iter().any(|x| *x != 0.0), "zero embedding");
    }
}
pub fn image_size(img: &DynamicImage, w: u32, h: u32) {
    assert_eq!(img.dimensions(), (w, h));
}
pub fn python(script: &str) -> String {
    let out = std::process::Command::new("uv")
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("imgutils"))
        .args(["run", "python", "-c", script])
        .output()
        .expect("Python reference requires uv and imgutils");
    assert!(
        out.status.success(),
        "Python reference failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("Python output must be UTF-8")
}
