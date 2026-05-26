use std::path::PathBuf;
use std::time::Instant;

/// Helper: run Python's ascii_drawing and return its output.
fn python_ascii(image_path: &str, levels: &str, width: u32) -> Option<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir(PathBuf::from(manifest_dir).join("imgutils"))
        .arg("run")
        .arg("python")
        .arg("-c")
        .arg(format!(
            "import sys; sys.path.insert(0, '.'); \
         from imgutils.ascii import ascii_drawing; \
         result = ascii_drawing('{}', levels='{}', width={}); \
         print(result, end='')",
            image_path, levels, width
        ));

    let output = cmd.output().ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout).ok()
    } else {
        None
    }
}

#[test]
#[ignore]
fn test_ascii_benchmark_and_compare() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/hutao.jpg",
    ];

    let levels = "@%#*+=-:. ";
    let widths = [40, 80];

    println!("============================================================");
    println!("ASCII Drawing / Resource Benchmark");
    println!("============================================================");

    for rel_path in &test_images {
        let mut image_path = PathBuf::from(manifest_dir);
        image_path.push(rel_path);
        if !image_path.exists() {
            eprintln!("SKIP (image not found): {rel_path}");
            continue;
        }
        let abs = image_path.to_string_lossy().to_string();

        for &width in &widths {
            // Load image once
            let img = image::open(&abs).expect("Failed to open image");

            // Rust ASCII
            let start = Instant::now();
            let rust_result = dghs_imgutils_rs::ascii::ascii_drawing(&img, levels, Some(width))
                .expect("Rust ASCII failed");
            let rust_dur = start.elapsed();

            println!("\n[{rel_path}] width={width}");
            println!("  Rust: {:?} ({} chars)", rust_dur, rust_result.len());

            // Python ASCII (optional, requires uv + imgutils)
            if let Some(py_result) = python_ascii(&abs, levels, width) {
                let py_lines: Vec<&str> = py_result.lines().collect();
                let rust_lines: Vec<&str> = rust_result.lines().collect();

                let match_count = rust_lines
                    .iter()
                    .zip(&py_lines)
                    .filter(|(r, p)| *r == *p)
                    .count();
                let total = rust_lines.len().min(py_lines.len());
                let ratio = if total > 0 {
                    match_count as f64 / total as f64
                } else {
                    0.0
                };

                println!(
                    "  Python vs Rust match: {match_count}/{total} lines ({:.1}%)",
                    ratio * 100.0
                );

                // Output samples
                println!("\n  Python output (first 3 lines):");
                for line in py_lines.iter().take(3) {
                    println!("    |{line}|");
                }
                println!("  Rust output (first 3 lines):");
                for line in rust_lines.iter().take(3) {
                    println!("    |{line}|");
                }
            } else {
                println!("  Python: SKIP (uv/imgutils not available)");
            }
        }
    }

    println!("\n============================================================");
    println!("Benchmark complete");
    println!("============================================================");
}

#[test]
#[ignore]
fn test_resource_image_access() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // Test listing available bg images
    println!("Testing list_bg_images()...");
    let start = Instant::now();
    let images = dghs_imgutils_rs::resource::list_bg_images();
    let list_dur = start.elapsed();
    println!("  {} images listed in {:?}", images.len(), list_dur);

    if !images.is_empty() {
        // Show first few
        for img in images.iter().take(5) {
            println!("  - {img}");
        }
        println!("  ... (showing first 5)");

        // Try loading a specific image (if available via cache)
        let first = &images[0];
        println!("  Trying to load: {first}");
        let start = Instant::now();
        match dghs_imgutils_rs::resource::get_bg_image(first) {
            Ok(_img) => {
                let load_dur = start.elapsed();
                println!("  Loaded in {:?}", load_dur);
            }
            Err(e) => {
                println!("  Failed to load: {e}");
            }
        }
    }
}

#[test]
#[ignore]
fn test_ascii_drawing_performance() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push("imgutils/test/testfile/skadi.jpg");
    if !image_path.exists() {
        eprintln!("SKIP: test image not found");
        return;
    }

    let img = image::open(&image_path).expect("Failed to open image");
    let levels = "@%#*+=-:. ";

    println!("ASCII Performance (skadi.jpg):");
    for &width in &[40, 80, 160, 320] {
        let start = Instant::now();
        let result = dghs_imgutils_rs::ascii::ascii_drawing(&img, levels, Some(width));
        let dur = start.elapsed();
        match result {
            Ok(aa) => {
                let lines = aa.lines().count();
                println!(
                    "  width={width:>3}: {:?} ({}x{} = {} chars)",
                    dur,
                    width,
                    lines,
                    aa.len()
                );
            }
            Err(e) => {
                println!("  width={width:>3}: FAILED - {e}");
            }
        }
    }
}
