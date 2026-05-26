use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonMetricsOutput {
    success: bool,
    error: Option<String>,
    lpips: PythonLpipsOutput,
    aesthetic: PythonAestheticOutput,
    laplacian: PythonLaplacianOutput,
    psnr: PythonPsnrOutput,
}

#[derive(Debug, serde::Deserialize)]
struct PythonLpipsOutput {
    features: Vec<Vec<f32>>,
    differences: Vec<Vec<f32>>,
    cluster_labels: Vec<i32>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonAestheticOutput {
    score: f64,
}

#[derive(Debug, serde::Deserialize)]
struct PythonLaplacianOutput {
    score: f64,
}

#[derive(Debug, serde::Deserialize)]
struct PythonPsnrOutput {
    score: f64,
}

#[test]
#[ignore]
fn test_metrics_precision_and_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/nude_girl.png",
        "imgutils/test/testfile/pose/tohsaka_rin.png",
        "imgutils/test/testfile/hutao.jpg",
        "imgutils/test/testfile/mostima_post.jpg",
    ];

    println!("============================================================");
    println!(" Metrics (LPIPS / Aesthetic / Laplacian / PSNR) Benchmark");
    println!("============================================================");

    // Build image paths
    let mut image_paths = Vec::new();
    for rel_path in &test_images {
        let mut path = PathBuf::from(manifest_dir);
        path.push(rel_path);
        assert!(path.exists(), "Test image not found: {:?}", path);
        image_paths.push(path);
    }

    // Run Python reference
    println!(" Running Python reference...");
    let py_start = Instant::now();
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_metrics.py");
    for path in &image_paths {
        cmd.arg(path);
    }
    let output = cmd.output().expect("Failed to run Python via uv");
    let py_duration = py_start.elapsed();
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        panic!(
            "Python error:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }
    let py_output: PythonMetricsOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python metrics failed: {:?}",
        py_output.error
    );
    println!(" Python done ({:?})", py_duration);

    // Load images for Rust
    let mut images = Vec::new();
    for path in &image_paths {
        let img = image::open(path).expect("Failed to open image in Rust");
        images.push(img);
    }

    // 1. LPIPS feature extraction
    println!(" Rust LPIPS feature extraction...");
    let rust_start = Instant::now();
    let rust_lpips_feats: Vec<_> = images
        .iter()
        .map(|img| dghs_imgutils_rs::metrics::lpips_extract_feature(img))
        .collect::<Result<Vec<_>, _>>()
        .expect("LPIPS feature extraction failed");
    let rust_lpips_feat_dur = rust_start.elapsed();

    // Compare LPIPS features (flatten first for comparison)
    let py_feats_flat: Vec<Vec<f32>> = py_output
        .lpips
        .features
        .iter()
        .map(|f| {
            // Python: flattened feature array per image
            f.clone()
        })
        .collect();

    for i in 0..images.len().min(py_feats_flat.len()) {
        let (ref f0, ref f1, ref f2, ref f3, ref f4) = rust_lpips_feats[i];
        let rust_flat: Vec<f32> = f0
            .iter()
            .chain(f1.iter())
            .chain(f2.iter())
            .chain(f3.iter())
            .chain(f4.iter())
            .copied()
            .collect();
        let py_flat = &py_feats_flat[i];
        let min_len = rust_flat.len().min(py_flat.len());
        let mut max_err = 0.0f32;
        for j in 0..min_len {
            let diff = (rust_flat[j] - py_flat[j]).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
        println!("  Image {} LPIPS feature max error: {:.6}", i, max_err);
        assert!(
            max_err < 0.05,
            "LPIPS feature max error for image {} too high: {}",
            i,
            max_err
        );
    }
    println!(" LPIPS feature extraction done ({:?})", rust_lpips_feat_dur);

    // 2. LPIPS differences
    println!(" Rust LPIPS pairwise differences...");
    let n = images.len();
    let mut rust_lpips_diffs = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = dghs_imgutils_rs::metrics::lpips_difference(
                &rust_lpips_feats[i],
                &rust_lpips_feats[j],
            )
            .expect("LPIPS difference failed");
            rust_lpips_diffs[i][j] = d;
            rust_lpips_diffs[j][i] = d;
        }
    }
    let py_diffs = &py_output.lpips.differences;
    let mut diff_max_err = 0.0f32;
    for i in 0..n.min(py_diffs.len()) {
        for j in 0..n.min(py_diffs[i].len()) {
            let diff = (rust_lpips_diffs[i][j] - py_diffs[i][j]).abs();
            if diff > diff_max_err {
                diff_max_err = diff;
            }
            assert!(
                diff < 0.03,
                "LPIPS diff [{},{}] error too high: Rust={}, Python={}",
                i,
                j,
                rust_lpips_diffs[i][j],
                py_diffs[i][j]
            );
        }
    }
    println!("  LPIPS diff max error: {:.6}", diff_max_err);
    println!(" LPIPS differences done");

    // 3. LPIPS clustering
    println!(" Rust LPIPS clustering...");
    let rust_lpips_clusters = dghs_imgutils_rs::metrics::lpips_clustering(&images, None)
        .expect("LPIPS clustering failed");
    let py_lpips_clusters = &py_output.lpips.cluster_labels;
    assert_eq!(rust_lpips_clusters.len(), py_lpips_clusters.len());
    for i in 0..rust_lpips_clusters.len() {
        assert_eq!(
            rust_lpips_clusters[i], py_lpips_clusters[i],
            "LPIPS cluster mismatch at {}: Rust={}, Python={}",
            i, rust_lpips_clusters[i], py_lpips_clusters[i]
        );
    }
    println!(" LPIPS clustering done (100% match)");

    // 4. Aesthetic score
    println!(" Rust aesthetic scoring...");
    let rust_aesthetic = dghs_imgutils_rs::metrics::anime_dbaesthetic(&images[0], None)
        .expect("Aesthetic scoring failed");
    let py_aesthetic = py_output.aesthetic.score as f32;
    let aesthetic_err = (rust_aesthetic - py_aesthetic).abs();
    println!(
        "  Aesthetic score: Rust={:.6}, Python={:.6}, err={:.6}",
        rust_aesthetic, py_aesthetic, aesthetic_err
    );
    assert!(
        aesthetic_err < 0.1,
        "Aesthetic score error too high: {}",
        aesthetic_err
    );
    println!(" Aesthetic scoring done");

    // 5. Laplacian score
    println!(" Rust Laplacian scoring...");
    let rust_laplacian = dghs_imgutils_rs::metrics::laplacian_score(&images[0]);
    let py_laplacian = py_output.laplacian.score as f32;
    let lap_err = (rust_laplacian - py_laplacian).abs();
    println!(
        "  Laplacian score: Rust={:.6}, Python={:.6}, err={:.6}",
        rust_laplacian, py_laplacian, lap_err
    );
    assert!(
        lap_err < 50.0,
        "Laplacian score error too high: {}",
        lap_err
    );
    println!(" Laplacian scoring done");

    // 6. PSNR
    println!(" Rust PSNR...");
    let rust_psnr = dghs_imgutils_rs::metrics::psnr(&images[0], &images[1]);
    let py_psnr = py_output.psnr.score;
    let psnr_err = (rust_psnr - py_psnr).abs();
    println!(
        "  PSNR: Rust={:.6}, Python={:.6}, err={:.6}",
        rust_psnr, py_psnr, psnr_err
    );
    assert!(psnr_err < 1.0, "PSNR error too high: {}", psnr_err);
    println!(" PSNR done");

    println!("\n============================================================");
    println!(" All metrics tests passed!");
    println!(
        " Python time: {:?}, Rust feature extraction: {:?}",
        py_duration, rust_lpips_feat_dur
    );
    println!("============================================================");
}
