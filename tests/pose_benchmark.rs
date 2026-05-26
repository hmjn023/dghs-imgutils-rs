use dghs_imgutils_rs::pose::dwpose_estimate;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonPoseKP {
    x: f32,
    y: f32,
    score: f32,
}

#[derive(Debug, serde::Deserialize)]
struct PythonPoseResult {
    num_keypoints: usize,
    keypoints: Vec<PythonPoseKP>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonPoseOutput {
    success: bool,
    error: Option<String>,
    num_persons: usize,
    results: Vec<PythonPoseResult>,
    num_keypoints: usize,
}

fn run_pose_precision_test(test_image_rel_path: &str) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push(test_image_rel_path);

    assert!(
        image_path.exists(),
        "Test image not found: {:?}",
        image_path
    );

    println!("\n  Pose estimation test, image: {}", test_image_rel_path);
    println!("------------------------------------------------------------");

    // 1. Python side
    let py_start = Instant::now();
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_pose.py")
        .arg(&image_path);

    let output = cmd.output().expect("Failed to execute python via uv");
    let py_duration = py_start.elapsed();

    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Python side failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }

    let py_output: PythonPoseOutput =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout))
            .expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python inference failed: {:?}",
        py_output.error
    );
    println!("  Python inference done ({:?})", py_duration);

    // 2. Rust side
    let image = image::open(&image_path).expect("Failed to open test image in Rust");
    let rust_start = Instant::now();
    let rust_results =
        dwpose_estimate(&image, true, None, None).expect("Rust pose estimation failed");
    let rust_duration = rust_start.elapsed();
    println!("  Rust inference done   ({:?})", rust_duration);

    println!(
        "  Detection count - Python: {} persons / Rust: {} persons",
        py_output.num_persons,
        rust_results.len()
    );
    println!(
        "  Keypoints per person - Python: {} / Rust: {}",
        py_output.num_keypoints,
        rust_results.first().map(|k| k.keypoints.len()).unwrap_or(0)
    );

    // Compare first person's keypoints if both return results
    if !py_output.results.is_empty() && !rust_results.is_empty() {
        let py_kps = &py_output.results[0].keypoints;
        let rust_kps = &rust_results[0].keypoints;
        let n = py_kps.len().min(rust_kps.len());
        let mut total_pos_error = 0.0f32;
        let mut total_score_error = 0.0f32;
        let mut max_pos_error = 0.0f32;
        let mut valid_count = 0;

        for i in 0..n {
            let dx = (rust_kps[i].x - py_kps[i].x).abs();
            let dy = (rust_kps[i].y - py_kps[i].y).abs();
            let pos_err = (dx * dx + dy * dy).sqrt();
            total_pos_error += pos_err;
            max_pos_error = max_pos_error.max(pos_err);
            total_score_error += (rust_kps[i].score - py_kps[i].score).abs();
            valid_count += 1;
        }

        if valid_count > 0 {
            let mean_pos_error = total_pos_error / valid_count as f32;
            let mean_score_error = total_score_error / valid_count as f32;
            println!(
                "  Position MAE: {:.4} px, Max pos err: {:.4} px, Score MAE: {:.6}",
                mean_pos_error, max_pos_error, mean_score_error
            );

            assert!(
                mean_pos_error <= 10.0,
                "Keypoint position MAE exceeds 10px: {:.4}",
                mean_pos_error
            );
            assert!(
                mean_score_error <= 0.15,
                "Keypoint score MAE exceeds 0.15: {:.6}",
                mean_score_error
            );
        }
    }

    let speedup = py_duration.as_secs_f64() / rust_duration.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
    println!("  PASS");
}

#[test]
#[ignore]
fn test_pose_precision_and_benchmark() {
    println!("============================================================");
    println!("  DWpose Precision Benchmark");
    println!("============================================================");

    run_pose_precision_test("imgutils/test/testfile/two_bikini_girls.png");
    run_pose_precision_test("imgutils/test/testfile/nude_girl.png");
    run_pose_precision_test("imgutils/test/testfile/skadi.jpg");

    println!("\n============================================================");
    println!("  All pose tests passed!");
    println!("============================================================");
}
