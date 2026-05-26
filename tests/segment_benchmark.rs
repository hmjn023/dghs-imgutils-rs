//! 領域分割 (segment) モジュールの Rust 版と Python 版の前景抽出マスクを、複数のテスト画像を用いて厳密に比較する自動精度検証テスト。

use dghs_imgutils_rs::segment::isnetis::get_isnetis_mask;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonStats {
    mean: f32,
    min: f32,
    max: f32,
    shape: Vec<i32>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonSegmentOutput {
    success: bool,
    error: Option<String>,
    mask_grid: Vec<Vec<f32>>,
    grid_step: usize,
    stats: PythonStats,
}

fn run_segment_precision_test(test_image_rel_path: &str) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push(test_image_rel_path);

    assert!(image_path.exists(), "Test image not found: {:?}", image_path);

    println!("\n📸 領域分割検証 画像: {}", test_image_rel_path);
    println!("------------------------------------------------------------");

    // 1. Python 側の推論実行 (グリッドサンプリングされたマスクと統計値を取得)
    let py_start = Instant::now();
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_segment.py")
        .arg(&image_path);

    let output = cmd.output().expect("Failed to execute python via uv");
    let py_duration = py_start.elapsed();

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        panic!(
            "Python side failed with error:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }

    let py_output: PythonSegmentOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python segment failed: {:?}",
        py_output.error
    );
    println!("✓ Python 版前景マスク抽出完了 (所要時間: {:?})", py_duration);

    // 2. Rust 側の推論実行 ( get_isnetis_mask )
    let image = image::open(&image_path).expect("Failed to open test image in Rust");
    let rust_start = Instant::now();
    let rust_mask = get_isnetis_mask(&image, 1024).expect("Rust get_isnetis_mask failed");
    let rust_duration = rust_start.elapsed();
    println!("✓ Rust 版前景マスク抽出完了   (所要時間: {:?})", rust_duration);

    let h = rust_mask.shape()[0];
    let w = rust_mask.shape()[1];

    println!(
        "🔍 マスク形状の比較 - Python: {:?} / Rust: {:?}",
        py_output.stats.shape,
        vec![h as i32, w as i32]
    );

    assert_eq!(
        py_output.stats.shape[0] as usize, h,
        "Mask height mismatch"
    );
    assert_eq!(
        py_output.stats.shape[1] as usize, w,
        "Mask width mismatch"
    );

    // 3. グリッドダウンサンプリングによる精密ピクセル比較
    let grid_step = py_output.grid_step;
    let mut total_absolute_error = 0.0f32;
    let mut grid_points_count = 0;

    let py_grid = &py_output.mask_grid;
    for (py_r, row) in py_grid.iter().enumerate() {
        let rust_r = py_r * grid_step;
        if rust_r >= h {
            continue;
        }
        for (py_c, &py_val) in row.iter().enumerate() {
            let rust_c = py_c * grid_step;
            if rust_c >= w {
                continue;
            }

            let rust_val = rust_mask[[rust_r, rust_c]];
            let diff = (rust_val - py_val).abs();
            total_absolute_error += diff;
            grid_points_count += 1;
        }
    }

    let grid_mae = if grid_points_count > 0 {
        total_absolute_error / grid_points_count as f32
    } else {
        0.0f32
    };

    // マスク全体の平均値
    let mut rust_sum = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            rust_sum += rust_mask[[y, x]] as f64;
        }
    }
    let rust_mean = (rust_sum / (h * w) as f64) as f32;

    let mean_diff = (rust_mean - py_output.stats.mean).abs();

    println!("  グリッド比較点数             : {} 点", grid_points_count);
    println!("  グリッド平均絶対誤差 (MAE)   : {:.8}", grid_mae);
    println!("  全体マスク平均値 - Python    : {:.8}", py_output.stats.mean);
    println!("  全体マスク平均値 - Rust      : {:.8}", rust_mean);
    println!("  全体平均値の差               : {:.8}", mean_diff);

    // 等価移植のしきい値アサーション: ダウンサンプリングMAE <= 0.02（2.0%）以下であることを保証
    assert!(
        grid_mae <= 0.02,
        "グリッドマスク誤差 MAE がしきい値(0.02)を超えています: {:.8}",
        grid_mae
    );
    assert!(
        mean_diff <= 0.02,
        "全体平均値の差がしきい値(0.02)を超えています: {:.8}",
        mean_diff
    );

    println!("🎉 領域分割画像 [{}] 検証パス！", test_image_rel_path);
}

#[test]
#[ignore]
fn test_segment_precision_and_benchmark() {
    println!("============================================================");
    println!("🧪 領域分割 (segment) ISNetIS の精度検証ベンチマーク");
    println!("============================================================");

    // 代表的なキャラクター画像を検証
    run_segment_precision_test("imgutils/test/testfile/skadi.jpg");
    run_segment_precision_test("imgutils/test/testfile/hutao.jpg");
    run_segment_precision_test("imgutils/test/testfile/nian.png");

    println!("\n============================================================");
    println!("🏆 全てのテスト画像で前景マスク領域分割の精度検証に合格しました！");
    println!("============================================================");
}
