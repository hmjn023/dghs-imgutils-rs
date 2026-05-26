//! 画像検証 (validate) モジュールの Rust 版と Python 版の推論結果を、テスト画像を用いて比較する自動精度検証テスト。

use dghs_imgutils_rs::validate::{
    anime_bangumi_char_score, anime_dbrating_score, anime_teen_score, safe_check_score,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct PythonValidateOutput {
    success: bool,
    error: Option<String>,
    safe: Option<HashMap<String, f64>>,
    dbrating: Option<HashMap<String, f64>>,
    bangumi_char: Option<HashMap<String, f64>>,
    teen: Option<HashMap<String, f64>>,
}

fn run_validate_precision_test(test_image_rel_path: &str) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push(test_image_rel_path);

    assert!(
        image_path.exists(),
        "Test image not found: {:?}",
        image_path
    );

    println!("\n📸 検証 画像: {}", test_image_rel_path);
    println!("------------------------------------------------------------");

    // 1. Python 推論
    let py_start = Instant::now();
    let cmd = std::process::Command::new("uv")
        .current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_validate.py")
        .arg(&image_path)
        .output()
        .expect("Failed to execute python via uv");
    let py_duration = py_start.elapsed();

    let stdout_str = String::from_utf8_lossy(&cmd.stdout);
    let stderr_str = String::from_utf8_lossy(&cmd.stderr);

    if !cmd.status.success() {
        panic!(
            "Python side failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }

    let py_output: PythonValidateOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python inference failed: {:?}",
        py_output.error
    );
    println!("✓ Python 版推論完了 (所要時間: {:?})", py_duration);

    // 2. Rust 推論
    let rust_start = Instant::now();
    let rust_safe = safe_check_score(&image_path.to_string_lossy(), None)
        .expect("Rust safe_check_score failed");
    let rust_dbrating = anime_dbrating_score(&image_path.to_string_lossy(), None)
        .expect("Rust anime_dbrating_score failed");
    let rust_bangumi = anime_bangumi_char_score(&image_path.to_string_lossy(), None)
        .expect("Rust anime_bangumi_char_score failed");
    let rust_teen = anime_teen_score(&image_path.to_string_lossy(), None)
        .expect("Rust anime_teen_score failed");
    let rust_duration = rust_start.elapsed();
    println!("✓ Rust 版推論完了 (所要時間: {:?})", rust_duration);

    // 3. スコア比較
    let tests: Vec<(&str, HashMap<String, f64>, HashMap<String, f32>)> = vec![
        (
            "safe",
            py_output.safe.clone().unwrap_or_default(),
            rust_safe,
        ),
        (
            "dbrating",
            py_output.dbrating.clone().unwrap_or_default(),
            rust_dbrating,
        ),
        (
            "bangumi_char",
            py_output.bangumi_char.clone().unwrap_or_default(),
            rust_bangumi,
        ),
        (
            "teen",
            py_output.teen.clone().unwrap_or_default(),
            rust_teen,
        ),
    ];

    for (name, py_scores, rust_scores) in &tests {
        println!("\n  --- {} ---", name);
        println!("  Python: {:?}", py_scores);
        println!("  Rust:   {:?}", rust_scores);

        let mut total_diff = 0.0f64;
        let mut count = 0;
        for (label, py_val) in py_scores {
            if let Some(rust_val) = rust_scores.get(label) {
                let diff = (py_val - *rust_val as f64).abs();
                total_diff += diff;
                count += 1;
            }
        }
        if count > 0 {
            let mae = total_diff / count as f64;
            println!("  MAE: {:.8}", mae);
            assert!(
                mae <= 0.05,
                "{} MAE {:.8} exceeds threshold 0.05",
                name,
                mae
            );
        }
    }

    println!("\n🎉 画像 [{}] 検証パス！", test_image_rel_path);
}

#[test]
#[ignore]
fn test_validate_precision_and_benchmark() {
    println!("============================================================");
    println!("🧪 画像検証 (validate) 各種 API の精度検証ベンチマーク");
    println!("============================================================");

    run_validate_precision_test("imgutils/test/testfile/skadi.jpg");

    println!("\n============================================================");
    println!("🏆 全ての検証器で精度アサーションに合格しました！");
    println!("============================================================");
}
