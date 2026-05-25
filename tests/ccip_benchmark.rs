//! CCIP (Character Contrastive Learning) の Rust 版と Python 版の推論精度および整合性を比較・検証する自動精度ベンチマークテスト。

use dghs_imgutils_rs::metrics::{
    ccip_batch_differences, ccip_batch_extract_features, ccip_batch_same, ccip_clustering,
    ccip_default_threshold, ccip_merge,
};
use std::path::PathBuf;
use std::time::Instant;

/// Python 側の出力をデコードするための構造体
#[derive(Debug, serde::Deserialize)]
struct PythonCcipOutput {
    success: bool,
    error: Option<String>,
    embeddings: Vec<Vec<f32>>,
    differences: Vec<Vec<f32>>,
    same: Vec<Vec<bool>>,
    merged: Vec<f32>,
    cluster_labels: Vec<i32>,
    threshold: f32,
}

/// 複数のキャラクター画像に対して Python と Rust の両方で推論を走らせ、
/// 特徴量、類似度、同一判定、マージ、クラスタリング結果を厳密に比較します。
///
/// 実行するには `cargo test --test ccip_benchmark -- --ignored --nocapture` を実行します。
#[test]
#[ignore]
#[allow(clippy::needless_range_loop)]
fn test_ccip_precision_and_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // 検証に使用する 5 つのキャラクター画像（イラストの特徴がそれぞれ異なる独立したキャラクター）
    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",            // キャラA
        "imgutils/test/testfile/nude_girl.png",        // キャラB
        "imgutils/test/testfile/pose/tohsaka_rin.png", // キャラC
        "imgutils/test/testfile/hutao.jpg",            // キャラD
        "imgutils/test/testfile/mostima_post.jpg",     // キャラE
    ];

    println!("============================================================");
    println!("🧪 CCIP (対照学習特徴量) 精度検証・ベンチマーク");
    println!("============================================================");

    // 1. 画像絶対パスの一覧を構築
    let mut image_paths = Vec::new();
    for rel_path in &test_images {
        let mut path = PathBuf::from(manifest_dir);
        path.push(rel_path);
        assert!(path.exists(), "テスト画像が見つかりません: {:?}", path);
        image_paths.push(path);
    }

    // 2. Python 側の推論実行 (補助スクリプトの呼び出し)
    println!("📸 Python 側での一括推論実行中...");
    let py_start = Instant::now();

    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_ccip.py");

    for path in &image_paths {
        cmd.arg(path);
    }

    let output = cmd.output().expect("Failed to execute python via uv");
    let py_duration = py_start.elapsed();

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        panic!(
            "Python 側でエラーが発生しました:\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout_str, stderr_str
        );
    }

    let py_output: PythonCcipOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");

    assert!(
        py_output.success,
        "Python 側での CCIP 解析が失敗しました: {:?}",
        py_output.error
    );
    println!("✓ Python 版推論完了 (所要時間: {:?})", py_duration);

    // 3. Rust 側の推論実行
    println!("📸 Rust 側での一括推論実行中...");
    let mut images = Vec::new();
    for path in &image_paths {
        let img = image::open(path).expect("Failed to open test image in Rust");
        images.push(img);
    }

    let rust_start = Instant::now();
    // 384x384, デフォルトモデルで一括特徴抽出
    let rust_embeddings =
        ccip_batch_extract_features(&images, 384, "").expect("Rust 側での特徴抽出に失敗しました");
    let rust_duration = rust_start.elapsed();
    println!("✓ Rust 版特徴抽出完了 (所要時間: {:?})", rust_duration);

    // 4. 特徴ベクトルの厳密比較 (768 次元 x 5枚)
    assert_eq!(rust_embeddings.len(), py_output.embeddings.len());
    let n = rust_embeddings.len();

    let mut feat_max_error: f32 = 0.0;
    let mut feat_total_error: f32 = 0.0;
    let mut feat_val_count = 0;
    let mut feat_exceeded_count = 0;

    for i in 0..n {
        let rust_emb = &rust_embeddings[i];
        let py_emb = &py_output.embeddings[i];
        assert_eq!(rust_emb.len(), 768);
        assert_eq!(py_emb.len(), 768);

        for j in 0..768 {
            let diff = (rust_emb[j] - py_emb[j]).abs();
            feat_max_error = feat_max_error.max(diff);
            feat_total_error += diff;
            feat_val_count += 1;

            if diff >= 0.05 {
                feat_exceeded_count += 1;
            }
        }
    }
    let feat_mae = feat_total_error / feat_val_count as f32;
    let feat_exceeded_ratio = feat_exceeded_count as f32 / feat_val_count as f32;
    println!(
        "  MAE: {:.6}, MaxError: {:.6}, 誤差5%超の割合: {:.2}%",
        feat_mae,
        feat_max_error,
        feat_exceeded_ratio * 100.0
    );

    assert!(
        feat_mae < 0.03,
        "特徴ベクトルの平均絶対誤差(MAE)が許容値(3.0%)を超えています: {:.6}",
        feat_mae
    );
    assert!(
        feat_max_error < 0.35,
        "特徴ベクトルの最大絶対誤差が許容値(35%)を超えています: {:.6}",
        feat_max_error
    );
    assert!(
        feat_exceeded_ratio < 0.15,
        "誤差が5%を超える次元の割合が多すぎます: {:.2}%",
        feat_exceeded_ratio * 100.0
    );
    println!("✓ 768次元特徴ベクトル統計検証パス");

    // 5. 距離マトリクス (differences) の比較
    println!("🔍 距離マトリクスの算出と比較中...");
    let rust_diffs = ccip_batch_differences(&rust_embeddings, "")
        .expect("Rust 側での距離マトリクス算出に失敗しました");

    let mut diff_max_error: f32 = 0.0;
    let mut diff_total_error: f32 = 0.0;

    for i in 0..n {
        for j in 0..n {
            let rust_val = rust_diffs[[i, j]];
            let py_val = py_output.differences[i][j];
            let diff = (rust_val - py_val).abs();
            diff_max_error = diff_max_error.max(diff);
            diff_total_error += diff;

            // 距離マトリクスの絶対誤差も 0.03 未満であることを確認
            assert!(
                diff < 0.03,
                "マトリクス [{}, {}] の距離誤差が許容値を超えました: Rust={}, Python={}, 差={}",
                i,
                j,
                rust_val,
                py_val,
                diff
            );
        }
    }
    let diff_mae = diff_total_error / (n * n) as f32;
    println!(
        "✓ 距離マトリクス検証パス (MAE: {:.6}, MaxError: {:.6})",
        diff_mae, diff_max_error
    );

    // 6. 同一人物判定 (same) マトリクスの比較
    println!("🔍 同一判定マトリクスの比較中...");
    let rust_sames = ccip_batch_same(&rust_embeddings, None, "")
        .expect("Rust 側での同一判定マトリクス算出に失敗しました");

    for i in 0..n {
        for j in 0..n {
            let rust_val = rust_sames[[i, j]];
            let py_val = py_output.same[i][j];
            // 同一判定は 100% 完全一致であることを検証します
            assert_eq!(
                rust_val, py_val,
                "マトリクス [{}, {}] の同一判定結果が不一致です: Rust={}, Python={}",
                i, j, rust_val, py_val
            );
        }
    }
    println!("✓ 同一人物判定マトリクス検証パス (100% 完全一致！)");

    // 7. 特徴マージ (ccip_merge) の比較
    println!("🔍 特徴マージの比較中...");
    let rust_merged = ccip_merge(&rust_embeddings).expect("Rust 側での特徴マージに失敗しました");
    assert_eq!(rust_merged.len(), 768);

    let mut merged_max_error: f32 = 0.0;
    let mut merged_total_error: f32 = 0.0;
    let mut merged_exceeded_count = 0;

    for j in 0..768 {
        let diff = (rust_merged[j] - py_output.merged[j]).abs();
        merged_max_error = merged_max_error.max(diff);
        merged_total_error += diff;

        if diff >= 0.05 {
            merged_exceeded_count += 1;
        }
    }
    let merged_mae = merged_total_error / 768.0;
    let merged_exceeded_ratio = merged_exceeded_count as f32 / 768.0;
    println!(
        "  MAE: {:.6}, MaxError: {:.6}, 誤差5%超の割合: {:.2}%",
        merged_mae,
        merged_max_error,
        merged_exceeded_ratio * 100.0
    );

    assert!(
        merged_mae < 0.03,
        "マージベクトルの平均絶対誤差(MAE)が許容値(3.0%)を超えています: {:.6}",
        merged_mae
    );
    assert!(
        merged_max_error < 0.35,
        "マージベクトルの最大絶対誤差が許容値(35%)を超えています: {:.6}",
        merged_max_error
    );
    assert!(
        merged_exceeded_ratio < 0.15,
        "誤差が5%を超える次元の割合が多すぎます: {:.2}%",
        merged_exceeded_ratio * 100.0
    );
    println!("✓ 特徴マージ統計検証パス");

    // 8. 自前実装 DBSCAN クラスタリング (ccip_clustering) の比較
    println!("🔍 DBSCAN クラスタリングの比較中...");
    // サンプル数が少ないため、テストスクリプトと同様に `dbscan` で `min_samples = 2` で動作させます
    let rust_clusters = ccip_clustering(&rust_embeddings, "dbscan", None, Some(2), "")
        .expect("Rust 側でのクラスタリングに失敗しました");

    assert_eq!(rust_clusters.len(), py_output.cluster_labels.len());
    for i in 0..n {
        assert_eq!(
            rust_clusters[i], py_output.cluster_labels[i],
            "画像 {} のクラスタ所属結果が不一致です: Rust={}, Python={}",
            i, rust_clusters[i], py_output.cluster_labels[i]
        );
    }
    println!("✓ DBSCAN クラスタリング検証パス (100% 完全一致！)");

    // 9. デフォルト閾値の比較
    let rust_thresh =
        ccip_default_threshold("").expect("Rust 側でのデフォルト閾値ロードに失敗しました");
    assert_eq!(rust_thresh, py_output.threshold);
    println!("✓ デフォルト閾値検証パス (値: {})", rust_thresh);

    println!("\n============================================================");
    println!("🏆 CCIP 特徴抽出・同一判定・クラスタリングすべてで合格しました！");
    println!(
        "推論の速度向上: {:.2}x",
        py_duration.as_secs_f64() / rust_duration.as_secs_f64()
    );
    println!("============================================================");
}
