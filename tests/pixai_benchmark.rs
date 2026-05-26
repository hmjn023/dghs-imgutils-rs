//! PixAI Tagger の Rust 版と Python 版の推論結果を、複数のテスト画像を用いて厳密に比較する自動ベンチマーク・精度検証テスト。

use dghs_imgutils_rs::tagging::pixai::get_pixai_tags;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Python 版の実行結果をデコードするための構造体
#[derive(Debug, serde::Deserialize)]
struct PythonTaggerOutput {
    success: bool,
    error: Option<String>,
    general: HashMap<String, f32>,
    character: HashMap<String, f32>,
    prediction: Vec<f32>,
}

/// 複数のテスト画像に対して Python と Rust の両方で推論を走らせ、誤差をループで一括検証します。
///
/// 実際に HuggingFace モデルの取得と Python プロセスの呼び出しを伴うため、
/// デフォルトではスキップされます。実行するには `cargo test --test pixai_benchmark -- --ignored` を実行します。
#[test]
#[ignore]
fn test_pixai_precision_and_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // 精度検証に使用するテスト画像のリスト（イラストの特徴が異なる複数の画像を定義）
    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg", // 1girl, 白髪, 赤目, スカディ
        "imgutils/test/testfile/nude_girl.png", // 1girl, ヌード
        "imgutils/test/testfile/two_bikini_girls.png", // 2girls, 水着, ビキニ
        "imgutils/test/testfile/pose/tohsaka_rin.png", // 1girl, 遠坂凛 (極端なアスペクト比)
        "imgutils/test/testfile/hutao.jpg", // 1girl, 原神の胡桃
        "imgutils/test/testfile/nian.png",  // 1girl, アークナイツのニェン
        "imgutils/test/testfile/dori.png",  // 1girl, 原神のドリー
        "imgutils/test/testfile/mostima_post.jpg", // 1girl, アークナイツのモスティマ
        "imgutils/test/testfile/genshin_post.jpg", // 複数キャラ, 原神
    ];

    println!("============================================================");
    println!("🧪 複数画像による PixAI Tagger 精度検証・ベンチマーク");
    println!("============================================================");

    for (idx, rel_path) in test_images.iter().enumerate() {
        let mut image_path = PathBuf::from(manifest_dir);
        image_path.push(rel_path);

        assert!(
            image_path.exists(),
            "テスト画像が見つかりません: {:?}",
            image_path
        );

        println!(
            "\n📸 [{}/{}] 画像検証: {}",
            idx + 1,
            test_images.len(),
            rel_path
        );
        println!("------------------------------------------------------------");

        // 1. Python 側の推論実行 (補助スクリプトの呼び出し)
        let py_start = Instant::now();

        let mut cmd = std::process::Command::new("uv");
        cmd.current_dir("imgutils")
            .arg("run")
            .arg("python")
            .arg("../benchmark/run_python_tagger.py")
            .arg(&image_path);

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

        let py_output: PythonTaggerOutput =
            serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");

        assert!(
            py_output.success,
            "Python 側でのタグ予測が失敗しました: {:?}",
            py_output.error
        );
        println!("✓ Python 版推論完了 (所要時間: {:?})", py_duration);

        // 2. Rust 側の推論実行
        let image = image::open(&image_path).expect("Failed to open test image in Rust");

        let rust_start = Instant::now();
        let rust_result =
            get_pixai_tags(&image, "v0.9", None).expect("Rust 側でのタグ予測に失敗しました");
        let rust_duration = rust_start.elapsed();
        println!("✓ Rust 版推論完了   (所要時間: {:?})", rust_duration);

        // 3. 予測確率配列のサイズチェック (13461 次元)
        assert_eq!(
            py_output.prediction.len(),
            13461,
            "Python prediction array size mismatch"
        );

        // 4. 統計の比較
        println!("🔍 予測結果の統計比較");
        println!(
            "  Python 版検出: general {}件 / character {}件",
            py_output.general.len(),
            py_output.character.len()
        );
        println!(
            "  Rust 版検出  : general {}件 / character {}件",
            rust_result.general.len(),
            rust_result.character.len()
        );

        // 5. 検出されたタグとその確信度スコアの誤差検証

        // 5. 検出されたタグ集合の IoU (一致率) を計算・アサーション
        // しきい値境界上のタグ（僅かなスコアズレで検出・未検出が分かれるタグ）による件数不一致を安全に許容します。
        let rust_keys: std::collections::HashSet<&String> =
            rust_result.general.iter().map(|(k, _)| k).collect();
        let py_keys: std::collections::HashSet<&String> = py_output.general.keys().collect();

        let intersection: std::collections::HashSet<_> =
            rust_keys.intersection(&py_keys).cloned().collect();
        let union: std::collections::HashSet<_> = rust_keys.union(&py_keys).cloned().collect();
        let tag_iou = intersection.len() as f32 / union.len() as f32;

        println!(
            "  一般タグ検出 IoU : {:.2}% (共通: {}件 / 和集合: {}件)",
            tag_iou * 100.0,
            intersection.len(),
            union.len()
        );
        assert!(
            tag_iou >= 0.90,
            "一般タグの検出一致率 (IoU) が 90% 未満です: {:.2}%",
            tag_iou * 100.0
        );

        // 6. 検出されたタグとその確信度スコアの誤差検証
        let mut total_absolute_error: f32 = 0.0;
        let mut max_absolute_error: f32 = 0.0;
        let mut compared_count = 0;
        let mut exceeded_errors = Vec::new();

        // general タグの比較
        for (tag, rust_score) in &rust_result.general {
            if let Some(py_score) = py_output.general.get(tag) {
                let diff = (rust_score - py_score).abs();
                total_absolute_error += diff;
                max_absolute_error = max_absolute_error.max(diff);
                compared_count += 1;

                if diff >= 0.04 {
                    exceeded_errors.push((tag.clone(), *rust_score, *py_score, diff));
                }
            }
        }

        // character タグの比較
        for (tag, rust_score) in &rust_result.character {
            if let Some(py_score) = py_output.character.get(tag) {
                let diff = (rust_score - py_score).abs();
                total_absolute_error += diff;
                max_absolute_error = max_absolute_error.max(diff);
                compared_count += 1;

                if diff >= 0.04 {
                    exceeded_errors.push((tag.clone(), *rust_score, *py_score, diff));
                }
            }
        }

        if !exceeded_errors.is_empty() {
            println!(
                "  ⚠️ 許容誤差(4%)を超えたタグが {} 件存在します:",
                exceeded_errors.len()
            );
            for (tag, rust_s, py_s, diff) in &exceeded_errors {
                println!(
                    "    - '{}': Rust={:.4}, Python={:.4}, 差={:.4}",
                    tag, rust_s, py_s, diff
                );
            }
        }

        let exceeded_ratio = if compared_count > 0 {
            exceeded_errors.len() as f32 / compared_count as f32
        } else {
            0.0
        };

        // 誤差が 4% を超えるタグの割合が全体の 10% 未満であること、かつ最大絶対誤差が 18% 未満であることを検証
        assert!(
            exceeded_ratio < 0.10,
            "誤差が4%を超えるタグの割合が多すぎます: {:.2}%",
            exceeded_ratio * 100.0
        );
        assert!(
            max_absolute_error < 0.18,
            "最大絶対誤差が許容限界(18%)を超えています: {:.4}",
            max_absolute_error
        );

        let mae = if compared_count > 0 {
            total_absolute_error / compared_count as f32
        } else {
            0.0
        };

        println!("🎉 検証パス！");
        println!("  比較したタグ件数 : {} 件", compared_count);
        println!("  平均絶対誤差(MAE): {:.8}", mae);
        println!("  最大絶対誤差      : {:.8}", max_absolute_error);
        println!(
            "  推論の速度向上   : {:.2}x",
            py_duration.as_secs_f64() / rust_duration.as_secs_f64()
        );
        println!("------------------------------------------------------------");
    }

    println!("\n============================================================");
    println!("🏆 すべてのテスト画像で精度検証ベンチマークに合格しました！");
    println!("============================================================");
}
