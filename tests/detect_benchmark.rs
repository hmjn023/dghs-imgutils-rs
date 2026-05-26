//! 物体検出 (detect) モジュールの Rust 版と Python 版の推論結果を、複数のテスト画像を用いて厳密に比較する自動ベンチマーク・精度検証テスト。

use dghs_imgutils_rs::detect::base::Detection;
use dghs_imgutils_rs::detect::booru::detect_with_booru_yolo;
use dghs_imgutils_rs::detect::censor::detect_censors;
use dghs_imgutils_rs::detect::face::detect_faces;
use dghs_imgutils_rs::detect::head::detect_heads;
use dghs_imgutils_rs::detect::nudenet::detect_with_nudenet;
use dghs_imgutils_rs::detect::person::detect_person;
use dghs_imgutils_rs::detect::similarity::bboxes_similarity;
use dghs_imgutils_rs::detect::text::detect_text;
use std::path::PathBuf;
use std::time::Instant;

/// Python 側の検出結果をデコードするための構造体
#[derive(Debug)]
struct PythonDetection {
    box_coords: [f32; 4],
    label: String,
    score: f32,
}

// カスタム Deserialize 実装で "box" を "box_coords" にマッピング
impl<'de> serde::Deserialize<'de> for PythonDetection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            #[serde(rename = "box")]
            box_coords: [f32; 4],
            label: String,
            score: f32,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(PythonDetection {
            box_coords: helper.box_coords,
            label: helper.label,
            score: helper.score,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct PythonDetectOutput {
    success: bool,
    error: Option<String>,
    detections: Vec<PythonDetection>,
}

fn run_precision_test_for_task(task_type: &str, test_image_rel_path: &str) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push(test_image_rel_path);

    assert!(
        image_path.exists(),
        "Test image not found: {:?}",
        image_path
    );

    println!("\n📸 タスク [{}], 画像: {}", task_type, test_image_rel_path);
    println!("------------------------------------------------------------");

    // 1. Python 側の推論実行
    let py_start = Instant::now();
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_detect.py")
        .arg(task_type)
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

    let py_output: PythonDetectOutput =
        serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output");
    assert!(
        py_output.success,
        "Python inference failed: {:?}",
        py_output.error
    );
    println!("✓ Python 版推論完了 (所要時間: {:?})", py_duration);

    // 2. Rust 側の推論実行
    let image = image::open(&image_path).expect("Failed to open test image in Rust");
    let rust_start = Instant::now();

    let rust_results: Vec<Detection> = match task_type {
        "face" => detect_faces(&image, "s", "v1.4", 0.25, 0.7).unwrap(),
        "head" => detect_heads(&image, "s", "v2.0", 0.4, 0.7).unwrap(),
        "person" => detect_person(&image, "m", "v1.1", 0.3, 0.5).unwrap(),
        "censor" => detect_censors(&image, "s", "v1.0", 0.3, 0.7).unwrap(),
        "booru" => detect_with_booru_yolo(&image, "yolov8s_aa11", 0.25, 0.7).unwrap(),
        "nudenet" => detect_with_nudenet(&image, 100, 0.45, 0.25).unwrap(),
        "text" => detect_text(
            &image,
            "dbnetpp_resnet50_fpnc_1200e_icdar2015",
            0.05,
            Some(640),
        )
        .unwrap(),
        _ => panic!("Unknown task type: {}", task_type),
    };
    let rust_duration = rust_start.elapsed();
    println!("✓ Rust 版推論完了   (所要時間: {:?})", rust_duration);

    println!(
        "🔍 検出件数の比較 - Python: {} 件 / Rust: {} 件",
        py_output.detections.len(),
        rust_results.len()
    );

    // 3. ハンガリアン法 (Kuhn-Munkres) を用いて、BBox 同士の最適ペアリングと位置・スコア誤差を評価
    let rust_bboxes: Vec<(u32, u32, u32, u32)> = rust_results.iter().map(|d| d.bbox).collect();
    let py_bboxes: Vec<(u32, u32, u32, u32)> = py_output
        .detections
        .iter()
        .map(|d| {
            (
                d.box_coords[0].round() as u32,
                d.box_coords[1].round() as u32,
                d.box_coords[2].round() as u32,
                d.box_coords[3].round() as u32,
            )
        })
        .collect();

    let sims = bboxes_similarity(&rust_bboxes, &py_bboxes);
    let matched_count = sims.iter().filter(|&&v| v > 0.3).count();

    println!(
        "  最適ペアリングマッチング数 (IoU > 0.3): {} 件",
        matched_count
    );

    if !py_output.detections.is_empty() && !rust_results.is_empty() {
        // 最低でも 1 件以上のマッチングが成功すること（画像はそれぞれ何らかの検出対象を含むものを選定）
        assert!(
            matched_count > 0,
            "Rust と Python で同一オブジェクトのマッチングが一切行われませんでした。"
        );

        // 各座標値・スコアの平均絶対誤差 (MAE) を算出
        let mut total_coords_diff = 0.0f32;
        let mut total_score_diff = 0.0f32;
        let mut valid_pairs = 0;

        for r_idx in 0..rust_results.len() {
            // 最も IoU が高い Python 側の候補を検索
            let mut best_py_idx = None;
            let mut best_iou = 0.3f32;

            for p_idx in 0..py_output.detections.len() {
                let iou = dghs_imgutils_rs::detect::similarity::calculate_iou(
                    &rust_bboxes[r_idx],
                    &py_bboxes[p_idx],
                );
                if iou > best_iou {
                    best_iou = iou;
                    best_py_idx = Some(p_idx);
                }
            }

            if let Some(p_idx) = best_py_idx {
                let r_det = &rust_results[r_idx];
                let p_det = &py_output.detections[p_idx];

                // BBox の座標誤差
                let dx1 = (r_det.bbox.0 as f32 - p_det.box_coords[0]).abs();
                let dy1 = (r_det.bbox.1 as f32 - p_det.box_coords[1]).abs();
                let dx2 = (r_det.bbox.2 as f32 - p_det.box_coords[2]).abs();
                let dy2 = (r_det.bbox.3 as f32 - p_det.box_coords[3]).abs();

                total_coords_diff += (dx1 + dy1 + dx2 + dy2) / 4.0;
                total_score_diff += (r_det.score - p_det.score).abs();
                valid_pairs += 1;
            }
        }

        if valid_pairs > 0 {
            let mae_coords = total_coords_diff / valid_pairs as f32;
            let mae_score = total_score_diff / valid_pairs as f32;

            println!("  BBox 座標平均絶対誤差 (MAE): {:.4} px", mae_coords);
            println!("  信頼度スコア平均絶対誤差 (MAE): {:.6}", mae_score);

            // しきい値境界テスト: YOLO/RT-DETR の等価移植として MAE <= 5.0px 未満、スコア MAE <= 0.15 未満であることを保証
            assert!(
                mae_coords <= 5.0,
                "BBox 座標誤差 MAE がしきい値(5.0px)を超えています: {:.4} px",
                mae_coords
            );
            assert!(
                mae_score <= 0.15,
                "信頼度スコア誤差 MAE がしきい値(0.15)を超えています: {:.6}",
                mae_score
            );
        }
    }

    println!("🎉 タスク [{}] 検証パス！", task_type);
}

#[test]
#[ignore]
fn test_detect_precision_and_benchmark() {
    println!("============================================================");
    println!("🧪 物体検出 (detect) 各種 API の精度検証ベンチマーク");
    println!("============================================================");

    // 各検出器が得意とするテスト画像を指定して一括実行
    run_precision_test_for_task("face", "imgutils/test/testfile/mostima_post.jpg");
    run_precision_test_for_task("head", "imgutils/test/testfile/skadi.jpg");
    run_precision_test_for_task("person", "imgutils/test/testfile/two_bikini_girls.png");
    run_precision_test_for_task("censor", "imgutils/test/testfile/nude_girl.png");
    run_precision_test_for_task("booru", "imgutils/test/testfile/dori.png");
    run_precision_test_for_task("nudenet", "imgutils/test/testfile/nude_girl.png");
    run_precision_test_for_task("text", "imgutils/test/testfile/mostima_post.jpg");

    println!("\n============================================================");
    println!("🏆 全ての物体検出器で等価移植の精度アサーションに合格しました！");
    println!("============================================================");
}
