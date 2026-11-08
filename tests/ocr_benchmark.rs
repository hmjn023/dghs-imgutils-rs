//! Opt-in Python parity check. Dependency and inference failures are test failures.
mod support;
use dghs_imgutils_rs::ocr::ocr;

#[derive(Debug, serde::Deserialize)]
struct PythonOcrOutput {
    results: Vec<Vec<PythonOcrItem>>,
}
#[derive(Debug, serde::Deserialize)]
struct PythonOcrItem {
    bbox: [i32; 4],
    text: String,
    score: f32,
}

#[test]
#[ignore = "requires Python imgutils, OCR models and reference images"]
fn test_ocr_precision_and_benchmark() {
    let paths: Vec<_> = ["skadi.jpg", "nude_girl.png", "hutao.jpg"]
        .iter()
        .map(|name| support::fixture(&format!("imgutils/test/testfile/{name}")))
        .collect();
    let output = std::process::Command::new("uv")
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/imgutils"))
        .args(["run", "python", "../benchmark/run_python_ocr.py"])
        .args(&paths)
        .output()
        .expect("Python reference requires uv and imgutils");
    assert!(
        output.status.success(),
        "Python OCR failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: PythonOcrOutput =
        serde_json::from_slice(&output.stdout).expect("invalid reference JSON");
    assert_eq!(paths.len(), expected.results.len(), "reference image count");
    assert!(
        expected.results.iter().any(|items| !items.is_empty()),
        "fixtures must exercise recognition"
    );
    for (path, items) in paths.iter().zip(expected.results) {
        let actual = ocr(&image::open(path).unwrap()).expect("Rust OCR failed");
        assert_eq!(
            actual.len(),
            items.len(),
            "OCR count differs for {}",
            path.display()
        );
        let mut unmatched = actual;
        for item in items {
            // Score parity is not checked: Rust returns detection confidence;
            // Python multiplies detection and recognition confidence.
            assert!(item.score.is_finite());
            let bbox = (item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]);
            let index = unmatched
                .iter()
                .position(|(b, text, score)| {
                    text == &item.text
                        && score.is_finite()
                        && dghs_imgutils_rs::detect::similarity::calculate_iou(
                            &(
                                b.0.max(0) as u32,
                                b.1.max(0) as u32,
                                b.2.max(0) as u32,
                                b.3.max(0) as u32,
                            ),
                            &(
                                bbox.0.max(0) as u32,
                                bbox.1.max(0) as u32,
                                bbox.2.max(0) as u32,
                                bbox.3.max(0) as u32,
                            ),
                        ) > 0.5
                })
                .unwrap_or_else(|| {
                    panic!("no matching OCR result for {item:?} in {}", path.display())
                });
            unmatched.remove(index);
        }
    }
}

#[test]
fn reference_preserves_image_boundaries() {
    let output: PythonOcrOutput =
        serde_json::from_str(r#"{"results":[[],[{"bbox":[1,2,3,4],"text":"hello","score":0.9}]]}"#)
            .unwrap();
    assert!(output.results[0].is_empty());
    assert_eq!(output.results[1][0].text, "hello");
    assert!(
        serde_json::from_str::<PythonOcrOutput>(
            r#"{"results":[{"bbox":[1,2,3,4],"text":"hello","score":0.9}]}"#
        )
        .is_err()
    );
}
