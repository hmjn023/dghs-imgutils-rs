mod support;
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore]
fn test_tagging_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let test_images = [
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/hutao.jpg",
    ];

    println!("============================================================");
    println!("Tagging Backend Benchmarks");
    println!("============================================================");

    let mut failures = Vec::new();
    for (idx, rel_path) in test_images.iter().enumerate() {
        let mut image_path = PathBuf::from(manifest_dir);
        image_path.push(rel_path);
        assert!(
            image_path.exists(),
            "Test image not found: {:?}",
            image_path
        );

        let image = image::open(&image_path).expect("Failed to open image");

        println!("\n[{}/{}] {}", idx + 1, test_images.len(), rel_path);

        // DeepDanbooru
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::deepdanbooru::get_deepdanbooru_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!(
                "  DeepDanbooru: {:?} (general={}, character={})",
                dur,
                r.general.len(),
                r.character.len()
            ),
            Err(e) => failures.push(format!("DeepDanbooru: {e}")),
        }

        // ML-Danbooru
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::mldanbooru::get_mldanbooru_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!("  ML-Danbooru:  {:?} (general={})", dur, r.general.len()),
            Err(e) => failures.push(format!("ML-Danbooru: {e}")),
        }

        // DeepGelbooru
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::deepgelbooru::get_deepgelbooru_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!(
                "  DeepGelbooru: {:?} (general={}, character={})",
                dur,
                r.general.len(),
                r.character.len()
            ),
            Err(e) => failures.push(format!("DeepGelbooru: {e}")),
        }

        // WD14 (SwinV2_v3)
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::wd14::get_wd14_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!(
                "  WD14:         {:?} (general={}, character={})",
                dur,
                r.general.len(),
                r.character.len()
            ),
            Err(e) => failures.push(format!("WD14: {e}")),
        }

        // Camie
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::camie::get_camie_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!(
                "  Camie:        {:?} (general={}, character={})",
                dur,
                r.general.len(),
                r.character.len()
            ),
            Err(e) => failures.push(format!("Camie: {e}")),
        }

        // PixAI (reference)
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::pixai::get_pixai_tags(&image, "v0.9", None);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!(
                "  PixAI:        {:?} (general={}, character={})",
                dur,
                r.general.len(),
                r.character.len()
            ),
            Err(e) => failures.push(format!("PixAI: {e}")),
        }
    }

    println!("\n============================================================");
    assert!(
        failures.is_empty(),
        "Model inference failures:\n{}",
        failures.join("\n")
    );
    println!("Benchmark complete");
    println!("============================================================");
}

#[test]
#[ignore]
fn test_deepdanbooru_vs_python() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut image_path = PathBuf::from(manifest_dir);
    image_path.push("imgutils/test/testfile/skadi.jpg");
    assert!(
        image_path.exists(),
        "Test image not found: {:?}",
        image_path
    );

    let py_output = support::python(&format!(
        "import json; from imgutils.tagging.deepdanbooru import get_deepdanbooru_tags; r,g,c=get_deepdanbooru_tags({}); print(json.dumps({{'general':g,'character':c}}))",
        serde_json::to_string(&image_path.to_string_lossy()).unwrap()
    ));
    let reference: serde_json::Value =
        serde_json::from_str(&py_output).expect("invalid Python reference");

    // Rust inference
    let image = image::open(&image_path).expect("Failed to open image");
    let result = dghs_imgutils_rs::tagging::deepdanbooru::get_deepdanbooru_tags_simple(&image);
    let result = result.expect("Rust DeepDanbooru inference failed");
    let expected = reference["general"]
        .as_object()
        .expect("Python general tags missing");
    let actual: std::collections::HashSet<_> =
        result.general.iter().map(|(k, _)| k.as_str()).collect();
    let expected: std::collections::HashSet<_> = expected.keys().map(String::as_str).collect();
    let union = actual.union(&expected).count();
    assert!(union > 0, "both tag sets are empty");
    let iou = actual.intersection(&expected).count() as f32 / union as f32;
    assert!(iou >= 0.9, "DeepDanbooru tag IoU {iou} < 0.9");
}
