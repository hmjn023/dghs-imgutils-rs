use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore]
fn test_tagging_benchmark() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let test_images = vec![
        "imgutils/test/testfile/skadi.jpg",
        "imgutils/test/testfile/hutao.jpg",
    ];

    println!("============================================================");
    println!("Tagging Backend Benchmarks");
    println!("============================================================");

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
            Err(e) => println!("  DeepDanbooru: FAILED - {}", e),
        }

        // ML-Danbooru
        let start = Instant::now();
        let result = dghs_imgutils_rs::tagging::mldanbooru::get_mldanbooru_tags_simple(&image);
        let dur = start.elapsed();
        match result {
            Ok(r) => println!("  ML-Danbooru:  {:?} (general={})", dur, r.general.len()),
            Err(e) => println!("  ML-Danbooru:  FAILED - {}", e),
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
            Err(e) => println!("  DeepGelbooru: FAILED - {}", e),
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
            Err(e) => println!("  WD14:         FAILED - {}", e),
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
            Err(e) => println!("  Camie:        FAILED - {}", e),
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
            Err(e) => println!("  PixAI:        FAILED - {}", e),
        }
    }

    println!("\n============================================================");
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

    // Python reference
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("-c")
        .arg(&format!(
            "import json, sys; sys.path.insert(0, '.'); from imgutils.tagging.deepdanbooru import get_deepdanbooru_tags; \
             rating, general, chars = get_deepdanbooru_tags('{}'); \
             print(json.dumps({{'success': True, 'general': general, 'character': chars}}))",
            image_path.display()
        ));
    let output = cmd.output().expect("Failed to execute python");
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("Python output: {}", stdout);
    }

    // Rust inference
    let image = image::open(&image_path).expect("Failed to open image");
    let result = dghs_imgutils_rs::tagging::deepdanbooru::get_deepdanbooru_tags_simple(&image);
    match result {
        Ok(r) => {
            println!("Rust general: {} tags", r.general.len());
            println!("Rust character: {} tags", r.character.len());
            assert!(!r.general.is_empty(), "Expected at least some general tags");
        }
        Err(e) => eprintln!("Rust DeepDanbooru failed: {}", e),
    }
}
