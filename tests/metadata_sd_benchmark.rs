//! metadata / sd モジュールの Rust 版と Python 版の出力を比較する自動精度検証ベンチマーク。

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use image::DynamicImage;

use dghs_imgutils_rs::metadata::geninfo::read_geninfo_parameters;
use dghs_imgutils_rs::metadata::lsb::{read_lsb_metadata, write_lsb_metadata};
use dghs_imgutils_rs::sd::metadata::parse_sdmeta_from_text;

#[derive(Debug, serde::Deserialize)]
struct PythonOutput {
    success: bool,
    error: Option<String>,
    geninfo: Option<String>,
    metadata: Option<HashMap<String, String>>,
    prompt: Option<String>,
    neg_prompt: Option<String>,
    parameters: Option<HashMap<String, String>>,
    dst_path: Option<String>,
}

fn run_python(command: &str, path: &str, extra_args: &[&str]) -> PythonOutput {
    let mut cmd = std::process::Command::new("uv");
    cmd.current_dir("imgutils")
        .arg("run")
        .arg("python")
        .arg("../benchmark/run_python_metadata_sd.py")
        .arg(command)
        .arg(path);
    for arg in extra_args {
        cmd.arg(arg);
    }

    let output = cmd.output().expect("Failed to execute python via uv");
    let stdout_str = String::from_utf8_lossy(&output.stdout);

    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Python side failed ({}):\nSTDOUT:\n{}\nSTDERR:\n{}",
            command, stdout_str, stderr_str
        );
    }

    serde_json::from_str(&stdout_str).expect("Failed to parse Python JSON output")
}

fn test_image_path(rel_path: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut path = PathBuf::from(manifest_dir);
    path.push(rel_path);
    path
}

#[test]
#[ignore]
fn test_metadata_sd_precision_and_benchmark() {
    println!("============================================================");
    println!("metadata + sd モジュール精度検証ベンチマーク");
    println!("============================================================");

    let test_png = test_image_path("imgutils/test/testfile/skadi.jpg");
    if !test_png.exists() {
        eprintln!("Test image not found: {:?}", test_png);
        eprintln!("Skipping precision test (no test images available)");
        return;
    }

    // 1. geninfo 読み取りテスト
    println!("\n--- 1. geninfo 読み取り ---");
    let py_start = Instant::now();
    let py_res = run_python("geninfo", test_png.to_str().unwrap(), &[]);
    let py_duration = py_start.elapsed();

    let rust_start = Instant::now();
    let rust_geninfo = read_geninfo_parameters(test_png.to_str().unwrap());
    let rust_duration = rust_start.elapsed();

    println!("  Python: {:?} ({:?})", py_res.geninfo, py_duration);
    println!("  Rust:   {:?} ({:?})", rust_geninfo, rust_duration);
    assert_eq!(py_res.geninfo, rust_geninfo, "geninfo mismatch");

    // 2. SD メタデータパーステスト
    println!("\n--- 2. SD メタデータパース ---");
    if let Some(ref text) = rust_geninfo {
        let rust_meta = parse_sdmeta_from_text(text);
        let py_meta = run_python("sd_parse", test_png.to_str().unwrap(), &[]);
        println!(
            "  Prompt:     Rust='{}' vs Python='{}'",
            rust_meta.prompt,
            py_meta.prompt.as_deref().unwrap_or("")
        );
        println!(
            "  NegPrompt:  Rust='{}' vs Python='{}'",
            rust_meta.neg_prompt,
            py_meta.neg_prompt.as_deref().unwrap_or("")
        );
        println!(
            "  Parameters: Rust={:?} vs Python={:?}",
            rust_meta.parameters,
            py_meta.parameters.as_ref().unwrap_or(&HashMap::new())
        );
    }

    // 3. LSB 書き込み／読み取りテスト
    println!("\n--- 3. LSB 書き込み／読み取り ---");
    let img = image::open(test_png.to_str().unwrap()).expect("Failed to open test image");
    let mut test_metadata = HashMap::new();
    test_metadata.insert("prompt".to_string(), "test cat".to_string());
    test_metadata.insert("steps".to_string(), "20".to_string());

    // Rust LSB write + read
    let rust_start = Instant::now();
    let encoded = write_lsb_metadata(&img, &test_metadata);
    let rust_read = read_lsb_metadata(&encoded);
    let rust_duration = rust_start.elapsed();
    println!("  Rust write+read: {:?} ({:?})", rust_read, rust_duration);
    assert_eq!(
        rust_read.get("prompt").map(|s| s.as_str()),
        Some("test cat")
    );

    // Python LSB read (from Rust-encoded image)
    let tmp_path = std::env::temp_dir().join("rust_lsb_test.png");
    encoded.save(&tmp_path).expect("Failed to save temp image");
    let py_start = Instant::now();
    let py_res = run_python("lsb_read", tmp_path.to_str().unwrap(), &[]);
    let py_duration = py_start.elapsed();
    println!(
        "  Python read (from Rust-encoded): {:?} ({:?})",
        py_res.metadata, py_duration
    );
    std::fs::remove_file(&tmp_path).unwrap_or(());

    // 4. Safetensors メタデータテスト
    println!("\n--- 4. Safetensors メタデータ ---");
    let st_path = test_image_path("imgutils/test/testfile/model_test.safetensors");
    if st_path.exists() {
        let py_res = run_python("safetensors", st_path.to_str().unwrap(), &[]);
        println!("  Python: {:?}", py_res.metadata);
        let rust_meta =
            dghs_imgutils_rs::sd::safetensors::read_safetensors_metadata(st_path.to_str().unwrap());
        println!("  Rust:   {:?}", rust_meta);
        if let Ok(ref rust_m) = rust_meta {
            if let Some(ref py_m) = py_res.metadata {
                assert_eq!(rust_m, py_m, "safetensors metadata mismatch");
            }
        }
    } else {
        println!("  Skipping (no safetensors test file available)");
    }

    println!("\n============================================================");
    println!("全てのテストが完了しました。");
    println!("============================================================");
}
