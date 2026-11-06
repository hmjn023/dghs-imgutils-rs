//! Opt-in real-model parity tests. Set `DGHS_EMBEDDINGGEMMA2_MODEL_DIR` to a Q8
//! export root, or omit it to use the HF cache. GPU preparation is automatic. Set `DGHS_EMBEDDINGGEMMA2_OPENVINO_DIR` to a separately prepared
//! OpenVINO export to compare its GPU outputs against the original CPU model.
//! To run the GPU comparison with the installed ONNX Runtime:
//! ```text
//! cargo test --no-default-features --features openvino --test embeddinggemma2_benchmark \
//!   embeddinggemma2_openvino_gpu_matches_cpu -- --ignored --nocapture
//! ```

use dghs_imgutils_rs::generic::embeddinggemma2::{
    EmbeddingGemma2Options, EmbeddingGemma2TextTask, embeddinggemma2_image_encode,
    embeddinggemma2_text_encode,
};
use dghs_imgutils_rs::inference::{Backend, SessionOptions, with_inference_options};
use image::{DynamicImage, Rgba, RgbaImage};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    texts: Vec<String>,
    text_embeddings: Vec<Vec<f32>>,
    image_embeddings: Vec<Vec<f32>>,
}

fn assert_embedding(actual: &[f32], expected: &[f32], min_cosine: f32) {
    assert_eq!(actual.len(), 768);
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm(actual) - 1.0).abs() < 1e-5);
    let cosine = actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f32>()
        / (norm(actual) * norm(expected));
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("cosine={cosine:.8}, max_abs_error={max_error:.8}");
    assert!(cosine >= min_cosine, "cosine {cosine} < {min_cosine}");
    assert!(max_error < 0.02, "max absolute error {max_error}");
}

fn pattern_image(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
        Rgba([
            (x * 7 % 256) as u8,
            (y * 3 % 256) as u8,
            ((x + y) % 256) as u8,
            (x % 256) as u8,
        ])
    }))
}

#[test]
#[ignore = "requires Q8 model files, Python uv, and ONNX Runtime"]
fn embeddinggemma2_python_rust_parity() {
    let model_dir = std::env::var("DGHS_EMBEDDINGGEMMA2_MODEL_DIR")
        .expect("set DGHS_EMBEDDINGGEMMA2_MODEL_DIR to the downloaded export root");
    let temp = tempfile::tempdir().unwrap();
    let mut images = Vec::new();
    let mut paths = Vec::new();
    for (i, (w, h)) in [(181, 97), (73, 149), (384, 384)].into_iter().enumerate() {
        let image = pattern_image(w, h);
        let path = temp.path().join(format!("image-{i}.png"));
        image.save(&path).unwrap();
        images.push(image);
        paths.push(path);
    }
    if let Ok(path) = std::env::var("DGHS_EMBEDDINGGEMMA2_TEST_IMAGE") {
        images.push(image::open(&path).unwrap());
        paths.push(path.into());
    }
    let output = std::process::Command::new("uv")
        .args([
            "run",
            "--script",
            "benchmark/run_python_embeddinggemma2.py",
            &model_dir,
            "--images",
        ])
        .args(&paths)
        .output()
        .expect("run Python reference");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference: Reference = serde_json::from_slice(&output.stdout).unwrap();
    let options = EmbeddingGemma2Options {
        model_dir: Some(model_dir.into()),
        max_soft_tokens: 70,
        ..Default::default()
    };
    with_inference_options(SessionOptions::default().with_backend(Backend::Cpu), || {
        let texts =
            embeddinggemma2_text_encode(&reference.texts, EmbeddingGemma2TextTask::Raw, &options)
                .unwrap();
        assert_eq!(texts.len(), reference.text_embeddings.len());
        for (actual, expected) in texts.iter().zip(&reference.text_embeddings) {
            assert_embedding(actual, expected, 0.99999);
        }
        let encoded = embeddinggemma2_image_encode(&images, &options).unwrap();
        assert_eq!(encoded.len(), reference.image_embeddings.len());
        for (actual, expected) in encoded.iter().zip(&reference.image_embeddings) {
            assert_embedding(actual, expected, 0.99);
        }
        let small = EmbeddingGemma2Options {
            dimensions: 256,
            ..options.clone()
        };
        let truncated = embeddinggemma2_text_encode(
            &reference.texts[..1],
            EmbeddingGemma2TextTask::Raw,
            &small,
        )
        .unwrap();
        let norm = texts[0][..256].iter().map(|x| x * x).sum::<f32>().sqrt();
        for (actual, full) in truncated[0].iter().zip(&texts[0]) {
            assert!((actual - full / norm).abs() < 1e-5);
        }
        assert!(
            embeddinggemma2_text_encode(
                &["<|image|>".into()],
                EmbeddingGemma2TextTask::Raw,
                &options
            )
            .is_err()
        );
        assert!(
            embeddinggemma2_text_encode(
                &["word ".repeat(9000)],
                EmbeddingGemma2TextTask::Raw,
                &options
            )
            .is_err()
        );
    });
}

/// Opt-in GPU regression: compares actual model outputs, including the input
/// that produced all-zero vectors with the OpenVINO GPU default precision.
#[cfg(feature = "openvino")]
#[test]
#[ignore = "requires cached EmbeddingGemma 2 model and Intel GPU"]
fn embeddinggemma2_openvino_gpu_matches_cpu() {
    use dghs_imgutils_rs::inference::Precision;
    let cpu_options = EmbeddingGemma2Options {
        model_dir: std::env::var_os("DGHS_EMBEDDINGGEMMA2_MODEL_DIR").map(Into::into),
        ..Default::default()
    };
    let prepared =
        std::env::var_os("DGHS_EMBEDDINGGEMMA2_OPENVINO_DIR").map(std::path::PathBuf::from);
    if let Some(dir) = &prepared {
        for component in ["model", "vision_encoder"] {
            let path = dir.join(format!("onnx/{component}_quantized.openvino.onnx"));
            assert!(
                path.is_file(),
                "missing prepared GPU model: {}",
                path.display()
            );
        }
    }
    let gpu_options = EmbeddingGemma2Options {
        model_dir: prepared.clone().or_else(|| cpu_options.model_dir.clone()),
        ..cpu_options.clone()
    };
    let texts = vec![
        "cat".to_string(),
        "赤い服を着たキャラクター".to_string(),
        // A single repeated RoPE position changes this query's final vector.
        "task: search result | query: cat".to_string(),
        "title: none | text: A cat wearing red clothes sits beside the window.".to_string(),
    ];
    let cases = [
        (70, pattern_image(181, 97)),
        (cpu_options.max_soft_tokens, pattern_image(73, 149)),
    ];
    let run = |session: SessionOptions, options: &EmbeddingGemma2Options| {
        with_inference_options(session, || {
            let text =
                embeddinggemma2_text_encode(&texts, EmbeddingGemma2TextTask::Raw, options).unwrap();
            assert_eq!(text.len(), texts.len());
            let mut images = Vec::new();
            for (budget, image) in &cases {
                let options = EmbeddingGemma2Options {
                    max_soft_tokens: *budget,
                    ..options.clone()
                };
                let batch =
                    embeddinggemma2_image_encode(std::slice::from_ref(image), &options).unwrap();
                assert_eq!(batch.len(), 1);
                images.extend(batch);
            }
            (text, images)
        })
    };
    let cpu = run(SessionOptions::for_backend(Backend::Cpu), &cpu_options);
    let min_cosine = 0.9999;
    for precision in [Precision::Auto, Precision::Fp32] {
        let gpu = run(
            SessionOptions::for_backend(Backend::OpenVino)
                .with_openvino_device_type("GPU")
                .with_precision(precision),
            &gpu_options,
        );
        for (text, (a, b)) in texts.iter().zip(gpu.0.iter().zip(&cpu.0)) {
            println!("GPU {precision:?} text {text:?}");
            assert_embedding(a, b, min_cosine);
        }
        for ((budget, image), (a, b)) in cases.iter().zip(gpu.1.iter().zip(&cpu.1)) {
            println!(
                "GPU {precision:?} image {}x{}, budget {budget}",
                image.width(),
                image.height()
            );
            assert_embedding(a, b, min_cosine);
        }
    }
}
