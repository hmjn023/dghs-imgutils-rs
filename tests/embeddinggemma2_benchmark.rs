//! Opt-in real-model parity test. Set DGHS_EMBEDDINGGEMMA2_MODEL_DIR to a Q8
//! export root, then run this test with --no-default-features -- --ignored.

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

#[test]
#[ignore = "requires Q8 model files, Python uv, and ONNX Runtime"]
fn embeddinggemma2_python_rust_parity() {
    let model_dir = std::env::var("DGHS_EMBEDDINGGEMMA2_MODEL_DIR")
        .expect("set DGHS_EMBEDDINGGEMMA2_MODEL_DIR to the downloaded export root");
    let temp = tempfile::tempdir().unwrap();
    let mut images = Vec::new();
    let mut paths = Vec::new();
    for (i, (w, h)) in [(181, 97), (73, 149), (384, 384)].into_iter().enumerate() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(w, h, |x, y| {
            Rgba([
                (x * 7 % 256) as u8,
                (y * 3 % 256) as u8,
                ((x + y) % 256) as u8,
                (x % 256) as u8,
            ])
        }));
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
    let options = EmbeddingGemma2Options {
        max_soft_tokens: 70,
        ..Default::default()
    };
    let texts = vec!["cat".to_string(), "赤い服を着たキャラクター".to_string()];
    let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(181, 97, |x, y| {
        Rgba([
            (x * 7 % 256) as u8,
            (y * 3 % 256) as u8,
            ((x + y) % 256) as u8,
            255,
        ])
    }));
    let run = |session: SessionOptions| {
        with_inference_options(session, || {
            let text = embeddinggemma2_text_encode(&texts, EmbeddingGemma2TextTask::Raw, &options)
                .unwrap();
            let images =
                embeddinggemma2_image_encode(std::slice::from_ref(&image), &options).unwrap();
            (text, images)
        })
    };
    let cpu = run(SessionOptions::for_backend(Backend::Cpu));
    for precision in [Precision::Auto, Precision::Fp32] {
        let gpu = run(SessionOptions::for_backend(Backend::OpenVino)
            .with_openvino_device_type("GPU")
            .with_precision(precision));
        for (a, b) in gpu.0.iter().zip(&cpu.0) {
            assert_embedding(a, b, 0.99);
        }
        for (a, b) in gpu.1.iter().zip(&cpu.1) {
            assert_embedding(a, b, 0.99);
        }
    }
}
