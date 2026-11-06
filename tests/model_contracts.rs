//! Opt-in real-model regression tests for input names, dimensions and assets.
//! Run scripts/test-model-contracts.py to isolate crashes between models.
mod support;
use dghs_imgutils_rs::{generic, metrics, ocr, tagging, validate};
use std::collections::HashMap;

fn tags(result: tagging::TagResult) {
    assert!(
        result
            .tag
            .iter()
            .chain(result.rest.values().flatten())
            .all(|(_, v)| v.is_finite())
    );
}
fn scores(values: HashMap<String, f32>) {
    assert!(!values.is_empty());
    assert!(values.values().all(|v| v.is_finite()));
}
macro_rules! tag_test {
    ($name:ident,$fn:path) => {
        #[test]
        #[ignore = "requires real model downloads and ONNX Runtime"]
        fn $name() {
            tags($fn(&support::image()).expect(concat!(stringify!($name), " inference failed")));
        }
    };
}
tag_test!(
    deepdanbooru,
    tagging::deepdanbooru::get_deepdanbooru_tags_simple
);
tag_test!(
    deepgelbooru,
    tagging::deepgelbooru::get_deepgelbooru_tags_simple
);
tag_test!(wd14, tagging::wd14::get_wd14_tags_simple);
tag_test!(camie, tagging::camie::get_camie_tags_simple);

macro_rules! classify_test {
    ($name:ident,$fn:path) => {
        #[test]
        #[ignore = "requires real model downloads and ONNX Runtime"]
        fn $name() {
            let f = tempfile::NamedTempFile::with_suffix(".png").unwrap();
            support::image().save(f.path()).unwrap();
            scores(
                $fn(f.path().to_str().unwrap(), None)
                    .expect(concat!(stringify!($name), " inference failed")),
            );
        }
    };
}
classify_test!(
    anime_classification,
    validate::classify::anime_classify_score
);
classify_test!(anime_real, validate::classify::anime_real_score);
classify_test!(anime_furry, validate::classify::anime_furry_score);
#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn aesthetic() {
    assert!(
        metrics::dbaesthetic::anime_dbaesthetic(&support::image(), None)
            .expect("aesthetic inference failed")
            .is_finite()
    );
}
#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn paddleocr() {
    let _ = ocr::ocr(&support::image()).expect("PaddleOCR pipeline failed");
}
#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn siglip_image() {
    let (emb, raw) = generic::siglip::siglip_image_encode(
        &[support::image()],
        "deepghs/siglip_onnx",
        "google/siglip-base-patch16-256-multilingual",
    )
    .expect("SigLIP image failed");
    support::embeddings(&emb, 1);
    support::embeddings(&raw, 1);
}
#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn siglip_text() {
    let (emb, raw) = generic::siglip::siglip_text_encode(
        &["an anime illustration".into(), "日本語の文章".into()],
        "deepghs/siglip_onnx",
        "google/siglip-base-patch16-256-multilingual",
    )
    .expect("SigLIP text failed");
    support::embeddings(&emb, 2);
    support::embeddings(&raw, 2);
}

#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn clip_long_text() {
    let texts = ["an anime illustration ".repeat(200), String::new()];
    let (emb, raw) = generic::clip::clip_text_encode(
        &texts,
        "deepghs/clip_onnx",
        "openai/clip-vit-base-patch32",
    )
    .expect("CLIP long-text inference failed");
    support::embeddings(&emb, texts.len());
    support::embeddings(&raw, texts.len());
}

#[test]
#[ignore = "requires real model downloads and ONNX Runtime"]
fn siglip_long_text() {
    let texts = ["日本語の長い文章 ".repeat(200), String::new()];
    let (emb, raw) = generic::siglip::siglip_text_encode(
        &texts,
        "deepghs/siglip_onnx",
        "google/siglip-base-patch16-256-multilingual",
    )
    .expect("SigLIP long-text inference failed");
    support::embeddings(&emb, texts.len());
    support::embeddings(&raw, texts.len());
}
