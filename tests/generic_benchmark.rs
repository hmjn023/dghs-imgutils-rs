//! Opt-in Rust model inference benchmarks. Each warmup and timed run must succeed.
//! `cargo test --no-default-features --features openvino --test generic_benchmark -- --ignored --nocapture`
//! Select the backend with DGHS_BACKEND/DGHS_ORT_DEVICE. No Python dependency.
mod support;
use dghs_imgutils_rs::generic::{clip, yoloseg};
const REPO: &str = "deepghs/clip_onnx";
const MODEL: &str = "openai/clip-vit-base-patch32";

#[test]
#[ignore = "downloads CLIP models and requires ONNX Runtime"]
fn generic_benchmark_clip_image_encode() {
    let image = support::image();
    support::benchmark("CLIP image", 3, || {
        let (emb, raw) = clip::clip_image_encode(std::slice::from_ref(&image), REPO, MODEL)
            .expect("CLIP image inference failed");
        support::embeddings(&emb, 1);
        support::embeddings(&raw, 1);
    });
}
#[test]
#[ignore = "downloads CLIP models and requires ONNX Runtime"]
fn generic_benchmark_clip_text_encode() {
    let texts = vec!["a cat".into(), "an anime illustration".into()];
    support::benchmark("CLIP text", 3, || {
        let (emb, raw) =
            clip::clip_text_encode(&texts, REPO, MODEL).expect("CLIP text inference failed");
        support::embeddings(&emb, texts.len());
        support::embeddings(&raw, texts.len());
    });
}
#[test]
#[ignore = "downloads CLIP models and requires ONNX Runtime"]
fn generic_benchmark_clip_predict() {
    let image = support::image();
    let texts = vec!["cat".into(), "dog".into(), "anime".into()];
    support::benchmark("CLIP predict", 3, || {
        let (image_emb, _) = clip::clip_image_encode(std::slice::from_ref(&image), REPO, MODEL)
            .expect("CLIP image inference failed");
        let (text_emb, _) =
            clip::clip_text_encode(&texts, REPO, MODEL).expect("CLIP text inference failed");
        let scale = clip::clip_logit_scale(REPO, MODEL).expect("CLIP metadata failed");
        let (similarity, logits, probabilities) = clip::clip_predict(&image_emb, &text_emb, scale);
        for values in [&similarity, &logits, &probabilities] {
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].len(), texts.len());
            assert!(values[0].iter().all(|x| x.is_finite()));
        }
        assert!((probabilities[0].iter().sum::<f32>() - 1.0).abs() < 1e-5);
    });
}
#[test]
#[ignore = "requires DGHS_YOLOSEG_REPO and DGHS_YOLOSEG_MODEL plus ONNX Runtime"]
fn generic_benchmark_yoloseg() {
    let repo = std::env::var("DGHS_YOLOSEG_REPO")
        .expect("set DGHS_YOLOSEG_REPO to an accessible YOLO-Seg repository");
    let model = std::env::var("DGHS_YOLOSEG_MODEL").expect("set DGHS_YOLOSEG_MODEL");
    let image = support::image();
    support::benchmark("YOLO-Seg", 3, || {
        yoloseg::yolo_seg_predict(&image, &repo, &model, 0.25, 0.7)
            .expect("YOLO-Seg inference failed")
    });
}
