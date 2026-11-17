//! Opt-in real-model smoke test with a cat-in-a-box image.
//! Set DGHS_SMOLVLM2_TEST_IMAGE and optionally DGHS_SMOLVLM2_MODEL_DIR.
use dghs_imgutils_rs::generic::smolvlm2::{SmolVlm2Error, SmolVlm2Options, smolvlm2_generate};
use dghs_imgutils_rs::inference::{Backend, SessionOptions, with_inference_options};

#[test]
#[ignore = "requires SmolVLM2 weights, ONNX Runtime, and a cat-in-a-box image"]
fn generates_english_from_image_and_prompt() {
    let image = image::open(std::env::var("DGHS_SMOLVLM2_TEST_IMAGE").unwrap()).unwrap();
    let options = SmolVlm2Options {
        model_dir: std::env::var_os("DGHS_SMOLVLM2_MODEL_DIR").map(Into::into),
        ..Default::default()
    };
    with_inference_options(SessionOptions::for_backend(Backend::Cpu), || {
        let prompt = "Describe this image in one sentence.";
        let caption = smolvlm2_generate(&image, prompt, &options).unwrap();
        println!("{caption}");
        assert!(caption.to_lowercase().contains("cat"));
        assert!(caption.to_lowercase().contains("box"));
        let answer = smolvlm2_generate(
            &image,
            "What animal is in the image? Answer with one word.",
            &options,
        )
        .unwrap();
        println!("{answer}");
        assert!(answer.to_lowercase().contains("cat"));
        assert_eq!(
            smolvlm2_generate(&image, prompt, &options).unwrap(),
            caption
        );

        let short = SmolVlm2Options {
            max_new_tokens: 1,
            ..options.clone()
        };
        assert!(
            !smolvlm2_generate(&image, prompt, &short)
                .unwrap()
                .is_empty()
        );
        for invalid in ["<image>".to_owned(), "cat ".repeat(8192)] {
            assert!(matches!(
                smolvlm2_generate(&image, &invalid, &options),
                Err(SmolVlm2Error::InvalidInput(_))
            ));
        }
    });
}
