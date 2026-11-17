//! Opt-in real-model English-to-Japanese translation tests.
//! Set DGHS_PLAMO3_MODEL_DIR to a local TeamFlaxia Q4 export, or omit for HF download.
use dghs_imgutils_rs::generic::plamo3::{Plamo3Error, Plamo3Options, plamo3_translate};
use dghs_imgutils_rs::inference::{Backend, SessionOptions, with_inference_options};

#[test]
#[ignore = "requires PLaMo 3 Translate 0.8B Q4 weights and ONNX Runtime"]
fn translates_english_and_keeps_requests_independent() {
    let options = Plamo3Options {
        model_dir: std::env::var_os("DGHS_PLAMO3_MODEL_DIR").map(Into::into),
        ..Default::default()
    };
    with_inference_options(SessionOptions::for_backend(Backend::Cpu), || {
        let source = "The image depicts a cardboard box with a cat inside it.";
        let translated = plamo3_translate(source, &options).unwrap();
        println!("{translated}");
        assert!(translated.contains("猫") && translated.contains("箱"));
        assert!(!translated.contains("<|plamo:"));

        let meeting =
            plamo3_translate("The meeting has been moved to next Monday.", &options).unwrap();
        println!("{meeting}");
        assert!(meeting.contains("来週") && meeting.contains("月曜"));
        let short = Plamo3Options {
            max_new_tokens: 1,
            ..options.clone()
        };
        assert!(matches!(
            plamo3_translate(source, &short),
            Err(Plamo3Error::GenerationLimit(1))
        ));
        assert_eq!(plamo3_translate(source, &options).unwrap(), translated);

        let long_source = "cat ".repeat(32768);
        assert!(matches!(
            plamo3_translate(&long_source, &options),
            Err(Plamo3Error::InvalidInput(_))
        ));
    });

    // Concurrent callers share one session but keep independent prompts/KV caches.
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let options = options.clone();
            std::thread::spawn(move || {
                with_inference_options(SessionOptions::for_backend(Backend::Cpu), || {
                    plamo3_translate("A cat is in a box.", &options).unwrap()
                })
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(results[0], results[1]);
    assert!(results[0].contains("猫") && results[0].contains("箱"));
}
