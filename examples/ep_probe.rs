use dghs_imgutils_rs::inference::{
    SessionOptions, create_onnx_session_with_options, probe_backends,
};
use std::env;

/// Prints provider availability and optionally creates one strict session.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = SessionOptions::from_env()?;
    println!("worker backend: {}", options.backend);
    println!("worker precision: {}", options.precision);
    println!(
        "ORT_DYLIB_PATH: {}",
        env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<default>".to_owned())
    );

    for capability in probe_backends() {
        println!(
            "backend={} provider={:?} available={} fp16={:?} bf16={:?} int8={:?} dynamic_shapes={:?} reason={:?}",
            capability.backend,
            capability.provider_name,
            capability.available,
            capability.supports_fp16,
            capability.supports_bf16,
            capability.supports_int8,
            capability.supports_dynamic_shapes,
            capability.reason,
        );
    }

    if let Some(model_path) = env::args().nth(1) {
        println!("creating strict session for {model_path}");
        let _session = create_onnx_session_with_options(model_path, &options)?;
        println!("session created successfully");
    }

    Ok(())
}
