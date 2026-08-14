use dghs_imgutils_rs::inference::{create_onnx_session, openvino_device_type};
use std::env;

/// Creates a minimal ONNX Runtime session to verify the configured execution provider.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::args()
        .nth(1)
        .ok_or("usage: cargo run --example intel_ep_probe -- <model.onnx>")?;

    let device = openvino_device_type();
    println!("creating ONNX Runtime session with requested device: {device}");
    let _session = create_onnx_session(model_path)?;
    println!("session created successfully");
    Ok(())
}
