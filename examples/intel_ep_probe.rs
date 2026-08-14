use dghs_imgutils_rs::inference::create_onnx_session;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::args()
        .nth(1)
        .ok_or("usage: cargo run --example intel_ep_probe -- <model.onnx>")?;

    let device = env::var("DGHS_ORT_DEVICE").unwrap_or_else(|_| "AUTO:NPU,GPU,CPU".to_owned());
    println!("creating ONNX Runtime session with OpenVINO device type: {device}");
    let _session = create_onnx_session(model_path)?;
    println!("session created successfully");
    Ok(())
}
