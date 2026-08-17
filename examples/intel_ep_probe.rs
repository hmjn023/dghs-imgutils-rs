use dghs_imgutils_rs::inference::{
    DeviceProvider, DeviceSelection, SessionOptions, create_onnx_session_with_options,
};
use ort::value::{Tensor, TensorElementType, ValueType};
use std::env;
use std::str::FromStr;

/// Creates a session and optionally runs a small single-input FP32 tensor.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut run_inference = false;
    let mut provider = None;
    let mut device = None;
    let mut model_path = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--run" => run_inference = true,
            "--provider" => {
                let value = args.next().ok_or("missing provider value")?;
                provider = Some(DeviceProvider::from_str(&value)?);
            }
            "--device" => device = Some(args.next().ok_or("missing device value")?),
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {value}").into());
            }
            value => {
                if model_path.replace(value.to_owned()).is_some() {
                    return Err("only one model path may be specified".into());
                }
            }
        }
    }

    let model_path = model_path.ok_or(
        "usage: cargo run --example intel_ep_probe -- [--run] [--provider intel_gpu|intel_npu] [--device 0|GPU.0] <model.onnx>",
    )?;
    let options = if let Some(provider) = provider {
        SessionOptions::for_device(DeviceSelection { provider, device })?
    } else {
        SessionOptions::from_env()?
    };
    let requested_device = options
        .openvino_device_type
        .clone()
        .unwrap_or_else(|| options.backend.to_string());
    println!("creating ONNX Runtime session with requested device: {requested_device}");
    let mut session = create_onnx_session_with_options(model_path, &options)?;
    println!("session created successfully");

    if run_inference {
        run_single_fp32_probe(&mut session)?;
    }

    Ok(())
}

fn run_single_fp32_probe(
    session: &mut ort::session::Session,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = session.inputs().first().ok_or("model has no inputs")?;
    let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
        return Err("probe only supports tensor inputs".into());
    };
    if *ty != TensorElementType::Float32 {
        return Err(format!("probe only supports FP32 input, got {ty}").into());
    }

    let shape = shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).map_err(|_| "input has a dynamic dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let element_count = shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or("input shape is too large")
    })?;
    let input_name = input.name().to_owned();
    let tensor = Tensor::from_array((shape, vec![1.0_f32; element_count]))?;
    let outputs = session.run(ort::inputs![input_name.as_str() => tensor])?;
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;
    println!(
        "inference completed: output_shape={output_shape:?} first_value={:?}",
        output_data.first()
    );
    Ok(())
}
