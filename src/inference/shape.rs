//! Validation of image model input contracts.
use super::InferenceError;
use ort::value::{TensorElementType, ValueType};

/// Returns (width, height) for a fixed RGB NCHW or NHWC tensor.
pub(crate) fn image_size(
    dtype: &ValueType,
    channels_last: bool,
) -> Result<(u32, u32), InferenceError> {
    let ValueType::Tensor { ty, shape, .. } = dtype else {
        return Err(InferenceError::InvalidShape(
            "image input is not a tensor".into(),
        ));
    };
    let (c, h, w) = if channels_last { (3, 1, 2) } else { (1, 2, 3) };
    if *ty != TensorElementType::Float32 || shape.len() != 4 || shape[c] != 3 {
        return Err(InferenceError::InvalidShape(format!(
            "expected a rank-4 RGB float32 input, got {dtype:?}"
        )));
    }
    let width = u32::try_from(shape[w]).ok().filter(|v| *v > 0);
    let height = u32::try_from(shape[h]).ok().filter(|v| *v > 0);
    match (width, height) {
        (Some(w), Some(h)) => Ok((w, h)),
        _ => Err(InferenceError::InvalidShape(format!(
            "image spatial dimensions must be positive and fixed: {shape:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dtype(shape: &[i64]) -> ValueType {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: shape.to_vec().into(),
            dimension_symbols: ort::value::SymbolicDimensions::empty(shape.len()),
        }
    }
    #[test]
    fn uses_model_resolution_and_layout() {
        assert_eq!(
            image_size(&dtype(&[-1, 3, 384, 448]), false).unwrap(),
            (448, 384)
        );
        assert_eq!(
            image_size(&dtype(&[-1, 448, 448, 3]), true).unwrap(),
            (448, 448)
        );
    }
    #[test]
    fn rejects_unknown_or_invalid_image_contracts() {
        for shape in [
            &[1, 3, -1, 448][..],
            &[1, 3, 0, 448],
            &[1, 1, 384, 384],
            &[1, 3, 384],
        ] {
            assert!(image_size(&dtype(shape), false).is_err());
        }
    }
}
