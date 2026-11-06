//! Validation of image model input contracts.
use super::InferenceError;
use ort::value::{TensorElementType, ValueType};

/// Returns (width, height) for a fixed RGB NCHW or NHWC tensor.
pub(crate) fn image_size(
    dtype: &ValueType,
    channels_last: bool,
) -> Result<(u32, u32), InferenceError> {
    image_size_with_fallback(dtype, channels_last, None)
}

/// Uses a caller-selected size only for symbolic (negative) spatial dimensions.
pub(crate) fn image_size_with_fallback(
    dtype: &ValueType,
    channels_last: bool,
    fallback: Option<u32>,
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
    let dimension = |value| {
        if value < 0 {
            fallback.filter(|v| *v > 0)
        } else {
            u32::try_from(value).ok().filter(|v| *v > 0)
        }
    };
    let width = dimension(shape[w]);
    let height = dimension(shape[h]);
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
    fn classifier_fallback_preserves_fixed_axes() {
        for (shape, expected) in [
            ([1, 3, -1, -1], (224, 224)),
            ([1, 3, 384, -1], (224, 384)),
            ([1, 3, -1, 448], (448, 224)),
        ] {
            assert_eq!(
                image_size_with_fallback(&dtype(&shape), false, Some(224)).unwrap(),
                expected
            );
        }
        assert!(image_size_with_fallback(&dtype(&[1, 3, 0, 448]), false, Some(224)).is_err());
        assert!(image_size_with_fallback(&dtype(&[1, 1, -1, -1]), false, Some(224)).is_err());
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
