//! Metadata-driven Pillow-style preprocessing for exported tagger models.
use crate::inference::InferenceError;
use image::{
    DynamicImage, GenericImageView, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use ndarray::Array4;
use serde_json::Value;

fn invalid(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(message.into())
}

fn size(value: &Value) -> Result<(u32, u32, bool), InferenceError> {
    let positive = |v: &Value| {
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("preprocess size must be positive"))
    };
    if value.is_number() {
        let n = positive(value)?;
        return Ok((n, n, true));
    }
    let a = value
        .as_array()
        .ok_or_else(|| invalid("preprocess size must be a scalar or [height, width]"))?;
    match a.as_slice() {
        [n] => {
            let n = positive(n)?;
            Ok((n, n, true))
        }
        [h, w] => Ok((positive(w)?, positive(h)?, false)),
        _ => Err(invalid("invalid preprocess size")),
    }
}
fn filter(stage: &Value) -> Result<FilterType, InferenceError> {
    match stage["interpolation"].as_str().unwrap_or("bilinear") {
        "nearest" => Ok(FilterType::Nearest),
        "bilinear" => Ok(FilterType::Triangle),
        "bicubic" => Ok(FilterType::CatmullRom),
        "lanczos" => Ok(FilterType::Lanczos3),
        name => Err(invalid(format!("unsupported interpolation: {name}"))),
    }
}
fn color(value: &Value) -> Result<[u8; 3], InferenceError> {
    match value {
        Value::Null => Ok([255; 3]),
        Value::String(s) if s == "white" => Ok([255; 3]),
        Value::String(s) if s == "black" => Ok([0; 3]),
        Value::Array(a) if a.len() == 3 => {
            let mut out = [0; 3];
            for (i, v) in a.iter().enumerate() {
                out[i] = v
                    .as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or_else(|| invalid("invalid background color"))?;
            }
            Ok(out)
        }
        _ => Err(invalid("unsupported background color")),
    }
}
/// Executes image operations and tensor scaling in the order specified by the model.
pub(crate) fn apply(image: &DynamicImage, stages: &[Value]) -> Result<Array4<f32>, InferenceError> {
    let mut img = image.clone();
    let mut tensor: Option<Array4<f32>> = None;
    for stage in stages {
        let kind = stage["type"]
            .as_str()
            .ok_or_else(|| invalid("preprocess stage has no type"))?;
        match kind {
            "to_tensor" | "maybe_to_tensor" => {
                if tensor.is_none() {
                    let rgb = img.to_rgb8();
                    tensor = Some(Array4::from_shape_fn(
                        (1, 3, rgb.height() as usize, rgb.width() as usize),
                        |(_, c, y, x)| rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
                    ));
                }
            }
            "normalize" => {
                let arr = tensor
                    .as_mut()
                    .ok_or_else(|| invalid("normalize requires to_tensor"))?;
                for c in 0..3 {
                    let mean = stage["mean"][c]
                        .as_f64()
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| invalid("invalid normalization mean"))?
                        as f32;
                    let std = stage["std"][c]
                        .as_f64()
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .ok_or_else(|| invalid("invalid normalization std"))?
                        as f32;
                    arr.slice_mut(ndarray::s![0, c, .., ..])
                        .mapv_inplace(|v| (v - mean) / std);
                }
            }
            "rescale" => {
                let factor = stage["rescale_factor"]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| invalid("invalid rescale factor"))?
                    as f32;
                tensor
                    .as_mut()
                    .ok_or_else(|| invalid("rescale requires to_tensor"))?
                    .mapv_inplace(|v| v * factor);
            }
            "resize" | "center_crop" | "pad_to_size" | "convert_rgb" => {
                if tensor.is_some() {
                    return Err(invalid("image operation after to_tensor is unsupported"));
                }
                if kind == "convert_rgb" {
                    img = super::force_image_background(&img, color(&stage["force_background"])?);
                    continue;
                }
                let (mut w, mut h, short) = size(&stage["size"])?;
                let (iw, ih) = img.dimensions();
                if iw == 0 || ih == 0 {
                    return Err(invalid("empty preprocessing image"));
                }
                match kind {
                    "resize" => {
                        if short {
                            if iw < ih {
                                h = (u64::from(w) * u64::from(ih) / u64::from(iw)) as u32;
                            } else {
                                w = (u64::from(h) * u64::from(iw) / u64::from(ih)) as u32;
                            }
                        }
                        img = img.resize_exact(w, h, filter(stage)?);
                    }
                    "pad_to_size" => {
                        let scale = (w as f64 / iw as f64).min(h as f64 / ih as f64);
                        let nw = ((iw as f64 * scale).round() as u32).clamp(1, w);
                        let nh = ((ih as f64 * scale).round() as u32).clamp(1, h);
                        let resized = img.resize_exact(nw, nh, filter(stage)?).to_rgb8();
                        let mut canvas =
                            RgbImage::from_pixel(w, h, Rgb(color(&stage["background_color"])?));
                        imageops::overlay(
                            &mut canvas,
                            &resized,
                            i64::from((w - nw) / 2),
                            i64::from((h - nh) / 2),
                        );
                        img = DynamicImage::ImageRgb8(canvas);
                    }
                    _ => {
                        let mut canvas = RgbImage::new(iw.max(w), ih.max(h));
                        imageops::overlay(
                            &mut canvas,
                            &img.to_rgb8(),
                            i64::from(w.saturating_sub(iw) / 2),
                            i64::from(h.saturating_sub(ih) / 2),
                        );
                        let x = ((canvas.width() - w) as f64 / 2.0).round_ties_even() as u32;
                        let y = ((canvas.height() - h) as f64 / 2.0).round_ties_even() as u32;
                        img = DynamicImage::ImageRgb8(canvas).crop_imm(x, y, w, h);
                    }
                }
            }
            _ => return Err(invalid(format!("unsupported preprocessing stage: {kind}"))),
        }
    }
    tensor.ok_or_else(|| invalid("preprocessing did not produce a tensor"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn explicit_resize_is_exact_and_to_tensor_scales() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 100, Rgb([255, 128, 0])));
        let t = apply(
            &img,
            &[
                json!({"type":"resize","size":[16,24]}),
                json!({"type":"to_tensor"}),
            ],
        )
        .unwrap();
        assert_eq!(t.shape(), [1, 3, 16, 24]);
        assert_eq!(t[[0, 0, 0, 0]], 1.0);
        assert!((t[[0, 1, 0, 0]] - 128.0 / 255.0).abs() < 1e-6);
    }
    #[test]
    fn camie_padding_uses_metadata_size_and_black_background() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 32, Rgb([255; 3])));
        let t=apply(&img,&[json!({"type":"pad_to_size","size":[512,512],"background_color":[0,0,0],"interpolation":"lanczos"}),json!({"type":"to_tensor"})]).unwrap();
        assert_eq!(t.shape(), [1, 3, 512, 512]);
        assert_eq!(t[[0, 0, 256, 0]], 0.0);
        assert_eq!(t[[0, 0, 256, 256]], 1.0);
    }
    #[test]
    fn scalar_resize_preserves_aspect_and_normalization_follows_scaling() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 20, Rgb([255; 3])));
        let t = apply(
            &img,
            &[
                json!({"type":"resize","size":8}),
                json!({"type":"to_tensor"}),
                json!({"type":"normalize","mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}),
            ],
        )
        .unwrap();
        assert_eq!(t.shape(), [1, 3, 16, 8]);
        assert_eq!(t[[0, 0, 0, 0]], 1.0);
    }
    #[test]
    fn clip_center_crop_selects_center_pixels() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(5, 2, |x, _| Rgb([(x * 50) as u8; 3])));
        let tensor = apply(
            &img,
            &[
                json!({"type":"center_crop","size":[2,2]}),
                json!({"type":"to_tensor"}),
            ],
        )
        .unwrap();
        assert_eq!(tensor.shape(), [1, 3, 2, 2]);
        assert!((tensor[[0, 0, 0, 0]] - 100.0 / 255.0).abs() < 1e-6);
        assert!((tensor[[0, 0, 0, 1]] - 150.0 / 255.0).abs() < 1e-6);
    }
    #[test]
    fn malformed_stages_fail_instead_of_silently_skipping() {
        let img = DynamicImage::new_rgb8(10, 10);
        for stage in [
            json!({"type":"resize","size":[]}),
            json!({"type":"resize","size":0}),
            json!({"type":"unknown"}),
            json!({"type":"normalize"}),
        ] {
            assert!(apply(&img, &[stage]).is_err());
        }
    }
}
