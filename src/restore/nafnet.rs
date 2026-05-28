//! NafNet モデルを用いた画像復元。
//!
//! 参照: <https://github.com/megvii-research/NAFNet>
//!
//! 対応モデル: REDS, GoPro, SIDD

use image::DynamicImage;
use ndarray::{Array3, Axis};

use crate::hub::hf_hub_download;
use crate::inference::get_or_create_session;
use crate::utils::area::area_batch_run;

use super::RestoreError;

/// 利用可能な NafNet モデル名。
pub const NAFNET_MODELS: &[&str] = &["REDS", "GoPro", "SIDD"];

const DEFAULT_TILE_SIZE: usize = 256;
const DEFAULT_TILE_OVERLAP: usize = 16;
const DEFAULT_BATCH_SIZE: usize = 4;

/// NafNet モデルファイルのパスを取得する。
fn get_nafnet_model_path(model: &str) -> Result<std::path::PathBuf, RestoreError> {
    if !NAFNET_MODELS.contains(&model) {
        return Err(RestoreError::InvalidArgument(format!(
            "Unknown NafNet model: {model}. Choose from: {:?}",
            NAFNET_MODELS
        )));
    }
    let filename = format!("NAFNet-{model}-width64.onnx");
    let path = hf_hub_download("deepghs/image_restoration", &filename, None, None)?;
    Ok(path)
}

/// NafNet の RGB チャンネル処理（内部関数）。
fn nafnet_process_rgb(
    model: &str,
    rgb_array: &Array3<f32>,
    tile_size: usize,
    tile_overlap: usize,
    batch_size: usize,
) -> Result<Array3<f32>, RestoreError> {
    let h = rgb_array.shape()[1];
    let w = rgb_array.shape()[2];

    let model_path = get_nafnet_model_path(model)?;
    let session = get_or_create_session(model_path)?;

    let input_ = rgb_array.to_owned().into_shape_with_order((1, 3, h, w))?;

    let output_ = area_batch_run(
        &input_,
        |ix| {
            let mut session = session
                .lock()
                .map_err(|e| RestoreError::Shape(format!("Session lock poisoned: {e}")))?;
            let tensor = ort::value::Tensor::from_array(ix.clone()).map_err(RestoreError::Ort)?;
            let outputs = session
                .run(ort::inputs!["input" => tensor])
                .map_err(RestoreError::Ort)?;
            let raw_output = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(RestoreError::Ort)?;
            let (out_shape, out_data) = raw_output;
            let out_shape_v: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
            let (b, c, h, w) = (
                out_shape_v[0],
                out_shape_v[1],
                out_shape_v[2],
                out_shape_v[3],
            );
            let out_data = out_data.to_owned();
            ndarray::Array4::from_shape_vec((b, c, h, w), out_data)
                .map_err(|e| RestoreError::Shape(e.to_string()))
        },
        1,
        tile_size,
        tile_overlap,
        batch_size,
        3,
        3,
    )?;

    let output_ = output_.mapv(|v| v.clamp(0.0, 1.0));
    Ok(output_.index_axis(Axis(0), 0).to_owned())
}

/// NafNet モデルで画像を復元する。
///
/// `model` には `"REDS"`, `"GoPro"`, `"SIDD"` のいずれかを指定する。
pub fn restore_with_nafnet(
    image: &DynamicImage,
    model: &str,
) -> Result<DynamicImage, RestoreError> {
    let has_alpha = image.color().has_alpha();
    let img_rgba = image.to_rgba8();
    let (w, h) = img_rgba.dimensions();
    let h = h as usize;
    let w = w as usize;

    // RGBA → float array [4, H, W]
    let mut input_array = Array3::<f32>::zeros((4, h, w));
    for (x, y, pixel) in img_rgba.enumerate_pixels() {
        input_array[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        input_array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        input_array[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        input_array[[3, y as usize, x as usize]] = pixel[3] as f32 / 255.0;
    }

    if has_alpha {
        let rgb = input_array.slice(ndarray::s![0..3, .., ..]).to_owned();
        let alpha = input_array.slice(ndarray::s![3, .., ..]).to_owned();

        let enhanced_rgb = nafnet_process_rgb(
            model,
            &rgb,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_OVERLAP,
            DEFAULT_BATCH_SIZE,
        )?;

        // Process alpha through same model (stack as RGB, average channels)
        let mut alpha_stacked = Array3::<f32>::zeros((3, h, w));
        for c in 0..3 {
            alpha_stacked
                .slice_mut(ndarray::s![c, .., ..])
                .assign(&alpha);
        }
        let enhanced_alpha_stacked = nafnet_process_rgb(
            model,
            &alpha_stacked,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_OVERLAP,
            DEFAULT_BATCH_SIZE,
        )?;
        let mut enhanced_alpha = Array3::<f32>::zeros((1, h, w));
        for y in 0..h {
            for x in 0..w {
                enhanced_alpha[[0, y, x]] = (enhanced_alpha_stacked[[0, y, x]]
                    + enhanced_alpha_stacked[[1, y, x]]
                    + enhanced_alpha_stacked[[2, y, x]])
                    / 3.0;
            }
        }

        let mut out_rgba = image::ImageBuffer::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let r = (enhanced_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (enhanced_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (enhanced_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a = (enhanced_alpha[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgba.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
            }
        }
        Ok(DynamicImage::ImageRgba8(out_rgba))
    } else {
        let rgb = input_array.slice(ndarray::s![0..3, .., ..]).to_owned();
        let enhanced_rgb = nafnet_process_rgb(
            model,
            &rgb,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_OVERLAP,
            DEFAULT_BATCH_SIZE,
        )?;

        let mut out_rgb = image::ImageBuffer::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let r = (enhanced_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (enhanced_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (enhanced_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }
        Ok(DynamicImage::ImageRgb8(out_rgb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_nafnet_invalid_model() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let result = restore_with_nafnet(&img, "INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_nafnet_model_list() {
        assert!(NAFNET_MODELS.contains(&"REDS"));
        assert!(NAFNET_MODELS.contains(&"GoPro"));
        assert!(NAFNET_MODELS.contains(&"SIDD"));
    }
}
