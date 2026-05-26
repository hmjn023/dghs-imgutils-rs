//! SCUNet モデルを用いた画像復元。
//!
//! 参照: <https://github.com/cszn/SCUNet>
//!
//! 対応モデル: GAN, PSNR

use image::DynamicImage;
use ndarray::{Array3, Array4, Axis};
use std::sync::Mutex;

use crate::hub::hf_hub_download;
use crate::inference::create_onnx_session;
use crate::utils::area::area_batch_run;

use super::RestoreError;

/// 利用可能な SCUNet モデル名。
pub const SCUNET_MODELS: &[&str] = &["GAN", "PSNR"];

const DEFAULT_TILE_SIZE: usize = 128;
const DEFAULT_TILE_OVERLAP: usize = 16;
const DEFAULT_BATCH_SIZE: usize = 4;

/// SCUNet モデルファイルのパスを取得する。
fn get_scunet_model_path(model: &str) -> Result<std::path::PathBuf, RestoreError> {
    if !SCUNET_MODELS.contains(&model) {
        return Err(RestoreError::InvalidArgument(format!(
            "Unknown SCUNet model: {model}. Choose from: {:?}",
            SCUNET_MODELS
        )));
    }
    let filename = format!("SCUNet-{model}.onnx");
    let path = hf_hub_download("deepghs/image_restoration", &filename, None, None)?;
    Ok(path)
}

/// SCUNet の RGB チャンネル処理（内部関数）。
fn scunet_process_rgb(
    model: &str,
    rgb_array: &Array3<f32>,
    tile_size: usize,
    tile_overlap: usize,
    batch_size: usize,
) -> Result<Array3<f32>, RestoreError> {
    let h = rgb_array.shape()[1];
    let w = rgb_array.shape()[2];

    let model_path = get_scunet_model_path(model)?;
    let session = create_onnx_session(model_path)?;
    let session = Mutex::new(session);

    let input_ = rgb_array
        .to_owned()
        .into_shape_with_order((1, 3, h, w))?;

    let output_ = area_batch_run(
        &input_,
        |ix| {
            let mut session = session.lock().unwrap();
            let tensor =
                ort::value::Tensor::from_array(ix.clone()).expect("Failed to create input tensor");
            let outputs = session
                .run(ort::inputs!["input" => tensor])
                .expect("SCUNet inference failed");
            let raw_output = outputs[0]
                .try_extract_tensor::<f32>()
                .expect("Failed to extract output tensor");
            let (out_shape, out_data) = raw_output;
            let out_shape_v: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
            let (b, c, h, w) = (out_shape_v[0], out_shape_v[1], out_shape_v[2], out_shape_v[3]);
            let out_data = out_data.to_owned();
            ndarray::Array4::from_shape_vec((b, c, h, w), out_data)
                .expect("Failed to create output array")
        },
        1,
        tile_size,
        tile_overlap,
        batch_size,
        3,
        3,
    );

    let output_ = output_.mapv(|v| v.clamp(0.0, 1.0));
    Ok(output_.index_axis(Axis(0), 0).to_owned())
}

/// SCUNet モデルで画像を復元する。
///
/// `model` には `"GAN"` または `"PSNR"` を指定する。
pub fn restore_with_scunet(
    image: &DynamicImage,
    model: &str,
) -> Result<DynamicImage, RestoreError> {
    let has_alpha = image.color().has_alpha();
    let img_rgba = image.to_rgba8();
    let (w, h) = img_rgba.dimensions();
    let h = h as usize;
    let w = w as usize;

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

        let enhanced_rgb = scunet_process_rgb(model, &rgb, DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP, DEFAULT_BATCH_SIZE)?;

        let mut alpha_stacked = Array3::<f32>::zeros((3, h, w));
        for c in 0..3 {
            alpha_stacked
                .slice_mut(ndarray::s![c, .., ..])
                .assign(&alpha);
        }
        let enhanced_alpha_stacked = scunet_process_rgb(
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
                let r =
                    (enhanced_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g =
                    (enhanced_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b =
                    (enhanced_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a =
                    (enhanced_alpha[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgba.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
            }
        }
        Ok(DynamicImage::ImageRgba8(out_rgba))
    } else {
        let rgb = input_array.slice(ndarray::s![0..3, .., ..]).to_owned();
        let enhanced_rgb = scunet_process_rgb(
            model,
            &rgb,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_OVERLAP,
            DEFAULT_BATCH_SIZE,
        )?;

        let mut out_rgb = image::ImageBuffer::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let r =
                    (enhanced_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g =
                    (enhanced_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b =
                    (enhanced_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
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
    fn test_scunet_invalid_model() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let result = restore_with_scunet(&img, "INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_scunet_model_list() {
        assert!(SCUNET_MODELS.contains(&"GAN"));
        assert!(SCUNET_MODELS.contains(&"PSNR"));
    }
}
