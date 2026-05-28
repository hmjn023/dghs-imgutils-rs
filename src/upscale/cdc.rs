//! CDC (Coupled Dual Convolution) モデルによるアニメ画像の 4x 超解像。
//!
//! 参照: <https://github.com/7eu7d7>
//! モデル: <https://huggingface.co/deepghs/cdc_anime_onnx>

use image::DynamicImage;
use ndarray::{Array3, Array4, Axis, s};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::hub::hf_hub_download;
use crate::inference::get_or_create_session;
use crate::utils::area::area_batch_run;

use super::UpscaleError;

/// CDC モデルの入力アライメント単位（16 ピクセル）。
const CDC_INPUT_UNIT: usize = 16;

/// デフォルトの CDC モデル名。
pub const DEFAULT_CDC_MODEL: &str = "HGSR-MHR-anime-aug_X4_320";

/// 利用可能な主な CDC モデル名。
pub const CDC_MODELS: &[&str] = &["HGSR-MHR-anime-aug_X4_320"];

/// CDC モデルのメタデータ（スケールファクター）。
struct CdcModelMeta {
    scale: usize,
}

/// CDC モデルのメタデータキャッシュ。
/// モデル名をキーとして、スケールファクターをキャッシュする。
static CDC_META_CACHE: Lazy<Mutex<HashMap<String, Arc<CdcModelMeta>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// CDC ONNX モデルのセッションとスケールファクターを取得する（キャッシュ付き）。
fn get_cdc_model(model: &str) -> Result<(Arc<Mutex<ort::session::Session>>, usize), UpscaleError> {
    // メタデータキャッシュを確認
    {
        let cache = CDC_META_CACHE
            .lock()
            .map_err(|e| UpscaleError::Shape(format!("CDC meta cache lock poisoned: {e}")))?;
        if let Some(meta) = cache.get(model) {
            let model_path = hf_hub_download(
                "deepghs/cdc_anime_onnx",
                &format!("{model}.onnx"),
                None,
                None,
            )?;
            let session = get_or_create_session(model_path)?;
            return Ok((session, meta.scale));
        }
    }

    // キャッシュなし: セッション作成 + warm-up 推論
    let model_path = hf_hub_download(
        "deepghs/cdc_anime_onnx",
        &format!("{model}.onnx"),
        None,
        None,
    )?;
    let session = get_or_create_session(&model_path)?;

    let scale = {
        let input_data = Array4::<f32>::zeros((1, 3, CDC_INPUT_UNIT, CDC_INPUT_UNIT));
        let tensor = ort::value::Tensor::from_array(input_data)?;
        let mut session_guard = session
            .lock()
            .map_err(|e| UpscaleError::Shape(format!("Session lock poisoned: {e}")))?;
        let outputs = session_guard
            .run(ort::inputs!["input" => tensor])
            .map_err(|e| UpscaleError::Shape(format!("CDC warm-up failed: {e}")))?;
        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| UpscaleError::Shape(format!("CDC output extract failed: {e}")))?;
        let (out_shape, _data) = output_tensor;
        let shape: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();

        if shape.len() != 6 {
            return Err(UpscaleError::Shape(format!(
                "CDC model expected 6D output, got {:?}",
                shape
            )));
        }
        let scale_h = shape[2];
        let scale_w = shape[4];
        if scale_h != scale_w {
            return Err(UpscaleError::Shape(format!(
                "CDC model scale mismatch: height_scale={scale_h}, width_scale={scale_w}"
            )));
        }
        scale_h as usize
    };

    // メタデータをキャッシュに保存
    {
        let mut cache = CDC_META_CACHE
            .lock()
            .map_err(|e| UpscaleError::Shape(format!("CDC meta cache lock poisoned: {e}")))?;
        cache.insert(model.to_string(), Arc::new(CdcModelMeta { scale }));
    }

    Ok((session, scale))
}

/// CDC の RGB アップスケール処理（内部関数）。
fn cdc_upscale_rgb(
    model: &str,
    rgb_array: &Array3<f32>,
    tile_size: usize,
    tile_overlap: usize,
    batch_size: usize,
) -> Result<Array3<f32>, UpscaleError> {
    let h = rgb_array.shape()[1];
    let w = rgb_array.shape()[2];
    let (session, scale) = get_cdc_model(model)?;

    let input_ = rgb_array.to_owned().into_shape_with_order((1, 3, h, w))?;

    let output_ = area_batch_run(
        &input_,
        |ix| {
            // Align to 16-pixel boundary
            let (_, _, ih, iw) = (ix.shape()[0], ix.shape()[1], ix.shape()[2], ix.shape()[3]);
            let pad_h = (CDC_INPUT_UNIT - ih % CDC_INPUT_UNIT) % CDC_INPUT_UNIT;
            let pad_w = (CDC_INPUT_UNIT - iw % CDC_INPUT_UNIT) % CDC_INPUT_UNIT;

            let padded = if pad_h > 0 || pad_w > 0 {
                let mut padded = Array4::<f32>::zeros((1, 3, ih + pad_h, iw + pad_w));
                // Reflect padding
                for c in 0..3 {
                    for y in 0..ih + pad_h {
                        for x in 0..iw + pad_w {
                            let sy = if y < ih { y } else { 2 * ih - y - 1 };
                            let sx = if x < iw { x } else { 2 * iw - x - 1 };
                            let sy = sy.min(ih - 1);
                            let sx = sx.min(iw - 1);
                            padded[[0, c, y, x]] = ix[[0, c, sy, sx]];
                        }
                    }
                }
                padded
            } else {
                ix.clone()
            };

            let tensor = ort::value::Tensor::from_array(padded).map_err(UpscaleError::Ort)?;
            let mut session_guard = session
                .lock()
                .map_err(|e| UpscaleError::Shape(format!("Session lock poisoned: {e}")))?;
            let outputs = session_guard
                .run(ort::inputs!["input" => tensor])
                .map_err(UpscaleError::Ort)?;
            let raw_output = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(UpscaleError::Ort)?;
            let (out_shape, out_data) = raw_output;
            let out_shape_vec: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
            let data = ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&out_shape_vec),
                out_data.to_owned(),
            )
            .map_err(|e| UpscaleError::Shape(e.to_string()))?;

            // Reshape from [1, 3, scale, H, scale, W] → [1, 3, scale*H, scale*W]
            let s = data.shape();
            let (b, c, sc1, h_, sc2, w_) = (s[0], s[1], s[2], s[3], s[4], s[5]);
            let reshaped = data
                .into_shape_with_order(ndarray::IxDyn(&[b, c, sc1 * h_, sc2 * w_]))
                .map_err(|e| UpscaleError::Shape(e.to_string()))?;

            // Crop to actual scaled size
            let actual_h = scale * ih;
            let actual_w = scale * iw;
            Ok::<_, UpscaleError>(
                reshaped
                    .slice(s![.., .., ..actual_h, ..actual_w])
                    .to_owned(),
            )
        },
        scale,
        tile_size,
        tile_overlap,
        batch_size,
        3,
        3,
    )?;

    let clipped = output_.mapv(|v| v.clamp(0.0, 1.0));
    Ok(clipped.index_axis(Axis(0), 0).to_owned())
}

/// CDC モデルで画像を 4x アップスケールする。
///
/// # Arguments
///
/// * `image` - 入力画像
/// * `model` - モデル名（デフォルト: `"HGSR-MHR-anime-aug_X4_320"`）
/// * `tile_size` - タイルサイズ（デフォルト: 512）
/// * `tile_overlap` - タイルオーバーラップ（デフォルト: 64）
/// * `batch_size` - バッチサイズ（デフォルト: 1）
pub fn upscale_with_cdc(
    image: &DynamicImage,
    model: &str,
    tile_size: u32,
    tile_overlap: u32,
    batch_size: u32,
) -> Result<DynamicImage, UpscaleError> {
    let tile_size = tile_size as usize;
    let tile_overlap = tile_overlap as usize;
    let batch_size = batch_size as usize;

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

        let upscaled_rgb = cdc_upscale_rgb(model, &rgb, tile_size, tile_overlap, batch_size)?;
        let (_, out_h, out_w) = (
            upscaled_rgb.shape()[0],
            upscaled_rgb.shape()[1],
            upscaled_rgb.shape()[2],
        );

        let mut alpha_stacked = Array3::<f32>::zeros((3, h, w));
        for c in 0..3 {
            alpha_stacked
                .slice_mut(ndarray::s![c, .., ..])
                .assign(&alpha);
        }
        let upscaled_alpha_stacked =
            cdc_upscale_rgb(model, &alpha_stacked, tile_size, tile_overlap, batch_size)?;
        let mut upscaled_alpha = Array3::<f32>::zeros((1, out_h, out_w));
        for y in 0..out_h {
            for x in 0..out_w {
                upscaled_alpha[[0, y, x]] = (upscaled_alpha_stacked[[0, y, x]]
                    + upscaled_alpha_stacked[[1, y, x]]
                    + upscaled_alpha_stacked[[2, y, x]])
                    / 3.0;
            }
        }

        let mut out_rgba = image::ImageBuffer::new(out_w as u32, out_h as u32);
        for y in 0..out_h {
            for x in 0..out_w {
                let r = (upscaled_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (upscaled_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (upscaled_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a = (upscaled_alpha[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgba.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
            }
        }
        Ok(DynamicImage::ImageRgba8(out_rgba))
    } else {
        let rgb = input_array.slice(ndarray::s![0..3, .., ..]).to_owned();
        let upscaled_rgb = cdc_upscale_rgb(model, &rgb, tile_size, tile_overlap, batch_size)?;
        let (_, out_h, out_w) = (
            upscaled_rgb.shape()[0],
            upscaled_rgb.shape()[1],
            upscaled_rgb.shape()[2],
        );

        let mut out_rgb = image::ImageBuffer::new(out_w as u32, out_h as u32);
        for y in 0..out_h {
            for x in 0..out_w {
                let r = (upscaled_rgb[[0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (upscaled_rgb[[1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (upscaled_rgb[[2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }
        Ok(DynamicImage::ImageRgb8(out_rgb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdc_model_list() {
        assert!(CDC_MODELS.contains(&"HGSR-MHR-anime-aug_X4_320"));
    }

    #[test]
    fn test_cdc_invalid_model() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
        let result = upscale_with_cdc(&img, "nonexistent_model", 512, 64, 1);
        assert!(result.is_err());
    }
}
