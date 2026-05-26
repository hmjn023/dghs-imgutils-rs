//! LineartAnime ニューラルネットワークによるエッジ検出。
//!
//! モデル: `deepghs/imgutils-models` の `lineart/lineart_anime.onnx`
//!
//! lineart.rs との差分:
//! - `align = 256`
//! - 正規化: `(x / 127.5) - 1.0` → `[-1, 1]`
//! - 後処理: `(x + 1.0) / 2.0` で [0, 1] に戻す

use crate::edge::canny::EdgeError;
use crate::edge::lineart::{blend_edge_mask, resize_image_align};
use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use std::sync::Mutex;

static LINEART_ANIME_SESSION: Lazy<Mutex<Option<Session>>> =
    Lazy::new(|| Mutex::new(None));

fn load_lineart_anime_session() -> Result<(), EdgeError> {
    if LINEART_ANIME_SESSION.lock().unwrap().is_some() {
        return Ok(());
    }
    let model_path =
        hf_hub_download("deepghs/imgutils-models", "lineart/lineart_anime.onnx", None, None)
            .map_err(|e| EdgeError::Processing(e.to_string()))?;
    let session = create_onnx_session(&model_path)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    *LINEART_ANIME_SESSION.lock().unwrap() = Some(session);
    Ok(())
}

/// LineartAnime モデルによるエッジマスク（float32 [H, W]）を返します。
///
/// * `path` - 入力画像パス
/// * `detect_resolution` - モデルへの入力解像度（デフォルト 512）
pub fn get_edge_by_lineart_anime(
    path: &str,
    detect_resolution: u32,
) -> Result<Vec<Vec<f32>>, EdgeError> {
    let img = image::open(path).map_err(EdgeError::ImageOpen)?;
    let (orig_w, orig_h) = img.dimensions();
    let rgb = force_image_background(&img, [255, 255, 255]);

    // align=256
    let resized = resize_image_align(&rgb, detect_resolution, 256);
    let (rw, rh) = resized.dimensions();

    // [-1, 1] 正規化: (x / 127.5) - 1.0
    let tensor = image_to_tensor_minus1_1(&resized, rw, rh);

    load_lineart_anime_session()?;
    let mut guard = LINEART_ANIME_SESSION.lock().unwrap();
    let session = guard.as_mut().unwrap();

    let tensor_val = ort::value::Tensor::from_array(tensor.clone())
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    let inputs = ort::inputs!["input" => tensor_val];
    let outputs = session
        .run(inputs)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;

    let (_, output_slice) = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    // [1, C, H', W'] → 後処理: (x + 1.0) / 2.0 → [0, 1]
    let channel_size = (rh * rw) as usize;
    let single_ch: Vec<f32> = output_slice[..channel_size]
        .iter()
        .map(|&v| (v + 1.0) / 2.0)
        .collect();

    // 元サイズにリサイズ
    let mask_img = mask_to_image(&single_ch, rw, rh);
    let scale = (orig_w as f32 / rw as f32).max(orig_h as f32 / rh as f32);
    let filter = if scale > 1.0 { FilterType::Lanczos3 } else { FilterType::CatmullRom };
    let mask_resized = DynamicImage::ImageRgb8(mask_img).resize_exact(orig_w, orig_h, filter);
    let mask_rgb = mask_resized.to_rgb8();

    // 1.0 - clip(0, 1) でエッジマスク反転
    let result: Vec<Vec<f32>> = (0..orig_h)
        .map(|y| {
            (0..orig_w)
                .map(|x| {
                    let v = mask_rgb.get_pixel(x, y)[0] as f32 / 255.0;
                    1.0 - v.clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect();

    Ok(result)
}

/// LineartAnime エッジ画像を PNG バイト列として返します。
pub fn edge_image_with_lineart_anime(
    path: &str,
    detect_resolution: u32,
    backcolor: [u8; 3],
    forecolor: Option<[u8; 3]>,
) -> Result<Vec<u8>, EdgeError> {
    let mask = get_edge_by_lineart_anime(path, detect_resolution)?;
    let img = image::open(path).map_err(EdgeError::ImageOpen)?;
    let rgb = force_image_background(&img, [255, 255, 255]);
    let rgb8 = rgb.to_rgb8();
    let h = mask.len() as u32;
    let w = if h > 0 { mask[0].len() as u32 } else { 0 };
    let output = blend_edge_mask(&rgb8, &mask, w, h, backcolor, forecolor);
    let mut buf = Vec::new();
    output
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    Ok(buf)
}

/// 画像を [-1, 1] 正規化して [1, 3, H, W] テンソルに変換
fn image_to_tensor_minus1_1(img: &DynamicImage, w: u32, h: u32) -> Array4<f32> {
    let rgb8 = img.to_rgb8();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 127.5 - 1.0;
        tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 127.5 - 1.0;
        tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 127.5 - 1.0;
    }
    tensor
}

/// 1D マスク値を [0, 255] RGB 画像（グレースケール）に変換
fn mask_to_image(mask: &[f32], w: u32, h: u32) -> image::RgbImage {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = (mask[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    img
}
