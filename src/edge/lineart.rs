//! Lineart ニューラルネットワークによるエッジ検出。
//!
//! モデル: `deepghs/imgutils-models` の `lineart/lineart.onnx` または `lineart_coarse.onnx`
//!
//! ## resize_image のアルゴリズム（Python バグ挙動に合わせた再現）
//!
//! Python の `resize_image(img, resolution=512, align=64)` は以下の動作をします:
//! ```python
//! k = float(resolution) / min(height, width)  # ← 計算するが align がある場合は使われない
//! new_height = int(math.ceil(height / align)) * align
//! new_width  = int(math.ceil(width  / align)) * align
//! ```
//! つまり `align` が指定されると元のサイズを align の倍数に切り上げるだけです。
//! `resolution` パラメータは align なしの場合のみ有効です。

use crate::edge::canny::EdgeError;
use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::create_onnx_session;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use once_cell::sync::Lazy;
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

static LINEART_CACHE: Lazy<Mutex<HashMap<String, Session>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Python の `resize_image(img, resolution, align=64)` と等価なリサイズを行います。
///
/// align が指定されている場合は元サイズの align 倍数への切り上げ（Python バグ挙動再現）。
/// align が 0 の場合は短辺を resolution に合わせてスケールします。
pub(crate) fn resize_image_align(img: &DynamicImage, resolution: u32, align: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let (new_w, new_h) = if align > 0 {
        // ceil(x / align) * align
        let new_h = ((h as f32 / align as f32).ceil() as u32) * align;
        let new_w = ((w as f32 / align as f32).ceil() as u32) * align;
        (new_w, new_h)
    } else {
        let k = resolution as f32 / w.min(h) as f32;
        ((w as f32 * k).round() as u32, (h as f32 * k).round() as u32)
    };

    // Python の cv2_resize: 拡大なら LANCZOS4、縮小なら AREA
    let scale = (new_w as f32 / w as f32).max(new_h as f32 / h as f32);
    let filter = if scale > 1.0 {
        FilterType::Lanczos3
    } else {
        FilterType::CatmullRom
    };
    img.resize_exact(new_w, new_h, filter)
}

fn load_lineart_session(coarse: bool) -> Result<(), EdgeError> {
    let key = if coarse { "coarse" } else { "normal" };
    {
        if LINEART_CACHE.lock().unwrap().contains_key(key) {
            return Ok(());
        }
    }
    let filename = if coarse {
        "lineart/lineart_coarse.onnx"
    } else {
        "lineart/lineart.onnx"
    };
    let model_path = hf_hub_download("deepghs/imgutils-models", filename, None, None)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    let session =
        create_onnx_session(&model_path).map_err(|e| EdgeError::Processing(e.to_string()))?;
    LINEART_CACHE
        .lock()
        .unwrap()
        .insert(key.to_string(), session);
    Ok(())
}

/// Lineart モデルによるエッジマスク（float32 [H, W]）を返します。
///
/// * `path` - 入力画像パス
/// * `coarse` - `true` で coarse モデル（より太くリッチな線）
/// * `detect_resolution` - モデルへの入力解像度（デフォルト 512）
pub fn get_edge_by_lineart(
    path: &str,
    coarse: bool,
    detect_resolution: u32,
) -> Result<Vec<Vec<f32>>, EdgeError> {
    let img = image::open(path).map_err(EdgeError::ImageOpen)?;
    let (orig_w, orig_h) = img.dimensions();
    let rgb = force_image_background(&img, [255, 255, 255]);

    // Python の resize_image(img, detect_resolution, align=64)
    let resized = resize_image_align(&rgb, detect_resolution, 64);
    let (rw, rh) = resized.dimensions();

    // [0, 1] 正規化 + CHW
    let tensor = image_to_tensor_01(&resized, rw, rh);

    load_lineart_session(coarse)?;
    let key = if coarse { "coarse" } else { "normal" };
    let mut cache = LINEART_CACHE.lock().unwrap();
    let session = cache.get_mut(key).unwrap();

    let tensor_val = ort::value::Tensor::from_array(tensor.clone())
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    let inputs =
        ort::inputs!["input" => tensor_val].map_err(|e| EdgeError::Processing(e.to_string()))?;
    let outputs = session
        .run(inputs)
        .map_err(|e| EdgeError::Processing(e.to_string()))?;

    let (_, output_slice) = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| EdgeError::Processing(e.to_string()))?;
    let raw = output_slice;

    // [1, 3, H', W'] → 平均をとって [H', W'] の 2D マスク
    // lineart のモデル出力は実質1チャンネルを3ch複製しているので最初のchを使う
    let channel_size = (rh * rw) as usize;
    let single_ch: Vec<f32> = raw[..channel_size].to_vec();

    // 元サイズにリサイズ（Python の cv2_resize 相当）
    let mask_img = mask_to_image(&single_ch, rw, rh);
    let scale = (orig_w as f32 / rw as f32).max(orig_h as f32 / rh as f32);
    let filter = if scale > 1.0 {
        FilterType::Lanczos3
    } else {
        FilterType::CatmullRom
    };
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

/// Lineart エッジ画像を PNG バイト列として返します。
pub fn edge_image_with_lineart(
    path: &str,
    coarse: bool,
    detect_resolution: u32,
    backcolor: [u8; 3],
    forecolor: Option<[u8; 3]>,
) -> Result<Vec<u8>, EdgeError> {
    let mask = get_edge_by_lineart(path, coarse, detect_resolution)?;
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

// -------- 内部ユーティリティ --------

/// 画像を [0, 1] 正規化して [1, 3, H, W] テンソルに変換
fn image_to_tensor_01(img: &DynamicImage, w: u32, h: u32) -> Array4<f32> {
    let rgb8 = img.to_rgb8();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }
    tensor
}

/// 1D マスク値を [0, 255] RGB 画像（グレースケール）に変換
fn mask_to_image(mask: &[f32], w: u32, h: u32) -> image::RgbImage {
    let mut img = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = (mask[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    img
}

/// エッジマスクと元画像・背景色を合成して RGB 画像を生成
pub(crate) fn blend_edge_mask(
    src: &image::RgbImage,
    mask: &[Vec<f32>],
    w: u32,
    h: u32,
    backcolor: [u8; 3],
    forecolor: Option<[u8; 3]>,
) -> DynamicImage {
    let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(w, h, Rgb(backcolor));
    for y in 0..h {
        for x in 0..w {
            let edge_val = mask[y as usize][x as usize].clamp(0.0, 1.0);
            if edge_val > 0.0 {
                let fore_pix = if let Some(fc) = forecolor {
                    Rgb(fc)
                } else {
                    *src.get_pixel(x, y)
                };
                let bg = canvas.get_pixel(x, y).0;
                let blended = [
                    (bg[0] as f32 * (1.0 - edge_val) + fore_pix[0] as f32 * edge_val).round() as u8,
                    (bg[1] as f32 * (1.0 - edge_val) + fore_pix[1] as f32 * edge_val).round() as u8,
                    (bg[2] as f32 * (1.0 - edge_val) + fore_pix[2] as f32 * edge_val).round() as u8,
                ];
                canvas.put_pixel(x, y, Rgb(blended));
            }
        }
    }
    DynamicImage::ImageRgb8(canvas)
}
