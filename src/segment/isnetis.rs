//! ISNetIS モデルを用いたアニメイラスト用の高精度キャラクターセグメンテーション（前景マスク抽出と合成）処理を提供します。

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{create_onnx_session, InferenceError};
use image::DynamicImage;
use ndarray::Array4;
use once_cell::sync::Lazy;
use std::sync::Mutex;

static ISNETIS_SESSION: Lazy<Mutex<Option<ort::session::Session>>> = Lazy::new(|| Mutex::new(None));

/// ISNetIS 用の画像前処理。
/// アスペクト比を維持して scale 解像度の長辺に合わせ、黒背景キャンバスの中央にパディング配置します。
pub fn preprocess_isnetis(
    image: &DynamicImage,
    scale: u32,
) -> Result<(Array4<f32>, (u32, u32), (u32, u32, u32, u32)), InferenceError> {
    let rgb_image = crate::image::force_image_background(image, [255, 255, 255]);
    let old_w = rgb_image.width();
    let old_h = rgb_image.height();

    // 長辺を scale に合わせ、アスペクト比を維持してリサイズ
    let (w, h) = if old_h > old_w {
        (
            ((scale as f32 * old_w as f32 / old_h as f32).round()) as u32,
            scale,
        )
    } else {
        (
            scale,
            ((scale as f32 * old_h as f32 / old_w as f32).round()) as u32,
        )
    };

    let resized = rgb_image.resize_exact(w, h, image::imageops::FilterType::Triangle);

    // パディング量の計算
    let ph = scale - h;
    let pw = scale - w;
    let pad_top = ph / 2;
    let pad_left = pw / 2;

    // scale x scale の黒背景キャンバスを作成
    let mut canvas = image::ImageBuffer::from_pixel(scale, scale, image::Rgb([0, 0, 0]));
    // リサイズした画像をパディング位置に貼り付け
    image::imageops::overlay(&mut canvas, &resized.to_rgb8(), pad_left as i64, pad_top as i64);

    // [1, 3, scale, scale] テンソルへ変換 (ピクセル値 / 255.0)
    let tensor = to_ndarray_chw(
        &image::DynamicImage::ImageRgb8(canvas),
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
    )
    .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    Ok((tensor, (old_w, old_h), (w, h, pad_left, pad_top)))
}

/// ISNetIS モデルを用いてキャラクター領域のアルファマスク（0.0 〜 1.0）を取得します。
pub fn get_isnetis_mask(
    image: &DynamicImage,
    scale: u32,
) -> Result<ndarray::Array2<f32>, InferenceError> {
    let mut lock = ISNETIS_SESSION.lock().map_err(|e| InferenceError::Initialization(e.to_string()))?;
    if lock.is_none() {
        let path = hf_hub_download("skytnt/anime-seg", "isnetis.onnx", None, None)
            .map_err(|e| InferenceError::Initialization(e.to_string()))?;
        *lock = Some(create_onnx_session(path)?);
    }
    let session = lock.as_mut().unwrap();

    // 1. 前処理
    let (input_tensor, old_size, (w, h, pad_left, pad_top)) = preprocess_isnetis(image, scale)?;

    // 2. 推論の実行 (入力名は 'img'、出力名は指定なしのため最初の出力を取得)
    let inputs = ort::inputs!["img" => ort::value::Tensor::from_array(input_tensor)?];
    let outputs = session.run(inputs)?;

    let output_val = outputs
        .iter()
        .next()
        .ok_or_else(|| InferenceError::InvalidShape("Missing output from ISNetIS".to_string()))?
        .1;

    let (shape, slice) = output_val
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let output_view_arr = ndarray::ArrayView::from_shape(shape_usize, slice)
        .map_err(|e| InferenceError::Initialization(e.to_string()))?;

    // 3. パディング部分をスライスで切り出し (リサイズ用に u8 2値化マスクとして作成)
    let mut sliced_mask_u8 = image::ImageBuffer::new(w, h);
    let view_shape = output_view_arr.shape();
    
    for y in 0..h {
        for x in 0..w {
            let val = match view_shape.len() {
                4 => output_view_arr[&[0, 0, (y + pad_top) as usize, (x + pad_left) as usize][..]],
                3 => output_view_arr[&[0, (y + pad_top) as usize, (x + pad_left) as usize][..]],
                2 => output_view_arr[&[(y + pad_top) as usize, (x + pad_left) as usize][..]],
                _ => return Err(InferenceError::InvalidShape("Unexpected ISNetIS output shape".to_string())),
            };
            let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
            sliced_mask_u8.put_pixel(x, y, image::Luma([pixel_val]));
        }
    }

    // 4. マスクを元の画像サイズ (old_w, old_h) に戻す
    let mask_img = image::DynamicImage::ImageLuma8(sliced_mask_u8);
    
    let resized_mask = mask_img.resize_exact(old_size.0, old_size.1, image::imageops::FilterType::Triangle);
    let resized_mask_luma = resized_mask.to_luma8();

    let mut final_mask = ndarray::Array2::<f32>::zeros((old_size.1 as usize, old_size.0 as usize));
    for y in 0..old_size.1 {
        for x in 0..old_size.0 {
            final_mask[[y as usize, x as usize]] = resized_mask_luma.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    Ok(final_mask)
}

/// キャラクター領域を分割し、指定された純色背景とアルファブレンド合成した RGB 画像を返します。
pub fn segment_with_isnetis(
    image: &DynamicImage,
    bg_color: [u8; 3],
    scale: u32,
) -> Result<(ndarray::Array2<f32>, DynamicImage), InferenceError> {
    let mask = get_isnetis_mask(image, scale)?;
    let old_w = image.width();
    let old_h = image.height();

    // 半透明PNGの場合でも動作するよう RGB 化して取得
    let rgb_img = image.to_rgb8();
    let mut canvas = image::ImageBuffer::new(old_w, old_h);

    for y in 0..old_h {
        for x in 0..old_w {
            let m_val = mask[[y as usize, x as usize]];
            let pixel = rgb_img.get_pixel(x, y);

            // アルファスタック背景ブレンド：前景 * alpha + 背景 * (1 - alpha)
            let r = ((pixel[0] as f32 * m_val) + (bg_color[0] as f32 * (1.0 - m_val))).round() as u8;
            let g = ((pixel[1] as f32 * m_val) + (bg_color[1] as f32 * (1.0 - m_val))).round() as u8;
            let b = ((pixel[2] as f32 * m_val) + (bg_color[2] as f32 * (1.0 - m_val))).round() as u8;

            canvas.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    Ok((mask, DynamicImage::ImageRgb8(canvas)))
}

/// キャラクター領域を分割し、背景を透過させた RGBA 画像を返します。
pub fn segment_rgba_with_isnetis(
    image: &DynamicImage,
    scale: u32,
) -> Result<(ndarray::Array2<f32>, DynamicImage), InferenceError> {
    let mask = get_isnetis_mask(image, scale)?;
    let old_w = image.width();
    let old_h = image.height();

    let rgb_img = image.to_rgb8();
    let mut canvas = image::ImageBuffer::new(old_w, old_h);

    for y in 0..old_h {
        for x in 0..old_w {
            let m_val = mask[[y as usize, x as usize]];
            let pixel = rgb_img.get_pixel(x, y);
            // マスク値をそのままアルファチャンネル（0..255）として埋め込み
            let alpha = (m_val * 255.0).round() as u8;

            canvas.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], alpha]));
        }
    }

    Ok((mask, DynamicImage::ImageRgba8(canvas)))
}
