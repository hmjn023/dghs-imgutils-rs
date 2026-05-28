//! 画像操作およびテンソル（ndarray）変換用の共通ユーティリティを提供します。

pub mod encode;
pub mod error;
pub mod layer;
pub mod load;

pub use encode::{rgb_decode, rgb_encode};
pub use error::ImageError;
pub use layer::{LayerItemBuilder, istack};
pub use load::{
    ImageSource, ImageSourceKind, check_image_source, load_image, load_image_with_background,
};

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;
use std::io::Read;

/// アスペクト比を維持しながら画像をリサイズし、余りを指定色でパディングします。
///
/// Pillow の `pad_image_to_size` と等価な処理を行います。
///
/// # 引数
///
/// * `img` - 入力画像
/// * `target_w` - 目標とする幅
/// * `target_h` - 目標とする高さ
/// * `bg_color` - 余白を埋める RGB の色（例: `[255, 255, 255]`）
pub fn pad_image_to_size(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    bg_color: [u8; 3],
) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w == target_w && h == target_h {
        return img.clone();
    }

    // アスペクト比を維持するための比率計算
    let scale_w = target_w as f32 / w as f32;
    let scale_h = target_h as f32 / h as f32;
    let scale = scale_w.min(scale_h);

    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    // リサイズを実行（Bilinear 相当の Triangle フィルタを採用）
    let resized = img.resize(new_w, new_h, FilterType::Triangle);

    // 背景色のキャンバスを作成
    let mut canvas = ImageBuffer::from_pixel(target_w, target_h, Rgb(bg_color));

    // 画像が中央になるようにオフセットを計算
    let x_offset = (target_w - new_w) / 2;
    let y_offset = (target_h - new_h) / 2;

    // 貼り付け
    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset as i64,
        y_offset as i64,
    );

    DynamicImage::ImageRgb8(canvas)
}

/// 画像がアルファチャンネルを持つ場合に、指定された背景色でアルファブレンドします。
///
/// Pillow の `add_background_for_rgba` と等価な処理を行います。
///
/// # 引数
///
/// * `img` - 入力画像
/// * `bg_color` - 背景色（例: `[255, 255, 255]`）
pub fn force_image_background(img: &DynamicImage, bg_color: [u8; 3]) -> DynamicImage {
    if img.color().has_alpha() {
        let (w, h) = img.dimensions();
        let mut canvas = ImageBuffer::from_pixel(w, h, Rgb(bg_color));
        let rgba_img = img.to_rgba8();

        for y in 0..h {
            for x in 0..w {
                let rgba_pixel = rgba_img.get_pixel(x, y);
                let alpha = rgba_pixel[3] as f32 / 255.0;
                let rgb_pixel = canvas.get_pixel_mut(x, y);

                rgb_pixel[0] = ((rgba_pixel[0] as f32 * alpha)
                    + (bg_color[0] as f32 * (1.0 - alpha)))
                    .round() as u8;
                rgb_pixel[1] = ((rgba_pixel[1] as f32 * alpha)
                    + (bg_color[1] as f32 * (1.0 - alpha)))
                    .round() as u8;
                rgb_pixel[2] = ((rgba_pixel[2] as f32 * alpha)
                    + (bg_color[2] as f32 * (1.0 - alpha)))
                    .round() as u8;
            }
        }
        DynamicImage::ImageRgb8(canvas)
    } else {
        DynamicImage::ImageRgb8(img.to_rgb8())
    }
}

/// `DynamicImage` を ONNX Runtime 入力用の `[1, 3, H, W]` テンソル（ndarray）に変換します。
///
/// ピクセル値は `0.0` 〜 `1.0` にスケーリングされた後、指定された平均 (`mean`) と標準偏差 (`std`) で正規化されます。
///
/// # 引数
///
/// * `img` - 変換対象の画像
/// * `mean` - 各チャンネル (R, G, B) の平均値
/// * `std` - 各チャンネル (R, G, B) の標準偏差
pub fn to_ndarray_chw(
    img: &DynamicImage,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Array4<f32>, ImageError> {
    let (width, height) = img.dimensions();
    let mut array = Array4::zeros((1, 3, height as usize, width as usize));

    let rgb_img = img.to_rgb8();
    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // 標準的な正規化処理: (x - mean) / std
        let r = (r - mean[0]) / std[0];
        let g = (g - mean[1]) / std[1];
        let b = (b - mean[2]) / std[2];

        // [Batch, Channel, Height, Width] の順序で配置
        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }

    Ok(array)
}

// ─── Batch load ───

/// 複数の画像ソースを一括で読み込みます。
///
/// * `sources` — 画像ソースのスライス
pub fn load_images(sources: &[ImageSource]) -> Result<Vec<DynamicImage>, ImageError> {
    sources.iter().map(|s| load_image(s)).collect()
}

// ─── Alpha channel check ───

/// 画像がアルファチャンネルを持つかどうかを判定します。
pub fn has_alpha_channel(img: &DynamicImage) -> bool {
    img.color().has_alpha()
}

// ─── Blob URL ───

/// 画像を Base64 エンコードされたデータ URI (blob URL) に変換します。
///
/// * `img` — 入力画像
/// * `format` — 画像形式 (`"jpg"`, `"png"`, `"webp"` 等)
pub fn to_blob_url(img: &DynamicImage, format: &str) -> Result<String, ImageError> {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    let format_upper = format.to_uppercase();
    let format_normalized = match format_upper.as_str() {
        "JPG" => "JPEG",
        other => other,
    };

    let mime = match format_normalized {
        "JPEG" => "image/jpeg",
        "PNG" => "image/png",
        "WEBP" => "image/webp",
        "GIF" => "image/gif",
        "BMP" => "image/bmp",
        _ => {
            return Err(ImageError::InvalidArgument(format!(
                "Unsupported format: {}",
                format
            )));
        }
    };

    let image_format = match format_normalized {
        "JPEG" => image::ImageFormat::Jpeg,
        "PNG" => image::ImageFormat::Png,
        "WEBP" => image::ImageFormat::WebP,
        "GIF" => image::ImageFormat::Gif,
        "BMP" => image::ImageFormat::Bmp,
        _ => {
            return Err(ImageError::InvalidArgument(format!(
                "Unsupported format: {}",
                format
            )));
        }
    };

    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image_format)
        .map_err(|e| ImageError::Processing(e.to_string()))?;
    let encoded = STANDARD.encode(buf.into_inner());

    Ok(format!("data:{};base64,{}", mime, encoded))
}

/// Base64 デコードされたデータ URI (blob URL) から画像を読み込みます。
pub fn load_image_from_blob_url(blob_url: &str) -> Result<DynamicImage, ImageError> {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;

    let parts: Vec<&str> = blob_url.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(ImageError::InvalidArgument(
            "Invalid blob URL format".to_string(),
        ));
    }

    let header = parts[0];
    let data = parts[1];

    // エンコーディングを確認 (base64 のみサポート)
    let is_base64 = header.contains(";base64");
    if !is_base64 {
        return Err(ImageError::InvalidArgument(
            "Only base64 encoding is supported".to_string(),
        ));
    }

    let decoded = STANDARD
        .decode(data)
        .map_err(|e| ImageError::InvalidArgument(format!("Base64 decode failed: {}", e)))?;

    image::load_from_memory(&decoded).map_err(|e| ImageError::Processing(e.to_string()))
}

// ─── URL download ───

/// URL から画像をダウンロードします。
///
/// GitHub URL (`?raw=True` 付与) や HuggingFace URL (`/blob/` → `/resolve/` 変換) を
/// 自動変換します。
pub fn download_image_from_url(url: &str) -> Result<DynamicImage, ImageError> {
    let actual_url = transform_url(url);
    let mut bytes = Vec::new();
    ureq::get(&actual_url)
        .call()
        .map_err(|e| {
            ImageError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| ImageError::Io(e))?;
    image::load_from_memory(&bytes).map_err(|e| ImageError::Processing(e.to_string()))
}

/// GitHub / HuggingFace の URL をダウンロード可能な raw URL に変換します。
fn transform_url(url: &str) -> String {
    // GitHub: ?raw=True を追加
    if url.contains("github.com") && url.contains("/blob/") && !url.contains("?raw=") {
        return format!("{}?raw=True", url);
    }
    // HuggingFace: /blob/ → /resolve/
    if url.contains("huggingface.co") && url.contains("/blob/") {
        return url.replace("/blob/", "/resolve/");
    }
    url.to_string()
}

// ─── Grid background ───

/// チェッカーボードパターンのグリッド背景画像を生成します。
///
/// 透過画像の可視化に使用します。
///
/// * `width` — 画像幅
/// * `height` — 画像高さ
/// * `step` — グリッドのマス目サイズ (`None` で自動計算)
/// * `forecolor` — 前景色 (グリッド線)
/// * `backcolor` — 背景色
pub fn grid_background(
    width: u32,
    height: u32,
    step: Option<u32>,
    forecolor: [u8; 3],
    backcolor: [u8; 3],
) -> DynamicImage {
    let s = step.unwrap_or_else(|| ((width as f64 * height as f64 / 800.0).sqrt() as u32).max(1));
    if s == 0 {
        return DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb(backcolor),
        ));
    }

    let mut img = image::RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let color = if (x / s + y / s) % 2 == 0 {
                backcolor
            } else {
                forecolor
            };
            img.put_pixel(x, y, image::Rgb(color));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// 画像の透過部分をグリッド背景で表示します。
///
/// * `img` — 入力画像 (RGBA)
/// * `step` — グリッドサイズ
/// * `forecolor` — グリッド前景色
/// * `backcolor` — グリッド背景色
pub fn grid_transparent(
    img: &DynamicImage,
    step: Option<u32>,
    forecolor: [u8; 3],
    backcolor: [u8; 3],
) -> DynamicImage {
    let (w, h) = img.dimensions();
    let bg = grid_background(w, h, step, forecolor, backcolor);
    let layers = vec![
        LayerItemBuilder::from_image(bg).with_alpha(1.0),
        LayerItemBuilder::from_image(img.clone()).with_alpha(1.0),
    ];
    match istack(layers, Some((w, h))) {
        Ok(rgba) => DynamicImage::ImageRgba8(rgba),
        Err(_) => img.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_pad_image_to_size() {
        // 100x50 の赤いダミー画像を作成
        let mut rgb_img = RgbImage::new(100, 50);
        for pixel in rgb_img.pixels_mut() {
            *pixel = Rgb([255, 0, 0]);
        }
        let img = DynamicImage::ImageRgb8(rgb_img);

        // 200x200 に白い背景でパディング
        let padded = pad_image_to_size(&img, 200, 200, [255, 255, 255]);

        assert_eq!(padded.width(), 200);
        assert_eq!(padded.height(), 200);

        let padded_rgb = padded.to_rgb8();
        // 左上の余白は背景色の白になっているはず
        assert_eq!(padded_rgb.get_pixel(0, 0), &Rgb([255, 255, 255]));
        // 中央は元の赤い画像がリサイズされて配置されているはず
        assert_eq!(padded_rgb.get_pixel(100, 100), &Rgb([255, 0, 0]));
    }

    #[test]
    fn test_to_ndarray_chw() {
        // 2x2 の青いダミー画像を作成
        let mut rgb_img = RgbImage::new(2, 2);
        for pixel in rgb_img.pixels_mut() {
            *pixel = Rgb([0, 0, 255]);
        }
        let img = DynamicImage::ImageRgb8(rgb_img);

        // mean = 0, std = 1 の正規化なし相当でテンソル変換
        let tensor = to_ndarray_chw(&img, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]).unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 2, 2]);

        // 青チャンネル (C=2) の値は 1.0 にスケーリングされているはず
        assert_eq!(tensor[[0, 2, 0, 0]], 1.0);
        // 赤チャンネル (C=0) は 0.0 のはず
        assert_eq!(tensor[[0, 0, 0, 0]], 0.0);
    }

    #[test]
    fn test_force_image_background() {
        use image::RgbaImage;
        // 2x2 の半透明な赤い RGBA 画像を作成
        let mut rgba_img = RgbaImage::new(2, 2);
        for pixel in rgba_img.pixels_mut() {
            *pixel = image::Rgba([255, 0, 0, 128]); // 半透明の赤
        }
        let img = DynamicImage::ImageRgba8(rgba_img);

        // 白背景 [255, 255, 255] の上に半透明の赤をアルファブレンド
        // 計算式: pixel * (alpha/255) + bg_color * (1 - alpha/255)
        // R: 255 * (128/255) + 255 * (1 - 128/255) = 255
        // G: 0 * (128/255) + 255 * (1 - 128/255) = 127
        // B: 0 * (128/255) + 255 * (1 - 128/255) = 127
        let blended = force_image_background(&img, [255, 255, 255]);
        assert_eq!(blended.width(), 2);
        assert_eq!(blended.height(), 2);
        let rgb = blended.to_rgb8();
        let pixel = rgb.get_pixel(0, 0);
        assert_eq!(pixel[0], 255);
        // 整数演算の誤差を考慮し、127〜128付近であることを確認
        assert!((pixel[1] as i16 - 127).abs() <= 1);
        assert!((pixel[2] as i16 - 127).abs() <= 1);
    }
}
