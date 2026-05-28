use napi_derive::napi;

/// NafNet モデルで画像を復元します。
///
/// `model` には `"REDS"`, `"GoPro"`, `"SIDD"` のいずれかを指定します。
#[napi]
pub fn restore_with_nafnet(path: String, model: Option<String>) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let model = model.as_deref().unwrap_or("REDS");
    let result = crate::restore::nafnet::restore_with_nafnet(&image, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("NafNet restoration failed: {}", e),
        )
    })?;
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode result image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}

/// SCUNet モデルで画像を復元します。
///
/// `model` には `"GAN"` または `"PSNR"` を指定します。
#[napi]
pub fn restore_with_scunet(path: String, model: Option<String>) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let model = model.as_deref().unwrap_or("GAN");
    let result = crate::restore::scunet::restore_with_scunet(&image, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("SCUNet restoration failed: {}", e),
        )
    })?;
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode result image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}

/// Adversarial noise をフィルタリングで除去します（ONNX モデル不要）。
#[napi]
pub fn remove_adversarial_noise(
    path: String,
    b_iters: Option<i32>,
    g_iters: Option<i32>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let b_iters = b_iters.unwrap_or(64).max(0) as usize;
    let g_iters = g_iters.unwrap_or(8).max(0) as usize;
    let result = crate::restore::adversarial::remove_adversarial_noise(&image, b_iters, g_iters)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Adversarial noise removal failed: {}", e),
            )
        })?;
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode result image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}

/// CDC モデルで画像を 4x アップスケールします。
#[napi]
pub fn upscale_with_cdc(
    path: String,
    model: Option<String>,
    tile_size: Option<u32>,
    tile_overlap: Option<u32>,
    batch_size: Option<u32>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;
    let model = model.unwrap_or_else(|| crate::upscale::cdc::DEFAULT_CDC_MODEL.to_string());
    let tile_size = tile_size.unwrap_or(512);
    let tile_overlap = tile_overlap.unwrap_or(64);
    let batch_size = batch_size.unwrap_or(1);
    let result =
        crate::upscale::cdc::upscale_with_cdc(&image, &model, tile_size, tile_overlap, batch_size)
            .map_err(|e| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    format!("CDC upscale failed: {}", e),
                )
            })?;
    let mut buf = std::io::Cursor::new(Vec::new());
    result
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode result image: {}", e),
            )
        })?;
    Ok(buf.into_inner())
}
