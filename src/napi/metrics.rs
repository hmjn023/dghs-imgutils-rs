use crate::metrics::laplacian::laplacian_score as core_laplacian_score;
use crate::metrics::psnr_::psnr as core_psnr;
use napi_derive::napi;

#[napi]
pub fn laplacian_score(path: String) -> napi::Result<f64> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path, e),
        )
    })?;
    let score = core_laplacian_score(&image);
    Ok(score as f64)
}

#[napi]
pub fn psnr(path1: String, path2: String) -> napi::Result<f64> {
    let img1 = image::open(&path1).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path1, e),
        )
    })?;
    let img2 = image::open(&path2).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image {}: {}", path2, e),
        )
    })?;
    let result = core_psnr(&img1, &img2);
    Ok(result)
}
