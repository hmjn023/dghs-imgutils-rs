//! グレースケール画像判定。
//!
//! Python 版の `is_greyscale` は PSNR（ピーク信号対雑音比）を使用します。
//!
//! アルゴリズム:
//! 1. 入力画像をグレースケール変換した画像 `L` を作成
//! 2. 元画像と `L` の RGB 間 PSNR を計算
//! 3. PSNR ≈ 30.37 dB 以上 → グレースケール画像と判定
//!
//! PSNR 閾値の導出:
//! `_GREYSCALE_K = -0.3578`, `_GREYSCALE_B = 10.867`
//! `threshold = -B / K ≈ 30.37`

use crate::image::force_image_background;

/// PSNR しきい値（Python 版と同一値）
const GREYSCALE_PSNR_THRESHOLD: f64 = 30.37;

/// 画像が実質的にグレースケール（白黒）かどうかを判定します。
///
/// # 引数
/// * `path` - 検査する画像のパス
///
/// # 戻り値
/// * `true` - グレースケール画像
/// * `false` - カラー画像
pub fn is_greyscale(path: &str) -> bool {
    match image::open(path) {
        Ok(img) => {
            let rgb = force_image_background(&img, [255, 255, 255]);
            let psnr_val = psnr_with_grey(&rgb);
            psnr_val >= GREYSCALE_PSNR_THRESHOLD
        }
        Err(_) => false,
    }
}

/// 画像の PSNR をグレースケール変換後の画像と比較した値を返します。
///
/// PSNR = 10 * log10(255^2 / MSE)
/// MSE が 0 の場合（完全一致 = 完全グレー）は f64::INFINITY を返します。
pub fn psnr_with_grey(img: &image::DynamicImage) -> f64 {
    let rgb8 = img.to_rgb8();
    let (w, h) = rgb8.dimensions();
    let n = (w * h * 3) as f64;

    let mut sum_sq_diff: f64 = 0.0;

    for (_, _, pixel) in rgb8.enumerate_pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        // Pillow の L (Greyscale) 変換: L = 0.299 * R + 0.587 * G + 0.114 * B
        let grey = 0.299 * r + 0.587 * g + 0.114 * b;

        sum_sq_diff += (r - grey).powi(2);
        sum_sq_diff += (g - grey).powi(2);
        sum_sq_diff += (b - grey).powi(2);
    }

    if sum_sq_diff == 0.0 {
        return f64::INFINITY;
    }

    let mse = sum_sq_diff / n;
    10.0 * (255.0_f64.powi(2) / mse).log10()
}
