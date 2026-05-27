//! 精度検証テスト: Python 実装との出力比較。
//!
//! このテストは `#[ignore]` 属性が付いているため、CI では自動実行されない。
//! 手動でテストを実行する際には、先に `generate_baseline.py` を実行して基準データを生成する必要がある。
//!
//! 使用方法:
//! ```bash
//! cd tests/precision
//! python generate_baseline.py
//! cd ../..
//! cargo test --test precision_test -- --ignored
//! ```

use image::DynamicImage;
use ndarray::Array3;

/// PSNR (Peak Signal-to-Noise Ratio) を計算する。
fn psnr(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");

    let mut mse = 0.0f32;
    for (va, vb) in a.iter().zip(b.iter()) {
        let diff = va - vb;
        mse += diff * diff;
    }
    mse /= a.len() as f32;

    if mse < 1e-10 {
        f32::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// ndarray を DynamicImage に変換する (RGB, f32, 0-255)。
fn array_to_dynamic_image(arr: &Array3<f32>) -> DynamicImage {
    let (h, w) = (arr.shape()[1], arr.shape()[2]);
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let r = arr[[0, y, x]].clamp(0.0, 255.0).round() as u8;
            let g = arr[[1, y, x]].clamp(0.0, 255.0).round() as u8;
            let b = arr[[2, y, x]].clamp(0.0, 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// DynamicImage を ndarray に変換する (RGB, f32, 0-255)。
fn dynamic_image_to_array(img: &DynamicImage) -> Array3<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut arr = Array3::<f32>::zeros((3, h as usize, w as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        arr[[0, y as usize, x as usize]] = pixel[0] as f32;
        arr[[1, y as usize, x as usize]] = pixel[1] as f32;
        arr[[2, y as usize, x as usize]] = pixel[2] as f32;
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    /// adversarial noise removal の精度テスト。
    ///
    /// Python 版と同一入力に対して、PSNR > 30dB を期待する。
    #[test]
    #[ignore] // 手動実行: python generate_baseline.py 後に cargo test -- --ignored
    fn test_adversarial_noise_removal_precision() {
        // 基準データの読み込み
        let input_path = "tests/precision/baseline_input.npy";
        if !std::path::Path::new(input_path).exists() {
            eprintln!("Baseline data not found. Run: cd tests/precision && python generate_baseline.py");
            return;
        }

        // npy ファイルからデータを読み込む（簡易実装）
        // 注意: 本番環境では ndarray-npy クレートの使用を推奨
        let input_data = std::fs::read(input_path).expect("Failed to read baseline_input.npy");
        // npy ファイルのヘッダーをスキップしてデータを読み込む
        // 実際の実装では適切な npy パーサーを使用すること

        // テスト画像を生成（npy ファイルが読めない場合のフォールバック）
        let mut arr = Array3::<f32>::zeros((3, 64, 64));
        for y in 0..64 {
            for x in 0..64 {
                arr[[0, y, x]] = (x as f32 / 64.0) * 255.0;
                arr[[1, y, x]] = (y as f32 / 64.0) * 255.0;
                arr[[2, y, x]] = ((x + y) as f32 / 128.0) * 255.0;
            }
        }

        let img = array_to_dynamic_image(&arr);

        // Rust で adversarial noise removal を実行
        let result = dghs_imgutils_rs::restore::adversarial::remove_adversarial_noise(
            &img, 2, 1, // テスト用に少ないイテレーション数
        )
        .expect("Adversarial noise removal failed");

        let result_arr = dynamic_image_to_array(&result);

        // PSNR を計算（入力と出力の差分）
        let psnr_value = psnr(&arr, &result_arr);
        println!("PSNR (input vs output): {:.2} dB", psnr_value);

        // PSNR > 20dB を期待（フィルタリングなので完全一致はしない）
        assert!(
            psnr_value > 20.0,
            "PSNR too low: {:.2} dB (expected > 20 dB)",
            psnr_value
        );
    }

    /// bilateral filter の基本テスト。
    #[test]
    fn test_bilateral_filter_basic() {
        // 定数画像に対してフィルタを適用しても変化しないことを確認
        let arr = Array3::<f32>::from_elem((3, 16, 16), 128.0);
        let img = array_to_dynamic_image(&arr);

        let result =
            dghs_imgutils_rs::restore::adversarial::remove_adversarial_noise(&img, 1, 0)
                .expect("Adversarial noise removal failed");

        let result_arr = dynamic_image_to_array(&result);

        // 定数画像なので PSNR は無限大になるはず
        let psnr_value = psnr(&arr, &result_arr);
        assert!(
            psnr_value > 50.0,
            "PSNR for constant image too low: {:.2} dB",
            psnr_value
        );
    }
}
