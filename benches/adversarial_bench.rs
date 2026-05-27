use std::time::Instant;

use image::DynamicImage;
use dghs_imgutils_rs::restore::adversarial::{remove_adversarial_noise, remove_adversarial_noise_with_params, AdversarialNoiseParams};

fn main() {
    println!("=== Adversarial Noise Removal Benchmark ===\n");

    // テスト画像の生成
    let sizes = vec![(64, 64), (128, 128), (256, 256)];

    for (w, h) in sizes {
        let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
            w, h, image::Rgb([128u8, 128, 128]),
        ));

        // ウォームアップ
        let _ = remove_adversarial_noise(&img, 2, 1);

        // デフォルトパラメータでのベンチマーク
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = remove_adversarial_noise(&img, 4, 1);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() / n_runs as f64 * 1000.0;
        println!("{w}x{h}: default params (b_iters=4, g_iters=1): avg {avg_ms:.2}ms");

        // カスタムパラメータでのベンチマーク
        let params = AdversarialNoiseParams {
            b_iters: 4,
            g_iters: 1,
            diameter_min: 4,
            diameter_max: 6,
            sigma_color_min: 6.0,
            sigma_color_max: 10.0,
            sigma_space_min: 6.0,
            sigma_space_max: 10.0,
            radius_min: 3,
            radius_max: 6,
            eps_min: 16.0,
            eps_max: 24.0,
        };
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = remove_adversarial_noise_with_params(&img, &params);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() / n_runs as f64 * 1000.0;
        println!("{w}x{h}: custom params: avg {avg_ms:.2}ms");
        println!();
    }

    // セッションキャッシュの効果測定（adversarial は ONNX を使わないので効果なし）
    println!("Note: adversarial noise removal does not use ONNX sessions.");
    println!("Session caching benefits NafNet, SCUNet, and CDC models.");
}
