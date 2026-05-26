//! 透過 PNG の自動トリミング。
//!
//! Python 版の `squeeze` / `squeeze_with_transparency` を移植します。
//!
//! ## squeeze アルゴリズム
//! 1. マスク（bool[H, W]）の非ゼロ行・列範囲を求める
//! 2. その範囲でクロップ
//!
//! ## squeeze_with_transparency アルゴリズム
//! 1. アルファチャンネルを `[0.0, 1.0]` に変換
//! 2. `threshold` 以上のピクセルを true とした bool マスクを作成
//! 3. 2D メディアンフィルタでノイズ除去（`median_filter` サイズ）
//! 4. `squeeze` に渡す

use image::{DynamicImage, GenericImageView};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SqueezeError {
    #[error("Failed to open image: {0}")]
    ImageOpen(#[from] image::ImageError),
    #[error("No foreground pixels found in mask")]
    EmptyMask,
}

/// bool マスクの非ゼロ領域でクロップします。
///
/// # 引数
/// * `img` - 入力画像（任意フォーマット、RGBA 保持）
/// * `mask` - bool マスク（true = 前景）。`img` と同じ H x W でなければなりません
///
/// # エラー
/// マスクが全 false（前景なし）の場合は `SqueezeError::EmptyMask`
pub fn squeeze(img: &DynamicImage, mask: &Vec<Vec<bool>>) -> Result<DynamicImage, SqueezeError> {
    let (w, h) = img.dimensions();
    let rows = mask.len() as u32;
    let cols = if rows > 0 { mask[0].len() as u32 } else { 0 };

    // 行方向（x_min, x_max）: マスクの列に非ゼロがある行インデックス
    // ここで Python の crop((y_min, x_min, y_max, x_max)) に注意:
    // Python の Pillow crop は (left, upper, right, lower) = (col_min, row_min, col_max, row_max)
    let mut row_min = rows;
    let mut row_max = 0u32;
    let mut col_min = cols;
    let mut col_max = 0u32;

    for r in 0..rows.min(h) {
        for c in 0..cols.min(w) {
            if mask[r as usize][c as usize] {
                if r < row_min {
                    row_min = r;
                }
                if r > row_max {
                    row_max = r;
                }
                if c < col_min {
                    col_min = c;
                }
                if c > col_max {
                    col_max = c;
                }
            }
        }
    }

    if row_min > row_max || col_min > col_max {
        return Err(SqueezeError::EmptyMask);
    }

    // Pillow crop: (left, upper, right, lower) = (col_min, row_min, col_max+1, row_max+1)
    let cropped = img.crop_imm(
        col_min,
        row_min,
        col_max - col_min + 1,
        row_max - row_min + 1,
    );
    Ok(cropped)
}

/// 透過 PNG の前景領域だけをクロップして返します。
///
/// # 引数
/// * `path` - 入力画像のパス（RGBA 画像であること）
/// * `threshold` - アルファ値の閾値（0.0〜1.0、デフォルト 0.7）
/// * `median_size` - メディアンフィルタのウィンドウサイズ（奇数、デフォルト 5）
pub fn squeeze_with_transparency(
    path: &str,
    threshold: f32,
    median_size: usize,
) -> Result<DynamicImage, SqueezeError> {
    let img = image::open(path).map_err(SqueezeError::ImageOpen)?;
    let (w, h) = img.dimensions();

    // アルファチャンネルからマスクを作成
    let rgba = img.to_rgba8();
    let mut mask: Vec<Vec<bool>> = (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    let alpha = rgba.get_pixel(x, y)[3] as f32 / 255.0;
                    alpha >= threshold
                })
                .collect()
        })
        .collect();

    // 2D メディアンフィルタ（scipy.ndimage.median_filter の等価実装）
    if median_size > 1 {
        mask = apply_median_filter(&mask, median_size);
    }

    squeeze(&img, &mask)
}

/// 2D bool マスクにメディアンフィルタを適用します。
/// ウィンドウ内の true の個数が過半数なら true、そうでなければ false。
fn apply_median_filter(mask: &Vec<Vec<bool>>, size: usize) -> Vec<Vec<bool>> {
    let h = mask.len();
    if h == 0 {
        return mask.clone();
    }
    let w = mask[0].len();
    let half = size / 2;
    let threshold = (size * size) / 2; // 過半数

    let mut result = vec![vec![false; w]; h];

    for r in 0..h {
        for c in 0..w {
            let mut count = 0usize;
            for dr in 0..size {
                let rr = (r + dr).saturating_sub(half);
                if rr >= h {
                    continue;
                }
                for dc in 0..size {
                    let cc = (c + dc).saturating_sub(half);
                    if cc >= w {
                        continue;
                    }
                    if mask[rr][cc] {
                        count += 1;
                    }
                }
            }
            result[r][c] = count > threshold;
        }
    }

    result
}
