//! 画像の最大サイズ制限リサイズ。
//!
//! Python 版: `align_maxsize(image, max_size)` → 長辺を `max_size` に合わせてアスペクト比維持リサイズ。

use image::{DynamicImage, imageops::FilterType};

/// 画像の長辺が `max_size` を超える場合、アスペクト比を維持しながらリサイズします。
///
/// # 引数
/// * `img` - 入力画像
/// * `max_size` - 最大辺長（px）
pub fn align_maxsize(img: &DynamicImage, max_size: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let longer_side = w.max(h);
    if longer_side <= max_size {
        return img.clone();
    }
    let r = max_size as f32 / longer_side as f32;
    let new_w = (w as f32 * r) as u32;
    let new_h = (h as f32 * r) as u32;
    img.resize_exact(new_w, new_h, FilterType::Triangle)
}
