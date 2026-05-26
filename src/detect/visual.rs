//! 物体検出結果 (BBox, Mask, Label, Score) を画像上に美しく可視化描画するためのユーティリティ。

use crate::detect::base::Detection;
use image::{DynamicImage, Rgba};

/// ラベル文字列から疑似ランダムに一意のカラー（RGB）を生成します。
pub fn label_to_color(label: &str) -> Rgba<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    let hash = hasher.finish();
    let r = (hash & 0xFF) as u8;
    let g = ((hash >> 8) & 0xFF) as u8;
    let b = ((hash >> 16) & 0xFF) as u8;
    // 明るめの色にするために最低輝度を補償
    Rgba([r.max(60), g.max(60), b.max(60), 255])
}

/// Linux システムの標準フォントディレクトリから使用可能な TrueType フォントを検索してロードします。
pub fn load_system_font() -> Option<ab_glyph::FontVec> {
    let font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ];
    for path in &font_paths {
        if std::path::Path::new(path).exists() {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(font) = ab_glyph::FontVec::try_from_vec(data) {
                    return Some(font);
                }
            }
        }
    }
    None
}

/// 検出されたオブジェクト（境界ボックス、半透明マスク、ラベル、信頼度スコア）を画像上に描画して可視化します。
pub fn detection_visualize(
    image: &DynamicImage,
    detections: &[Detection],
    text_padding: u32,
    fontsize: f32,
    max_short_edge_size: Option<u32>,
    mask_alpha: f32,
    no_label: bool,
) -> DynamicImage {
    let original_width = image.width();
    let original_height = image.height();

    // アスペクト比を維持しつつ、最短辺を制限して縮小リサイズ
    let (new_width, new_height) = if let Some(limit) = max_short_edge_size {
        let short_edge = original_width.min(original_height);
        if limit < short_edge {
            let r = limit as f32 / short_edge as f32;
            let nw = ((original_width as f32 * r).ceil()) as u32;
            let nh = ((original_height as f32 * r).ceil()) as u32;
            (nw, nh)
        } else {
            (original_width, original_height)
        }
    } else {
        (original_width, original_height)
    };

    let mut visual_image = image.to_rgba8();
    if (new_width, new_height) != (original_width, original_height) {
        visual_image = image::imageops::resize(
            image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );
    }

    let system_font = if !no_label { load_system_font() } else { None };

    // スコアの昇順にソート（スコアが高いアイテムが前面に描画されるようにする）
    let mut sorted_detections = detections.to_vec();
    sorted_detections.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

    for detect_item in &sorted_detections {
        let (x0_orig, y0_orig, x1_orig, y1_orig) = detect_item.bbox;

        // リサイズ後の座標へスケール変換
        let x0 = ((x0_orig as f32) * (new_width as f32) / (original_width as f32)).round() as u32;
        let y0 = ((y0_orig as f32) * (new_height as f32) / (original_height as f32)).round() as u32;
        let x1 = ((x1_orig as f32) * (new_width as f32) / (original_width as f32)).round() as u32;
        let y1 = ((y1_orig as f32) * (new_height as f32) / (original_height as f32)).round() as u32;

        let color = label_to_color(&detect_item.label);

        // 1. セグメンテーションマスクの描画 (存在する場合)
        if let Some(ref mask) = detect_item.mask {
            let mask_h = mask.shape()[0] as u32;
            let mask_w = mask.shape()[1] as u32;
            let mut mask_u8 = image::ImageBuffer::new(mask_w, mask_h);
            for y in 0..mask_h {
                for x in 0..mask_w {
                    let val = mask[[y as usize, x as usize]];
                    let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
                    mask_u8.put_pixel(x, y, image::Luma([pixel_val]));
                }
            }
            let mask_img = image::DynamicImage::ImageLuma8(mask_u8);
            let resized_mask = mask_img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);
            let resized_mask_luma = resized_mask.to_luma8();

                // 画素ループでアルファ重ね合わせ
                for y in 0..new_height {
                    for x in 0..new_width {
                        let m_val = resized_mask_luma.get_pixel(x, y)[0] as f32 / 255.0;
                        if m_val > 0.0 {
                            let pixel = visual_image.get_pixel_mut(x, y);
                            let alpha = m_val * mask_alpha;
                            pixel[0] = ((color[0] as f32 * alpha) + (pixel[0] as f32 * (1.0 - alpha))) as u8;
                            pixel[1] = ((color[1] as f32 * alpha) + (pixel[1] as f32 * (1.0 - alpha))) as u8;
                            pixel[2] = ((color[2] as f32 * alpha) + (pixel[2] as f32 * (1.0 - alpha))) as u8;
                        }
                    }
                }
        }

        // 2. 境界ボックス (BBox) 枠線の描画
        let rect = imageproc::rect::Rect::at(x0 as i32, y0 as i32).of_size(
            (x1.saturating_sub(x0)).max(1),
            (y1.saturating_sub(y0)).max(1),
        );
        // 太さ 2px 相当を表現するため少し重ねて描画
        imageproc::drawing::draw_hollow_rect_mut(&mut visual_image, rect, color);
        let inner_rect = imageproc::rect::Rect::at((x0 + 1) as i32, (y0 + 1) as i32).of_size(
            (x1.saturating_sub(x0).saturating_sub(2)).max(1),
            (y1.saturating_sub(y0).saturating_sub(2)).max(1),
        );
        imageproc::drawing::draw_hollow_rect_mut(&mut visual_image, inner_rect, color);

        // 3. テキストラベルと信頼度スコアの描画
        if !no_label {
            if let Some(ref font) = system_font {
                let label_text = format!("{}: {:.2}%", detect_item.label, detect_item.score * 100.0);
                let scale = ab_glyph::PxScale::from(fontsize);

                // テキストの幅・高さを大雑把に計算 (ポータブルな文字幅近似)
                let text_w = (label_text.len() as f32 * fontsize * 0.55) as u32;
                let text_h = fontsize as u32;

                let box_y = if y0 >= text_h + text_padding * 2 {
                    y0 - text_h - text_padding * 2
                } else {
                    y0
                };

                // 半透明背景ボックスの塗りつぶし描画
                let alpha = 0.5f32;
                for y_offset in 0..(text_h + text_padding * 2) {
                    for x_offset in 0..(text_w + text_padding * 2) {
                        let px = x0 + x_offset;
                        let py = box_y + y_offset;
                        if px < new_width && py < new_height {
                            let pixel = visual_image.get_pixel_mut(px, py);
                            pixel[0] = ((color[0] as f32 * alpha) + (pixel[0] as f32 * (1.0 - alpha))) as u8;
                            pixel[1] = ((color[1] as f32 * alpha) + (pixel[1] as f32 * (1.0 - alpha))) as u8;
                            pixel[2] = ((color[2] as f32 * alpha) + (pixel[2] as f32 * (1.0 - alpha))) as u8;
                        }
                    }
                }

                // テキスト自体の描画 (文字色は黒)
                imageproc::drawing::draw_text_mut(
                    &mut visual_image,
                    image::Rgba([0, 0, 0, 255]),
                    (x0 + text_padding) as i32,
                    (box_y + text_padding) as i32,
                    scale,
                    font,
                    &label_text,
                );
            }
        }
    }

    DynamicImage::ImageRgba8(visual_image)
}
