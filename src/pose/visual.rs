use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};

use super::format::OP18KeyPointSet;

const BODY_CONNECTS: &[(usize, usize)] = &[
    (2, 3),
    (2, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (2, 12),
    (12, 13),
    (13, 14),
    (2, 1),
    (1, 15),
    (15, 17),
    (1, 16),
    (16, 18),
];

const BODY_COLORS: &[Rgba<u8>] = &[
    Rgba([255, 0, 0, 255]),
    Rgba([255, 85, 0, 255]),
    Rgba([255, 170, 0, 255]),
    Rgba([255, 255, 0, 255]),
    Rgba([170, 255, 0, 255]),
    Rgba([85, 255, 0, 255]),
    Rgba([0, 255, 0, 255]),
    Rgba([0, 255, 85, 255]),
    Rgba([0, 255, 170, 255]),
    Rgba([0, 255, 255, 255]),
    Rgba([0, 170, 255, 255]),
    Rgba([0, 85, 255, 255]),
    Rgba([0, 0, 255, 255]),
    Rgba([85, 0, 255, 255]),
    Rgba([170, 0, 255, 255]),
    Rgba([255, 0, 255, 255]),
    Rgba([255, 0, 170, 255]),
    Rgba([255, 0, 85, 255]),
];

const HAND_EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

const FOOT_EDGES: &[(usize, usize)] = &[(2, 0), (2, 1)];

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgba<u8> {
    let hi = (h * 6.0).floor() as i32 % 6;
    let f = h * 6.0 - hi as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match hi {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
}

fn draw_body(img: &mut image::RgbaImage, keypoints: &OP18KeyPointSet, threshold: f32) {
    for (idx, &(i, j)) in BODY_CONNECTS.iter().enumerate() {
        let kp_i = &keypoints.keypoints[i - 1];
        let kp_j = &keypoints.keypoints[j - 1];
        if kp_i.score >= threshold && kp_j.score >= threshold {
            let color = BODY_COLORS[idx.min(BODY_COLORS.len() - 1)];
            draw_line_segment_mut(img, (kp_i.x, kp_i.y), (kp_j.x, kp_j.y), color);
        }
    }
}

fn draw_hands(img: &mut image::RgbaImage, keypoints: &OP18KeyPointSet, threshold: f32) {
    let n_edges = HAND_EDGES.len();
    for hands in [keypoints.left_hand(), keypoints.right_hand()] {
        for (ie, &(e0, e1)) in HAND_EDGES.iter().enumerate() {
            let kp0 = &hands[e0];
            let kp1 = &hands[e1];
            if kp0.score >= threshold && kp1.score >= threshold {
                let color = hsv_to_rgb(ie as f32 / n_edges as f32, 1.0, 1.0);
                draw_line_segment_mut(img, (kp0.x, kp0.y), (kp1.x, kp1.y), color);
            }
        }
    }
}

fn draw_feet(img: &mut image::RgbaImage, keypoints: &OP18KeyPointSet, threshold: f32) {
    let n_edges = FOOT_EDGES.len();
    for feet in [keypoints.left_foot(), keypoints.right_foot()] {
        for (ie, &(e0, e1)) in FOOT_EDGES.iter().enumerate() {
            let kp0 = &feet[e0];
            let kp1 = &feet[e1];
            if kp0.score >= threshold && kp1.score >= threshold {
                let color = hsv_to_rgb(ie as f32 / n_edges as f32, 1.0, 1.0);
                draw_line_segment_mut(img, (kp0.x, kp0.y), (kp1.x, kp1.y), color);
            }
        }
    }
}

fn draw_face(img: &mut image::RgbaImage, keypoints: &OP18KeyPointSet, threshold: f32) {
    let point_size = 4.5f32;
    let color = Rgba([0, 255, 0, 255]);
    for kp in keypoints.face() {
        if kp.score >= threshold {
            draw_filled_circle_mut(
                img,
                (kp.x as i32, kp.y as i32),
                (point_size * 0.5).round() as i32,
                color,
            );
        }
    }
}

/// Visualize OP18 keypoints on an image.
///
/// Draws skeleton bones and keypoints colored by body part.
///
/// # Arguments
/// * `image` - Input image
/// * `keypoints_list` - List of OP18KeyPointSet to visualize
/// * `threshold` - Keypoint confidence threshold (default 0.3)
/// * `min_edge_size` - If the shorter edge exceeds this, the image is downscaled first
/// * `draw_body` - Whether to draw body skeleton
/// * `draw_hands` - Whether to draw hand skeleton
/// * `draw_feet` - Whether to draw foot skeleton
/// * `draw_face` - Whether to draw face keypoints
///
/// # Returns
/// RGBA image with visualized keypoints
pub fn op18_visualize(
    image: &DynamicImage,
    keypoints_list: &[OP18KeyPointSet],
    threshold: f32,
    min_edge_size: Option<u32>,
    show_body: bool,
    show_hands: bool,
    show_feet: bool,
    show_face: bool,
) -> image::RgbaImage {
    let (w, h) = image.dimensions();
    let scale_factor = if let Some(limit) = min_edge_size {
        let min_edge = w.min(h);
        if min_edge > limit {
            Some(min_edge as f32 / limit as f32)
        } else {
            None
        }
    } else {
        None
    };

    let canvas = if let Some(r) = scale_factor {
        let new_w = (w as f32 / r).round() as u32;
        let new_h = (h as f32 / r).round() as u32;
        image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
    } else {
        image.clone()
    };

    let mut img = canvas.to_rgba8();

    for keypoints in keypoints_list {
        let mut kps_scaled = keypoints.clone();
        if let Some(r) = scale_factor {
            kps_scaled.scale(1.0 / r);
        }

        if show_body {
            draw_body(&mut img, &kps_scaled, threshold);
        }
        if show_hands {
            draw_hands(&mut img, &kps_scaled, threshold);
        }
        if show_feet {
            draw_feet(&mut img, &kps_scaled, threshold);
        }
        if show_face {
            draw_face(&mut img, &kps_scaled, threshold);
        }
    }

    img
}
