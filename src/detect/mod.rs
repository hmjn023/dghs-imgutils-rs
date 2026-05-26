//! 物体検出 (detect) モジュール

pub mod base;
pub mod booru;
pub mod censor;
pub mod eye;
pub mod face;
pub mod halfbody;
pub mod hand;
pub mod head;
pub mod nudenet;
pub mod person;
pub mod similarity;
pub mod text;
pub mod visual;

pub use base::{BBox, Detection};
pub use booru::detect_with_booru_yolo;
pub use censor::detect_censors;
pub use eye::detect_eyes;
pub use face::detect_faces;
pub use halfbody::detect_halfbody;
pub use hand::detect_hands;
pub use head::detect_heads;
pub use nudenet::detect_with_nudenet;
pub use person::detect_person;
pub use similarity::{
    bboxes_similarity, calculate_iou, calculate_mask_iou, detection_similarity,
    detection_with_mask_similarity, masks_similarity,
};
pub use text::detect_text;
pub use visual::detection_visualize;
