//! 領域分割 (segment) モジュール

pub mod isnetis;

pub use isnetis::{get_isnetis_mask, segment_rgba_with_isnetis, segment_with_isnetis};
