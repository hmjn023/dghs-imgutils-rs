//! 画像操作モジュール。
//!
//! - 最大サイズ合わせ (`align`)
//! - 透過PNG の自動トリミング (`squeeze`)
//! - 検出部位の自動検閲 (`censor`)

pub mod align;
pub mod squeeze;
pub mod censor;

pub use align::align_maxsize;
pub use squeeze::{squeeze, squeeze_with_transparency};
pub use censor::{CensorMethod, censor_area, censor_areas, censor_nsfw};
