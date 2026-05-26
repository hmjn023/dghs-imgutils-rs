//! 画像操作モジュール。
//!
//! - 最大サイズ合わせ (`align`)
//! - 透過PNG の自動トリミング (`squeeze`)
//! - 検出部位の自動検閲 (`censor`)
//! - 画像ベース検閲 (`imgcensor`)

pub mod align;
pub mod censor;
pub mod imgcensor;
pub mod squeeze;

pub use align::align_maxsize;
pub use censor::{
    CensorError, CensorMethod, censor_area, censor_areas, censor_nsfw, censor_nsfw_image,
};
pub use imgcensor::{CensorImageFit, censor_area_image, censor_areas_image};
pub use squeeze::{squeeze, squeeze_with_transparency};
