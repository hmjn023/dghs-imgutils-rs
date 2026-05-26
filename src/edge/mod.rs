//! エッジ検出モジュール。
//!
//! Canny アルゴリズム（モデル不要）および Lineart / LineartAnime ニューラルネットワーク
//! による線画生成機能を提供します。

pub mod canny;
pub mod lineart;
pub mod lineart_anime;

pub use canny::{edge_image_with_canny, get_edge_by_canny};
pub use lineart::{edge_image_with_lineart, get_edge_by_lineart};
pub use lineart_anime::{edge_image_with_lineart_anime, get_edge_by_lineart_anime};
