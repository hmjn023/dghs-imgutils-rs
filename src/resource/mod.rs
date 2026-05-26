//! リソース管理（背景画像、データセットファイル等）機能を提供します。

pub mod background;

pub use background::{get_bg_image, list_bg_images, random_bg_image};
