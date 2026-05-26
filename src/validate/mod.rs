//! 画像検証モジュール。
//!
//! - 破損ファイルチェック (`truncate`)
//! - グレースケール判定 (`color`)
//! - AI 生成・モノクロ・年齢層・NSFWなど各種分類 (`classify`)
//! - NSFW 特殊モデル (`nsfw`)

pub mod classify;
pub mod color;
pub mod nsfw;
pub mod truncate;

pub use classify::{
    anime_classify, anime_classify_score, anime_completeness, anime_completeness_score,
    anime_furry, anime_furry_score, anime_portrait, anime_portrait_score, anime_rating,
    anime_rating_score, anime_real, anime_real_score, anime_style_age, anime_style_age_score,
    get_ai_created_score, get_monochrome_score, is_ai_created, is_monochrome,
};
pub use color::is_greyscale;
pub use nsfw::{nsfw_pred, nsfw_pred_score};
pub use truncate::is_truncated_file;
