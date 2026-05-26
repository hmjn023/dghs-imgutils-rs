//! 画像検証モジュール。
//!
//! - 破損ファイルチェック (`truncate`)
//! - グレースケール判定 (`color`)
//! - AI 生成・モノクロ・年齢層・NSFWなど各種分類 (`classify`)
//! - NSFW 特殊モデル (`nsfw`)

pub mod truncate;
pub mod color;
pub mod classify;
pub mod nsfw;

pub use truncate::is_truncated_file;
pub use color::is_greyscale;
pub use classify::{
    get_ai_created_score, is_ai_created,
    get_monochrome_score, is_monochrome,
    anime_classify_score, anime_classify,
    anime_real_score, anime_real,
    anime_rating_score, anime_rating,
    anime_completeness_score, anime_completeness,
    anime_portrait_score, anime_portrait,
    anime_furry_score, anime_furry,
    anime_style_age_score, anime_style_age,
};
pub use nsfw::{nsfw_pred_score, nsfw_pred};
