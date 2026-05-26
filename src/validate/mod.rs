//! 画像検証モジュール。
//!
//! - 破損ファイルチェック (`truncate`)
//! - グレースケール判定 (`color`)
//! - AI 生成・モノクロ・年齢層・NSFWなど各種分類 (`classify`)
//! - NSFW 特殊モデル (`nsfw`)
//! - セーフ判定 (`safe`)
//! - Danbooru レーティング (`dbrating`)
//! - バングミキャラクター種類 (`bangumi_char`)
//! - ティーンレーティング (`teen`)

pub mod bangumi_char;
pub mod classify;
pub mod color;
pub mod dbrating;
pub mod nsfw;
pub mod safe;
pub mod teen;
pub mod truncate;

pub use bangumi_char::{anime_bangumi_char, anime_bangumi_char_score};
pub use classify::{
    anime_classify, anime_classify_score, anime_completeness, anime_completeness_score,
    anime_furry, anime_furry_score, anime_portrait, anime_portrait_score, anime_rating,
    anime_rating_score, anime_real, anime_real_score, anime_style_age, anime_style_age_score,
    get_ai_created_score, get_monochrome_score, is_ai_created, is_monochrome,
};
pub use color::is_greyscale;
pub use dbrating::{anime_dbrating, anime_dbrating_score};
pub use nsfw::{nsfw_pred, nsfw_pred_score};
pub use safe::{safe_check, safe_check_score};
pub use teen::{anime_teen, anime_teen_score};
pub use truncate::is_truncated_file;
