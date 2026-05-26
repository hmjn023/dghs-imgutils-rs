//! Stable Diffusion 関連のメタデータ読み書き機能を提供します。
//!
//! # サブモジュール
//!
//! - `metadata` — A1111 形式の SD メタデータのパース (`SDMetaData`)
//! - `nai` — NovelAI 形式のメタデータのパース (`NAIMetaData`)
//! - `safetensors` — Safetensors モデルファイルのヘッダーメタデータの読み取り

pub mod metadata;
pub mod nai;
pub mod safetensors;

pub use metadata::{SDMetaData, parse_sdmeta_from_text};
pub use nai::{NAIMetaData, get_naimeta_from_image};
pub use safetensors::read_safetensors_metadata;
