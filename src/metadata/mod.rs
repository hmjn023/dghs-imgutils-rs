//! 画像メタデータの読み書き機能を提供します。
//!
//! # サブモジュール
//!
//! - `geninfo` — PNG iTXt/tEXt チャンクおよび EXIF/GIF からの生成情報 (geninfo) の読み取り
//! - `lsb` — LSB ステガノグラフィによるメタデータの埋め込み・抽出

pub mod geninfo;
pub mod lsb;

pub use geninfo::{read_geninfo_exif, read_geninfo_gif, read_geninfo_parameters, read_geninfo_raw};
pub use lsb::{read_lsb_metadata, read_lsb_raw_bytes, write_lsb_metadata, write_lsb_raw_bytes};
