//! アニメ・イラスト画像の自動タグ付け（タガー）機能を提供します。

pub mod pixai;
pub mod deepdanbooru;
pub mod mldanbooru;
pub mod deepgelbooru;
pub mod wd14;
pub mod camie;
pub mod tag_match;
pub mod format;
pub mod order;
pub mod overlap;
pub mod blacklist;
pub mod character;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// タギング操作で発生しうるエラー
#[derive(Debug, thiserror::Error)]
pub enum TaggingError {
    /// 推論処理のエラー
    #[error("Inference error: {0}")]
    Inference(#[from] crate::inference::InferenceError),

    /// HuggingFace Hub からのモデル取得エラー
    #[error("Hub error: {0}")]
    Hub(#[from] crate::hub::HubError),

    /// CSV ファイルの解析エラー
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// JSON (ipsカラム) の解析エラー
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// ONNX Runtime クレートのエラー
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    /// 画像操作処理のエラー
    #[error("Image error: {0}")]
    Image(#[from] crate::image::error::ImageError),

    /// ファイル I/O エラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// 無効な引数やモデル指定
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

/// タガー推論の結果を表す共通構造体。
///
/// Python の順序付き辞書 (dict) の挙動と等価にするため、
/// 各タグリストは確率の降順（同じ確率の場合はタグ名の昇順）でソートされた
/// `(タグ名, 確信度)` のペアのベクタとして表現されます。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagResult {
    /// 一般的な概念・属性タグ (category = 0)
    pub general: Vec<(String, f32)>,

    /// キャラクター名タグ (category = 4 等)
    pub character: Vec<(String, f32)>,

    /// その他の未定義または追加のカテゴリ別タグ
    pub rest: HashMap<String, Vec<(String, f32)>>,

    /// すべてのカテゴリを結合し、確率の降順でソートしたタグリスト
    pub tag: Vec<(String, f32)>,

    /// 検出されたキャラクターからマッピングされた原作 IP (著作物名) のリスト。
    ///
    /// 登場回数の降順（同じ回数の場合は名前の昇順）でソートされています。
    pub ips: Vec<String>,
}
