//! dghs-imgutils-rs ライブラリのルート。
//!
//! アニメ・イラスト画像向けの画像処理・機械学習推論機能を提供します。
//! Python ライブラリ [`dghs-imgutils`](https://github.com/deepghs/imgutils) の Rust 実装です。

/// HuggingFace Hub からモデルファイルをダウンロードする機能。
pub mod hub;

/// 画像処理とテンソル前処理のユーティリティ。
pub mod image;

/// ONNX Runtime 推論セッション管理。
pub mod inference;

/// 画像タギング（タグ自動予測）機能。
pub mod tagging;

/// メトリック評価（類似度、同一判定、クラスタリング）機能。
pub mod metrics;

/// 物体検出（アニメ顔、頭部、全身、検閲等）機能。
pub mod detect;

/// 領域分割（キャラクター背景透過、単色背景切り抜き）機能。
pub mod segment;

/// エッジ検出（Canny / Lineart / LineartAnime）機能。
pub mod edge;

/// 姿勢推定（DWpose / OpenPose 18 キーポイント検出）機能。
pub mod pose;

/// 画像検証（破損チェック・グレースケール・AI生成・NSFW 等）機能。
pub mod validate;

/// 画像操作（リサイズ・自動トリミング・自動検閲）機能。
pub mod operate;

#[cfg(feature = "napi")]
/// JS/TS ネイティブバインディング (napi-rs) 用の公開インターフェース。
pub mod napi;
