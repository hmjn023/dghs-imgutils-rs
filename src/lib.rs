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
