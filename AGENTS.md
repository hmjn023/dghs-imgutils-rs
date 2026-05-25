# AGENTS.md

このドキュメントは、AIコーディングエージェントがこのリポジトリを操作する際のガイドラインです。

## プロジェクト概要

**dghs-imgutils-rs** は、Pythonライブラリ [`dghs-imgutils`](https://github.com/deepghs/imgutils) の Rust 実装プロジェクトです。

アニメ・イラスト画像向けの画像処理・機械学習推論機能を、Rust で高性能かつクロスプラットフォームに提供することを目標としています。

最終的には **[napi-rs](https://napi.rs/)** を使って TypeScript / Node.js からもネイティブバインディング経由で呼び出せるようにすることを目指しています。

参照元の Python ライブラリ（`imgutils/` サブディレクトリ）が提供する機能：

- タクエ（差分）検出・クラスタリング（LPIPS）
- キャラクター画像の対照学習特徴抽出（CCIP）
- オブジェクト検出（顔・頭部・人物）
- エッジ検出・線画生成
- モノクロ画像判定
- 破損画像チェック
- 画像タギング（WD14, Deepdanbooru）
- キャラクターセグメンテーション

## リポジトリ構成

```
dghs-imgutils-rs/
├── AGENTS.md           # このファイル
├── Cargo.toml          # Rust パッケージ設定
├── mise.toml           # ツールバージョン管理（Rust, Python, Node, Bun, uv）
├── src/
│   └── main.rs         # Rust エントリポイント（現在は雛形）
└── imgutils/           # 参照元 Python ライブラリ（サブモジュール or コピー）
    ├── imgutils/       # Python パッケージ本体
    ├── test/           # Python テスト
    ├── docs/           # ドキュメント
    └── ...
```

## 開発環境セットアップ

ツールのバージョン管理には [mise](https://mise.jdx.dev/) を使用しています。

```bash
# mise でツールをインストール
mise install

# Rust ビルド確認
cargo build

# Rust テスト実行
cargo test
```

## 技術スタック

| 用途 | ツール |
|------|--------|
| メイン言語 | Rust (latest) |
| TSバインディング | napi-rs |
| Python参照実装 | Python 3.12 + uv |
| ランタイム (JS) | Node.js (latest), Bun (latest) |
| ビルドツール | Cargo |
| ツール管理 | mise |

## 主な依存クレート

| クレート | 用途 |
|----------|------|
| `hf-hub` | HuggingFace Hub からのモデルダウンロード・キャッシュ管理 |
| `thiserror` | エラー型の定義 |
| `serde` / `serde_json` | JSON のシリアライズ／デシリアライズ |

## napi-rs バインディング方針

- Rust のコアロジックは `src/lib.rs` 以下の純粋な Rust クレートとして実装する
- napi-rs 用のエントリポイントは `src/napi/` または専用のバイナリクレートとして分離する
- `#[napi]` マクロを使って TypeScript 向けに公開する関数・型を定義する
- 生成される `.d.ts` の型定義が直感的になるよう、型設計に注意する
- Node.js のネイティブアドオン（`.node` ファイル）は `npm` パッケージとして配布することを想定する

```bash
# napi-rs CLI のインストール（初回のみ）
npm install -g @napi-rs/cli

# バインディングのビルド（開発時）
napi build --platform

# TypeScript 型定義の生成
napi build --platform --dts
```

## コーディング規約

### Rust

- **Edition**: Rust 2024
- フォーマットには `cargo fmt` を使用してください
- リントは `cargo clippy -- -D warnings` で確認してください
- 新しい機能には必ずドキュメントコメント（`///`）を付与してください
- `unsafe` コードは最小限に留め、使用する場合はコメントで理由を明記してください

### ファイル操作

- `src/main.rs` はエントリポイントです。ライブラリコードは `src/lib.rs` や `src/` 配下のモジュールに分離してください
- `imgutils/` ディレクトリは**読み取り専用**の参照として扱ってください。Python コードを直接変更しないでください

## よく使うコマンド

```bash
# ビルド
cargo build

# リリースビルド
cargo build --release

# テスト
cargo test

# フォーマット
cargo fmt

# Lint
cargo clippy

# Python参照実装の確認（imgutils/ 内）
cd imgutils && uv run python -c "import imgutils; print(imgutils.__version__)"
```

## 実装方針

1. **参照実装の尊重**: Python の `imgutils` ライブラリの API 設計・挙動を参考にしながら、Rust らしい API を設計してください
2. **パフォーマンス優先**: Rust の強みである安全性とパフォーマンスを活かした実装を心がけてください
3. **段階的実装**: 全機能を一度に実装しようとせず、コア機能から始めて段階的に拡張してください
4. **テスト駆動**: 各モジュールにユニットテストを必ず追加してください

## 注意事項

- このプロジェクトは現在初期段階（`src/main.rs` が雛形のみ）です
- Python の参照実装（`imgutils/`）は Rust 実装の仕様参照として使用してください
- ONNX Runtime や推論ライブラリの選定は今後の設計上の重要な決定事項です
