# AGENTS.md

このドキュメントは、AIコーディングエージェントがこのリポジトリを操作する際のガイドラインです。

## プロジェクト概要

**dghs-imgutils-rs** は、Pythonライブラリ [`dghs-imgutils`](https://github.com/deepghs/imgutils) の Rust 実装プロジェクトです。

アニメ・イラスト画像向けの画像処理・機械学習推論機能を、Rust で高性能かつクロスプラットフォームに提供することを目標としています。

**[napi-rs](https://napi.rs/)** を使って TypeScript / Node.js からもネイティブバインディング経由で呼び出し可能です。

## 実装済みモジュール

| モジュール | 説明 | 状態 |
|-----------|------|------|
| `tagging` | 自動タグ付け（WD14, DeepDanbooru, PixAI, Camie など） | 実装済み |
| `detect` | 物体検出（顔、頭部、人物、目、手、上半身、検閲、テキスト） | 実装済み |
| `segment` | キャラクターセグメンテーション・背景透過（ISNetIS） | 実装済み |
| `edge` | エッジ検出・線画生成（Canny, Lineart, LineartAnime） | 実装済み |
| `metrics` | キャラクター類似度（CCIP）、美的スコアリング、LPIPS | 実装済み |
| `validate` | 画像検証（破損チェック、モノクロ判定、AI生成判定、NSFW） | 実装済み |
| `ocr` | テキスト検出・認識（PaddleOCR） | 実装済み |
| `pose` | 姿勢推定（DWpose / OpenPose 18キーポイント） | 実装済み |
| `metadata` | 画像メタデータ読み書き、LSBステガノグラフィ | 実装済み |
| `sd` | Stable Diffusion / NovelAI メタデータパーサー | 実装済み |
| `operate` | 画像リサイズ、自動トリミング、自動検閲 | 実装済み |
| `restore` | 画像復元（NafNet, SCUNet, 敵対的ノイズ除去） | 実装済み |
| `upscale` | 超解像（CDCアップスケーリング） | 実装済み |
| `ascii` | アスキーアート生成 | 実装済み |
| `resource` | 背景画像データセット管理 | 実装済み |
| `generic` | 汎用MLエンジン（YOLO, CLIP, SigLIP, TIMM分類） | 実装済み |
| `hub` | HuggingFace Hub モデルダウンロード・キャッシュ | 実装済み |
| `image` | 画像I/O、フォーマット変換、テンソル変換 | 実装済み |
| `inference` | ONNX Runtime セッション管理 | 実装済み |
| `utils` | 共通ユーティリティ（キャッシュ、ストレージ） | 実装済み |
| `config` | パッケージメタ情報 | 実装済み |
| `napi` | Node.js ネイティブバインディング | 実装済み |

## リポジトリ構成

```
dghs-imgutils-rs/
├── AGENTS.md           # このファイル
├── README.md           # 英語版プロジェクトドキュメント
├── README.ja.md        # 日本語版プロジェクトドキュメント
├── CONTRIBUTING.md     # 英語版開発者ガイド
├── CONTRIBUTING.ja.md  # 日本語版開発者ガイド
├── Cargo.toml          # Rust パッケージ設定
├── package.json        # Node.js パッケージ設定
├── mise.toml           # ツールバージョン管理（Rust, Python, Node, Bun, uv）
├── src/
│   ├── lib.rs          # ライブラリルート
│   ├── main.rs         # バイナリエントリポイント
│   ├── config.rs       # パッケージメタ情報
│   ├── hub/            # HuggingFace Hub 統合
│   ├── image/          # 画像I/Oユーティリティ
│   ├── inference/      # ONNX Runtime セッション管理
│   ├── generic/        # 汎用MLエンジン
│   ├── tagging/        # 自動タグ付け
│   ├── detect/         # 物体検出
│   ├── segment/        # セグメンテーション
│   ├── edge/           # エッジ検出
│   ├── metrics/        # 類似度メトリクス
│   ├── validate/       # 画像検証
│   ├── ocr/            # OCR
│   ├── pose/           # 姿勢推定
│   ├── metadata/       # メタデータ
│   ├── sd/             # SD メタデータ
│   ├── operate/        # 画像操作
│   ├── restore/        # 画像復元
│   ├── upscale/        # 超解像
│   ├── ascii/          # アスキーアート
│   ├── resource/       # リソース管理
│   ├── utils/          # 共通ユーティリティ
│   └── napi/           # Node.js バインディング
└── imgutils/           # 参照元 Python ライブラリ（読み取り専用）
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
| メイン言語 | Rust (2024 edition) |
| 推論エンジン | ort (ONNX Runtime) v2.0.0-rc.12 |
| TSバインディング | napi-rs |
| Python参照実装 | Python 3.12 + uv |
| ランタイム (JS) | Node.js (latest), Bun (latest) |
| ビルドツール | Cargo |
| ツール管理 | mise |

## 主な依存クレート

| クレート | 用途 |
|----------|------|
| `ort` | ONNX Runtime 推論（CUDA, TensorRT, OpenVINO, DirectML 対応） |
| `hf-hub` | HuggingFace Hub からのモデルダウンロード・キャッシュ管理 |
| `image` | 画像の読み込み・変換・保存 |
| `ndarray` | 多次元配列・テンソル操作 |
| `thiserror` | エラー型の定義 |
| `serde` / `serde_json` | JSON のシリアライズ／デシリアライズ |
| `imageproc` | 画像処理アルゴリズム |
| `napi` / `napi-derive` | Node.js ネイティブバインディング |

## napi-rs バインディング

Node.js バインディングは `src/napi/` ディレクトリに実装されています。

```bash
# バインディングのビルド（リリース）
npm run build

# バインディングのビルド（デバッグ）
npm run build:debug

# TypeScript 型定義は自動生成される
# -> index.d.ts
```

### 公開されている主な API（TypeScript）

```typescript
// タグ付け
getPixaiTags(path, modelName?, thresholds?): PixaiTagResult

// キャラクター類似度
ccipGetEmbedding(path, modelName?): number[]
ccipDistance(emb1, emb2, modelName?): number
ccipSame(emb1, emb2, threshold?, modelName?): boolean
ccipCluster(embeddings, eps?, minSamples?, modelName?): number[]

// 物体検出
detectFaces(path, level?, version?, confThreshold?, iouThreshold?): NapiDetection[]
detectHeads(path, level?, version?, confThreshold?, iouThreshold?): NapiDetection[]
detectPerson(path, level?, version?, confThreshold?, iouThreshold?): NapiDetection[]

// セグメンテーション
getIsnetisMask(path, scale?): number[][]
segmentRgbaWithIsnetis(path, scale?): number[]
```

## コーディング規約

### Rust

- **Edition**: Rust 2024
- フォーマットには `cargo fmt` を使用してください
- リントは `cargo clippy -- -D warnings` で確認してください
- 新しい機能には必ずドキュメントコメント（`///`）を付与してください
- `unsafe` コードは最小限に留め、使用する場合はコメントで理由を明記してください

### 命名規則

- モジュール: `snake_case`（例: `tagging`, `detect_faces`）
- 型: `PascalCase`（例: `TagResult`, `Detection`）
- 関数: `snake_case`（例: `get_pixai_tags`, `detect_faces`）
- 定数: `SCREAMING_SNAKE_CASE`（例: `DEFAULT_MODEL`）

### ファイル操作

- `src/main.rs` はバイナリエントリポイントです。ライブラリコードは `src/lib.rs` や `src/` 配下のモジュールに分離してください
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
cargo clippy -- -D warnings

# API ドキュメント生成
cargo doc --open

# Node.js バインディングビルド
npm run build

# Python参照実装の確認（imgutils/ 内）
cd imgutils && uv run python -c "import imgutils; print(imgutils.__version__)"
```

## 実装方針

1. **参照実装の尊重**: Python の `imgutils` ライブラリの API 設計・挙動を参考にしながら、Rust らしい API を設計してください
2. **パフォーマンス優先**: Rust の強みである安全性とパフォーマンスを活かした実装を心がけてください
3. **段階的実装**: 全機能を一度に実装しようとせず、コア機能から始めて段階的に拡張してください
4. **テスト駆動**: 各モジュールにユニットテストを必ず追加してください
5. **エラーハンドリング**: `thiserror` を使用した型安全なエラー処理を徹底してください

## モデル管理

モデルファイルは HuggingFace Hub からオンデマンドでダウンロードされ、ローカルにキャッシュされます。

- キャッシュディレクトリ: `~/.cache/huggingface/`（環境変数 `HF_HOME` で変更可能）
- ストレージディレクトリ: `~/.cache/dghs-imgutils-rs/`（環境変数 `IU_HOME` で変更可能）

## 注意事項

- Python の参照実装（`imgutils/`）は Rust 実装の仕様参照として使用してください
- `ort` クレート v2.0.0-rc.12 を使用中（CUDA, TensorRT, OpenVINO, DirectML 対応）
- napi-rs バインディングは `napi` フィーチャーフラグで有効化（デフォルトで有効）
