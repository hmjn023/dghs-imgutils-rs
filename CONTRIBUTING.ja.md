# dghs-imgutils-rs への貢献

貢献に興味を持っていただきありがとうございます！このガイドでは開発の始め方を説明します。

## 目次

- [開発環境](#開発環境)
- [プロジェクト構造](#プロジェクト構造)
- [ビルド方法](#ビルド方法)
- [テスト](#テスト)
- [コーディング規約](#コーディング規約)
- [変更の提出](#変更の提出)

## 開発環境

### 必要なもの

- [Rust](https://www.rust-lang.org/tools/install)（最新安定版）
- [Node.js](https://nodejs.org/)（最新LTS）
- [mise](https://mise.jdx.dev/)（オプション、ツールバージョン管理用）

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/your-org/dghs-imgutils-rs.git
cd dghs-imgutils-rs

# ツールバージョンのインストール（mise 使用時）
mise install

# Rust ライブラリのビルド
cargo build

# テストの実行
cargo test
```

## プロジェクト構造

```
src/
├── lib.rs          # ライブラリルート（公開モジュール宣言）
├── main.rs         # バイナリエントリポイント
├── config.rs       # パッケージメタ情報（バージョン、タイトルなど）
├── hub/            # HuggingFace Hub モデルダウンロード・キャッシュ
├── image/          # 画像I/O、フォーマット変換、ユーティリティ
├── inference/      # ONNX Runtime セッション管理
├── generic/        # 汎用MLエンジン（YOLO, CLIP, SigLIP など）
├── tagging/        # 自動タグ付け（WD14, DeepDanbooru, PixAI など）
├── detect/         # 物体検出（顔、頭部、人物など）
├── segment/        # セグメンテーション・背景透過
├── edge/           # エッジ検出（Canny, Lineart）
├── metrics/        # 類似度メトリクス（CCIP, LPIPS, 美的スコア）
├── validate/       # 画像検証（破損、モノクロ、NSFW）
├── ocr/            # テキスト検出・認識
├── pose/           # 姿勢推定（DWpose）
├── metadata/       # 画像メタデータ・ステガノグラフィ
├── sd/             # Stable Diffusion メタデータパーサー
├── operate/        # 画像操作（リサイズ、トリミング、検閲）
├── restore/        # 画像復元（NafNet, SCUNet）
├── upscale/        # 超解像（CDC）
├── ascii/          # アスキーアート生成
├── resource/       # 背景画像管理
├── utils/          # 共通ユーティリティ（キャッシュ、ストレージ）
└── napi/           # Node.js ネイティブバインディング（napi-rs）
```

### 主要な設計原則

1. **Rust コア、オプションバインディング**: コアロジックは `src/` 配下の純粋な Rust モジュールに配置。`napi` フィーチャーフラグで Node.js バインディングを有効化。
2. **Python 参照実装**: `imgutils/` ディレクトリにオリジナルの Python 実装が読み取り専用の参照として含まれています。
3. **HuggingFace Hub**: モデルは HuggingFace Hub からオンデマンドでダウンロードされ、ローカルにキャッシュされます。

## ビルド方法

```bash
# デバッグビルド
cargo build

# リリースビルド
cargo build --release

# napi-rs バインディングのビルド
npm run build

# デバッグバインディングのビルド
npm run build:debug
```

## テスト

```bash
# 全 Rust テストの実行
cargo test

# 特定モジュールのテスト実行
cargo test tagging::

# 出力付きテスト実行
cargo test -- --nocapture
```

## コーディング規約

### Rust

- **エディション**: Rust 2024
- **フォーマッタ**: コミット前に `cargo fmt` を使用
- **リンター**: `cargo clippy -- -D warnings` で警告をチェック
- **ドキュメント**: すべての公開アイテムにドキュメントコメント（`///`）を追加
- **安全性**: `unsafe` コードは最小限に留め、使用時は理由を記録

```bash
# コードのフォーマット
cargo fmt

# リントチェック
cargo clippy -- -D warnings
```

### 命名規則

- モジュール: `snake_case`（例: `tagging`, `detect_faces`）
- 型: `PascalCase`（例: `TagResult`, `Detection`）
- 関数: `snake_case`（例: `get_pixai_tags`, `detect_faces`）
- 定数: `SCREAMING_SNAKE_CASE`（例: `DEFAULT_MODEL`）

### ドキュメント

すべての公開 API にドキュメントコメントを追加してください：

```rust
/// アニメ画像内の顔を検出します。
///
/// # 引数
///
/// * `image` - 入力画像
/// * `level` - モデルのレベル（"s" または "n"）
/// * `version` - モデルバージョン（"v0", "v1", "v1.3", "v1.4"）
/// * `conf_threshold` - 信頼度しきい値（デフォルト: 0.25）
/// * `iou_threshold` - NMS の IoU しきい値（デフォルト: 0.7）
///
/// # 戻り値
///
/// バウンディングボックスとスコアを含む検出結果のベクタ。
pub fn detect_faces(
    image: &DynamicImage,
    level: &str,
    version: &str,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>, InferenceError> {
    // ...
}
```

## 変更の提出

1. リポジトリを**フォーク**
2. 機能または修正用の**ブランチを作成**
3. 新機能には**テストを記述**
4. **チェックを実行**:
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   ```
5. 明確なメッセージで**コミット**
6. 変更内容の説明を添えて**Pull Requestを開く**

### コミットメッセージ

明確で説明的なコミットメッセージを使用してください：

```
feat: WD14 タギングサポートの追加
fix: 顔検出での RGBA 画像の処理
docs: segment モジュールの API ドキュメント更新
refactor: 共通 YOLO 予測ロジックの抽出
```

### Pull Request ガイドライン

- PR は単一の機能または修正に集中させてください
- 新機能にはテストを含めてください
- 公開 API を追加する場合はドキュメントを更新してください
- すべての CI チェックが通ることを確認してください

## 質問がある場合

Issue を開いて質問や議論をしてください。
