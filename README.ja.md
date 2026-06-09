# dghs-imgutils-rs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-edition.svg)](https://www.rust-lang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-native%20addon-green.svg)](https://nodejs.org/)

[dghs-imgutils](https://github.com/deepghs/imgutils) の高性能 Rust 実装です。アニメ・イラスト画像向けの画像処理・機械学習推論機能を提供します。

## 機能一覧

| モジュール | 説明 |
|-----------|------|
| **tagging** | 自動タグ付け（WD14, DeepDanbooru, PixAI など） |
| **detect** | 物体検出（顔、頭部、人物、検閲部位、テキストなど） |
| **segment** | キャラクターセグメンテーション・背景透過（ISNetIS） |
| **edge** | エッジ検出・線画生成（Canny, Lineart） |
| **metrics** | キャラクター類似度（CCIP）、美的スコアリング、LPIPS |
| **validate** | 画像検証（破損チェック、モノクロ判定、AI生成判定、NSFW判定） |
| **ocr** | テキスト検出・認識（PaddleOCR） |
| **pose** | 姿勢推定（DWpose / OpenPose 18キーポイント） |
| **metadata** | 画像メタデータ読み書き、LSBステガノグラフィ |
| **sd** | Stable Diffusion / NovelAI メタデータパーサー |
| **operate** | 画像リサイズ、自動トリミング、自動検閲 |
| **restore** | 画像復元（NafNet, SCUNet, 敵対的ノイズ除去） |
| **upscale** | 超解像（CDCアップスケーリング） |
| **ascii** | アスキーアート生成 |
| **resource** | 背景画像データセット管理 |

## インストール

### Rust

```toml
[dependencies]
dghs-imgutils-rs = "0.1.0"
```

### Node.js / TypeScript

```bash
npm install dghs-imgutils-rs
```

## クイックスタート

### Rust

```rust
use image::open;
use dghs_imgutils_rs::tagging::pixai::get_pixai_tags;
use dghs_imgutils_rs::detect::detect_faces;
use dghs_imgutils_rs::segment::segment_rgba_with_isnetis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = open("anime_character.jpg")?;

    // 自動タグ付け
    let tags = get_pixai_tags(&image, "v0.9", None)?;
    println!("タグ: {:?}", tags.general);

    // 顔検出
    let faces = detect_faces(&image, "s", "v1.4", 0.25, 0.7)?;
    println!("検出された顔: {}個", faces.len());

    // 背景透過
    let rgba = segment_rgba_with_isnetis(&image, 1024)?;
    rgba.save("character_transparent.png")?;

    Ok(())
}
```

### TypeScript / Node.js

```typescript
import {
  getPixaiTags,
  detectFaces,
  segmentRgbaWithIsnetis,
  ccipGetEmbedding,
  ccipSame,
} from 'dghs-imgutils-rs';

// 自動タグ付け
const tags = getPixaiTags('./anime_character.jpg');
console.log('一般タグ:', tags.general);
console.log('キャラクタータグ:', tags.character);

// 顔検出
const faces = detectFaces('./anime_character.jpg');
console.log(`検出された顔: ${faces.length}個`);
faces.forEach((face, i) => {
  console.log(`顔 #${i + 1}: bbox=[${face.bbox.x1}, ${face.bbox.y1}, ${face.bbox.x2}, ${face.bbox.y2}], スコア=${face.score}`);
});

// 背景透過（PNGバッファを返す）
const pngBytes = segmentRgbaWithIsnetis('./anime_character.jpg');

// キャラクター類似度判定
const emb1 = ccipGetEmbedding('./character_a.jpg');
const emb2 = ccipGetEmbedding('./character_b.jpg');
const isSame = ccipSame(emb1, emb2);
console.log(`同一キャラクター: ${isSame}`);
```

## サポートされているモデル

モデルは初回使用時に [HuggingFace Hub](https://huggingface.co/) から自動的にダウンロードされ、ローカルにキャッシュされます。

| 機能 | モデルリポジトリ |
|------|------------------|
| 顔検出 | `deepghs/anime_face_detection` |
| 頭部検出 | `deepghs/anime_head_detection` |
| 人物検出 | `deepghs/anime_person_detection` |
| WD14 タガー | `deepghs/wd14_tagger_with_embeddings` |
| PixAI タガー | `deepghs/pixai-tagger-v0.9-onnx` |
| CCIP（キャラクター） | `deepghs/ccip_onnx` |
| ISNetIS セグメンテーション | `skytnt/anime-seg` |
| Lineart | `deepghs/imgutils-models` |
| NafNet 画像復元 | `deepghs/image_restoration` |
| CDC 超解像 | `deepghs/cdc_anime_onnx` |
| PaddleOCR | `deepghs/paddleocr` |

## プロジェクト構造

```
dghs-imgutils-rs/
├── src/
│   ├── lib.rs          # ライブラリルート
│   ├── main.rs         # バイナリエントリポイント
│   ├── config.rs       # パッケージメタ情報
│   ├── hub/            # HuggingFace Hub 統合
│   ├── image/          # 画像I/Oユーティリティ
│   ├── inference/      # ONNX Runtime セッション管理
│   ├── generic/        # 汎用MLエンジン（YOLO, CLIP など）
│   ├── tagging/        # 自動タグ付けモジュール
│   ├── detect/         # 物体検出モジュール
│   ├── segment/        # セグメンテーションモジュール
│   ├── edge/           # エッジ検出モジュール
│   ├── metrics/        # 類似度・美的メトリクス
│   ├── validate/       # 画像検証モジュール
│   ├── ocr/            # OCRモジュール
│   ├── pose/           # 姿勢推定モジュール
│   ├── metadata/       # メタデータ読み書き
│   ├── sd/             # SD/NovelAI メタデータパーサー
│   ├── operate/        # 画像操作
│   ├── restore/        # 画像復元
│   ├── upscale/        # 超解像
│   ├── ascii/          # アスキーアート生成
│   ├── resource/       # リソース管理
│   ├── utils/          # 共通ユーティリティ
│   └── napi/           # Node.js ネイティブバインディング
├── Cargo.toml
├── package.json
└── imgutils/           # Python参照実装（読み取り専用）
```

## ビルド方法

```bash
# Rust ライブラリ
cargo build --release

# Node.js ネイティブアドオン
npm run build

# TypeScript 型定義の生成
npm run build  # --dts フラグを含む
```

## ドキュメント

```bash
# Rust API ドキュメントの生成と閲覧
cargo doc --open
```

## Python 参照実装

本プロジェクトは Python ライブラリ [dghs-imgutils](https://github.com/deepghs/imgutils) の Rust 移植版です。Python のソースコードは `imgutils/` ディレクトリに読み取り専用の参照として含まれています。

## ライセンス

Apache License 2.0
