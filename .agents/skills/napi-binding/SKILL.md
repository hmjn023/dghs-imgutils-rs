---
name: napi-binding
description: Rustで実装した高性能推論エンジンを napi-rs を用いて TypeScript/JavaScript からネイティブ速度で直接呼び出し可能にするネイティブバインディングの構築と、Rustテストとビルドの両立性を確保する検証一貫ガイドライン。
---

# NAPI-RS による JS/TS ネイティブバインディングの構築と検証ガイドライン

本ガイドラインは、Rust で実装された画像処理・機械学習推論ロジックを **napi-rs** を用いて TypeScript / Node.js 向けのアドオンとしてパッケージ化し、**「Rust 単体テスト実行」と「JS/TS アドオンビルド」の競合を完全に排除した状態で開発・検証を一貫して行うためのベストプラクティス** を規定します。

---

## 📌 1. アーキテクチャ設計の方針

napi-rs の導入にあたり、Rust コアロジックの再利用性と堅牢性を損なわないために、以下の原則に従います。

1. **コアロジックの非侵襲性 (Non-invasive)**
   `src/tagging/` や `src/metrics/` などの Rust コアコードには `#[napi]` アノテーションを一切記述しません。これにより、純粋な Rust ライブラリとしてのポータビリティを 100% 維持します。
2. **エントリポイントの完全隔離 (`src/napi/`)**
   JS/TS 露出用のラッパーコードはすべて専用のモジュール `src/napi/mod.rs` に集約し、`#[cfg(feature = "napi")]` 条件付きコンパイルで制御します。
3. **ユーザーフレンドリーな高レベル API の設計**
   JS/TS 側からは `DynamicImage` などの複雑な Rust 型を直接扱うのが難しいため、ラッパー関数は「画像ファイルパス (`String`)」や「プレーンな JS オブジェクト/Map」を受け取り、Rust 内部で高速デコードと自動前処理（アルファブレンド等）を一貫して行う設計とします。

---

## 🛠️ 2. 環境構築と設定手順

### ステップ 1: `package.json` の作成と初期化
プロジェクトのルートに `package.json` を作成し、アドオンビルド用の CLI 依存関係を追加します。

```json
{
  "name": "dghs-imgutils-rs",
  "version": "0.1.0",
  "main": "index.js",
  "types": "index.d.ts",
  "napi": {
    "name": "dghs-imgutils-rs"
  },
  "scripts": {
    "build": "napi build --platform --dts index.d.ts --release --cargo-flags=--lib",
    "build:debug": "napi build --platform --dts index.d.ts --cargo-flags=--lib"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.4",
    "typescript": "^5.4.5"
  }
}
```

> [!IMPORTANT]
> **`--cargo-flags=--lib` の必須性**
> プロジェクト内に `src/main.rs` (バイナリターゲット) が存在する場合、`napi build` のリンクフェーズで実行可能バイナリが Node.js の C シンボルを解決しようとしてリンクエラーを発生させます。
> `--cargo-flags=--lib` を付与することで、Cargo に対し「ライブラリターゲットのみをビルドする」よう指示し、バイナリターゲットの競合によるエラーを完全に防止できます。

---

### ステップ 2: `Cargo.toml` の更新 (機能フラグの切り分け)
通常の `cargo test` (静的リンク) 実行時に Node.js のシンボルが未定義でリンクエラーになる問題を完全に解決するため、**`napi` 依存関係をオプショナル（機能フラグ制御）**にします。

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
# ... 既存の依存 ...
napi = { version = "2.16.11", features = ["async", "serde-json", "napi4"], optional = true }
napi-derive = { version = "2.16.9", optional = true }

[build-dependencies]
napi-build = "2.1.4"

[features]
default = ["napi"]
napi = ["dep:napi", "dep:napi-derive"]
```

---

### ステップ 3: `build.rs` の作成
ルートに `build.rs` を作成し、napi-rs のネイティブリンクを設定します。

```rust
fn main() {
    napi_build::setup();
}
```

---

### ステップ 4: `src/lib.rs` でのモジュール公開
`napi` モジュールを機能フラグ `#[cfg(feature = "napi")]` の下で条件付きエクスポートします。

```rust
#[cfg(feature = "napi")]
/// JS/TS ネイティブバインディング (napi-rs) 用の公開インターフェース。
pub mod napi;
```

---

## ✍️ 3. ラッパー API の実装パターン (`src/napi/mod.rs`)

`src/napi/mod.rs` に `#[napi]` マクロを用いて、JS/TS 側のプリミティブ型と Rust 側のコア API を繋ぐブリッジ関数を記述します。

### パターン A: 画像パスを受け取って構造体を返す (PixAI Tagger)
```rust
use crate::tagging::pixai::get_pixai_tags as core_get_pixai_tags;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct PixaiTagResult {
    pub general: HashMap<String, f64>,
    pub character: HashMap<String, f64>,
}

#[napi]
pub fn get_pixai_tags(
    path: String,
    model_name: Option<String>,
    thresholds: Option<HashMap<String, f64>>,
) -> napi::Result<PixaiTagResult> {
    // パスから画像を開いてデコード (Rustコアの自動前処理へ流す)
    let image = image::open(&path)
        .map_err(|e| napi::Error::new(napi::Status::InvalidArg, format!("Failed to open image: {}", e)))?;

    let model = model_name.as_deref().unwrap_or("v0.9");
    let core_thresholds = thresholds.map(|map| {
        map.into_iter().map(|(k, v)| (k, v as f32)).collect::<HashMap<String, f32>>()
    });

    let result = core_get_pixai_tags(&image, model, core_thresholds)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("Prediction failed: {}", e)))?;

    Ok(PixaiTagResult {
        general: result.general.into_iter().map(|(k, v)| (k, v as f64)).collect(),
        character: result.character.into_iter().map(|(k, v)| (k, v as f64)).collect(),
    })
}
```

### パターン B: ベクトルの多次元配列を扱う (CCIP クラスタリング)
DBSCAN のような多次元配列（`Vec<Vec<f64>>`）を扱う場合、`ndarray::Array2` への変換や戻り値の型合わせを確実に行います。

```rust
use crate::metrics::ccip::ccip_clustering as core_ccip_clustering;

#[napi]
pub fn ccip_cluster(
    embeddings: Vec<Vec<f64>>,
    eps: Option<f64>,
    min_samples: Option<i32>,
    model_name: Option<String>,
) -> napi::Result<Vec<i32>> {
    let model = model_name.as_deref().unwrap_or("");
    let core_embs: Vec<Vec<f32>> = embeddings
        .into_iter()
        .map(|emb| emb.into_iter().map(|v| v as f32).collect())
        .collect();

    let core_eps = eps.map(|v| v as f32);
    let core_min_samples = min_samples.map(|v| v.max(0) as usize);

    let clusters = core_ccip_clustering(&core_embs, "dbscan", core_eps, core_min_samples, model)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("Clustering failed: {}", e)))?;

    Ok(clusters.into_iter().map(|v| v as i32).collect())
}
```

---

## 🧪 4. テストと二重検証 (Validation Suite)

堅牢性を担保するため、**「① Rust 側テストスイート」** と **「② JS/TS 側実稼働スクリプト」** の双方で厳密に検証を行います。

### ① Rust テストの実行 (リンクエラーの完全回避)
`napi` 依存関係を完全に排除してリンクエラーを防ぐため、**必ず `--no-default-features` を付与** してテストを実行します。

```bash
# 精度検証テストを CPU / 検出アクセラレータ上で実行
rtk cargo test --no-default-features --test ccip_benchmark -- --ignored
rtk cargo test --no-default-features --test pixai_benchmark -- --ignored
```

---

### ② JS/TS 呼び出し検証スクリプトの作成と実行
ビルドされた `.node` バイナリが実際に JS/TS からロードされ、100% 正確に機能することを検証する TypeScript スクリプト `test_napi.ts` を作成します。

```typescript
import { getPixaiTags, ccipGetEmbedding, ccipDistance, ccipSame, ccipCluster } from './index.js';
import * as path from 'path';

async function main() {
  console.log('🧪 JS/TS ネイティブアドオン動作検証開始\n');
  const imgPath = path.resolve('imgutils/test/testfile/skadi.jpg');

  // 1. PixAI Tagger の検証
  const tags = getPixaiTags(imgPath);
  console.log('✓ PixAI タグ予測成功:', Object.keys(tags.general).slice(0, 3));

  // 2. CCIP 特徴量抽出の検証
  const emb = ccipGetEmbedding(imgPath);
  console.log(`✓ CCIP 特徴抽出成功: ${emb.length}次元`);

  // 3. DBSCAN クラスタリングの検証
  const clusterIds = ccipCluster([emb]);
  console.log('✓ DBSCAN クラスタリング成功:', clusterIds);
}

main();
```

#### **ビルドと検証の実行**:
```bash
# 依存関係のインストール
npm install

# アドオンビルド (自動的に napi feature が有効になり、index.js/index.d.ts/.node が生成されます)
npm run build

# TS 検証テストの直接実行 (Bun 等を使用)
rtk bun test_napi.ts
```

---

## 🔍 5. トラブルシューティング

### Q1: `cargo test` 実行時に `undefined symbol: napi_get_global` などのリンクエラーが出る
* **原因**: `cargo test` 実行時に default feature である `"napi"` が有効になっており、テストバイナリに napi コードがリンクされようとしてシンボル不足になっています。
* **対策**: テスト実行時は必ず `--no-default-features` フラグを付与してください。
  `rtk cargo test --no-default-features ...`

### Q2: `npm run build` 時にバイナリターゲットのリンクエラーが発生する
* **原因**: プロジェクト内の `src/main.rs` も一緒にコンパイルされ、実行ファイルが Node.js のシンボルを解決しようとしています。
* **対策**: `package.json` のビルドスクリプトに `--cargo-flags=--lib` が付与されているか確認してください。これによりライブラリのビルドに一本化され、バイナリターゲットが無視されます。
