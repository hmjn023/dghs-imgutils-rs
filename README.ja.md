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
dghs-imgutils-rs = "0.3.0"
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

### ローカル環境の一括セットアップ

このリポジトリはRust 2024、Node.js、`mise`を使います。`mise`を使う場合は、
次の手順で固定された開発ツールを導入し、RustライブラリとNode.js native
addonをビルドできます。

```bash
git clone https://github.com/hmjn023/dghs-imgutils-rs.git
cd dghs-imgutils-rs
mise install
export HF_HOME="$PWD/.cache/huggingface"
export IU_HOME="$PWD/.cache/imgutils"
cargo build --release
cargo test --lib --no-default-features
npm install
npm run build
```

モデルは初回使用時にHuggingFaceからダウンロードされます。標準のhome directory
を使わない場合は、アプリケーション起動前に指定します。

```bash
export HF_HOME=/var/cache/dghs/huggingface
export IU_HOME=/var/cache/dghs/imgutils
```

native addonはONNX Runtimeやvendorのdriver libraryを同梱しません。アプリケーション
起動前に、対象hardwareと互換性のあるRuntime/provider配布物を用意してください。

### Provider別のCargo feature

Cargo featureはproviderのRust bindingをビルドするもので、provider runtimeを
ダウンロード・同梱するものではありません。

| 対象 | Cargo feature | 必要なRuntime/host |
|---|---|---|
| CPU | `napi`またはprovider featureなし | ONNX Runtime CPU共有ライブラリ |
| Intel CPU/GPU/NPU | `openvino` | OpenVINO対応ONNX RuntimeとIntel driver/runtime |
| NVIDIA GPU | `cuda` | CUDA対応ONNX Runtime、NVIDIA driver、CUDA、cuDNN |
| NVIDIA TensorRT | `cuda,tensorrt` | TensorRTとCUDA一式 |
| AMD Radeon GPU | `amd-gpu` | ROCm、MIGraphX、MIGraphX対応ONNX Runtime |
| AMD XDNA/Vitis-AI | `amd-npu` | 対象機種用Vitis-AI/XDNA runtime（experimental） |

Node.js addonではdefaultの`napi` featureが有効です。全providerのbindingを含む
ビルドも可能ですが、ロードするRuntimeが実際にproviderを含んでいる必要があります。

```bash
cargo build --release --all-features
```

`--all-features`はcompile確認であり、対象hostで全providerが利用可能である証拠では
ありません。

### ONNX Runtimeの配置とdynamic loading

このcrateの`ort`はdynamic loadingを使うため、ONNX Runtimeを静的リンク・同梱
しません。`ORT_DYLIB_PATH`には互換性のある`libonnxruntime.so`を1つ指定し、
Execution Provider共有ライブラリとvendor依存ライブラリをdynamic loaderから
探索できるようにします。

`ORT_DYLIB_PATH`はONNX Runtime core libraryの単一パスであり、複数パスを並べる
変数ではありません。ただし、1つのカスタムRuntime配布物が複数の互換providerを
含む場合は、同一processで複数EPを利用できます。同じ配布物のファイルをまとめて
配置してください。

```text
libonnxruntime.so
libonnxruntime_providers_*.so
libonnxruntime_providers_shared.so
vendor library（CUDA/cuDNN、ROCm/MIGraphX、OpenVINO、TensorRTなど）
```

```bash
export ORT_DYLIB_PATH=/opt/dghs/ort-profile/lib/libonnxruntime.so
export LD_LIBRARY_PATH=/opt/dghs/ort-profile/lib:${LD_LIBRARY_PATH:-}
```

異なるORT、ROCm、CUDA、OpenVINO、TensorRTの配布物からcore/providerを混在させない
でください。詳細は[ONNX Runtime Execution Provider一覧](https://onnxruntime.ai/docs/execution-providers/)、
[provider build guide](https://onnxruntime.ai/docs/build/eps.html)、`ort`の
[dynamic linking guide](https://github.com/pykeio/ort/blob/main/docs/content/setup/linking.mdx)
を参照してください。

### 共通workerの起動

各workerは、1つのprovider feature set、互換性のあるRuntime profile、明示的なbackend、
専用のcache directoryで起動します。

このrepositoryはlibrary/native addonを提供しますが、常駐RPC workerやschedulerは
提供しません。利用側applicationがRuntime profileごとにprocessを起動し、適切なprocessへ
requestをroutingします。

```bash
# CPU workerの例
cargo build --release --features napi
export DGHS_BACKEND=cpu
export DGHS_PRECISION=fp32
export ORT_DYLIB_PATH=/opt/dghs/ort-cpu/lib/libonnxruntime.so
export IU_HOME=/var/cache/dghs/cpu
```

実際のジョブを送る前に、capability matrixを表示し、小さなONNX modelでstrict
sessionを作成します。

```bash
cargo run --no-default-features --features openvino --example ep_probe -- \
  path/to/model.onnx
```

workerに対応するfeature setへ置き換えてください。probeはロード済みRuntimeに
providerがあるかを確認しますが、provider probeの成功だけでは特定modelのcompile・
実行成功までは保証しません。

### 環境変数

これらはsession作成時に読み込まれます。programmaticな`configureInference`はworkerの
defaultとして環境変数より優先され、N-API関数の最後の`inferenceOptions`は1回の呼び出し
だけ上書きします。

| 変数 | 例 | 用途 |
|---|---|---|
| `DGHS_BACKEND` | `amd-gpu` | `auto`、`cpu`、`openvino`、`cuda`、`tensorrt`、`directml`、`amd-gpu`、`amd-npu` |
| `DGHS_PRECISION` | `fp16` | `auto`、`fp32`、`fp16`、`bf16`、`int8` |
| `DGHS_DEVICE_ID` | `0` | GPU/providerのdevice ordinal |
| `DGHS_ORT_DEVICE` | `GPU` | OpenVINO device policy（`CPU`、`GPU`、`NPU`、`AUTO`など） |
| `DGHS_VITIS_CONFIG` | `/opt/vitis-ai/model.json` | Vitis-AI compiler/provider設定（必須） |
| `DGHS_EP_CACHE_DIR` | `/var/cache/dghs/ep` | Vitis-AIのprovider compile/cache directory（AMD NPU専用、MIGraphXでは使用しません） |
| `DGHS_MIGRAPHX_INT8_CALIBRATION_TABLE` | `/opt/models/calibration.table` | MIGraphX INT8 calibration table |
| `DGHS_MIGRAPHX_EXHAUSTIVE_TUNE` | `true` | MIGraphX exhaustive tuningを有効化 |

Vitis-AI workerやRuntime profileごとに`DGHS_EP_CACHE_DIR`を分けてください。MIGraphXは
provider固有の設定を使います。cache identityにはmodel hash、backend、precision、device、
runtime、provider fingerprintが含まれます。

### AMD Radeon GPU / XDNA2 NPU worker

AMDアクセラレーションはworker単位の明示的な契約です。providerをbinaryへ組み込み、
互換性のあるORT profileを使います。MIGraphXとVitis-AIが異なるRuntime配布物になる
場合はGPU/NPUを別processまたは別containerに分けます。1つのRuntime profileが両方を
含む場合は、1 process内の呼び出し単位選択を利用できます。

```bash
# AMD GPU / MIGraphX worker
cargo build --release --no-default-features --features amd-gpu
export DGHS_BACKEND=amd-gpu
export DGHS_PRECISION=fp16       # fp32, fp16, int8
export ORT_DYLIB_PATH=/opt/ort-migraphx/lib/libonnxruntime.so

# AMD XDNA2 / Vitis-AI worker（experimental）
cargo build --release --no-default-features --features amd-npu
export DGHS_BACKEND=amd-npu
export DGHS_PRECISION=bf16       # manifestが対応するモデルだけ
export DGHS_VITIS_CONFIG=/opt/vitis-ai/wd14.json
export DGHS_EP_CACHE_DIR=/var/cache/dghs-imgutils/vitis
export ORT_DYLIB_PATH=/opt/ort-vitis/lib/libonnxruntime.so
```

#### AMD Radeon GPU / MIGraphXの前提

対象GPUに対応したROCmを先に導入し、その後MIGraphXを導入します。Ubuntuでは
ROCmのrepository設定後に、次のようにMIGraphX packageを導入できます。

```bash
# 正確なOS/GPU手順はROCm公式ガイドに従う
sudo apt update
sudo apt install -y migraphx

# kernel/ROCmが認識したGPU architectureを確認
rocminfo | rg 'Name:.*gfx'
```

Radeon 8060Sでは`gfx1151`が想定されますが、実際のdriverが出力する値を正とします。
ONNX Runtimeを自前ビルドする場合は、MIGraphX用のbuild pathを使い、インストール先を
`--migraphx_home`で指定します。

```bash
./build.sh --config Release --parallel --build_shared_lib \
  --use_migraphx --migraphx_home /opt/rocm
```

利用可能なら、対応バージョンが固定されたprebuilt ORT/MIGraphX配布物を使ってください。
[ONNX Runtime MIGraphX guide](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
と[ROCm MIGraphX installation guide](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/install/install-migraphx.html)
で対応versionの組み合わせを確認し、`ep_probe`実行時にはRuntimeとROCmのlibrary
directoryを`LD_LIBRARY_PATH`へ追加します。

利用プログラム側から、最初のmodel API呼び出し前にworkerを選択できます。
programmatic設定は`DGHS_*`環境変数より優先されます。

```rust
use dghs_imgutils_rs::inference::{
    configure_inference, Backend, Precision, SessionOptions,
};

configure_inference(
    SessionOptions::for_backend(Backend::AmdGpu)
        .with_precision(Precision::Fp16)
        .with_device_id(0),
)?;
```

TypeScript側からも同じ設定が可能です。

```typescript
import { configureInference, getInferenceConfiguration } from 'dghs-imgutils-rs';

configureInference({ backend: 'amd-gpu', precision: 'fp16', deviceId: 0 });
console.log(getInferenceConfiguration());
```

UIから呼び出しごとに選択する場合は、モデルを使うN-API関数の最後の引数に
任意の`inferenceOptions`を渡せます。既存の引数を壊さず、リクエスト単位で
backend、precision、deviceを選択できます。

```typescript
import { getPixaiTags } from 'dghs-imgutils-rs';

const gpu = await getPixaiTags(imagePath, undefined, undefined, {
  backend: 'amd-gpu',
  precision: 'fp16',
  deviceId: 0,
});

const cpu = await getPixaiTags(imagePath, undefined, undefined, {
  backend: 'cpu',
  precision: 'fp32',
});
```

同じ最後の引数はtagging、detection、segmentation、OCR、pose、restore/upscale、
generic model、metrics、validation、NSFW modelのAPIで利用できます。session cacheは
model/backend/precision/deviceごとに分離されるため、元の設定へ戻したときはwarm
sessionを再利用できます。処理中の呼び出しは開始時に選択したsessionを使い続け、
UIの変更は新しい呼び出しにだけ反映されます。

OpenVINOを使う場合は、Rustでは`.with_openvino_device_type("GPU")`、TypeScriptでは
`openvinoDeviceType: 'GPU'`を指定します。

ロード済みruntimeのproviderを確認するには`probeInferenceBackends()`を使います。
ONNX Runtimeとprovider libraryはprocess単位なので、設定はapplication起動時に一度
行ってください。`configureInference`は起動時のデフォルトであり、呼び出し単位では
ロード済みruntimeに存在するproviderの中から`inferenceOptions`で選択できます。
同一Rust processで複数backendを使う場合は、
`create_onnx_session_with_options`または`get_or_create_session_with_options`へ明示的に
optionsを渡します。

呼び出し単位の選択で、起動後に`ORT_DYLIB_PATH`を差し替えることはできません。
GPUとNPUで異なるONNX Runtime/provider配布物が必要な場合は、GPU workerとNPU workerを
別プロセスまたは別コンテナで起動し、UI/routerから適切なworkerへリクエストを送ります。

`DGHS_BACKEND=amd-gpu`、`amd-npu`、`cpu`はstrict指定です。provider初期化、
モデル非対応、compile失敗は`BackendUnavailable`、`ModelUnsupported`、
`CompilationFailed`として返し、明示的なAMD workerがCPUへ黙ってfallbackする
ことはありません。既存APIとの互換性のため`auto`も残し、従来の自動provider
選択を維持します。

`amd-gpu` featureは[ONNX Runtime MIGraphX provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)を有効化します。
[Radeon 8060SのGPU target](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)は`gfx1151`です。ROCmとkernelの組み合わせは実機で検証し、
`device_id`、FP16、INT8 calibration、`exhaustive_tune`はMIGraphX固有設定として
扱います。このcrateはMIGraphXのBF16を汎用対応とは宣言せず、永続`.mxr`
compiled model cacheもv1の必須要件にしません。プロセス内session cacheのkeyは
モデルSHA-256、backend、precision、device、runtime/provider fingerprintを含みます。

`amd-npu` featureは[Vitis-AI provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)を有効化します。`DGHS_VITIS_CONFIG`を
必ず指定し、cache keyにはモデル内容とruntime identityを使います。Vitisはgraphを
NPUとCPUへ分割できるため、provider登録成功だけではNPU実行の証拠になりません。
subgraph数、CPU fallback、cold/warm latencyをprobeとbenchmarkで記録します。
[`amdxdna`](https://github.com/amd/xdna-driver)、XRT、`/dev/accel/accel0`、kernel、runtime、compiled modelを対象機で確認
するまでexperimentalとして扱います。

#### AMD XDNA / Vitis-AIの前提

すべてのRyzen AI/XDNA targetを一括で有効化する汎用Linux installerはありません。
上流ONNX RuntimeのVitis-AI Linux対応は主にAMD Adaptable SoC向けで、Ryzen AI Software
はtargetとOSごとに手順が異なります。[Vitis-AI EPのinstallation guide](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
と対象機種のXDNA/XRT手順に従い、同じworker userからdeviceを確認してください。

```bash
test -e /dev/accel/accel0
ls -l /dev/accel/accel0
```

modelはVitis互換のdeployment artifactが必要です。通常はBF16またはINT8へ量子化した
ONNX、static shape、明示的なcompiler configurationを使用します。このcrateはFP32
graphを自動BF16変換しません。初回Vitis sessionは数分compileする場合があるため、
専用cache directoryを使い、cold/warm latencyを測定してください。provider登録が
成功しても、未対応subgraphがCPUで実行される場合があります。

ONNX Runtime本体とprovider共有ライブラリはcrateに同梱されません。互換性のある
同一配布物から揃え、core libraryとprovider libraryを別runtimeから混在させないで
ください。Ubuntu、Arch、Proxmox LXC対応はCargo featureだけでは保証せず、実機の
runtime matrixで検証します。

`ModelManifest`でモデルSHA-256、opset、BF16/INT8対応variant、static shape、compiler
設定を管理します。`Precision::Auto`はFP32 ONNXを自動的にBF16へ変換する保証ではありません。

例:

```json
{
  "name": "wd14",
  "model_sha256": "<sha256>",
  "opset": 17,
  "preferred_backends": ["amd-npu", "amd-gpu", "cpu"],
  "default_precision": "bf16",
  "supported_precisions": ["bf16", "int8"],
  "static_shape": true,
  "compiler_config": "/opt/vitis-ai/wd14.json"
}
```

workerのprovider能力を表示し、任意でstrict sessionを作成するprobeです。

```bash
cargo run --no-default-features --features amd-npu --example ep_probe -- \
  path/to/model.onnx
```

build wiringとpolicy logicの検証matrix:

```bash
cargo check --no-default-features
cargo check --no-default-features --features amd-gpu,amd-npu
cargo check --no-default-features --features cuda,tensorrt,directml,openvino
cargo check --all-features
cargo check --no-default-features --examples
cargo test --lib --no-default-features
npm run build
```

これらはfeature wiringとpolicy logicを検証するもので、実機実行の証拠では
ありません。deployment reportにはruntime build、kernel/driver、`gfx1151`
またはXDNA device discovery、model hash、provider compile結果、subgraph/CPU
fallbackの証拠、cold/warm latencyを記録してください。
storage directoryを作成できないtest環境ではstorage-directory testが失敗する
ことがあります。これはAMD providerの合否とは分離して扱ってください。

### NVIDIA CUDA

このcrateはCUDA、TensorRT、DirectML、OpenVINO、MIGraphX、Vitis-AIをoptionalな
Cargo featureとして公開します。プロセス起動時に`ort`が`ORT_DYLIB_PATH`から一つの
ONNX Runtime共有ライブラリをロードするため、使用するhardwareに対応したruntimeを
指定してください。

NVIDIA GPUでは、CUDA対応のONNX Runtime共有ライブラリを指定します。Intel向けの `onnxruntime-openvino` 配布物にはCUDA Execution Providerは含まれません。

```bash
# host driverを確認
nvidia-smi

# 使用するprovider bindingをビルド
cargo build --release --features cuda
# TensorRTもCUDAの前提を使う
cargo build --release --features cuda,tensorrt

# CUDA対応ONNX Runtimeの配置先を指定
export ORT_DYLIB_PATH="/opt/onnxruntime-cuda/lib/libonnxruntime.so"
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/opt/cudnn
export TENSORRT_HOME=/opt/tensorrt       # TensorRT workerだけで使用
export LD_LIBRARY_PATH="/opt/onnxruntime-cuda/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib:${TENSORRT_HOME}/lib:${LD_LIBRARY_PATH:-}"

# ロードしたランタイムにCUDA EPがあれば自動検出される
npm run build
```

同じRuntime profileに`libonnxruntime_providers_cuda.so`と、TensorRT使用時は対応する
TensorRT provider libraryが必要です。CUDA、cuDNN、TensorRT、NVIDIA driver、ONNX
Runtimeのversionを対応matrixに合わせてください。[CUDA EP要件](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、
[TensorRT EP要件](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)、
[ONNX Runtime install matrix](https://onnxruntime.ai/docs/install/)を正とします。
ORTを自前ビルドする場合は、リリースに対応した`--use_cuda`、`--cuda_home`、
`--use_tensorrt`、`--tensorrt_home`を公式build guideに従って指定し、別配布物の
libraryを混在させないでください。

### Windows DirectML

DirectMLはWindows専用です。DirectML対応のONNX Runtime配布物を使い、
`onnxruntime.dll`とprovider DLLを同じprofileに置き、`directml` featureを有効にします。

```powershell
cargo build --release --features directml
$env:DGHS_BACKEND = "directml"
$env:DGHS_PRECISION = "fp32"
$env:ORT_DYLIB_PATH = "C:\opt\onnxruntime-directml\onnxruntime.dll"
$env:PATH = "C:\opt\onnxruntime-directml;" + $env:PATH
npm run build
```

DirectMLのdevice availabilityはWindows graphics driverとロードしたRuntimeに依存します。
ジョブを送る前に`probeInferenceBackends()`または`ep_probe`で確認してください。新規の
Windows deploymentでは、インストールしたONNX Runtime versionに対応する
[DirectML/WinML guidance](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
も確認します。

### Intel NPU / GPU（XPU、Linux）

以下のOpenVINOセットアップはLinux向けです。Windowsでは `onnxruntime.dll` を使用し、対応するOpenVINOランタイムDLLを別途インストールして、そのディレクトリをアプリケーション起動前に `PATH` へ追加してください。

このリポジトリは `ort` の動的ロードと OpenVINO Execution Provider を使います。Python APIを使うのではなく、公式 `onnxruntime-openvino` パッケージに含まれるネイティブ共有ライブラリだけをRust/Node.jsからロードします。

まず、OpenVINO対応のONNX Runtime一式をプロジェクト内に展開します。

```bash
uv pip install --target .ort-runtime \
  --python-version 3.13 --only-binary=:all: --no-deps \
  onnxruntime-openvino==1.24.1

export ORT_DYLIB_PATH="$PWD/.ort-runtime/onnxruntime/capi/libonnxruntime.so.1.24.1"
```

systemのOpenVINOを使う場合は、native application用の環境設定scriptを先に読み込みます。

```bash
source /opt/intel/openvino/setupvars.sh
cargo build --release --features openvino
```

実行時にデバイスを選択します。`NPU` は Intel NPU、`GPU` は Intel XPU（GPU）です。

```bash
# Intel NPU
export DGHS_ORT_DEVICE=NPU

# Intel XPU/GPU。LinuxではIntel OpenCL ICDを明示する
export DGHS_ORT_DEVICE=GPU
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/
```

未設定時は `AUTO:NPU,GPU,CPU` で、NPU → GPU → CPUの順に選択します。アプリケーションを起動するプロセスに `ORT_DYLIB_PATH` と上記の環境変数を設定してください。GPUを使うLinux環境では、Intel GPUドライバーと `/etc/OpenCL/vendors/` 内のICDファイルが必要です。

strictにproviderを選択する場合はbackendも指定します。

```bash
export DGHS_BACKEND=openvino
export DGHS_PRECISION=fp16
export DGHS_ORT_DEVICE=GPU   # CPU、GPU、NPU、AUTOなど
```

Intel GPU/NPUのkernel driver、Level Zero/OpenCL runtime、device pluginは別途必要です。
同じdriverとOpenVINO環境を読み込んだ状態で`ep_probe`を実行してください。Rustの
compile成功だけではIntel hardwareの対応確認にはなりません。

```bash
# ネイティブアドオンをビルド
npm run build
```

動作確認用の最小モデルでRust側のセッション作成を確認できます。

```bash
cargo run --no-default-features --example intel_ep_probe -- \
  .ort-runtime/onnxruntime/datasets/sigmoid.onnx
```

`onnxruntime-openvino` の共有ライブラリは、[ONNX Runtime OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) の公式配布物です。別のOpenVINO共有ライブラリを混在させず、同じ `capi` ディレクトリのファイル一式を使ってください。

## ドキュメント

```bash
# Rust API ドキュメントの生成と閲覧
cargo doc --open
```

## Python 参照実装

本プロジェクトは Python ライブラリ [dghs-imgutils](https://github.com/deepghs/imgutils) の Rust 移植版です。Python のソースコードは `imgutils/` ディレクトリに読み取り専用の参照として含まれています。

## ライセンス

Apache License 2.0
