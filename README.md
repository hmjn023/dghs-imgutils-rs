# dghs-imgutils-rs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-edition.svg)](https://www.rust-lang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-native%20addon-green.svg)](https://nodejs.org/)

A high-performance Rust implementation of [dghs-imgutils](https://github.com/deepghs/imgutils) — an anime/illustration image processing and ML inference library.

## Features

| Module | Description |
|--------|-------------|
| **tagging** | Automatic tagging (WD14, DeepDanbooru, PixAI, etc.) |
| **detect** | Object detection (face, head, person, censor, text, etc.) |
| **segment** | Character segmentation & background removal (ISNetIS) |
| **edge** | Edge detection & line art generation (Canny, Lineart) |
| **metrics** | Character similarity (CCIP), aesthetic scoring, LPIPS |
| **validate** | Image validation (truncated, monochrome, AI-generated, NSFW) |
| **ocr** | Text detection & recognition (PaddleOCR) |
| **pose** | Pose estimation (DWpose / OpenPose 18 keypoints) |
| **metadata** | Image metadata read/write, LSB steganography |
| **sd** | Stable Diffusion / NovelAI metadata parser |
| **operate** | Image resize, auto-trim, auto-censor |
| **restore** | Image restoration (NafNet, SCUNet, adversarial noise removal) |
| **upscale** | Super-resolution (CDC upscaling) |
| **ascii** | ASCII art generation |
| **resource** | Background image dataset management |

## Installation

### Rust

```toml
[dependencies]
dghs-imgutils-rs = "0.3.0"
```

### Node.js / TypeScript

```bash
npm install dghs-imgutils-rs
```

## Quick Start

### Rust

```rust
use image::open;
use dghs_imgutils_rs::tagging::pixai::get_pixai_tags;
use dghs_imgutils_rs::detect::detect_faces;
use dghs_imgutils_rs::segment::segment_rgba_with_isnetis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = open("anime_character.jpg")?;

    // Auto-tagging
    let tags = get_pixai_tags(&image, "v0.9", None)?;
    println!("Tags: {:?}", tags.general);

    // Face detection
    let faces = detect_faces(&image, "s", "v1.4", 0.25, 0.7)?;
    println!("Detected {} faces", faces.len());

    // Background removal
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

// Auto-tagging (async to avoid blocking the event loop)
const tags = await getPixaiTags('./anime_character.jpg');
console.log('General tags:', tags.general);
console.log('Character tags:', tags.character);

// Face detection
const faces = detectFaces('./anime_character.jpg');
console.log(`Detected ${faces.length} faces`);
faces.forEach((face, i) => {
  console.log(`Face #${i + 1}: bbox=[${face.bbox.x1}, ${face.bbox.y1}, ${face.bbox.x2}, ${face.bbox.y2}], score=${face.score}`);
});

// Background removal (returns PNG buffer)
const pngBytes = await segmentRgbaWithIsnetis('./anime_character.jpg');

// Character similarity
const emb1 = await ccipGetEmbedding('./character_a.jpg');
const emb2 = await ccipGetEmbedding('./character_b.jpg');
const isSame = ccipSame(emb1, emb2);
console.log(`Same character: ${isSame}`);
```

## Supported Models

Models are automatically downloaded from [HuggingFace Hub](https://huggingface.co/) on first use and cached locally.

| Feature | Model Repository |
|---------|------------------|
| Face Detection | `deepghs/anime_face_detection` |
| Head Detection | `deepghs/anime_head_detection` |
| Person Detection | `deepghs/anime_person_detection` |
| WD14 Tagger | `deepghs/wd14_tagger_with_embeddings` |
| PixAI Tagger | `deepghs/pixai-tagger-v0.9-onnx` |
| CCIP (Character) | `deepghs/ccip_onnx` |
| ISNetIS Segmentation | `skytnt/anime-seg` |
| Lineart | `deepghs/imgutils-models` |
| NafNet Restoration | `deepghs/image_restoration` |
| CDC Super-Resolution | `deepghs/cdc_anime_onnx` |
| PaddleOCR | `deepghs/paddleocr` |

## Project Structure

```
dghs-imgutils-rs/
├── src/
│   ├── lib.rs          # Library root
│   ├── main.rs         # Binary entry point
│   ├── config.rs       # Package metadata
│   ├── hub/            # HuggingFace Hub integration
│   ├── image/          # Image I/O utilities
│   ├── inference/      # ONNX Runtime session management
│   ├── generic/        # Generic ML engine (YOLO, CLIP, etc.)
│   ├── tagging/        # Auto-tagging modules
│   ├── detect/         # Object detection modules
│   ├── segment/        # Segmentation modules
│   ├── edge/           # Edge detection modules
│   ├── metrics/        # Similarity & aesthetic metrics
│   ├── validate/       # Image validation modules
│   ├── ocr/            # OCR modules
│   ├── pose/           # Pose estimation modules
│   ├── metadata/       # Metadata read/write
│   ├── sd/             # SD/NovelAI metadata parser
│   ├── operate/        # Image operations
│   ├── restore/        # Image restoration
│   ├── upscale/        # Super-resolution
│   ├── ascii/          # ASCII art generation
│   ├── resource/       # Resource management
│   ├── utils/          # Common utilities
│   └── napi/           # Node.js native bindings
├── Cargo.toml
├── package.json
└── imgutils/           # Python reference implementation (read-only)
```

## Building

```bash
# Rust library
cargo build --release

# Node.js native addon
npm run build

# Generate TypeScript type definitions
npm run build  # includes --dts flag
```

### End-to-end local setup

The repository uses Rust 2024, Node.js, and `mise` for the development
toolchain. If `mise` is installed, the pinned tools can be installed with:

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

Models are downloaded from HuggingFace on first use. Set these variables when
the default home directories are not suitable:

```bash
export HF_HOME=/var/cache/dghs/huggingface
export IU_HOME=/var/cache/dghs/imgutils
```

The native addon does not bundle ONNX Runtime or vendor driver libraries.
Choose one compatible runtime/provider distribution before starting the
application.

### Provider build matrix

Cargo features compile provider bindings into the application; they do not
download or bundle the provider runtime:

| Target | Cargo features | Runtime/host prerequisite |
|---|---|---|
| CPU | `napi` or no provider feature | ONNX Runtime CPU shared library |
| Intel CPU/GPU/NPU | `openvino` | OpenVINO-enabled ONNX Runtime and Intel driver/runtime |
| NVIDIA GPU | `cuda` | CUDA-enabled ONNX Runtime, NVIDIA driver, CUDA, cuDNN |
| NVIDIA TensorRT | `cuda,tensorrt` | TensorRT plus the CUDA prerequisites |
| AMD Radeon GPU | `amd-gpu` | ROCm, MIGraphX, and MIGraphX-enabled ONNX Runtime |
| AMD XDNA/Vitis-AI | `amd-npu` | Target-specific Vitis-AI/XDNA runtime; experimental |

For a Node.js addon, the default `napi` feature is already enabled. A build
that contains every provider binding is possible, but the loaded runtime must
actually contain the corresponding providers:

```bash
cargo build --release --all-features
```

`--all-features` is a compile check, not proof that every provider is available
on the host.

The opt-in `all-providers` feature is equivalent when a caller needs to name
the complete provider set explicitly:

```bash
cargo build --release --features all-providers
```

The N-API build script selects `cuda,openvino` on Linux and `openvino` on
Windows automatically. It does not bundle a provider runtime or a hardware
driver; the same native addon can therefore use the compatible runtime that is
installed on the host.

### ONNX Runtime loading

The `ort` configuration in this project uses dynamic loading, so it does not
download or statically link a bundled ONNX Runtime. `ORT_DYLIB_PATH` must point
to a compatible `libonnxruntime.so`; execution-provider shared libraries and
their vendor dependencies must be discoverable by the dynamic loader.

The crate exposes CUDA, TensorRT, DirectML, OpenVINO, MIGraphX, and Vitis-AI
as optional Cargo features. At process startup, `ort` loads exactly one ONNX
Runtime shared library from `ORT_DYLIB_PATH`; choose a runtime distribution
that contains the provider you want to use.

`ORT_DYLIB_PATH` is a single path to the ONNX Runtime core library. It is not a
colon-separated list. One custom ONNX Runtime distribution may contain several
compatible providers, so one process can still use multiple EPs. Keep the
following files from the same distribution together:

```text
libonnxruntime.so
libonnxruntime_providers_*.so
libonnxruntime_providers_shared.so
vendor libraries (CUDA/cuDNN, ROCm/MIGraphX, OpenVINO, or TensorRT)
```

Example runtime environment:

```bash
export ORT_DYLIB_PATH=/opt/dghs/ort-profile/lib/libonnxruntime.so
export LD_LIBRARY_PATH=/opt/dghs/ort-profile/lib:${LD_LIBRARY_PATH:-}
```

On Linux, the library prepares common CUDA/cuDNN and OpenVINO core/TBB
dependencies before provider discovery. It searches the directory containing
`ORT_DYLIB_PATH`, the provider-path environment variables, and standard
runtime directories, so applications do not need handwritten `dlopen` or
preload code. OpenVINO GPU/NPU plugins and Level Zero/OpenCL driver libraries
are left to the provider's normal load boundary to avoid cross-version symbol
collisions. Set `DGHS_OPENVINO_LIBRARY_PATH` when an OpenVINO installation is
outside those locations. The ORT core, provider libraries, and vendor runtime
must still come from compatible distributions, and the Intel/NVIDIA kernel
driver remains a host prerequisite.

Do not mix the core library or provider libraries from different ORT, ROCm,
CUDA, OpenVINO, or TensorRT builds. See the ONNX Runtime [Execution Provider
overview](https://onnxruntime.ai/docs/execution-providers/), [provider build
guide](https://onnxruntime.ai/docs/build/eps.html), and the `ort` [dynamic
linking guide](https://github.com/pykeio/ort/blob/main/docs/content/setup/linking.mdx).

### Common worker startup

Each worker should use one provider feature set, one compatible runtime profile,
and its own cache directory. Set an explicit backend when the worker needs a
strict device contract; otherwise the library can use its automatic policy:

This repository provides the library/native addon, not a long-running RPC
worker or scheduler. The application that embeds it is responsible for
starting one process per runtime profile and routing requests to that process.

```bash
# CPU worker example
cargo build --release --features napi
export DGHS_BACKEND=cpu
export DGHS_PRECISION=fp32
export ORT_DYLIB_PATH=/opt/dghs/ort-cpu/lib/libonnxruntime.so
export IU_HOME=/var/cache/dghs/cpu
```

Before sending real jobs, print the capability matrix and create a strict test
session with a small ONNX model:

```bash
cargo run --no-default-features --features openvino --example ep_probe -- \
  path/to/model.onnx
```

Use the feature set matching the worker. The probe checks provider availability
in the loaded runtime; a successful provider probe alone does not prove that a
particular model compiles or runs on that provider.

With `DGHS_BACKEND=auto`, the library creates the requested model session for
each provider candidate before accepting it and uses CPU as the final fallback.
An explicit backend, or an explicit `DGHS_ORT_DEVICE` such as `GPU` or `NPU`,
is strict and returns the initialization/session error instead of silently
falling back.

### Environment variables

These variables are read when a session is created. A programmatic
`configureInference` call overrides them for the worker default, and the
`inferenceOptions` final argument overrides the selection for one N-API call.

| Variable | Example | Purpose |
|---|---|---|
| `DGHS_BACKEND` | `amd-gpu` | `auto`, `cpu`, `openvino`, `cuda`, `tensorrt`, `directml`, `amd-gpu`, or `amd-npu` |
| `DGHS_PRECISION` | `fp16` | `auto`, `fp32`, `fp16`, `bf16`, or `int8` |
| `DGHS_DEVICE_ID` | `0` | GPU/provider device ordinal |
| `DGHS_ORT_DEVICE` | `GPU` | OpenVINO device policy (`CPU`, `GPU`, `NPU`, `AUTO`, ...) |
| `DGHS_OPENVINO_LIBRARY_PATH` | `/opt/intel/openvino/runtime/lib/intel64` | Additional OpenVINO runtime search path on Linux |
| `DGHS_VITIS_CONFIG` | `/opt/vitis-ai/model.json` | Required Vitis-AI compiler/provider configuration |
| `DGHS_EP_CACHE_DIR` | `/var/cache/dghs/ep` | Vitis-AI provider compilation/cache directory (AMD NPU only; not used by MIGraphX) |
| `DGHS_MIGRAPHX_INT8_CALIBRATION_TABLE` | `/opt/models/calibration.table` | MIGraphX INT8 calibration table |
| `DGHS_MIGRAPHX_EXHAUSTIVE_TUNE` | `true` | Enable MIGraphX exhaustive tuning |

Use separate `DGHS_EP_CACHE_DIR` values for separate Vitis-AI workers and
runtime profiles. MIGraphX uses its provider-specific options instead. Cache
identities include the model hash, backend, precision, device, runtime, and
provider fingerprints.

### AMD Radeon GPU / XDNA2 NPU workers

AMD acceleration is an explicit worker contract. Build the provider into the
binary and use a compatible ORT profile. If MIGraphX and Vitis-AI are supplied
by different runtime builds, run GPU and NPU as separate processes/containers;
if one runtime profile contains both providers, per-call selection can remain
in one process:

```bash
# AMD GPU / MIGraphX worker
cargo build --release --no-default-features --features amd-gpu
export DGHS_BACKEND=amd-gpu
export DGHS_PRECISION=fp16       # fp32, fp16, or int8
export ORT_DYLIB_PATH=/opt/ort-migraphx/lib/libonnxruntime.so

# AMD XDNA2 / Vitis-AI worker (experimental)
cargo build --release --no-default-features --features amd-npu
export DGHS_BACKEND=amd-npu
export DGHS_PRECISION=bf16       # only for a model manifest that supports it
export DGHS_VITIS_CONFIG=/opt/vitis-ai/wd14.json
export DGHS_EP_CACHE_DIR=/var/cache/dghs-imgutils/vitis
export ORT_DYLIB_PATH=/opt/ort-vitis/lib/libonnxruntime.so
```

#### AMD Radeon GPU / MIGraphX prerequisites

Install a ROCm release supported by the target GPU and then install MIGraphX.
For Ubuntu, the MIGraphX package is available after ROCm is configured:

```bash
# Follow the ROCm installation guide for the exact OS/GPU first.
sudo apt update
sudo apt install -y migraphx

# Confirm that the kernel/ROCm stack sees the target architecture.
rocminfo | rg 'Name:.*gfx'
```

The Radeon 8060S target is expected to report `gfx1151`, but the output from
the installed driver is authoritative. When building ONNX Runtime yourself,
use its MIGraphX build path and point `--migraphx_home` at the installed
MIGraphX/ROCm prefix:

```bash
./build.sh --config Release --parallel --build_shared_lib \
  --use_migraphx --migraphx_home /opt/rocm
```

Use the matching prebuilt ORT/MIGraphX distribution when one is available;
the ONNX Runtime [MIGraphX guide](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
and [ROCm MIGraphX installation guide](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/install/install-migraphx.html)
define the supported version combinations. Set `LD_LIBRARY_PATH` to the
runtime and ROCm library directories before running `ep_probe`.

The application can select the worker programmatically before its first model
call. This takes precedence over `DGHS_*` variables:

```rust
use dghs_imgutils_rs::inference::{
    configure_inference, DeviceProvider, DeviceSelection, Precision, SessionOptions,
};

let options = SessionOptions::for_device(
    DeviceSelection::new(DeviceProvider::AmdGpu).with_device("0"),
)?
.with_precision(Precision::Fp16);
configure_inference(options)?;
```

The same control is available to TypeScript callers:

```typescript
import { configureInference, getInferenceConfiguration } from 'dghs-imgutils-rs';

configureInference({ provider: 'amd_gpu', precision: 'fp16' });
configureInference({ provider: 'cuda', device: '1' });
console.log(getInferenceConfiguration());
```

When automatic Intel GPU/NPU selection is used on Linux, call the Rust
`init_onnx_runtime()` function or `configureInference(...)` during
single-threaded application startup, before creating inference worker threads.
This lets the library prepare the OpenVINO dependencies and Intel OpenCL ICD
before ONNX Runtime loads its provider; an existing `OCL_ICD_VENDORS` or
`OCL_ICD_FILENAMES` policy is always preserved.

`provider` is one of `cpu`, `cuda`, `tensorrt`, `directml`, `intel_gpu`,
`intel_npu`, `amd_gpu`, `amd_npu`, or `openvino`. `device` is optional: CUDA,
TensorRT, AMD GPU, and DirectML accept an ordinal string such as `"1"`; Intel
providers accept `"0"`, `"GPU.0"`, or `"NPU.0"`. Omitting it delegates the
provider's default device selection to the runtime. The old `backend`,
`deviceId`, and `openvinoDeviceType` fields remain as compatibility aliases.

For UI-driven selection, model-backed N-API functions also accept an optional
`inferenceOptions` object as their final argument. This is call-local, so a UI
can choose a backend per request while existing calls remain source-compatible:

```typescript
import { getPixaiTags } from 'dghs-imgutils-rs';

const gpu = await getPixaiTags(imagePath, undefined, undefined, {
  provider: 'amd_gpu',
  precision: 'fp16',
  device: '0',
});

const cpu = await getPixaiTags(imagePath, undefined, undefined, {
  provider: 'cpu',
  precision: 'fp32',
});
```

The same final argument is available on the tagging, detection, segmentation,
OCR, pose, restoration/upscale, generic-model, metrics, validation, and NSFW
model APIs. Sessions are cached separately by model/backend/precision/device,
so switching back can reuse a warm session. A request already in progress keeps
the session it selected; changing the UI selection affects new calls only.

For Intel OpenVINO, use `DeviceProvider::IntelGpu` or
`DeviceProvider::IntelNpu` in Rust, and `provider: 'intel_gpu'` or
`provider: 'intel_npu'` in TypeScript. Add an optional `device: '0'` (or
`'GPU.0'`/`'NPU.0'`) when a specific device is required. The legacy
`.with_openvino_device_type("GPU")` and `openvinoDeviceType: 'GPU'` forms are
still supported.

Call `probeInferenceBackends({ provider: 'intel_npu' })` to inspect providers
using a programmatic device policy, or call it without arguments for the
current worker configuration.
Configuration is process-wide because ONNX Runtime and its provider libraries
are process-wide; `configureInference` is the startup default, while
`inferenceOptions` selects among providers available in that loaded runtime.
For multiple
backends in one Rust process, use `create_onnx_session_with_options` or
`get_or_create_session_with_options` directly.

Call-local selection cannot replace `ORT_DYLIB_PATH` after the process starts.
If GPU and NPU require different ONNX Runtime/provider distributions, keep the
GPU and NPU workers in separate processes or containers and let the UI/router
send each request to the appropriate worker.

`DGHS_BACKEND=amd-gpu`, `amd-npu`, and `cpu` are strict selections. Provider
initialization, model support, and compilation failures are returned as
`BackendUnavailable`, `ModelUnsupported`, or `CompilationFailed`; an explicit
AMD worker never silently falls back to CPU. The `auto` setting remains
available for compatibility and validates each provider candidate by creating
the requested model session before selecting it; CPU is the final fallback.

The `amd-gpu` feature enables the [ONNX Runtime MIGraphX provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html).
The [Radeon 8060S](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
target is `gfx1151`; it must be verified against the installed ROCm and
kernel stack at deployment time. MIGraphX `device_id`, FP16, INT8 calibration,
and `exhaustive_tune` are provider-specific options. BF16 is not advertised as
a generic MIGraphX capability by this crate, and persistent `.mxr` compiled
model files are intentionally not a v1 requirement. Sessions are cached in
process using the model SHA-256, backend, precision, device, runtime, and
provider fingerprints.

The `amd-npu` feature enables the [Vitis-AI provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html). Vitis requires an explicit
`DGHS_VITIS_CONFIG`; its cache key is derived from the model content and
runtime identity. Vitis may partition a graph between NPU and CPU, so provider
registration alone is not an NPU validation: record subgraph counts, CPU
fallback, and cold/warm latency with the probe and benchmark harness. This
worker is experimental until [`amdxdna`](https://github.com/amd/xdna-driver), XRT, `/dev/accel/accel0`, the kernel,
runtime, and the compiled model have been verified on the target machine.

#### AMD XDNA / Vitis-AI prerequisites

There is no generic Linux installer that makes every Ryzen AI/XDNA target
ready. Upstream ONNX Runtime documents Vitis-AI Linux support primarily for
AMD Adaptable SoCs; Ryzen AI software packages are target/OS-specific. Follow
the [Vitis-AI EP installation guide](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
and the target's XDNA/XRT documentation, then verify the device from the same
worker account:

```bash
test -e /dev/accel/accel0
ls -l /dev/accel/accel0
```

The model must be a Vitis-compatible deployment artifact. Use a BF16 or INT8
quantized ONNX model, normally with static shapes and an explicit compiler
configuration; an FP32 graph is not automatically converted to BF16 by this
crate. The first Vitis session can compile for minutes, so use a dedicated
cache directory and benchmark cold and warm starts. A successful provider
registration does not prove that the graph ran on the NPU because unsupported
subgraphs may execute on the CPU.

ONNX Runtime and provider shared libraries are not bundled by this crate and
must come from a compatible distribution. Do not mix the core library with
provider libraries from another runtime build. Ubuntu, Arch, and Proxmox LXC
support are deployment claims to validate in a runtime matrix, not guaranteed
by the Cargo feature alone.

Use `ModelManifest` for deployment-specific facts such as the model SHA-256,
opset, supported BF16/INT8 variants, static shapes, and compiler configuration.
The generic `Precision::Auto` setting does not convert an FP32 ONNX graph into
BF16 automatically.

For example:

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

Print the worker capability matrix and optionally create a strict session with:

```bash
cargo run --no-default-features --features amd-npu --example ep_probe -- \
  path/to/model.onnx
```

The build-only validation matrix is:

```bash
cargo check --no-default-features
cargo check --no-default-features --features amd-gpu,amd-npu
cargo check --no-default-features --features cuda,tensorrt,directml,openvino
cargo check --all-features
cargo check --no-default-features --examples
cargo test --lib --no-default-features
npm run build
```

These commands validate feature wiring and policy logic, not hardware
execution. A deployment report must additionally record the runtime build,
kernel/driver, `gfx1151` or XDNA device discovery, model hash, provider
compile result, subgraph/CPU-fallback evidence, and cold/warm latency.
The storage-directory test may fail when the test process has no writable
storage environment; keep that environment failure separate from AMD provider
acceptance.

### NVIDIA CUDA support

Use a CUDA-enabled ONNX Runtime distribution when running on an NVIDIA GPU. The OpenVINO distribution in the Intel section does not contain the CUDA Execution Provider.

```bash
# Verify the host driver before installing the user-space runtime.
nvidia-smi

# Build this crate with the provider bindings you intend to use.
cargo build --release --features cuda
# TensorRT uses the CUDA prerequisites too.
cargo build --release --features cuda,tensorrt

# Point to a CUDA-enabled ONNX Runtime distribution.
export ORT_DYLIB_PATH="/opt/onnxruntime-cuda/lib/libonnxruntime.so"
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/opt/cudnn
export TENSORRT_HOME=/opt/tensorrt       # only for TensorRT workers
export LD_LIBRARY_PATH="/opt/onnxruntime-cuda/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib:${TENSORRT_HOME}/lib:${LD_LIBRARY_PATH:-}"

# The provider is detected automatically when the loaded runtime contains it.
npm run build
```

The same runtime profile must provide `libonnxruntime_providers_cuda.so` and,
for TensorRT, the matching TensorRT provider library. CUDA, cuDNN, TensorRT,
the NVIDIA driver, and ONNX Runtime must match the version matrix. The
official [CUDA EP requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[TensorRT EP requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html),
and [ONNX Runtime install matrix](https://onnxruntime.ai/docs/install/) are the
source of truth. If building ORT from source, use the release-matched
`--use_cuda`, `--cuda_home`, `--use_tensorrt`, and `--tensorrt_home` options
from the official build guide; do not mix the resulting libraries with a
different distribution.

### Windows DirectML support

DirectML is available only on Windows. Use a DirectML-enabled ONNX Runtime
distribution, keep `onnxruntime.dll` and its provider DLLs together, and build
the binding with the `directml` feature:

```powershell
cargo build --release --features directml
$env:DGHS_BACKEND = "directml"
$env:DGHS_PRECISION = "fp32"
$env:ORT_DYLIB_PATH = "C:\opt\onnxruntime-directml\onnxruntime.dll"
$env:PATH = "C:\opt\onnxruntime-directml;" + $env:PATH
npm run build
```

DirectML device availability depends on the Windows graphics driver and the
loaded Runtime. Use `probeInferenceBackends()` or `ep_probe` before dispatching
jobs. For new Windows deployments, compare the [DirectML/WinML guidance](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
with the installed ONNX Runtime version.

### Intel NPU / GPU (XPU) support (Linux)

The OpenVINO setup below is Linux-specific. On Windows, use `onnxruntime.dll`, install the matching OpenVINO runtime DLLs separately, and add their directory to `PATH` before starting the application.

This project uses the `ort` dynamic loader with the OpenVINO Execution Provider. It does not call a Python API; the native shared libraries included in the official `onnxruntime-openvino` package are loaded directly by Rust/Node.js.

Extract the OpenVINO-enabled ONNX Runtime distribution into the project:

```bash
uv pip install --target .ort-runtime \
  --python-version 3.13 --only-binary=:all: --no-deps \
  onnxruntime-openvino==1.24.1

export ORT_DYLIB_PATH="$PWD/.ort-runtime/onnxruntime/capi/libonnxruntime.so.1.24.1"
```

For a system OpenVINO installation, the library can resolve the standard
runtime libraries itself. Source the vendor setup script only when it is needed
to configure the driver/ICD environment, or point the library at a custom
runtime directory:

```bash
source /opt/intel/openvino/setupvars.sh
export DGHS_OPENVINO_LIBRARY_PATH=/opt/intel/openvino/runtime/lib/intel64
cargo build --release --features openvino
```

Select the device when starting the application. `NPU` selects the Intel NPU and `GPU` selects the Intel XPU/GPU.

```bash
# Intel NPU
export DGHS_ORT_DEVICE=NPU

# Intel XPU/GPU. The library selects the Intel OpenCL ICD automatically when
# OCL_ICD_VENDORS is not already set.
export DGHS_ORT_DEVICE=GPU
# Optional override:
# export OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd
```

When unset, the library inspects the Intel devices exposed by the host and
builds an OpenVINO policy such as `AUTO:NPU,GPU,CPU`, `AUTO:GPU,CPU`, or
`AUTO:CPU`. The actual model session is then created for each automatic
candidate; CPU is the final fallback. Set `ORT_DYLIB_PATH` and the device
variables in the process that runs the application. GPU execution on Linux
requires the Intel GPU driver and an Intel ICD file. NPU execution also
requires access to the `/dev/accel/*` node (normally membership in the
`render` group). If
`/etc/OpenCL/vendors/intel.icd` (or the standard
`/usr/share/OpenCL/vendors/intel.icd`) exists, the library selects it before
OpenVINO initializes during the startup initialization described above. An
existing `OCL_ICD_VENDORS` or `OCL_ICD_FILENAMES` value is preserved.

For a strict provider selection, set the crate backend as well:

```bash
export DGHS_BACKEND=openvino
export DGHS_PRECISION=fp16
export DGHS_ORT_DEVICE=GPU   # CPU, GPU, NPU, AUTO, or another OpenVINO policy
```

The Intel GPU/NPU kernel driver, Level Zero/OpenCL runtime, and device plugin
must be installed separately. `ep_probe` must be run after the same driver and
OpenVINO environment has been loaded; a successful Rust build alone does not
validate Intel hardware support.

```bash
# Build the native addon
npm run build
```

Use the included minimal probe to verify session creation:

```bash
cargo run --no-default-features --features openvino --example intel_ep_probe -- --run \
  --provider intel_npu --device 0 .ort-runtime/onnxruntime/datasets/sigmoid.onnx
```

An explicit `provider` is strict: if that device cannot initialize or the model
cannot be committed, the library returns an error instead of hiding the failure
behind CPU fallback. The automatic policy is the mode that may fall back to
CPU. The probe with `--run` validates both session creation and a real
inference call; use `ep_probe` for a provider capability diagnostic.

The `onnxruntime-openvino` shared libraries come from the [official ONNX Runtime OpenVINO Execution Provider distribution](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html). Keep the files from the same `capi` directory together and do not mix them with a different OpenVINO installation.

## Documentation

```bash
# Generate and open Rust API documentation
cargo doc --open
```

## Python Reference

This project is a Rust port of the Python library [dghs-imgutils](https://github.com/deepghs/imgutils). The Python source code is included in the `imgutils/` directory as a read-only reference.

## License

Apache License 2.0
