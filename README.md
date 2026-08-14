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
dghs-imgutils-rs = "0.1.0"
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

### ONNX Runtime loading

The `ort` configuration in this project uses dynamic loading, so it does not download or statically link a bundled ONNX Runtime. `ORT_DYLIB_PATH` must point to a compatible `libonnxruntime.so`; the execution-provider shared libraries must be available beside it. See the Intel OpenVINO section below for the tested distribution.

The Cargo configuration keeps the CUDA Execution Provider enabled alongside OpenVINO. At process startup, `ort` loads exactly one ONNX Runtime shared library from `ORT_DYLIB_PATH`; choose a runtime distribution that contains the provider you want to use.

### NVIDIA CUDA support

Use a CUDA-enabled ONNX Runtime distribution when running on an NVIDIA GPU. The OpenVINO distribution in the Intel section does not contain the CUDA Execution Provider.

```bash
# Point to a CUDA-enabled ONNX Runtime distribution.
export ORT_DYLIB_PATH="/opt/onnxruntime-cuda/lib/libonnxruntime.so"
export LD_LIBRARY_PATH="/opt/onnxruntime-cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# The provider is detected automatically when the loaded runtime contains it.
npm run build
```

The same directory must provide `libonnxruntime_providers_cuda.so`, and compatible CUDA and cuDNN runtimes must be installed. The ONNX Runtime source build supports the `--use_cuda` option; do not mix the core library and provider libraries from different distributions. See the [CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

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

Select the device when starting the application. `NPU` selects the Intel NPU and `GPU` selects the Intel XPU/GPU.

```bash
# Intel NPU
export DGHS_ORT_DEVICE=NPU

# Intel XPU/GPU. Select the Intel OpenCL ICD on Linux.
export DGHS_ORT_DEVICE=GPU
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/
```

When unset, the library uses `AUTO:NPU,GPU,CPU` and tries NPU, then GPU, then CPU. Set `ORT_DYLIB_PATH` and the device variables in the process that runs the application. GPU execution on Linux requires the Intel GPU driver and the ICD files in `/etc/OpenCL/vendors/`.

```bash
# Build the native addon
npm run build
```

Use the included minimal probe to verify session creation:

```bash
cargo run --no-default-features --example intel_ep_probe -- \
  .ort-runtime/onnxruntime/datasets/sigmoid.onnx
```

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
