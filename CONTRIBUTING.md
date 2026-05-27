# Contributing to dghs-imgutils-rs

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Building](#building)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Development Environment

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- [Node.js](https://nodejs.org/) (latest LTS)
- [mise](https://mise.jdx.dev/) (optional, for tool version management)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/dghs-imgutils-rs.git
cd dghs-imgutils-rs

# Install tool versions (if using mise)
mise install

# Build the Rust library
cargo build

# Run tests
cargo test
```

## Project Structure

```
src/
├── lib.rs          # Library root (public module declarations)
├── main.rs         # Binary entry point
├── config.rs       # Package metadata (version, title, etc.)
├── hub/            # HuggingFace Hub model download & caching
├── image/          # Image I/O, format conversion, utilities
├── inference/      # ONNX Runtime session management
├── generic/        # Generic ML engine (YOLO, CLIP, SigLIP, etc.)
├── tagging/        # Auto-tagging (WD14, DeepDanbooru, PixAI, etc.)
├── detect/         # Object detection (face, head, person, etc.)
├── segment/        # Segmentation & background removal
├── edge/           # Edge detection (Canny, Lineart)
├── metrics/        # Similarity metrics (CCIP, LPIPS, aesthetic)
├── validate/       # Image validation (truncated, monochrome, NSFW)
├── ocr/            # Text detection & recognition
├── pose/           # Pose estimation (DWpose)
├── metadata/       # Image metadata & steganography
├── sd/             # Stable Diffusion metadata parser
├── operate/        # Image operations (resize, trim, censor)
├── restore/        # Image restoration (NafNet, SCUNet)
├── upscale/        # Super-resolution (CDC)
├── ascii/          # ASCII art generation
├── resource/       # Background image management
├── utils/          # Common utilities (cache, storage)
└── napi/           # Node.js native bindings (napi-rs)
```

### Key Design Principles

1. **Rust core, optional bindings**: Core logic lives in pure Rust modules under `src/`. The `napi` feature flag enables Node.js bindings.
2. **Python reference**: The `imgutils/` directory contains the original Python implementation as a read-only reference.
3. **HuggingFace Hub**: Models are downloaded on-demand from HuggingFace Hub and cached locally.

## Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with napi-rs bindings
npm run build

# Build debug bindings
npm run build:debug
```

## Testing

```bash
# Run all Rust tests
cargo test

# Run tests for a specific module
cargo test tagging::

# Run with output
cargo test -- --nocapture
```

## Code Style

### Rust

- **Edition**: Rust 2024
- **Formatter**: Use `cargo fmt` before committing
- **Linter**: Run `cargo clippy -- -D warnings` to check for warnings
- **Documentation**: Add doc comments (`///`) to all public items
- **Safety**: Minimize `unsafe` code; document reasons when used

```bash
# Format code
cargo fmt

# Check lints
cargo clippy -- -D warnings
```

### Naming Conventions

- Modules: `snake_case` (e.g., `tagging`, `detect_faces`)
- Types: `PascalCase` (e.g., `TagResult`, `Detection`)
- Functions: `snake_case` (e.g., `get_pixai_tags`, `detect_faces`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_MODEL`)

### Documentation

All public APIs should have doc comments:

```rust
/// Detect faces in an anime image.
///
/// # Arguments
///
/// * `image` - Input image
/// * `level` - Model level ("s" or "n")
/// * `version` - Model version ("v0", "v1", "v1.3", "v1.4")
/// * `conf_threshold` - Confidence threshold (default: 0.25)
/// * `iou_threshold` - IoU threshold for NMS (default: 0.7)
///
/// # Returns
///
/// A vector of detected faces with bounding boxes and scores.
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

## Submitting Changes

1. **Fork** the repository
2. **Create a branch** for your feature or fix
3. **Write tests** for new functionality
4. **Run checks**:
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   ```
5. **Commit** with a clear message
6. **Open a Pull Request** with a description of your changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add WD14 tagging support
fix: handle RGBA images in face detection
docs: update API documentation for segment module
refactor: extract common YOLO prediction logic
```

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation if adding public APIs
- Ensure all CI checks pass

## Questions?

Feel free to open an issue for questions or discussions.
