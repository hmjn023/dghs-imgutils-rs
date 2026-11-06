# Inference and precision tests

The default suite needs no model downloads:

```sh
cargo test --no-default-features --lib --tests
cargo clippy -- -D warnings
```

Model tests are explicitly ignored in that suite. Running an ignored test opts
into its prerequisites: a compatible ONNX Runtime, the requested backend,
network access or cached public Hugging Face assets, and any named fixtures.
Missing assets and inference errors must fail; they must not return early as a
successful test. Python parity tests additionally require `uv`, the `imgutils`
submodule and its dependencies.

## Model input regression tests

Run each model in its own process so a native crash cannot hide later results:

```sh
python scripts/test-model-contracts.py --backend cpu --image /path/to/image.jpg
python scripts/test-model-contracts.py --features openvino --backend openvino \
  --device GPU --image /path/to/image.jpg
# Optional positional case names select a subset:
python scripts/test-model-contracts.py --backend cpu deepdanbooru siglip_text
```

Set `ORT_DYLIB_PATH` when using a custom ONNX Runtime library. The runner builds
with default features disabled (no N-API), writes per-case logs and `results.json`
to `target/model-contract-results`, and returns nonzero for any panic, signal or
timeout. These are inference smoke tests, not numerical parity checks. Backend
selection follows the library's usual fallback policy; selecting GPU alone does
not certify that every node ran on GPU. A synthetic image is used if `--image`
is omitted; use an image containing text to exercise OCR recognition.

`generic_benchmark` and `restore_upscale_benchmark` measure real Rust inference
after warmup. They are not Python speed comparisons. CLIP uses the complete
`openai/clip-vit-base-patch32` model path. YOLO segmentation requires explicit
`DGHS_YOLOSEG_REPO` and `DGHS_YOLOSEG_MODEL` values for an accessible model.

## Independent filter reference

Generate both the input and expected output using Python/OpenCV, then compare
the Rust filter with matching, deterministic bilateral-filter parameters:

```sh
uv run --with numpy --with pillow --with opencv-python-headless \
  tests/precision/generate_baseline.py --output-dir /tmp/imgutils-baseline
DGHS_PRECISION_BASELINE_DIR=/tmp/imgutils-baseline \
  cargo test --no-default-features --test precision_test -- --ignored --nocapture
```

The reference is an actual filtered PNG, not the input image or a Rust-produced
baseline. The test requires PSNR above 35 dB and at least 3 dB improvement over
returning the unchanged input. Missing or identity references fail. This covers
bilateral filtering only; it does not establish guided-filter or model parity.

OpenCV and Rust use different border handling and bilateral kernels. The filter
comparison also checks the border and interior separately with matched final
rounding; passing means approximate agreement, not identical algorithms. OCR
comparisons check score finiteness only: Rust returns detection confidence,
whereas Python multiplies detection and recognition confidence.
