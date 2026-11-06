"""Generate deterministic OpenCV reference images for Rust's bilateral filter.

Run with: uv run --with numpy --with pillow --with opencv-python-headless \
    tests/precision/generate_baseline.py --output-dir /tmp/imgutils-baseline

This uses the same OpenCV bilateralFilter operation as imgutils.restore, with
fixed parameters (two iterations, diameter 5, sigmaColor 20, sigmaSpace 8).
PNG preserves the exact uint8 input both implementations receive.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:64, :64]
    image = np.stack([x * 255 / 64, y * 255 / 64, (x + y) * 255 / 128], axis=-1)
    image = np.clip(image + np.random.default_rng(42).normal(0, 10, image.shape), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(args.output_dir / "baseline_input.png")
    output = image.astype(np.float32)
    for _ in range(2):
        output = cv2.bilateralFilter(output, 5, 20, 8)
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(args.output_dir / "baseline_adversarial.png")
    print(f"Wrote input and Python reference to {args.output_dir}")


if __name__ == "__main__":
    main()
