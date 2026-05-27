"""
精度検証用の基準データを生成するスクリプト。

使用方法:
    python generate_baseline.py

生成されるファイル:
    - baseline_input.npy: テスト入力画像 (RGB, float32, 0-255)
    - baseline_adversarial.npy: adversarial noise removal 結果
"""

import numpy as np
from PIL import Image

def generate_test_image(width: int = 64, height: int = 64) -> np.ndarray:
    """テスト用の画像を生成する。"""
    # グラデーションパターン
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            img[y, x, 0] = (x / width) * 255.0  # R
            img[y, x, 1] = (y / height) * 255.0  # G
            img[y, x, 2] = ((x + y) / (width + height)) * 255.0  # B
    return img


def add_noise(img: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
    """ノイズを追加する。"""
    noise = np.random.randn(*img.shape).astype(np.float32) * noise_level
    return np.clip(img + noise, 0, 255).astype(np.float32)


def main():
    np.random.seed(42)  # 再現性のため

    # テスト画像生成
    img = generate_test_image(64, 64)
    noisy_img = add_noise(img, noise_level=10.0)

    # 保存
    np.save("baseline_input.npy", noisy_img)
    print(f"Saved baseline_input.npy: shape={noisy_img.shape}, dtype={noisy_img.dtype}")

    # adversarial noise removal は Rust で実行するため、ここでは入力のみ生成
    print("Run Rust tests to generate adversarial output for comparison.")


if __name__ == "__main__":
    main()
