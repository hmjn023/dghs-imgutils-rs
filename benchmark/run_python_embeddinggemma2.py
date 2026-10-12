# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnxruntime>=1.24", "tokenizers>=0.22", "Pillow"]
# ///
"""Independent Pillow/NumPy reference for the published EmbeddingGemma 2 ONNX export.

Preprocessing follows Hugging Face's Gemma4ImageProcessor:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/image_processing_pil_gemma4.py
This compares language bindings and preprocessing, not ONNX versus PyTorch accuracy.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


def preprocess(path, budget):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    factor = math.sqrt(budget * 48**2 / (width * height))
    out_h = math.floor(factor * height / 48) * 48
    out_w = math.floor(factor * width / 48) * 48
    if out_h == 0:
        out_h, out_w = 48, min(width // height, budget) * 48
    elif out_w == 0:
        out_w, out_h = 48, min(height // width, budget) * 48
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255
    rows, cols = out_h // 16, out_w // 16
    patches = pixels.reshape(rows, 16, cols, 16, 3).transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(rows * cols, 768)
    positions = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), axis=-1).reshape(-1, 2)
    padded_pixels = np.zeros((1, budget * 9, 768), dtype=np.float32)
    padded_positions = np.full((1, budget * 9, 2), -1, dtype=np.int64)
    padded_pixels[0, :len(patches)] = patches
    padded_positions[0, :len(patches)] = positions
    return {"pixel_values": padded_pixels, "pixel_position_ids": padded_positions}, len(patches) // 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=70)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    model = ort.InferenceSession(str(args.model_dir / "onnx/model_quantized.onnx"), options,
                                providers=["CPUExecutionProvider"])
    vision = ort.InferenceSession(str(args.model_dir / "onnx/vision_encoder_quantized.onnx"), options,
                                 providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    def encode(text, features=None):
        ids = np.array([tokenizer.encode(text).ids], dtype=np.int64)
        empty = np.zeros((0, 512), dtype=np.float32)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids),
                 "image_features": empty if features is None else features,
                 "video_features": empty, "audio_features": empty}
        return model.run(["sentence_embedding"], feeds)[0][0].tolist()

    texts = ["task: search result | query: 赤い服を着た猫", "title: none | text: A cat wearing red clothes.",
             "task: sentence similarity | query: Unicode: café 🐈", ""]
    result = {"texts": texts, "text_embeddings": [encode(text) for text in texts], "image_embeddings": []}
    for path in args.images:
        inputs, count = preprocess(path, args.budget)
        features = vision.run(["image_features"], inputs)[0]
        assert features.shape == (count, 512)
        text = "<|image>" + "<|image|>" * count + "<image|>"
        result["image_embeddings"].append(encode(text, features))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
