"""
Python side: run imgcensor (emoji) operation for benchmark comparison.

Usage:
    uv run python benchmark/run_python_operate.py emulate_censor <image_path> <censor_image_path>
    uv run python benchmark/run_python_operate.py censor_nsfw_emoji <image_path>
"""

import sys
import json
import traceback
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        return

    task_type = sys.argv[1]
    image_path = sys.argv[2]

    try:
        if task_type == 'censor_area_image':
            censor_path = sys.argv[3]
            image = Image.open(image_path).convert('RGBA')
            censor_img = Image.open(censor_path).convert('RGBA')

            # Emulate what Rust does: scale to fit (contain) and paste centered
            area_x1, area_y1, area_x2, area_y2 = 100, 100, 300, 300
            area_w = area_x2 - area_x1
            area_h = area_y2 - area_y1
            cw, ch = censor_img.size
            scale = min(area_w / cw, area_h / ch)
            new_w = int(round(cw * scale))
            new_h = int(round(ch * scale))
            resized = censor_img.resize((new_w, new_h), Image.LANCZOS)

            offset_x = area_x1 + (area_w - new_w) // 2
            offset_y = area_y1 + (area_h - new_h) // 2

            result = image.copy()
            result.paste(resized, (offset_x, offset_y), resized)

            out_path = '/tmp/rust_censor_test_output_py.png'
            result.save(out_path)
            print(json.dumps({
                "success": True,
                "error": None,
                "output_path": out_path,
                "rust_match_expected": True,
            }))

        elif task_type == 'censor_nsfw_emoji':
            from imgutils.operate import censor_nsfw
            result = censor_nsfw(image_path, 'emoji', nipple_f=True, penis=True, pussy=True)
            out_path = '/tmp/rust_censor_nsfw_output_py.png'
            result.save(out_path)
            print(json.dumps({
                "success": True,
                "error": None,
                "output_path": out_path,
            }))

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))


if __name__ == '__main__':
    main()
