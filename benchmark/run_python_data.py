import sys
import json
import numpy as np
from PIL import Image
from imgutils.data import load_image, rgb_encode, istack


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        img = load_image(image_path, mode='RGB', force_background='white')
        w, h = img.size

        encoded = rgb_encode(img, order_='CHW', use_float=True)
        shape = list(encoded.shape)
        values = encoded.flatten()[:100].tolist()
        values = [[v] for v in values]

        output = {
            "success": True,
            "type": str(type(img).__name__),
            "width": w,
            "height": h,
            "channels": len(img.getbands()),
            "shape": shape,
            "values": values,
        }
    except Exception as e:
        output = {"success": False, "error": str(e)}

    print(json.dumps(output))


if __name__ == '__main__':
    main()
