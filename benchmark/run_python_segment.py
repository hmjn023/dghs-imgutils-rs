import sys
import json
import traceback

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing image path"}))
        return

    image_path = sys.argv[1]

    try:
        from imgutils.segment import get_isnetis_mask
        mask = get_isnetis_mask(image_path)
        
        # マスク全体（例えば 1024x1024 等）をそのまま出力すると JSON が数MBから数十MBになり
        # 標準出力での転送やパースが極めて遅くなるため、32ピクセルごとのグリッドダウンサンプリングを行います。
        # これにより全体の形状が極めて高精度に保たれたまま、データ量を数KBに削減できます。
        grid_step = 32
        downsampled = mask[::grid_step, ::grid_step].tolist()

        # マスク全体の簡単な統計値も付加して堅牢性を補強
        stats = {
            "mean": float(mask.mean()),
            "min": float(mask.min()),
            "max": float(mask.max()),
            "shape": list(mask.shape)
        }

        print(json.dumps({
            "success": True,
            "error": None,
            "mask_grid": downsampled,
            "grid_step": grid_step,
            "stats": stats
        }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))

if __name__ == '__main__':
    main()
