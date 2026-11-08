"""Return OCR results grouped by input image; missing dependencies/inference fail."""
import json
import sys


def main():
    if len(sys.argv) < 2:
        raise ValueError("provide at least one image path")
    from imgutils.ocr import ocr

    results = []
    for path in sys.argv[1:]:
        items = ocr(path)
        results.append([
            {"bbox": [int(v) for v in bbox], "text": text, "score": float(score)}
            for bbox, text, score in items
        ])
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
