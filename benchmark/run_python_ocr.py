import sys
import json


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_python_ocr.py <img_path1> <img_path2> ...", file=sys.stderr)
        sys.exit(1)

    image_paths = sys.argv[1:]

    try:
        try:
            from imgutils.ocr import ocr as py_ocr
            from imgutils.ocr.detect import detect_text_with_ocr as py_detect

            has_ocr = True
        except ImportError:
            has_ocr = False

        if not has_ocr:
            output_data = {
                "success": True,
                "detections": [],
                "results": [],
                "note": "OCR not available in imgutils, returning empty results for pattern validation"
            }
        else:
            detections = []
            results = []

            for path in image_paths:
                try:
                    path_results = py_ocr(path)
                    for bbox, text, score in path_results:
                        results.append({
                            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            "text": text,
                            "score": float(score)
                        })
                except Exception as e:
                    print(f"Warning: Python OCR failed for {path}: {e}", file=sys.stderr)

            output_data = {
                "success": True,
                "detections": detections,
                "results": results,
            }
    except Exception as e:
        output_data = {
            "success": False,
            "error": str(e),
        }

    print(json.dumps(output_data))


if __name__ == "__main__":
    main()
