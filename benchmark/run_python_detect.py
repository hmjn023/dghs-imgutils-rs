import sys
import json
import traceback

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        return

    task_type = sys.argv[1]
    image_path = sys.argv[2]

    try:
        if task_type == 'face':
            from imgutils.detect import detect_faces
            res = detect_faces(image_path)
        elif task_type == 'head':
            from imgutils.detect import detect_heads
            res = detect_heads(image_path)
        elif task_type == 'person':
            from imgutils.detect import detect_person
            res = detect_person(image_path)
        elif task_type == 'censor':
            from imgutils.detect import detect_censors
            res = detect_censors(image_path)
        elif task_type == 'booru':
            from imgutils.detect import detect_with_booru_yolo
            res = detect_with_booru_yolo(image_path)
        elif task_type == 'nudenet':
            from imgutils.detect import detect_with_nudenet
            res = detect_with_nudenet(image_path)
        elif task_type == 'text':
            from imgutils.detect import detect_text
            res = detect_text(image_path)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        detections = []
        for box, label, score in res:
            detections.append({
                "box": list(box),
                "label": label,
                "score": float(score)
            })

        print(json.dumps({
            "success": True,
            "error": None,
            "detections": detections
        }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))

if __name__ == '__main__':
    main()
