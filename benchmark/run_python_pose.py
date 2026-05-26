import sys
import json
import traceback

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        return

    image_path = sys.argv[1]

    try:
        from imgutils.pose import dwpose_estimate, op18_visualize

        keypoints_list = dwpose_estimate(image_path)

        results = []
        for kps in keypoints_list:
            keypoints = []
            for kp in kps.all:
                keypoints.append({
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "score": float(kp[2]),
                })
            results.append({
                "num_keypoints": len(keypoints),
                "keypoints": keypoints,
            })

        print(json.dumps({
            "success": True,
            "error": None,
            "num_persons": len(results),
            "results": results,
            "num_keypoints": len(results[0]["keypoints"]) if results else 0,
        }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))

if __name__ == '__main__':
    main()
