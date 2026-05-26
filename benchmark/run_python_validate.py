import sys
import json
from imgutils.validate import (
    safe_check_score,
    anime_dbrating_score,
    anime_bangumi_char_score,
    anime_teen_score,
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_python_validate.py <img_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        result = {
            "success": True,
            "safe": dict(safe_check_score(image_path)),
            "dbrating": dict(anime_dbrating_score(image_path)),
            "bangumi_char": dict(anime_bangumi_char_score(image_path)),
            "teen": dict(anime_teen_score(image_path)),
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
        }

    print(json.dumps(result))

if __name__ == '__main__':
    main()
