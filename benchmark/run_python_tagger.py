import sys
import json
from imgutils.tagging.pixai import get_pixai_tags, _raw_predict

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_python_tagger.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # get_pixai_tags をデフォルト閾値で実行
        general, character = get_pixai_tags(image_path, model_name='v0.9')
        # _raw_predict を実行して予測確率配列 (prediction) を取得
        prediction = _raw_predict(image_path, model_name='v0.9')['prediction'].tolist()

        output_data = {
            "success": True,
            "general": general,
            "character": character,
            "prediction": prediction
        }
    except Exception as e:
        output_data = {
            "success": False,
            "error": str(e)
        }

    print(json.dumps(output_data))

if __name__ == '__main__':
    main()
