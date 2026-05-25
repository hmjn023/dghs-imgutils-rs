import sys
import json
import numpy as np
from imgutils.metrics import (
    ccip_batch_extract_features,
    ccip_batch_differences,
    ccip_batch_same,
    ccip_merge,
    ccip_clustering,
    ccip_default_threshold
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_python_ccip.py <img_path1> <img_path2> ...", file=sys.stderr)
        sys.exit(1)

    image_paths = sys.argv[1:]

    try:
        # 1. 各画像の特徴ベクトル (embeddings) 抽出
        feats = ccip_batch_extract_features(image_paths)
        
        # 2. Pairwise 距離マトリクス計算
        differences = ccip_batch_differences(image_paths)
        
        # 3. 同一キャラ判定マトリクス計算
        same_matrix = ccip_batch_same(image_paths)
        
        # 4. 特徴ベクトルのマージ (全画像)
        merged_feat = ccip_merge(image_paths)
        
        # 5. DBSCAN クラスタリング実行 (min_samples=2)
        # ※サンプル数が少ないため、デフォルトの optics ではなく dbscan で min_samples=2 で動作させます
        cluster_labels = ccip_clustering(image_paths, method='dbscan', min_samples=2)
        
        # 6. デフォルト閾値の取得
        threshold = ccip_default_threshold()

        output_data = {
            "success": True,
            "embeddings": feats.tolist(),
            "differences": differences.tolist(),
            "same": same_matrix.tolist(),
            "merged": merged_feat.tolist(),
            "cluster_labels": cluster_labels,
            "threshold": threshold
        }
    except Exception as e:
        output_data = {
            "success": False,
            "error": str(e)
        }

    print(json.dumps(output_data))

if __name__ == '__main__':
    main()
