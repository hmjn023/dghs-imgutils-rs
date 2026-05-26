import sys
import json
import numpy as np
from imgutils.metrics import (
    lpips_extract_feature,
    lpips_difference,
    lpips_clustering,
    anime_dbaesthetic,
    laplacian_score,
    psnr,
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_python_metrics.py <img_path1> <img_path2> ...", file=sys.stderr)
        sys.exit(1)

    image_paths = sys.argv[1:]

    try:
        # LPIPS features
        lpips_feats = [lpips_extract_feature(p) for p in image_paths]
        lpips_feats_flat = []
        for feat_tuple in lpips_feats:
            flat = np.concatenate([f.flatten() for f in feat_tuple])
            lpips_feats_flat.append(flat.tolist())

        # LPIPS pairwise differences
        n = len(image_paths)
        lpips_diffs = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = lpips_difference(lpips_feats[i], lpips_feats[j])
                lpips_diffs[i][j] = d
                lpips_diffs[j][i] = d

        # LPIPS clustering
        lpips_clusters = lpips_clustering(image_paths)

        # Aesthetic score (first image)
        aesthetic = anime_dbaesthetic(image_paths[0], fmt=('score',))
        aesthetic_score = aesthetic[0] if isinstance(aesthetic, tuple) else 0.0

        # Laplacian score (first image)
        lap = laplacian_score(image_paths[0])

        # PSNR (between first two images)
        psnr_val = psnr(image_paths[0], image_paths[1]) if len(image_paths) >= 2 else 0.0

        output_data = {
            "success": True,
            "lpips": {
                "features": lpips_feats_flat,
                "differences": lpips_diffs,
                "cluster_labels": lpips_clusters,
            },
            "aesthetic": {
                "score": aesthetic_score,
            },
            "laplacian": {
                "score": lap,
            },
            "psnr": {
                "score": psnr_val,
            },
        }
    except Exception as e:
        import traceback
        output_data = {
            "success": False,
            "error": str(e) + "\n" + traceback.format_exc(),
        }

    print(json.dumps(output_data))

if __name__ == '__main__':
    main()
