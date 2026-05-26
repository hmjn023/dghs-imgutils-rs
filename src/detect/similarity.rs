//! 境界ボックス (BBox) およびセグメンテーションマスクの類似度 (IoU) 計算ユーティリティ。
//! Kuhn-Munkres（ハンガリアンアルゴリズム）を用いてペアリングを最適化します。

use crate::detect::base::{BBox, Detection};

/// 二つの境界ボックスの Intersection over Union (IoU) を計算します。
pub fn calculate_iou(box1: &BBox, box2: &BBox) -> f32 {
    let x1 = box1.0.max(box2.0) as f32;
    let y1 = box1.1.max(box2.1) as f32;
    let x2 = box1.2.min(box2.2) as f32;
    let y2 = box1.3.min(box2.3) as f32;

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = ((box1.2 - box1.0) as f32) * ((box1.3 - box1.1) as f32);
    let area2 = ((box2.2 - box2.0) as f32) * ((box2.3 - box2.1) as f32);

    intersection / (area1 + area2 - intersection + 1e-6)
}

/// 二つの境界ボックス配列の間の類似度リストを計算します。
/// Kuhn-Munkres アルゴリズムによる最適ペアリングを行います。
pub fn bboxes_similarity(
    bboxes1: &[BBox],
    bboxes2: &[BBox],
) -> Vec<f32> {
    let m = bboxes1.len();
    let n = bboxes2.len();

    if m == 0 && n == 0 {
        return Vec::new();
    }

    // pathfinding::kuhn_munkres は行数 <= 列数を要求するため、必要に応じて転置する
    let is_transposed = m > n;
    let matrix_to_solve = if is_transposed {
        pathfinding::matrix::Matrix::from_fn(n, m, |(j, i)| {
            let iou = calculate_iou(&bboxes1[i], &bboxes2[j]);
            (iou * 1000000.0) as i32
        })
    } else {
        pathfinding::matrix::Matrix::from_fn(m, n, |(i, j)| {
            let iou = calculate_iou(&bboxes1[i], &bboxes2[j]);
            (iou * 1000000.0) as i32
        })
    };

    // Kuhn-Munkres（最大重みマッチング）を実行
    let (_total_weight, assignments) = pathfinding::prelude::kuhn_munkres(&matrix_to_solve);

    let mut matched_similarities = Vec::new();
    if is_transposed {
        // matrix_to_solve の行は bboxes2 (長さ n)
        // assignments は bboxes1 のインデックス配列
        for (j, &i) in assignments.iter().enumerate() {
            let iou = calculate_iou(&bboxes1[i], &bboxes2[j]);
            matched_similarities.push(iou);
        }
    } else {
        // matrix_to_solve の行は bboxes1 (長さ m)
        // assignments は bboxes2 のインデックス配列
        for (i, &j) in assignments.iter().enumerate() {
            let iou = calculate_iou(&bboxes1[i], &bboxes2[j]);
            matched_similarities.push(iou);
        }
    }

    // Python の np.zeros(max_len) での初期化と similarities の配置を再現
    let max_len = m.max(n);
    let mut padded_similarities = vec![0.0f32; max_len];
    for (i, &sim) in matched_similarities.iter().enumerate() {
        if i < max_len {
            padded_similarities[i] = sim;
        }
    }

    padded_similarities
}

/// 検出結果リストの間の類似度リストを計算します（クラスラベルごとにマッチングを制限）。
pub fn detection_similarity(
    detect1: &[Detection],
    detect2: &[Detection],
) -> Vec<f32> {
    use std::collections::HashSet;

    // 両方の検出結果からユニークなラベルを収集
    let mut labels = HashSet::new();
    for d in detect1 {
        labels.insert(d.label.clone());
    }
    for d in detect2 {
        labels.insert(d.label.clone());
    }

    let mut sorted_labels: Vec<String> = labels.into_iter().collect();
    sorted_labels.sort();

    let mut sims = Vec::new();
    for current_label in sorted_labels {
        let bboxes1: Vec<BBox> = detect1
            .iter()
            .filter(|d| d.label == current_label)
            .map(|d| d.bbox)
            .collect();
        let bboxes2: Vec<BBox> = detect2
            .iter()
            .filter(|d| d.label == current_label)
            .map(|d| d.bbox)
            .collect();

        sims.extend(bboxes_similarity(&bboxes1, &bboxes2));
    }

    sims
}

/// 二つのマスクの Intersection over Union (IoU) を計算します。
pub fn calculate_mask_iou(
    mask1: &ndarray::Array2<f32>,
    mask2: &ndarray::Array2<f32>,
    threshold: f32,
) -> f32 {
    let mut intersection = 0.0f32;
    let mut union = 0.0f32;

    let h = mask1.shape()[0];
    let w = mask1.shape()[1];

    for y in 0..h {
        for x in 0..w {
            let m1_val = mask1[[y, x]];
            let m2_val = mask2[[y, x]];
            let m1_bool = m1_val >= threshold;
            let m2_bool = m2_val >= threshold;

            if m1_bool && m2_bool {
                intersection += 1.0;
            }
            if m1_bool || m2_bool {
                union += 1.0;
            }
        }
    }

    intersection / (union + 1e-6)
}

/// 二つのマスク配列の間の類似度リストを計算します。
pub fn masks_similarity(
    masks1: &[ndarray::Array2<f32>],
    masks2: &[ndarray::Array2<f32>],
    threshold: f32,
) -> Vec<f32> {
    let m = masks1.len();
    let n = masks2.len();

    if m == 0 && n == 0 {
        return Vec::new();
    }

    let is_transposed = m > n;
    let matrix_to_solve = if is_transposed {
        pathfinding::matrix::Matrix::from_fn(n, m, |(j, i)| {
            let iou = calculate_mask_iou(&masks1[i], &masks2[j], threshold);
            (iou * 1000000.0) as i32
        })
    } else {
        pathfinding::matrix::Matrix::from_fn(m, n, |(i, j)| {
            let iou = calculate_mask_iou(&masks1[i], &masks2[j], threshold);
            (iou * 1000000.0) as i32
        })
    };

    let (_total_weight, assignments) = pathfinding::prelude::kuhn_munkres(&matrix_to_solve);

    let mut matched_similarities = Vec::new();
    if is_transposed {
        for (j, &i) in assignments.iter().enumerate() {
            let iou = calculate_mask_iou(&masks1[i], &masks2[j], threshold);
            matched_similarities.push(iou);
        }
    } else {
        for (i, &j) in assignments.iter().enumerate() {
            let iou = calculate_mask_iou(&masks1[i], &masks2[j], threshold);
            matched_similarities.push(iou);
        }
    }

    let max_len = m.max(n);
    let mut padded_similarities = vec![0.0f32; max_len];
    for (i, &sim) in matched_similarities.iter().enumerate() {
        if i < max_len {
            padded_similarities[i] = sim;
        }
    }

    padded_similarities
}

/// インスタンスセグメンテーション結果（マスク付き）の類似度リストを計算します。
pub fn detection_with_mask_similarity(
    detect1: &[Detection],
    detect2: &[Detection],
    threshold: f32,
) -> Vec<f32> {
    use std::collections::HashSet;

    let mut labels = HashSet::new();
    for d in detect1 {
        labels.insert(d.label.clone());
    }
    for d in detect2 {
        labels.insert(d.label.clone());
    }

    let mut sorted_labels: Vec<String> = labels.into_iter().collect();
    sorted_labels.sort();

    let mut sims = Vec::new();
    for current_label in sorted_labels {
        let masks1: Vec<ndarray::Array2<f32>> = detect1
            .iter()
            .filter(|d| d.label == current_label && d.mask.is_some())
            .map(|d| d.mask.clone().unwrap())
            .collect();
        let masks2: Vec<ndarray::Array2<f32>> = detect2
            .iter()
            .filter(|d| d.label == current_label && d.mask.is_some())
            .map(|d| d.mask.clone().unwrap())
            .collect();

        sims.extend(masks_similarity(&masks1, &masks2, threshold));
    }

    sims
}
