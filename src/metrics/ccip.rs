//! CCIP (Character Contrastive Learning) の特徴抽出、距離測定、同一判定、クラスタリングの実装。

use crate::hub::hf_hub_download;
use crate::image::to_ndarray_chw;
use crate::inference::{create_onnx_session, run_onnx_session};
use crate::metrics::CcipError;

use image::DynamicImage;
use ndarray::{Array2, Array4};
use std::collections::HashMap;

const DEFAULT_REPO_ID: &str = "deepghs/ccip_onnx";
const DEFAULT_MODEL_NAME: &str = "ccip-caformer-24-randaug-pruned";

/// `metrics.json` のデコード用構造体
#[derive(serde::Deserialize)]
struct MetricsConfig {
    threshold: f32,
}

/// `cluster.json` のデコード用構造体
#[derive(serde::Deserialize)]
struct ClusterConfigItem {
    eps: f32,
    min_samples: usize,
}

/// 単一画像からキャラクターの 768 次元特徴ベクトル（Embedding）を抽出します。
///
/// # 引数
/// * `image` - 解析対象の画像オブジェクト
/// * `size` - 入力解像度（デフォルト `384`）
/// * `model` - モデル名（デフォルト `ccip-caformer-24-randaug-pruned`）
pub fn ccip_extract_feature(
    image: &DynamicImage,
    size: u32,
    model: &str,
) -> Result<Vec<f32>, CcipError> {
    let feats = ccip_batch_extract_features(std::slice::from_ref(image), size, model)?;
    Ok(feats[0].clone())
}

/// 複数画像から一括でキャラクターの特徴ベクトル（Embedding）を抽出します。
///
/// # 引数
/// * `images` - 画像オブジェクトの配列スライス
/// * `size` - 入力解像度（デフォルト `384`）
/// * `model` - モデル名（デフォルト `ccip-caformer-24-randaug-pruned`）
pub fn ccip_batch_extract_features(
    images: &[DynamicImage],
    size: u32,
    model: &str,
) -> Result<Vec<Vec<f32>>, CcipError> {
    if images.is_empty() {
        return Ok(Vec::new());
    }

    let model = if model.is_empty() {
        DEFAULT_MODEL_NAME
    } else {
        model
    };
    let filename = format!("{}/model_feat.onnx", model);
    let model_path = hf_hub_download(DEFAULT_REPO_ID, &filename, Some("model"), None)?;

    // 前処理用パラメータ (CCIP 共通)
    let mean = [0.48145466_f32, 0.4578275_f32, 0.40821073_f32];
    let std = [0.26862954_f32, 0.2613026_f32, 0.2757771_f32];

    let n = images.len();
    let mut batch_tensor = Array4::<f32>::zeros((n, 3, size as usize, size as usize));

    for (i, img) in images.iter().enumerate() {
        // アルファチャンネルがある場合、白背景にアルファブレンド
        let rgb_img = crate::image::force_image_background(img, [255, 255, 255]);
        // 指定サイズに歪みリサイズ
        let resized = rgb_img.resize_exact(size, size, image::imageops::FilterType::Triangle);
        // CHW テンソル取得
        let chw = to_ndarray_chw(&resized, &mean, &std)?;

        batch_tensor
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&chw.slice(ndarray::s![0, .., .., ..]));
    }

    let mut session = create_onnx_session(&model_path)?;
    let outputs = run_onnx_session(&mut session, "input", &batch_tensor)?;

    let output_value = outputs.get("output").ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No 'output' tensor found".to_string())
    })?;

    let (shape, prediction) = output_value.try_extract_tensor::<f32>()?;
    if shape.len() != 2 || shape[0] != n as i64 || shape[1] != 768 {
        return Err(CcipError::Inference(
            crate::inference::InferenceError::InvalidShape(format!(
                "Expected shape [{}, 768], got {:?}",
                n, shape
            )),
        ));
    }

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * 768;
        let end = start + 768;
        results.push(prediction[start..end].to_vec());
    }

    Ok(results)
}

/// 選択されたモデルのデフォルト同一判定しきい値をロードして返却します。
pub fn ccip_default_threshold(model: &str) -> Result<f32, CcipError> {
    let model = if model.is_empty() {
        DEFAULT_MODEL_NAME
    } else {
        model
    };
    let filename = format!("{}/metrics.json", model);
    let metrics_path = hf_hub_download(DEFAULT_REPO_ID, &filename, Some("model"), None)?;

    let file = std::fs::File::open(metrics_path)?;
    let config: MetricsConfig = serde_json::from_reader(file)?;
    Ok(config.threshold)
}

/// 2つのキャラクター画像または特徴ベクトル間の「距離（差異度）」を計算します。
/// 距離が小さいほど、同一キャラクターである可能性が高くなります。
pub fn ccip_difference(x: &[f32], y: &[f32], model: &str) -> Result<f32, CcipError> {
    let diffs = ccip_batch_differences(&[x.to_vec(), y.to_vec()], model)?;
    Ok(diffs[[0, 1]])
}

/// 2つのキャラクターが同一人物であるかをデフォルトのしきい値に基づいて判定します。
pub fn ccip_same(
    x: &[f32],
    y: &[f32],
    threshold: Option<f32>,
    model: &str,
) -> Result<bool, CcipError> {
    let diff = ccip_difference(x, y, model)?;
    let limit = match threshold {
        Some(t) => t,
        None => ccip_default_threshold(model)?,
    };
    Ok(diff <= limit)
}

/// 複数の特徴ベクトルのすべてのペアに対する距離マトリクス (`N x N`) を一括で算出します。
/// この処理には ONNX の `model_metrics.onnx` を用いた推論が使用されます。
pub fn ccip_batch_differences(
    embeddings: &[Vec<f32>],
    model: &str,
) -> Result<Array2<f32>, CcipError> {
    if embeddings.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    let model = if model.is_empty() {
        DEFAULT_MODEL_NAME
    } else {
        model
    };
    let filename = format!("{}/model_metrics.onnx", model);
    let model_path = hf_hub_download(DEFAULT_REPO_ID, &filename, Some("model"), None)?;

    let n = embeddings.len();
    let mut input_tensor = Array2::<f32>::zeros((n, 768));
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != 768 {
            return Err(CcipError::InvalidArgument(format!(
                "Embedding at index {} has invalid dimension: expected 768, got {}",
                i,
                emb.len()
            )));
        }
        for j in 0..768 {
            input_tensor[[i, j]] = emb[j];
        }
    }

    let mut session = create_onnx_session(&model_path)?;
    let input_value = ort::value::Tensor::from_array(input_tensor)?;
    let inputs = ort::inputs!["input" => input_value];
    let outputs = session.run(inputs)?;

    let output_value = outputs.get("output").ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No 'output' tensor found".to_string())
    })?;

    let (shape, prediction) = output_value.try_extract_tensor::<f32>()?;
    if shape.len() != 2 || shape[0] != n as i64 || shape[1] != n as i64 {
        return Err(CcipError::Inference(
            crate::inference::InferenceError::InvalidShape(format!(
                "Expected metric output shape [{}, {}], got {:?}",
                n, n, shape
            )),
        ));
    }

    let mut result_matrix = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result_matrix[[i, j]] = prediction[i * n + j];
        }
    }

    Ok(result_matrix)
}

/// 複数の画像特徴ベクトル間の同一キャラクター判定結果をマトリクス (`N x N`) で返します。
pub fn ccip_batch_same(
    embeddings: &[Vec<f32>],
    threshold: Option<f32>,
    model: &str,
) -> Result<Array2<bool>, CcipError> {
    let diffs = ccip_batch_differences(embeddings, model)?;
    let limit = match threshold {
        Some(t) => t,
        None => ccip_default_threshold(model)?,
    };

    let n = embeddings.len();
    let mut same_matrix = Array2::from_elem((n, n), false);
    for i in 0..n {
        for j in 0..n {
            same_matrix[[i, j]] = diffs[[i, j]] <= limit;
        }
    }

    Ok(same_matrix)
}

/// 指定モデルおよびクラスタリング手法に対するデフォルトパラメータ `(eps, min_samples)` を取得します。
pub fn ccip_default_clustering_params(
    model: &str,
    method: &str,
) -> Result<(f32, usize), CcipError> {
    let model = if model.is_empty() {
        DEFAULT_MODEL_NAME
    } else {
        model
    };

    if method == "dbscan" {
        let threshold = ccip_default_threshold(model)?;
        return Ok((threshold, 2));
    }
    if method == "optics" {
        return Ok((0.5, 5));
    }

    let filename = format!("{}/cluster.json", model);
    let cluster_path = hf_hub_download(DEFAULT_REPO_ID, &filename, Some("model"), None)?;
    let file = std::fs::File::open(cluster_path)?;
    let config: HashMap<String, ClusterConfigItem> = serde_json::from_reader(file)?;

    // optics_best 等のエイリアス解決
    let resolved_method = if method == "optics_best" {
        "optics"
    } else {
        method
    };

    let item = config.get(resolved_method).ok_or_else(|| {
        CcipError::InvalidArgument(format!("Unknown clustering method: {}", method))
    })?;

    Ok((item.eps, item.min_samples))
}

/// 複数の特徴ベクトルについて、自前実装の DBSCAN クラスタリングアルゴリズムを用いて
/// キャラクターごとにクラスタ分類を行います。
///
/// # 戻り値
/// クラスタIDのリスト（N要素）。ノイズ（未分類）点は `-1`、他の所属クラスタは `0, 1, 2...` となります。
pub fn ccip_clustering(
    embeddings: &[Vec<f32>],
    method: &str,
    eps: Option<f32>,
    min_samples: Option<usize>,
    model: &str,
) -> Result<Vec<i32>, CcipError> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let (default_eps, default_min_samples) = ccip_default_clustering_params(model, method)?;
    let eps = eps.unwrap_or(default_eps);
    let min_samples = min_samples.unwrap_or(default_min_samples);

    let n = embeddings.len();
    let batch_diff = ccip_batch_differences(embeddings, model)?;

    // 各点 i の近傍（距離が eps 以下のインデックス）を取得するクロージャ
    let get_neighbors =
        |i: usize| -> Vec<usize> { (0..n).filter(|&j| batch_diff[[i, j]] <= eps).collect() };

    // ラベル配列初期化: -2 = 未訪問, -1 = ノイズ/アウトライヤ, 0以上 = クラスタID
    let mut labels = vec![-2; n];
    let mut cluster_id = 0;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }

        let neighbors = get_neighbors(i);
        if neighbors.len() < min_samples {
            labels[i] = -1; // 一旦ノイズとしてマーク
            continue;
        }

        // 新しいクラスタを開始
        labels[i] = cluster_id;
        let mut seed_set = Vec::new();
        for &neighbor in &neighbors {
            if neighbor != i {
                seed_set.push(neighbor);
            }
        }

        // 近傍点の幅優先（BFS）探索
        let mut idx = 0;
        while idx < seed_set.len() {
            let curr = seed_set[idx];
            idx += 1;

            if labels[curr] == -1 {
                // 過去にノイズと判定されたが、コア点の近傍にあるため境界点として救済
                labels[curr] = cluster_id;
                continue;
            }
            if labels[curr] != -2 {
                continue; // 分類済み
            }

            labels[curr] = cluster_id;
            let curr_neighbors = get_neighbors(curr);
            if curr_neighbors.len() >= min_samples {
                // curr もコア点であるため、未訪問またはノイズである近傍をキューに合流
                for &nb in &curr_neighbors {
                    if (labels[nb] == -2 || labels[nb] == -1) && !seed_set.contains(&nb) {
                        seed_set.push(nb);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    Ok(labels)
}

/// 複数の特徴ベクトルを、長さ情報と角度を考慮して数学的に単一の特徴ベクトルにマージします。
/// 同一キャラクターの複数写真から「代表ベクトル」を作成する際に使用されます。
pub fn ccip_merge(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, CcipError> {
    if embeddings.is_empty() {
        return Err(CcipError::InvalidArgument(
            "Empty embeddings list".to_string(),
        ));
    }
    let dim = embeddings[0].len();
    let n = embeddings.len();

    // 1. 各ベクトルの長さを計算し、L2 正規化を行う
    let mut normalized_embs = vec![vec![0.0; dim]; n];
    let mut lengths = vec![0.0; n];
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != dim {
            return Err(CcipError::InvalidArgument(format!(
                "Embedding at index {} has invalid dimension: expected {}, got {}",
                i,
                dim,
                emb.len()
            )));
        }
        let len = (emb.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        lengths[i] = len;
        let active_len = if len == 0.0 { 1.0 } else { len };
        for j in 0..dim {
            normalized_embs[i][j] = emb[j] / active_len;
        }
    }

    // 2. 正規化ベクトルの平均ベクトルを算出
    let mut mean_emb = vec![0.0; dim];
    for j in 0..dim {
        let mut sum = 0.0;
        for row in normalized_embs.iter().take(n) {
            sum += row[j];
        }
        mean_emb[j] = sum / n as f32;
    }

    // 3. 平均ベクトルを再度 L2 正規化し、元のベクトルの長さの平均（lengths.mean()）を乗算
    let mean_len = (mean_emb.iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let active_mean_len = if mean_len == 0.0 { 1.0 } else { mean_len };

    let sum_lengths: f32 = lengths.iter().sum();
    let avg_length = sum_lengths / n as f32;

    let mut ret = vec![0.0; dim];
    for j in 0..dim {
        ret[j] = (mean_emb[j] / active_mean_len) * avg_length;
    }

    Ok(ret)
}
