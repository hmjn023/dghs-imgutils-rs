use crate::metrics::ccip::{
    ccip_batch_same as core_ccip_batch_same, ccip_clustering as core_ccip_clustering,
    ccip_difference as core_ccip_difference, ccip_extract_feature as core_ccip_extract_feature,
    ccip_merge as core_ccip_merge, ccip_same as core_ccip_same,
};
use crate::tagging::pixai::get_pixai_tags as core_get_pixai_tags;
use image::DynamicImage;
use napi_derive::napi;
use std::collections::HashMap;

/// PixAI Tagger の予測結果を JS/TS 側で表現する構造体
#[napi(object)]
pub struct PixaiTagResult {
    /// 一般的なイラスト属性タグ（1girl, long_hair など）とそれぞれの確信度スコア
    pub general: HashMap<String, f64>,
    /// キャラクター名タグ（遠坂凛, ニェンなど）とそれぞれの確信度スコア
    pub character: HashMap<String, f64>,
}

/// 指定した画像ファイルパスから PixAI Tagger モデルを用いてアニメ調イラストのタグとその確信度スコアを予測します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: モデル名（デフォルトは `"v0.9"`）
/// * `thresholds`: 一般タグおよびキャラクタタグのしきい値マップ（オプション）
#[napi]
pub fn get_pixai_tags(
    path: String,
    model_name: Option<String>,
    thresholds: Option<HashMap<String, f64>>,
) -> napi::Result<PixaiTagResult> {
    // 1. パスから画像を開く (Rust側で透過PNG白ブレンド含む前処理が自動適用されます)
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let model = model_name.as_deref().unwrap_or("v0.9");

    let core_thresholds = thresholds.map(|map| {
        map.into_iter()
            .map(|(k, v)| (k, v as f32))
            .collect::<HashMap<String, f32>>()
    });

    // 2. コア予測の実行
    let result = core_get_pixai_tags(&image, model, core_thresholds).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("PixAI tagger prediction failed: {}", e),
        )
    })?;

    // 3. f32 -> f64 に変換して返す
    let general = result
        .general
        .into_iter()
        .map(|(k, v)| (k, v as f64))
        .collect();
    let character = result
        .character
        .into_iter()
        .map(|(k, v)| (k, v as f64))
        .collect();

    Ok(PixaiTagResult { general, character })
}

/// 指定した画像ファイルパスから CCIP モデルを用いてキャラクター対照学習特徴量（768次元ベクトル）を抽出します。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `model_name`: CCIPモデル名（オプション。省略時は `"ccip-caformer-24-randaug-pruned"`）
#[napi]
pub fn ccip_get_embedding(path: String, model_name: Option<String>) -> napi::Result<Vec<f64>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let model = model_name.as_deref().unwrap_or("");

    // CCIP の入力解像度はデフォルト 384
    let embedding = core_ccip_extract_feature(&image, 384, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("CCIP embedding extraction failed: {}", e),
        )
    })?;

    Ok(embedding.into_iter().map(|v| v as f64).collect())
}

/// 2つの CCIP 特徴ベクトルのペア距離（違い度）を算出します。
///
/// * `emb1`: 1つ目の768次元特徴ベクトル
/// * `emb2`: 2つ目の768次元特徴ベクトル
/// * `model_name`: メトリック評価に使用するモデル名（オプション。省略時は `"ccip-caformer-24-randaug-pruned"`）
#[napi]
pub fn ccip_distance(
    emb1: Vec<f64>,
    emb2: Vec<f64>,
    model_name: Option<String>,
) -> napi::Result<f64> {
    if emb1.len() != 768 || emb2.len() != 768 {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
                "Both embeddings must be exactly 768 dimensions. Got {} and {}.",
                emb1.len(),
                emb2.len()
            ),
        ));
    }

    let model = model_name.as_deref().unwrap_or("");
    let e1: Vec<f32> = emb1.into_iter().map(|v| v as f32).collect();
    let e2: Vec<f32> = emb2.into_iter().map(|v| v as f32).collect();

    let dist = core_ccip_difference(&e1, &e2, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("CCIP distance calculation failed: {}", e),
        )
    })?;

    Ok(dist as f64)
}

/// 2つの CCIP 特徴ベクトルを比較し、同一のキャラクターであるかどうかを判定します。
///
/// * `emb1`: 1つ目の768次元特徴ベクトル
/// * `emb2`: 2つ目の768次元特徴ベクトル
/// * `threshold`: 同一キャラクターと判定する最大距離のしきい値（オプション。省略時はモデルの最適デフォルト値がロードされます）
/// * `model_name`: 使用するモデル名（オプション。省略時は `"ccip-caformer-24-randaug-pruned"`）
#[napi]
pub fn ccip_same(
    emb1: Vec<f64>,
    emb2: Vec<f64>,
    threshold: Option<f64>,
    model_name: Option<String>,
) -> napi::Result<bool> {
    if emb1.len() != 768 || emb2.len() != 768 {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
                "Both embeddings must be exactly 768 dimensions. Got {} and {}.",
                emb1.len(),
                emb2.len()
            ),
        ));
    }

    let model = model_name.as_deref().unwrap_or("");
    let e1: Vec<f32> = emb1.into_iter().map(|v| v as f32).collect();
    let e2: Vec<f32> = emb2.into_iter().map(|v| v as f32).collect();
    let th = threshold.map(|v| v as f32);

    let is_same = core_ccip_same(&e1, &e2, th, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("CCIP same check failed: {}", e),
        )
    })?;

    Ok(is_same)
}

/// 複数の特徴ベクトルの一括同一判定（ペアごとの同一キャラクター判定結果マトリクス）を行います。
///
/// * `embeddings`: 比較する複数の768次元特徴ベクトルの配列
/// * `threshold`: しきい値（オプション）
/// * `model_name`: 使用するモデル名（オプション。省略時は `"ccip-caformer-24-randaug-pruned"`）
#[napi]
pub fn ccip_batch_same(
    embeddings: Vec<Vec<f64>>,
    threshold: Option<f64>,
    model_name: Option<String>,
) -> napi::Result<Vec<Vec<bool>>> {
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != 768 {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!(
                    "Embedding at index {} must be exactly 768 dimensions. Got {}.",
                    i,
                    emb.len()
                ),
            ));
        }
    }

    let model = model_name.as_deref().unwrap_or("");
    let core_embs: Vec<Vec<f32>> = embeddings
        .into_iter()
        .map(|emb| emb.into_iter().map(|v| v as f32).collect())
        .collect();
    let th = threshold.map(|v| v as f32);

    let matrix = core_ccip_batch_same(&core_embs, th, model).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("CCIP batch same check failed: {}", e),
        )
    })?;

    let mut result = Vec::with_capacity(matrix.nrows());
    for r in 0..matrix.nrows() {
        result.push(matrix.row(r).to_vec());
    }

    Ok(result)
}

/// 同一キャラクターの複数の特徴ベクトルを L2 重み付きでマージし、キャラクターを代表する一つの特徴ベクトルを生成します。
///
/// * `embeddings`: マージする複数の768次元特徴ベクトルの配列
#[napi]
pub fn ccip_merge(embeddings: Vec<Vec<f64>>) -> napi::Result<Vec<f64>> {
    if embeddings.is_empty() {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            "Cannot merge an empty array of embeddings".to_string(),
        ));
    }
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != 768 {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!(
                    "Embedding at index {} must be exactly 768 dimensions. Got {}.",
                    i,
                    emb.len()
                ),
            ));
        }
    }

    let core_embs: Vec<Vec<f32>> = embeddings
        .into_iter()
        .map(|emb| emb.into_iter().map(|v| v as f32).collect())
        .collect();

    let merged = core_ccip_merge(&core_embs).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("CCIP merge failed: {}", e),
        )
    })?;

    Ok(merged.into_iter().map(|v| v as f64).collect())
}

/// 複数の特徴ベクトルについて、自前で構築した高速 DBSCAN アルゴリズムを用いて同一キャラクターごとにクラスタリング（分類）します。
///
/// * `embeddings`: クラスタリング対象の特徴ベクトル配列
/// * `eps`: 近傍範囲の半径（オプション。省略時はモデルの最適デフォルト値がロードされます）
/// * `min_samples`: クラスタを形成するために必要な最小データ数（オプション）
/// * `model_name`: 使用するモデル名（オプション。省略時は `"ccip-caformer-24-randaug-pruned"`）
#[napi]
pub fn ccip_cluster(
    embeddings: Vec<Vec<f64>>,
    eps: Option<f64>,
    min_samples: Option<i32>,
    model_name: Option<String>,
) -> napi::Result<Vec<i32>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != 768 {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!(
                    "Embedding at index {} must be exactly 768 dimensions. Got {}.",
                    i,
                    emb.len()
                ),
            ));
        }
    }

    let model = model_name.as_deref().unwrap_or("");
    let core_embs: Vec<Vec<f32>> = embeddings
        .into_iter()
        .map(|emb| emb.into_iter().map(|v| v as f32).collect())
        .collect();
    let core_eps = eps.map(|v| v as f32);
    let core_min_samples = min_samples.map(|v| v.max(0) as usize);

    let clusters = core_ccip_clustering(&core_embs, "dbscan", core_eps, core_min_samples, model)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("CCIP clustering failed: {}", e),
            )
        })?;

    Ok(clusters.into_iter().map(|v| v as i32).collect())
}

/// 指定したローカル画像ファイルを読み込み、RGBA 画像の場合は白背景をブレンドして RGB 画像として返します。
/// 戻り値は RGBA (R, G, B, A) のピクセル配列です。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
/// * `force_background`: 背景色の RGB 値（例: `[255, 255, 255]`）。省略時は白背景。
#[napi]
pub fn load_image_from_file(
    path: String,
    force_background: Option<Vec<f64>>,
) -> napi::Result<Vec<u8>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {e}"),
        )
    })?;

    let bg = force_background.map(|v| {
        if v.len() != 3 {
            [255, 255, 255]
        } else {
            [v[0] as u8, v[1] as u8, v[2] as u8]
        }
    });

    let img = if let Some(bg) = bg {
        if img.color().has_alpha() {
            crate::image::force_image_background(&img, bg)
        } else {
            img
        }
    } else {
        img
    };

    let rgba = img.to_rgba8();
    Ok(rgba.into_raw())
}

/// 指定した URL から画像をダウンロードして読み込みます。
#[napi]
pub fn load_image_from_url(url: String) -> napi::Result<Vec<u8>> {
    use crate::image::load::{ImageSource, load_image};
    let img = load_image(&ImageSource::HttpUrl(&url)).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to load from URL: {e}"),
        )
    })?;
    let rgba = img.to_rgba8();
    Ok(rgba.into_raw())
}

/// RGB エンコード: 画像をテンソル（CHW 順序、float32）に変換します。
/// 戻り値はフラットな f64 配列です（形状は [C, H, W]）。
#[napi]
pub fn rgb_encode(
    path: String,
    order: Option<String>,
    use_float: Option<bool>,
) -> napi::Result<Vec<f64>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {e}"),
        )
    })?;
    let order = order.unwrap_or_else(|| "CHW".to_string());
    let use_float = use_float.unwrap_or(true);

    let encoded = crate::image::rgb_encode(&img, &order, use_float).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("RGB encode failed: {e}"),
        )
    })?;

    Ok(encoded.iter().map(|&v| v as f64).collect())
}

/// RGB デコード: テンソル（CHW 順序、float32）を画像に変換し、ファイルに保存します。
#[napi]
pub fn rgb_decode(
    tensor: Vec<f64>,
    shape_c: u32,
    shape_h: u32,
    shape_w: u32,
    order: Option<String>,
    output_path: String,
) -> napi::Result<()> {
    let order = order.unwrap_or_else(|| "CHW".to_string());
    let arr = ndarray::Array3::from_shape_vec(
        (shape_c as usize, shape_h as usize, shape_w as usize),
        tensor.into_iter().map(|v| v as f32).collect(),
    )
    .map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Invalid tensor shape: {e}"),
        )
    })?;

    let img = crate::image::rgb_decode(&arr.view(), &order).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("RGB decode failed: {e}"),
        )
    })?;

    img.save(&output_path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to save image: {e}"),
        )
    })?;

    Ok(())
}

/// 複数の画像レイヤーを合成します。
/// `layers` は画像パスの配列、`alphas` は各レイヤーの透明度の配列です。
/// 戻り値は RGBA ピクセル配列です。
#[napi]
pub fn istack(
    layers: Vec<String>,
    alphas: Vec<f64>,
    output_width: Option<u32>,
    output_height: Option<u32>,
) -> napi::Result<Vec<u8>> {
    if layers.len() != alphas.len() {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            "layers and alphas must have the same length".to_string(),
        ));
    }

    let items: Vec<crate::image::LayerItemBuilder> = layers
        .iter()
        .zip(alphas.iter())
        .map(|(path, alpha)| {
            let img = image::open(path)
                .unwrap_or_else(|_| DynamicImage::ImageRgba8(image::RgbaImage::new(1, 1)));
            crate::image::LayerItemBuilder::from_image(img).with_alpha(*alpha as f32)
        })
        .collect();

    let size = output_width.zip(output_height);

    let result = crate::image::istack(items, size).map_err(|e| {
        napi::Error::new(napi::Status::GenericFailure, format!("istack failed: {e}"))
    })?;

    Ok(result.into_raw())
}

/// ストレージディレクトリのパスを取得します。
#[napi]
pub fn get_storage_dir() -> String {
    crate::utils::storage::get_storage_dir()
        .to_string_lossy()
        .to_string()
}

/// パッケージのバージョンを取得します。
#[napi]
pub fn get_version() -> String {
    crate::config::VERSION.to_string()
}

/// パッケージのタイトルを取得します。
#[napi]
pub fn get_title() -> String {
    crate::config::TITLE.to_string()
}

pub mod detect;
pub mod edge;
pub mod generic;
pub mod image_utils;
pub mod metadata_sd;
pub mod metrics;
pub mod ocr;
pub mod operate;
pub mod pose;
pub mod resource_ascii;
pub mod restore_upscale;
pub mod segment;
pub mod tagging;
pub mod utils;
pub mod validate;
