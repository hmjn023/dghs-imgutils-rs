//! napi-rs バインディング: resource (背景画像) + ascii (アスキーアート) モジュール。

use napi_derive::napi;
use std::path::PathBuf;

/// 利用可能な背景画像のファイル名一覧を取得します。
///
/// 内部で `deepghs/anime-bg` データセットの `images.csv` を HuggingFace Hub から
/// ダウンロードし、全ての画像ファイル名を配列として返します。
#[napi]
pub fn list_bg_images() -> napi::Result<Vec<String>> {
    let images = crate::resource::list_bg_images();
    if images.is_empty() {
        return Err(napi::Error::new(
            napi::Status::GenericFailure,
            "Failed to list background images from HuggingFace dataset".to_string(),
        ));
    }
    Ok(images)
}

/// 指定したファイル名の背景画像を読み込み、一時ファイルに保存し、そのパスを返します。
///
/// 画像は `~/.cache/dghs-imgutils/bg/` にキャッシュされます。
/// キャッシュに既に存在する場合はダウンロードせずにそのパスを返します。
///
/// * `filename` - 画像ファイル名（例: `"000000.jpg"`）
#[napi]
pub fn get_bg_image(filename: String) -> napi::Result<String> {
    let cache_dir = crate::utils::storage::get_storage_dir().join("bg");
    std::fs::create_dir_all(&cache_dir).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to create cache directory: {e}"),
        )
    })?;

    let cached_path = cache_dir.join(&filename);
    if cached_path.exists() {
        return Ok(cached_path.to_string_lossy().to_string());
    }

    let _img = crate::resource::get_bg_image(&filename).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to load background image '{filename}': {e}"),
        )
    })?;

    // After get_bg_image, the file should exist in cache
    if cached_path.exists() {
        Ok(cached_path.to_string_lossy().to_string())
    } else {
        Err(napi::Error::new(
            napi::Status::GenericFailure,
            format!("Background image '{filename}' was not cached properly"),
        ))
    }
}

/// 利用可能な背景画像からランダムに1枚を選び、そのファイルパスを返します。
///
/// * `output_dir` - 画像を保存するディレクトリパス（省略時はシステムの一時ディレクトリ）
#[napi]
pub fn random_bg_image(output_dir: Option<String>) -> napi::Result<String> {
    let img = crate::resource::random_bg_image().map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to load random background image: {e}"),
        )
    })?;

    let dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to create output directory: {e}"),
        )
    })?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let output_path = dir.join(format!("random_bg_{}.jpg", timestamp));

    img.save(&output_path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to save background image: {e}"),
        )
    })?;

    Ok(output_path.to_string_lossy().to_string())
}

/// 画像ファイルをアスキーアート文字列に変換します。
///
/// * `path` - 画像ファイルのローカル絶対パス
/// * `levels` - 明るさの順に並べた文字列（オプション。デフォルト: `"@%#*+=-:. "`）
/// * `width` - 出力のアスキーアートの幅（文字数、オプション。省略時は元画像の幅）
#[napi]
pub fn ascii_drawing(
    path: String,
    levels: Option<String>,
    width: Option<u32>,
) -> napi::Result<String> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {path}: {e}"),
        )
    })?;

    let levels = levels.unwrap_or_else(|| "@%#*+=-:. ".to_string());

    let result = crate::ascii::ascii_drawing(&img, &levels, width).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("ASCII drawing failed: {e}"),
        )
    })?;

    Ok(result)
}
