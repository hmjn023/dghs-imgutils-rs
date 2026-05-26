//! 背景画像の管理機能。
//!
//! HuggingFace Dataset [`deepghs/anime-bg`](https://huggingface.co/datasets/deepghs/anime-bg)
//! からアニメ背景画像をダウンロード・キャッシュ・取得します。
//!
//! # 使用例
//!
//! ```no_run
//! use dghs_imgutils_rs::resource::{list_bg_images, get_bg_image, random_bg_image};
//!
//! // 利用可能な背景画像一覧
//! let images = list_bg_images();
//! println!("利用可能な背景画像: {} 枚", images.len());
//!
//! // 特定の画像を読み込み
//! let img = get_bg_image("000000.jpg").unwrap();
//!
//! // ランダムな画像を読み込み
//! let random = random_bg_image().unwrap();
//! ```

use crate::hub::hf_hub_download;
use image::DynamicImage;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// 背景画像管理に関するエラー型。
#[derive(Debug, Error)]
pub enum BackgroundError {
    /// Hub からのダウンロードエラー
    #[error("Hub download error: {0}")]
    Hub(#[from] crate::hub::HubError),

    /// I/O エラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// 画像読み込みエラー
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// CSV パースエラー
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// 指定された背景画像が見つからない
    #[error("Background image not found: {0}")]
    NotFound(String),

    /// Tar アーカイブ内にファイルが見つからない
    #[error("File {0} not found in tar archive {1}")]
    TarEntryNotFound(String, String),

    /// 不正なキャッシュ状態
    #[error("Cache error: {0}")]
    Cache(String),
}

/// `deepghs/anime-bg` データセットのリポジトリ ID。
const BG_REPO: &str = "deepghs/anime-bg";

/// CSV のヘッダー行。
const CSV_FILENAME: &str = "images.csv";

/// 背景画像をキャッシュするサブディレクトリ名。
const BG_CACHE_SUBDIR: &str = "bg";

/// CSV のレコードを表す構造体。
#[derive(Debug, Clone, serde::Deserialize)]
struct CsvRecord {
    archive: String,
    filename: String,
    #[allow(dead_code)]
    mimetype: String,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    file_size: u64,
}

/// 背景画像のキャッシュディレクトリを取得します。
fn get_bg_cache_dir() -> PathBuf {
    crate::utils::storage::get_storage_dir().join(BG_CACHE_SUBDIR)
}

/// 抽出済みの画像ファイルのパスを計算します。
fn get_cached_image_path(filename: &str) -> PathBuf {
    get_bg_cache_dir().join(filename)
}

/// CSV をダウンロード（未キャッシュの場合）してパースし、画像ファイル名の一覧を返します。
///
/// 返される Vec は `images.csv` に含まれる全画像のファイル名（例: `"000000.jpg"`）です。
pub fn list_bg_images() -> Vec<String> {
    let cache_dir = get_bg_cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();

    let csv_path = match hf_hub_download(BG_REPO, CSV_FILENAME, Some("dataset"), None) {
        Ok(path) => path,
        Err(_) => return Vec::new(),
    };

    let mut images = Vec::new();
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    for result in rdr.deserialize::<CsvRecord>() {
        if let Ok(record) = result {
            images.push(record.filename);
        }
    }

    images
}

/// 指定されたファイル名の背景画像を読み込みます。
///
/// 画像がキャッシュディレクトリに存在しない場合、対応する tar アーカイブを
/// HuggingFace からダウンロードし、ファイルを抽出してキャッシュした後に読み込みます。
///
/// # 引数
///
/// * `filename` - 画像ファイル名（例: `"000000.jpg"`）
///
/// # エラー
///
/// * ファイル名が CSV 内に見つからない場合
/// * tar アーカイブ内にファイルが見つからない場合
/// * 画像の読み込みに失敗した場合
pub fn get_bg_image(filename: &str) -> Result<DynamicImage, BackgroundError> {
    let cache_dir = get_bg_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let cached_path = get_cached_image_path(filename);
    if cached_path.exists() {
        return Ok(image::open(&cached_path)?);
    }

    let csv_path = hf_hub_download(BG_REPO, CSV_FILENAME, Some("dataset"), None)?;
    let mut rdr = csv::Reader::from_path(&csv_path)?;

    let mut target_archive: Option<String> = None;
    for r in rdr.deserialize() {
        let record: CsvRecord = r?;
        if record.filename == filename {
            target_archive = Some(record.archive);
            break;
        }
    }

    let archive_name =
        target_archive.ok_or_else(|| BackgroundError::NotFound(filename.to_string()))?;

    let archive_path = format!("images/{}", archive_name);
    let tar_path = hf_hub_download(BG_REPO, &archive_path, Some("dataset"), None)?;

    extract_file_from_tar(&tar_path, filename, &cache_dir)?;

    if cached_path.exists() {
        Ok(image::open(&cached_path)?)
    } else {
        Err(BackgroundError::TarEntryNotFound(
            filename.to_string(),
            archive_name,
        ))
    }
}

/// tar アーカイブから指定されたファイルを抽出し、キャッシュディレクトリに保存します。
fn extract_file_from_tar(
    tar_path: &Path,
    filename: &str,
    output_dir: &Path,
) -> Result<(), BackgroundError> {
    let file = std::fs::File::open(tar_path)?;
    let mut archive = tar::Archive::new(file);

    for entry in archive.entries()? {
        let mut entry = entry?;
        if let Ok(entry_path) = entry.path() {
            if entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == filename)
            {
                let output_path = output_dir.join(filename);
                entry.unpack(&output_path)?;
                return Ok(());
            }
        }
    }

    Err(BackgroundError::TarEntryNotFound(
        filename.to_string(),
        tar_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string(),
    ))
}

/// 利用可能な背景画像からランダムに1枚を選び、読み込みます。
///
/// 内部的には `list_bg_images()` で一覧を取得し、ランダムに選んだファイル名を
/// `get_bg_image()` に渡します。
///
/// # エラー
///
/// * 画像一覧が空の場合
/// * 選択した画像の読み込みに失敗した場合
pub fn random_bg_image() -> Result<DynamicImage, BackgroundError> {
    let images = list_bg_images();
    if images.is_empty() {
        return Err(BackgroundError::Cache(
            "No background images available".to_string(),
        ));
    }

    use rand::Rng;
    let idx = rand::thread_rng().gen_range(0..images.len());
    let filename = &images[idx];
    get_bg_image(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_bg_images() {
        let images = list_bg_images();
        // CSV がダウンロードできている場合、少なくとも1件以上あるはず
        if !images.is_empty() {
            assert!(images[0].ends_with(".jpg"));
        }
    }

    #[test]
    fn test_get_bg_image_not_found() {
        let result = get_bg_image("nonexistent.jpg");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_cached_image_path() {
        let path = get_cached_image_path("test.jpg");
        assert!(path.ends_with("test.jpg"));
    }
}
