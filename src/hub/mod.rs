//! HuggingFace Hub からモデルファイルをダウンロードする機能を提供します。
//!
//! Python の `huggingface_hub.hf_hub_download` に相当する機能です。
//! ダウンロードしたファイルは HuggingFace 標準のキャッシュディレクトリ
//! (`~/.cache/huggingface/hub/`) に保存され、2 回目以降はキャッシュから取得します。
//!
//! # 環境変数
//!
//! | 変数名 | 説明 |
//! |--------|------|
//! | `HF_HOME` | キャッシュディレクトリのルート（デフォルト: `~/.cache/huggingface`） |
//! | `HF_ENDPOINT` | HuggingFace エンドポイント URL（デフォルト: `https://huggingface.co`） |
//! | `HF_HUB_TOKEN` | 認証トークン（プライベートリポジトリへのアクセスに使用） |
//!
//! # 使用例
//!
//! ```no_run
//! use dghs_imgutils_rs::hub::hf_hub_download;
//!
//! let path = hf_hub_download("deepghs/ccip_onnx", "ccip-caformer-24-randaug-pruned/model_feat.onnx", None, None).unwrap();
//! println!("ダウンロード先: {}", path.display());
//! ```

pub mod error;
pub mod model_list;

pub use error::HubError;

use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use std::path::PathBuf;

/// HuggingFace Hub からファイルをダウンロードし、ローカルキャッシュのパスを返します。
///
/// キャッシュに既に存在する場合はダウンロードをスキップします。
/// ダウンロード時は `indicatif` によるプログレスバーが表示されます。
///
/// # 引数
///
/// * `repo_id` - リポジトリ ID（例: `"deepghs/ccip_onnx"`）
/// * `filename` - リポジトリ内のファイルパス（例: `"model/model_feat.onnx"`）
/// * `repo_type` - リポジトリの種類（`"model"`, `"dataset"`, `"space"` のいずれか。`None` の場合は `"model"`）
/// * `token` - 認証トークン（`None` の場合は `HF_HUB_TOKEN` 環境変数、
///   または `huggingface-cli login` で設定したトークンを使用）
///
/// # 戻り値
///
/// キャッシュされたファイルのローカルパス。
///
/// # エラー
///
/// - ネットワークエラー
/// - 認証エラー（プライベートリポジトリに対してトークンが無効・未設定）
/// - ファイルが見つからない場合
pub fn hf_hub_download(
    repo_id: &str,
    filename: &str,
    repo_type: Option<&str>,
    token: Option<String>,
) -> Result<PathBuf, HubError> {
    // HF_HUB_TOKEN 環境変数をフォールバックとして利用
    let token = token.or_else(|| std::env::var("HF_HUB_TOKEN").ok());

    let api = ApiBuilder::from_env().with_token(token).build()?;

    let r_type = match repo_type.unwrap_or("model") {
        "dataset" => RepoType::Dataset,
        "space" => RepoType::Space,
        _ => RepoType::Model,
    };

    let repo = api.repo(Repo::new(repo_id.to_string(), r_type));

    // キャッシュヒット時はそのまま返し、ミス時はダウンロード（進捗バー付き）
    let path = repo.get(filename)?;
    Ok(path)
}

/// `model_list::ALL_MODELS` に登録されているすべてのモデルファイルをダウンロードします。
///
/// ダウンロード中は、各ファイルのダウンロード進捗が表示されます。
/// すでにキャッシュされているファイルは一瞬でパスが取得されます。
pub fn download_all_models(token: Option<String>) -> Result<Vec<PathBuf>, HubError> {
    let mut paths = Vec::new();
    let total = model_list::ALL_MODELS.len();
    for (i, entry) in model_list::ALL_MODELS.iter().enumerate() {
        println!(
            "[{}/{}] ダウンロード中: {} ({} から {})",
            i + 1,
            total,
            entry.description,
            entry.repo_id,
            entry.filename
        );
        let path = hf_hub_download(
            entry.repo_id,
            entry.filename,
            Some(entry.repo_type),
            token.clone(),
        )?;
        println!("✓ 保存先: {}", path.display());
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 実際に HuggingFace へアクセスするため、デフォルトではスキップ。
    /// 実行するには: `cargo test -- --ignored`
    #[test]
    #[ignore]
    fn test_download_ccip_metrics() {
        let path = hf_hub_download(
            "deepghs/ccip_onnx",
            "ccip-caformer-24-randaug-pruned/metrics.json",
            None,
            None,
        )
        .expect("ダウンロードに失敗しました");

        assert!(
            path.exists(),
            "ダウンロードしたファイルが存在しない: {}",
            path.display()
        );
        assert!(path.metadata().unwrap().len() > 0, "ファイルが空です");

        println!("✓ ダウンロード先: {}", path.display());
    }
}
