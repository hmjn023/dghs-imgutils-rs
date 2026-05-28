//! ユーティリティ (utils) モジュールの napi-rs バインディング。

use napi_derive::napi;

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
