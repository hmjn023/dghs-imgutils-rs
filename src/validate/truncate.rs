//! 画像ファイルの物理的破損チェック。
//!
//! Python の `PIL.Image.open(path).load()` が OSError を出すかどうかで判定する
//! 挙動と等価に、Rust の `image::open(path)` でのデコードを試みます。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TruncateError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// 画像ファイルが物理的に破損しているかどうかを検査します。
///
/// `image::open` によるフルデコードを試み、エラーが発生した場合は `true` を返します。
///
/// # 引数
/// * `path` - 検査するファイルのパス
///
/// # 戻り値
/// * `true` - ファイルが破損している（または画像として解釈不能）
/// * `false` - ファイルは正常
pub fn is_truncated_file(path: &str) -> bool {
    // to_rgb8() まで実行することでデコードを完全に行う（PIL の .load() と等価）
    match image::open(path) {
        Ok(img) => {
            // フルデコード（全ピクセル読み込み）
            let _ = img.to_rgb8();
            false
        }
        Err(_) => true,
    }
}
