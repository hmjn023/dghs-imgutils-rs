use crate::sd::safetensors::read_safetensors_metadata as core_read_safetensors_metadata;
use napi_derive::napi;
use std::collections::HashMap;

/// PNG 画像ファイルから生成情報 (geninfo / parameters) を読み取ります。
///
/// A1111 WebUI が PNG の iTXt/tEXt チャンクに書き込む `parameters` フィールドを取得します。
///
/// * `path`: 画像ファイルの絶対パスまたは相対パス
#[napi]
pub fn read_geninfo(path: String) -> napi::Result<Option<String>> {
    let result = crate::metadata::geninfo::read_geninfo_parameters(&path);
    Ok(result)
}

/// 画像の LSB (Least Significant Bit) ステガノグラフィからメタデータを読み取ります。
///
/// * `path`: 画像ファイルのパス
#[napi]
pub fn read_lsb_metadata(path: String) -> napi::Result<HashMap<String, String>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {e}"),
        )
    })?;
    let result = crate::metadata::lsb::read_lsb_metadata(&img);
    Ok(result)
}

/// メタデータを画像の LSB に書き込み、新しい画像ファイルとして保存します。
///
/// * `src_path`: 元画像のパス
/// * `dst_path`: 出力先の画像パス
/// * `metadata`: 書き込むキー／値マップ
#[napi]
pub fn write_lsb_metadata(
    src_path: String,
    dst_path: String,
    metadata: HashMap<String, String>,
) -> napi::Result<()> {
    let img = image::open(&src_path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open source image: {e}"),
        )
    })?;
    let result = crate::metadata::lsb::write_lsb_metadata(&img, &metadata);
    result.save(&dst_path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to save image: {e}"),
        )
    })?;
    Ok(())
}

/// 画像ファイルから A1111 形式の Stable Diffusion メタデータをパースします。
///
/// * `path`: 画像ファイルのパス
#[napi(object)]
pub struct SdMetadataResult {
    pub prompt: String,
    pub neg_prompt: String,
    pub parameters: HashMap<String, String>,
}

/// 画像ファイルから SD メタデータを抽出します。
#[napi]
pub fn get_sd_metadata(path: String) -> napi::Result<Option<SdMetadataResult>> {
    let geninfo = crate::metadata::geninfo::read_geninfo_parameters(&path);
    let text = match geninfo {
        Some(t) => t,
        None => return Ok(None),
    };
    let meta = crate::sd::metadata::parse_sdmeta_from_text(&text);
    Ok(Some(SdMetadataResult {
        prompt: meta.prompt,
        neg_prompt: meta.neg_prompt,
        parameters: meta.parameters,
    }))
}

/// Safetensors モデルファイルからメタデータを読み取ります。
///
/// * `path`: `.safetensors` ファイルのパス
#[napi]
pub fn read_safetensors_metadata(path: String) -> napi::Result<HashMap<String, String>> {
    core_read_safetensors_metadata(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to read safetensors metadata: {e}"),
        )
    })
}
