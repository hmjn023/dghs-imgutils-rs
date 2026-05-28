use crate::sd::safetensors::read_safetensors_metadata as core_read_safetensors_metadata;
use napi_derive::napi;
use std::collections::HashMap;

/// PNG 画像ファイルから生成情報 (geninfo / parameters) を読み取ります。
#[napi]
pub fn read_geninfo(path: String) -> napi::Result<Option<String>> {
    let result = crate::metadata::geninfo::read_geninfo_parameters(&path);
    Ok(result)
}

/// EXIF メタデータから生成情報を読み取ります。
#[napi]
pub fn read_geninfo_exif(path: String) -> napi::Result<Option<String>> {
    let result = crate::metadata::geninfo::read_geninfo_exif(&path);
    Ok(result)
}

/// GIF メタデータから生成情報を読み取ります。
#[napi]
pub fn read_geninfo_gif(path: String) -> napi::Result<Option<String>> {
    let result = crate::metadata::geninfo::read_geninfo_gif(&path);
    Ok(result)
}

/// 画像から生成情報の生データ（全フィールド）を読み取ります。
#[napi]
pub fn read_geninfo_raw(path: String) -> napi::Result<Option<HashMap<String, String>>> {
    let result = crate::metadata::geninfo::read_geninfo_raw(&path);
    Ok(result)
}

/// 画像の LSB (Least Significant Bit) ステガノグラフィからメタデータを読み取ります。
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

/// LSB から生バイト列を読み取ります。
#[napi]
pub fn read_lsb_raw_bytes(path: String) -> napi::Result<Vec<u8>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {e}"),
        )
    })?;
    let result = crate::metadata::lsb::read_lsb_raw_bytes(&img).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("LSB read failed: {e}"),
        )
    })?;
    Ok(result)
}

/// メタデータを画像の LSB に書き込み、新しい画像ファイルとして保存します。
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

/// 生バイト列を画像の LSB に書き込み、新しい画像ファイルとして保存します。
#[napi]
pub fn write_lsb_raw_bytes(src_path: String, dst_path: String, data: Vec<u8>) -> napi::Result<()> {
    let img = image::open(&src_path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open source image: {e}"),
        )
    })?;
    let result = crate::metadata::lsb::write_lsb_raw_bytes(&img, &data);
    result.save(&dst_path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to save image: {e}"),
        )
    })?;
    Ok(())
}

/// 画像ファイルから A1111 形式の Stable Diffusion メタデータをパースします。
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

/// NovelAI メタデータを画像から読み取ります。
#[napi(object)]
pub struct NaiMetadataResult {
    pub software: String,
    pub source: String,
    pub title: Option<String>,
    pub description: Option<String>,
}

/// 画像から NovelAI メタデータを抽出します。
#[napi]
pub fn get_nai_metadata(path: String) -> napi::Result<Option<NaiMetadataResult>> {
    let img = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image: {e}"),
        )
    })?;
    let meta = crate::sd::nai::get_naimeta_from_image(&img);
    match meta {
        Some(m) => Ok(Some(NaiMetadataResult {
            software: m.software,
            source: m.source,
            title: m.title,
            description: m.description,
        })),
        None => Ok(None),
    }
}

/// 画像から SD メタデータを抽出します（画像ファイル直接読み取り版）。
#[napi]
pub fn get_sdmeta_from_image(path: String) -> napi::Result<Option<SdMetadataResult>> {
    let meta = crate::sd::metadata::get_sdmeta_from_image(&path);
    match meta {
        Some(m) => Ok(Some(SdMetadataResult {
            prompt: m.prompt,
            neg_prompt: m.neg_prompt,
            parameters: m.parameters,
        })),
        None => Ok(None),
    }
}

/// 画像ファイルに SD メタデータを保存します。
///
/// * `src_path` — 元画像のファイルパス
/// * `dst_path` — 保存先ファイルパス
/// * `prompt` — メインプロンプト
/// * `neg_prompt` — ネガティブプロンプト
/// * `parameters` — パラメーターのキーバリューマップ
#[napi]
pub fn save_image_with_sdmeta(
    src_path: String,
    dst_path: String,
    prompt: String,
    neg_prompt: String,
    parameters: HashMap<String, String>,
) -> napi::Result<()> {
    let meta = crate::sd::metadata::SDMetaData {
        prompt,
        neg_prompt,
        parameters,
    };
    crate::sd::metadata::save_image_with_sdmeta(&src_path, &dst_path, &meta).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to save image with SD metadata: {e}"),
        )
    })
}

/// PNG 画像に `parameters` テキストチャンクを書き込みます。
#[napi]
pub fn write_geninfo_parameters(
    src_path: String,
    dst_path: String,
    geninfo: String,
) -> napi::Result<()> {
    crate::metadata::geninfo::write_geninfo_parameters(&src_path, &dst_path, &geninfo).map_err(
        |e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to write geninfo parameters: {e}"),
            )
        },
    )
}

/// JPEG/WebP 画像に EXIF UserComment を書き込みます。
#[napi]
pub fn write_geninfo_exif(src_path: String, dst_path: String, geninfo: String) -> napi::Result<()> {
    crate::metadata::geninfo::write_geninfo_exif(&src_path, &dst_path, &geninfo).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to write geninfo EXIF: {e}"),
        )
    })
}

/// GIF 画像にコメント拡張ブロックを書き込みます。
#[napi]
pub fn write_geninfo_gif(src_path: String, dst_path: String, geninfo: String) -> napi::Result<()> {
    crate::metadata::geninfo::write_geninfo_gif(&src_path, &dst_path, &geninfo).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to write geninfo GIF: {e}"),
        )
    })
}

/// Safetensors モデルファイルからメタデータを読み取ります。
#[napi]
pub fn read_safetensors_metadata(path: String) -> napi::Result<HashMap<String, String>> {
    core_read_safetensors_metadata(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("Failed to read safetensors metadata: {e}"),
        )
    })
}
