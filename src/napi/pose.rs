use crate::pose::dwpose::dwpose_estimate as core_dwpose_estimate;
use crate::pose::format::OP18KeyPointSet as CoreOP18KeyPointSet;
use crate::pose::visual::op18_visualize as core_op18_visualize;
use napi_derive::napi;

#[napi(object)]
pub struct NapiOP18KeyPoint {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

#[napi(object)]
pub struct NapiOP18KeyPointSet {
    pub keypoints: Vec<NapiOP18KeyPoint>,
    pub bbox: Vec<f64>,
}

impl From<CoreOP18KeyPointSet> for NapiOP18KeyPointSet {
    fn from(ks: CoreOP18KeyPointSet) -> Self {
        let keypoints = ks
            .keypoints
            .iter()
            .map(|kp| NapiOP18KeyPoint {
                x: kp.x as f64,
                y: kp.y as f64,
                score: kp.score as f64,
            })
            .collect();
        let bbox = ks.bbox.iter().map(|&v| v as f64).collect();
        NapiOP18KeyPointSet { keypoints, bbox }
    }
}

/// DWpose (RTMPose-x) モデルを用いて画像から人体キーポイントを推定します。
///
/// 内部で人物検出（detect_person）を自動実行し、検出された各人物に対して
/// キーポイント推定（133+1=134点のOpenPose 18形式）を行います。
///
/// * `path`: 画像ファイルのローカル絶対パスまたは相対パス
#[napi]
pub fn dwpose_estimate(path: String) -> napi::Result<Vec<NapiOP18KeyPointSet>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let results = core_dwpose_estimate(&image, true, None, None).map_err(|e| {
        napi::Error::new(
            napi::Status::GenericFailure,
            format!("DWpose estimation failed: {}", e),
        )
    })?;

    Ok(results.into_iter().map(NapiOP18KeyPointSet::from).collect())
}

/// OP18 キーポイント推定結果を画像上に可視化し、PNG バッファとして返します。
///
/// * `path`: 元画像のファイルパス
/// * `keypoints_list`: キーポイント推定結果のリスト
#[napi]
pub fn op18_visualize(
    path: String,
    keypoints_list: Vec<NapiOP18KeyPointSet>,
) -> napi::Result<Vec<u8>> {
    let image = image::open(&path).map_err(|e| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("Failed to open image at {}: {}", path, e),
        )
    })?;

    let core_kps: Vec<CoreOP18KeyPointSet> = keypoints_list
        .into_iter()
        .map(|napi_kps| {
            let keypoints = napi_kps
                .keypoints
                .into_iter()
                .map(|kp| crate::pose::format::OP18KeyPoint {
                    x: kp.x as f32,
                    y: kp.y as f32,
                    score: kp.score as f32,
                })
                .collect();
            let bbox: [f32; 4] = [
                napi_kps.bbox[0] as f32,
                napi_kps.bbox[1] as f32,
                napi_kps.bbox[2] as f32,
                napi_kps.bbox[3] as f32,
            ];
            CoreOP18KeyPointSet { keypoints, bbox }
        })
        .collect();

    let vis_img = core_op18_visualize(&image, &core_kps, 0.3, Some(512), true, true, true, true);

    let mut buf = std::io::Cursor::new(Vec::new());
    vis_img
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to encode visualized image: {}", e),
            )
        })?;

    Ok(buf.into_inner())
}
