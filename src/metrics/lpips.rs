use crate::hub::hf_hub_download;
use crate::image::force_image_background;
use crate::inference::{create_onnx_session, run_onnx_session};

use image::DynamicImage;
use ndarray::Array4;
use ort::inputs;

const REPO_ID: &str = "deepghs/imgutils-models";
const FEATURE_FILENAME: &str = "lpips/lpips_feature.onnx";
const DIFF_FILENAME: &str = "lpips/lpips_diff.onnx";
const INPUT_SIZE: u32 = 400;

fn preprocess(image: &DynamicImage) -> Result<ndarray::Array4<f32>, crate::metrics::CcipError> {
    let rgb = force_image_background(image, [255, 255, 255]);
    let resized = rgb.resize_exact(
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let rgb8 = resized.to_rgb8();
    let (w, h) = rgb8.dimensions();
    let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in rgb8.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        tensor[[0, 0, y as usize, x as usize]] = (r - 0.5) / 0.5;
        tensor[[0, 1, y as usize, x as usize]] = (g - 0.5) / 0.5;
        tensor[[0, 2, y as usize, x as usize]] = (b - 0.5) / 0.5;
    }
    Ok(tensor)
}

pub type LpipsFeatures = (
    Array4<f32>,
    Array4<f32>,
    Array4<f32>,
    Array4<f32>,
    Array4<f32>,
);

fn extract_feat_array(value: &ort::value::Value) -> Result<Array4<f32>, crate::metrics::CcipError> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Array4::from_shape_vec(
        (dims[0], dims[1], dims[2], dims[3]),
        data.to_vec(),
    )?)
}

pub fn lpips_extract_feature(
    image: &DynamicImage,
) -> Result<LpipsFeatures, crate::metrics::CcipError> {
    let model_path = hf_hub_download(REPO_ID, FEATURE_FILENAME, Some("model"), None)?;
    let mut session = create_onnx_session(&model_path)?;
    let tensor = preprocess(image)?;
    let outputs = run_onnx_session(&mut session, "input", &tensor)?;

    let feat_names = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"];
    let mut feats = Vec::new();
    for name in &feat_names {
        let value = outputs.get(*name).ok_or_else(|| {
            crate::inference::InferenceError::InvalidShape(format!("No '{}' tensor found", name))
        })?;
        feats.push(extract_feat_array(value)?);
    }

    Ok((
        feats.remove(0),
        feats.remove(0),
        feats.remove(0),
        feats.remove(0),
        feats.remove(0),
    ))
}

pub fn lpips_difference(
    f1: &LpipsFeatures,
    f2: &LpipsFeatures,
) -> Result<f32, crate::metrics::CcipError> {
    let model_path = hf_hub_download(REPO_ID, DIFF_FILENAME, Some("model"), None)?;
    let mut session = create_onnx_session(&model_path)?;

    let tx0 = ort::value::Tensor::from_array(f1.0.clone())?;
    let tx1 = ort::value::Tensor::from_array(f1.1.clone())?;
    let tx2 = ort::value::Tensor::from_array(f1.2.clone())?;
    let tx3 = ort::value::Tensor::from_array(f1.3.clone())?;
    let tx4 = ort::value::Tensor::from_array(f1.4.clone())?;

    let ty0 = ort::value::Tensor::from_array(f2.0.clone())?;
    let ty1 = ort::value::Tensor::from_array(f2.1.clone())?;
    let ty2 = ort::value::Tensor::from_array(f2.2.clone())?;
    let ty3 = ort::value::Tensor::from_array(f2.3.clone())?;
    let ty4 = ort::value::Tensor::from_array(f2.4.clone())?;

    let outputs = session.run(inputs![
        "feat_x_0" => tx0,
        "feat_x_1" => tx1,
        "feat_x_2" => tx2,
        "feat_x_3" => tx3,
        "feat_x_4" => tx4,
        "feat_y_0" => ty0,
        "feat_y_1" => ty1,
        "feat_y_2" => ty2,
        "feat_y_3" => ty3,
        "feat_y_4" => ty4,
    ])?;

    let output_value = outputs.get("output").ok_or_else(|| {
        crate::inference::InferenceError::InvalidShape("No 'output' tensor found".to_string())
    })?;
    let (_shape, data) = output_value.try_extract_tensor::<f32>()?;
    Ok(data[0])
}

pub fn lpips_clustering(
    images: &[DynamicImage],
    threshold: Option<f32>,
) -> Result<Vec<i32>, crate::metrics::CcipError> {
    let thr = threshold.unwrap_or(0.45);
    if images.is_empty() {
        return Ok(Vec::new());
    }

    let n = images.len();
    let feats: Vec<LpipsFeatures> = images
        .iter()
        .map(|img| lpips_extract_feature(img))
        .collect::<Result<Vec<_>, _>>()?;

    let mut diff_matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = lpips_difference(&feats[i], &feats[j])?;
            diff_matrix[i][j] = d;
            diff_matrix[j][i] = d;
        }
    }

    let get_neighbors =
        |i: usize| -> Vec<usize> { (0..n).filter(|&j| diff_matrix[i][j] <= thr).collect() };

    let mut labels = vec![-2i32; n];
    let mut cluster_id = 0i32;
    let min_samples = 2;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbors = get_neighbors(i);
        if neighbors.len() < min_samples {
            labels[i] = -1;
            continue;
        }
        labels[i] = cluster_id;
        let mut seed_set: Vec<usize> = neighbors.iter().filter(|&&nb| nb != i).copied().collect();
        let mut idx = 0;
        while idx < seed_set.len() {
            let curr = seed_set[idx];
            idx += 1;
            if labels[curr] == -1 {
                labels[curr] = cluster_id;
                continue;
            }
            if labels[curr] != -2 {
                continue;
            }
            labels[curr] = cluster_id;
            let curr_neighbors = get_neighbors(curr);
            if curr_neighbors.len() >= min_samples {
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
