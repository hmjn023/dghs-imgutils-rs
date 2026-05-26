use ndarray::{Array4, ArrayView4, s};

pub fn area_batch_run<F>(
    origin_input: &Array4<f32>,
    func: F,
    scale: usize,
    tile_size: usize,
    tile_overlap: usize,
    batch_size: usize,
    input_channels: usize,
    output_channels: usize,
) -> Array4<f32>
where
    F: Fn(&Array4<f32>) -> Array4<f32>,
{
    let shape = origin_input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    assert_eq!(
        channels, input_channels,
        "Input channels mismatch: expected {input_channels}, got {channels}"
    );

    let tile = tile_size.min(height).min(width);
    let stride = tile.saturating_sub(tile_overlap);
    if stride == 0 {
        return func(origin_input);
    }

    let mut h_indices: Vec<usize> = (0..height.saturating_sub(tile)).step_by(stride).collect();
    if height > tile {
        h_indices.push(height - tile);
    } else {
        h_indices.push(0);
    }
    h_indices.sort();
    h_indices.dedup();

    let mut w_indices: Vec<usize> = (0..width.saturating_sub(tile)).step_by(stride).collect();
    if width > tile {
        w_indices.push(width - tile);
    } else {
        w_indices.push(0);
    }
    w_indices.sort();
    w_indices.dedup();

    let out_h = height * scale;
    let out_w = width * scale;

    let mut sum_ = Array4::<f32>::zeros((batch, output_channels, out_h, out_w));
    let mut weight = Array4::<f32>::zeros((batch, output_channels, out_h, out_w));

    let mut all_patches: Vec<Array4<f32>> = Vec::new();
    let mut all_indices: Vec<(usize, usize)> = Vec::new();

    for &h_idx in &h_indices {
        for &w_idx in &w_indices {
            let patch = origin_input
                .slice(s![.., .., h_idx..h_idx + tile, w_idx..w_idx + tile])
                .to_owned();
            all_patches.push(patch);
            all_indices.push((h_idx, w_idx));
        }
    }

    let num_patches = all_patches.len();
    let mut results: Vec<(usize, usize, Array4<f32>)> = Vec::with_capacity(num_patches);

    for (i, chunk) in all_patches.chunks(batch_size).enumerate() {
        let views: Vec<ArrayView4<f32>> = chunk.iter().map(|a| a.view()).collect();
        let batch_chunk = if chunk.len() == batch_size {
            ndarray::concatenate(ndarray::Axis(0), &views).unwrap()
        } else {
            let mut padded_views = views.clone();
            let last_view = views.last().unwrap().to_owned();
            while padded_views.len() < batch_size {
                padded_views.push(last_view.view());
            }
            ndarray::concatenate(ndarray::Axis(0), &padded_views).unwrap()
        };

        let output = func(&batch_chunk);
        let start_idx = i * batch_size;

        for (j, _) in chunk.iter().enumerate() {
            let (h_idx, w_idx) = all_indices[start_idx + j];
            let out_slice = output
                .index_axis(ndarray::Axis(0), j)
                .to_owned()
                .insert_axis(ndarray::Axis(0));
            results.push((h_idx, w_idx, out_slice));
        }
    }

    for (h_idx, w_idx, output) in results {
        let h_min = h_idx * scale;
        let h_max = (h_idx + tile) * scale;
        let w_min = w_idx * scale;
        let w_max = (w_idx + tile) * scale;

        {
            let mut sum_slice = sum_.slice_mut(s![.., .., h_min..h_max, w_min..w_max]);
            for (s, o) in sum_slice.iter_mut().zip(output.iter()) {
                *s += o;
            }
        }

        {
            let mut weight_slice = weight.slice_mut(s![.., .., h_min..h_max, w_min..w_max]);
            for w in weight_slice.iter_mut() {
                *w += 1.0;
            }
        }
    }

    let inv_weight = weight.mapv(|w| if w > 0.0 { 1.0 / w } else { 0.0 });
    sum_ * inv_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_batch_run_identity() {
        let input = Array4::<f32>::ones((1, 3, 64, 64));
        let identity = |x: &Array4<f32>| x.clone();
        let result = area_batch_run(&input, identity, 1, 32, 4, 4, 3, 3);
        assert_eq!(result.shape(), &[1, 3, 64, 64]);
    }

    #[test]
    fn test_area_batch_run_small() {
        let input =
            Array4::<f32>::from_shape_fn((1, 1, 16, 16), |(_, _, h, w)| (h * 16 + w) as f32);
        let identity = |x: &Array4<f32>| x.clone();
        let result = area_batch_run(&input, identity, 1, 16, 4, 4, 1, 1);
        assert_eq!(result.shape(), &[1, 1, 16, 16]);
    }
}
