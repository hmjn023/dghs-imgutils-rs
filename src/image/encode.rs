use image::DynamicImage;
use ndarray::{Array3, ArrayView3};

use super::error::ImageError;

fn get_hwc_map_encode(order_: &str) -> [usize; 3] {
    let mut map = [0usize; 3];
    for (i, c) in order_.to_uppercase().chars().enumerate() {
        match c {
            'H' => map[i] = 0,
            'W' => map[i] = 1,
            'C' => map[i] = 2,
            _ => {}
        }
    }
    map
}

fn get_hwc_map_decode(order_: &str) -> [usize; 3] {
    let order = order_.to_uppercase();
    let mut map = [0usize; 3];
    let chars: Vec<char> = order.chars().collect();
    for (target, c) in ['H', 'W', 'C'].iter().enumerate() {
        map[target] = chars.iter().position(|&x| x == *c).unwrap_or(target);
    }
    map
}

pub fn rgb_encode(
    image: &DynamicImage,
    order_: &str,
    use_float: bool,
) -> Result<Array3<f32>, ImageError> {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();

    let mut arr = Array3::<f32>::zeros((h as usize, w as usize, 3));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        arr[[y as usize, x as usize, 0]] = pixel[0] as f32;
        arr[[y as usize, x as usize, 1]] = pixel[1] as f32;
        arr[[y as usize, x as usize, 2]] = pixel[2] as f32;
    }

    if use_float {
        arr.mapv_inplace(|v| v / 255.0);
    }

    let map = get_hwc_map_encode(order_);
    let permuted = arr.permuted_axes([map[0], map[1], map[2]]);
    Ok(permuted)
}

pub fn rgb_decode(tensor: &ArrayView3<f32>, order_: &str) -> Result<DynamicImage, ImageError> {
    let hwc_map = get_hwc_map_decode(order_);
    let hwc = tensor
        .view()
        .permuted_axes([hwc_map[0], hwc_map[1], hwc_map[2]]);

    let (h, w, c) = hwc.dim();
    if c != 3 {
        return Err(ImageError::Shape(format!("Expected 3 channels, got {c}")));
    }

    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r = hwc[[y, x, 0]];
            let g = hwc[[y, x, 1]];
            let b = hwc[[y, x, 2]];
            let idx = (y * w + x) * 3;
            buf[idx] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
            buf[idx + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
            buf[idx + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    let img = image::ImageBuffer::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| ImageError::Shape("Failed to create image buffer".into()))?;
    Ok(DynamicImage::ImageRgb8(img))
}

pub fn rgb_decode_raw(data: &ArrayView3<f32>, order_: &str) -> Result<DynamicImage, ImageError> {
    rgb_decode(data, order_)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, RgbImage};

    fn create_test_image() -> DynamicImage {
        let mut img = RgbImage::new(4, 3);
        for y in 0..3 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Rgb([x as u8 * 64, y as u8 * 64, 128]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_rgb_encode_chw() {
        let img = create_test_image();
        let encoded = rgb_encode(&img, "CHW", true).unwrap();
        assert_eq!(encoded.shape(), &[3, 3, 4]);
    }

    #[test]
    fn test_rgb_encode_hwc() {
        let img = create_test_image();
        let encoded = rgb_encode(&img, "HWC", true).unwrap();
        assert_eq!(encoded.shape(), &[3, 4, 3]);
    }

    #[test]
    fn test_rgb_encode_uint8() {
        let img = create_test_image();
        let encoded = rgb_encode(&img, "CHW", false).unwrap();
        assert_eq!(encoded.shape(), &[3, 3, 4]);
        assert!(
            (encoded[[0, 0, 0]] - 0.0).abs() < f32::EPSILON
                || (encoded[[0, 0, 0]] - 255.0).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_rgb_decode_roundtrip() {
        let img = create_test_image();
        let encoded = rgb_encode(&img, "CHW", true).unwrap();
        let decoded = rgb_decode(&encoded.view(), "CHW").unwrap();
        assert_eq!(decoded.dimensions(), img.dimensions());

        let orig_rgb = img.to_rgb8();
        let dec_rgb = decoded.to_rgb8();
        for y in 0..3 {
            for x in 0..4 {
                let op = orig_rgb.get_pixel(x, y);
                let dp = dec_rgb.get_pixel(x, y);
                let diff = (op[0] as i16 - dp[0] as i16).abs()
                    + (op[1] as i16 - dp[1] as i16).abs()
                    + (op[2] as i16 - dp[2] as i16).abs();
                assert!(
                    diff <= 1,
                    "Roundtrip mismatch at ({x},{y}): ({},{},{}) vs ({},{},{})",
                    op[0],
                    op[1],
                    op[2],
                    dp[0],
                    dp[1],
                    dp[2]
                );
            }
        }
    }

    #[test]
    fn test_rgb_decode_hwc_roundtrip() {
        let img = create_test_image();
        let encoded = rgb_encode(&img, "HWC", true).unwrap();
        let decoded = rgb_decode(&encoded.view(), "HWC").unwrap();
        assert_eq!(decoded.dimensions(), img.dimensions());
    }
}
