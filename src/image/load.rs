use image::DynamicImage;
use url::Url;

use super::error::ImageError;

pub enum ImageSource<'a> {
    Path(&'a str),
    Bytes(&'a [u8]),
    BlobUrl(&'a str),
    HttpUrl(&'a str),
    Image(&'a DynamicImage),
}

pub fn load_image(source: &ImageSource) -> Result<DynamicImage, ImageError> {
    let img = match source {
        ImageSource::Path(path) => image::open(path)?,
        ImageSource::Bytes(bytes) => image::load_from_memory(bytes)?,
        ImageSource::BlobUrl(url) => load_image_from_blob_url(url)?,
        ImageSource::HttpUrl(url) => download_image_from_url(url)?,
        ImageSource::Image(img) => (*img).clone(),
    };
    Ok(img)
}

pub fn load_image_with_background(
    source: &ImageSource,
    force_background: Option<[u8; 3]>,
) -> Result<DynamicImage, ImageError> {
    let img = load_image(source)?;
    if let Some(bg) = force_background {
        if img.color().has_alpha() {
            return Ok(crate::image::force_image_background(&img, bg));
        }
    }
    Ok(img)
}

pub fn check_image_source(source: &str) -> ImageSourceKind {
    if is_valid_image_blob_url(source) {
        ImageSourceKind::Blob
    } else if is_http_url(source) {
        ImageSourceKind::Http
    } else {
        ImageSourceKind::File
    }
}

pub enum ImageSourceKind {
    File,
    Blob,
    Http,
}

fn is_valid_image_blob_url(s: &str) -> bool {
    if let Some(rest) = s.strip_prefix("data:image/") {
        rest.contains(',')
    } else {
        false
    }
}

fn is_http_url(s: &str) -> bool {
    if let Ok(parsed) = Url::parse(s) {
        matches!(parsed.scheme(), "http" | "https")
    } else {
        false
    }
}

fn load_image_from_blob_url(blob_url: &str) -> Result<DynamicImage, ImageError> {
    let (header, data) = blob_url
        .split_once(',')
        .ok_or_else(|| ImageError::InvalidArgument("Invalid blob URL: missing comma".into()))?;

    let encoding = header.rsplit(';').next().unwrap_or("");
    if encoding != "base64" {
        return Err(ImageError::InvalidArgument(format!(
            "Unsupported blob encoding: {encoding:?}"
        )));
    }

    use base64::Engine as _;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| ImageError::InvalidArgument(format!("Base64 decode error: {e}")))?;

    let img = image::load_from_memory(&decoded)?;
    Ok(img)
}

fn download_image_from_url(url: &str) -> Result<DynamicImage, ImageError> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| ImageError::Http(format!("HTTP download failed: {e}")))?;

    let mut bytes: Vec<u8> = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(ImageError::Io)?;

    let img = image::load_from_memory(&bytes)?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, RgbImage};

    #[test]
    fn test_load_image_from_path() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let loaded = load_image(&ImageSource::Image(&img)).unwrap();
        assert_eq!(loaded.dimensions(), (10, 10));
    }

    #[test]
    fn test_load_image_from_bytes() {
        let mut rgba_img = image::RgbaImage::new(2, 2);
        for p in rgba_img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let dyn_img = DynamicImage::ImageRgba8(rgba_img);
        let mut buf = std::io::Cursor::new(Vec::new());
        dyn_img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let bytes = buf.into_inner();
        let loaded = load_image(&ImageSource::Bytes(&bytes)).unwrap();
        assert_eq!(loaded.dimensions(), (2, 2));
    }

    #[test]
    fn test_check_image_source() {
        assert!(matches!(
            check_image_source("test.png"),
            ImageSourceKind::File
        ));
        assert!(matches!(
            check_image_source("https://example.com/image.jpg"),
            ImageSourceKind::Http
        ));
        assert!(matches!(
            check_image_source("data:image/png;base64,abc"),
            ImageSourceKind::Blob
        ));
    }

    #[test]
    fn test_is_http_url() {
        assert!(is_http_url("https://example.com"));
        assert!(is_http_url("http://example.com"));
        assert!(!is_http_url("ftp://example.com"));
        assert!(!is_http_url("relative/path"));
    }

    #[test]
    fn test_is_valid_image_blob_url() {
        assert!(is_valid_image_blob_url("data:image/png;base64,abc123"));
        assert!(!is_valid_image_blob_url("data:text/plain,hello"));
        assert!(!is_valid_image_blob_url("not_a_blob"));
    }
}
