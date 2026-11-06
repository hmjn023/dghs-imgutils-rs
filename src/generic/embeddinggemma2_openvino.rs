//! Lazy, process-safe preparation of the validated Q8 OpenVINO exports.
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{GenericError, invalid, model_sha256};

#[path = "embeddinggemma2_native.rs"]
mod native;
#[path = "embeddinggemma2_onnx.rs"]
mod onnx;
const REVISION: &str = "native-blob-v2";

fn complete(root: &Path, stem: &str) -> bool {
    ["onnx", "blob"].iter().all(|extension| {
        root.join("onnx")
            .join(format!("{stem}.openvino-compiled.{extension}"))
            .metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
    })
}

/// Converts a single encoder only when its content-addressed cache is missing.
pub(super) fn prepare(graph: &Path, weights: &Path) -> Result<PathBuf, GenericError> {
    use crate::inference::{Precision, current_inference_options};
    crate::inference::init_onnx_runtime()?;
    let options = current_inference_options()?;
    let precision = match options.precision {
        Precision::Auto | Precision::Fp32 => "f32",
        Precision::Fp16 => "f16",
        _ => {
            return Err(invalid(
                "automatic Intel GPU preparation supports auto, fp32, or fp16 precision",
            ));
        }
    };
    let device = options
        .openvino_device_type
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(crate::inference::openvino_device_type)
        .trim()
        .to_ascii_uppercase();
    static READY: once_cell::sync::Lazy<
        std::sync::Mutex<std::collections::HashMap<String, PathBuf>>,
    > = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let root = crate::utils::storage::get_storage_dir();
    let key = format!(
        "{}:{}:{}:{device}:{precision}",
        root.display(),
        model_sha256(graph)?,
        model_sha256(weights)?
    );
    let mut ready = READY.lock().map_err(|e| invalid(e.to_string()))?;
    if let Some(path) = ready.get(&key)
        && path.is_file()
        && path.with_extension("blob").is_file()
    {
        return Ok(path.clone());
    }
    let native = native::Native::new(&device, precision)?;
    let cache = crate::utils::storage::get_storage_dir()
        .join("embeddinggemma2/openvino")
        .join(format!("{:x}", Sha256::digest(&native.identity)));
    let prepared = prepare_with(graph, weights, &cache, |source, output, stem| {
        let graph = source.join("onnx").join(format!("{stem}.onnx"));
        let original = fs::read(&graph)?;
        let expected = if stem == "model_quantized" {
            "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953"
        } else {
            "bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3"
        };
        if format!("{:x}", Sha256::digest(&original)) != expected {
            return Err(invalid(
                "unsupported Q8 graph revision for automatic Intel GPU preparation",
            ));
        }
        fs::write(&graph, onnx::patch(&original, stem == "model_quantized")?)?;
        fs::create_dir_all(output.join("onnx"))?;
        let blob = format!("{stem}.openvino-compiled.blob");
        let raw = output.join("raw.blob");
        native.compile(&graph, &raw)?;
        pack_blob(&raw, &output.join("onnx").join(&blob))?;
        fs::remove_file(raw)?;
        fs::write(
            output
                .join("onnx")
                .join(format!("{stem}.openvino-compiled.onnx")),
            onnx::wrapper(&original, &blob, &native.sdk_version)?,
        )?;
        Ok(())
    })?;
    ready.insert(key, prepared.clone());
    Ok(prepared)
}

fn prepare_with(
    graph: &Path,
    weights: &Path,
    cache: &Path,
    convert: impl FnOnce(&Path, &Path, &str) -> Result<(), GenericError>,
) -> Result<PathBuf, GenericError> {
    let stem = graph
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| matches!(*s, "model_quantized" | "vision_encoder_quantized"))
        .ok_or_else(|| invalid("unsupported OpenVINO Q8 component"))?;
    let identity = format!(
        "{:x}",
        Sha256::digest(format!(
            "{REVISION}\n{}\n{}",
            model_sha256(graph)?,
            model_sha256(weights)?
        ))
    );
    fs::create_dir_all(cache)?;
    let destination = cache.join(&identity);
    let result = destination
        .join("onnx")
        .join(format!("{stem}.openvino-compiled.onnx"));
    // File locks work across Rust callers, Node workers, and separate processes;
    // the OS releases them even if a converter or its parent crashes.
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(cache.join(format!("{identity}.lock")))?;
    lock.lock()?;
    if complete(&destination, stem) {
        return Ok(result);
    }
    if destination.exists() {
        fs::remove_dir_all(&destination)?;
    }
    let temporary = tempfile::tempdir_in(cache)?;
    let source = temporary.path().join("source");
    fs::create_dir_all(source.join("onnx"))?;
    fs::copy(graph, source.join("onnx").join(format!("{stem}.onnx")))?;
    let staged_weights = source.join("onnx").join(format!("{stem}.onnx_data"));
    if fs::hard_link(weights.canonicalize()?, &staged_weights).is_err() {
        fs::copy(weights, staged_weights)?;
    }

    let output = temporary.path().join("output");
    tracing::info!(component = stem, "preparing Intel GPU model on first use");
    convert(&source, &output, stem)?;
    if !complete(&output, stem) {
        return Err(invalid(
            "OpenVINO conversion did not produce all model files",
        ));
    }
    fs::rename(output, destination)?;
    Ok(result)
}

/// ORT 1.29 OpenVINO BinManager v1 container with one native GPU partition.
/// The metadata is BSON; the native payload begins at a 64 KiB boundary.
fn pack_blob(raw: &Path, output: &Path) -> Result<(), GenericError> {
    use std::io::Write;
    fn document(entries: Vec<(u8, &str, Vec<u8>)>) -> Vec<u8> {
        let mut out = vec![0; 4];
        for (kind, key, value) in entries {
            out.push(kind);
            out.extend_from_slice(key.as_bytes());
            out.push(0);
            out.extend(value);
        }
        out.push(0);
        let size = out.len() as i32;
        out[..4].copy_from_slice(&size.to_le_bytes());
        out
    }
    fn text(value: &str) -> Vec<u8> {
        let mut out = ((value.len() + 1) as i32).to_le_bytes().to_vec();
        out.extend_from_slice(value.as_bytes());
        out.push(0);
        out
    }
    let size = fs::metadata(raw)?.len();
    let entry = document(vec![
        (18, "data_offset", 65536_i64.to_le_bytes().to_vec()),
        (18, "size", (size as i64).to_le_bytes().to_vec()),
    ]);
    let metadata = document(vec![
        (2, "version", text("1.0.0")),
        (2, "producer", text("dghs-imgutils-rs")),
        (
            3,
            "blob_metadata_map",
            document(vec![(3, "OpenVINO", entry)]),
        ),
    ]);
    let mut file = File::create(output)?;
    for value in [0x4e49425f5045564fu64, 1, 40, 40, metadata.len() as u64] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.write_all(&metadata)?;
    file.write_all(&vec![0; 65536 - 40 - metadata.len()])?;
    std::io::copy(&mut File::open(raw)?, &mut file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn cache_reuses_recovers_and_invalidates() {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join("model_quantized.onnx");
        let weights = dir.path().join("model_quantized.onnx_data");
        let cache = dir.path().join("cache");
        fs::write(&graph, b"graph").unwrap();
        fs::write(&weights, b"weights").unwrap();
        let calls = AtomicUsize::new(0);
        let convert = |_: &Path, output: &Path, stem: &str| {
            calls.fetch_add(1, Ordering::SeqCst);
            fs::create_dir_all(output.join("onnx"))?;
            for ext in ["onnx", "blob"] {
                fs::write(
                    output
                        .join("onnx")
                        .join(format!("{stem}.openvino-compiled.{ext}")),
                    b"data",
                )?;
            }
            Ok(())
        };
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| prepare_with(&graph, &weights, &cache, convert).unwrap());
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let first = prepare_with(&graph, &weights, &cache, convert).unwrap();
        fs::remove_file(first.with_extension("blob")).unwrap();
        assert_eq!(
            prepare_with(&graph, &weights, &cache, convert).unwrap(),
            first
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        fs::write(&weights, b"changed weights").unwrap();
        assert_ne!(
            prepare_with(&graph, &weights, &cache, convert).unwrap(),
            first
        );
        fs::write(&graph, b"changed graph").unwrap();
        assert!(prepare_with(&graph, &weights, &cache, |_, _, _| Err(invalid("failed"))).is_err());
        assert!(prepare_with(&graph, &weights, &cache, |_, _, _| Ok(())).is_err());
        assert!(
            prepare_with(&graph, &weights, &cache, convert)
                .unwrap()
                .is_file()
        );
    }
}
