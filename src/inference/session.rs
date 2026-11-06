//! Session identity and model hashing helpers.

use crate::inference::InferenceError;
use crate::inference::backend::SessionOptions;
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ModelFileIdentity {
    canonical_path: PathBuf,
    file_size: u64,
    modified: Option<(u64, u32)>,
}

/// Memoizes content hashes while a model's cheap filesystem identity is unchanged.
static MODEL_HASH_CACHE: Lazy<Mutex<HashMap<ModelFileIdentity, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Identity of a compiled ONNX Runtime session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub model_path: PathBuf,
    pub model_sha256: String,
    pub backend: crate::inference::backend::Backend,
    pub precision: crate::inference::backend::Precision,
    pub device_id: usize,
    pub runtime_fingerprint: String,
    pub provider_fingerprint: String,
}

/// Recognizes prepared OpenVINO EPContext wrappers with adjacent XML and BIN files.
pub(crate) fn is_openvino_ir_wrapper(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        name.as_encoded_bytes().ends_with(b".openvino.onnx")
            || name
                .as_encoded_bytes()
                .ends_with(b".openvino-compiled.onnx")
    })
}

impl SessionKey {
    /// Creates a session identity from the model path and worker options.
    /// Adjacent `<model>.onnx_data` weights participate in the hash, and their
    /// snapshot directory is preserved when the graph itself is a symlink.
    pub fn from_path(
        path: impl AsRef<Path>,
        options: &SessionOptions,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let mut external_name = path.as_os_str().to_os_string();
        external_name.push("_data");
        let weights = PathBuf::from(external_name);
        if weights.is_file() {
            Self::from_model_files(path, options, &[weights])
        } else {
            Self::from_model_files(path, options, &[])
        }
    }

    /// Hashes a prepared OpenVINO wrapper with its required XML and BIN companions.
    pub(crate) fn from_openvino_ir(
        path: &Path,
        options: &SessionOptions,
    ) -> Result<Self, InferenceError> {
        if path.file_name().is_some_and(|name| {
            name.as_encoded_bytes()
                .ends_with(b".openvino-compiled.onnx")
        }) {
            Self::from_model_files(path, options, &[path.with_extension("blob")])
        } else {
            Self::from_model_files(
                path,
                options,
                &[path.with_extension("xml"), path.with_extension("bin")],
            )
        }
    }

    fn from_model_files(
        path: &Path,
        options: &SessionOptions,
        companions: &[PathBuf],
    ) -> Result<Self, InferenceError> {
        let mut hash = model_sha256(path)?;
        let model_path = if companions.is_empty() {
            fs::canonicalize(path)?
        } else {
            for file in companions {
                hash.push('-');
                hash.push_str(&model_sha256(file).map_err(|error| {
                    InferenceError::Initialization(format!(
                        "cannot hash model companion {}: {error}",
                        file.display()
                    ))
                })?);
            }
            hash = format!("{:x}", Sha256::digest(hash.as_bytes()));
            // Resolve the directory only. The graph and its companions can each
            // be symlinks to separate blobs in a Hugging Face snapshot.
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            fs::canonicalize(parent)?.join(path.file_name().ok_or_else(|| {
                InferenceError::Initialization("model path has no filename".to_owned())
            })?)
        };

        Ok(Self {
            model_path,
            model_sha256: hash,
            backend: options.backend,
            precision: options.precision,
            device_id: options.device_id,
            runtime_fingerprint: runtime_fingerprint(),
            provider_fingerprint: options.provider_fingerprint(),
        })
    }

    /// Cache key accepted by the Vitis-AI provider.
    pub fn vitis_cache_key(&self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            self.model_sha256,
            self.precision,
            self.device_id,
            short_fingerprint(&self.runtime_fingerprint),
            short_fingerprint(&self.provider_fingerprint),
        )
    }
}

/// Computes a content hash rather than relying on a mutable model path.
pub fn model_sha256(path: impl AsRef<Path>) -> Result<String, InferenceError> {
    let canonical_path = fs::canonicalize(path.as_ref())?;
    let metadata = fs::metadata(&canonical_path)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs(), duration.subsec_nanos()));
    let identity = ModelFileIdentity {
        canonical_path: canonical_path.clone(),
        file_size: metadata.len(),
        modified,
    };

    if let Some(hash) = MODEL_HASH_CACHE
        .lock()
        .map_err(|error| {
            InferenceError::Initialization(format!("Model hash cache lock poisoned: {error}"))
        })?
        .get(&identity)
        .cloned()
    {
        return Ok(hash);
    }

    let mut file = File::open(&canonical_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    MODEL_HASH_CACHE
        .lock()
        .map_err(|error| {
            InferenceError::Initialization(format!("Model hash cache lock poisoned: {error}"))
        })?
        .insert(identity, hash.clone());

    Ok(hash)
}

/// Identifies the dynamically loaded ONNX Runtime and relevant worker inputs.
pub fn runtime_fingerprint() -> String {
    let dylib = env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "default".to_owned());
    let metadata = fs::metadata(&dylib)
        .ok()
        .map(|metadata| {
            format!(
                "size={};modified={:?}",
                metadata.len(),
                metadata.modified().ok()
            )
        })
        .unwrap_or_else(|| "metadata=unavailable".to_owned());
    format!("ort-dylib={dylib};{metadata}")
}

fn short_fingerprint(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn openvino_ir_companions_are_required_and_invalidate_session_identity() {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join("model.openvino.onnx");
        let xml = graph.with_extension("xml");
        let bin = graph.with_extension("bin");
        fs::write(&graph, b"wrapper").unwrap();
        let options = SessionOptions::default();
        let key = || SessionKey::from_openvino_ir(&graph, &options);
        assert!(
            key()
                .unwrap_err()
                .to_string()
                .contains("model.openvino.xml")
        );
        fs::write(&xml, b"xml").unwrap();
        assert!(
            key()
                .unwrap_err()
                .to_string()
                .contains("model.openvino.bin")
        );
        fs::write(&bin, b"weights").unwrap();
        let first = key().unwrap();
        fs::write(&xml, b"new xml").unwrap();
        let second = key().unwrap();
        assert_ne!(first.model_sha256, second.model_sha256);
        fs::write(&bin, b"new weights").unwrap();
        assert_ne!(second.model_sha256, key().unwrap().model_sha256);
    }

    #[test]
    fn compiled_blob_is_required_and_invalidates_session_identity() {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join("model.openvino-compiled.onnx");
        fs::write(&graph, b"wrapper").unwrap();
        let options = SessionOptions::default();
        assert!(is_openvino_ir_wrapper(&graph));
        assert!(SessionKey::from_openvino_ir(&graph, &options).is_err());
        fs::write(graph.with_extension("blob"), b"blob").unwrap();
        let first = SessionKey::from_openvino_ir(&graph, &options).unwrap();
        fs::write(graph.with_extension("blob"), b"changed blob").unwrap();
        assert_ne!(
            first.model_sha256,
            SessionKey::from_openvino_ir(&graph, &options)
                .unwrap()
                .model_sha256
        );
    }

    #[test]
    fn ordinary_graphs_ignore_adjacent_openvino_ir_files() {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join("model.openvino.onnx");
        fs::write(&graph, b"graph").unwrap();
        let options = SessionOptions::default();
        let original = SessionKey::from_path(&graph, &options).unwrap();
        fs::write(graph.with_extension("xml"), b"unrelated xml").unwrap();
        fs::write(graph.with_extension("bin"), b"unrelated weights").unwrap();
        let key = SessionKey::from_path(&graph, &options).unwrap();
        assert_eq!(key, original);
        assert_eq!(key.model_sha256, model_sha256(&graph).unwrap());
    }

    #[test]
    fn external_weight_changes_invalidate_session_identity() {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join("model.onnx");
        let weights = dir.path().join("model.onnx_data");
        fs::write(&graph, b"graph").unwrap();
        fs::write(&weights, b"first weights").unwrap();
        let options = SessionOptions::default();
        let first = SessionKey::from_path(&graph, &options).unwrap();
        let legacy_hash = format!(
            "{:x}",
            Sha256::digest(format!(
                "{}-{}",
                model_sha256(&graph).unwrap(),
                model_sha256(&weights).unwrap()
            ))
        );
        assert_eq!(first.model_sha256, legacy_hash);
        fs::write(&weights, b"different weights").unwrap();
        let second = SessionKey::from_path(&graph, &options).unwrap();
        assert_ne!(first.model_sha256, second.model_sha256);
        assert_eq!(first.model_sha256.len(), 64);
    }

    #[cfg(unix)]
    #[test]
    fn external_weights_keep_hub_snapshot_directory() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        let blob = dir.path().join("graph-blob");
        fs::write(&blob, b"graph").unwrap();
        let graph = snapshot.join("model.onnx");
        std::os::unix::fs::symlink(&blob, &graph).unwrap();
        fs::write(snapshot.join("model.onnx_data"), b"weights").unwrap();
        let key = SessionKey::from_path(&graph, &SessionOptions::default()).unwrap();
        assert_eq!(key.model_path, graph);
        assert_ne!(key.model_path, blob);
    }

    #[test]
    fn model_hash_changes_when_model_bytes_change() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"first").unwrap();
        let first = model_sha256(file.path()).unwrap();
        file.as_file_mut().set_len(0).unwrap();
        file.write_all(b"second").unwrap();
        let second = model_sha256(file.path()).unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn model_hash_is_reused_for_unchanged_file_identity() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"unchanged").unwrap();

        let first = model_sha256(file.path()).unwrap();
        let second = model_sha256(file.path()).unwrap();

        assert_eq!(first, second);
        let cache = MODEL_HASH_CACHE.lock().unwrap();
        assert!(cache.values().any(|hash| hash == &first));
    }

    #[test]
    fn vitis_cache_key_contains_model_and_runtime_identity() {
        let key = SessionKey {
            model_path: PathBuf::from("model.onnx"),
            model_sha256: "abc".to_owned(),
            backend: crate::inference::backend::Backend::AmdNpu,
            precision: crate::inference::backend::Precision::Bf16,
            device_id: 0,
            runtime_fingerprint: "runtime".to_owned(),
            provider_fingerprint: "provider".to_owned(),
        };

        assert!(key.vitis_cache_key().starts_with("abc-bf16-0-"));
    }
}
