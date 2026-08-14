//! Session identity and model hashing helpers.

use crate::inference::InferenceError;
use crate::inference::backend::SessionOptions;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

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

impl SessionKey {
    pub fn from_path(
        path: impl AsRef<Path>,
        options: &SessionOptions,
    ) -> Result<Self, InferenceError> {
        let model_path = fs::canonicalize(path.as_ref())?;
        let model_sha256 = model_sha256(&model_path)?;

        Ok(Self {
            model_path,
            model_sha256,
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
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
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
