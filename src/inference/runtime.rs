//! Runtime dependency setup for dynamically loaded execution providers.
//!
//! Some Linux ONNX Runtime CUDA builds resolve cuDNN symbols from the global
//! loader scope. `ort`'s generic preload helper intentionally uses a local
//! handle, so the provider can report as available and still fail when a
//! session is created. The small preload below mirrors `LD_PRELOAD` for the
//! standard system/CUDA installation paths.

/// Prepares provider dependencies before the first ONNX Runtime API call.
pub(crate) fn prepare() {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        use once_cell::sync::OnceCell;

        static PREPARED: OnceCell<()> = OnceCell::new();
        PREPARED.get_or_init(prepare_linux);
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn prepare_linux() {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};
    use std::mem::ManuallyDrop;
    use std::path::Path;
    use tracing::{debug, info};

    let mut loaded = 0;
    for library_name in CUDA_LIBRARIES.iter().chain(CUDNN_LIBRARIES) {
        let Some(path) = TRUSTED_LIBRARY_DIRECTORIES
            .iter()
            .map(|directory| Path::new(directory).join(library_name))
            .find(|path| path.is_file())
        else {
            continue;
        };

        // SAFETY: only known vendor library filenames from standard system
        // installation directories are opened. Handles are intentionally
        // leaked so ONNX Runtime can safely use their symbols for the whole
        // process lifetime, matching the semantics of LD_PRELOAD.
        match unsafe { Library::open(Some(&path), RTLD_LAZY | RTLD_GLOBAL) } {
            Ok(library) => {
                let _ = ManuallyDrop::new(library);
                loaded += 1;
                debug!(path = %path.display(), "[ort] preloaded CUDA dependency");
            }
            Err(error) => {
                debug!(
                    path = %path.display(),
                    error = %error,
                    "[ort] CUDA dependency preload failed"
                );
            }
        }
    }

    if loaded > 0 {
        info!(count = loaded, "[ort] preloaded CUDA/cuDNN dependencies");
    }
}

#[cfg(all(test, target_os = "linux", feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn known_cuda_abis_are_covered() {
        assert!(CUDA_LIBRARIES.contains(&"libcudart.so.13"));
        assert!(CUDA_LIBRARIES.contains(&"libcudart.so.12"));
        assert!(CUDNN_LIBRARIES.contains(&"libcudnn.so.9"));
        assert!(CUDNN_LIBRARIES.contains(&"libcudnn.so.8"));
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const TRUSTED_LIBRARY_DIRECTORIES: &[&str] = &[
    "/opt/cuda/lib64",
    "/opt/cuda/lib",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib",
    "/usr/local/lib",
];

#[cfg(all(target_os = "linux", feature = "cuda"))]
const CUDA_LIBRARIES: &[&str] = &[
    "libcudart.so.13",
    "libcudart.so.12",
    "libcublasLt.so.13",
    "libcublasLt.so.12",
    "libcublas.so.13",
    "libcublas.so.12",
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libnccl.so.2",
];

#[cfg(all(target_os = "linux", feature = "cuda"))]
const CUDNN_LIBRARIES: &[&str] = &[
    "libcudnn.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn.so.8",
    "libcudnn_cnn_infer.so.8",
    "libcudnn_cnn_train.so.8",
    "libcudnn_ops_infer.so.8",
    "libcudnn_ops_train.so.8",
    "libcudnn_adv_infer.so.8",
    "libcudnn_adv_train.so.8",
];
