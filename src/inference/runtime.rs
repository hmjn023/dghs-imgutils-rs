//! Runtime dependency preparation and Intel device discovery.
//!
//! `ort` dynamically loads the ONNX Runtime core, while the core dynamically
//! loads execution-provider libraries.  The provider libraries in turn load
//! vendor runtimes such as OpenVINO, Level Zero, and the Intel GPU compiler.
//! Preparing those dependencies here keeps provider setup in the library and
//! lets a consumer use the same native addon without hand-written preload
//! logic.

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
use std::env;
#[cfg(target_os = "linux")]
use std::fs::{self, OpenOptions};
#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
use std::path::PathBuf;

/// Prepares known provider dependencies before the first ONNX Runtime call.
///
/// On Linux, callers that rely on automatic Intel OpenCL ICD selection must
/// run this during single-threaded application startup. The OpenCL loader has
/// no provider-option API for an ICD path, so its process-wide startup policy
/// must be established before worker threads or provider libraries are used.
pub(crate) fn prepare() {
    #[cfg(any(feature = "cuda", feature = "openvino"))]
    {
        use once_cell::sync::OnceCell;

        static PREPARED: OnceCell<()> = OnceCell::new();
        PREPARED.get_or_init(prepare_platform);
    }
}

/// Prepares device-specific dependencies after ONNX Runtime itself has been
/// initialized but before the provider is registered.
#[cfg(feature = "openvino")]
pub(crate) fn prepare_openvino_device(device_type: &str) {
    #[cfg(all(target_os = "linux", feature = "openvino"))]
    if device_type
        .trim()
        .to_ascii_uppercase()
        .split([':', ','])
        .any(|token| token.trim() == "NPU" || token.trim().starts_with("NPU."))
    {
        prepare_linux_npu_dependencies();
    }

    #[cfg(not(all(target_os = "linux", feature = "openvino")))]
    let _ = device_type;
}

/// Returns the OpenVINO device policy used when the caller did not specify one.
///
/// OpenVINO still performs the final device validation.  This lightweight
/// host inspection only orders the devices that are actually exposed by the
/// operating system, so a missing driver or inaccessible device is handled by
/// the provider/session probe rather than being treated as available.
pub(crate) fn default_openvino_device() -> String {
    #[cfg(target_os = "linux")]
    {
        let devices = detected_intel_devices();
        let has_npu = devices.contains(&"NPU");
        let has_gpu = devices.contains(&"GPU");

        match (has_npu, has_gpu) {
            (true, true) => "AUTO:NPU,GPU,CPU".to_owned(),
            (true, false) => "AUTO:NPU,CPU".to_owned(),
            (false, true) => "AUTO:GPU,CPU".to_owned(),
            (false, false) => "AUTO:CPU".to_owned(),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // OpenVINO itself performs device discovery on platforms where this
        // crate cannot use Linux sysfs, such as Windows.
        "AUTO:NPU,GPU,CPU".to_owned()
    }
}

/// Returns the Intel accelerators visible to the current process environment.
pub(crate) fn detected_intel_devices() -> Vec<&'static str> {
    #[cfg(target_os = "linux")]
    {
        let mut devices = Vec::new();
        if has_intel_drm_device() {
            devices.push("GPU");
        }
        if has_intel_accelerator_device() {
            devices.push("NPU");
        }
        devices
    }

    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// Provides a stable summary for diagnostics and provider probe output.
#[cfg(any(feature = "openvino", test))]
pub(crate) fn detected_intel_device_summary() -> String {
    #[cfg(target_os = "linux")]
    {
        let devices = detected_intel_devices();
        if devices.is_empty() {
            "none".to_owned()
        } else {
            devices.join(",")
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        "platform-auto".to_owned()
    }
}

#[cfg(target_os = "linux")]
/// Reports whether an accessible Intel DRM render node is available.
fn has_intel_drm_device() -> bool {
    has_accessible_intel_render_node()
}

#[cfg(target_os = "linux")]
/// Reports whether an accessible Intel accelerator node is exposed by sysfs.
fn has_intel_accelerator_device() -> bool {
    let Ok(entries) = fs::read_dir("/sys/class/accel") else {
        return false;
    };

    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name_string = name.to_string_lossy();
        name_string.starts_with("accel")
            && is_intel_pci_device(&entry.path().join("device"))
            && is_device_node_accessible(Path::new("/dev/accel").join(&name).as_path())
    })
}

#[cfg(target_os = "linux")]
/// Finds an Intel DRM render node that the current process can open read/write.
fn has_accessible_intel_render_node() -> bool {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return false;
    };

    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name_string = name.to_string_lossy();
        name_string.starts_with("renderD")
            && is_intel_pci_device(&entry.path().join("device"))
            && is_device_node_accessible(Path::new("/dev/dri").join(&name).as_path())
    })
}

#[cfg(target_os = "linux")]
/// Checks the read/write access required by the provider runtime for a device
/// node.
fn is_device_node_accessible(path: &Path) -> bool {
    OpenOptions::new().read(true).write(true).open(path).is_ok()
}

#[cfg(target_os = "linux")]
/// Checks the Linux PCI vendor identifier for Intel (`0x8086`).
fn is_intel_pci_device(device_path: &Path) -> bool {
    fs::read_to_string(device_path.join("vendor"))
        .map(|vendor| vendor.trim().eq_ignore_ascii_case("0x8086"))
        .unwrap_or(false)
}

#[cfg(any(feature = "cuda", feature = "openvino"))]
/// Prepares platform-wide provider dependencies before ONNX Runtime is used.
fn prepare_platform() {
    #[cfg(target_os = "linux")]
    {
        #[cfg(feature = "cuda")]
        prepare_linux_libraries(
            "CUDA",
            &[
                "libcudart.so.13",
                "libcudart.so.12",
                "libcudart.so.11",
                "libcublasLt.so.13",
                "libcublasLt.so.12",
                "libcublasLt.so.11",
                "libcublas.so.13",
                "libcublas.so.12",
                "libcublas.so.11",
                "libnvrtc.so.13",
                "libnvrtc.so.12",
                "libnvrtc.so.11",
                "libcurand.so.10",
                "libcufft.so.11",
                "libcufft.so.10",
                "libnccl.so.2",
            ],
            &[
                "libcudart.so.",
                "libcublasLt.so.",
                "libcublas.so.",
                "libnvrtc.so.",
                "libcurand.so.",
                "libcufft.so.",
                "libnccl.so.",
            ],
        );
        #[cfg(feature = "cuda")]
        prepare_linux_libraries(
            "cuDNN",
            &[
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
            ],
            &[
                "libcudnn.so.",
                "libcudnn_cnn.so.",
                "libcudnn_ops.so.",
                "libcudnn_graph.so.",
                "libcudnn_heuristic.so.",
                "libcudnn_adv.so.",
                "libcudnn_engines_precompiled.so.",
                "libcudnn_engines_runtime_compiled.so.",
                "libcudnn_cnn_infer.so.",
                "libcudnn_cnn_train.so.",
                "libcudnn_ops_infer.so.",
                "libcudnn_ops_train.so.",
                "libcudnn_adv_infer.so.",
                "libcudnn_adv_train.so.",
            ],
        );
        #[cfg(feature = "openvino")]
        prepare_linux_openvino();
    }

    #[cfg(target_os = "windows")]
    {
        // Windows searches the directory containing the ONNX Runtime DLL and
        // the process DLL directories for provider dependencies.  The
        // provider-specific setup remains a no-op here; OpenVINO's strict
        // session commit is still the source of truth for availability.
    }
}

#[cfg(all(target_os = "linux", feature = "openvino"))]
/// Preloads the OpenVINO core libraries needed by the execution provider.
fn prepare_linux_openvino() {
    configure_intel_opencl_icd();
    prepare_linux_libraries(
        "OpenVINO core",
        &["libopenvino.so", "libopenvino_c.so", "libtbb.so.12"],
        &["libopenvino.so.", "libopenvino_c.so.", "libtbb.so."],
    );

    // Do not preload the OpenVINO device plugins, vendor compiler/driver
    // libraries, or the ORT provider itself. OpenVINO/ORT must load those
    // through their normal provider boundary; a RTLD_GLOBAL preload can
    // change private symbol resolution and abort the process when a GPU/NPU
    // plugin initializes.
}

#[cfg(all(target_os = "linux", feature = "openvino", not(test)))]
/// Selects Intel's standard OpenCL ICD during the documented single-threaded
/// startup phase while preserving an application-provided loader policy.
fn configure_intel_opencl_icd() {
    use tracing::info;

    // Respect an application/container policy when it has already selected an
    // ICD. OCL_ICD_FILENAMES is also honored because the ocl-icd loader gives
    // it precedence over the vendor directory scan.
    if env::var_os("OCL_ICD_VENDORS").is_some()
        || env::var_os("OCL_ICD_FILENAMES").is_some()
        || !has_accessible_intel_render_node()
    {
        return;
    }

    for path in [
        "/etc/OpenCL/vendors/intel.icd",
        "/usr/share/OpenCL/vendors/intel.icd",
    ] {
        let path = Path::new(path);
        if path.is_file() {
            // SAFETY: `prepare` is the process startup initialization hook and
            // runs once before ONNX Runtime/provider loading. Applications
            // must call `init_onnx_runtime` or `configure_inference` before
            // starting worker threads when relying on this automatic ICD
            // selection. Existing loader policy is preserved above.
            unsafe { env::set_var("OCL_ICD_VENDORS", path) };
            info!(path = %path.display(), "[ort] selected Intel OpenCL ICD");
            break;
        }
    }
}

#[cfg(all(target_os = "linux", feature = "openvino", test))]
fn configure_intel_opencl_icd() {}

#[cfg(all(target_os = "linux", feature = "openvino"))]
/// Makes Intel's Level Zero libraries discoverable by the OpenVINO NPU plugin.
fn prepare_linux_npu_dependencies() {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};
    use once_cell::sync::OnceCell;
    use tracing::{debug, info};

    static PREPARED: OnceCell<()> = OnceCell::new();
    PREPARED.get_or_init(|| {
        if !has_intel_accelerator_device() {
            return;
        }

        let Some(path) = find_library("libze_intel_npu.so.1", &["libze_intel_npu.so."])
        else {
            debug!("[ort] Intel NPU Level Zero runtime was not found");
            return;
        };

        let Some(loader) = find_library("libze_loader.so.1", &["libze_loader.so."]) else {
            debug!("[ort] Intel NPU Level Zero loader was not found");
            return;
        };
        let Some(tracing) =
            find_library("libze_tracing_layer.so.1", &["libze_tracing_layer.so."])
        else {
            debug!("[ort] Intel Level Zero tracing layer was not found");
            return;
        };

        // OpenVINO's NPU plugin resolves these three libraries by SONAME. Load
        // the same sequence explicitly so the later name-based dlopen can reuse
        // the handles even when the distro keeps the NPU driver in a multiarch
        // directory absent from the process's startup search path.
        for (label, library, flags) in [
            ("Level Zero loader", loader, RTLD_LAZY | RTLD_GLOBAL),
            ("Level Zero tracing layer", tracing, RTLD_LAZY | RTLD_GLOBAL),
            ("Intel NPU Level Zero runtime", path, RTLD_LAZY),
        ] {
            // SAFETY: paths are selected from explicit runtime configuration or
            // standard system/provider directories, and only known Intel runtime
            // library names are opened. Handles stay alive for the process because
            // OpenVINO may resolve them again during provider initialization.
            match unsafe { Library::open(Some(&library), flags) } {
                Ok(handle) => {
                    std::mem::forget(handle);
                    info!(label, path = %library.display(), "[ort] prepared Intel NPU dependency");
                }
                Err(error) => {
                    debug!(label, path = %library.display(), error = %error, "[ort] Intel NPU dependency preload failed");
                    return;
                }
            }
        }
    });
}

/// Rejects strict Intel device policies that the current process cannot open.
///
/// Sysfs presence alone is not enough: Linux commonly exposes `/dev/accel/*`
/// as `root:render` and a service user may see the NPU in sysfs without having
/// permission to initialize it.  AUTO policies intentionally remain allowed
/// to use their CPU fallback.
#[cfg(any(feature = "openvino", test))]
pub(crate) fn validate_openvino_device_policy(device_type: &str) -> Result<(), String> {
    #[cfg(not(target_os = "linux"))]
    {
        // Windows and macOS do not expose the Linux device nodes used by this
        // lightweight preflight. OpenVINO remains the source of truth there.
        let _ = device_type;
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        let normalized = device_type.trim().to_ascii_uppercase();
        if normalized.starts_with("AUTO") {
            return Ok(());
        }

        let devices = detected_intel_devices();
        for requested in ["GPU", "NPU"] {
            let requested_by_policy = normalized == requested
                || normalized.starts_with(&format!("{requested}."))
                || normalized.replace(':', ",").split(',').any(|token| {
                    token.trim() == requested || token.trim().starts_with(&format!("{requested}."))
                });
            if requested_by_policy && !devices.contains(&requested) {
                let device_summary = if devices.is_empty() {
                    "none".to_owned()
                } else {
                    devices.join(",")
                };
                return Err(format!(
                    "Intel {requested} is not accessible to this process (detected accessible devices: {device_summary})"
                ));
            }
        }

        Ok(())
    }
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Preloads known provider libraries from the runtime search path so ORT can
/// resolve them even when the host linker path does not include their folder.
fn prepare_linux_libraries(label: &str, exact_names: &[&str], prefixes: &[&str]) {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};
    use std::collections::HashSet;
    use tracing::{debug, info};

    let mut prepared = false;
    let mut loaded_families = HashSet::new();
    for name in exact_names {
        let family = matching_prefixes(name, prefixes).next().unwrap_or(name);
        if loaded_families.contains(family) {
            continue;
        }

        let Some(path) = find_library(name, prefixes) else {
            continue;
        };

        // SAFETY: paths are selected from explicit runtime configuration or
        // standard system/provider directories, and only known vendor/ORT
        // library names are opened.  The handle is intentionally leaked so
        // later ONNX Runtime/provider loads can resolve the global symbols.
        match unsafe { Library::open(Some(&path), RTLD_LAZY | RTLD_GLOBAL) } {
            Ok(library) => {
                prepared = true;
                loaded_families.insert(family);
                std::mem::forget(library);
                debug!(label, path = %path.display(), "[ort] preloaded runtime dependency");
            }
            Err(error) => {
                debug!(
                    label,
                    path = %path.display(),
                    error = %error,
                    "[ort] runtime dependency preload failed"
                );
            }
        }
    }

    if prepared {
        info!(label, "[ort] runtime dependency preparation completed");
    }
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Finds an exact library or the newest versioned library in the configured
/// runtime directories.
fn find_library(name: &str, prefixes: &[&str]) -> Option<PathBuf> {
    let directories = runtime_search_directories();

    for directory in &directories {
        let exact = directory.join(name);
        if exact.is_file() {
            return Some(exact);
        }
    }

    // `prefixes` contains one version prefix for each library family.  Only
    // use the prefix belonging to this exact library name; otherwise looking
    // up `libopenvino_c.so` could accidentally return `libopenvino.so.*`.
    for prefix in matching_prefixes(name, prefixes) {
        for directory in &directories {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            let matches = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|file| file.to_str())
                        .is_some_and(|file| file.starts_with(prefix))
                })
                .collect::<Vec<_>>();
            if let Some(path) = select_highest_versioned_library(matches, prefix) {
                return Some(path);
            }
        }
    }

    None
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Selects the highest numeric SONAME suffix rather than relying on
/// lexicographic path ordering (`.12` must win over `.9`).
fn select_highest_versioned_library(mut matches: Vec<PathBuf>, prefix: &str) -> Option<PathBuf> {
    matches.sort_by_key(|path| {
        let version = path
            .file_name()
            .and_then(|file| file.to_str())
            .and_then(|file| file.strip_prefix(prefix))
            .map(|suffix| {
                suffix
                    .split('.')
                    .map_while(|component| component.parse::<u64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (version, path.clone())
    });
    matches.into_iter().next_back()
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Returns the library-family prefixes that match an exact library name.
fn matching_prefixes<'a>(name: &str, prefixes: &'a [&str]) -> impl Iterator<Item = &'a str> {
    prefixes
        .iter()
        .copied()
        .filter(move |prefix| name.starts_with(prefix.trim_end_matches('.')))
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Builds the ordered library search path used by provider preparation.
fn runtime_search_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();

    if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            // Prefer the directory beside the loaded ORT core so that its
            // provider and vendor libraries cannot be mixed with another
            // installation found earlier in the process environment.
            push_unique(&mut directories, parent.to_path_buf());
        }
    }

    for variable in [
        "DGHS_OPENVINO_LIBRARY_PATH",
        "OPENVINO_LIB_PATH",
        "ORT_PROVIDER_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ] {
        append_env_paths(&mut directories, variable);
    }

    for directory in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/lib/openvino",
        "/opt/intel/openvino/runtime/lib/intel64",
        "/opt/openvino/runtime/lib/intel64",
        "/opt/openvino/runtime/lib/intel64/Release",
    ] {
        push_unique(&mut directories, PathBuf::from(directory));
    }

    directories
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Appends existing directories referenced by a path-list environment value.
fn append_env_paths(directories: &mut Vec<PathBuf>, variable: &str) {
    let Some(value) = env::var_os(variable) else {
        return;
    };

    for path in env::split_paths(&value) {
        if path.is_dir() {
            push_unique(directories, path);
        } else if path.is_file()
            && let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            push_unique(directories, parent.to_path_buf());
        }
    }
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
/// Adds a directory once while preserving search priority.
fn push_unique(directories: &mut Vec<PathBuf>, path: PathBuf) {
    if !directories.iter().any(|existing| existing == &path) {
        directories.push(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_is_idempotent() {
        prepare();
        prepare();
    }

    #[test]
    fn default_policy_contains_cpu_fallback() {
        assert!(
            default_openvino_device()
                .to_ascii_uppercase()
                .contains("CPU")
        );
    }

    #[test]
    fn device_summary_is_stable() {
        let summary = detected_intel_device_summary();
        assert!(!summary.is_empty());
    }

    #[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
    #[test]
    fn library_version_prefixes_stay_in_their_family() {
        let prefixes = ["libopenvino.so.", "libopenvino_c.so."];

        assert_eq!(
            matching_prefixes("libopenvino_c.so", &prefixes).collect::<Vec<_>>(),
            vec!["libopenvino_c.so."]
        );
    }

    #[cfg(all(target_os = "linux", any(feature = "cuda", feature = "openvino")))]
    #[test]
    fn newest_library_version_wins_numeric_order() {
        let selected = select_highest_versioned_library(
            vec![
                PathBuf::from("/tmp/libcudart.so.9"),
                PathBuf::from("/tmp/libcudart.so.12"),
                PathBuf::from("/tmp/libcudart.so.11"),
            ],
            "libcudart.so.",
        );

        assert_eq!(selected, Some(PathBuf::from("/tmp/libcudart.so.12")));
    }

    #[test]
    fn auto_policy_does_not_require_unavailable_accelerators() {
        assert!(validate_openvino_device_policy("AUTO:NPU,GPU,CPU").is_ok());
    }
}
