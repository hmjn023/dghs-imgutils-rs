//! Dynamic OpenVINO C API binding used only during automatic model preparation.
use super::{GenericError, invalid};
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;

type Handle = *mut c_void;
type Free = unsafe extern "C" fn(Handle);
struct Owned {
    ptr: Handle,
    free: Free,
}
impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: Each handle was created by this library and is freed exactly once.
        unsafe { (self.free)(self.ptr) }
    }
}
#[repr(C)]
struct Version {
    build: *const c_char,
    description: *const c_char,
}

pub(super) struct Native {
    core: Owned,
    library: Library,
    device: CString,
    precision: CString,
    pub identity: String,
    pub sdk_version: String,
}
fn string(value: &str) -> Result<CString, GenericError> {
    CString::new(value).map_err(|_| invalid("OpenVINO path or option contains NUL"))
}
// All function signatures below are from the public OpenVINO C headers.
// Symbols and handles never outlive the owning Library.
impl Native {
    /// Opens the installed C runtime and identifies the target device.
    pub fn new(device: &str, precision: &str) -> Result<Self, GenericError> {
        let names = if cfg!(target_os = "windows") {
            vec!["openvino_c.dll"]
        } else if cfg!(target_os = "macos") {
            vec!["libopenvino_c.dylib"]
        } else {
            vec!["libopenvino_c.so", "libopenvino_c.so.2641"]
        };
        let mut candidates = Vec::new();
        if let Some(path) = std::env::var_os("ORT_DYLIB_PATH")
            && let Some(parent) = Path::new(&path).parent()
        {
            candidates.extend(names.iter().map(|n| parent.join(n)));
        }
        candidates.extend(names.iter().map(std::path::PathBuf::from));
        let device_c = string(device)?;
        let precision_c = string(precision)?;
        // SAFETY: Load the installed OpenVINO C runtime, resolving only documented C symbols.
        let library=unsafe { candidates.iter().find_map(|p|Library::new(p).ok()) }
            .ok_or_else(||invalid("automatic Intel GPU preparation requires the OpenVINO C runtime (openvino_c); Python is not required"))?;
        // SAFETY: C ABI signatures match ov_core.h; allocated strings/handles are freed by their owner.
        unsafe {
            let create = library
                .get::<unsafe extern "C" fn(*mut Handle) -> i32>(b"ov_core_create\0")
                .map_err(|e| invalid(e.to_string()))?;
            let free = *library
                .get::<Free>(b"ov_core_free\0")
                .map_err(|e| invalid(e.to_string()))?;
            let mut ptr = std::ptr::null_mut();
            if create(&mut ptr) != 0 {
                return Err(invalid("OpenVINO core creation failed"));
            }
            let core = Owned { ptr, free };
            let mut v = Version {
                build: std::ptr::null(),
                description: std::ptr::null(),
            };
            let get = library
                .get::<unsafe extern "C" fn(*mut Version) -> i32>(b"ov_get_openvino_version\0")
                .map_err(|e| invalid(e.to_string()))?;
            let vf = library
                .get::<unsafe extern "C" fn(*mut Version)>(b"ov_version_free\0")
                .map_err(|e| invalid(e.to_string()))?;
            if get(&mut v) != 0 {
                return Err(invalid("OpenVINO version unavailable"));
            }
            let version = CStr::from_ptr(v.build).to_string_lossy().into_owned();
            vf(&mut v);
            let mut native = Self {
                core,
                library,
                device: device_c,
                precision: precision_c,
                sdk_version: version.split('.').take(2).collect::<Vec<_>>().join("."),
                identity: version,
            };
            for property in ["FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "DEVICE_UUID"] {
                native.identity.push_str(&native.property(property)?);
            }
            native.identity.push_str(ort::info());
            // Linux GPU binaries can change without an OpenVINO version change.
            // Include the loaded driver/plugin file identities in the disk cache key.
            #[cfg(target_os = "linux")]
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                let paths: std::collections::BTreeSet<_> = maps
                    .lines()
                    .filter_map(|line| line.split_whitespace().last())
                    .filter(|path| {
                        path.starts_with('/')
                            && ["libigdrcl", "libze_intel_gpu", "openvino_intel_gpu"]
                                .iter()
                                .any(|name| path.contains(name))
                    })
                    .collect();
                for path in paths {
                    if let Ok(metadata) = std::fs::metadata(path) {
                        native.identity.push_str(&format!(
                            "|{path}:{}:{:?}",
                            metadata.len(),
                            metadata.modified().ok()
                        ));
                    }
                }
            }
            native.identity.push_str(device);
            native.identity.push_str(precision);
            Ok(native)
        }
    }
    fn check(&self, status: i32) -> Result<(), GenericError> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: Error text is borrowed from OpenVINO and copied while the library is alive.
        unsafe {
            let error = self
                .library
                .get::<unsafe extern "C" fn() -> *const c_char>(b"ov_get_last_err_msg\0")
                .map_err(|e| invalid(e.to_string()))?;
            let ptr = error();
            Err(invalid(format!(
                "OpenVINO native preparation failed ({status}): {}",
                if ptr.is_null() {
                    "unknown error".into()
                } else {
                    CStr::from_ptr(ptr).to_string_lossy()
                }
            )))
        }
    }
    fn property(&self, name: &str) -> Result<String, GenericError> {
        // SAFETY: Output is an OpenVINO-owned string, released with ov_free after copying.
        unsafe {
            let get =
                self.library
                    .get::<unsafe extern "C" fn(
                        Handle,
                        *const c_char,
                        *const c_char,
                        *mut *mut c_char,
                    ) -> i32>(b"ov_core_get_property\0")
                    .map_err(|e| invalid(e.to_string()))?;
            let free = self
                .library
                .get::<unsafe extern "C" fn(*const c_char)>(b"ov_free\0")
                .map_err(|e| invalid(e.to_string()))?;
            let mut ptr = std::ptr::null_mut();
            self.check(get(
                self.core.ptr,
                self.device.as_ptr(),
                string(name)?.as_ptr(),
                &mut ptr,
            ))?;
            let value = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            free(ptr);
            Ok(value)
        }
    }
    /// Compiles the corrected ONNX graph and exports the native GPU executable.
    pub fn compile(&self, graph: &Path, blob: &Path) -> Result<(), GenericError> {
        let graph = string(
            graph
                .to_str()
                .ok_or_else(|| invalid("OpenVINO model path must be UTF-8"))?,
        )?;
        let blob = string(
            blob.to_str()
                .ok_or_else(|| invalid("OpenVINO cache path must be UTF-8"))?,
        )?;
        // SAFETY: C ABI signatures match the public headers. Model and compiled handles
        // are RAII-owned and destroyed before self.library; all input strings stay alive.
        unsafe {
            let read=self.library.get::<unsafe extern "C" fn(Handle,*const c_char,*const c_char,*mut Handle)->i32>(b"ov_core_read_model\0").map_err(|e|invalid(e.to_string()))?;
            let model_free = *self
                .library
                .get::<Free>(b"ov_model_free\0")
                .map_err(|e| invalid(e.to_string()))?;
            let compiled_free = *self
                .library
                .get::<Free>(b"ov_compiled_model_free\0")
                .map_err(|e| invalid(e.to_string()))?;
            let compile =
                self.library
                    .get::<unsafe extern "C" fn(
                        Handle,
                        Handle,
                        *const c_char,
                        usize,
                        *mut Handle,
                        ...
                    ) -> i32>(b"ov_core_compile_model\0")
                    .map_err(|e| invalid(e.to_string()))?;
            let export = self
                .library
                .get::<unsafe extern "C" fn(Handle, *const c_char) -> i32>(
                    b"ov_compiled_model_export_model\0",
                )
                .map_err(|e| invalid(e.to_string()))?;
            let mut ptr = std::ptr::null_mut();
            self.check(read(
                self.core.ptr,
                graph.as_ptr(),
                std::ptr::null(),
                &mut ptr,
            ))?;
            let model = Owned {
                ptr,
                free: model_free,
            };
            let mut ptr = std::ptr::null_mut();
            self.check(compile(
                self.core.ptr,
                model.ptr,
                self.device.as_ptr(),
                2,
                &mut ptr,
                c"INFERENCE_PRECISION_HINT".as_ptr(),
                self.precision.as_ptr(),
            ))?;
            let compiled = Owned {
                ptr,
                free: compiled_free,
            };
            self.check(export(compiled.ptr, blob.as_ptr()))
        }
    }
}
