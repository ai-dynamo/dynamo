// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SYCL vectorized_copy kernel FFI bindings.
//!
//! Runtime-loads `libvectorized_copy_sycl.so` and provides a safe Rust
//! wrapper around the C FFI defined in `sycl/vectorized_copy_ffi.h`.
//!
//! The shared library is built separately:
//! ```sh
//! cd lib/kvbm-kernels/sycl && make
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use kvbm_kernels::sycl_vectorized_copy::SyclVectorizedCopy;
//!
//! // `ze_context` and `ze_device` are raw Level Zero handles from syclrc.
//! let svc = SyclVectorizedCopy::new(ze_context, ze_device)?;
//!
//! // src_addrs_dev / dst_addrs_dev are device pointers to u64 arrays,
//! // already uploaded by the caller via BCS memcpy.
//! svc.run(src_addrs_dev, dst_addrs_dev, copy_size, num_pairs)?;
//! ```

use std::ffi::c_void;
use std::sync::OnceLock;

/// Default library search paths (in order).
const LIB_SEARCH_PATHS: &[&str] = &[
    "libvectorized_copy_sycl.so",
    // Relative to kvbm-kernels crate root:
    "sycl/libvectorized_copy_sycl.so",
    "lib/kvbm-kernels/sycl/libvectorized_copy_sycl.so",
];

/// Override library path via environment variable.
const LIB_ENV_VAR: &str = "SYCL_VC_LIB_PATH";

// ---------------------------------------------------------------------------
// Raw FFI types
// ---------------------------------------------------------------------------

/// Opaque handle returned by `sycl_vc_init`.
#[repr(C)]
struct SyclVcState {
    _opaque: [u8; 0],
}

/// Loaded function pointers from the shared library.
struct SyclVcLib {
    _lib: libloading::Library,
    init: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut SyclVcState,
    run: unsafe extern "C" fn(*mut SyclVcState, u64, u64, u64, i32) -> i32,
    destroy: unsafe extern "C" fn(*mut SyclVcState),
}

// Safety: the C library is thread-safe (sycl::queue is internally synchronized).
unsafe impl Send for SyclVcLib {}
unsafe impl Sync for SyclVcLib {}

/// Load the library once (process-wide).
fn load_lib() -> Result<&'static SyclVcLib, String> {
    static LIB: OnceLock<Result<SyclVcLib, String>> = OnceLock::new();
    let result = LIB.get_or_init(|| {
        // Check env override first.
        let paths: Vec<String> = if let Ok(p) = std::env::var(LIB_ENV_VAR) {
            vec![p]
        } else {
            LIB_SEARCH_PATHS.iter().map(|s| s.to_string()).collect()
        };

        let mut last_err = String::new();
        for path in &paths {
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => {
                    let init = unsafe {
                        *lib.get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut SyclVcState>(
                            b"sycl_vc_init\0",
                        )
                        .map_err(|e| format!("symbol sycl_vc_init: {e}"))?
                    };
                    let run = unsafe {
                        *lib.get::<unsafe extern "C" fn(*mut SyclVcState, u64, u64, u64, i32) -> i32>(
                            b"sycl_vc_run\0",
                        )
                        .map_err(|e| format!("symbol sycl_vc_run: {e}"))?
                    };
                    let destroy = unsafe {
                        *lib.get::<unsafe extern "C" fn(*mut SyclVcState)>(
                            b"sycl_vc_destroy\0",
                        )
                        .map_err(|e| format!("symbol sycl_vc_destroy: {e}"))?
                    };
                    return Ok(SyclVcLib { _lib: lib, init, run, destroy });
                }
                Err(e) => {
                    last_err = format!("{path}: {e}");
                }
            }
        }
        Err(format!(
            "Failed to load libvectorized_copy_sycl.so. \
             Set {LIB_ENV_VAR} to override. Last error: {last_err}"
        ))
    });
    result.as_ref().map_err(|e| e.clone())
}

// ---------------------------------------------------------------------------
// Safe wrapper
// ---------------------------------------------------------------------------

/// RAII wrapper around the SYCL vectorized_copy FFI state.
///
/// Holds an opaque `SyclVcState*` and calls `sycl_vc_destroy` on drop.
pub struct SyclVectorizedCopy {
    state: *mut SyclVcState,
    lib: &'static SyclVcLib,
}

// Safety: the underlying sycl::queue is thread-safe.
unsafe impl Send for SyclVectorizedCopy {}
unsafe impl Sync for SyclVectorizedCopy {}

impl SyclVectorizedCopy {
    /// Create a new SYCL vectorized_copy context.
    ///
    /// `ze_context` and `ze_device` are raw Level Zero handles
    /// (e.g., from `ZeDevice::ze_context()` and `ZeDevice::ze_device()`).
    pub fn new(ze_context: *mut c_void, ze_device: *mut c_void) -> Result<Self, String> {
        let lib = load_lib()?;
        let state = unsafe { (lib.init)(ze_context, ze_device) };
        if state.is_null() {
            return Err("sycl_vc_init returned NULL".into());
        }
        Ok(Self { state, lib })
    }

    /// Submit the vectorized_copy kernel and block until completion.
    ///
    /// `src_addrs_dev` / `dst_addrs_dev` are **device** pointers to arrays
    /// of `u64` addresses (already uploaded by the caller via BCS).
    pub fn run(
        &self,
        src_addrs_dev: u64,
        dst_addrs_dev: u64,
        copy_size: u64,
        num_pairs: i32,
    ) -> Result<(), String> {
        let rc = unsafe {
            (self.lib.run)(self.state, src_addrs_dev, dst_addrs_dev, copy_size, num_pairs)
        };
        if rc != 0 {
            Err(format!("sycl_vc_run returned error code {rc}"))
        } else {
            Ok(())
        }
    }
}

impl Drop for SyclVectorizedCopy {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { (self.lib.destroy)(self.state) };
        }
    }
}
