// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Calls the NVTX C library (`libnvToolsExt.so`) directly via FFI so that
//! annotation names can be runtime `&str` values (not just string literals).
//!
//! # Gating (two-level)
//!
//! | Cargo feature `nvtx` | `DYN_ENABLE_RUST_NVTX` env | Effect                                    |
//! |----------------------|-----------------------|-------------------------------------------|
//! | off (default)        | any                   | macros compile to nothing; zero overhead  |
//! | on                   | unset                 | one `Relaxed` load per site (~1 ns)       |
//! | on                   | `1` / `true` / `yes`  | NVTX FFI calls (~50 ns/annotation)        |
//!
//! # Usage
//!
//! ```rust,ignore
//! let _r = dynamo_nvtx_range!("preprocess.tokenize"); // RAII — pops at scope end
//! dynamo_nvtx_push!("codec.encode");
//! dynamo_nvtx_pop!();
//! dynamo_nvtx_name_thread!("tokio-worker-0");
//! ```
//!
//! # Build
//!
//! ```bash
//! cargo build --profile profiling --features nvtx
//! ```
//! Requires `libnvToolsExt.so` at runtime (CUDA Toolkit or NVHPC).

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

// ── FFI declarations (feature-gated) ────────────────────────────────────────

#[cfg(feature = "nvtx")]
mod ffi {
    use std::ffi::c_char;

    #[link(name = "nvToolsExt")]
    unsafe extern "C" {
        pub fn nvtxRangePushA(message: *const c_char) -> std::ffi::c_int;
        pub fn nvtxRangePop() -> std::ffi::c_int;
        pub fn nvtxNameOsThreadA(thread_id: u32, name: *const c_char);
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Initialise the NVTX subsystem from the `DYN_ENABLE_RUST_NVTX` environment variable.
/// Must be called once at runtime startup before any annotation macros fire.
/// No-op when the `nvtx` Cargo feature is off.
pub fn init() {
    #[cfg(feature = "nvtx")]
    {
        let enabled = std::env::var("DYN_ENABLE_RUST_NVTX")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        NVTX_ENABLED.store(enabled, Ordering::Relaxed);
        if enabled {
            tracing::info!("NVTX annotations enabled (DYN_ENABLE_RUST_NVTX)");
        }
    }
}

/// Returns `true` when the `nvtx` feature is compiled in **and** `DYN_ENABLE_RUST_NVTX` is set.
#[inline(always)]
pub fn enabled() -> bool {
    #[cfg(feature = "nvtx")]
    {
        return NVTX_ENABLED.load(Ordering::Relaxed);
    }
    #[allow(unreachable_code)]
    false
}

/// Push an NVTX range onto the calling thread's stack.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn push_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            if let Ok(cstr) = std::ffi::CString::new(name) {
                unsafe { ffi::nvtxRangePushA(cstr.as_ptr()) };
            }
        }
    }
    let _ = name;
}

/// Pop the innermost NVTX range from the calling thread's stack.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn pop_impl() {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            unsafe { ffi::nvtxRangePop() };
        }
    }
}

/// Name the current OS thread in the Nsight Systems timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn name_current_thread_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            // libnvToolsExt expects the native OS thread id (u32 on Linux).
            #[cfg(target_os = "linux")]
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as u32 };
            #[cfg(not(target_os = "linux"))]
            let tid = 0u32;
            if let Ok(cstr) = std::ffi::CString::new(name) {
                unsafe { ffi::nvtxNameOsThreadA(tid, cstr.as_ptr()) };
            }
        }
    }
    let _ = name;
}

// ── RAII guard ───────────────────────────────────────────────────────────────

/// RAII guard that pops an NVTX range when dropped.
/// Construct with [`dynamo_nvtx_range!`].
#[cfg(feature = "nvtx")]
pub struct NvtxRangeGuard {
    active: bool,
}

/// Zero-sized no-op guard used when the `nvtx` feature is off.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRangeGuard;

impl NvtxRangeGuard {
    #[doc(hidden)]
    pub fn new(name: &str) -> Self {
        #[cfg(feature = "nvtx")]
        {
            let active = NVTX_ENABLED.load(Ordering::Relaxed);
            if active {
                if let Ok(cstr) = std::ffi::CString::new(name) {
                    unsafe { ffi::nvtxRangePushA(cstr.as_ptr()) };
                }
            }
            return NvtxRangeGuard { active };
        }
        // Feature off: suppress unused-variable warning and return unit-like guard
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = name;
            NvtxRangeGuard {}
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        if self.active {
            unsafe { ffi::nvtxRangePop() };
        }
    }
}

#[cfg(not(feature = "nvtx"))]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {}
}

// ── Macros ───────────────────────────────────────────────────────────────────

/// Push a named NVTX range onto the calling thread's stack.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_push {
    ($name:expr) => {
        $crate::nvtx::push_impl($name)
    };
}

/// Pop the innermost NVTX range from the calling thread's stack.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_pop {
    () => {
        $crate::nvtx::pop_impl()
    };
}

/// Open a named NVTX range that closes automatically at end of scope.
///
/// ```rust,ignore
/// let _r = dynamo_nvtx_range!("preprocess.tokenize");
/// // range closes here
/// ```
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_range {
    ($name:expr) => {
        $crate::nvtx::NvtxRangeGuard::new($name)
    };
}

/// Annotate the current OS thread in the Nsight Systems timeline.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_name_thread {
    ($name:expr) => {
        $crate::nvtx::name_current_thread_impl($name)
    };
}
