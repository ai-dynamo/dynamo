// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Delegates to the [`nvtx`](https://docs.rs/nvtx) crate for the actual NVTX
//! calls. That crate compiles the vendored, header-only **NVTX v3** headers
//! (`nvtx3/nvToolsExt.h`), so there is no link against the legacy
//! `libnvToolsExt.so` — which CUDA 12.9+ / CUDA 13 removed.
//!
//! # Gating (two-level)
//!
//! | Cargo feature `nvtx` | `DYN_ENABLE_RUST_NVTX` env | Effect                                    |
//! |----------------------|----------------------------|-------------------------------------------|
//! | off (default)        | any                        | macros compile to nothing; zero overhead  |
//! | on                   | unset                      | one `Relaxed` load per site (~1 ns)       |
//! | on                   | `1` / `true` / `yes`       | NVTX v3 calls (~50 ns/annotation)         |
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
//! NVTX v3 is header-only, so no shared library is required at runtime; the
//! profiler (e.g. Nsight Systems) injects the implementation when it attaches.

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

// ── Public API ───────────────────────────────────────────────────────────────

/// Initialise the NVTX subsystem from the `DYN_ENABLE_RUST_NVTX` environment variable.
/// Must be called once at runtime startup before any annotation macros fire.
/// No-op when the `nvtx` Cargo feature is off.
pub fn init() {
    #[cfg(feature = "nvtx")]
    {
        let enabled = crate::config::env_is_truthy("DYN_ENABLE_RUST_NVTX");
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
            nvtx::range_push!("{name}");
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
            nvtx::range_pop!();
        }
    }
}

/// Open a **correlated** NVTX range and return an opaque id to pass to
/// [`range_end_impl`].
///
/// Unlike [`push_impl`]/[`pop_impl`], a correlated range is *not* tied to the
/// calling thread's push/pop stack, so it may be opened and closed across
/// `await` points, on different threads, and while other ranges interleave —
/// which is exactly what async callers (e.g. the Python workers, whose
/// coroutines interleave on one event-loop thread) need. Prefer the RAII
/// [`NvtxRangeGuard`] / push-pop pair for synchronous scopes.
///
/// Returns `0` when NVTX is disabled (feature off, or `DYN_ENABLE_RUST_NVTX`
/// unset); [`range_end_impl`] treats `0` as "no range".
#[inline(always)]
pub fn range_start_impl(name: &str) -> i64 {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            return i64::from(nvtx::range_start!("{name}"));
        }
    }
    let _ = name;
    0
}

/// Close a correlated NVTX range previously opened by [`range_start_impl`].
/// No-op for id `0`.
#[inline(always)]
pub fn range_end_impl(id: i64) {
    #[cfg(feature = "nvtx")]
    {
        if id != 0 {
            let range_id = id as i32;
            nvtx::range_end!(range_id);
        }
    }
    let _ = id;
}

/// Name the current OS thread in the Nsight Systems timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn name_current_thread_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            // The `nvtx` crate resolves the current OS thread id internally.
            nvtx::name_thread!("{name}");
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
                nvtx::range_push!("{name}");
            }
            NvtxRangeGuard { active }
        }
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
            nvtx::range_pop!();
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
