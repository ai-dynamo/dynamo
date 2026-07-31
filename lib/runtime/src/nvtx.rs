// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Backed by the [`nvtx`] crate, which vendors the **NVTX v3 header-only C API**
//! and compiles it in via `cc`. Nothing is linked or dlopened at runtime: the v3
//! headers compile to inert stubs that bind to the profiler's injection library
//! through `NVTX_INJECTION64_PATH` the first time an annotation fires. With no
//! profiler attached each call is a predicted-not-taken branch.
//!
//! This is why v3 matters on CUDA 13: CUDA 13 removed `libnvToolsExt.so`, which
//! the older v2 C API (and cudarc's binding to it) resolved by `dlopen`. v3 needs
//! no such library, so markers work on CUDA 13 natively — same mechanism TRT-LLM's
//! own markers already use.
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
//! let _r = dynamo_nvtx_range!("preprocess.tokenize"); // RAII — ends at scope end
//! let _r = dynamo_nvtx_range!("request.{}", request_id); // format args accepted
//! dynamo_nvtx_mark!("engine.first_token");
//! dynamo_nvtx_push!("codec.encode");
//! dynamo_nvtx_pop!();
//! dynamo_nvtx_name_thread!("tokio-worker-0");
//! ```
//!
//! # Range flavors
//!
//! [`dynamo_nvtx_range!`] uses NVTX **start/end** ranges, which carry an explicit id
//! and are therefore correct in async code: the guard may be created on one tokio
//! worker thread and dropped on another after an `.await`. [`dynamo_nvtx_push!`] /
//! [`dynamo_nvtx_pop!`] use the thread-local nested range stack and must be paired
//! on the same thread with no await in between.
//!
//! # Build
//!
//! ```bash
//! cargo build --profile profiling --features nvtx
//! ```

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

/// Push an NVTX range onto the calling thread's nested range stack.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn push_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            nvtx::range_push!("{}", name);
        }
    }
    let _ = name;
}

/// Pop the innermost NVTX range from the calling thread's nested range stack.
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

/// Emit an instantaneous NVTX marker on the calling thread's timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn mark_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            nvtx::mark!("{}", name);
        }
    }
    let _ = name;
}

/// Name the current OS thread in the Nsight Systems timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn name_current_thread_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            // The nvtx crate resolves the OS tid itself (SYS_gettid on Linux).
            nvtx::name_thread!("{}", name);
        }
    }
    let _ = name;
}

// ── RAII guard ───────────────────────────────────────────────────────────────

/// RAII guard that ends an NVTX start/end range when dropped.
/// Construct with [`dynamo_nvtx_range!`].
///
/// Start/end ranges (rather than the thread-nested push/pop stack) are used so the
/// guard stays correct when it is held across an `.await` and the task resumes on a
/// different tokio worker thread.
#[cfg(feature = "nvtx")]
pub struct NvtxRangeGuard {
    /// NVTX range id; `None` when annotations are disabled at runtime.
    id: Option<i32>,
}

/// Zero-sized no-op guard used when the `nvtx` feature is off.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRangeGuard;

impl NvtxRangeGuard {
    #[doc(hidden)]
    pub fn new(name: &str) -> Self {
        // Exactly one of these blocks survives cfg-stripping and becomes the
        // function's tail expression.
        #[cfg(feature = "nvtx")]
        {
            let id = NVTX_ENABLED
                .load(Ordering::Relaxed)
                .then(|| nvtx::range_start!("{}", name));
            NvtxRangeGuard { id }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = name;
            NvtxRangeGuard
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            nvtx::range_end!(id);
        }
    }
}

#[cfg(not(feature = "nvtx"))]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {}
}

// ── Macros ───────────────────────────────────────────────────────────────────

/// Push a named NVTX range onto the calling thread's nested range stack.
///
/// Must be paired with [`dynamo_nvtx_pop!`] on the same thread, with no `.await`
/// in between — use [`dynamo_nvtx_range!`] in async code.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_push {
    ($fmt:literal, $($arg:tt)+) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::push_impl(&::std::format!($fmt, $($arg)+))
        }
    };
    ($name:expr) => {
        $crate::nvtx::push_impl($name)
    };
}

/// Pop the innermost NVTX range from the calling thread's nested range stack.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_pop {
    () => {
        $crate::nvtx::pop_impl()
    };
}

/// Emit an instantaneous NVTX marker on the calling thread's timeline.
///
/// ```rust,ignore
/// dynamo_nvtx_mark!("engine.first_token");
/// dynamo_nvtx_mark!("engine.first_token.{}", request_id);
/// ```
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_mark {
    ($fmt:literal, $($arg:tt)+) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::mark_impl(&::std::format!($fmt, $($arg)+))
        }
    };
    ($name:expr) => {
        $crate::nvtx::mark_impl($name)
    };
}

/// Open a named NVTX range that closes automatically at end of scope.
///
/// ```rust,ignore
/// let _r = dynamo_nvtx_range!("preprocess.tokenize");
/// let _r = dynamo_nvtx_range!("preprocess.tokenize.{}", model);
/// // range closes here
/// ```
///
/// Safe to hold across `.await`: the range is an NVTX start/end pair, not a
/// thread-local push/pop.
///
/// The formatting arm only allocates when annotations are enabled at runtime.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_range {
    ($fmt:literal, $($arg:tt)+) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::NvtxRangeGuard::new(&::std::format!($fmt, $($arg)+))
        } else {
            $crate::nvtx::NvtxRangeGuard::new("")
        }
    };
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

#[cfg(test)]
mod tests {
    /// Fire every annotation entry point, in both macro arms.
    fn exercise_all_sites() {
        dynamo_nvtx_name_thread!("nvtx-unit-test");
        dynamo_nvtx_mark!("test.mark");
        dynamo_nvtx_mark!("test.mark.{}", 1);

        dynamo_nvtx_push!("test.push");
        dynamo_nvtx_push!("test.push.{}", 2);
        dynamo_nvtx_pop!();
        dynamo_nvtx_pop!();

        let _outer = dynamo_nvtx_range!("test.range");
        let _inner = dynamo_nvtx_range!("test.range.{}", 3);
    }

    /// Exercises the NVTX v3 entry points end to end. With no profiler attached the
    /// header-only stubs are inert, so this asserts the calls resolve and return
    /// rather than that anything is recorded — the regression it guards is the v2
    /// failure mode, where a missing `libnvToolsExt.so` aborted the process.
    ///
    /// `NVTX_ENABLED` is set directly rather than through `DYN_ENABLE_RUST_NVTX` +
    /// [`init`]: this binary runs its tests in parallel threads, where mutating the
    /// process environment is unsound.
    #[test]
    fn annotations_are_safe_with_no_profiler_attached() {
        #[cfg(feature = "nvtx")]
        {
            use std::sync::atomic::Ordering;
            let restore = super::NVTX_ENABLED.swap(true, Ordering::Relaxed);
            assert!(super::enabled());
            exercise_all_sites();
            super::NVTX_ENABLED.store(restore, Ordering::Relaxed);
        }

        // Every site must also be callable with annotations off.
        #[cfg(not(feature = "nvtx"))]
        assert!(!super::enabled());
        exercise_all_sites();
    }

    /// `init` reads the env var. Run in-process without mutating the environment by
    /// asserting only the default: unset (or unparseable) means disabled.
    #[test]
    fn init_defaults_to_disabled() {
        if std::env::var("DYN_ENABLE_RUST_NVTX").is_err() {
            super::init();
            assert!(!super::enabled());
        }
    }
}
