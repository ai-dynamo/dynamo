// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Backed by the `nvtx` crate, which vendors NVIDIA's header-only **NVTX v3** C API
//! and compiles it in with `cc`. There is no link-time dependency on any NVTX shared
//! library: the v3 headers compile to inert stubs that lazily `dlopen` the profiler's
//! injection library named by `NVTX_INJECTION64_PATH` the first time an annotation
//! fires. With no profiler attached, nothing is loaded at all.
//!
//! This is why v3 matters on CUDA 13: CUDA 13 removed `libnvToolsExt.so`, which the
//! older v2 C API (and cudarc's binding to it) resolved by `dlopen` — a hard failure
//! when the file is absent. v3 has no such library to find, so markers work on CUDA 13
//! natively — the same mechanism TensorRT-LLM's own Python markers use, which puts both
//! on one timeline.
//!
//! # Gating (two-level)
//!
//! | Cargo feature `nvtx` | `DYN_NVTX` env | Effect                                        |
//! |----------------------|----------------------------|-----------------------------------------------|
//! | off (default)        | any                        | macros compile to nothing; zero overhead      |
//! | on                   | unset                      | inlined `Relaxed` load + branch per site      |
//! | on                   | `1`/`true`/`on`/`yes`      | two allocations + an FFI call per annotation  |
//!
//! The disabled-but-compiled-in row is not free, only cheap. Each macro tests
//! [`enabled()`] in the caller, which inlines to a relaxed load of a static `AtomicBool`
//! and a branch that predicts perfectly; the name and its `core::fmt::Arguments` are
//! built only on the taken side. Verified by disassembling a release build with the
//! repo's `profiling` profile: the disabled path is a load, a `test`, and a jump to the
//! function tail, with no call and no `Arguments` construction.
//!
//! The enabled path is **not** cheap: the `nvtx` crate renders the name with
//! `to_string()` and then copies it into a `CString`, so each annotation costs two heap
//! allocations on top of the injection-library call. Budget on the order of a few
//! hundred nanoseconds, not tens, and keep that in mind when reading a profile of a
//! per-token path — the observer effect is real.
//!
//! # Caveats inherited from the `nvtx` crate
//!
//! - **Range ids are narrowed to `i32`.** `nvtx-sys/export.c` declares
//!   `int ffi_range_start(...)` / `void ffi_range_end(int)`, but NVTX's `nvtxRangeId_t`
//!   is `uint64_t`. If an injection library ever handed back an id with bits above 31
//!   set, the id passed to `nvtxRangeEnd` would differ from the one returned and the
//!   range would never close — an unterminated bar on the timeline.
//!
//!   Measured against Nsight Systems 2026.4.1: ids are allocated sequentially from 1,
//!   so a 300,000-range capture (200k raw start/end pairs plus 100k guard-managed
//!   ranges) saw a maximum id of 200,000, zero negative ids, and all 300,000 ranges
//!   closed. The narrowing has ~2^31 of headroom per process at that allocation rate
//!   and does not bite in practice. Re-check if the injection library ever changes its
//!   id scheme; widening it means a local `cc` shim over the v3 headers, since the
//!   narrowing is in the vendored C, not in the Rust declarations [`ffi`] corrects.
//! - **Interior NULs in names are sanitized, not fatal.** The upstream crate builds its
//!   `CString` with `CString::new(..).expect(..)`, so an interpolated name carrying a NUL
//!   would panic the annotated request path. [`ffi`] below owns that conversion instead
//!   and substitutes U+FFFD, because a profiling aid must never be able to abort the
//!   work it is measuring.
//! - **The upstream `ffi_range_end` binding does not match its C definition.** In `nvtx`
//!   1.3.0, `nvtx-sys/export.c` defines `void ffi_range_end(int)` while `src/bindings.rs`
//!   declares it `-> c_int`. Every guarded range end would cross that mismatched ABI, so
//!   [`ffi`] declares the whole surface itself, matching `export.c` exactly.
//!
//! # Usage
//!
//! ```rust,ignore
//! let _r = dynamo_nvtx_range!("preprocess.tokenize"); // RAII — ends at scope end
//! let _r = dynamo_nvtx_range!("request.{}", request_id); // format args also work
//! dynamo_nvtx_push!("codec.encode");
//! dynamo_nvtx_pop!();
//! dynamo_nvtx_mark!("stream.first_token");
//! dynamo_nvtx_name_thread!("tokio-worker-0");
//! ```
//!
//! # Range flavors
//!
//! [`dynamo_nvtx_range!`] opens an NVTX **start/end** range, which carries an explicit
//! id and is therefore correct in async code: the guard may be created on one tokio
//! worker thread and dropped on another after an `.await`. [`dynamo_nvtx_push!`] /
//! [`dynamo_nvtx_pop!`] use the thread-local nested range stack and must be paired on
//! the same thread with no `.await` in between.
//!
//! # Name arguments
//!
//! A string literal is treated as a **format string**, so a literal containing `{}` or
//! `{name}` interpolates rather than appearing verbatim. `dynamo_nvtx_range!("map{}")`
//! fails to compile; `dynamo_nvtx_range!("a{b}")` silently captures a local `b`. Use
//! `dynamo_nvtx_range!("{}", "map{}")` for a literal brace.
//!
//! # Build
//!
//! ```bash
//! cargo build --profile profiling --features dynamo-runtime/nvtx
//! ```
//!
//! Prefer `dynamo-runtime/nvtx` over a bare `--features nvtx` at the workspace root:
//! the latter also selects the `nvtx` feature of `kvbm-engine`, whose markers have no
//! `DYN_NVTX` runtime switch and would fire unconditionally.

use anyhow::Context as _;

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

/// Name of the runtime switch. Shared verbatim with the Python layer
/// (`dynamo.common.utils.nvtx_utils`), which parses it with the same vocabulary.
pub const ENV_VAR: &str = "DYN_NVTX";

/// Corrected declarations for the C shim that `nvtx` 1.3.0 compiles and links.
///
/// The upstream `src/bindings.rs` is not usable as-is: it declares
/// `ffi_range_end` as returning `c_int` where `nvtx-sys/export.c` defines it
/// `void`, and it builds every name with `CString::new(..).expect(..)`, which
/// turns an interior NUL in an interpolated marker name into a panic on the
/// path being profiled. These declarations match `export.c` exactly and the
/// conversion below cannot fail.
///
/// The symbols come from the static `libnvtx.a` that the `nvtx` crate's build
/// script compiles with `cc`; depending on that crate is what puts them on the
/// link line. A rename upstream surfaces as a link error, not as silent drift.
#[cfg(feature = "nvtx")]
mod ffi {
    use std::ffi::{CString, c_char, c_int};

    #[link(name = "nvtx")]
    unsafe extern "C" {
        fn ffi_range_push(message: *const c_char) -> c_int;
        fn ffi_range_pop() -> c_int;
        fn ffi_range_start(message: *const c_char) -> c_int;
        fn ffi_range_end(id: c_int);
        fn ffi_mark(message: *const c_char);
        fn ffi_name_thread(name: *const c_char);
    }

    /// Render a marker name into a NUL-terminated C string, infallibly.
    ///
    /// An interior NUL is replaced with U+FFFD rather than rejected: the name
    /// is a human-readable label on a timeline, so a mangled label is a far
    /// better outcome than either a lost range or a panic. The allocation
    /// count is unchanged from the upstream path (one `String`, one
    /// `CString`); the `replace` only runs on the rare bad name.
    fn c_name<M: std::fmt::Display>(name: M) -> CString {
        let rendered = name.to_string();
        let cleaned = if rendered.as_bytes().contains(&0) {
            rendered.replace('\0', "\u{fffd}")
        } else {
            rendered
        };
        // No interior NUL remains, so this cannot fail; the fallback keeps the
        // guarantee local instead of resting on that reasoning alone.
        CString::new(cleaned).unwrap_or_default()
    }

    pub fn range_push<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `int ffi_range_push(const char *)` in
        // nvtx-sys/export.c; the pointer is valid for the call's duration.
        unsafe { ffi_range_push(name.as_ptr()) };
    }

    pub fn range_pop() {
        // SAFETY: signature matches `int ffi_range_pop(void)`.
        unsafe { ffi_range_pop() };
    }

    pub fn range_start<M: std::fmt::Display>(name: M) -> c_int {
        let name = c_name(name);
        // SAFETY: signature matches `int ffi_range_start(const char *)`.
        unsafe { ffi_range_start(name.as_ptr()) }
    }

    pub fn range_end(id: c_int) {
        // SAFETY: signature matches `void ffi_range_end(int)`. `id` is one
        // previously returned by `range_start`.
        unsafe { ffi_range_end(id) };
    }

    pub fn mark<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `void ffi_mark(const char *)`.
        unsafe { ffi_mark(name.as_ptr()) };
    }

    pub fn name_thread<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `void ffi_name_thread(const char *)`.
        unsafe { ffi_name_thread(name.as_ptr()) };
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Parse the switch value, rejecting anything outside the shared vocabulary.
///
/// Split out from [`init`] so the contract can be tested without mutating the
/// process environment, which is unsound while other tests run in parallel.
fn parse_switch(raw: Option<&str>) -> anyhow::Result<bool> {
    match raw {
        // Delegated to the canonical parser rather than hand-rolled: the
        // accepted set (1/true/on/yes, 0/false/off/no, empty) is exactly what
        // `nvtx_utils._parse_enabled` mirrors on the Python side.
        Some(value) => crate::config::parse_bool(value)
            .with_context(|| format!("{ENV_VAR} is not a recognized boolean")),
        None => Ok(false),
    }
}

/// Initialise the NVTX subsystem from the [`ENV_VAR`] environment variable.
/// Must be called once at runtime startup before any annotation macros fire.
///
/// The value is validated even when the `nvtx` Cargo feature is off. `DYN_NVTX`
/// is one switch shared with the Python layer, which rejects an unrecognized
/// value outright; if this half quietly read `DYN_NVTX=maybe` as "off", the same
/// capture would half-start and the operator would be told nothing. Rejecting it
/// in both halves is what makes it a single switch.
///
/// Storing the flag and emitting markers remain no-ops without the feature.
pub fn init() -> anyhow::Result<()> {
    let enabled = match std::env::var(ENV_VAR) {
        Ok(raw) => parse_switch(Some(&raw))?,
        Err(std::env::VarError::NotPresent) => parse_switch(None)?,
        Err(std::env::VarError::NotUnicode(raw)) => {
            anyhow::bail!("{ENV_VAR}={raw:?} is not valid unicode")
        }
    };

    #[cfg(feature = "nvtx")]
    {
        NVTX_ENABLED.store(enabled, Ordering::Relaxed);
        if enabled {
            tracing::info!("NVTX annotations enabled ({ENV_VAR})");
        }
    }
    #[cfg(not(feature = "nvtx"))]
    let _ = enabled;

    Ok(())
}

/// Returns `true` when the `nvtx` feature is compiled in **and** `DYN_NVTX` is set.
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
pub fn push_impl<M: std::fmt::Display>(name: M) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            ffi::range_push(name);
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
            ffi::range_pop();
        }
    }
}

/// Record an instantaneous NVTX marker on the calling thread's timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn mark_impl<M: std::fmt::Display>(name: M) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            ffi::mark(name);
        }
    }
    let _ = name;
}

/// Name the current OS thread in the Nsight Systems timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn name_current_thread_impl<M: std::fmt::Display>(name: M) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            ffi::name_thread(name);
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
    #[inline(always)]
    pub fn new<M: std::fmt::Display>(name: M) -> Self {
        // Exactly one of these blocks survives cfg-stripping and becomes the
        // function's tail expression.
        #[cfg(feature = "nvtx")]
        {
            // Re-checked rather than assumed: the macros gate on `enabled()`
            // first, but `new` is public and must not open a range behind the
            // switch's back.
            let id = NVTX_ENABLED
                .load(Ordering::Relaxed)
                .then(|| ffi::range_start(name));
            NvtxRangeGuard { id }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = name;
            NvtxRangeGuard
        }
    }

    /// A guard that ends nothing — the cold half of the macros' `enabled()` branch.
    #[doc(hidden)]
    #[inline(always)]
    pub fn inactive() -> Self {
        #[cfg(feature = "nvtx")]
        {
            NvtxRangeGuard { id: None }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            NvtxRangeGuard
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            ffi::range_end(id);
        }
    }
}

#[cfg(not(feature = "nvtx"))]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {}
}

// ── Macros ───────────────────────────────────────────────────────────────────
//
// Every macro tests `enabled()` *first*, in the caller. That inlines to a
// `Relaxed` load plus a predictable branch, and — critically — keeps the
// `core::fmt::Arguments` construction and the call into this module on the cold
// side of it. Passing `format_args!` straight into a helper instead would build
// the 32-byte `Arguments` on the caller's stack unconditionally and defeat
// inlining, costing ~5-15ns per site even with annotations switched off. That
// matters: `worker.egress.*` and `frontend.http.*.sse_chunk` fire per token.
//
// Consequence to know about: in the `$name:expr` arm the name expression is
// only evaluated when annotations are enabled. Do not pass an expression whose
// side effects you depend on.

/// Push a named NVTX range onto the calling thread's nested range stack.
///
/// Accepts either a `Display` expression or `format!`-style arguments.
/// Must be paired with [`dynamo_nvtx_pop!`] on the same thread, with no `.await`
/// in between — use [`dynamo_nvtx_range!`] in async code.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_push {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::push_impl(::core::format_args!($fmt $(, $arg)*));
        }
    };
    ($name:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::push_impl($name);
        }
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

/// Record an instantaneous NVTX marker on the calling thread's timeline.
/// Accepts either a `Display` expression or `format!`-style arguments.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_mark {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::mark_impl(::core::format_args!($fmt $(, $arg)*));
        }
    };
    ($name:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::mark_impl($name);
        }
    };
}

/// Open a named NVTX range that closes automatically at end of scope.
///
/// ```rust,ignore
/// let _r = dynamo_nvtx_range!("preprocess.tokenize");
/// let _r = dynamo_nvtx_range!("backend.decode.{}", request_id);
/// // range closes here
/// ```
///
/// Safe to hold across `.await`: the range is an NVTX start/end pair, not a
/// thread-local push/pop.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_range {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::NvtxRangeGuard::new(::core::format_args!($fmt $(, $arg)*))
        } else {
            $crate::nvtx::NvtxRangeGuard::inactive()
        }
    };
    ($name:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::NvtxRangeGuard::new($name)
        } else {
            $crate::nvtx::NvtxRangeGuard::inactive()
        }
    };
}

/// Annotate the current OS thread in the Nsight Systems timeline.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_name_thread {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::name_current_thread_impl(::core::format_args!($fmt $(, $arg)*));
        }
    };
    ($name:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::name_current_thread_impl($name);
        }
    };
}

#[cfg(test)]
mod tests {
    /// Serialises the tests that toggle [`super::NVTX_ENABLED`]. The flag is
    /// process-global and this binary runs its tests on parallel threads, so
    /// without this a test asserting "disabled" can observe another test's
    /// "enabled" window.
    #[cfg(feature = "nvtx")]
    static NVTX_ENABLED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Exclusive access to [`NVTX_ENABLED`] for the lifetime of the guard, with the
    /// previous value restored on drop.
    ///
    /// Restoring in `Drop` rather than at the end of the test body is what makes a
    /// failing assertion safe: an `assert!` or `.expect()` that panics mid-test
    /// would otherwise leave the flag `true`, and — because the lock deliberately
    /// ignores poisoning — the next test to assert "disabled" would observe that
    /// leaked value and fail for an unrelated reason.
    #[cfg(feature = "nvtx")]
    struct EnabledFlag {
        _lock: std::sync::MutexGuard<'static, ()>,
        restore: bool,
    }

    #[cfg(feature = "nvtx")]
    impl EnabledFlag {
        /// Take the lock and set the flag, remembering the value to restore.
        fn set(value: bool) -> Self {
            use std::sync::atomic::Ordering;
            // Ignore poisoning: a panic in another test must not cascade into
            // unrelated failures here.
            let _lock = NVTX_ENABLED_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let restore = super::NVTX_ENABLED.swap(value, Ordering::Relaxed);
            Self { _lock, restore }
        }

        /// Flip the flag mid-test. Still restored to the original value on drop.
        fn store(&self, value: bool) {
            super::NVTX_ENABLED.store(value, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(feature = "nvtx")]
    impl Drop for EnabledFlag {
        fn drop(&mut self) {
            super::NVTX_ENABLED.store(self.restore, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Counts its own drops, so a test can prove a value that travelled across an
    /// `.await` (and possibly across tokio worker threads) was dropped exactly once.
    struct DropSpy(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Fire every annotation entry point, in both macro arms.
    fn exercise_all_sites() {
        dynamo_nvtx_name_thread!("nvtx-unit-test");
        dynamo_nvtx_mark!("test.mark");
        dynamo_nvtx_mark!("test.mark.{}", 1);

        dynamo_nvtx_push!("test.push");
        dynamo_nvtx_push!("test.push.{}", 2);
        dynamo_nvtx_pop!();
        dynamo_nvtx_pop!();

        // Non-literal arm: a `Display` expression rather than a format string.
        let name = String::from("test.range.dynamic");
        let _dynamic = dynamo_nvtx_range!(&name);
        let _outer = dynamo_nvtx_range!("test.range");
        let _inner = dynamo_nvtx_range!("test.range.{}", 3);
    }

    /// Exercises the NVTX v3 entry points end to end. With no profiler attached the
    /// header-only stubs are inert, so this asserts the calls resolve and return
    /// rather than that anything is recorded — the regression it guards is the v2
    /// failure mode, where a missing `libnvToolsExt.so` aborted the process.
    ///
    /// `NVTX_ENABLED` is set directly rather than through `DYN_NVTX` +
    /// [`super::init`]: this binary runs its tests in parallel threads, where mutating
    /// the process environment is unsound.
    #[test]
    fn annotations_are_safe_with_no_profiler_attached() {
        #[cfg(feature = "nvtx")]
        {
            let _flag = EnabledFlag::set(true);
            assert!(super::enabled());
            exercise_all_sites();
        }

        // Every site must also be callable with annotations off.
        #[cfg(not(feature = "nvtx"))]
        assert!(!super::enabled());
        exercise_all_sites();
    }

    /// The switch must accept exactly what `nvtx_utils._parse_enabled` accepts on
    /// the Python side, and reject everything else rather than reading it as "off".
    /// A value that disables one half and raises in the other is not one switch.
    #[test]
    fn switch_parsing_matches_the_python_half() {
        assert!(!super::parse_switch(None).expect("unset is a valid off"));

        for on in ["1", "true", "TRUE", " on ", "yes"] {
            assert!(super::parse_switch(Some(on)).expect("recognized"), "{on:?}");
        }
        for off in ["0", "false", "off", "no", ""] {
            assert!(
                !super::parse_switch(Some(off)).expect("recognized"),
                "{off:?}"
            );
        }
        for junk in ["maybe", "2", "enabled", "-1"] {
            let err = super::parse_switch(Some(junk))
                .expect_err("an unrecognized value must not read as off");
            assert!(
                err.to_string().contains(super::ENV_VAR),
                "the error must name the variable the operator set: {err}"
            );
        }
    }

    /// A marker name is a label on a timeline; it must never be able to abort the
    /// work being profiled. The upstream crate's `CString::new(..).expect(..)`
    /// would panic here, which is why the conversion is owned locally.
    #[test]
    fn an_interior_nul_in_a_name_does_not_panic() {
        #[cfg(feature = "nvtx")]
        let _flag = EnabledFlag::set(true);

        let hostile = "test.range.\u{0}injected";
        let _range = dynamo_nvtx_range!("{}", hostile);
        dynamo_nvtx_mark!("{}", hostile);
        dynamo_nvtx_push!("{}", hostile);
        dynamo_nvtx_pop!();
        dynamo_nvtx_name_thread!("{}", hostile);
    }

    // ── Send-ness ────────────────────────────────────────────────────────────

    /// Compile-time assertion helper.
    fn assert_send<T: Send>() {}

    /// The whole reason [`super::NvtxRangeGuard`] holds a start/end range id rather
    /// than relying on the thread-nested push/pop stack is that it must be able to
    /// live inside a future — which requires `Send` for `tokio::spawn`. This is a
    /// compile-time assertion; it must hold in both feature configurations.
    #[test]
    fn range_guard_is_send() {
        assert_send::<super::NvtxRangeGuard>();
        // A future that holds the guard across an await must itself stay `Send`,
        // which is the property `tokio::spawn` actually requires.
        assert_send::<std::pin::Pin<Box<dyn Future<Output = ()> + Send>>>();
    }

    // ── RAII correctness across threads and awaits ───────────────────────────

    /// The guard must be constructible on one thread and droppable on another.
    /// With the old push/pop stack this was silently wrong (the pop landed on the
    /// wrong thread's stack); with start/end ranges it is well defined.
    #[test]
    fn range_guard_can_be_dropped_on_a_different_thread() {
        #[cfg(feature = "nvtx")]
        let _flag = EnabledFlag::set(true);

        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let guard = dynamo_nvtx_range!("test.cross_thread");
        let spy = DropSpy(drops.clone());
        let created_on = std::thread::current().id();

        let dropped_on = std::thread::spawn(move || {
            // `guard` and `spy` are moved in and dropped here, off the creating thread.
            drop(guard);
            drop(spy);
            std::thread::current().id()
        })
        .join()
        .expect("guard drop on another thread must not panic");

        assert_ne!(
            created_on, dropped_on,
            "the guard must have changed threads"
        );
        assert_eq!(
            drops.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "guard must be dropped exactly once"
        );
    }

    /// Hold a range guard across `.await` points on a multi-threaded runtime, with
    /// enough tasks and yields that tokio's work-stealing scheduler resumes at least
    /// some of them on a different worker thread than the one that created the guard.
    ///
    /// Nothing about the recorded NVTX data can be asserted (no profiler is
    /// attached), so this asserts what is observable: every task runs to completion
    /// without panicking, and every guard is dropped exactly once.
    #[test]
    fn range_guard_survives_await_on_multi_thread_runtime() {
        #[cfg(feature = "nvtx")]
        let _flag = EnabledFlag::set(true);

        const TASKS: usize = 16;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("multi-thread runtime");

        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        runtime.block_on(async {
            let mut handles = Vec::with_capacity(TASKS);
            for i in 0..TASKS {
                let drops = drops.clone();
                handles.push(tokio::spawn(async move {
                    // Guard + drop sentinel share a scope, so the sentinel's
                    // drop count is the guard's drop count.
                    let _range = dynamo_nvtx_range!("test.async.range.{}", i);
                    let _spy = DropSpy(drops);

                    // Yield and park: both give the scheduler an opportunity to
                    // resume this task on a different worker thread.
                    tokio::task::yield_now().await;
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    tokio::task::yield_now().await;
                }));
            }
            for handle in handles {
                handle.await.expect("task must not panic");
            }
        });

        drop(runtime);

        // Which thread each task resumed on is scheduler-dependent and so is not
        // asserted; the cross-thread drop itself is covered deterministically by
        // `range_guard_can_be_dropped_on_a_different_thread`.
        assert_eq!(
            drops.load(std::sync::atomic::Ordering::Relaxed),
            TASKS,
            "every guard scope must be dropped exactly once"
        );
    }

    // ── Nesting and non-LIFO ordering ────────────────────────────────────────

    /// Start/end ranges carry an explicit id, so guards need not be closed in the
    /// order they were opened. Both nested (LIFO) and interleaved (non-LIFO)
    /// lifetimes must run cleanly.
    #[test]
    fn ranges_nest_and_interleave() {
        #[cfg(feature = "nvtx")]
        let _flag = EnabledFlag::set(true);

        // Nested, closed LIFO.
        {
            let _outer = dynamo_nvtx_range!("test.nest.outer");
            let _middle = dynamo_nvtx_range!("test.nest.middle");
            let _inner = dynamo_nvtx_range!("test.nest.inner");
        }

        // Interleaved: `first` outlives `second`'s start but is closed before it.
        let first = dynamo_nvtx_range!("test.interleave.first");
        let second = dynamo_nvtx_range!("test.interleave.second");
        drop(first);
        let third = dynamo_nvtx_range!("test.interleave.third");
        drop(second);
        drop(third);

        // push/pop is a thread-local stack and must stay balanced on one thread.
        dynamo_nvtx_push!("test.stack.outer");
        dynamo_nvtx_push!("test.stack.inner");
        dynamo_nvtx_pop!();
        dynamo_nvtx_pop!();
    }

    // ── Macro arm coverage ───────────────────────────────────────────────────

    /// One representative shape per macro arm: a bare literal, a literal with
    /// format args, and a non-literal `Display` expression. The exhaustive
    /// enumeration this replaced re-tested `macro_rules!` itself; what actually
    /// protects call sites is that both arms still match at all.
    fn exercise_every_macro_arm() {
        let owned = String::from("test.arg.string");
        let number: u32 = 7;

        dynamo_nvtx_mark!("test.mark.literal");
        dynamo_nvtx_mark!("test.mark.{}", number);
        dynamo_nvtx_mark!(&owned);

        dynamo_nvtx_push!("test.push.literal");
        dynamo_nvtx_push!("test.push.{}", number);
        dynamo_nvtx_push!(&owned);
        for _ in 0..3 {
            dynamo_nvtx_pop!();
        }

        dynamo_nvtx_name_thread!("test.thread.literal");
        dynamo_nvtx_name_thread!("test.thread.{}", number);
        dynamo_nvtx_name_thread!(&owned);

        let _r1 = dynamo_nvtx_range!("test.range.literal");
        let _r2 = dynamo_nvtx_range!("test.range.{}", number);
        let _r3 = dynamo_nvtx_range!(&owned);
    }

    /// Run [`exercise_every_macro_arm`] with annotations off — the state every
    /// non-profiled process is in — and, when the feature is compiled in, again
    /// with them on.
    #[test]
    fn every_macro_arm_compiles_and_runs() {
        #[cfg(feature = "nvtx")]
        {
            let flag = EnabledFlag::set(false);
            exercise_every_macro_arm();

            flag.store(true);
            exercise_every_macro_arm();
        }

        #[cfg(not(feature = "nvtx"))]
        exercise_every_macro_arm();
    }

    // ── Enabled/disabled transitions ─────────────────────────────────────────

    /// Without the Cargo feature the runtime switch cannot turn anything on:
    /// `enabled()` is a compile-time `false`.
    #[cfg(not(feature = "nvtx"))]
    #[test]
    fn enabled_is_always_false_without_the_feature() {
        assert!(!super::enabled());
        // `init` still validates the switch without the feature, so it returns a
        // `Result`. The outcome depends on whatever `DYN_NVTX` the developer's
        // shell happens to carry and is asserted by
        // `switch_parsing_matches_the_python_half`; what matters here is that no
        // value of it can turn annotations on in this build.
        let _ = super::init();
        assert!(!super::enabled());
    }

    /// `enabled()` is exactly a read of the `NVTX_ENABLED` atomic, and flipping it
    /// back and forth is observable at every step.
    #[cfg(feature = "nvtx")]
    #[test]
    fn enabled_tracks_the_atomic_in_both_directions() {
        let flag = EnabledFlag::set(false);
        assert!(!super::enabled());
        flag.store(true);
        assert!(super::enabled());
        flag.store(false);
        assert!(!super::enabled());
    }

    /// A guard samples the runtime switch once, at construction: a range opened
    /// while disabled must stay a no-op even if annotations are switched on before
    /// it drops (otherwise the drop would end a range that was never started).
    #[cfg(feature = "nvtx")]
    #[test]
    fn guard_samples_the_switch_at_construction() {
        let flag = EnabledFlag::set(false);
        let disabled = dynamo_nvtx_range!("test.switch.disabled");
        assert!(
            disabled.id.is_none(),
            "a range opened while disabled must not allocate an NVTX id"
        );

        flag.store(true);
        let enabled = dynamo_nvtx_range!("test.switch.enabled");
        assert!(
            enabled.id.is_some(),
            "a range opened while enabled must allocate an NVTX id"
        );

        // Dropping the disabled guard after the switch flipped on must not end a
        // range it never started.
        drop(disabled);
        drop(enabled);
    }
}
