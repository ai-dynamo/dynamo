// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Backed by NVIDIA's header-only **NVTX v3** C API through the `nvtx-sys` crate, which
//! compiles the headers in (only when the `nvtx` feature is on). There is no link-time
//! dependency on any NVTX shared library: the v3 headers compile to inert stubs that
//! lazily `dlopen` the profiler's injection library named by `NVTX_INJECTION64_PATH` the
//! first time an annotation fires. With no profiler attached, nothing is loaded at all.
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
//! The enabled path is **not** cheap: each annotation renders its name into a `String`,
//! copies it into a `CString` and calls into the injection library. Measured at 4096
//! concurrent streams on mock workers, a range costs about 0.1 µs with no profiler attached
//! and about 1.3 µs under `nsys -t nvtx`. Keep the second figure in mind when reading a
//! profile of a per-chunk path: the observer effect is real.
//!
//! # The `nvtx-sys` binding
//!
//! `ffi` wraps `nvtx-sys` 2.x, NVIDIA's own Rust binding from the NVTX repository. It
//! replaced the community `nvtx` 1.x crate, which exposed only message-only calls and
//! narrowed range ids to `int`. `nvtx-sys` gives this module what it needs:
//!
//! - **Range ids are `u64` end to end.** NVTX's `nvtxRangeId_t` is `uint64_t`, and so is
//!   the id held by [`NvtxRangeGuard`], so no id an injection library returns is truncated.
//! - **Event attributes carry a `u64` payload.** `nvtxRangeStartEx` / `nvtxMarkEx` take an
//!   attributes struct with the request key as payload (see "Request-scoped annotations").
//!
//! Only `nvtx-sys` is used, not the higher-level `nvtx` crate: these few calls need none of
//! its defaults (`tracing` integration, colors, a thread-id crate). Its build script runs
//! bindgen, so building with the feature needs libclang. Interior NULs in names are
//! sanitized rather than fatal: `ffi` substitutes U+FFFD because a profiling aid must
//! never be able to abort the work it is measuring.
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
//! # Request-scoped annotations
//!
//! Annotations that belong to one request carry a 64-bit **request key** so a request seen
//! misbehaving in OpenTelemetry can be found in an `nsys` report, while every range name
//! stays static (so `nsys stats --report nvtx_sum` aggregates per stage, not per request).
//! The message and payload format ([`rid_key`], [`format_request_message`],
//! [`IDENTITY_STAGE`]) is the cross-language contract; it is specified, with its pinned
//! test vectors, in the `request_id` submodule (`nvtx/request_id.rs`) and re-exported here.
//!
//! **Explicit markers** — [`dynamo_nvtx_request_mark!`] / [`dynamo_nvtx_request_range!`]
//! produce request-scoped ranges and marks. They test [`enabled()`] first, so nothing is formatted or allocated while annotations are off. A
//! non-empty `attrs` also emits an identity mark carrying those attributes. Where one
//! request opens several ranges (the worker's ingress and per-chunk egress ranges, the
//! router's selection ranges), [`request_key`] hashes the id once and
//! [`dynamo_nvtx_keyed_range!`] reuses it.
//!
//! Raw-trace recipes against the nsys SQLite export:
//!
//! ```sql
//! -- per-stage aggregation over closed ranges (`nsys stats --report nvtx_sum` does this
//! -- natively for whole-request captures; it overflows on windowed ones with open ranges)
//! SELECT text AS stage, count(*) AS n, avg("end" - start) / 1e6 AS avg_ms
//! FROM NVTX_EVENTS WHERE eventType = 60 AND uint64Value IS NOT NULL
//!   AND "end" <= (SELECT duration FROM ANALYSIS_DETAILS) GROUP BY text;
//! -- request id -> key
//! SELECT DISTINCT uint64Value FROM NVTX_EVENTS
//! WHERE eventType = 34 AND (text = 'dynamo.request rid=<rid>' OR instr(text, 'dynamo.request rid=<rid> ') = 1);
//! -- every event of that request
//! SELECT text, start, "end", globalTid FROM NVTX_EVENTS WHERE uint64Value = :key ORDER BY start;
//! ```
//!
//! Cost: with the feature off, none of this is compiled. With the feature on and
//! `DYN_NVTX` off, explicit sites cost the usual relaxed load plus branch.
//!
//! # Build
//!
//! ```bash
//! cargo build --profile profiling --features dynamo-runtime/nvtx
//! ```

use anyhow::Context as _;

mod request_id;

pub use request_id::{IDENTITY_STAGE, MAX_VALUE_BYTES, format_request_message, rid_key};

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

/// Name of the runtime switch. Shared verbatim with the Python layer
/// (`dynamo.common.utils.nvtx_utils`), which parses it with the same vocabulary.
pub const ENV_VAR: &str = "DYN_NVTX";

/// Thin wrappers over `nvtx-sys`, NVIDIA's binding to the header-only NVTX v3 C API.
///
/// Range ids are `u64` (`nvtxRangeId_t`) end to end. Name conversion below cannot fail: an
/// interior NUL would otherwise turn an interpolated name into a panic on the path being
/// profiled.
#[cfg(feature = "nvtx")]
mod ffi {
    use nvtx_sys::{EventAttributes, MessageType, MessageValue, PayloadType, PayloadValue};
    use std::ffi::CString;

    /// Render a marker name into a NUL-terminated C string, infallibly.
    ///
    /// An interior NUL is replaced with U+FFFD rather than rejected: the name
    /// is a human-readable label on a timeline, so a mangled label is a far
    /// better outcome than either a lost range or a panic. The `replace` only
    /// runs on the rare bad name.
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

    /// Event attributes carrying an ASCII message and a `u64` payload. `name` must outlive
    /// the NVTX call the attributes are passed to.
    fn payload_attributes(name: &CString, payload: u64) -> EventAttributes {
        EventAttributes {
            version: nvtx_sys::NVTX_VERSION as u16,
            size: nvtx_sys::NVTX_EVENT_ATTRIBUTES_SIZE as u16,
            category: 0,
            colorType: 0, // NVTX_COLOR_UNKNOWN
            color: 0,
            payloadType: PayloadType::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 as i32,
            reserved0: 0,
            payload: PayloadValue { ullValue: payload },
            messageType: MessageType::NVTX_MESSAGE_TYPE_ASCII as i32,
            message: MessageValue {
                ascii: name.as_ptr(),
            },
        }
    }

    /// The calling thread's OS id, which is what `nvtxNameOsThread` expects.
    #[cfg(target_os = "linux")]
    fn os_thread_id() -> u32 {
        // SAFETY: SYS_gettid takes no arguments and cannot fail. A raw syscall because the
        // gettid() wrapper needs glibc 2.30 and the wheel targets manylinux_2_28.
        unsafe { libc::syscall(libc::SYS_gettid) as u32 }
    }

    #[cfg(target_os = "macos")]
    fn os_thread_id() -> u32 {
        let mut tid: u64 = 0;
        // SAFETY: a null thread means the calling thread; `tid` is a valid out pointer.
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid as u32
    }

    pub fn range_push<M: std::fmt::Display>(name: M) {
        let _ = nvtx_sys::range_push_ascii(&c_name(name));
    }

    pub fn range_pop() {
        let _ = nvtx_sys::range_pop();
    }

    pub fn range_start<M: std::fmt::Display>(name: M) -> u64 {
        nvtx_sys::range_start_ascii(&c_name(name))
    }

    pub fn range_start_payload<M: std::fmt::Display>(name: M, payload: u64) -> u64 {
        let name = c_name(name);
        nvtx_sys::range_start_ex(&payload_attributes(&name, payload))
    }

    pub fn range_end(id: u64) {
        nvtx_sys::range_end(id);
    }

    pub fn mark<M: std::fmt::Display>(name: M) {
        nvtx_sys::mark_ascii(&c_name(name));
    }

    pub fn mark_payload<M: std::fmt::Display>(name: M, payload: u64) {
        let name = c_name(name);
        nvtx_sys::mark_ex(&payload_attributes(&name, payload));
    }

    pub fn name_thread<M: std::fmt::Display>(name: M) {
        nvtx_sys::name_os_thread_ascii(os_thread_id(), &c_name(name));
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

/// Record a marker with an optional precomputed request key. Prefer
/// [`dynamo_nvtx_keyed_mark!`].
#[doc(hidden)]
#[inline(always)]
pub fn keyed_mark_impl<M: std::fmt::Display>(stage: M, key: Option<u64>) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            match key {
                Some(key) => ffi::mark_payload(stage, key),
                None => ffi::mark(stage),
            }
        }
    }
    #[cfg(not(feature = "nvtx"))]
    let _ = (stage, key);
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
    id: Option<u64>,
}

/// Zero-sized no-op guard used when the `nvtx` feature is off.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRangeGuard;

impl NvtxRangeGuard {
    /// Open a range named `name` if annotations are on. Prefer [`dynamo_nvtx_range!`].
    #[doc(hidden)]
    #[inline(always)]
    pub fn new<M: std::fmt::Display>(name: M) -> Self {
        Self::new_keyed(name, None)
    }

    /// Open a range named `stage`, with `key` as the NVTX payload when it is `Some` and
    /// without a payload when it is `None`, if annotations are on. Prefer
    /// [`dynamo_nvtx_keyed_range!`] or [`dynamo_nvtx_request_range!`].
    ///
    /// The single place a guard opens a range: `stage` is rendered and the FFI is
    /// called only when the `nvtx` feature is on and `DYN_NVTX` is set.
    #[doc(hidden)]
    #[inline(always)]
    pub fn new_keyed<M: std::fmt::Display>(stage: M, key: Option<u64>) -> Self {
        #[cfg(feature = "nvtx")]
        {
            // Re-checked rather than assumed: the macros gate on `enabled()`
            // first, but this is public and must not open a range behind the
            // switch's back.
            let id = NVTX_ENABLED.load(Ordering::Relaxed).then(|| match key {
                Some(key) => ffi::range_start_payload(stage, key),
                None => ffi::range_start(stage),
            });
            NvtxRangeGuard { id }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = (stage, key);
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
// matters: `worker.egress.*` fires once per streamed chunk.
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

// ── Request-scoped messages ──────────────────────────────────────────────────

/// [`rid_key`]`(rid)` while annotations are on, `None` while they are off, so nothing is
/// hashed on the off path.
///
/// Compute it once where one request opens several ranges, or one range per response
/// chunk, and pass it to [`dynamo_nvtx_keyed_range!`].
#[inline(always)]
#[must_use]
pub fn request_key(rid: &str) -> Option<u64> {
    if enabled() { Some(rid_key(rid)) } else { None }
}

/// Record a request-scoped marker. Prefer [`dynamo_nvtx_request_mark!`].
#[doc(hidden)]
#[inline(never)]
pub fn request_mark_impl(stage: &str, rid: &str, attrs: &[(&str, &str)]) {
    #[cfg(feature = "nvtx")]
    {
        if !enabled() {
            return;
        }
        let key = rid_key(rid);
        if !attrs.is_empty() {
            ffi::mark_payload(format_request_message(rid, attrs), key);
        }
        ffi::mark_payload(stage, key);
    }
    #[cfg(not(feature = "nvtx"))]
    let _ = (stage, rid, attrs);
}

/// Open a request-scoped range. Prefer [`dynamo_nvtx_request_range!`].
#[doc(hidden)]
#[inline(never)]
pub fn request_range_impl(stage: &str, rid: &str, attrs: &[(&str, &str)]) -> NvtxRangeGuard {
    let key = rid_key(rid);
    #[cfg(feature = "nvtx")]
    if enabled() && !attrs.is_empty() {
        ffi::mark_payload(format_request_message(rid, attrs), key);
    }
    #[cfg(not(feature = "nvtx"))]
    let _ = attrs;
    NvtxRangeGuard::new_keyed(stage, Some(key))
}

/// Record a request-scoped NVTX marker: message `<stage>`, payload [`rid_key`]`(rid)`.
///
/// `$attrs` is an optional `&[(&str, &str)]`; when non-empty an identity mark
/// (`dynamo.request rid=<rid> k=v...`, same payload) is emitted first. Nothing is formatted or allocated unless
/// annotations are enabled. Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_request_mark {
    ($stage:expr, $rid:expr) => {
        $crate::dynamo_nvtx_request_mark!($stage, $rid, &[])
    };
    ($stage:expr, $rid:expr, $attrs:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::request_mark_impl($stage, $rid, $attrs);
        }
    };
}

/// Open a request-scoped NVTX range (message `<stage>`, payload [`rid_key`]`(rid)`) that
/// closes when the returned guard drops. Safe to hold across `.await`.
///
/// `$attrs` is an optional `&[(&str, &str)]`; when non-empty an identity mark
/// (`dynamo.request rid=<rid> k=v...`, same payload) is emitted first. Nothing is formatted or allocated unless
/// annotations are enabled. Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_request_range {
    ($stage:expr, $rid:expr) => {
        $crate::dynamo_nvtx_request_range!($stage, $rid, &[])
    };
    ($stage:expr, $rid:expr, $attrs:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::request_range_impl($stage, $rid, $attrs)
        } else {
            $crate::nvtx::NvtxRangeGuard::inactive()
        }
    };
}

/// Open an NVTX range named `$stage` that closes when the returned guard drops, with the
/// precomputed request key `$key: Option<u64>` (from [`nvtx::request_key`](crate::nvtx::request_key))
/// as its payload. A `None` key opens the same range without a payload.
///
/// Unlike [`dynamo_nvtx_request_range!`], which hashes the request id at every call,
/// this reuses one key, so per-chunk ranges cost no more than unkeyed ones. It never emits
/// an identity mark. Safe to hold across `.await`. Zero-cost when the `nvtx` Cargo feature
/// is off.
#[macro_export]
macro_rules! dynamo_nvtx_keyed_range {
    ($stage:expr, $key:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::NvtxRangeGuard::new_keyed($stage, $key)
        } else {
            $crate::nvtx::NvtxRangeGuard::inactive()
        }
    };
}

/// Record an NVTX marker named `$stage` with the precomputed request key `$key:
/// Option<u64>` as its payload; a `None` key records it without one. The marker
/// counterpart of [`dynamo_nvtx_keyed_range!`]. Zero-cost when the `nvtx` Cargo feature
/// is off.
#[macro_export]
macro_rules! dynamo_nvtx_keyed_mark {
    ($stage:expr, $key:expr) => {
        if $crate::nvtx::enabled() {
            $crate::nvtx::keyed_mark_impl($stage, $key);
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
    #[cfg(feature = "nvtx")]
    use super::request_id::PIN_A;

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

    fn assert_send<T: Send>() {}

    /// The whole reason [`super::NvtxRangeGuard`] holds a start/end range id rather
    /// than relying on the thread-nested push/pop stack is that it must be able to
    /// live inside a future — which requires `Send` for `tokio::spawn`. This is a
    /// compile-time assertion; it must hold in both feature configurations.
    #[test]
    fn range_guard_is_send() {
        assert_send::<super::NvtxRangeGuard>();
    }

    // ── Macro arm coverage ───────────────────────────────────────────────────

    /// One representative shape per macro arm: a bare literal, a literal with
    /// format args, and a non-literal `Display` expression.
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

        drop(disabled);
        drop(enabled);
    }

    /// The constructors own the runtime gate themselves (the macros' `enabled()`
    /// check is only the fast path): called directly while disabled, both the keyed
    /// and unkeyed arms of `new_keyed`, and `new`, must return an inactive guard.
    #[cfg(feature = "nvtx")]
    #[test]
    fn constructors_gate_on_the_switch_without_the_macros() {
        use super::NvtxRangeGuard;
        let flag = EnabledFlag::set(false);
        assert!(NvtxRangeGuard::new("test.ctor").id.is_none());
        assert!(NvtxRangeGuard::new_keyed("test.ctor", None).id.is_none());
        assert!(NvtxRangeGuard::new_keyed("test.ctor", Some(7)).id.is_none());

        flag.store(true);
        assert!(NvtxRangeGuard::new("test.ctor").id.is_some());
        assert!(NvtxRangeGuard::new_keyed("test.ctor", None).id.is_some());
        assert!(NvtxRangeGuard::new_keyed("test.ctor", Some(7)).id.is_some());
    }

    #[test]
    fn request_macros_run_in_every_arm() {
        #[cfg(feature = "nvtx")]
        let flag = EnabledFlag::set(false);
        dynamo_nvtx_request_mark!("test.mark", "rid-1");
        dynamo_nvtx_request_mark!("test.mark", "rid-1", &[("k", "v")]);
        let _r = dynamo_nvtx_request_range!("test.range", "rid-1");
        #[cfg(feature = "nvtx")]
        {
            flag.store(true);
            dynamo_nvtx_request_mark!("test.mark", "rid-1", &[("k", "v")]);
            let _r = dynamo_nvtx_request_range!("test.range", "rid-1", &[("k", "v")]);
        }
    }

    /// `request_key` hashes only while enabled, and the keyed range opens with or
    /// without a key.
    #[test]
    fn keyed_range_runs_in_every_arm() {
        #[cfg(feature = "nvtx")]
        let flag = EnabledFlag::set(false);
        assert_eq!(super::request_key("a"), None);
        let _keyed = dynamo_nvtx_keyed_range!("test.keyed", super::request_key("a"));
        let _unkeyed = dynamo_nvtx_keyed_range!("test.keyed", None);
        dynamo_nvtx_keyed_mark!("test.keyed_mark", super::request_key("a"));
        #[cfg(feature = "nvtx")]
        {
            flag.store(true);
            let key = super::request_key("a");
            dynamo_nvtx_keyed_mark!("test.keyed_mark", key);
            dynamo_nvtx_keyed_mark!("test.keyed_mark", None);
            assert_eq!(key, Some(PIN_A));
            let keyed = dynamo_nvtx_keyed_range!("test.keyed", key);
            let unkeyed = dynamo_nvtx_keyed_range!("test.keyed", None);
            assert!(keyed.id.is_some(), "a keyed range opened while enabled");
            assert!(
                unkeyed.id.is_some(),
                "an unkeyed range opened while enabled"
            );
        }
    }

    /// Nothing is evaluated or formatted while disabled.
    #[cfg(feature = "nvtx")]
    #[test]
    fn request_macros_skip_formatting_when_disabled() {
        let _flag = EnabledFlag::set(false);
        let evaluated = std::cell::Cell::new(false);
        let probe = || {
            evaluated.set(true);
            "x"
        };
        dynamo_nvtx_request_mark!("s", probe());
        let _guard = dynamo_nvtx_request_range!("s", probe());
        assert!(!evaluated.get());
    }
}
