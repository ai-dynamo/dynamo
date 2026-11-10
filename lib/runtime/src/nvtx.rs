// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Backed by NVIDIA's header-only **NVTX v3** C API, vendored under `third_party/nvtx3`
//! and compiled with `cc` (from `build.rs`, only when the `nvtx` feature is on) together
//! with a small C shim, `src/nvtx/shim.c`. There is no link-time dependency on any NVTX
//! shared library: the v3 headers compile to inert stubs that lazily `dlopen` the
//! profiler's injection library named by `NVTX_INJECTION64_PATH` the first time an
//! annotation fires. With no profiler attached, nothing is loaded at all.
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
//! # The C shim
//!
//! The `nvtx` crate this module used to wrap only exposed message-only calls and had two
//! defects, both resolved by owning the shim:
//!
//! - **Range ids are `u64` end to end.** NVTX's `nvtxRangeId_t` is `uint64_t`; the crate
//!   narrowed it to `int`, which would leave a range unterminated if an injection library
//!   ever returned an id above 2^31. The shim returns and takes `uint64_t`, and so does
//!   [`NvtxRangeGuard`].
//! - **The declarations match the C definitions.** [`ffi`] declares exactly what
//!   `shim.c` defines (the crate's Rust binding disagreed with its C on `ffi_range_end`'s
//!   return type), and a mismatch is a link error rather than silent drift.
//!
//! The shim also exposes the event-attribute entry points the crate lacked, which carry a
//! `u64` payload (see "Request-scoped annotations"). Interior NULs in names are sanitized
//! rather than fatal: [`ffi`] substitutes U+FFFD because a profiling aid must never be able
//! to abort the work it is measuring.
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
//! Everything goes to the default NVTX domain:
//!
//! - **Request-scoped range or mark**: ASCII message = the stage name only
//!   (`http-request`, `kv_router.route_request:prefill`, ...), event payload = unsigned
//!   64-bit [`rid_key`]`(rid)`. In the nsys SQLite export these are `NVTX_EVENTS.text` and
//!   `NVTX_EVENTS.uint64Value`.
//! - **Request identity mark** (`eventType` 34): `dynamo.request rid=<rid>[ key=value]...`
//!   with the same payload, mapping the key back to the id (see [`format_request_message`]
//!   for ordering and sanitizing). Several may exist per request; readers merge them.
//!
//! [`rid_key`] is FNV-1a-64 of the UTF-8 request id with the top bit cleared so it stays
//! positive in SQLite's signed INTEGER. The Python layer pins the same test vectors.
//!
//! Two sources produce them:
//!
//! - **Explicit markers** — [`dynamo_nvtx_request_mark!`] / [`dynamo_nvtx_request_range!`]
//!   (and the `dynamo._core.nvtx_request_*` Python bindings). They test [`enabled()`]
//!   first, so nothing is formatted or allocated while annotations are off. A non-empty
//!   `attrs` also emits an identity mark carrying those attributes. Where one request opens
//!   several ranges (the worker's ingress and per-chunk egress ranges, the router's
//!   selection ranges), [`request_key`] hashes the id once and [`dynamo_nvtx_keyed_range!`]
//!   reuses it.
//! - **Tracing mirror** (`span_layer`, feature `nvtx` only) — a `tracing` layer installed
//!   by `logging::init` only when `DYN_NVTX` is truthy. It mirrors every INFO-or-above
//!   span that belongs to a request (own `request_id` / `dynamo.request.id` field, or
//!   inherited from an ancestor span) into an NVTX start/end range that ends when the
//!   span closes; the stage is the span name plus a lowercase `:<phase>` when the span has
//!   a `phase` field. A span that takes its id from its own field, while its parent does not
//!   already carry that id, also emits the identity mark (with `trace=` and `xrid=` when
//!   known). Spans with no request id are ignored, and it never touches
//!   OpenTelemetry data. It sits behind the console logger's filter, so it mirrors exactly
//!   the spans `DYN_LOG` already creates (the default `info` covers the whole catalog;
//!   `request_span` spans are always on) and never changes a callsite's cached interest.
//!   A layer that is absent rather than `None` is what keeps the off path at one cached
//!   check per disabled `debug!`: see `registry_dispatch!` in `logging.rs`.
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
//! `DYN_NVTX` off, explicit sites cost the usual relaxed load plus branch and the layer is
//! not installed. Enabled, each mirrored span costs a field visit, one message `String`
//! and the NVTX call (about the 1.3 µs above under `nsys`).
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
pub(crate) mod span_layer;

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

/// Name of the runtime switch. Shared verbatim with the Python layer
/// (`dynamo.common.utils.nvtx_utils`), which parses it with the same vocabulary.
pub const ENV_VAR: &str = "DYN_NVTX";

/// Declarations for `src/nvtx/shim.c`, compiled into the `dynamo_nvtx_shim` static library by
/// `build.rs` (which also puts it on the link line).
///
/// They match the C definitions exactly, with range ids as `uint64_t` end to end. A rename
/// or signature drift on either side surfaces as a link error, not as silent corruption.
/// Name conversion below cannot fail: an interior NUL would otherwise turn an interpolated
/// name into a panic on the path being profiled.
#[cfg(feature = "nvtx")]
mod ffi {
    use std::ffi::{CString, c_char};

    #[link(name = "dynamo_nvtx_shim", kind = "static")]
    unsafe extern "C" {
        fn dynamo_nvtx_range_start(message: *const c_char) -> u64;
        fn dynamo_nvtx_range_start_payload(message: *const c_char, payload: u64) -> u64;
        fn dynamo_nvtx_range_end(id: u64);
        fn dynamo_nvtx_mark(message: *const c_char);
        fn dynamo_nvtx_mark_payload(message: *const c_char, payload: u64);
        fn dynamo_nvtx_range_push(message: *const c_char);
        fn dynamo_nvtx_range_pop();
        fn dynamo_nvtx_name_thread(name: *const c_char);
    }

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

    pub fn range_push<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `void dynamo_nvtx_range_push(const char *)` in
        // shim.c; the pointer is valid for the call's duration.
        unsafe { dynamo_nvtx_range_push(name.as_ptr()) };
    }

    pub fn range_pop() {
        // SAFETY: signature matches `void dynamo_nvtx_range_pop(void)`.
        unsafe { dynamo_nvtx_range_pop() };
    }

    pub fn range_start<M: std::fmt::Display>(name: M) -> u64 {
        let name = c_name(name);
        // SAFETY: signature matches `uint64_t dynamo_nvtx_range_start(const char *)`.
        unsafe { dynamo_nvtx_range_start(name.as_ptr()) }
    }

    pub fn range_start_payload<M: std::fmt::Display>(name: M, payload: u64) -> u64 {
        let name = c_name(name);
        // SAFETY: signature matches
        // `uint64_t dynamo_nvtx_range_start_payload(const char *, uint64_t)`.
        unsafe { dynamo_nvtx_range_start_payload(name.as_ptr(), payload) }
    }

    pub fn range_end(id: u64) {
        // SAFETY: signature matches `void dynamo_nvtx_range_end(uint64_t)`. `id` is one
        // previously returned by `range_start*`.
        unsafe { dynamo_nvtx_range_end(id) };
    }

    pub fn mark<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `void dynamo_nvtx_mark(const char *)`.
        unsafe { dynamo_nvtx_mark(name.as_ptr()) };
    }

    pub fn mark_payload<M: std::fmt::Display>(name: M, payload: u64) {
        let name = c_name(name);
        // SAFETY: signature matches `void dynamo_nvtx_mark_payload(const char *, uint64_t)`.
        unsafe { dynamo_nvtx_mark_payload(name.as_ptr(), payload) };
    }

    pub fn name_thread<M: std::fmt::Display>(name: M) {
        let name = c_name(name);
        // SAFETY: signature matches `void dynamo_nvtx_name_thread(const char *)`.
        unsafe { dynamo_nvtx_name_thread(name.as_ptr()) };
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

/// Whether `DYN_NVTX` currently reads as on, without touching the runtime flag.
///
/// An unset, unparsable or non-unicode value reads as off; [`init`] is what reports
/// those. Used by `logging::init`, which runs before [`init`], to decide whether to
/// install the tracing mirror layer.
#[cfg(feature = "nvtx")]
pub(crate) fn switch_from_env() -> bool {
    std::env::var(ENV_VAR)
        .ok()
        .and_then(|raw| parse_switch(Some(&raw)).ok())
        .unwrap_or(false)
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
    id: Option<u64>,
}

/// Zero-sized no-op guard used when the `nvtx` feature is off.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRangeGuard;

impl NvtxRangeGuard {
    #[doc(hidden)]
    #[inline(always)]
    pub fn new<M: std::fmt::Display>(name: M) -> Self {
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

    /// Open a request-scoped range (static `stage`, `key` as the NVTX payload) if
    /// annotations are on. Prefer [`dynamo_nvtx_request_range!`].
    #[doc(hidden)]
    pub fn new_with_payload<M: std::fmt::Display>(stage: M, key: u64) -> Self {
        #[cfg(feature = "nvtx")]
        {
            let id = NVTX_ENABLED
                .load(Ordering::Relaxed)
                .then(|| ffi::range_start_payload(stage, key));
            NvtxRangeGuard { id }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = (stage, key);
            NvtxRangeGuard
        }
    }

    /// Open a range named `stage`, with `key` as the NVTX payload when it is `Some` and
    /// without a payload when it is `None`, if annotations are on. Prefer
    /// [`dynamo_nvtx_keyed_range!`].
    #[doc(hidden)]
    #[inline(always)]
    pub fn new_keyed<M: std::fmt::Display>(stage: M, key: Option<u64>) -> Self {
        match key {
            Some(key) => Self::new_with_payload(stage, key),
            None => Self::new(stage),
        }
    }

    /// Open a request-scoped range without consulting the runtime flag. Only for the
    /// tracing mirror layer, which is installed solely when `DYN_NVTX` is on and runs before
    /// [`init`] has stored the flag.
    #[cfg(feature = "nvtx")]
    pub(crate) fn new_with_payload_unchecked<M: std::fmt::Display>(stage: M, key: u64) -> Self {
        NvtxRangeGuard {
            id: Some(ffi::range_start_payload(stage, key)),
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

/// Longest value (in bytes) emitted for any `rid=`/`key=` token.
pub const MAX_VALUE_BYTES: usize = 128;

/// Keys that always come first after `rid=`, in this order.
const LEADING_KEYS: [&str; 3] = ["trace", "xrid", "engine_rid"];

/// Append `value` sanitized for a space-separated token: whitespace becomes `_`,
/// control characters are dropped, and the result is cut to [`MAX_VALUE_BYTES`] on a
/// char boundary. Returns whether anything was written.
fn push_sanitized(out: &mut String, value: &str) -> bool {
    let start = out.len();
    for ch in value.chars() {
        let ch = if ch.is_whitespace() {
            '_'
        } else if ch.is_control() {
            continue;
        } else {
            ch
        };
        if out.len() - start + ch.len_utf8() > MAX_VALUE_BYTES {
            break;
        }
        out.push(ch);
    }
    out.len() > start
}

fn push_attr(out: &mut String, key: &str, value: &str) {
    let rollback = out.len();
    out.push(' ');
    let key_start = out.len();
    out.extend(key.chars().map(|c| {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
            c
        } else {
            '_'
        }
    }));
    if out.len() == key_start {
        out.truncate(rollback);
        return;
    }
    out.push('=');
    if !push_sanitized(out, value) {
        out.truncate(rollback);
    }
}

/// Stage name of the request identity mark.
pub const IDENTITY_STAGE: &str = "dynamo.request";

/// The 64-bit NVTX payload that identifies a request: FNV-1a-64 of the UTF-8 request id
/// with the top bit cleared, so the value stays positive in SQLite's signed INTEGER.
///
/// Stable across processes and languages (the Python tests pin the same vectors), which is
/// what lets the frontend's and the workers' events for one request be joined.
#[must_use]
pub const fn rid_key(rid: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let bytes = rid.as_bytes();
    let mut hash = OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    hash & 0x7FFF_FFFF_FFFF_FFFF
}

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

/// Render `<stage> rid=<rid>[ key=value]...`.
///
/// Used with [`IDENTITY_STAGE`] for the request identity mark. `trace`, `xrid` and
/// `engine_rid` come first (in that order), then the remaining `attrs` sorted by key.
/// Values are sanitized and bounded (see [`MAX_VALUE_BYTES`]) and attributes whose value
/// ends up empty are omitted. This allocates; callers must test [`enabled()`] first (the
/// `dynamo_nvtx_request_*` macros do).
pub fn format_request_message(stage: &str, rid: &str, attrs: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(stage.len() + rid.len() + 32 + attrs.len() * 24);
    push_sanitized(&mut out, stage);
    out.push_str(" rid=");
    push_sanitized(&mut out, rid);

    for key in LEADING_KEYS {
        if let Some((_, value)) = attrs.iter().find(|(k, _)| *k == key) {
            push_attr(&mut out, key, value);
        }
    }
    let mut rest: Vec<&(&str, &str)> = attrs
        .iter()
        .filter(|(k, _)| !LEADING_KEYS.contains(k))
        .collect();
    rest.sort_by_key(|(k, _)| *k);
    for (key, value) in rest {
        push_attr(&mut out, key, value);
    }
    out
}

/// Message of the identity mark for `rid` with `attrs`.
#[cfg(feature = "nvtx")]
pub(crate) fn identity_message(rid: &str, attrs: &[(&str, &str)]) -> String {
    format_request_message(IDENTITY_STAGE, rid, attrs)
}

/// Emit the identity mark without consulting the runtime flag. Only for the tracing mirror
/// layer (see [`NvtxRangeGuard::new_with_payload_unchecked`]).
#[cfg(feature = "nvtx")]
pub(crate) fn identity_mark_unchecked(message: &str, key: u64) {
    ffi::mark_payload(message, key);
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
            ffi::mark_payload(identity_message(rid, attrs), key);
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
        ffi::mark_payload(identity_message(rid, attrs), key);
    }
    #[cfg(not(feature = "nvtx"))]
    let _ = attrs;
    NvtxRangeGuard::new_with_payload(stage, key)
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

    const PIN_A: u64 = 3_414_815_163_700_866_188;
    const PIN_UUID: u64 = 1_346_822_055_457_142_067;

    // ── Request message formatter ────────────────────────────────────────────

    #[test]
    fn format_minimal_message() {
        assert_eq!(
            super::format_request_message("http-request", "abc", &[]),
            "http-request rid=abc"
        );
    }

    #[test]
    fn format_orders_leading_keys_then_sorted_rest() {
        let msg = super::format_request_message(
            "stage",
            "r1",
            &[
                ("zeta", "z"),
                ("engine_rid", "e"),
                ("alpha", "a"),
                ("xrid", "x"),
                ("trace", "t"),
            ],
        );
        assert_eq!(
            msg,
            "stage rid=r1 trace=t xrid=x engine_rid=e alpha=a zeta=z"
        );
    }

    #[test]
    fn format_omits_empty_and_fully_dropped_values() {
        let msg = super::format_request_message(
            "s",
            "r",
            &[("xrid", ""), ("a", "\u{1}\u{7f}"), ("b", "ok"), ("", "v")],
        );
        assert_eq!(msg, "s rid=r b=ok");
    }

    #[test]
    fn format_sanitizes_whitespace_and_control_chars() {
        let msg = super::format_request_message(
            "s",
            "a b\tc\nd",
            &[("xrid", "x y\u{0}z\r\n"), ("k=ey v", "v")],
        );
        assert_eq!(msg, "s rid=a_b_c_d xrid=x_yz__ k_ey_v=v");
        assert_eq!(msg.matches(' ').count(), 3);
    }

    #[test]
    fn format_truncates_on_char_boundary() {
        let long = "a".repeat(500);
        let msg = super::format_request_message("s", &long, &[]);
        assert_eq!(msg, format!("s rid={}", "a".repeat(super::MAX_VALUE_BYTES)));

        // 3-byte chars: 128 / 3 = 42 whole chars (126 bytes); never split one.
        let wide = "\u{20ac}".repeat(100);
        let msg = super::format_request_message("s", "r", &[("xrid", &wide)]);
        let value = msg.strip_prefix("s rid=r xrid=").expect("prefix");
        assert_eq!(value.chars().count(), 42);
        assert_eq!(value.len(), 126);

        // Exactly at the limit is kept whole.
        let exact = "b".repeat(super::MAX_VALUE_BYTES);
        assert!(super::format_request_message("s", &exact, &[]).ends_with(&exact));
    }

    #[test]
    fn format_matches_the_parser_regex_shape() {
        let msg = super::format_request_message(
            "kv_router.route_request:prefill",
            "123e4567-e89b-12d3-a456-426614174000",
            &[
                ("trace", "0af7651916cd43dd8448eb211c80319c"),
                ("xrid", "my id"),
            ],
        );
        let mut tokens = msg.split(' ');
        assert_eq!(tokens.next(), Some("kv_router.route_request:prefill"));
        assert!(tokens.next().expect("rid").starts_with("rid="));
        for rest in tokens {
            assert!(rest.contains('='), "{rest}");
        }
    }

    #[test]
    fn rid_key_is_fnv1a_with_top_bit_cleared() {
        // Pinned; the Python tests assert the same vectors.
        assert_eq!(super::rid_key(""), 5_472_609_002_491_880_229);
        assert_eq!(super::rid_key("a"), PIN_A);
        assert_eq!(
            super::rid_key("6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42"),
            PIN_UUID
        );
        for rid in ["", "a", "zz", "6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42"] {
            assert!(super::rid_key(rid) < 1 << 63, "{rid:?}");
        }
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
        #[cfg(feature = "nvtx")]
        {
            flag.store(true);
            let key = super::request_key("a");
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
