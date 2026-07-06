// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gated prefill-path instrumentation (DYN_PREFILL_TRACE).
//!
//! Localizes where a disagg PREFILL request spends its pre-decode time and
//! whether the frontend keeps each CTX worker densely fed. Entirely zero-cost
//! when `DYN_PREFILL_TRACE` is unset: the gate is parsed ONCE via `OnceLock`
//! (mirroring the `DYN_STALL_OP_TRACE` pattern in `kv_router.rs`), and every
//! emit site early-returns on the cached bool before touching the gauge or
//! formatting any field.
//!
//! Two signals (both `tracing::warn!`, distinct targets):
//! - `dynamo_prefill_trace`: ONE per request with select_ms / dispatch_gap_ms /
//!   prefill_first_ms / prefill_total_ms — localizes the time.
//! - `dynamo_prefill_inflight`: emitted on each in-flight increment, showing
//!   per-CTX-worker concurrency. inflight ≫ 1 ⇒ frontend feeds densely;
//!   inflight ≈ 1–2 ⇒ frontend under-dispatches.

use std::sync::OnceLock;
use std::time::Instant;

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerId;

/// Parse `DYN_PREFILL_TRACE` once. `1`/`true` enable; anything else (or unset)
/// disables. Cached for the process lifetime so the hot path pays only an
/// atomic load when the flag is off.
#[inline]
pub(super) fn prefill_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("DYN_PREFILL_TRACE")
            .ok()
            .is_some_and(|v| v == "1" || v == "true")
    })
}

/// Process-global per-CTX-worker in-flight prefill gauge. `WorkerId` is a
/// numeric internal id, but per the scheduling guardrail we keep this on a
/// standard hash collection (DashMap's default SipHash) — it is an
/// instrumentation side-table, not a routing hot-path map. Only allocated when
/// the flag is on (first `inflight_inc` call).
fn inflight_map() -> &'static DashMap<WorkerId, usize> {
    static MAP: OnceLock<DashMap<WorkerId, usize>> = OnceLock::new();
    MAP.get_or_init(DashMap::new)
}

/// RAII guard for the per-CTX-worker in-flight prefill count. Constructing it
/// (via [`inflight_guard`]) increments the gauge for `worker` and WARNs the new
/// depth; dropping it decrements. This makes the gauge leak-proof across every
/// exit path of `execute_prefill` (early error returns, `?`, success, and the
/// spawned task being cancelled) without scattering manual decrements.
///
/// `None` is returned when the flag is off or no target worker is known, so the
/// hot path holds an `Option<InflightGuard>` that does nothing when off.
pub(super) struct InflightGuard {
    worker: WorkerId,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Some(mut entry) = inflight_map().get_mut(&self.worker) {
            *entry = entry.saturating_sub(1);
        }
    }
}

/// Increment the in-flight count for `worker`, WARN the new depth, and return a
/// guard that decrements on drop. Returns `None` (no-op) when the flag is off
/// or `worker` is `None` (e.g. the Unavailable fallback path where the router
/// picks the worker internally and no target id is known at dispatch).
pub(super) fn inflight_guard(worker: Option<WorkerId>) -> Option<InflightGuard> {
    if !prefill_trace_enabled() {
        return None;
    }
    let worker = worker?;
    let inflight = {
        let mut entry = inflight_map().entry(worker).or_insert(0);
        *entry += 1;
        *entry
    };
    tracing::warn!(
        target: "dynamo_prefill_inflight",
        worker = worker,
        inflight = inflight,
        "prefill in-flight gauge (increment)"
    );
    Some(InflightGuard { worker })
}

/// Per-request lifecycle handle. Carries the timestamps captured in the
/// generate() path (arrival `a`, post-select `b`) across the spawn boundary
/// into `execute_prefill`, which fills in dispatch (`c`), first-response (`d`),
/// and stream-end (`e`) and emits the single per-request summary on drop-free
/// completion via [`PrefillTrace::emit`].
///
/// Cheap to construct ([`PrefillTrace::new`] returns `None` when the flag is
/// off), so callers thread `Option<PrefillTrace>` and skip all work when off.
pub(super) struct PrefillTrace {
    request_id: String,
    worker_id: Option<WorkerId>,
    isl_tokens: usize,
    /// a: request arrival/entry into PrefillRouter::generate.
    arrival: Instant,
    /// b: after find_best_match_details/resolve returned the CTX worker.
    selected: Instant,
    /// c: just before generate_to_worker dispatch (set in execute_prefill).
    dispatched: Option<Instant>,
    /// d: first response item from the CTX worker (set in execute_prefill).
    first_response: Option<Instant>,
}

impl PrefillTrace {
    /// Construct a trace anchored at arrival `a`. Returns `None` (zero work
    /// downstream) when `DYN_PREFILL_TRACE` is off.
    pub(super) fn new(request_id: &str, isl_tokens: usize, arrival: Instant) -> Option<Self> {
        if !prefill_trace_enabled() {
            return None;
        }
        Some(Self {
            request_id: request_id.to_string(),
            worker_id: None,
            isl_tokens,
            arrival,
            // selected is rewritten by mark_selected; default to arrival so
            // select_ms is 0 if a caller never calls it.
            selected: arrival,
            dispatched: None,
            first_response: None,
        })
    }

    /// Record b (worker resolved) and the target CTX worker id.
    pub(super) fn mark_selected(&mut self, worker_id: Option<WorkerId>) {
        self.selected = Instant::now();
        self.worker_id = worker_id;
    }

    /// Record c (just before generate_to_worker dispatch).
    pub(super) fn mark_dispatched(&mut self) {
        self.dispatched = Some(Instant::now());
    }

    /// Record d (first response item from the CTX worker), once.
    pub(super) fn mark_first_response(&mut self) {
        if self.first_response.is_none() {
            self.first_response = Some(Instant::now());
        }
    }

    /// Emit the single per-request summary. `e` (stream end) is `Instant::now()`
    /// at call time. Gaps:
    ///   select_ms        = b - a
    ///   dispatch_gap_ms  = c - b
    ///   prefill_first_ms = d - c
    ///   prefill_total_ms = e - c
    pub(super) fn emit(self) {
        let end = Instant::now();
        let select_ms = self.selected.duration_since(self.arrival).as_millis() as u64;
        // c falls back to b if a synchronous path never marked dispatch.
        let dispatched = self.dispatched.unwrap_or(self.selected);
        let dispatch_gap_ms = dispatched.duration_since(self.selected).as_millis() as u64;
        let prefill_first_ms = self
            .first_response
            .map(|d| d.duration_since(dispatched).as_millis() as u64);
        let prefill_total_ms = end.duration_since(dispatched).as_millis() as u64;

        tracing::warn!(
            target: "dynamo_prefill_trace",
            request_id = %self.request_id,
            worker = self.worker_id,
            isl_tokens = self.isl_tokens,
            select_ms = select_ms,
            dispatch_gap_ms = dispatch_gap_ms,
            prefill_first_ms = prefill_first_ms,
            prefill_total_ms = prefill_total_ms,
            "prefill request lifecycle"
        );
    }
}
