// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EPP-side driver for the router's request-classifier lifecycle (DEP #13891).
//!
//! # Why this exists
//!
//! The router's pluggable flow control (`RequestClassifier::classify` /
//! `on_event`) only runs for a request that has a **registered lifecycle**.
//! `LocalScheduler::classify_request` gates on two conditions, and the EPP
//! currently fails both:
//!
//! 1. `request.mode.tracked_request_id()` must be `Some`. The EPP calls
//!    [`ManagedKvRouter::find_best_match`] with `context_id: None` and
//!    `update_states: false`, which yields `ScheduleMode::QueryOnly`, whose
//!    `tracked_request_id()` is `None`.
//! 2. `classifier.has_request(request_id)` must be true. That is only set by
//!    `begin_request`, reached through `KvRouter::begin_request_lifecycle`.
//!
//! Consequence today: installing a classifier alongside the EPP is silently
//! inert — no `classify()`, no `ClassifyEvent`, and no warning.
//!
//! # What this module provides
//!
//! [`EppRequestLifecycle`] is the EPP's counterpart to the frontend's
//! `routing_host::request_guard`. The frontend drives the router's
//! `RequestLifecycle` from the response stream it owns; the EPP has no response
//! stream, so the driver is instead pumped from the ext_proc stream in
//! [`crate::server`], whose `RequestContext` has exactly the right lifetime.
//!
//! The EPP already produces every signal the event set needs:
//!
//! | Router guard method       | EPP signal                                            |
//! |---------------------------|-------------------------------------------------------|
//! | `begin_request`           | `x-request-id` resolved or minted (request headers)   |
//! | `selected(worker)`        | `Router::route_decode` returns                        |
//! | `sent(worker)`            | routing headers returned to Envoy                     |
//! | `responding()`            | first non-empty `ResponseBody` chunk                  |
//! | `observe_context_tokens`  | `usage` parsed from the SSE or unary response body    |
//! | `complete()`              | ext_proc stream teardown with usage                   |
//! | `abort()`                 | Envoy disconnect (`tx.closed()`), or guard drop       |
//!
//! Adopting the guard also closes `TODO(epp-atomic-admission)`: the router's
//! `impl Drop for RequestLifecycle` calls `abort(None)`, so dropping `pick()`
//! on an Envoy disconnect retracts the admission instead of leaving
//! `add_request` half-applied.
//!
//! # Router-side gaps this scaffold depends on
//!
//! Nothing in this module can be implemented until the following land in
//! `dynamo-llm` / `dynamo-kv-router`. Each is called out again at its use site.
//!
//! - **TODO(router-lifecycle-visibility):** `KvRouter::begin_request_lifecycle`
//!   (`lib/llm/src/kv_router.rs`) and `KvScheduler::begin_request_lifecycle`
//!   (`lib/llm/src/kv_router/scheduler.rs`) are `pub(crate)`. `dynamo-ext-proc`
//!   is a separate crate and cannot call them at any value of
//!   `track_lifecycle`. They must become `pub`, or a `pub` wrapper must be
//!   exposed on `ManagedKvRouter`.
//!
//! - **TODO(router-lifecycle-entrypoint):** there is no public selection entry
//!   point that sets `track_lifecycle: true`. `find_best_match` hardcodes
//!   `FindBestMatchAdmission::WithAdmission { track_lifecycle: false }`
//!   (`lib/llm/src/kv_router.rs`), and `FindBestMatchAdmission` itself is
//!   `pub(super)`, so an external caller cannot even name the enum. A public
//!   entry point taking `context_id: Some(request_id)`, `update_states: true`
//!   and `track_lifecycle: true` is required.
//!
//! - **TODO(router-lifecycle-export):** `RequestLifecycle` is
//!   `#[doc(hidden)]` and `ClassifyEvent` / `RequestClassifier` are not
//!   re-exported from `dynamo-llm`'s public surface. The EPP needs a stable
//!   path to the guard type to hold it in [`EppRequestLifecycle`].
//!
//! - **TODO(router-classifier-installer):** `KvRouter::with_request_classifier`
//!   and `KvScheduler::install_request_classifier` are `#[doc(hidden)]` and
//!   carry "TODO: wire a production installer". The EPP has no supported way
//!   to install a classifier even once the above are addressed. Related:
//!   `runner.rs` rejects linked worker-selection policies unless
//!   `DYN_EPP_MODE=standalone`, which must be reconciled with plugins being
//!   usable in dynamo mode.
//!
//! - **TODO(router-sent-semantics):** the router's `Sent` event means
//!   "dispatched to the worker". The EPP can only report "routing headers
//!   returned to Envoy", which is a *commit*, not a send — Envoy may still fail
//!   to connect and never dispatch. The router tolerates the inverse case (a
//!   terminal event with no prior `Sent`) but not a `Sent` that never became a
//!   dispatch. The event needs either a documented weaker meaning for
//!   non-dispatching routers or a distinct `Committed` variant.
//!
//! - **TODO(router-prefill-lifecycle):** the guard tracks a single worker.
//!   Disaggregated EPP places a prefill worker and a decode worker and emits
//!   both as headers. There is no way to register or terminate the prefill leg
//!   (see the existing `TODO(epp-prefill-booking)` in [`crate::epp`]).

use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::picker::ResponseUsage;

/// Placeholder for `dynamo_llm::kv_router::scheduling::RequestLifecycle`.
///
/// TODO(router-lifecycle-export): replace with the real guard once it is
/// exported. Kept as a distinct type so every call site below already has the
/// correct shape and only the import has to change.
#[derive(Debug)]
pub struct RouterRequestLifecycle;

/// Terminal cause handed to the router when a request ends without completing.
///
/// Mirrors the router's `AbortCause` (`dyn Error + Send + Sync + 'static`).
pub type AbortCause = dyn std::error::Error + Send + Sync + 'static;

/// Why an EPP-driven lifecycle reached its terminal state.
#[derive(Debug)]
pub enum LifecycleOutcome {
    /// The ext_proc stream ended normally after the worker responded.
    Completed { usage: Option<ResponseUsage> },
    /// Envoy disconnected, the pick failed, or the guard was dropped.
    Aborted { error: Option<Arc<AbortCause>> },
}

/// EPP-side driver for one request's router lifecycle.
///
/// Owned by the ext_proc `RequestContext` for the life of the stream. Every
/// method is infallible and idempotent so the ext_proc state machine can call
/// them without branching on prior phase — the router guard already ignores
/// out-of-order transitions.
#[derive(Debug)]
pub struct EppRequestLifecycle {
    /// `None` when no classifier is installed, so the driver degrades to a
    /// no-op rather than forcing every call site to branch.
    guard: Option<Box<RouterRequestLifecycle>>,
}

impl EppRequestLifecycle {
    /// Register `request_id` with the classifier and take ownership of the
    /// resulting guard.
    ///
    /// TODO(router-lifecycle-visibility): needs a public
    /// `begin_request_lifecycle` on `ManagedKvRouter`.
    ///
    /// TODO(epp-request-id-collision): the router's `begin_request` returns
    /// `DuplicateClassificationRequestId` when the id is already live. The EPP
    /// keys on the client-supplied `x-request-id`, so two concurrent clients
    /// sharing one turns into a hard error. Decide whether the EPP mints its
    /// own id (and what then correlates `add_request` / `free`) or surfaces the
    /// duplicate as a 400.
    pub fn begin(_request_id: &str) -> Self {
        todo!("blocked on router-lifecycle-visibility")
    }

    /// A lifecycle for a request that will not be classified — all transitions
    /// are no-ops. Used when no classifier is installed.
    pub fn disabled() -> Self {
        Self { guard: None }
    }

    /// True when this driver is backed by a real router guard.
    pub fn is_active(&self) -> bool {
        self.guard.is_some()
    }

    /// Record the decode worker chosen by selection, before any dispatch
    /// commitment. Maps to `RequestLifecycle::selected`.
    pub fn selected(&mut self, _worker: WorkerWithDpRank) {
        todo!("blocked on router-lifecycle-export")
    }

    /// Record that routing headers were handed back to Envoy.
    ///
    /// TODO(router-sent-semantics): this is a commit, not a send. See the
    /// module docs.
    pub fn committed(&mut self, _worker: WorkerWithDpRank) {
        todo!("blocked on router-lifecycle-export")
    }

    /// Record the prefill worker for a disaggregated placement.
    ///
    /// TODO(router-prefill-lifecycle): the router guard holds one worker and
    /// has no prefill leg, so this cannot be forwarded today.
    pub fn prefill_selected(&mut self, _worker: WorkerWithDpRank) {
        todo!("blocked on router-prefill-lifecycle")
    }

    /// Record the first non-empty response chunk observed from Envoy. Maps to
    /// `RequestLifecycle::responding`.
    pub fn responding(&mut self) {
        todo!("blocked on router-lifecycle-export")
    }

    /// Raise the recorded context total from parsed `usage`.
    ///
    /// TODO(epp-usage-availability): for streaming responses the EPP only sees
    /// `usage` when the client sent `stream_options.include_usage`, so
    /// `ClassifyEvent::Completed.context_tokens` is frequently `None` here.
    /// `observe_output_tokens` is not an alternative — the EPP does not
    /// tokenize output. The DEP should document `context_tokens` as
    /// best-effort.
    pub fn observe_usage(&mut self, _usage: &ResponseUsage) {
        todo!("blocked on router-lifecycle-export")
    }

    /// Drive the lifecycle to its terminal state. Maps to
    /// `RequestLifecycle::complete` or `::abort`.
    pub fn finish(&mut self, _outcome: LifecycleOutcome) {
        todo!("blocked on router-lifecycle-export")
    }
}

impl Drop for EppRequestLifecycle {
    /// The router guard's own `Drop` emits `Aborted`, which is what retracts an
    /// admission when Envoy disconnects mid-`pick()`. Dropping this wrapper
    /// must therefore drop the inner guard rather than suppress it.
    fn drop(&mut self) {
        // Intentionally empty: the inner guard's Drop does the work.
    }
}
