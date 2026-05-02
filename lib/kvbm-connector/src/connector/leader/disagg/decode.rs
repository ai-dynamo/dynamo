// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared decode-side types used by the coordinator and decode leader.
//!
//! The legacy `RemotePrefillCoordinator` and `DecodeCoordinator` trait have
//! been removed; the unified `ConditionalDisaggCoordinator` now owns all
//! decode-side per-request state and is held directly as a concrete
//! `Arc<ConditionalDisaggCoordinator>` by the leaders.

use std::sync::Arc;

use futures::future::BoxFuture;
use kvbm_disagg_protocol::SessionId;
use kvbm_engine::disagg::session::Session;

#[derive(Clone)]
pub struct BeginOutcome {
    pub session_id: SessionId,
    pub session: Arc<dyn Session>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemotePrefillStatus {
    Active,
    Failed,
    Released,
}

/// Sink the coordinator calls when a per-request session enters a
/// terminal failure state outside the normal cooperative-detach path.
///
/// Production wiring: `DecodeDisaggLeader` implements this and routes
/// the call to `cleanup_failed_request`, which fires
/// `worker_hook.mark_failed_onboarding` so vLLM can unblock the
/// pending async load.  Without the sink, the lifecycle watchdog
/// would evict the coordinator's session state on watchdog/Failed,
/// but vLLM would never learn the load failed and the request would
/// hang in `Onboarding` indefinitely.
pub trait CdFailureSink: Send + Sync {
    /// Called when the session for `request_id` fires a
    /// `LifecycleEvent::Failed` OR the watchdog fires before any
    /// terminal event arrives.  `reason` is a short human-readable
    /// description (the peer's failure reason or the watchdog
    /// fallback).  Implementations should be idempotent; the
    /// coordinator only invokes the sink once per request.
    fn on_session_failure(&self, request_id: String, reason: String) -> BoxFuture<'static, ()>;
}
