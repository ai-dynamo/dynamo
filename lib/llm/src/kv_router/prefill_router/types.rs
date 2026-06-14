// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::config::RouterConfigOverride;
use dynamo_kv_router::protocols::RouterBackpressureReason;
use dynamo_runtime::pipeline::OccupancyPermit;

use crate::protocols::common::preprocessor::{BootstrapInfo, PrefillResult, TraceLink};

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
pub(super) enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready.
    Bootstrap {
        bootstrap_info: BootstrapInfo,
        /// Prefill worker ID used for topology-aware decode routing.
        worker_id: u64,
    },
    /// Synchronous prefill completed with result.
    Completed {
        result: PrefillResult,
        /// Prefill worker ID when available for topology-aware decode routing.
        worker_id: Option<u64>,
        /// Prefill worker's `engine.generate` span pointer for the decode side
        /// to render as an OTel `Link` via `PreprocessedRequest.migration_link`.
        worker_link: Option<TraceLink>,
    },
}

pub(super) enum PrefillResolveDecision {
    Resolved {
        worker_id: u64,
        dp_rank: Option<u32>,
        bootstrap_info: BootstrapInfo,
        /// Occupancy booking for LL/P2C/DAW; held across the spawned prefill so
        /// `direct()` dispatch is reflected in the load counter. `None` for
        /// KV/RoundRobin/Random.
        permit: Option<OccupancyPermit>,
    },
    Unavailable,
    NotActivated,
    /// Bootstrap endpoint unavailable after a worker was selected.
    /// Carries the selected worker so the synchronous prefill path can commit
    /// that selection instead of re-entering router selection.
    NoBootstrapEndpoint {
        worker_id: u64,
        dp_rank: Option<u32>,
        /// Same occupancy booking as `Resolved`; the synchronous path holds it
        /// across `execute_prefill` so this branch tracks load too.
        permit: Option<OccupancyPermit>,
    },
    Backpressure {
        reason: RouterBackpressureReason,
        queued_isl_tokens: usize,
        max_queued_isl_tokens: Option<usize>,
    },
}

/// Structured outcome from `PrefillRouter::query_prefill_worker`.
///
/// Backpressure is surfaced as a variant rather than being collapsed into a
/// generic error so callers (Rust resolve_prefill_worker, C FFI shim) can
/// distinguish a queue-saturation reject from an ordinary lookup failure and
/// translate it into a retryable signal upstream.
pub enum PrefillQueryOutcome {
    Routed {
        worker_id: u64,
        dp_rank: Option<u32>,
        /// Atomic occupancy booking when the SimpleRouter selected via
        /// `select_and_reserve` (LL/P2C/DAW). `None` for KV (own accounting),
        /// RoundRobin/Random (peeked, no counter), and pinned workers.
        permit: Option<OccupancyPermit>,
    },
    Backpressure {
        reason: RouterBackpressureReason,
        queued_isl_tokens: usize,
        max_queued_isl_tokens: Option<usize>,
    },
}

pub(super) fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    }
}
