// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `metrics` module protocol: on-demand runtime snapshot.
//!
//! A small, dev-/test-oriented snapshot of the numbers an engineer most often
//! wants to eyeball without standing up Prometheus: per-pool block populations
//! (focus on G2; G3 included when present) and the count of in-flight disagg
//! sessions. Production observability is unchanged — each leader keeps
//! exporting the full Prometheus surface via
//! `kvbm_observability::start_metrics_server`. This handler is read-only and
//! sources its numbers from the same `prometheus::Registry`, so values match
//! exactly what Prometheus would scrape at the same instant.

use serde::{Deserialize, Serialize};

/// Velo handler name for the on-demand metrics snapshot.
pub const SNAPSHOT_HANDLER: &str = "kvbm.leader.control.metrics.snapshot";

/// Request — no parameters; the target leader is addressed by the velo call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshotRequest {}

/// A single leader's current runtime numbers.
///
/// Top-level fields are absolute counts at `gathered_at_unix_ms`; per-pool
/// breakdown carries one entry per tier the leader has configured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshotResponse {
    /// Wall-clock at gather time, milliseconds since the Unix epoch. Used by
    /// the UI to detect stale snapshots and by tests to assert monotonicity.
    pub gathered_at_unix_ms: u64,

    /// Number of disagg sessions currently held open by the leader's
    /// `SessionManager`. `0` when disagg is wired but idle. Not tier-scoped.
    pub sessions_inflight: u64,

    /// One entry per pool the leader has configured. Order is stable —
    /// G2 first, then G3 if present, then any further tiers sorted by their
    /// `pool` label. G1 (device) is filtered out: it is not part of the
    /// "what does this leader hold" picture the snapshot is meant to surface.
    pub pools: Vec<PoolBreakdown>,
}

/// Per-pool block populations.
///
/// All four fields are read from the leader's Prometheus `Registry` so they
/// match what `/metrics` would report:
///
/// | Field      | Metric                          |
/// |------------|---------------------------------|
/// | `mutable`  | `kvbm_inflight_mutable{pool}`   |
/// | `immutable`| `kvbm_inflight_immutable{pool}` |
/// | `reset`    | `kvbm_reset_pool_size{pool}`    |
/// | `inactive` | `kvbm_inactive_pool_size{pool}` |
///
/// The UI derives `pinned = mutable + immutable` and
/// `available = reset + inactive` (the latter matches `BlockManager::available`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PoolBreakdown {
    /// Pool label as it appears on the Prometheus `pool` label (e.g. `"G2"`).
    pub pool: String,
    /// MutableBlocks currently held outside the pool (in flight on the
    /// allocation side).
    pub mutable: u64,
    /// ImmutableBlocks currently held outside the pool (in flight on the
    /// readout side — e.g. matched and pinned by a live session).
    pub immutable: u64,
    /// Reset pool size — free blocks that have been zeroed and are immediately
    /// reusable.
    pub reset: u64,
    /// Inactive pool size — populated but evictable (LRU tail).
    pub inactive: u64,
}

#[cfg(feature = "client")]
pub use client::MetricsClient;

#[cfg(feature = "client")]
mod client {
    use super::*;
    use crate::control::ControlError;
    use crate::control::client::ControlChannel;

    /// Client for the opt-in `metrics` control module.
    #[derive(Clone)]
    pub struct MetricsClient {
        chan: ControlChannel,
    }

    impl MetricsClient {
        pub(crate) fn new(chan: ControlChannel) -> Self {
            Self { chan }
        }

        /// Fetch the leader's current runtime snapshot.
        pub async fn snapshot(&self) -> Result<MetricsSnapshotResponse, ControlError> {
            self.chan
                .call(SNAPSHOT_HANDLER, &MetricsSnapshotRequest::default())
                .await
        }
    }
}
