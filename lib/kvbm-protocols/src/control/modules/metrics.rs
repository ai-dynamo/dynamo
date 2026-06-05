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

use std::collections::BTreeMap;

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

    /// Conditional-disagg metrics, mirrored from the leader's `kvbm_cd_*`
    /// Prometheus surface. `None` when the leader exports no CD metrics (e.g. a
    /// pure remote-search leader) OR when talking to a leader built before this
    /// field existed (`#[serde(default)]` keeps the wire format compatible). The
    /// point of carrying these is so the hub `/v1/metrics` fan-out is a complete
    /// CD-observability surface WITHOUT each leader standing up its own
    /// `/metrics` axum on a fixed port.
    #[serde(default)]
    pub cd: Option<CdSnapshot>,
}

/// Conditional-disagg metrics mirrored from the leader's `kvbm_cd_*` Prometheus
/// families. All counters are absolute/cumulative at `gathered_at_unix_ms`;
/// labeled counters keep one map entry per label-value combination (label values
/// joined by `,` when a metric has more than one label). Sourced from the same
/// `prometheus::Registry` the per-leader `/metrics` exporter would serve, so the
/// numbers match a Prometheus scrape at the same instant.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CdSnapshot {
    /// `kvbm_cd_prefill_decisions_total`, keyed by the `decision` label
    /// (e.g. `"local_prefill"`, `"remote_prefill"`).
    pub prefill_decisions: BTreeMap<String, u64>,
    /// `kvbm_cd_remote_prefill_declined_total`, keyed by the decline `reason`.
    pub remote_prefill_declined: BTreeMap<String, u64>,
    /// `kvbm_cd_prefill_output_residual_total`, keyed by its label.
    pub prefill_output_residual: BTreeMap<String, u64>,
    /// `kvbm_cd_local_prefill_tokens_total`.
    pub local_prefill_tokens_total: u64,
    /// `kvbm_cd_remote_prefill_tokens_total`.
    pub remote_prefill_tokens_total: u64,
    /// `kvbm_cd_remote_prefill_window_tokens_total`.
    pub remote_prefill_window_tokens_total: u64,
    /// `kvbm_cd_prefill_computed_tokens_total`.
    pub prefill_computed_tokens_total: u64,
    /// `kvbm_cd_prefill_pulled_tokens_total`.
    pub prefill_pulled_tokens_total: u64,
    /// `kvbm_cd_prefill_local_hit_tokens_total` — prefill's own cache hit within
    /// decode's prefix window (over-pulled today; the PNCT opportunity).
    pub prefill_local_hit_tokens_total: u64,
    /// `kvbm_cd_remote_prefill_tokens` histogram (per-request remote prefill size).
    pub remote_prefill_tokens: HistogramSummary,
    /// `kvbm_cd_prefix_cache_hit_tokens` histogram.
    pub prefix_cache_hit_tokens: HistogramSummary,
    /// `kvbm_cd_prefill_computed_tokens` histogram.
    pub prefill_computed_tokens: HistogramSummary,
}

/// A Prometheus histogram reduced to its cumulative buckets + count + sum — the
/// full distribution, JSON-serializable for the velo snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HistogramSummary {
    /// Total observation count (`_count`).
    pub count: u64,
    /// Sum of all observed values (`_sum`).
    pub sum: f64,
    /// Cumulative buckets in `le` order (the implicit `+Inf` bucket equals `count`).
    pub buckets: Vec<HistogramBucket>,
}

/// One cumulative histogram bucket.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HistogramBucket {
    /// Upper bound (`le`).
    pub le: f64,
    /// Cumulative count of observations `<= le`.
    pub count: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The CD snapshot survives a JSON round-trip intact — this is the wire the
    /// new wheel (leader) serializes and the new hub deserializes in the
    /// `/v1/metrics` fan-out. Covers labeled-counter maps + a populated histogram.
    #[test]
    fn metrics_snapshot_cd_json_round_trip() {
        let cd = CdSnapshot {
            prefill_decisions: BTreeMap::from([
                ("local_prefill".to_string(), 12u64),
                ("remote_prefill".to_string(), 7u64),
            ]),
            remote_prefill_tokens_total: 9000,
            prefill_computed_tokens_total: 6400,
            prefill_pulled_tokens_total: 41000,
            prefill_local_hit_tokens_total: 12000,
            remote_prefill_tokens: HistogramSummary {
                count: 7,
                sum: 9000.0,
                buckets: vec![
                    HistogramBucket { le: 256.0, count: 2 },
                    HistogramBucket {
                        le: 1024.0,
                        count: 7,
                    },
                ],
            },
            ..Default::default()
        };
        let snap = MetricsSnapshotResponse {
            gathered_at_unix_ms: 1_700_000_000_000,
            sessions_inflight: 3,
            pools: vec![],
            cd: Some(cd.clone()),
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: MetricsSnapshotResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.cd, Some(cd));
    }

    /// A snapshot from a leader/hub built before `cd` existed omits the field;
    /// `#[serde(default)]` must deserialize it to `None` (mixed-version safety).
    #[test]
    fn metrics_snapshot_cd_defaults_to_none_when_absent() {
        let json = r#"{"gathered_at_unix_ms":1,"sessions_inflight":0,"pools":[]}"#;
        let snap: MetricsSnapshotResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(snap.cd, None);
    }
}
