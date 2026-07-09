// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`SnapshotPublisher`] — the single push surface for per-rank component
//! snapshots.
//!
//! Replaces the previous pull-driven model (tokio poll task at 100 ms calls
//! a Python closure that does a dict read). Engines now call
//! [`SnapshotPublisher::publish`] directly from their stat-logger threads;
//! the call fans out to two consumers inline:
//!
//! 1. **Rust [`ComponentGauges`]** — atomic gauge writes for the
//!    `dynamo_component_*` `/metrics` surface, no GIL required.
//! 2. **`WorkerMetricsPublisher`** (one per rank) — NATS publish of
//!    `kv_used_blocks` for the KV router's load signal.
//!
//! No tokio task. No 100 ms cadence. No Python registry on the unified
//! path. Event-driven flushing — every engine push immediately updates
//! both consumers.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_llm::kv_router::publisher::WorkerMetricsPublisher;
use parking_lot::Mutex;

use crate::engine::ComponentSnapshot;
use crate::metrics::ComponentGauges;

/// Fan-out target for per-rank `ComponentSnapshot` writes from engine
/// stat-loggers. Stable for the engine's lifetime — engines stash the
/// `Arc<SnapshotPublisher>` once during setup and call `publish` from
/// any thread thereafter.
pub struct SnapshotPublisher {
    gauges: Arc<ComponentGauges>,
    router_publishers: HashMap<u32, Arc<WorkerMetricsPublisher>>,
    /// First-failure log per rank; suppresses noise on sustained NATS
    /// outages.
    warned_ranks: Mutex<HashSet<u32>>,
    /// Number of engine source blocks represented by one router block.
    block_ratio: u64,
}

impl SnapshotPublisher {
    pub fn new(
        gauges: Arc<ComponentGauges>,
        router_publishers: HashMap<u32, Arc<WorkerMetricsPublisher>>,
    ) -> Self {
        Self {
            gauges,
            router_publishers,
            warned_ranks: Mutex::new(HashSet::new()),
            block_ratio: 1,
        }
    }

    pub(crate) fn new_with_block_ratio(
        gauges: Arc<ComponentGauges>,
        router_publishers: HashMap<u32, Arc<WorkerMetricsPublisher>>,
        block_ratio: u64,
    ) -> Self {
        debug_assert!(block_ratio > 0);
        Self {
            gauges,
            router_publishers,
            warned_ranks: Mutex::new(HashSet::new()),
            block_ratio,
        }
    }

    /// Push a snapshot for `dp_rank`. Atomically updates the per-rank
    /// `dynamo_component_*` gauges and emits the `kv_used_blocks` router
    /// signal. Hot path — no allocation, no GIL acquisition.
    ///
    /// Ranks not declared at construction are silently dropped (engine
    /// emitting for an unknown rank is a misconfiguration the framework
    /// can't recover from cleanly).
    pub fn publish(&self, dp_rank: u32, snap: ComponentSnapshot) {
        let snap = self.scale_snapshot(snap);
        self.gauges.update(&snap);
        if let Some(rp) = self.router_publishers.get(&dp_rank)
            && let Err(e) = rp.publish(Some(dp_rank), None, Some(snap.kv_used_blocks))
        {
            if self.warned_ranks.lock().insert(dp_rank) {
                tracing::warn!(dp_rank, error = %e, "router signal publish failed; suppressing further");
            } else {
                tracing::debug!(dp_rank, error = %e, "router signal publish failed");
            }
        }
    }

    fn scale_snapshot(&self, mut snap: ComponentSnapshot) -> ComponentSnapshot {
        snap.kv_used_blocks = snap.kv_used_blocks.div_ceil(self.block_ratio);
        snap.kv_total_blocks /= self.block_ratio;
        snap
    }

    /// Declared dp_ranks. Stable for the publisher's lifetime.
    pub fn dp_ranks(&self) -> Vec<u32> {
        let mut ranks: Vec<u32> = self.router_publishers.keys().copied().collect();
        ranks.sort_unstable();
        ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ComponentSnapshot;
    use crate::metrics::{ComponentGauges, EngineMetrics, TestHierarchy};

    fn make_publisher() -> SnapshotPublisher {
        let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
        let gauges = Arc::new(ComponentGauges::new(&metrics, &[0]).expect("component gauges"));
        SnapshotPublisher::new(gauges, HashMap::new())
    }

    #[test]
    fn coalescing_scales_capacity_down_and_usage_up() {
        let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
        let gauges = Arc::new(ComponentGauges::new(&metrics, &[0]).expect("component gauges"));
        let publisher = SnapshotPublisher::new_with_block_ratio(gauges, HashMap::new(), 16);
        let scaled = publisher.scale_snapshot(ComponentSnapshot {
            kv_used_blocks: 33,
            kv_total_blocks: 33,
            ..ComponentSnapshot::default()
        });
        assert_eq!(scaled.kv_used_blocks, 3);
        assert_eq!(scaled.kv_total_blocks, 2);
    }

    /// Publishing for an undeclared rank is a no-op (no panic, no crash).
    /// Production safety: a misbehaving engine emitting a stray rank
    /// shouldn't tear the worker down.
    #[test]
    fn publish_unknown_rank_is_noop() {
        let publisher = make_publisher();
        publisher.publish(
            42,
            ComponentSnapshot {
                kv_used_blocks: 5,
                kv_total_blocks: 100,
                gpu_cache_usage: 0.5,
                kv_cache_hit_rate: Some(0.3),
                dp_rank: 42,
            },
        );
        // No panic — test passes.
    }

    /// `publish` updates gauges synchronously — operators see the new
    /// values on the next /metrics scrape with no poll-task delay.
    #[test]
    fn publish_updates_gauges_synchronously() {
        let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
        let gauges = Arc::new(ComponentGauges::new(&metrics, &[0]).expect("component gauges"));
        let publisher = SnapshotPublisher::new(gauges, HashMap::new());

        publisher.publish(
            0,
            ComponentSnapshot {
                kv_used_blocks: 7,
                kv_total_blocks: 100,
                gpu_cache_usage: 0.07,
                kv_cache_hit_rate: Some(0.25),
                dp_rank: 0,
            },
        );

        let text = metrics
            .hierarchy()
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .expect("expfmt");
        assert!(
            text.contains("dynamo_component_total_blocks") && text.contains("100"),
            "total_blocks not in /metrics: {text}"
        );
        assert!(
            text.contains("dynamo_component_gpu_cache_usage_percent"),
            "gpu_cache_usage_percent not in /metrics: {text}"
        );
    }

    /// Constructor seeds each rank at zero so empty `GaugeVec` families
    /// still render. Without this, the prometheus encoder skips them and
    /// `/metrics` is missing `total_blocks` / `gpu_cache_usage_percent`.
    #[test]
    fn seeded_ranks_render_in_metrics() {
        let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
        let _gauges = ComponentGauges::new(&metrics, &[0, 1]).expect("component gauges");
        let text = metrics
            .hierarchy()
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .expect("expfmt");
        assert!(
            text.contains(r#"dynamo_component_total_blocks{"#) && text.contains(r#"dp_rank="0""#),
            "rank 0 total_blocks not seeded: {text}"
        );
        assert!(
            text.contains(r#"dp_rank="1""#),
            "rank 1 total_blocks not seeded: {text}"
        );
        // kv_cache_hit_rate is intentionally not seeded (tri-state None).
        assert!(
            !text.contains("dynamo_component_kv_cache_hit_rate"),
            "kv_cache_hit_rate should not be seeded: {text}"
        );
    }
}
