// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `metrics` control module: on-demand runtime snapshot.
//!
//! Opt-in via `control.metrics = true` *and* requires the leader to have
//! been built with a [`SharedKvbmObservability`] handle (i.e. via
//! `InstanceLeaderBuilder::with_runtime`). Reads the same Prometheus
//! [`Registry`](prometheus::Registry) that
//! `kvbm_observability::start_metrics_server` exposes in production, so
//! values match what `/metrics` would report at the same instant.
//!
//! Wire types live in `kvbm_protocols::control::modules::metrics`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use prometheus::proto::{LabelPair, MetricFamily};
use velo::{Handler, Messenger};

use kvbm_observability::SharedKvbmObservability;
use kvbm_protocols::control::modules::metrics::{
    MetricsSnapshotRequest, MetricsSnapshotResponse, PoolBreakdown, SNAPSHOT_HANDLER,
};
use kvbm_protocols::control::{ControlReply, ModuleId};

use crate::leader::InstanceLeader;
use crate::leader::control::ControlModule;

/// The `metrics` control module.
pub struct MetricsModule {
    leader: Arc<InstanceLeader>,
    observability: SharedKvbmObservability,
}

impl MetricsModule {
    /// Build a metrics module backed by `leader` and the shared observability
    /// registry. The observability handle is typically
    /// `leader.observability().expect(...)` — the caller decides whether to
    /// register the module based on `leader.observability().is_some()`.
    pub fn new(leader: Arc<InstanceLeader>, observability: SharedKvbmObservability) -> Self {
        Self {
            leader,
            observability,
        }
    }
}

impl ControlModule for MetricsModule {
    fn id(&self) -> ModuleId {
        ModuleId::Metrics
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        let leader = self.leader.clone();
        let obs = self.observability.clone();
        let handler = Handler::typed_unary_async(SNAPSHOT_HANDLER, move |ctx| {
            let leader = leader.clone();
            let obs = obs.clone();
            async move {
                let _req: MetricsSnapshotRequest = ctx.input;
                let snapshot = build_snapshot(&leader, &obs);
                let reply: ControlReply<MetricsSnapshotResponse> = ControlReply::Ok(snapshot);
                Ok::<ControlReply<MetricsSnapshotResponse>, anyhow::Error>(reply)
            }
        })
        .build();
        messenger
            .register_handler(handler)
            .map_err(|e| anyhow::anyhow!("velo register_handler({SNAPSHOT_HANDLER}): {e}"))?;
        Ok(())
    }
}

fn build_snapshot(
    leader: &InstanceLeader,
    obs: &SharedKvbmObservability,
) -> MetricsSnapshotResponse {
    let pools = roll_up_pool_gauges(&obs.registry().gather());
    let sessions_inflight = leader.session_manager().len() as u64;
    let gathered_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    MetricsSnapshotResponse {
        gathered_at_unix_ms,
        sessions_inflight,
        pools,
    }
}

/// Which of the four block-population gauges a `MetricFamily` represents.
#[derive(Clone, Copy)]
enum GaugeKind {
    Mutable,
    Immutable,
    Reset,
    Inactive,
}

impl GaugeKind {
    fn from_family_name(name: &str) -> Option<Self> {
        match name {
            "kvbm_inflight_mutable" => Some(Self::Mutable),
            "kvbm_inflight_immutable" => Some(Self::Immutable),
            "kvbm_reset_pool_size" => Some(Self::Reset),
            "kvbm_inactive_pool_size" => Some(Self::Inactive),
            _ => None,
        }
    }
}

/// Fold the four per-pool gauges across all `MetricFamily`s into one
/// [`PoolBreakdown`] per pool label.
///
/// `G1` (device tier) is filtered out — the snapshot answers "what does this
/// leader hold in host/disk pools" and `G1` is not owned by the leader's
/// `BlockManager`. Output order is stable: `G2` first, then `G3`, then any
/// other pools sorted lexicographically (so test fixtures with synthetic
/// pool labels remain deterministic).
fn roll_up_pool_gauges(families: &[MetricFamily]) -> Vec<PoolBreakdown> {
    let mut by_pool: BTreeMap<String, PoolBreakdown> = BTreeMap::new();
    for family in families {
        let Some(kind) = GaugeKind::from_family_name(family.name()) else {
            continue;
        };
        for metric in family.get_metric() {
            let pool = match metric
                .get_label()
                .iter()
                .find(|l: &&LabelPair| l.name() == "pool")
            {
                Some(label) => label.value().to_string(),
                None => continue,
            };
            if pool == "G1" {
                continue;
            }
            // Gauges are f64; pool sizes are always non-negative integers in
            // practice. Clamp to 0 to guard against a future negative gauge.
            let value = metric.get_gauge().value().max(0.0) as u64;
            let entry = by_pool.entry(pool.clone()).or_insert(PoolBreakdown {
                pool,
                mutable: 0,
                immutable: 0,
                reset: 0,
                inactive: 0,
            });
            match kind {
                GaugeKind::Mutable => entry.mutable = value,
                GaugeKind::Immutable => entry.immutable = value,
                GaugeKind::Reset => entry.reset = value,
                GaugeKind::Inactive => entry.inactive = value,
            }
        }
    }
    let mut out: Vec<PoolBreakdown> = by_pool.into_values().collect();
    out.sort_by(|a, b| {
        pool_rank(&a.pool)
            .cmp(&pool_rank(&b.pool))
            .then_with(|| a.pool.cmp(&b.pool))
    });
    out
}

/// Stable sort rank: G2 first, G3 second, everything else after.
fn pool_rank(pool: &str) -> u8 {
    match pool {
        "G2" => 0,
        "G3" => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::proto::{Gauge, LabelPair, Metric, MetricFamily, MetricType};

    fn family(name: &str, samples: &[(&str, f64)]) -> MetricFamily {
        let mut mf = MetricFamily::default();
        mf.set_name(name.to_string());
        mf.set_field_type(MetricType::GAUGE);
        let metrics: Vec<Metric> = samples
            .iter()
            .map(|(pool, value)| {
                let mut m = Metric::default();
                let mut l = LabelPair::default();
                l.set_name("pool".to_string());
                l.set_value(pool.to_string());
                m.set_label(vec![l]);
                let mut g = Gauge::default();
                g.set_value(*value);
                m.set_gauge(g);
                m
            })
            .collect();
        mf.set_metric(metrics);
        mf
    }

    #[test]
    fn filters_g1_and_orders_g2_before_g3() {
        let fams = vec![
            family(
                "kvbm_inflight_mutable",
                &[("G1", 1.0), ("G3", 5.0), ("G2", 2.0)],
            ),
            family(
                "kvbm_inflight_immutable",
                &[("G1", 7.0), ("G2", 3.0), ("G3", 11.0)],
            ),
            family("kvbm_reset_pool_size", &[("G2", 100.0), ("G3", 200.0)]),
            family("kvbm_inactive_pool_size", &[("G2", 50.0), ("G3", 150.0)]),
            // Unrelated family must be ignored.
            family("kvbm_allocations_total", &[("G2", 999.0)]),
        ];
        let pools = roll_up_pool_gauges(&fams);

        assert_eq!(pools.len(), 2, "G1 dropped, G2 and G3 retained");
        assert_eq!(pools[0].pool, "G2");
        assert_eq!(pools[1].pool, "G3");

        assert_eq!(pools[0].mutable, 2);
        assert_eq!(pools[0].immutable, 3);
        assert_eq!(pools[0].reset, 100);
        assert_eq!(pools[0].inactive, 50);

        assert_eq!(pools[1].mutable, 5);
        assert_eq!(pools[1].immutable, 11);
        assert_eq!(pools[1].reset, 200);
        assert_eq!(pools[1].inactive, 150);
    }

    #[test]
    fn missing_g3_yields_only_g2_row() {
        let fams = vec![
            family("kvbm_inflight_mutable", &[("G2", 1.0)]),
            family("kvbm_inflight_immutable", &[("G2", 2.0)]),
            family("kvbm_reset_pool_size", &[("G2", 3.0)]),
            family("kvbm_inactive_pool_size", &[("G2", 4.0)]),
        ];
        let pools = roll_up_pool_gauges(&fams);
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].pool, "G2");
    }

    #[test]
    fn empty_families_yield_empty_pools() {
        let pools = roll_up_pool_gauges(&[]);
        assert!(pools.is_empty());
    }

    #[test]
    fn metric_without_pool_label_is_skipped() {
        let mut mf = MetricFamily::default();
        mf.set_name("kvbm_inflight_mutable".to_string());
        mf.set_field_type(MetricType::GAUGE);
        let mut m = Metric::default();
        // No `pool` label.
        let mut g = Gauge::default();
        g.set_value(42.0);
        m.set_gauge(g);
        mf.set_metric(vec![m]);
        let pools = roll_up_pool_gauges(&[mf]);
        assert!(pools.is_empty(), "unlabeled metrics should be skipped");
    }

    #[test]
    fn negative_gauge_value_clamps_to_zero() {
        let fams = vec![family("kvbm_inflight_mutable", &[("G2", -3.0)])];
        let pools = roll_up_pool_gauges(&fams);
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].mutable, 0);
    }
}
