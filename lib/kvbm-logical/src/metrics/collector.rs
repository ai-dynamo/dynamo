// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom `prometheus::core::Collector` that reads raw atomics at scrape time.
//!
//! External labels (e.g. `instance_id`, `worker_id`) are appended at collection time,
//! not baked in at metric creation time.

use std::sync::{Arc, RwLock};

use prometheus::core::{Collector, Desc};
use prometheus::proto::{Gauge, LabelPair, Metric, MetricFamily, MetricType};

use super::pool_metrics::BlockPoolMetrics;

/// Metric definitions: (name, help, type).
const COUNTER_DEFS: &[(&str, &str)] = &[
    (
        "kvbm_allocations_total",
        "Total blocks allocated from pools",
    ),
    (
        "kvbm_allocations_from_reset_total",
        "Total blocks allocated from the reset pool",
    ),
    (
        "kvbm_evictions_total",
        "Total blocks evicted from inactive pool",
    ),
    (
        "kvbm_registrations_total",
        "Total blocks registered (CompleteBlock -> ImmutableBlock)",
    ),
    (
        "kvbm_duplicate_blocks_total",
        "Total duplicate blocks created (Allow policy)",
    ),
    (
        "kvbm_registration_dedup_total",
        "Total block registrations deduplicated (Reject policy)",
    ),
    (
        "kvbm_stagings_total",
        "Total MutableBlock -> CompleteBlock transitions",
    ),
    (
        "kvbm_match_hashes_requested_total",
        "Total hashes requested in match_blocks calls",
    ),
    (
        "kvbm_match_blocks_returned_total",
        "Total blocks returned from match_blocks calls",
    ),
    (
        "kvbm_scan_hashes_requested_total",
        "Total hashes requested in scan_matches calls",
    ),
    (
        "kvbm_scan_blocks_returned_total",
        "Total blocks returned from scan_matches calls",
    ),
];

const GAUGE_DEFS: &[(&str, &str)] = &[
    (
        "kvbm_inflight_mutable",
        "Current MutableBlocks held outside pool",
    ),
    (
        "kvbm_inflight_immutable",
        "Current ImmutableBlocks held outside pool",
    ),
    ("kvbm_reset_pool_size", "Current reset pool size"),
    ("kvbm_inactive_pool_size", "Current inactive pool size"),
];

/// Aggregates metrics from multiple `BlockPoolMetrics` sources and exports
/// them as Prometheus `MetricFamily` protos with per-pool-type labels.
#[derive(Clone)]
pub struct MetricsAggregator {
    inner: Arc<Inner>,
}

struct Inner {
    sources: RwLock<Vec<Arc<BlockPoolMetrics>>>,
    external_labels: RwLock<Vec<(String, String)>>,
    descs: Vec<Desc>,
}

impl MetricsAggregator {
    /// Create a new `MetricsAggregator`.
    pub fn new() -> Self {
        let mut descs = Vec::with_capacity(COUNTER_DEFS.len() + GAUGE_DEFS.len());
        for (name, help) in COUNTER_DEFS {
            descs.push(
                Desc::new(
                    name.to_string(),
                    help.to_string(),
                    vec!["pool".to_string()],
                    Default::default(),
                )
                .expect("valid desc"),
            );
        }
        for (name, help) in GAUGE_DEFS {
            descs.push(
                Desc::new(
                    name.to_string(),
                    help.to_string(),
                    vec!["pool".to_string()],
                    Default::default(),
                )
                .expect("valid desc"),
            );
        }

        Self {
            inner: Arc::new(Inner {
                sources: RwLock::new(Vec::new()),
                external_labels: RwLock::new(Vec::new()),
                descs,
            }),
        }
    }

    /// Register a `BlockPoolMetrics` source (called by `BlockManager::build()`).
    pub fn register_source(&self, source: Arc<BlockPoolMetrics>) {
        self.inner
            .sources
            .write()
            .expect("sources lock poisoned")
            .push(source);
    }

    /// Set external labels appended at scrape time (e.g. `instance_id`, `worker_id`).
    pub fn set_external_labels(&self, labels: Vec<(String, String)>) {
        *self
            .inner
            .external_labels
            .write()
            .expect("external_labels lock poisoned") = labels;
    }

    /// Register this collector with a `prometheus::Registry`.
    pub fn register_with(&self, registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.clone()))
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for MetricsAggregator {
    fn desc(&self) -> Vec<&Desc> {
        self.inner.descs.iter().collect()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        let sources = self.inner.sources.read().expect("sources lock poisoned");
        let ext_labels = self
            .inner
            .external_labels
            .read()
            .expect("external_labels lock poisoned");

        let mut families: Vec<MetricFamily> = Vec::new();

        for source in sources.iter() {
            let snap = source.snapshot();
            let pool_label = source.type_label();

            let mut base_labels: Vec<LabelPair> = Vec::with_capacity(1 + ext_labels.len());
            let mut pool_lp = LabelPair::default();
            pool_lp.set_name("pool".to_string());
            pool_lp.set_value(pool_label.to_string());
            base_labels.push(pool_lp);
            for (k, v) in ext_labels.iter() {
                let mut lp = LabelPair::default();
                lp.set_name(k.clone());
                lp.set_value(v.clone());
                base_labels.push(lp);
            }

            // Counter values in order matching COUNTER_DEFS
            let counter_values: [u64; 11] = [
                snap.allocations,
                snap.allocations_from_reset,
                snap.evictions,
                snap.registrations,
                snap.duplicate_blocks,
                snap.registration_dedup,
                snap.stagings,
                snap.match_hashes_requested,
                snap.match_blocks_returned,
                snap.scan_hashes_requested,
                snap.scan_blocks_returned,
            ];

            for (i, (name, help)) in COUNTER_DEFS.iter().enumerate() {
                let mut m = Metric::default();
                m.set_label(base_labels.clone());
                let mut c = prometheus::proto::Counter::default();
                c.set_value(counter_values[i] as f64);
                m.set_counter(c);

                let mut mf = MetricFamily::default();
                mf.set_name(name.to_string());
                mf.set_help(help.to_string());
                mf.set_field_type(MetricType::COUNTER);
                mf.set_metric(vec![m]);
                families.push(mf);
            }

            // Gauge values in order matching GAUGE_DEFS
            let gauge_values: [i64; 4] = [
                snap.inflight_mutable,
                snap.inflight_immutable,
                snap.reset_pool_size,
                snap.inactive_pool_size,
            ];

            for (i, (name, help)) in GAUGE_DEFS.iter().enumerate() {
                let mut m = Metric::default();
                m.set_label(base_labels.clone());
                let mut g = Gauge::default();
                g.set_value(gauge_values[i] as f64);
                m.set_gauge(g);

                let mut mf = MetricFamily::default();
                mf.set_name(name.to_string());
                mf.set_help(help.to_string());
                mf.set_field_type(MetricType::GAUGE);
                mf.set_metric(vec![m]);
                families.push(mf);
            }
        }

        // Merge families with the same name (when multiple sources)
        if sources.len() > 1 {
            let mut merged: Vec<MetricFamily> = Vec::new();
            for mut family in families {
                if let Some(existing) = merged.iter_mut().find(|f| f.name() == family.name()) {
                    existing.mut_metric().extend(family.take_metric());
                } else {
                    merged.push(family);
                }
            }
            merged
        } else {
            families
        }
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use prometheus::core::Collector;

    #[test]
    fn test_empty_aggregator_collects_nothing() {
        let agg = MetricsAggregator::new();
        let families = agg.collect();
        assert!(families.is_empty());
    }

    #[test]
    fn test_single_source_collect() {
        let agg = MetricsAggregator::new();
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));

        metrics.inc_allocations(10);
        metrics.inc_evictions(3);
        metrics.set_reset_pool_size(42);

        agg.register_source(metrics);

        let families = agg.collect();
        assert_eq!(families.len(), COUNTER_DEFS.len() + GAUGE_DEFS.len());

        // Find allocations counter
        let alloc_family = families
            .iter()
            .find(|f| f.get_name() == "kvbm_allocations_total")
            .expect("should have allocations family");
        assert_eq!(alloc_family.get_field_type(), MetricType::COUNTER);
        let m = &alloc_family.get_metric()[0];
        assert_eq!(m.get_counter().value(), 10.0);
        assert_eq!(m.get_label()[0].get_name(), "pool");
        assert_eq!(m.get_label()[0].get_value(), "G1");

        // Find reset_pool_size gauge
        let reset_family = families
            .iter()
            .find(|f| f.get_name() == "kvbm_reset_pool_size")
            .expect("should have reset pool size family");
        assert_eq!(reset_family.get_field_type(), MetricType::GAUGE);
        assert_eq!(reset_family.get_metric()[0].get_gauge().value(), 42.0);
    }

    #[test]
    fn test_external_labels() {
        let agg = MetricsAggregator::new();
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        agg.register_source(metrics);

        agg.set_external_labels(vec![
            ("instance_id".to_string(), "node-1".to_string()),
            ("worker_id".to_string(), "w0".to_string()),
        ]);

        let families = agg.collect();
        let alloc_family = families
            .iter()
            .find(|f| f.get_name() == "kvbm_allocations_total")
            .unwrap();
        let labels = alloc_family.get_metric()[0].get_label();
        assert_eq!(labels.len(), 3); // pool + 2 external
        assert_eq!(labels[1].get_name(), "instance_id");
        assert_eq!(labels[1].get_value(), "node-1");
        assert_eq!(labels[2].get_name(), "worker_id");
        assert_eq!(labels[2].get_value(), "w0");
    }

    #[test]
    fn test_multiple_sources_merged() {
        let agg = MetricsAggregator::new();

        let m1 = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let m2 = Arc::new(BlockPoolMetrics::new("G2".to_string()));

        m1.inc_allocations(5);
        m2.inc_allocations(10);

        agg.register_source(m1);
        agg.register_source(m2);

        let families = agg.collect();

        // Families should be merged by name
        let alloc_family = families
            .iter()
            .find(|f| f.get_name() == "kvbm_allocations_total")
            .expect("should have allocations family");
        assert_eq!(alloc_family.get_metric().len(), 2);

        let values: Vec<f64> = alloc_family
            .get_metric()
            .iter()
            .map(|m| m.get_counter().value())
            .collect();
        assert!(values.contains(&5.0));
        assert!(values.contains(&10.0));
    }

    #[test]
    fn test_register_with_prometheus_registry() {
        let agg = MetricsAggregator::new();
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        metrics.inc_allocations(42);
        agg.register_source(metrics);

        let registry = prometheus::Registry::new();
        agg.register_with(&registry)
            .expect("should register successfully");

        let gathered = registry.gather();
        assert!(!gathered.is_empty());

        let alloc_family = gathered
            .iter()
            .find(|f| f.get_name() == "kvbm_allocations_total")
            .expect("should find allocations in gathered metrics");
        assert_eq!(alloc_family.get_metric()[0].get_counter().value(), 42.0);
    }

    #[test]
    fn test_descs_match_definitions() {
        let agg = MetricsAggregator::new();
        let descs = agg.desc();
        assert_eq!(descs.len(), COUNTER_DEFS.len() + GAUGE_DEFS.len());
    }
}
