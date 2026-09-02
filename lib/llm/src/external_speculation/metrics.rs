// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};
use prometheus::{Histogram, IntCounterVec, IntGauge};

const QUARANTINED_DRAFT_LEASES_METRIC: &str =
    "router_external_speculation_quarantined_draft_leases";
const QUARANTINED_DRAFT_LEASES_HELP: &str =
    "External-speculation draft leases retained through a cleanup bound";

pub(crate) struct ExternalSpeculationMetrics {
    selections_total: IntCounterVec,
    lifecycle_total: IntCounterVec,
    selection_duration_seconds: Histogram,
    dispatch_duration_seconds: Histogram,
    quarantined_draft_leases: IntGauge,
}

static EXTERNAL_SPECULATION_METRICS: OnceLock<Arc<ExternalSpeculationMetrics>> = OnceLock::new();

impl ExternalSpeculationMetrics {
    pub(crate) fn from_component(component: &Component) -> Arc<Self> {
        EXTERNAL_SPECULATION_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                Arc::new(Self {
                    selections_total: metrics
                        .create_intcountervec(
                            "router_external_speculation_selections_total",
                            "External-speculation rank selections by pool and cache-match basis",
                            &["pool", "basis"],
                            &[],
                        )
                        .expect("failed to create external-speculation selection counter"),
                    lifecycle_total: metrics
                        .create_intcountervec(
                            "router_external_speculation_lifecycle_total",
                            "External-speculation pair lifecycle transitions",
                            &["outcome"],
                            &[],
                        )
                        .expect("failed to create external-speculation lifecycle counter"),
                    selection_duration_seconds: metrics
                        .create_histogram(
                            "router_external_speculation_selection_duration_seconds",
                            "Time to derive keys and reserve an external-speculation pair",
                            &[],
                            Some(prometheus::exponential_buckets(0.000_01, 3.0, 14).unwrap()),
                        )
                        .expect("failed to create external-speculation selection histogram"),
                    dispatch_duration_seconds: metrics
                        .create_histogram(
                            "router_external_speculation_dispatch_duration_seconds",
                            "Time to exact-dispatch an external-speculation target",
                            &[],
                            Some(prometheus::exponential_buckets(0.000_1, 3.0, 14).unwrap()),
                        )
                        .expect("failed to create external-speculation dispatch histogram"),
                    quarantined_draft_leases: metrics
                        .create_intgauge(
                            QUARANTINED_DRAFT_LEASES_METRIC,
                            QUARANTINED_DRAFT_LEASES_HELP,
                            &[],
                        )
                        .expect("failed to create external-speculation quarantine gauge"),
                })
            })
            .clone()
    }

    pub(crate) fn observe_selection(&self, pool: &'static str, cache_hit: bool) {
        self.selections_total
            .with_label_values(&[pool, if cache_hit { "hit" } else { "miss" }])
            .inc();
    }

    pub(crate) fn observe_lifecycle(&self, outcome: &'static str) {
        self.lifecycle_total.with_label_values(&[outcome]).inc();
    }

    pub(crate) fn observe_selection_duration(&self, seconds: f64) {
        self.selection_duration_seconds.observe(seconds);
    }

    pub(crate) fn observe_dispatch_duration(&self, seconds: f64) {
        self.dispatch_duration_seconds.observe(seconds);
    }

    pub(crate) fn begin_quarantine(&self) {
        self.quarantined_draft_leases.inc();
        self.observe_lifecycle("quarantine_started");
    }

    pub(crate) fn end_quarantine(&self) {
        self.quarantined_draft_leases.dec();
        self.observe_lifecycle("quarantine_released");
    }
}

#[cfg(test)]
mod tests {
    use prometheus::core::Collector;

    use super::*;

    #[test]
    fn quarantine_gauge_preserves_legacy_prometheus_series() {
        let gauge = IntGauge::new(
            QUARANTINED_DRAFT_LEASES_METRIC,
            QUARANTINED_DRAFT_LEASES_HELP,
        )
        .unwrap();
        gauge.inc();

        let families = gauge.collect();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].name(), QUARANTINED_DRAFT_LEASES_METRIC);
        assert_eq!(families[0].get_metric()[0].get_gauge().value(), 1.0);
    }
}
