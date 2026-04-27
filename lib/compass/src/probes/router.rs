// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use prometheus::{Histogram, HistogramOpts, Registry};

static ROUTER_PROBES: OnceLock<RouterPhaseMetrics> = OnceLock::new();

pub struct RouterPhaseMetrics {
    pub cost_compute_ms: Histogram,
    pub kv_overlap_score_ms: Histogram,
    pub worker_select_ms: Histogram,
    pub dispatch_ms: Histogram,
}

fn probe_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.001, 2.0, 18).unwrap()
}

impl RouterPhaseMetrics {
    pub fn register(registry: &Registry) -> Result<(), prometheus::Error> {
        let metrics = ROUTER_PROBES.get_or_init(|| {
            let buckets = probe_buckets();
            Self {
                cost_compute_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_router_cost_compute_ms",
                        "Router cost computation time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("router_cost_compute_ms"),
                kv_overlap_score_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_router_kv_overlap_score_ms",
                        "Router KV overlap scoring time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("router_kv_overlap_score_ms"),
                worker_select_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_router_worker_select_ms",
                        "Router worker selection time in milliseconds",
                    )
                    .buckets(buckets.clone()),
                )
                .expect("router_worker_select_ms"),
                dispatch_ms: Histogram::with_opts(
                    HistogramOpts::new(
                        "dynamo_compass_router_dispatch_ms",
                        "Router dispatch time in milliseconds",
                    )
                    .buckets(buckets),
                )
                .expect("router_dispatch_ms"),
            }
        });
        registry.register(Box::new(metrics.cost_compute_ms.clone()))?;
        registry.register(Box::new(metrics.kv_overlap_score_ms.clone()))?;
        registry.register(Box::new(metrics.worker_select_ms.clone()))?;
        registry.register(Box::new(metrics.dispatch_ms.clone()))?;
        Ok(())
    }

    pub fn get() -> Option<&'static Self> {
        ROUTER_PROBES.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_probe_registration() {
        let registry = Registry::new();
        RouterPhaseMetrics::register(&registry).unwrap();
        let probes = RouterPhaseMetrics::get().unwrap();
        probes.cost_compute_ms.observe(0.15);
        probes.kv_overlap_score_ms.observe(0.42);
        probes.worker_select_ms.observe(0.18);
        probes.dispatch_ms.observe(0.09);
    }
}
