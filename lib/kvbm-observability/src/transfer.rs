// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for transfer, cache, and object-store activity.

use std::time::Duration;

use kvbm_common::KvbmTransferRoute;
use prometheus::{
    Gauge, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts, Registry,
};

/// Exact v1-compatible metrics that existing dashboards and tests consume.
#[derive(Clone)]
pub struct CompatMetrics {
    pub offload_blocks_d2h: IntCounter,
    pub offload_blocks_h2d: IntCounter,
    pub offload_blocks_d2d: IntCounter,
    pub offload_blocks_d2o: IntCounter,
    pub onboard_blocks_h2d: IntCounter,
    pub onboard_blocks_d2d: IntCounter,
    pub onboard_blocks_o2d: IntCounter,
    pub matched_tokens: IntCounter,
    pub host_cache_hit_rate: Gauge,
    pub disk_cache_hit_rate: Gauge,
    pub object_cache_hit_rate: Gauge,
    pub object_read_failures: IntCounter,
    pub object_write_failures: IntCounter,
}

impl CompatMetrics {
    pub fn new() -> Self {
        Self {
            offload_blocks_d2h: IntCounter::with_opts(Opts::new(
                "kvbm_offload_blocks_d2h",
                "The number of offload blocks from device to host",
            ))
            .expect("valid metric"),
            offload_blocks_h2d: IntCounter::with_opts(Opts::new(
                "kvbm_offload_blocks_h2d",
                "The number of offload blocks from host to disk",
            ))
            .expect("valid metric"),
            offload_blocks_d2d: IntCounter::with_opts(Opts::new(
                "kvbm_offload_blocks_d2d",
                "The number of offload blocks from device to disk (bypassing host memory)",
            ))
            .expect("valid metric"),
            offload_blocks_d2o: IntCounter::with_opts(Opts::new(
                "kvbm_offload_blocks_d2o",
                "The number of offload blocks from device to object storage",
            ))
            .expect("valid metric"),
            onboard_blocks_h2d: IntCounter::with_opts(Opts::new(
                "kvbm_onboard_blocks_h2d",
                "The number of onboard blocks from host to device",
            ))
            .expect("valid metric"),
            onboard_blocks_d2d: IntCounter::with_opts(Opts::new(
                "kvbm_onboard_blocks_d2d",
                "The number of onboard blocks from disk to device",
            ))
            .expect("valid metric"),
            onboard_blocks_o2d: IntCounter::with_opts(Opts::new(
                "kvbm_onboard_blocks_o2d",
                "The number of onboard blocks from object storage to device",
            ))
            .expect("valid metric"),
            matched_tokens: IntCounter::with_opts(Opts::new(
                "kvbm_matched_tokens",
                "The number of matched tokens",
            ))
            .expect("valid metric"),
            host_cache_hit_rate: Gauge::with_opts(Opts::new(
                "kvbm_host_cache_hit_rate",
                "Host cache hit rate (0.0-1.0) from the sliding window",
            ))
            .expect("valid metric"),
            disk_cache_hit_rate: Gauge::with_opts(Opts::new(
                "kvbm_disk_cache_hit_rate",
                "Disk cache hit rate (0.0-1.0) from the sliding window",
            ))
            .expect("valid metric"),
            object_cache_hit_rate: Gauge::with_opts(Opts::new(
                "kvbm_object_cache_hit_rate",
                "Object storage cache hit rate (0.0-1.0) from the sliding window",
            ))
            .expect("valid metric"),
            object_read_failures: IntCounter::with_opts(Opts::new(
                "kvbm_object_read_failures",
                "The number of failed object storage read operations (blocks)",
            ))
            .expect("valid metric"),
            object_write_failures: IntCounter::with_opts(Opts::new(
                "kvbm_object_write_failures",
                "The number of failed object storage write operations (blocks)",
            ))
            .expect("valid metric"),
        }
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.offload_blocks_d2h.clone()))?;
        registry.register(Box::new(self.offload_blocks_h2d.clone()))?;
        registry.register(Box::new(self.offload_blocks_d2d.clone()))?;
        registry.register(Box::new(self.offload_blocks_d2o.clone()))?;
        registry.register(Box::new(self.onboard_blocks_h2d.clone()))?;
        registry.register(Box::new(self.onboard_blocks_d2d.clone()))?;
        registry.register(Box::new(self.onboard_blocks_o2d.clone()))?;
        registry.register(Box::new(self.matched_tokens.clone()))?;
        registry.register(Box::new(self.host_cache_hit_rate.clone()))?;
        registry.register(Box::new(self.disk_cache_hit_rate.clone()))?;
        registry.register(Box::new(self.object_cache_hit_rate.clone()))?;
        registry.register(Box::new(self.object_read_failures.clone()))?;
        registry.register(Box::new(self.object_write_failures.clone()))?;
        Ok(())
    }

    pub fn record_transfer_success(&self, route: KvbmTransferRoute, blocks: u64) {
        match route {
            KvbmTransferRoute::OffloadD2H => self.offload_blocks_d2h.inc_by(blocks),
            KvbmTransferRoute::OffloadH2D => self.offload_blocks_h2d.inc_by(blocks),
            KvbmTransferRoute::OffloadD2D => self.offload_blocks_d2d.inc_by(blocks),
            KvbmTransferRoute::OffloadD2O => self.offload_blocks_d2o.inc_by(blocks),
            KvbmTransferRoute::OnboardH2D => self.onboard_blocks_h2d.inc_by(blocks),
            KvbmTransferRoute::OnboardD2D => self.onboard_blocks_d2d.inc_by(blocks),
            KvbmTransferRoute::OnboardO2D => self.onboard_blocks_o2d.inc_by(blocks),
        }
    }

    pub fn set_cache_hit_rates(&self, host: f64, disk: f64, object: f64) {
        self.host_cache_hit_rate.set(host);
        self.disk_cache_hit_rate.set(disk);
        self.object_cache_hit_rate.set(object);
    }
}

impl Default for CompatMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Additive metrics that complement the strict-compat counters.
#[derive(Clone)]
pub struct TransferMetrics {
    pub transfer_duration_seconds: HistogramVec,
    pub transfer_failures_total: IntCounterVec,
    pub transfer_inflight: IntGaugeVec,
    pub cache_blocks_hit_total: IntCounterVec,
}

impl TransferMetrics {
    pub fn new() -> Self {
        Self {
            transfer_duration_seconds: HistogramVec::new(
                HistogramOpts::new(
                    "kvbm_transfer_duration_seconds",
                    "End-to-end transfer duration by operation, route, and outcome",
                ),
                &["operation", "route", "outcome"],
            )
            .expect("valid metric"),
            transfer_failures_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_transfer_failures_total",
                    "Transfer failures by operation, route, and reason",
                ),
                &["operation", "route", "reason"],
            )
            .expect("valid metric"),
            transfer_inflight: IntGaugeVec::new(
                Opts::new(
                    "kvbm_transfer_inflight",
                    "Logical transfers currently in flight",
                ),
                &["operation"],
            )
            .expect("valid metric"),
            cache_blocks_hit_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_cache_blocks_hit_total",
                    "Blocks served from each cache tier",
                ),
                &["tier"],
            )
            .expect("valid metric"),
        }
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.transfer_duration_seconds.clone()))?;
        registry.register(Box::new(self.transfer_failures_total.clone()))?;
        registry.register(Box::new(self.transfer_inflight.clone()))?;
        registry.register(Box::new(self.cache_blocks_hit_total.clone()))?;
        Ok(())
    }

    pub fn begin_transfer(&self, route: KvbmTransferRoute) {
        self.transfer_inflight
            .with_label_values(&[route.operation_label()])
            .inc();
    }

    pub fn finish_transfer(
        &self,
        route: KvbmTransferRoute,
        duration: Duration,
        outcome: &'static str,
    ) {
        self.transfer_inflight
            .with_label_values(&[route.operation_label()])
            .dec();
        self.transfer_duration_seconds
            .with_label_values(&[route.operation_label(), route.as_label(), outcome])
            .observe(duration.as_secs_f64());
    }

    pub fn record_failure(&self, route: KvbmTransferRoute, reason: &'static str, count: u64) {
        self.transfer_failures_total
            .with_label_values(&[route.operation_label(), route.as_label(), reason])
            .inc_by(count);
    }

    pub fn record_cache_hits(&self, tier: &'static str, blocks: u64) {
        self.cache_blocks_hit_total
            .with_label_values(&[tier])
            .inc_by(blocks);
    }
}

impl Default for TransferMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compat_metric_names() {
        let compat = CompatMetrics::new();
        let registry = Registry::new();
        compat.register(&registry).unwrap();

        let gathered = registry.gather();
        let names: Vec<_> = gathered.iter().map(|mf| mf.name()).collect();

        assert!(names.contains(&"kvbm_offload_blocks_d2h"));
        assert!(names.contains(&"kvbm_onboard_blocks_d2d"));
        assert!(names.contains(&"kvbm_matched_tokens"));
        assert!(names.contains(&"kvbm_object_write_failures"));
        assert!(!names.contains(&"kvbm_offload_blocks_d2h_total"));
        assert!(!names.contains(&"kvbm_matched_tokens_total"));
    }
}
