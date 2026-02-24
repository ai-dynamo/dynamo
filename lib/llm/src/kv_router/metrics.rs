// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the KV router.
//!
//! This module centralizes all router-side Prometheus metric definitions:
//! - [`WorkerLoadMetrics`]: Per-worker active decode blocks and prefill tokens gauges.
//! - [`RoutingOverheadMetrics`]: Per-request routing phase latency histograms.
//! - [`RouterRequestMetrics`]: Per-request aggregate histograms (TTFT, ITL, tokens, KV hit rate).
//!
//! See also: `docs/pages/observability/metrics.md` (Router Metrics section).

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{
    frontend_service, labels, name_prefix, routing_overhead,
};
use prometheus::{IntGaugeVec, Opts};

use crate::http::service::metrics::generate_log_buckets;

// ---------------------------------------------------------------------------
// Worker load metrics (gauges)
// ---------------------------------------------------------------------------

/// Per-worker active load gauges, published by `ActiveSequencesMultiWorker`
/// and cleaned up by `KvWorkerMonitor` when workers disappear.
pub struct WorkerLoadMetrics {
    pub active_decode_blocks: IntGaugeVec,
    pub active_prefill_tokens: IntGaugeVec,
}

impl WorkerLoadMetrics {
    pub fn observe(
        &self,
        worker_id: u64,
        dp_rank: u32,
        worker_type: &str,
        active_blocks: usize,
        active_tokens: usize,
    ) {
        let worker_id_str = worker_id.to_string();
        let dp_rank_str = dp_rank.to_string();
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
        self.active_decode_blocks
            .with_label_values(labels)
            .set(active_blocks as i64);
        self.active_prefill_tokens
            .with_label_values(labels)
            .set(active_tokens as i64);
    }
}

pub static WORKER_LOAD_METRICS: LazyLock<WorkerLoadMetrics> = LazyLock::new(|| WorkerLoadMetrics {
    active_decode_blocks: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
            ),
            "Active KV cache decode blocks per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_active_decode_blocks gauge"),
    active_prefill_tokens: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
            ),
            "Active prefill tokens queued per worker",
        ),
        &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
    )
    .expect("Failed to create worker_active_prefill_tokens gauge"),
});

/// Register the worker load gauges with the given Prometheus registry.
pub fn register_worker_load_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*WORKER_LOAD_METRICS;
    registry.register(Box::new(m.active_decode_blocks.clone()))?;
    registry.register(Box::new(m.active_prefill_tokens.clone()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Routing overhead metrics (histograms)
// ---------------------------------------------------------------------------

/// Per-request routing phase latency histograms (milliseconds).
pub struct RoutingOverheadMetrics {
    pub block_hashing: prometheus::Histogram,
    pub indexer_find_matches: prometheus::Histogram,
    pub seq_hashing: prometheus::Histogram,
    pub scheduling: prometheus::Histogram,
    pub total: prometheus::Histogram,
}

pub static ROUTING_OVERHEAD_METRICS: LazyLock<RoutingOverheadMetrics> = LazyLock::new(|| {
    // Buckets from 0.0001ms (0.1μs) to ~10ms, exponential with factor 2
    let buckets = prometheus::exponential_buckets(0.0001, 2.0, 18)
        .expect("exponential buckets should not fail");
    let make = |suffix: &str, help: &str| {
        let name = format!("{}_{}", name_prefix::ROUTING_OVERHEAD, suffix);
        prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(name, help).buckets(buckets.clone()),
        )
        .expect("histogram creation should not fail")
    };
    RoutingOverheadMetrics {
        block_hashing: make(
            routing_overhead::BLOCK_HASHING_MS,
            "Time spent computing block hashes in milliseconds",
        ),
        indexer_find_matches: make(
            routing_overhead::INDEXER_FIND_MATCHES_MS,
            "Time spent in indexer find_matches in milliseconds",
        ),
        seq_hashing: make(
            routing_overhead::SEQ_HASHING_MS,
            "Time spent computing sequence hashes in milliseconds",
        ),
        scheduling: make(
            routing_overhead::SCHEDULING_MS,
            "Time spent in scheduler worker selection in milliseconds",
        ),
        total: make(
            routing_overhead::TOTAL_MS,
            "Total routing overhead per request in milliseconds",
        ),
    }
});

impl RoutingOverheadMetrics {
    /// Observe routing overhead timings in milliseconds.
    pub fn observe(
        &self,
        hash_elapsed: Duration,
        find_matches_elapsed: Duration,
        seq_hash_elapsed: Duration,
        total_elapsed: Duration,
    ) {
        self.block_hashing
            .observe(hash_elapsed.as_secs_f64() * 1000.0);
        self.indexer_find_matches.observe(
            find_matches_elapsed
                .saturating_sub(hash_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.seq_hashing.observe(
            seq_hash_elapsed
                .saturating_sub(find_matches_elapsed)
                .as_secs_f64()
                * 1000.0,
        );
        self.scheduling
            .observe(total_elapsed.saturating_sub(seq_hash_elapsed).as_secs_f64() * 1000.0);
        self.total.observe(total_elapsed.as_secs_f64() * 1000.0);
    }
}

/// Register the routing overhead histograms with the given Prometheus registry.
pub fn register_routing_overhead_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*ROUTING_OVERHEAD_METRICS;
    registry.register(Box::new(m.block_hashing.clone()))?;
    registry.register(Box::new(m.indexer_find_matches.clone()))?;
    registry.register(Box::new(m.seq_hashing.clone()))?;
    registry.register(Box::new(m.scheduling.clone()))?;
    registry.register(Box::new(m.total.clone()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Router request metrics (component-scoped aggregate histograms + counter)
// ---------------------------------------------------------------------------

/// Aggregate per-request metrics observed at the router level.
/// Component-scoped via `from_component()` to get automatic namespace/component labels.
/// Uses `router_` prefix to distinguish from `ResponseMetricCollector` metrics.
pub struct RouterRequestMetrics {
    pub requests_total: prometheus::IntCounter,
    pub time_to_first_token_seconds: prometheus::Histogram,
    pub inter_token_latency_seconds: prometheus::Histogram,
    pub input_sequence_tokens: prometheus::Histogram,
    pub output_sequence_tokens: prometheus::Histogram,
    pub kv_hit_rate: prometheus::Histogram,
}

static ROUTER_REQUEST_METRICS: OnceLock<Arc<RouterRequestMetrics>> = OnceLock::new();

impl RouterRequestMetrics {
    fn new(
        requests_total: prometheus::IntCounter,
        time_to_first_token_seconds: prometheus::Histogram,
        inter_token_latency_seconds: prometheus::Histogram,
        input_sequence_tokens: prometheus::Histogram,
        output_sequence_tokens: prometheus::Histogram,
        kv_hit_rate: prometheus::Histogram,
    ) -> Self {
        Self {
            requests_total,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            input_sequence_tokens,
            output_sequence_tokens,
            kv_hit_rate,
        }
    }

    // TODO: move all `router_*` metric name strings to `prometheus_names.rs` constants
    // for consistency with the other metric families (routing_overhead, frontend_service).

    /// Create from a Component, memoized in a static OnceLock.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_REQUEST_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let requests_total = metrics
                    .create_intcounter(
                        "router_requests_total",
                        "Total number of requests processed by the router",
                        &[],
                    )
                    .expect("failed to create router_requests_total");
                let time_to_first_token_seconds = metrics
                    .create_histogram(
                        "router_time_to_first_token_seconds",
                        "Time to first token observed at the router",
                        &[],
                        Some(generate_log_buckets(0.001, 480.0, 18)),
                    )
                    .expect("failed to create router_time_to_first_token_seconds");
                let inter_token_latency_seconds = metrics
                    .create_histogram(
                        "router_inter_token_latency_seconds",
                        "Average inter-token latency observed at the router",
                        &[],
                        Some(generate_log_buckets(0.001, 2.0, 13)),
                    )
                    .expect("failed to create router_inter_token_latency_seconds");
                let input_sequence_tokens = metrics
                    .create_histogram(
                        "router_input_sequence_tokens",
                        "Input sequence length in tokens observed at the router",
                        &[],
                        Some(generate_log_buckets(50.0, 128000.0, 12)),
                    )
                    .expect("failed to create router_input_sequence_tokens");
                let output_sequence_tokens = metrics
                    .create_histogram(
                        "router_output_sequence_tokens",
                        "Output sequence length in tokens observed at the router",
                        &[],
                        Some(generate_log_buckets(50.0, 32000.0, 10)),
                    )
                    .expect("failed to create router_output_sequence_tokens");
                let kv_hit_rate = metrics
                    .create_histogram(
                        "router_kv_hit_rate",
                        "Predicted KV cache hit rate at routing time (0.0-1.0)",
                        &[],
                        Some(prometheus::linear_buckets(0.0, 0.05, 21).unwrap()),
                    )
                    .expect("failed to create router_kv_hit_rate");
                Arc::new(Self::new(
                    requests_total,
                    time_to_first_token_seconds,
                    inter_token_latency_seconds,
                    input_sequence_tokens,
                    output_sequence_tokens,
                    kv_hit_rate,
                ))
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{Encoder, TextEncoder};

    fn gather_pef(registry: &prometheus::Registry) -> String {
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&registry.gather(), &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    #[test]
    fn test_worker_load_metrics_pef() {
        let registry = prometheus::Registry::new();
        let metrics = WorkerLoadMetrics {
            active_decode_blocks: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
                    ),
                    "Active KV cache decode blocks per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
            active_prefill_tokens: IntGaugeVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
                    ),
                    "Active prefill tokens queued per worker",
                ),
                &[labels::WORKER_ID, labels::DP_RANK, labels::WORKER_TYPE],
            )
            .unwrap(),
        };
        registry
            .register(Box::new(metrics.active_decode_blocks.clone()))
            .unwrap();
        registry
            .register(Box::new(metrics.active_prefill_tokens.clone()))
            .unwrap();

        metrics.observe(123, 0, "decode", 42, 100);

        let output = gather_pef(&registry);
        let expected = "\
# HELP dynamo_frontend_worker_active_decode_blocks Active KV cache decode blocks per worker
# TYPE dynamo_frontend_worker_active_decode_blocks gauge
dynamo_frontend_worker_active_decode_blocks{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 42
# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens queued per worker
# TYPE dynamo_frontend_worker_active_prefill_tokens gauge
dynamo_frontend_worker_active_prefill_tokens{dp_rank=\"0\",worker_id=\"123\",worker_type=\"decode\"} 100
";
        assert_eq!(
            output, expected,
            "\nActual PEF:\n{output}\nExpected PEF:\n{expected}"
        );
    }

    #[test]
    fn test_routing_overhead_metric_names_pef() {
        let registry = prometheus::Registry::new();
        let buckets = prometheus::exponential_buckets(0.0001, 2.0, 18).unwrap();
        let make = |suffix: &str, help: &str| {
            let name = format!("{}_{}", name_prefix::ROUTING_OVERHEAD, suffix);
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(name, help).buckets(buckets.clone()),
            )
            .unwrap()
        };

        let total = make(
            routing_overhead::TOTAL_MS,
            "Total routing overhead per request in milliseconds",
        );
        registry.register(Box::new(total.clone())).unwrap();
        total.observe(1.5);

        let output = gather_pef(&registry);
        assert!(
            output.contains("# HELP dynamo_routing_overhead_total_ms"),
            "PEF missing HELP for routing overhead metric"
        );
        assert!(
            output.contains("# TYPE dynamo_routing_overhead_total_ms histogram"),
            "PEF missing TYPE for routing overhead metric"
        );
        assert!(
            output.contains("dynamo_routing_overhead_total_ms_count 1"),
            "PEF missing observation count"
        );
    }

    #[test]
    fn test_routing_overhead_saturating_sub() {
        let buckets = prometheus::exponential_buckets(0.0001, 2.0, 18).unwrap();
        let make = |name: &str| {
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(name, "test").buckets(buckets.clone()),
            )
            .unwrap()
        };
        let metrics = RoutingOverheadMetrics {
            block_hashing: make("test_block_hashing_ms"),
            indexer_find_matches: make("test_find_matches_ms"),
            seq_hashing: make("test_seq_hashing_ms"),
            scheduling: make("test_scheduling_ms"),
            total: make("test_total_ms"),
        };

        // Out-of-order durations: each phase < previous (would panic without saturating_sub)
        metrics.observe(
            Duration::from_millis(10),
            Duration::from_millis(5),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        // Reaching here without panic confirms saturating_sub works
    }
}
