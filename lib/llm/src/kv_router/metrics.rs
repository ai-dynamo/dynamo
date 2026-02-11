// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the KV router.
//!
//! This module centralizes all router-side Prometheus metric definitions:
//! - [`WorkerLoadMetrics`]: Per-worker active decode blocks and prefill tokens gauges.
//! - [`RoutingOverheadMetrics`]: Per-request routing phase latency histograms.

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::frontend_service;
use prometheus::{IntGaugeVec, Opts};

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
                "dynamo_frontend_{}",
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
            ),
            "Active KV cache decode blocks per worker",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_active_decode_blocks gauge"),
    active_prefill_tokens: IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
            ),
            "Active prefill tokens queued per worker",
        ),
        &["worker_id", "dp_rank", "worker_type"],
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
    let make = |name: &str, help: &str| {
        prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(name, help).buckets(buckets.clone()),
        )
        .expect("histogram creation should not fail")
    };
    RoutingOverheadMetrics {
        block_hashing: make(
            "dynamo_routing_overhead_block_hashing_ms",
            "Time spent computing block hashes in milliseconds",
        ),
        indexer_find_matches: make(
            "dynamo_routing_overhead_indexer_find_matches_ms",
            "Time spent in indexer find_matches in milliseconds",
        ),
        seq_hashing: make(
            "dynamo_routing_overhead_seq_hashing_ms",
            "Time spent computing sequence hashes in milliseconds",
        ),
        scheduling: make(
            "dynamo_routing_overhead_scheduling_ms",
            "Time spent in scheduler worker selection in milliseconds",
        ),
        total: make(
            "dynamo_routing_overhead_total_ms",
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
        self.indexer_find_matches
            .observe((find_matches_elapsed - hash_elapsed).as_secs_f64() * 1000.0);
        self.seq_hashing
            .observe((seq_hash_elapsed - find_matches_elapsed).as_secs_f64() * 1000.0);
        self.scheduling
            .observe((total_elapsed - seq_hash_elapsed).as_secs_f64() * 1000.0);
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
}

static ROUTER_REQUEST_METRICS: OnceLock<Arc<RouterRequestMetrics>> = OnceLock::new();

use crate::http::service::metrics::generate_log_buckets;

impl RouterRequestMetrics {
    fn new(
        requests_total: prometheus::IntCounter,
        time_to_first_token_seconds: prometheus::Histogram,
        inter_token_latency_seconds: prometheus::Histogram,
        input_sequence_tokens: prometheus::Histogram,
        output_sequence_tokens: prometheus::Histogram,
    ) -> Self {
        Self {
            requests_total,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            input_sequence_tokens,
            output_sequence_tokens,
        }
    }

    /// Create from a Component, memoized in a static OnceLock.
    pub fn from_component(component: &Component) -> Arc<Self> {
        ROUTER_REQUEST_METRICS
            .get_or_init(|| {
                let metrics = component.metrics();
                let result = (|| -> anyhow::Result<Arc<Self>> {
                    let requests_total = metrics.create_intcounter(
                        "router_requests_total",
                        "Total number of requests processed by the router",
                        &[],
                    )?;
                    let time_to_first_token_seconds = metrics.create_histogram(
                        "router_time_to_first_token_seconds",
                        "Time to first token observed at the router",
                        &[],
                        Some(generate_log_buckets(0.001, 480.0, 18)),
                    )?;
                    let inter_token_latency_seconds = metrics.create_histogram(
                        "router_inter_token_latency_seconds",
                        "Average inter-token latency observed at the router",
                        &[],
                        Some(generate_log_buckets(0.001, 2.0, 13)),
                    )?;
                    let input_sequence_tokens = metrics.create_histogram(
                        "router_input_sequence_tokens",
                        "Input sequence length in tokens observed at the router",
                        &[],
                        Some(generate_log_buckets(50.0, 128000.0, 12)),
                    )?;
                    let output_sequence_tokens = metrics.create_histogram(
                        "router_output_sequence_tokens",
                        "Output sequence length in tokens observed at the router",
                        &[],
                        Some(generate_log_buckets(50.0, 32000.0, 10)),
                    )?;
                    Ok(Arc::new(Self::new(
                        requests_total,
                        time_to_first_token_seconds,
                        inter_token_latency_seconds,
                        input_sequence_tokens,
                        output_sequence_tokens,
                    )))
                })();

                match result {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create router request metrics from component: {}. Using unregistered metrics as fallback.",
                            e
                        );
                        Arc::new(Self::new_unregistered())
                    }
                }
            })
            .clone()
    }

    /// Fallback for tests or when a MetricsRegistry is not available.
    pub fn new_unregistered() -> Self {
        Self::new(
            prometheus::IntCounter::new(
                "router_requests_total",
                "Total number of requests processed by the router",
            )
            .unwrap(),
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    "router_time_to_first_token_seconds",
                    "Time to first token observed at the router",
                )
                .buckets(generate_log_buckets(0.001, 480.0, 18)),
            )
            .unwrap(),
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    "router_inter_token_latency_seconds",
                    "Average inter-token latency observed at the router",
                )
                .buckets(generate_log_buckets(0.001, 2.0, 13)),
            )
            .unwrap(),
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    "router_input_sequence_tokens",
                    "Input sequence length in tokens observed at the router",
                )
                .buckets(generate_log_buckets(50.0, 128000.0, 12)),
            )
            .unwrap(),
            prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    "router_output_sequence_tokens",
                    "Output sequence length in tokens observed at the router",
                )
                .buckets(generate_log_buckets(50.0, 32000.0, 10)),
            )
            .unwrap(),
        )
    }
}
