// SPDX-FileCopyrightText: Copyright (c) 2026-2027 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend pipeline stage and finer-grained perf metrics.
//! Used by both runtime (route, transport_roundtrip) and llm (preprocess, postprocess, tokenize, template, detokenize).

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Counter, Histogram, HistogramOpts, HistogramVec, Opts, Registry};

use super::prometheus_names::{frontend_perf, name_prefix};
use crate::MetricsRegistry;

fn frontend_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::FRONTEND, suffix)
}

/// Per-stage latency: preprocess, route, transport_roundtrip, postprocess.
pub static STAGE_DURATION_SECONDS: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            frontend_metric_name(frontend_perf::STAGE_DURATION_SECONDS),
            "Pipeline stage duration (seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0,
        ]),
        &["stage"],
    )
    .expect("stage_duration_seconds histogram vec")
});

/// Tokenization time in preprocessor (gather_tokens).
pub static TOKENIZE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            frontend_metric_name(frontend_perf::TOKENIZE_SECONDS),
            "Tokenization time in preprocessor (seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .expect("tokenize_seconds histogram")
});

/// Template application time in preprocessor (apply_template).
pub static TEMPLATE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            frontend_metric_name(frontend_perf::TEMPLATE_SECONDS),
            "Template application time in preprocessor (seconds)",
        )
        .buckets(vec![
            0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
        ]),
    )
    .expect("template_seconds histogram")
});

/// Cumulative detokenization time across all tokens (microseconds).
/// Use `rate(total) / rate(count)` in Prometheus to derive per-token average.
pub static DETOKENIZE_TOTAL_US: Lazy<Counter> = Lazy::new(|| {
    Counter::with_opts(Opts::new(
        frontend_metric_name(frontend_perf::DETOKENIZE_TOTAL_US),
        "Cumulative detokenization time (microseconds)",
    ))
    .expect("detokenize_total_us counter")
});

/// Total number of tokens detokenized.
pub static DETOKENIZE_TOKEN_COUNT: Lazy<Counter> = Lazy::new(|| {
    Counter::with_opts(Opts::new(
        frontend_metric_name(frontend_perf::DETOKENIZE_TOKEN_COUNT),
        "Total tokens detokenized",
    ))
    .expect("detokenize_token_count counter")
});

/// Guards idempotency for the `MetricsRegistry` registration path.
static REGISTERED: OnceCell<()> = OnceCell::new();

/// Guards idempotency for the raw `prometheus::Registry` registration path.
/// Kept separate from `REGISTERED` so that calling `ensure_frontend_perf_metrics_registered`
/// first does not silently prevent the metrics from being registered in the prometheus registry.
static PROMETHEUS_REGISTERED: OnceCell<()> = OnceCell::new();

/// Register frontend perf metrics with the given registry. Idempotent.
pub fn ensure_frontend_perf_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry
            .add_metric(Box::new(STAGE_DURATION_SECONDS.clone()))
            .ok();
        registry.add_metric(Box::new(TOKENIZE_SECONDS.clone())).ok();
        registry.add_metric(Box::new(TEMPLATE_SECONDS.clone())).ok();
        registry
            .add_metric(Box::new(DETOKENIZE_TOTAL_US.clone()))
            .ok();
        registry
            .add_metric(Box::new(DETOKENIZE_TOKEN_COUNT.clone()))
            .ok();
    });
}

/// Register frontend perf metrics with a raw Prometheus registry (e.g. for LLM HTTP service /metrics).
/// Idempotent. Call this when the service exposes /metrics from its own registry.
pub fn ensure_frontend_perf_metrics_registered_prometheus(
    registry: &Registry,
) -> Result<(), prometheus::Error> {
    if PROMETHEUS_REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(STAGE_DURATION_SECONDS.clone()))?;
    registry.register(Box::new(TOKENIZE_SECONDS.clone()))?;
    registry.register(Box::new(TEMPLATE_SECONDS.clone()))?;
    registry.register(Box::new(DETOKENIZE_TOTAL_US.clone()))?;
    registry.register(Box::new(DETOKENIZE_TOKEN_COUNT.clone()))?;
    let _ = PROMETHEUS_REGISTERED.set(());
    Ok(())
}
