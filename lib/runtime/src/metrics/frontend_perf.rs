// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend pipeline stage and finer-grained perf metrics.
//! Used by both runtime (route, transport_roundtrip) and llm (preprocess, postprocess, tokenize, template, detokenize).

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Histogram, HistogramOpts, HistogramVec, Registry};

use super::prometheus_names::{name_prefix, frontend_perf};
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
        .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0]),
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
        .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
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
        .buckets(vec![0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
    )
    .expect("template_seconds histogram")
});

/// Per-token detokenization cost (microseconds).
pub static DETOKENIZE_PER_TOKEN_US: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            frontend_metric_name(frontend_perf::DETOKENIZE_PER_TOKEN_US),
            "Detokenization cost per token (microseconds)",
        )
        .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
    )
    .expect("detokenize_per_token_us histogram")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

/// Register frontend perf metrics with the given registry. Idempotent.
pub fn ensure_frontend_perf_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry
            .add_metric(Box::new(STAGE_DURATION_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKENIZE_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TEMPLATE_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(DETOKENIZE_PER_TOKEN_US.clone()))
            .ok();
    });
}

/// Register frontend perf metrics with a raw Prometheus registry (e.g. for LLM HTTP service /metrics).
/// Idempotent. Call this when the service exposes /metrics from its own registry.
pub fn ensure_frontend_perf_metrics_registered_prometheus(
    registry: &Registry,
) -> Result<(), prometheus::Error> {
    if REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(STAGE_DURATION_SECONDS.clone()))?;
    registry.register(Box::new(TOKENIZE_SECONDS.clone()))?;
    registry.register(Box::new(TEMPLATE_SECONDS.clone()))?;
    registry.register(Box::new(DETOKENIZE_PER_TOKEN_US.clone()))?;
    let _ = REGISTERED.set(());
    Ok(())
}
