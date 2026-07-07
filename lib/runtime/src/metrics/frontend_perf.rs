// SPDX-FileCopyrightText: Copyright (c) 2026-2027 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend pipeline stage and finer-grained perf metrics.
//! Used by both runtime (route, transport_roundtrip) and llm (preprocess, postprocess, tokenize, template, detokenize).

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{
    Counter, Histogram, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
};

use super::prometheus_names::{frontend_perf, labels, name_prefix};
use crate::MetricsRegistry;

pub use super::prometheus_names::frontend_perf::{STAGE_DISPATCH, STAGE_PREPROCESS, STAGE_ROUTE};

fn frontend_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::FRONTEND, suffix)
}

/// Per-stage inflight request count: preprocess, route, dispatch.
/// Labels: stage (pipeline stage), phase (prefill/decode/aggregated or empty for preprocess).
pub static STAGE_REQUESTS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            frontend_metric_name(frontend_perf::STAGE_REQUESTS),
            "Number of requests currently in the given pipeline stage",
        ),
        &["stage", "phase"],
    )
    .expect("failed to create dynamo_frontend_stage_requests gauge")
});

/// RAII guard that increments a per-stage gauge on creation and decrements on drop.
///
/// Used to track how many requests are in each frontend pipeline stage at any given time.
/// Create with [`StageGuard::new`] at stage entry; the gauge decrements automatically when
/// the guard is dropped (end of scope, explicit drop, or stream completion).
pub struct StageGuard {
    gauge: prometheus::IntGauge,
}

impl StageGuard {
    /// Increment the stage gauge and return a guard that decrements on drop.
    ///
    /// * `stage` — pipeline stage name; use `frontend_perf::STAGE_{PREPROCESS,ROUTE,DISPATCH}`
    ///   constants from [`crate::metrics::prometheus_names`].
    /// * `phase` — request phase; use [`RequestPhase::to_string`] output
    ///   (`"prefill"|"decode"|"aggregated"`), or `""` for stages without a phase.
    pub fn new(stage: &str, phase: &str) -> Self {
        let gauge = STAGE_REQUESTS.with_label_values(&[stage, phase]);
        gauge.inc();
        Self { gauge }
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
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

/// Cumulative L1 tokenizer cache hits. The cache is enabled unless `DYN_TOKENIZER_CACHE=0`.
pub static TOKENIZER_CACHE_HITS_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::with_opts(Opts::new(
        frontend_metric_name(frontend_perf::TOKENIZER_CACHE_HITS_TOTAL),
        "Cumulative L1 tokenizer prefix-cache hits",
    ))
    .expect("tokenizer_cache_hits_total counter")
});

/// Cumulative L1 tokenizer cache misses. The cache is enabled unless `DYN_TOKENIZER_CACHE=0`.
pub static TOKENIZER_CACHE_MISSES_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::with_opts(Opts::new(
        frontend_metric_name(frontend_perf::TOKENIZER_CACHE_MISSES_TOTAL),
        "Cumulative L1 tokenizer prefix-cache misses",
    ))
    .expect("tokenizer_cache_misses_total counter")
});

/// Tokens returned from the L1 tokenizer prefix cache, labeled by served model name.
pub static TOKENIZER_CACHE_CACHED_TOKENS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            frontend_metric_name(frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL),
            "Total tokens returned from the L1 tokenizer prefix cache",
        ),
        &[labels::MODEL],
    )
    .expect("tokenizer_cache_cached_tokens_total counter vec")
});

/// Tokens freshly encoded after an L1 tokenizer prefix-cache lookup, labeled by served model name.
pub static TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            frontend_metric_name(frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL),
            "Total tokens freshly encoded after an L1 tokenizer prefix-cache lookup",
        ),
        &[labels::MODEL],
    )
    .expect("tokenizer_cache_uncached_tokens_total counter vec")
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
        registry.add_metric(Box::new(STAGE_REQUESTS.clone())).ok();
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
        registry
            .add_metric(Box::new(TOKENIZER_CACHE_HITS_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKENIZER_CACHE_MISSES_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.clone()))
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
    registry.register(Box::new(STAGE_REQUESTS.clone()))?;
    registry.register(Box::new(STAGE_DURATION_SECONDS.clone()))?;
    registry.register(Box::new(TOKENIZE_SECONDS.clone()))?;
    registry.register(Box::new(TEMPLATE_SECONDS.clone()))?;
    registry.register(Box::new(DETOKENIZE_TOTAL_US.clone()))?;
    registry.register(Box::new(DETOKENIZE_TOKEN_COUNT.clone()))?;
    registry.register(Box::new(TOKENIZER_CACHE_HITS_TOTAL.clone()))?;
    registry.register(Box::new(TOKENIZER_CACHE_MISSES_TOTAL.clone()))?;
    registry.register(Box::new(TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.clone()))?;
    registry.register(Box::new(TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.clone()))?;
    let _ = PROMETHEUS_REGISTERED.set(());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tokenizer_cache_token_metrics_registered(
        families: &[prometheus::proto::MetricFamily],
        model: &str,
    ) {
        for name in [
            "dynamo_frontend_tokenizer_cache_cached_tokens_total",
            "dynamo_frontend_tokenizer_cache_uncached_tokens_total",
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .unwrap_or_else(|| panic!("missing metric family {name}"));
            assert!(family.get_metric().iter().any(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == labels::MODEL && label.value() == model)
            }));
        }
    }

    #[test]
    fn test_tokenizer_cache_token_metrics_registered_with_model_label() {
        let model = "frontend-perf-registration-test-model";
        let _ = TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.with_label_values(&[model]);
        let _ = TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.with_label_values(&[model]);

        let metrics_registry = MetricsRegistry::new();
        ensure_frontend_perf_metrics_registered(&metrics_registry);
        assert_tokenizer_cache_token_metrics_registered(
            &metrics_registry.get_prometheus_registry().gather(),
            model,
        );

        let prometheus_registry = Registry::new();
        ensure_frontend_perf_metrics_registered_prometheus(&prometheus_registry).unwrap();
        assert_tokenizer_cache_token_metrics_registered(&prometheus_registry.gather(), model);
    }

    #[test]
    fn test_stage_guard_inc_dec() {
        let gauge = STAGE_REQUESTS.with_label_values(&["test_stage", "test_phase"]);
        assert_eq!(gauge.get(), 0);

        {
            let _guard = StageGuard::new("test_stage", "test_phase");
            assert_eq!(gauge.get(), 1);

            {
                let _guard2 = StageGuard::new("test_stage", "test_phase");
                assert_eq!(gauge.get(), 2);
            }
            // guard2 dropped
            assert_eq!(gauge.get(), 1);
        }
        // guard dropped
        assert_eq!(gauge.get(), 0);
    }

    #[test]
    fn test_stage_guard_different_labels() {
        let preprocess = STAGE_REQUESTS.with_label_values(&["preprocess_t", ""]);
        let route_prefill = STAGE_REQUESTS.with_label_values(&["route_t", "prefill"]);
        let route_decode = STAGE_REQUESTS.with_label_values(&["route_t", "decode"]);

        let _g1 = StageGuard::new("preprocess_t", "");
        let _g2 = StageGuard::new("route_t", "prefill");
        let _g3 = StageGuard::new("route_t", "decode");

        assert_eq!(preprocess.get(), 1);
        assert_eq!(route_prefill.get(), 1);
        assert_eq!(route_decode.get(), 1);

        drop(_g2);
        assert_eq!(preprocess.get(), 1);
        assert_eq!(route_prefill.get(), 0);
        assert_eq!(route_decode.get(), 1);
    }
}
