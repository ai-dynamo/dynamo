// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the two metric collection paths.
//!
//! `/metrics` is scraped every 5s on every pod, so a regression in
//! `prometheus_expfmt_combined` is paid fleet-wide whether or not OTLP export
//! is enabled. Run this against `main` and against a branch to compare.
//!
//! `metric_families_combined` is the added path; it only runs when OTLP export
//! is on, at `OTEL_METRIC_EXPORT_INTERVAL` (default 60s).

use criterion::{Criterion, criterion_group, criterion_main};
use dynamo_runtime::metrics::MetricsRegistry;
use std::hint::black_box;
use std::sync::Arc;

/// Roughly the shape of a worker's exposition: a few hundred native series
/// plus an engine payload with colon-named families and histograms.
fn registry(native_families: usize, engine_families: usize) -> MetricsRegistry {
    let registry = MetricsRegistry::new();

    for i in 0..native_families {
        let counter = prometheus::IntCounterVec::new(
            prometheus::Opts::new(format!("dynamo_bench_metric_{i}"), "Benchmark counter"),
            &["model", "endpoint"],
        )
        .expect("counter");
        for model in ["a", "b", "c"] {
            counter.with_label_values(&[model, "generate"]).inc_by(7);
        }
        registry
            .get_prometheus_registry()
            .register(Box::new(counter))
            .expect("register");
    }

    let mut text = String::new();
    for i in 0..engine_families {
        text.push_str(&format!(
            "# HELP vllm:metric_{i} Engine gauge {i}\n\
             # TYPE vllm:metric_{i} gauge\n\
             vllm:metric_{i}{{engine=\"0\",model_name=\"llama\"}} {i}\n\
             # HELP vllm:latency_{i}_seconds Engine histogram {i}\n\
             # TYPE vllm:latency_{i}_seconds histogram\n\
             vllm:latency_{i}_seconds_bucket{{model_name=\"llama\",le=\"0.1\"}} 1\n\
             vllm:latency_{i}_seconds_bucket{{model_name=\"llama\",le=\"1\"}} 4\n\
             vllm:latency_{i}_seconds_bucket{{model_name=\"llama\",le=\"+Inf\"}} 9\n\
             vllm:latency_{i}_seconds_sum{{model_name=\"llama\"}} 2.5\n\
             vllm:latency_{i}_seconds_count{{model_name=\"llama\"}} 9\n"
        ));
    }
    registry.add_expfmt_callback(Arc::new(move || Ok(text.clone())));

    registry
}

fn bench_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_collection");

    for (native, engine) in [(50, 20), (200, 60)] {
        let registry = registry(native, engine);
        let label = format!("{native}native_{engine}engine");

        // The scrape path. Unchanged behaviour; benched to catch regressions
        // from the shared-merger refactor.
        group.bench_function(format!("expfmt_combined/{label}"), |b| {
            b.iter(|| black_box(registry.prometheus_expfmt_combined().expect("expfmt")))
        });

        // The export path: same collection, plus parsing engine text into
        // families instead of appending it as a string.
        group.bench_function(format!("families_combined/{label}"), |b| {
            b.iter(|| black_box(registry.metric_families_combined().expect("families")))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_collection);
criterion_main!(benches);
