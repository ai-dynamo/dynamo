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
use prometheus::Encoder;
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

    let mut typed_families = Vec::new();
    for i in 0..engine_families {
        let mut family = prometheus::proto::MetricFamily::new();
        family.set_name(format!("vllm:metric_{i}"));
        family.set_help(format!("Engine gauge {i}"));
        family.set_field_type(prometheus::proto::MetricType::GAUGE);
        let mut metric = prometheus::proto::Metric::new();
        let mut gauge = prometheus::proto::Gauge::new();
        gauge.set_value(i as f64);
        metric.set_gauge(gauge);
        family.mut_metric().push(metric);
        typed_families.push(family);
    }
    // Both surfaces are fed independently, so bench them as they run in
    // production: the scrape appends the engine's exposition text, the export
    // takes the same metrics typed.
    let expfmt_text = {
        let mut buffer = Vec::new();
        prometheus::TextEncoder::new()
            .encode(&typed_families, &mut buffer)
            .expect("encode");
        String::from_utf8(buffer).expect("utf8")
    };
    registry.add_expfmt_callback(Arc::new(move || Ok(expfmt_text.clone())));
    registry.add_typed_callback(Arc::new(move || Ok(typed_families.clone())));

    registry
}

fn bench_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_collection");

    for (native, engine) in [(50, 20), (200, 60)] {
        let registry = registry(native, engine);
        let label = format!("{native}native_{engine}engine");

        // The scrape path: unchanged behaviour, benched to catch regressions
        // from sharing the family merger with the export path.
        group.bench_function(format!("expfmt_combined/{label}"), |b| {
            b.iter(|| black_box(registry.prometheus_expfmt_combined().expect("expfmt")))
        });

        // The export path: the same metrics delivered typed and merged into
        // the native families, instead of appended as text.
        group.bench_function(format!("families_combined/{label}"), |b| {
            b.iter(|| black_box(registry.metric_families_combined().expect("families")))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_collection);
criterion_main!(benches);
