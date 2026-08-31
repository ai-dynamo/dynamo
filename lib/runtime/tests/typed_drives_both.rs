// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Can one typed collection drive both OTLP and /metrics?

use dynamo_runtime::metrics::{MetricsRegistry, prom_typed};
use prometheus::Encoder;
use std::sync::Arc;

const TYPED: &str = include_str!("data/vllm-same-registry-typed.json");
const TEXT: &str = include_str!("data/vllm-same-registry.txt");

fn registry_with_typed_engine() -> MetricsRegistry {
    let registry = MetricsRegistry::new();
    // A native Dynamo metric, already rendered by TextEncoder today.
    let counter =
        prometheus::IntCounter::new("dynamo_requests_total", "Dynamo requests").expect("counter");
    counter.inc_by(5);
    registry
        .get_prometheus_registry()
        .register(Box::new(counter))
        .expect("register");
    // Engine metrics arriving typed instead of as text.
    registry.add_typed_callback(Arc::new(|| {
        Ok(prom_typed::build_families(
            serde_json::from_str(TYPED).expect("typed fixture"),
        ))
    }));
    registry
}

/// /metrics rendered from the same typed collection that feeds OTLP.
fn metrics_from_typed(registry: &MetricsRegistry) -> String {
    let families = registry.metric_families_combined().expect("combined");
    let mut buf = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&families, &mut buf)
        .expect("encode");
    String::from_utf8(buf).expect("utf8")
}

/// Both surfaces from one collection: Dynamo's own metrics and the engine's
/// both render, and the engine families are the ones the engine exposed.
#[test]
fn one_typed_collection_can_render_metrics_and_feed_otlp() {
    let registry = registry_with_typed_engine();
    let rendered = metrics_from_typed(&registry);

    assert!(
        rendered.contains("dynamo_requests_total 5"),
        "native metric missing"
    );
    assert!(
        rendered.contains("vllm:num_requests_running"),
        "engine gauge missing"
    );
    assert!(
        rendered.contains("vllm:time_to_first_token_seconds_bucket"),
        "engine histogram missing"
    );
    assert!(rendered.contains("_created"), "_created must survive");
    assert!(!rendered.contains("le=\"inf\""), "invalid infinity bound");

    // Same collection is what OTLP consumes.
    let families = registry.metric_families_combined().expect("combined");
    assert!(families.iter().any(|f| f.name().starts_with("vllm:")));
    assert!(families.iter().any(|f| f.name() == "dynamo_requests_total"));
}

/// Every engine series the engine exposed still appears when /metrics is
/// rendered from typed. Compares series identity, ignoring numeric formatting
/// of the bucket bound, which is a rendering choice rather than a metric.
#[test]
fn no_engine_series_is_lost_when_metrics_is_rendered_from_typed() {
    let rendered = metrics_from_typed(&registry_with_typed_engine());

    /// Parse to (name, sorted labels) so label ordering -- which the two
    /// renderers disagree on and which carries no meaning -- cannot register
    /// as a lost series. Numeric bucket bounds are canonicalised so `le="1"`
    /// and `le="1.0"` compare equal.
    fn key(text: &str) -> std::collections::BTreeSet<(String, Vec<(String, String)>)> {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| l.rfind(' ').map(|i| &l[..i]))
            .filter(|head| head.starts_with("vllm"))
            .map(|head| match head.find('{') {
                Some(open) => {
                    let close = head.rfind('}').unwrap_or(head.len());
                    let mut labels: Vec<(String, String)> = head[open + 1..close]
                        .split(',')
                        .filter(|p| !p.trim().is_empty())
                        .filter_map(|p| p.split_once('='))
                        .map(|(k, v)| {
                            let k = k.trim().to_string();
                            let v = v.trim().trim_matches('"');
                            let v = if k == "le" || k == "quantile" {
                                v.parse::<f64>()
                                    .map(|f| format!("{f:?}"))
                                    .unwrap_or_else(|_| v.into())
                            } else {
                                v.to_string()
                            };
                            (k, v)
                        })
                        .collect();
                    labels.sort();
                    (head[..open].trim().to_string(), labels)
                }
                None => (head.trim().to_string(), Vec::new()),
            })
            .collect()
    }

    let engine = key(TEXT);
    let ours = key(&rendered);
    let missing: Vec<_> = engine.difference(&ours).take(5).collect();
    assert!(
        missing.is_empty(),
        "{} engine series lost, e.g. {missing:?}",
        engine.difference(&ours).count()
    );
    assert!(
        engine.len() > 300,
        "fixture should be substantial: {}",
        engine.len()
    );
}
