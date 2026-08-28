// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The typed callback must reach the structured path without any text step.

use dynamo_runtime::metrics::{MetricsRegistry, prom_typed};
use std::sync::Arc;

/// Families registered through the typed callback appear in the structured
/// collection, and `/metrics` is untouched by them -- the two paths stay
/// independent, so adopting typed cannot alter what scrapers already see.
#[test]
fn typed_callback_feeds_structured_path_only() {
    let registry = MetricsRegistry::new();
    let typed: Vec<prom_typed::TypedFamily> = serde_json::from_str(
        r#"[{"name":"vllm:num_requests_running","help":"Running","type":"gauge",
             "samples":[{"name":"vllm:num_requests_running",
                         "labels":{"model_name":"llama"},"value":4.0}]}]"#,
    )
    .expect("fixture");
    registry.add_typed_callback(Arc::new(move || {
        Ok(prom_typed::build_families(
            serde_json::from_str(
                r#"[{"name":"vllm:num_requests_running","help":"Running","type":"gauge",
                     "samples":[{"name":"vllm:num_requests_running",
                                 "labels":{"model_name":"llama"},"value":4.0}]}]"#,
            )
            .expect("fixture"),
        ))
    }));
    drop(typed);

    let families = registry.metric_families_combined().expect("combined");
    let found = families
        .iter()
        .find(|f| f.name() == "vllm:num_requests_running")
        .expect("typed family missing from structured collection");
    assert_eq!(found.help(), "Running");
    assert_eq!(found.get_metric()[0].get_gauge().value(), 4.0);

    let text = registry.prometheus_expfmt_combined().expect("text");
    assert!(
        !text.contains("vllm:num_requests_running"),
        "typed families must not leak into /metrics; that surface stays the engine's own rendering"
    );
}

/// A failing typed callback is skipped, not fatal: one broken engine must not
/// empty the collection for every other.
#[test]
fn failing_typed_callback_is_skipped() {
    let registry = MetricsRegistry::new();
    registry.add_typed_callback(Arc::new(|| anyhow::bail!("engine exploded")));
    let families = registry
        .metric_families_combined()
        .expect("collection survives");
    assert!(families.is_empty());
}
