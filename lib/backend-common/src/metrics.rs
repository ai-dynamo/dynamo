// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`EngineMetrics`] — slim metrics-only handle for [`LLMEngine`] authors.
//!
//! `Worker` constructs an `EngineMetrics` from the endpoint's
//! [`MetricsHierarchy`] and hands it to the engine via
//! [`LLMEngine::register_prometheus`]. Engines never see the full `Endpoint`
//! — only the surface needed to bridge a foreign registry into the runtime's
//! `/metrics` output via [`EngineMetrics::add_expfmt_callback`].
//!
//! Native Rust `create_*` (with model labels merged in) is intentionally not
//! exposed here yet — `dynamo_runtime::metrics::create_metric` injects only
//! hierarchy labels (`dynamo_namespace`, `dynamo_component`,
//! `dynamo_endpoint`, `worker_id`). When a Rust engine needs to emit
//! native metrics labelled with `model`/`model_name`, add label-aware
//! `create_*` wrappers on `EngineMetrics` that merge in `auto_labels`
//! at construction time.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_runtime::metrics::{
    MetricsHierarchy, PrometheusExpositionFormatCallback, prometheus_names::labels,
};

use crate::engine::EngineConfig;

/// Metrics handle passed to [`LLMEngine::register_prometheus`].
/// Not `Clone` — engines should retain returned instruments, not this object.
pub struct EngineMetrics {
    hierarchy: Arc<dyn MetricsHierarchy>,
    auto_labels: Arc<HashMap<String, String>>,
}

impl EngineMetrics {
    /// Wrap a hierarchy without a model identity. Auto-labels omit
    /// `model` / `model_name`.
    pub fn from_hierarchy<H>(hierarchy: H) -> Self
    where
        H: MetricsHierarchy + 'static,
    {
        let arc: Arc<dyn MetricsHierarchy> = Arc::new(hierarchy);
        let labels = compute_auto_labels(&*arc, None, None);
        Self {
            hierarchy: arc,
            auto_labels: Arc::new(labels),
        }
    }

    /// Wrap a hierarchy plus model identity from [`EngineConfig`].
    pub fn with_engine_config<H>(hierarchy: H, engine_config: &EngineConfig) -> Self
    where
        H: MetricsHierarchy + 'static,
    {
        let arc: Arc<dyn MetricsHierarchy> = Arc::new(hierarchy);
        let labels = compute_auto_labels(
            &*arc,
            Some(&engine_config.model),
            engine_config.served_model_name.as_deref(),
        );
        Self {
            hierarchy: arc,
            auto_labels: Arc::new(labels),
        }
    }

    /// Borrow the underlying hierarchy. Exposed for the FFI bridge.
    pub fn hierarchy(&self) -> &Arc<dyn MetricsHierarchy> {
        &self.hierarchy
    }

    /// Precomputed auto-labels for the FFI bridge.
    pub fn auto_labels(&self) -> &Arc<HashMap<String, String>> {
        &self.auto_labels
    }

    /// Register a scrape callback for a foreign Prometheus registry.
    /// Auto-labels are not injected — the callback owns its own labelling.
    pub fn add_expfmt_callback(&self, callback: PrometheusExpositionFormatCallback) {
        self.hierarchy
            .get_metrics_registry()
            .add_expfmt_callback(callback);
    }
}

/// Standalone hierarchy for tests — no parent, no DRT, no connection_id.
#[cfg(any(test, feature = "testing"))]
#[derive(Default)]
pub struct TestHierarchy {
    registry: dynamo_runtime::metrics::MetricsRegistry,
}

#[cfg(any(test, feature = "testing"))]
impl TestHierarchy {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(any(test, feature = "testing"))]
impl MetricsHierarchy for TestHierarchy {
    fn basename(&self) -> String {
        "test".to_string()
    }
    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        Vec::new()
    }
    fn get_metrics_registry(&self) -> &dynamo_runtime::metrics::MetricsRegistry {
        &self.registry
    }
}

/// `served_model_name` wins over `model` for the `model_name` label;
/// both default to `model` otherwise.
fn compute_auto_labels(
    hierarchy: &dyn MetricsHierarchy,
    model: Option<&str>,
    served_model_name: Option<&str>,
) -> HashMap<String, String> {
    let mut out = HashMap::new();

    // Hierarchy chain is [DRT, namespace, component, endpoint]; chain[0]
    // is the DRT basename which doesn't correspond to a labelled level.
    let parents = hierarchy.parent_hierarchies();
    let mut chain: Vec<String> = parents.iter().map(|p| p.basename()).collect();
    chain.push(hierarchy.basename());

    let mut put = |idx: usize, key: &str| {
        if let Some(v) = chain.get(idx).filter(|s| !s.is_empty()) {
            out.insert(key.to_string(), v.clone());
        }
    };
    put(1, labels::NAMESPACE);
    put(2, labels::COMPONENT);
    put(3, labels::ENDPOINT);

    if let Some(id) = hierarchy.connection_id() {
        out.insert(labels::WORKER_ID.to_string(), format!("{:x}", id));
    }

    if let Some(m) = model.filter(|s| !s.is_empty()) {
        out.insert(labels::MODEL.to_string(), m.to_string());
        out.insert(labels::MODEL_NAME.to_string(), m.to_string());
    }
    if let Some(s) = served_model_name.filter(|s| !s.is_empty()) {
        out.insert(labels::MODEL_NAME.to_string(), s.to_string());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_engine_config_populates_model_labels() {
        let config = EngineConfig {
            model: "/local/path/qwen".to_string(),
            served_model_name: Some("qwen3-0.6b".to_string()),
            ..Default::default()
        };
        let labels = EngineMetrics::with_engine_config(TestHierarchy::new(), &config)
            .auto_labels()
            .clone();
        assert_eq!(labels.get("model").unwrap(), "/local/path/qwen");
        // served_model_name wins over model for the `model_name` label.
        assert_eq!(labels.get("model_name").unwrap(), "qwen3-0.6b");
    }

    #[test]
    fn empty_model_strings_do_not_emit_labels() {
        let config = EngineConfig {
            model: String::new(),
            served_model_name: Some(String::new()),
            ..Default::default()
        };
        let labels = EngineMetrics::with_engine_config(TestHierarchy::new(), &config)
            .auto_labels()
            .clone();
        assert!(!labels.contains_key("model"));
        assert!(!labels.contains_key("model_name"));
    }

    #[test]
    fn add_expfmt_callback_appears_in_combined_scrape() {
        let m = EngineMetrics::from_hierarchy(TestHierarchy::new());
        m.add_expfmt_callback(Arc::new(|| Ok("# external metric\n".to_string())));
        let text = m
            .hierarchy()
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .expect("expfmt");
        assert!(text.contains("# external metric"));
    }
}
