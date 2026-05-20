// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`EngineMetrics`] — slim metrics-only handle for [`LLMEngine`] authors,
//! plus [`LifecycleGauges`] — framework-owned gauges emitted independently
//! of engine opt-in (cleanup_time, drain_time, model_load_time).
//!
//! `Worker` constructs an `EngineMetrics` from the endpoint's
//! [`MetricsHierarchy`] and hands it to the engine via
//! [`LLMEngine::setup_metrics`]. Engines never see the full `Endpoint` —
//! only the surface needed to bridge a foreign registry into the runtime's
//! `/metrics` output via [`EngineMetrics::add_expfmt_callback`].

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_runtime::metrics::{
    MetricsHierarchy, PrometheusExpositionFormatCallback, create_metric, prometheus_names::labels,
};

use crate::engine::EngineConfig;
use crate::error::{BackendError, DynamoError, ErrorType};

/// Metrics handle passed to [`LLMEngine::setup_metrics`].
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

/// Helper: build the `const_labels` slice that goes into `create_metric`
/// — strips out the auto-injected hierarchy labels (namespace, component,
/// endpoint, worker_id) so the runtime can re-inject them. Keeps
/// model/model_name and anything else engine-specific.
fn const_labels(metrics: &EngineMetrics) -> Vec<(&str, &str)> {
    let auto_injected = [
        labels::NAMESPACE,
        labels::COMPONENT,
        labels::ENDPOINT,
        labels::WORKER_ID,
    ];
    metrics
        .auto_labels()
        .iter()
        .filter(|(k, _)| !auto_injected.contains(&k.as_str()))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

fn gauge_err(name: &str, e: anyhow::Error) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(format!("gauge create {name}: {e}"))
        .build()
}

/// Framework-owned lifecycle gauges. Emitted by `Worker` independent of
/// the engine — operators see cleanup/drain timing + model load time in
/// `/metrics` regardless of engine opt-in. The gauge names land in the
/// `dynamo_component_*` namespace via the runtime's
/// `build_component_metric_name`.
pub struct LifecycleGauges {
    cleanup_time_seconds: prometheus::Gauge,
    drain_time_seconds: prometheus::Gauge,
    #[allow(dead_code)]
    model_load_time_seconds: prometheus::Gauge,
}

impl LifecycleGauges {
    /// Construct + register the three lifecycle gauges. `model_load_time`
    /// is seeded with the measured `engine.start()` elapsed; cleanup and
    /// drain start at 0 and are populated during shutdown.
    pub fn new(metrics: &EngineMetrics, model_load_time_seconds: f64) -> Result<Self, DynamoError> {
        let labels = const_labels(metrics);
        let hierarchy = metrics.hierarchy().as_ref();
        let build = |name: &str, help: &str| {
            create_metric::<prometheus::Gauge, _>(hierarchy, name, help, &labels, None, None)
                .map_err(|e| gauge_err(name, e))
        };
        let cleanup = build(
            "cleanup_time_seconds",
            "Time spent releasing engine resources during shutdown. Set \
             by the framework once after engine.cleanup() returns.",
        )?;
        let drain = build(
            "drain_time_seconds",
            "Time spent draining in-flight work before cleanup. Stays at \
             0 for engines without a drain hook.",
        )?;
        let model_load = build(
            "model_load_time_seconds",
            "Time engine.start() took to return. Set once at Worker setup.",
        )?;
        model_load.set(model_load_time_seconds);
        Ok(Self {
            cleanup_time_seconds: cleanup,
            drain_time_seconds: drain,
            model_load_time_seconds: model_load,
        })
    }

    /// Record cleanup latency. `Worker` calls this exactly once during
    /// shutdown after `engine.cleanup()` returns.
    pub fn observe_cleanup_time(&self, seconds: f64) {
        self.cleanup_time_seconds.set(seconds);
    }

    /// Record drain latency. `Worker` calls this exactly once during
    /// graceful shutdown after `engine.drain()` returns.
    pub fn observe_drain_time(&self, seconds: f64) {
        self.drain_time_seconds.set(seconds);
    }
}

/// Framework-owned per-rank engine gauges in the `dynamo_component_*`
/// namespace. Written by [`crate::snapshot_publisher::SnapshotPublisher`]
/// on every engine push — no GIL acquisition, no tokio poll task, just
/// atomic stores into the runtime's `MetricsRegistry`.
///
/// Replaces the previous Python `LLMBackendMetrics` registry on the
/// unified path. Legacy entry points still use the Python class for
/// their own `dynamo_component_*` surface; the two registries never
/// share a process.
pub struct ComponentGauges {
    total_blocks: prometheus::IntGaugeVec,
    gpu_cache_usage_percent: prometheus::GaugeVec,
    kv_cache_hit_rate: prometheus::GaugeVec,
}

impl ComponentGauges {
    pub fn new(metrics: &EngineMetrics) -> Result<Self, DynamoError> {
        let const_label_values = const_labels(metrics);
        let hierarchy = metrics.hierarchy().as_ref();
        let build_int = |name: &str, help: &str| {
            create_metric::<prometheus::IntGaugeVec, _>(
                hierarchy,
                name,
                help,
                &const_label_values,
                None,
                Some(&["dp_rank"]),
            )
            .map_err(|e| gauge_err(name, e))
        };
        let build_f64 = |name: &str, help: &str| {
            create_metric::<prometheus::GaugeVec, _>(
                hierarchy,
                name,
                help,
                &const_label_values,
                None,
                Some(&["dp_rank"]),
            )
            .map_err(|e| gauge_err(name, e))
        };
        let total_blocks = build_int(
            "total_blocks",
            "Total number of KV cache blocks available on the worker.",
        )?;
        let gpu_cache_usage_percent = build_f64(
            "gpu_cache_usage_percent",
            "GPU cache usage as a percentage (0.0-1.0).",
        )?;
        let kv_cache_hit_rate = build_f64(
            "kv_cache_hit_rate",
            "Prefix cache hit rate (0.0-1.0). Portable across engines.",
        )?;
        Ok(Self {
            total_blocks,
            gpu_cache_usage_percent,
            kv_cache_hit_rate,
        })
    }

    /// Apply a snapshot to the per-rank gauges. Atomic stores; safe to
    /// call concurrently from multiple engine threads as long as each
    /// targets a distinct `dp_rank`.
    pub fn update(&self, snap: &crate::engine::ComponentSnapshot) {
        let rank = snap.dp_rank.to_string();
        self.total_blocks
            .with_label_values(&[&rank])
            .set(snap.kv_total_blocks as i64);
        self.gpu_cache_usage_percent
            .with_label_values(&[&rank])
            .set(snap.gpu_cache_usage as f64);
        // hit_rate is tri-state: None means "no data yet / no prefix
        // cache" — skip the gauge update entirely. 0.0 is a legitimate
        // measurement (no hits) and DOES write.
        if let Some(hr) = snap.kv_cache_hit_rate {
            self.kv_cache_hit_rate
                .with_label_values(&[&rank])
                .set(hr as f64);
        }
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

    /// Multi-level test hierarchy. Mirrors production's
    /// [drt, namespace, component, endpoint] chain so
    /// `compute_auto_labels` emits dynamo_namespace / dynamo_component /
    /// dynamo_endpoint — the entries that collide with `create_metric`'s
    /// auto-injection if `LifecycleGauges::new` doesn't filter them out.
    struct NamedHierarchy {
        registry: dynamo_runtime::metrics::MetricsRegistry,
        name: String,
        parents: Vec<NamedHierarchy>,
    }

    impl MetricsHierarchy for NamedHierarchy {
        fn basename(&self) -> String {
            self.name.clone()
        }
        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            self.parents
                .iter()
                .map(|p| p as &dyn MetricsHierarchy)
                .collect()
        }
        fn get_metrics_registry(&self) -> &dynamo_runtime::metrics::MetricsRegistry {
            &self.registry
        }
    }

    fn leaf(name: &str) -> NamedHierarchy {
        NamedHierarchy {
            registry: Default::default(),
            name: name.to_string(),
            parents: Vec::new(),
        }
    }

    /// Regression: `LifecycleGauges::new` must filter auto-injected
    /// hierarchy labels out of `metrics.auto_labels()` before handing to
    /// `create_metric` — otherwise `create_metric` rejects with "Label
    /// already auto-added" and the gauges silently fail to register.
    #[test]
    fn lifecycle_gauges_register_and_observe() {
        let endpoint = NamedHierarchy {
            registry: Default::default(),
            name: "generate".to_string(),
            parents: vec![leaf("drt"), leaf("dyn"), leaf("backend")],
        };
        let config = EngineConfig {
            model: "test-model".to_string(),
            ..Default::default()
        };
        let metrics = EngineMetrics::with_engine_config(endpoint, &config);
        // Sanity: auto_labels contains the entries that would collide with
        // create_metric's auto-injection.
        let auto = metrics.auto_labels();
        assert_eq!(auto.get(labels::NAMESPACE).map(String::as_str), Some("dyn"));
        assert_eq!(
            auto.get(labels::COMPONENT).map(String::as_str),
            Some("backend")
        );
        assert_eq!(
            auto.get(labels::ENDPOINT).map(String::as_str),
            Some("generate")
        );

        let lifecycle = LifecycleGauges::new(&metrics, 0.0).expect("construct lifecycle gauges");
        lifecycle.observe_cleanup_time(1.5);
        lifecycle.observe_drain_time(0.25);

        let text = metrics
            .hierarchy()
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .expect("expfmt");
        // Find the data row for each gauge (skip HELP/TYPE comment lines)
        // and assert the observed value is the trailing token. Per-line
        // matching avoids false-positives where some other gauge on the
        // registry happens to read the same value.
        let data_line = |gauge: &str| -> Option<&str> {
            text.lines()
                .find(|l| l.starts_with(gauge) && !l.starts_with('#'))
        };
        let cleanup = data_line("dynamo_component_cleanup_time_seconds")
            .unwrap_or_else(|| panic!("cleanup gauge data row missing: {text}"));
        let drain = data_line("dynamo_component_drain_time_seconds")
            .unwrap_or_else(|| panic!("drain gauge data row missing: {text}"));
        assert!(cleanup.ends_with(" 1.5"), "cleanup value wrong: {cleanup}");
        assert!(drain.ends_with(" 0.25"), "drain value wrong: {drain}");
    }
}
