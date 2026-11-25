// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics registry trait and implementation for Prometheus metrics
//!
//! This module provides a trait-based interface for creating and managing Prometheus metrics
//! with automatic label injection and hierarchical naming support.

pub mod frontend_perf;
pub mod otlp_export;
pub mod prom_typed;
pub mod prometheus_names;
pub mod quic_response;
pub mod request_plane;
pub mod tokio_perf;
pub mod transport_metrics;
pub mod work_handler_perf;
pub mod work_handler_pool;

#[cfg(test)]
mod tests;

use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;

use once_cell::sync::Lazy;
use std::collections::HashMap;

// Import commonly used items to avoid verbose prefixes
use prometheus_names::{build_metric_name, labels, name_prefix, sanitize_prometheus_label};

// Prometheus imports
use prometheus::Encoder;

/// Validate that a label slice has no duplicate keys.
/// Returns Ok(()) when all keys are unique; otherwise returns an error naming the duplicate key.
fn validate_no_duplicate_label_keys(labels: &[(&str, &str)]) -> anyhow::Result<()> {
    let mut seen_keys = std::collections::HashSet::new();
    for (key, _) in labels {
        if !seen_keys.insert(*key) {
            return Err(anyhow::anyhow!(
                "Duplicate label key '{}' found in labels",
                key
            ));
        }
    }
    Ok(())
}

/// ==============================
/// Prometheus section
/// ==============================
/// Trait that defines common behavior for Prometheus metric types
pub trait PrometheusMetric: prometheus::core::Collector + Clone + Send + Sync + 'static {
    /// Create a new metric with the given options
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error>
    where
        Self: Sized;

    /// Create a new metric with histogram options and custom buckets
    /// This is a default implementation that will panic for non-histogram metrics
    fn with_histogram_opts_and_buckets(
        _opts: prometheus::HistogramOpts,
        _buckets: Option<Vec<f64>>,
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!("with_histogram_opts_and_buckets is not implemented for this metric type");
    }

    /// Create a histogram vector with custom buckets and label names.
    fn with_histogram_opts_buckets_and_label_names(
        _opts: prometheus::HistogramOpts,
        _buckets: Option<Vec<f64>>,
        _label_names: &[&str],
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!(
            "with_histogram_opts_buckets_and_label_names is not implemented for this metric type"
        );
    }

    /// Create a new metric with counter options and label names (for CounterVec)
    /// This is a default implementation that will panic for non-countervec metrics
    fn with_opts_and_label_names(
        _opts: prometheus::Opts,
        _label_names: &[&str],
    ) -> Result<Self, prometheus::Error>
    where
        Self: Sized,
    {
        panic!("with_opts_and_label_names is not implemented for this metric type");
    }
}

// Implement the trait for Counter, IntCounter, and Gauge
impl PrometheusMetric for prometheus::Counter {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::Counter::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::IntCounter {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::IntCounter::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::Gauge {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::Gauge::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::IntGauge {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        prometheus::IntGauge::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::GaugeVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "GaugeVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::GaugeVec::new(opts, label_names)
    }
}

impl PrometheusMetric for prometheus::IntGaugeVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "IntGaugeVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::IntGaugeVec::new(opts, label_names)
    }
}

impl PrometheusMetric for prometheus::IntCounterVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "IntCounterVec requires label names, use with_opts_and_label_names instead".to_string(),
        ))
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::IntCounterVec::new(opts, label_names)
    }
}

// Implement the trait for Histogram
impl PrometheusMetric for prometheus::Histogram {
    fn with_opts(opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        // Convert Opts to HistogramOpts
        let histogram_opts = prometheus::HistogramOpts::new(opts.name, opts.help);
        prometheus::Histogram::with_opts(histogram_opts)
    }

    fn with_histogram_opts_and_buckets(
        mut opts: prometheus::HistogramOpts,
        buckets: Option<Vec<f64>>,
    ) -> Result<Self, prometheus::Error> {
        if let Some(custom_buckets) = buckets {
            opts = opts.buckets(custom_buckets);
        }
        prometheus::Histogram::with_opts(opts)
    }
}

impl PrometheusMetric for prometheus::HistogramVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        Err(prometheus::Error::Msg(
            "HistogramVec requires histogram options and label names".to_string(),
        ))
    }

    fn with_histogram_opts_buckets_and_label_names(
        mut opts: prometheus::HistogramOpts,
        buckets: Option<Vec<f64>>,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        if let Some(custom_buckets) = buckets {
            opts = opts.buckets(custom_buckets);
        }
        prometheus::HistogramVec::new(opts, label_names)
    }
}

// Implement the trait for CounterVec
impl PrometheusMetric for prometheus::CounterVec {
    fn with_opts(_opts: prometheus::Opts) -> Result<Self, prometheus::Error> {
        // This will panic - CounterVec needs label names
        panic!("CounterVec requires label names, use with_opts_and_label_names instead");
    }

    fn with_opts_and_label_names(
        opts: prometheus::Opts,
        label_names: &[&str],
    ) -> Result<Self, prometheus::Error> {
        prometheus::CounterVec::new(opts, label_names)
    }
}

/// ==============================
/// Metrics section
/// ==============================
/// Public helper function to create metrics - accessible for Python bindings
pub fn create_metric<T: PrometheusMetric, H: MetricsHierarchy + ?Sized>(
    hierarchy: &H,
    metric_name: &str,
    metric_desc: &str,
    labels: &[(&str, &str)],
    buckets: Option<Vec<f64>>,
    const_labels: Option<&[&str]>,
) -> anyhow::Result<T> {
    create_metric_with_options(
        hierarchy,
        MetricOptions::default(),
        metric_name,
        metric_desc,
        labels,
        buckets,
        const_labels,
    )
}

/// How [`create_metric_with_options`] names and registers a metric. The
/// default is what [`create_metric`] does.
#[derive(Clone, Copy, Debug)]
pub struct MetricOptions {
    /// The metric is named `{prefix}_{metric_name}`. Only this prefix is ever
    /// used: none is inferred from the metric name.
    pub prefix: &'static str,
    /// When the hierarchy's registry refuses the metric, log a warning and
    /// return it unregistered, still usable but not exposed, instead of
    /// failing.
    pub warn_on_registration_failure: bool,
}

impl Default for MetricOptions {
    fn default() -> Self {
        Self {
            prefix: name_prefix::COMPONENT,
            warn_on_registration_failure: false,
        }
    }
}

/// [`create_metric`] under explicit [`MetricOptions`]: the same validation,
/// auto-labels from the hierarchy, construction and registration.
pub fn create_metric_with_options<T: PrometheusMetric, H: MetricsHierarchy + ?Sized>(
    hierarchy: &H,
    options: MetricOptions,
    metric_name: &str,
    metric_desc: &str,
    labels: &[(&str, &str)],
    buckets: Option<Vec<f64>>,
    const_labels: Option<&[&str]>,
) -> anyhow::Result<T> {
    // Validate that user-provided labels don't have duplicate keys
    validate_no_duplicate_label_keys(labels)?;
    // Note: stored labels functionality has been removed

    let basename = hierarchy.basename();
    let parent_hierarchies = hierarchy.parent_hierarchies();

    // Build hierarchy path as vector of strings: parent names + [basename]
    let mut hierarchy_names: Vec<String> =
        parent_hierarchies.iter().map(|p| p.basename()).collect();
    hierarchy_names.push(basename.clone());

    let metric_name = build_metric_name(options.prefix, metric_name);

    // Build updated_labels: auto-labels first, then `labels` + stored labels
    let mut updated_labels: Vec<(String, String)> = Vec::new();

    // Auto-label injection: Always add dynamo_namespace, dynamo_component, dynamo_endpoint labels
    // based on the hierarchy. Label constants defined in prometheus_names.rs labels module.
    //
    // Python counterpart: components/src/dynamo/common/utils/prometheus.py register_engine_metrics_callback()

    // Validate that user-provided labels don't conflict with auto-generated labels
    for (key, _) in labels {
        if *key == labels::NAMESPACE
            || *key == labels::COMPONENT
            || *key == labels::ENDPOINT
            || *key == labels::WORKER_ID
        {
            return Err(anyhow::anyhow!(
                "Label '{}' is automatically added by auto-label injection and cannot be manually set",
                key
            ));
        }
    }

    // Also validate that vector label names (const_labels) don't collide with auto-injected
    // const labels. A variable label named "worker_id" would conflict with the auto-injected
    // worker_id const label, causing a prometheus registration error or ambiguous output.
    if let Some(label_names) = const_labels {
        for name in label_names.iter() {
            if *name == labels::NAMESPACE
                || *name == labels::COMPONENT
                || *name == labels::ENDPOINT
                || *name == labels::WORKER_ID
            {
                return Err(anyhow::anyhow!(
                    "Variable label name '{}' conflicts with auto-injected const label and cannot be used",
                    name
                ));
            }
        }
    }

    // Add auto-generated labels with sanitized values
    // Hierarchy: [drt, namespace, component, endpoint]
    if hierarchy_names.len() > 1 {
        let namespace = &hierarchy_names[1];
        if !namespace.is_empty() {
            let valid_namespace = sanitize_prometheus_label(namespace)?;
            if !valid_namespace.is_empty() {
                updated_labels.push((labels::NAMESPACE.to_string(), valid_namespace));
            }
        }
    }
    if hierarchy_names.len() > 2 {
        let component = &hierarchy_names[2];
        if !component.is_empty() {
            let valid_component = sanitize_prometheus_label(component)?;
            if !valid_component.is_empty() {
                updated_labels.push((labels::COMPONENT.to_string(), valid_component));
            }
        }
    }
    if hierarchy_names.len() > 3 {
        let endpoint = &hierarchy_names[3];
        if !endpoint.is_empty() {
            let valid_endpoint = sanitize_prometheus_label(endpoint)?;
            if !valid_endpoint.is_empty() {
                updated_labels.push((labels::ENDPOINT.to_string(), valid_endpoint));
            }
        }
    }

    // Auto-inject worker_id label from the hierarchy's connection_id (discovery instance ID).
    // This provides a stable per-worker identity label so metrics from different workers
    // serving the same endpoint can be distinguished without relying on Kubernetes labels.
    if let Some(conn_id) = hierarchy.connection_id() {
        updated_labels.push((labels::WORKER_ID.to_string(), format!("{:x}", conn_id)));
    }

    // Add user labels
    updated_labels.extend(
        labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );
    // Note: stored labels functionality has been removed

    // Handle different metric types
    let prometheus_metric = if std::any::TypeId::of::<T>()
        == std::any::TypeId::of::<prometheus::CounterVec>()
    {
        // Special handling for CounterVec with label names
        // const_labels parameter is required for CounterVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for CounterVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("CounterVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::GaugeVec>() {
        // Special handling for GaugeVec with label names
        // const_labels parameter is required for GaugeVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for GaugeVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("GaugeVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::HistogramVec>() {
        let mut opts = prometheus::HistogramOpts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("HistogramVec requires const_labels parameter"))?;
        T::with_histogram_opts_buckets_and_label_names(opts, buckets, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::Histogram>() {
        // Special handling for Histogram with custom buckets
        // buckets parameter is valid for Histogram, const_labels is not used
        if const_labels.is_some() {
            return Err(anyhow::anyhow!(
                "const_labels parameter is not valid for Histogram"
            ));
        }
        let mut opts = prometheus::HistogramOpts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        T::with_histogram_opts_and_buckets(opts, buckets)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::IntCounterVec>() {
        // Special handling for IntCounterVec with label names
        // const_labels parameter is required for IntCounterVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for IntCounterVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("IntCounterVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::IntGaugeVec>() {
        // Special handling for IntGaugeVec with label names
        // const_labels parameter is required for IntGaugeVec
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for IntGaugeVec"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        let label_names = const_labels
            .ok_or_else(|| anyhow::anyhow!("IntGaugeVec requires const_labels parameter"))?;
        T::with_opts_and_label_names(opts, label_names)?
    } else {
        // Standard handling for Counter, IntCounter, Gauge, IntGauge
        // buckets and const_labels parameters are not valid for these types
        if buckets.is_some() {
            return Err(anyhow::anyhow!(
                "buckets parameter is not valid for Counter, IntCounter, Gauge, or IntGauge"
            ));
        }
        if const_labels.is_some() {
            return Err(anyhow::anyhow!(
                "const_labels parameter is not valid for Counter, IntCounter, Gauge, or IntGauge"
            ));
        }
        let mut opts = prometheus::Opts::new(&metric_name, metric_desc);
        for (key, value) in &updated_labels {
            opts = opts.const_label(key.clone(), value.clone());
        }
        T::with_opts(opts)?
    };

    let collector: Box<dyn prometheus::core::Collector> = Box::new(prometheus_metric.clone());
    let registry = hierarchy.get_metrics_registry();
    if options.warn_on_registration_failure {
        registry.add_metric_or_warn(collector, &metric_name);
    } else {
        registry.add_metric(collector)?;
    }

    Ok(prometheus_metric)
}

/// Wrapper struct that provides access to metrics functionality
/// This struct is accessed via the `.metrics()` method on DistributedRuntime, Namespace, Component, and Endpoint
pub struct Metrics<H: MetricsHierarchy> {
    hierarchy: H,
}

impl<H: MetricsHierarchy> Metrics<H> {
    pub fn new(hierarchy: H) -> Self {
        Self { hierarchy }
    }

    // TODO: Add support for additional Prometheus metric types:
    // - Counter: ✅ IMPLEMENTED - create_counter()
    // - CounterVec: ✅ IMPLEMENTED - create_countervec()
    // - Gauge: ✅ IMPLEMENTED - create_gauge()
    // - GaugeVec: ✅ IMPLEMENTED - create_gaugevec()
    // - GaugeHistogram: create_gauge_histogram() - for gauge histograms
    // - Histogram: ✅ IMPLEMENTED - create_histogram()
    // - HistogramVec: ✅ IMPLEMENTED - create_histogramvec()
    // - Info: create_info() - for info metrics with labels
    // - IntCounter: ✅ IMPLEMENTED - create_intcounter()
    // - IntCounterVec: ✅ IMPLEMENTED - create_intcountervec()
    // - IntGauge: ✅ IMPLEMENTED - create_intgauge()
    // - IntGaugeVec: ✅ IMPLEMENTED - create_intgaugevec()
    // - Stateset: create_stateset() - for state-based metrics
    // - Summary: create_summary() - for quantiles and sum/count metrics
    // - SummaryVec: create_summary_vec() - for labeled summaries
    // - Untyped: create_untyped() - for untyped metrics
    //
    // NOTE: The order of create_* methods below is mirrored in lib/bindings/python/rust/lib.rs::Metrics
    // Keep them synchronized when adding new metric types

    /// Create a Counter metric
    pub fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Counter> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create a CounterVec metric with label names (for dynamic labels)
    pub fn create_countervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::CounterVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create a Gauge metric
    pub fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Gauge> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create a GaugeVec metric with label names (for dynamic labels)
    pub fn create_gaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::GaugeVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create a Histogram metric with custom buckets
    pub fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<prometheus::Histogram> {
        create_metric(&self.hierarchy, name, description, labels, buckets, None)
    }

    /// Create a HistogramVec metric with custom buckets and label names.
    pub fn create_histogramvec(
        &self,
        name: &str,
        description: &str,
        label_names: &[&str],
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<prometheus::HistogramVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            labels,
            buckets,
            Some(label_names),
        )
    }

    /// Create an IntCounter metric
    pub fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounter> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create an IntCounterVec metric with label names (for dynamic labels)
    pub fn create_intcountervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounterVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create an IntGauge metric
    pub fn create_intgauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGauge> {
        create_metric(&self.hierarchy, name, description, labels, None, None)
    }

    /// Create an IntGaugeVec metric with label names (for dynamic labels)
    pub fn create_intgaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGaugeVec> {
        create_metric(
            &self.hierarchy,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Get metrics in Prometheus text format
    pub fn prometheus_expfmt(&self) -> anyhow::Result<String> {
        self.hierarchy
            .get_metrics_registry()
            .prometheus_expfmt_combined()
    }
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsHierarchy: Send + Sync {
    // ========================================================================
    // Required methods - must be implemented by all types
    // ========================================================================

    /// Get the name of this hierarchy (without any hierarchy prefix)
    fn basename(&self) -> String;

    /// Get the parent hierarchies as actual objects (not strings)
    /// Returns a vector of hierarchy references, ordered from root to immediate parent.
    /// For example, an Endpoint would return [DRT, Namespace, Component].
    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy>;

    /// Get a reference to this hierarchy's metrics registry
    fn get_metrics_registry(&self) -> &MetricsRegistry;

    // ========================================================================
    // Provided methods - have default implementations
    // ========================================================================

    /// Get the connection ID (discovery instance ID) for this hierarchy level.
    ///
    /// Returns `Some(id)` when the hierarchy has access to the DistributedRuntime
    /// (e.g. Namespace, Component, Endpoint). Used by `create_metric()` to auto-inject
    /// the `worker_id` label. Returns `None` by default.
    fn connection_id(&self) -> Option<u64> {
        None
    }

    /// Access the metrics interface for this hierarchy
    /// This is a provided method that works for any type implementing MetricsHierarchy
    fn metrics(&self) -> Metrics<&Self>
    where
        Self: Sized,
    {
        Metrics::new(self)
    }
}

// Blanket implementation for references to types that implement MetricsHierarchy
impl<T: MetricsHierarchy + ?Sized> MetricsHierarchy for &T {
    fn basename(&self) -> String {
        (**self).basename()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        (**self).parent_hierarchies()
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        (**self).get_metrics_registry()
    }

    fn connection_id(&self) -> Option<u64> {
        (**self).connection_id()
    }
}

/// Type alias for runtime callback functions to reduce complexity
///
/// This type represents an Arc-wrapped callback function that can be:
/// - Shared efficiently across multiple threads and contexts
/// - Cloned without duplicating the underlying closure
/// - Used in generic contexts requiring 'static lifetime
///
/// The Arc wrapper is included in the type to make sharing explicit.
pub type PrometheusUpdateCallback = Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync + 'static>;

/// One collection: the merged families, plus the UNIT metadata that came with
/// the typed ones.
///
/// Kept beside the families because `prometheus::proto::MetricFamily` has no
/// unit field, and the compatibility spec requires the unit reach OTLP when the
/// source declares one.
pub struct CollectedFamilies {
    pub families: Vec<prometheus::proto::MetricFamily>,
    /// Family name -> Prometheus unit word, for families that declared one.
    pub units: HashMap<String, String>,
    /// Family name -> Prometheus type word. Carried because the proto model has
    /// no Info variant: an Info family arrives here as a gauge, and the spec
    /// converts it to a non-monotonic Sum rather than a Gauge.
    pub prom_types: HashMap<String, String>,
}

/// Type alias for exposition text callback functions that return Prometheus text
pub type PrometheusExpositionFormatCallback =
    Arc<dyn Fn() -> anyhow::Result<String> + Send + Sync + 'static>;

/// Returns already-built families, so engine metrics reach the structured
/// path without being rendered to text and parsed back.
pub type PrometheusTypedCallback =
    Arc<dyn Fn() -> anyhow::Result<Vec<prom_typed::BuiltFamily>> + Send + Sync + 'static>;

/// Structure to hold Prometheus registries and associated callbacks for a given hierarchy.
///
/// All fields are Arc-wrapped, so cloning shares state. This ensures metrics registered
/// on cloned instances (e.g., cloned Client/Endpoint) are visible to the original.
#[derive(Clone)]
pub struct MetricsRegistry {
    /// The Prometheus registry for this hierarchy.
    /// Arc-wrapped so clones share the same registry (metrics registered on clones are visible everywhere).
    pub prometheus_registry: Arc<std::sync::RwLock<prometheus::Registry>>,

    /// Child registries included when emitting combined `/metrics` output.
    ///
    /// Why this exists:
    /// - Previously, `create_metric()` registered every collector into *all* parent registries
    ///   (Endpoint → Component → Namespace → DRT) so scraping the root registry included everything.
    /// - That fan-out caused Prometheus collisions when different endpoints tried to register the
    ///   same metric name with different const-labels (descriptor mismatch).
    ///
    /// We now register metrics only into the local hierarchy registry to avoid collisions.
    /// `child_registries` rebuilds “what to scrape” as a tree of registries so `/metrics` can:
    /// - traverse registries recursively,
    /// - merge metric families into one exposition payload,
    /// - warn/drop exact duplicate series, while allowing same metric name with different labels.
    child_registries: Arc<std::sync::RwLock<Vec<MetricsRegistry>>>,

    /// Update callbacks invoked before metrics are scraped.
    /// Wrapped in Arc to preserve callbacks across clones (prevents callback loss when MetricsRegistry is cloned).
    pub prometheus_update_callbacks: Arc<std::sync::RwLock<Vec<PrometheusUpdateCallback>>>,

    /// Callbacks returning engine exposition text; these feed `/metrics` only.
    #[allow(clippy::type_complexity)]
    pub prometheus_expfmt_callbacks:
        Arc<std::sync::RwLock<Vec<PrometheusExpositionFormatCallback>>>,

    /// Callbacks returning engine metric families; these feed OTLP only.
    /// Wrapped in Arc to preserve callbacks across clones (e.g. vLLM callbacks
    /// registered at Endpoint remain reachable at DRT).
    pub prometheus_typed_callbacks: Arc<std::sync::RwLock<Vec<PrometheusTypedCallback>>>,
}

impl std::fmt::Debug for MetricsRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsRegistry")
            .field("prometheus_registry", &"<RwLock<Registry>>")
            .field(
                "prometheus_update_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_update_callbacks.read().unwrap().len()
                ),
            )
            .field(
                "prometheus_expfmt_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_expfmt_callbacks.read().unwrap().len()
                ),
            )
            .field(
                "prometheus_typed_callbacks",
                &format!(
                    "<RwLock<Vec<Callback>>> with {} callbacks",
                    self.prometheus_typed_callbacks.read().unwrap().len()
                ),
            )
            .finish()
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry with an empty Prometheus registry and callback lists
    pub fn new() -> Self {
        Self::from_prometheus_registry(prometheus::Registry::new())
    }

    fn from_prometheus_registry(registry: prometheus::Registry) -> Self {
        Self {
            prometheus_registry: Arc::new(std::sync::RwLock::new(registry)),
            child_registries: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_update_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_expfmt_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
            prometheus_typed_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// Add a child registry to be included in combined /metrics output.
    ///
    /// Dedup is by the child `MetricsRegistry`'s own `Arc` (shared by its
    /// clones), so registering the same registry, or a clone of it, again is
    /// a no-op.
    pub fn add_child_registry(&self, child: &MetricsRegistry) {
        let child_ptr = Arc::as_ptr(&child.prometheus_registry);
        let mut guard = self.child_registries.write().unwrap();
        if guard
            .iter()
            .any(|r| Arc::as_ptr(&r.prometheus_registry) == child_ptr)
        {
            return;
        }
        guard.push(child.clone());
    }

    /// Attach a plain `prometheus::Registry` as a child, so both `/metrics`
    /// (`prometheus_expfmt_combined`) and the OTLP export
    /// (`metric_families_combined`) include its families. The registry is
    /// shared, not copied: collectors registered into it later are seen too.
    ///
    /// For a component that owns its own `prometheus::Registry`. The registry
    /// stays attached until the returned handle is dropped, so a component that
    /// is torn down and rebuilt on the same runtime does not leave its old
    /// metrics behind. Attach each registry once: this is not enforced, and a
    /// registry attached twice is gathered twice, and every series in it
    /// then warns as a duplicate on each scrape and export.
    #[must_use = "the registry is detached when the handle is dropped"]
    pub fn add_prometheus_registry(
        &self,
        registry: prometheus::Registry,
    ) -> AttachedPrometheusRegistry {
        let child = Self::from_prometheus_registry(registry);
        self.add_child_registry(&child);
        AttachedPrometheusRegistry {
            parent: Arc::downgrade(&self.child_registries),
            child: child.prometheus_registry,
        }
    }

    fn registries_for_combined_scrape(&self) -> Vec<MetricsRegistry> {
        // Traverse child registries recursively so `prometheus_expfmt()` on any hierarchy
        // (DRT/namespace/component/endpoint) includes metrics from its descendants.
        //
        // Dedup by underlying Prometheus registry pointer so multiple paths (e.g. also registering
        // directly on the root) won't duplicate output.
        fn visit(
            registry: &MetricsRegistry,
            out: &mut Vec<MetricsRegistry>,
            seen: &mut HashSet<*const std::sync::RwLock<prometheus::Registry>>,
        ) {
            let ptr = Arc::as_ptr(&registry.prometheus_registry);
            if !seen.insert(ptr) {
                return;
            }

            out.push(registry.clone());

            let children: Vec<MetricsRegistry> = registry
                .child_registries
                .read()
                .unwrap()
                .iter()
                .cloned()
                .collect();
            for child in children {
                visit(&child, out, seen);
            }
        }

        let mut out = Vec::new();
        let mut seen: HashSet<*const std::sync::RwLock<prometheus::Registry>> = HashSet::new();
        visit(self, &mut out, &mut seen);
        out
    }

    /// Combine metrics across this registry and all registered children into one Prometheus exposition output.
    ///
    /// Engine metrics are appended as the text the engine produced. This is the
    /// scrape path and is deliberately independent of the typed path that feeds
    /// OTLP: `/metrics` output stays byte-for-byte what the engine emitted.
    ///
    /// - Families are merged by name. The first source to declare a name fixes its
    ///   HELP and TYPE; a later family that disagrees is warned and skipped, and
    ///   every other family is still emitted.
    /// - Multiple series for the same name are allowed if labels differ.
    /// - Exact duplicate series (same name + identical label pairs) are warned and dropped.
    pub fn prometheus_expfmt_combined(&self) -> anyhow::Result<String> {
        let registries = self.registries_for_combined_scrape();
        run_update_callbacks(&registries);

        let merged = FamilyMerger::from_gathered(&registries, &MERGE_CONFLICTS_TOTAL).into_sorted();

        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&merged, &mut buffer)?;
        let mut result = String::from_utf8(buffer)?;

        // Append expfmt callbacks deterministically in registry order.
        let mut expfmt = String::new();
        for registry in &registries {
            let text = registry.execute_expfmt_callbacks();
            if !text.is_empty() {
                if !expfmt.is_empty() && !expfmt.ends_with('\n') {
                    expfmt.push('\n');
                }
                expfmt.push_str(&text);
            }
        }

        if !expfmt.is_empty() {
            if !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(&expfmt);
        }

        Ok(result)
    }

    /// Every metric family: Dynamo's own from `gather()`, plus engine families
    /// delivered typed. This is the OTLP export path.
    ///
    /// Engine metrics cross the binding as a structure rather than as
    /// exposition text, so their name, type and help arrive authoritative
    /// instead of being re-derived by a parser.
    ///
    /// Each call re-executes the typed callbacks, which cross into Python and
    /// take the GIL, so callers should collect on a deliberate cadence.
    ///
    /// A family that conflicts with one already merged is skipped, as in
    /// [`Self::prometheus_expfmt_combined`], so collection cannot fail.
    pub fn metric_families_combined(&self) -> CollectedFamilies {
        let registries = self.registries_for_combined_scrape();
        run_update_callbacks(&registries);

        let mut merger = FamilyMerger::from_gathered(&registries, &MERGE_CONFLICTS_TOTAL);

        // The unit travels beside the family, keyed by name: the proto model
        // has no unit field, so it would otherwise be dropped here.
        let mut units: HashMap<String, String> = HashMap::new();
        let mut prom_types: HashMap<String, String> = HashMap::new();
        for registry in &registries {
            for built in registry.execute_typed_callbacks() {
                let name = built.family.name().to_string();
                // Publish the side metadata only once the family is accepted.
                // The maps are keyed by name, so metadata from a family the
                // merger rejects would otherwise be applied to the family that
                // did survive under that name -- a rejected `info` could make a
                // surviving gauge encode as a non-monotonic Sum.
                if !merger.add_or_skip(built.family) {
                    continue;
                }
                // First writer wins, matching how the merger itself resolves a
                // conflict: the first family fixes help and type, and later
                // ones must agree. A later family can be accepted and still
                // have every sample dropped as a duplicate series, so letting
                // it overwrite the unit would label the surviving data with a
                // unit from a source that contributed none of it.
                if !built.unit.is_empty() {
                    units.entry(name.clone()).or_insert(built.unit);
                }
                prom_types.entry(name).or_insert(built.prom_type);
            }
        }

        CollectedFamilies {
            families: merger.into_sorted(),
            units,
            prom_types,
        }
    }

    /// Add a callback function that receives a reference to any MetricsHierarchy
    pub fn add_update_callback(&self, callback: PrometheusUpdateCallback) {
        self.prometheus_update_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Add an exposition text callback that returns Prometheus text.
    ///
    /// This feeds `/metrics` only. Engines that also want to be exported over
    /// OTLP must register a typed callback via [`Self::add_typed_callback`].
    pub fn add_expfmt_callback(&self, callback: PrometheusExpositionFormatCallback) {
        self.prometheus_expfmt_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Execute all exposition text callbacks and return their concatenated text
    pub fn execute_expfmt_callbacks(&self) -> String {
        // Snapshot under the lock, then invoke without it held; see
        // `execute_typed_callbacks` for why holding it across the calls
        // deadlocks against registration from Python.
        let callbacks: Vec<PrometheusExpositionFormatCallback> = self
            .prometheus_expfmt_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let mut result = String::new();
        for callback in callbacks.iter() {
            match callback() {
                Ok(text) => {
                    if !text.is_empty() {
                        if !result.is_empty() && !result.ends_with('\n') {
                            result.push('\n');
                        }
                        result.push_str(&text);
                    }
                }
                Err(e) => {
                    tracing::error!("Error executing exposition text callback: {e}");
                }
            }
        }
        result
    }

    /// Register a callback that returns typed families directly.
    ///
    /// This is how engine metrics reach OTLP.
    pub fn add_typed_callback(&self, callback: PrometheusTypedCallback) {
        self.prometheus_typed_callbacks
            .write()
            .unwrap()
            .push(callback);
    }

    /// Collect from every typed callback, logging and skipping failures so one
    /// broken engine cannot empty the whole collection.
    pub fn execute_typed_callbacks(&self) -> Vec<prom_typed::BuiltFamily> {
        // Snapshot under the lock, then invoke without it held. The callbacks
        // take the Python GIL, and registration arrives from Python holding the
        // GIL and blocking on `write()` -- holding the read lock across the
        // calls would deadlock those two against each other.
        let callbacks: Vec<PrometheusTypedCallback> = self
            .prometheus_typed_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();

        callbacks
            .iter()
            .filter_map(|callback| match callback() {
                Ok(families) => Some(families),
                Err(error) => {
                    tracing::warn!(%error, "typed metrics callback failed");
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Execute all update callbacks and return their results
    pub fn execute_update_callbacks(&self) -> Vec<anyhow::Result<()>> {
        // Snapshot before invoking, for the same reason as the typed and
        // exposition callbacks: a callback may take the Python GIL or register
        // another callback, and holding the read lock across the call deadlocks
        // against a registration that holds the GIL and wants the write lock.
        let callbacks: Vec<PrometheusUpdateCallback> = self
            .prometheus_update_callbacks
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();

        callbacks.iter().map(|callback| callback()).collect()
    }

    /// Add a Prometheus metric collector to this registry
    pub fn add_metric(
        &self,
        collector: Box<dyn prometheus::core::Collector>,
    ) -> anyhow::Result<()> {
        self.prometheus_registry
            .write()
            .unwrap()
            .register(collector)
            .map_err(|e| anyhow::anyhow!("Failed to register metric: {}", e))
    }

    /// Add a Prometheus metric collector, logging a warning on failure instead of returning an error.
    pub fn add_metric_or_warn(&self, collector: Box<dyn prometheus::core::Collector>, name: &str) {
        if let Err(e) = self.add_metric(collector) {
            tracing::warn!(error = %e, metric = name, "Failed to register metric");
        }
    }

    /// Get a read guard to the Prometheus registry for scraping
    pub fn get_prometheus_registry(&self) -> std::sync::RwLockReadGuard<'_, prometheus::Registry> {
        self.prometheus_registry.read().unwrap()
    }

    /// Returns true if a metric with the given name already exists in the Prometheus registry
    pub fn has_metric_named(&self, metric_name: &str) -> bool {
        self.prometheus_registry
            .read()
            .unwrap()
            .gather()
            .iter()
            .any(|mf| mf.name() == metric_name)
    }
}

/// Run every registry's update callbacks, logging rather than propagating
/// failures so one broken callback cannot poison the whole collection.
fn run_update_callbacks(registries: &[MetricsRegistry]) {
    for registry in registries {
        for result in registry.execute_update_callbacks() {
            if let Err(error) = result {
                tracing::error!(%error, "metrics update callback failed");
            }
        }
    }
}

/// Merges metric families from several sources under one set of rules.
///
/// Extracted so the text path (`prometheus_expfmt_combined`) and the structured
/// path (`metric_families_combined`) cannot drift: a series accepted by one must
/// be accepted by the other, or `/metrics` and OTLP would disagree.
struct FamilyMerger<'a> {
    by_name: HashMap<String, prometheus::proto::MetricFamily>,
    seen_series: HashSet<String>,
    conflicts: &'a prometheus::IntCounter,
}

/// Process-wide count of metric families a combined gather skipped because
/// their help or type conflicted with a family already merged under the same
/// name. Counts every skip on every gather, so a conflict stays alertable after
/// its one warning has rotated out of the logs. Unlabeled: the family name is in
/// that warning, and as a label it could take as many values as there are
/// families.
pub(crate) static MERGE_CONFLICTS_TOTAL: Lazy<prometheus::IntCounter> = Lazy::new(|| {
    prometheus::IntCounter::new(
        format!(
            "{}_{}",
            name_prefix::METRICS,
            prometheus_names::metrics_merge::CONFLICTS_TOTAL
        ),
        "Metric families skipped while merging a combined gather because their help or type \
         conflicted with an earlier family of the same name",
    )
    .expect("metrics merge conflicts counter")
});

/// Register [`MERGE_CONFLICTS_TOTAL`] on a DistributedRuntime's root registry,
/// so it is exported wherever that tree is: `/metrics` and OTLP.
pub(crate) fn register_merge_conflicts_counter(registry: &MetricsRegistry) {
    registry.add_metric_or_warn(
        Box::new(MERGE_CONFLICTS_TOTAL.clone()),
        prometheus_names::metrics_merge::CONFLICTS_TOTAL,
    );
}

/// True the first time a help/type conflict is reported for `name` in this
/// process, false after that.
fn first_conflict_for(name: &str) -> bool {
    static REPORTED: Lazy<Mutex<HashSet<String>>> = Lazy::new(Default::default);
    let mut reported = REPORTED.lock();
    if reported.contains(name) {
        return false;
    }
    reported.insert(name.to_owned());
    true
}

/// A family whose help or type disagrees with the family already merged under
/// its name.
#[derive(Debug)]
struct FamilyConflict {
    name: String,
    existing_type: prometheus::proto::MetricType,
    existing_help: String,
    incoming_type: prometheus::proto::MetricType,
    incoming_help: String,
}

impl<'a> FamilyMerger<'a> {
    fn new(conflicts: &'a prometheus::IntCounter) -> Self {
        Self {
            by_name: HashMap::new(),
            seen_series: HashSet::new(),
            conflicts,
        }
    }

    /// Merge every registry's families. A family whose help or type conflicts
    /// with one already merged under its name is reported and skipped, as the
    /// typed path in `metric_families_combined` does: one bad family must not
    /// take every other family off both `/metrics` and OTLP.
    fn from_gathered(
        registries: &[MetricsRegistry],
        conflicts: &'a prometheus::IntCounter,
    ) -> Self {
        let mut merger = Self::new(conflicts);
        for registry in registries {
            for family in registry.get_prometheus_registry().gather() {
                merger.add_or_skip(family);
            }
        }
        merger
    }

    /// Merge `family`, or report the conflict and skip it. Returns whether the
    /// family was merged.
    fn add_or_skip(&mut self, family: prometheus::proto::MetricFamily) -> bool {
        match self.add_family(family) {
            Ok(()) => true,
            Err(conflict) => {
                self.report_conflict(&conflict);
                false
            }
        }
    }

    fn report_conflict(&self, conflict: &FamilyConflict) {
        self.conflicts.inc();
        // A conflict persists across gathers, so warn once per name rather
        // than at the scrape and export rate; repeats are logged at debug.
        if first_conflict_for(&conflict.name) {
            tracing::warn!(
                metric_name = %conflict.name,
                existing_type = ?conflict.existing_type,
                existing_help = %conflict.existing_help,
                incoming_type = ?conflict.incoming_type,
                incoming_help = %conflict.incoming_help,
                "skipping metric family"
            );
        } else {
            tracing::debug!(
                metric_name = %conflict.name,
                existing_type = ?conflict.existing_type,
                existing_help = %conflict.existing_help,
                incoming_type = ?conflict.incoming_type,
                incoming_help = %conflict.incoming_help,
                "skipping metric family"
            );
        }
    }

    fn add_family(
        &mut self,
        mut family: prometheus::proto::MetricFamily,
    ) -> Result<(), FamilyConflict> {
        let name = family.name().to_string();

        let entry = self.by_name.entry(name.clone()).or_insert_with(|| {
            let mut out = prometheus::proto::MetricFamily::new();
            out.set_name(name.clone());
            out.set_help(family.help().to_string());
            out.set_field_type(family.get_field_type());
            out
        });

        if entry.help() != family.help() || entry.get_field_type() != family.get_field_type() {
            return Err(FamilyConflict {
                existing_type: entry.get_field_type(),
                existing_help: entry.help().to_string(),
                incoming_type: family.get_field_type(),
                incoming_help: family.take_help(),
                name,
            });
        }

        for metric in family.take_metric() {
            let mut labels: Vec<(String, String)> = metric
                .get_label()
                .iter()
                .map(|lp| (lp.name().to_string(), lp.value().to_string()))
                .collect();
            labels.sort_by(|(ka, va), (kb, vb)| (ka, va).cmp(&(kb, vb)));

            let key = format!(
                "{}|{}",
                name,
                labels
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",")
            );

            if !self.seen_series.insert(key) {
                tracing::warn!(
                    metric_name = %name,
                    labels = ?labels,
                    "Duplicate Prometheus series while merging; dropping later sample"
                );
                continue;
            }

            entry.mut_metric().push(metric);
        }

        Ok(())
    }

    fn into_sorted(self) -> Vec<prometheus::proto::MetricFamily> {
        let mut merged: Vec<prometheus::proto::MetricFamily> = self.by_name.into_values().collect();
        merged.sort_by(|a, b| a.name().cmp(b.name()));
        merged
    }
}

/// A `prometheus::Registry` attached to a [`MetricsRegistry`] tree by
/// [`MetricsRegistry::add_prometheus_registry`]. Dropping it detaches the
/// registry.
pub struct AttachedPrometheusRegistry {
    parent: std::sync::Weak<std::sync::RwLock<Vec<MetricsRegistry>>>,
    child: Arc<std::sync::RwLock<prometheus::Registry>>,
}

impl Drop for AttachedPrometheusRegistry {
    fn drop(&mut self) {
        if let Some(parent) = self.parent.upgrade() {
            // Never panic in `drop`: a poisoned lock would otherwise make every
            // later drop panic, and a panic while unwinding aborts.
            parent
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|registry| !Arc::ptr_eq(&registry.prometheus_registry, &self.child));
        }
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test_metricsregistry_units {
    use super::prometheus_names::build_component_metric_name;
    use super::*;

    #[test]
    fn test_build_component_metric_name_with_prefix() {
        // Test that build_component_metric_name correctly prepends the dynamo_component prefix
        let result = build_component_metric_name("requests");
        assert_eq!(result, "dynamo_component_requests");

        let result = build_component_metric_name("counter");
        assert_eq!(result, "dynamo_component_counter");
    }

    #[test]
    fn test_metrics_registry_entry_callbacks() {
        use crate::MetricsRegistry;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test 1: Basic callback execution with counter increments
        {
            let registry = MetricsRegistry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Add callbacks with different increment values
            for increment in [1, 10, 100] {
                let counter_clone = counter.clone();
                registry.add_update_callback(Arc::new(move || {
                    counter_clone.fetch_add(increment, Ordering::SeqCst);
                    Ok(())
                }));
            }

            // Verify counter starts at 0
            assert_eq!(counter.load(Ordering::SeqCst), 0);

            // First execution
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert!(results.iter().all(|r| r.is_ok()));
            assert_eq!(counter.load(Ordering::SeqCst), 111); // 1 + 10 + 100

            // Second execution - callbacks should be reusable
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert_eq!(counter.load(Ordering::SeqCst), 222); // 111 + 111

            // Test cloning - cloned entry shares callbacks (callbacks are Arc-wrapped)
            let cloned = registry.clone();
            assert_eq!(cloned.execute_update_callbacks().len(), 3);
            assert_eq!(counter.load(Ordering::SeqCst), 333); // 222 + 111

            // Original still has callbacks and shares the same Arc
            registry.execute_update_callbacks();
            assert_eq!(counter.load(Ordering::SeqCst), 444); // 333 + 111
        }

        // Test 2: Mixed success and error callbacks
        {
            let registry = MetricsRegistry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Successful callback
            let counter_clone = counter.clone();
            registry.add_update_callback(Arc::new(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }));

            // Error callback
            registry.add_update_callback(Arc::new(|| Err(anyhow::anyhow!("Simulated error"))));

            // Another successful callback
            let counter_clone = counter.clone();
            registry.add_update_callback(Arc::new(move || {
                counter_clone.fetch_add(10, Ordering::SeqCst);
                Ok(())
            }));

            // Execute and verify mixed results
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 3);
            assert!(results[0].is_ok());
            assert!(results[1].is_err());
            assert!(results[2].is_ok());

            // Verify error message
            assert_eq!(
                results[1].as_ref().unwrap_err().to_string(),
                "Simulated error"
            );

            // Verify successful callbacks still executed
            assert_eq!(counter.load(Ordering::SeqCst), 11); // 1 + 10

            // Execute again - errors should be consistent
            let results = registry.execute_update_callbacks();
            assert!(results[1].is_err());
            assert_eq!(counter.load(Ordering::SeqCst), 22); // 11 + 11
        }

        // Test 3: Empty registry
        {
            let registry = MetricsRegistry::new();
            let results = registry.execute_update_callbacks();
            assert_eq!(results.len(), 0);
        }
    }
}

#[cfg(test)]
mod test_create_metric_with_options {
    use super::*;
    use prometheus::core::Collector;

    /// A hierarchy level without a runtime, with a registry of its own.
    struct Level {
        name: &'static str,
        parents: Vec<Level>,
        registry: MetricsRegistry,
        connection_id: Option<u64>,
    }

    impl Level {
        fn new(name: &'static str, parents: Vec<Level>, connection_id: Option<u64>) -> Self {
            Self {
                name,
                parents,
                registry: MetricsRegistry::new(),
                connection_id,
            }
        }

        /// Component `comp` of namespace `ns-1`, on connection `0x1a`.
        fn component() -> Self {
            let root = Self::new("", Vec::new(), None);
            let namespace = Self::new("ns-1", Vec::new(), None);
            Self::new("comp", vec![root, namespace], Some(0x1a))
        }
    }

    impl MetricsHierarchy for Level {
        fn basename(&self) -> String {
            self.name.to_string()
        }

        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            self.parents
                .iter()
                .map(|parent| parent as &dyn MetricsHierarchy)
                .collect()
        }

        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }

        fn connection_id(&self) -> Option<u64> {
            self.connection_id
        }
    }

    const CUSTOM: MetricOptions = MetricOptions {
        prefix: "dynamo_custom",
        warn_on_registration_failure: false,
    };

    /// A metric's name and its constant labels, as sorted `name=value` pairs.
    fn described(metric: &impl Collector) -> (String, Vec<String>) {
        let desc = metric.desc()[0];
        let labels = desc
            .const_label_pairs
            .iter()
            .map(|pair| format!("{}={}", pair.name(), pair.value()))
            .collect();
        (desc.fq_name.clone(), labels)
    }

    /// A prefix applies only when passed explicitly, and a metric named under
    /// one carries the same hierarchy labels as one named by default.
    #[test]
    fn an_explicit_prefix_names_the_metric_and_keeps_the_hierarchy_labels() {
        let component = Level::component();
        let custom: prometheus::IntCounterVec = create_metric_with_options(
            &component,
            CUSTOM,
            "requests_total",
            "help",
            &[],
            None,
            Some(&["dp_rank"]),
        )
        .unwrap();
        let default = component
            .metrics()
            .create_intcountervec("requests_total", "help", &["dp_rank"], &[])
            .unwrap();
        // A name that merely looks prefixed is still named by default.
        let lookalike: prometheus::IntCounter =
            create_metric(&component, "dynamo_custom_total", "help", &[], None, None).unwrap();

        let (name, labels) = described(&custom);
        assert_eq!(name, "dynamo_custom_requests_total");
        assert_eq!(
            labels,
            [
                "dynamo_component=comp",
                "dynamo_namespace=ns_1",
                "worker_id=1a"
            ]
        );
        assert_eq!(
            described(&default),
            ("dynamo_component_requests_total".to_string(), labels)
        );
        assert_eq!(
            described(&lookalike).0,
            "dynamo_component_dynamo_custom_total"
        );
    }

    /// Explicit options keep every label check, and choose what a registry's
    /// refusal does: fail, or keep the metric unregistered with a warning.
    #[test]
    fn explicit_options_keep_validation_and_choose_how_a_refusal_ends() {
        let component = Level::component();
        let gauge = |options, labels: &[(&str, &str)], label_names: &[&str]| {
            create_metric_with_options::<prometheus::IntGaugeVec, _>(
                &component,
                options,
                "depth",
                "help",
                labels,
                None,
                Some(label_names),
            )
        };
        assert!(gauge(CUSTOM, &[("worker_id", "7")], &["dp_rank"]).is_err());
        assert!(gauge(CUSTOM, &[], &["dynamo_namespace"]).is_err());
        assert!(gauge(CUSTOM, &[("a", "1"), ("a", "2")], &["dp_rank"]).is_err());

        let first = gauge(CUSTOM, &[], &["dp_rank"]).unwrap();
        assert!(gauge(CUSTOM, &[], &["dp_rank"]).is_err());
        let warn = MetricOptions {
            warn_on_registration_failure: true,
            ..CUSTOM
        };
        let refused = gauge(warn, &[], &["dp_rank"]).unwrap();
        first.with_label_values(&["0"]).set(3);
        refused.with_label_values(&["0"]).set(5);

        // The registry exposes only the first; the refused one still counts.
        let exposed = component.registry.prometheus_expfmt_combined().unwrap();
        let samples: Vec<&str> = exposed
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        assert_eq!(
            samples,
            [
                r#"dynamo_custom_depth{dp_rank="0",dynamo_component="comp",dynamo_namespace="ns_1",worker_id="1a"} 3"#
            ]
        );
        assert_eq!(refused.with_label_values(&["0"]).get(), 5);
    }
}

#[cfg(test)]
mod test_metric_families_combined {
    use super::*;
    use std::sync::Arc as StdArc;

    fn gauge_family(name: &str, help: &str, value: f64) -> prometheus::proto::MetricFamily {
        let mut family = prometheus::proto::MetricFamily::new();
        family.set_name(name.to_string());
        family.set_help(help.to_string());
        family.set_field_type(prometheus::proto::MetricType::GAUGE);
        let mut metric = prometheus::proto::Metric::new();
        let mut gauge = prometheus::proto::Gauge::new();
        gauge.set_value(value);
        metric.set_gauge(gauge);
        family.mut_metric().push(metric);
        family
    }

    fn registry() -> MetricsRegistry {
        let registry = MetricsRegistry::new();
        let counter = prometheus::IntCounter::new("dynamo_native_total", "Native counter").unwrap();
        counter.inc_by(7);
        registry
            .get_prometheus_registry()
            .register(Box::new(counter))
            .unwrap();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![prom_typed::BuiltFamily {
                unit: String::new(),
                prom_type: "gauge".to_string(),
                family: gauge_family("vllm:num_requests_running", "Running requests", 4.0),
            }])
        }));
        registry
    }

    fn names(fs: &[prometheus::proto::MetricFamily]) -> Vec<&str> {
        fs.iter().map(|f| f.name()).collect()
    }

    /// Two registries in one tree declare the same family name with different
    /// types. The first declaration wins, the conflicting one is skipped, and
    /// every other family survives, on both the scrape and the export.
    #[test]
    fn conflicting_family_is_skipped_and_everything_else_survives() {
        let root = registry();
        let first =
            prometheus::IntGauge::new("dynamo_conflict", "Declared first, as a gauge").unwrap();
        first.set(1);
        root.add_metric(Box::new(first)).unwrap();

        let child = MetricsRegistry::new();
        let conflicting =
            prometheus::IntCounter::new("dynamo_conflict", "Declared again, as a counter").unwrap();
        conflicting.inc_by(5);
        child.add_metric(Box::new(conflicting)).unwrap();
        child
            .add_metric(Box::new(
                prometheus::IntCounter::new("dynamo_child_total", "Child counter").unwrap(),
            ))
            .unwrap();
        root.add_child_registry(&child);

        // Other tests can count conflicts concurrently, so only a lower bound
        // holds on the process-wide counter.
        let conflicts_before = MERGE_CONFLICTS_TOTAL.get();
        let exported = root.metric_families_combined();
        assert_eq!(
            names(&exported.families),
            [
                "dynamo_child_total",
                "dynamo_conflict",
                "dynamo_native_total",
                "vllm:num_requests_running"
            ]
        );
        let conflict = exported
            .families
            .iter()
            .find(|family| family.name() == "dynamo_conflict")
            .unwrap();
        assert_eq!(
            conflict.get_field_type(),
            prometheus::proto::MetricType::GAUGE
        );
        assert_eq!(conflict.get_metric().len(), 1);
        assert_eq!(conflict.get_metric()[0].get_gauge().value(), 1.0);

        let text = root
            .prometheus_expfmt_combined()
            .expect("one conflict must not fail the scrape");
        assert!(
            MERGE_CONFLICTS_TOTAL.get() >= conflicts_before + 2,
            "the export and the scrape must each count the conflict"
        );
        let declared: Vec<&str> = text
            .lines()
            .filter_map(|line| line.strip_prefix("# TYPE "))
            .collect();
        assert_eq!(
            declared,
            [
                "dynamo_child_total counter",
                "dynamo_conflict gauge",
                "dynamo_native_total counter"
            ],
            "{text}"
        );
        assert!(
            text.lines().any(|line| line == "dynamo_conflict 1"),
            "{text}"
        );
        assert!(
            !text.lines().any(|line| line == "dynamo_conflict 5"),
            "{text}"
        );
    }

    /// A conflict persists across gathers. Every gather counts it, so it stays
    /// alertable, and the first one records the name so later reports drop to
    /// debug.
    #[test]
    fn every_gather_counts_a_conflict() {
        // The set of reported names is process-wide: use a name no other test does.
        const NAME: &str = "dynamo_test_conflict_counted_every_gather";
        let first = MetricsRegistry::new();
        first
            .add_metric(Box::new(
                prometheus::IntGauge::new(NAME, "Declared first, as a gauge").unwrap(),
            ))
            .unwrap();
        let second = MetricsRegistry::new();
        second
            .add_metric(Box::new(
                prometheus::IntCounter::new(NAME, "Declared again, as a counter").unwrap(),
            ))
            .unwrap();
        let registries = [first, second];
        // A counter of its own, so conflicts in concurrent tests don't move it.
        let conflicts = prometheus::IntCounter::new("test_conflicts_total", "test").unwrap();

        for _ in 0..2 {
            let merged = FamilyMerger::from_gathered(&registries, &conflicts).into_sorted();
            assert_eq!(names(&merged), [NAME]);
            assert_eq!(
                merged[0].get_field_type(),
                prometheus::proto::MetricType::GAUGE
            );
        }
        assert_eq!(conflicts.get(), 2);
        assert!(
            !first_conflict_for(NAME),
            "the first gather must record the conflicting name"
        );
    }

    /// A plain `prometheus::Registry` attached with `add_prometheus_registry`
    /// reaches both the scrape and the export, including collectors registered
    /// after it was attached, and leaves both when the handle is dropped.
    #[test]
    fn attached_prometheus_registry_is_scraped_exported_and_detached() {
        let root = registry();
        let own = prometheus::Registry::new();
        own.register(Box::new(
            prometheus::IntCounter::new("dynamo_frontend_early_total", "Early").unwrap(),
        ))
        .unwrap();
        let attached = root.add_prometheus_registry(own.clone());
        own.register(Box::new(
            prometheus::IntGauge::new("dynamo_frontend_late", "Late").unwrap(),
        ))
        .unwrap();

        let ours = ["dynamo_frontend_early_total", "dynamo_frontend_late"];
        let exported = root.metric_families_combined().families;
        let text = root.prometheus_expfmt_combined().unwrap();
        for name in ours {
            assert!(names(&exported).contains(&name), "{name} not exported");
            assert_eq!(
                text.matches(&format!("# TYPE {name} ")).count(),
                1,
                "{name} not declared exactly once:\n{text}"
            );
        }

        drop(attached);
        let exported = root.metric_families_combined().families;
        let text = root.prometheus_expfmt_combined().unwrap();
        for name in ours {
            assert!(!names(&exported).contains(&name), "{name} still exported");
            assert!(!text.contains(name), "{name} still scraped:\n{text}");
        }
        // The rest of the tree is untouched.
        assert!(names(&exported).contains(&"dynamo_native_total"));
    }

    #[test]
    fn typed_and_native_families_share_one_collection() {
        let collected = registry().metric_families_combined();
        assert!(names(&collected.families).contains(&"vllm:num_requests_running"));
        assert!(names(&collected.families).contains(&"dynamo_native_total"));
    }

    /// `units` and `prom_types` are keyed by family name, so a family the
    /// merger rejects must not leave its metadata behind: it would then be
    /// applied to whichever family did survive under that name. Here the
    /// rejected entry claims `info`, which would flip the surviving gauge to a
    /// non-monotonic Sum on the OTLP side.
    /// The merger resolves a conflict first-writer-wins: the first family
    /// fixes help and type and later ones must agree. The metadata maps must
    /// resolve the same way. A later family can pass that check and still have
    /// every sample dropped as a duplicate series, so a last-writer-wins unit
    /// would label the surviving data with a unit from a source that
    /// contributed none of it.
    #[test]
    fn metadata_from_a_source_that_contributed_no_samples_does_not_win() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![
                prom_typed::BuiltFamily {
                    unit: "bytes".to_string(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:probe", "Same help", 1.0),
                },
                // Same name, help and type, so the merger accepts it -- but the
                // labels match too, so every sample is dropped as a duplicate.
                prom_typed::BuiltFamily {
                    unit: "seconds".to_string(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:probe", "Same help", 2.0),
                },
            ])
        }));

        let collected = registry.metric_families_combined();

        let family = collected
            .families
            .iter()
            .find(|f| f.name() == "vllm:probe")
            .expect("family");
        assert_eq!(
            family.get_metric()[0].get_gauge().value(),
            1.0,
            "the surviving sample should be the first source's"
        );
        assert_eq!(
            collected.units.get("vllm:probe").map(String::as_str),
            Some("bytes"),
            "unit came from a source whose samples were all dropped"
        );
    }

    #[test]
    fn rejected_family_leaves_no_metadata_behind() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| {
            Ok(vec![
                prom_typed::BuiltFamily {
                    unit: String::new(),
                    prom_type: "gauge".to_string(),
                    family: gauge_family("vllm:cache_usage", "Cache usage", 0.5),
                },
                // Same name, different help -> rejected by the merger.
                prom_typed::BuiltFamily {
                    unit: "seconds".to_string(),
                    prom_type: "info".to_string(),
                    family: gauge_family("vllm:cache_usage", "A different help", 1.0),
                },
            ])
        }));

        let collected = registry.metric_families_combined();

        assert_eq!(
            collected
                .prom_types
                .get("vllm:cache_usage")
                .map(String::as_str),
            Some("gauge"),
            "the rejected family's type overwrote the accepted family's"
        );
        assert!(
            !collected.units.contains_key("vllm:cache_usage"),
            "the rejected family's unit was published for a family that has none"
        );
    }

    /// The two surfaces are fed independently: typed callbacks reach OTLP,
    /// exposition-text callbacks reach `/metrics`. A registry that registers
    /// only one appears on only that surface -- which is why the Python
    /// helpers always register both.
    #[test]
    fn typed_callbacks_feed_otlp_and_not_the_scrape() {
        let registry = registry();

        let text = registry.prometheus_expfmt_combined().expect("text");
        assert!(
            text.contains("dynamo_native_total"),
            "native metrics must still render on /metrics"
        );
        assert!(
            !text.contains("vllm:num_requests_running"),
            "typed callbacks feed OTLP only; /metrics is fed by expfmt callbacks"
        );

        registry.add_expfmt_callback(StdArc::new(|| {
            Ok("# HELP vllm:num_requests_running Running requests\n\
                # TYPE vllm:num_requests_running gauge\n\
                vllm:num_requests_running 4\n"
                .to_string())
        }));
        let text = registry.prometheus_expfmt_combined().expect("text");
        assert!(
            text.contains("vllm:num_requests_running"),
            "engine exposition text must be appended verbatim"
        );
    }

    /// Collection must not hold the callback lock while invoking callbacks.
    ///
    /// Registration arrives from Python holding the GIL and blocks on
    /// `write()`; a collection that held the read lock while calling into
    /// Python would deadlock the two against each other. Registering from
    /// inside a callback reproduces that ordering deterministically, without
    /// depending on a thread interleaving.
    ///
    /// Run on a worker thread with a deadline so a regression fails the test
    /// rather than hanging the suite forever.
    #[test]
    fn collection_does_not_hold_the_callback_lock() {
        for (name, collect) in [
            (
                "typed",
                Box::new(|r: &MetricsRegistry| {
                    r.execute_typed_callbacks();
                }) as Box<dyn Fn(&MetricsRegistry) + Send>,
            ),
            (
                "expfmt",
                Box::new(|r: &MetricsRegistry| {
                    r.execute_expfmt_callbacks();
                }),
            ),
        ] {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let registry = MetricsRegistry::new();
                let reentrant = registry.clone();
                registry.add_typed_callback(StdArc::new(move || {
                    reentrant.add_typed_callback(StdArc::new(|| Ok(Vec::new())));
                    Ok(Vec::new())
                }));
                let reentrant = registry.clone();
                registry.add_expfmt_callback(StdArc::new(move || {
                    reentrant.add_expfmt_callback(StdArc::new(|| Ok(String::new())));
                    Ok(String::new())
                }));
                collect(&registry);
                let _ = tx.send(());
            });

            assert!(
                rx.recv_timeout(std::time::Duration::from_secs(10)).is_ok(),
                "{name} collection deadlocked: the callback lock was held while a \
                 callback registered another"
            );
        }
    }

    /// A failing engine callback is skipped, never fatal: one broken engine
    /// must not empty the collection for every other.
    #[test]
    fn failing_typed_callback_is_skipped() {
        let registry = MetricsRegistry::new();
        registry.add_typed_callback(StdArc::new(|| anyhow::bail!("engine exploded")));
        assert!(registry.metric_families_combined().families.is_empty());
    }
}
