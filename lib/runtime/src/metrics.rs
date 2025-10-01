// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics registry trait and implementation for Prometheus metrics
//!
//! This module provides a trait-based interface for creating and managing Prometheus metrics
//! with automatic label injection and hierarchical naming support.

pub mod prometheus_names;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use crate::component::ComponentBuilder;
use anyhow;
use once_cell::sync::Lazy;
use regex::Regex;
use std::any::Any;
use std::collections::HashMap;

// Import commonly used items to avoid verbose prefixes
use prometheus_names::{
    COMPONENT_NATS_METRICS, DRT_NATS_METRICS, build_component_metric_name, labels, name_prefix,
    nats_client, nats_service, sanitize_prometheus_label, sanitize_prometheus_name, work_handler,
};

// Pipeline imports for endpoint creation
use crate::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
    network::Ingress,
};
use crate::protocols::annotated::Annotated;
use crate::stream;
use crate::stream::StreamExt;

// If set to true, then metrics will be labeled with the namespace, component, and endpoint labels.
// These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.
pub const USE_AUTO_LABELS: bool = true;

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

/// Private helper function to create metrics - not accessible to trait implementors
fn create_metric<T: PrometheusMetric, R: MetricsRegistry + ?Sized>(
    registry: &R,
    metric_name: &str,
    metric_desc: &str,
    labels: &[(&str, &str)],
    buckets: Option<Vec<f64>>,
    const_labels: Option<&[&str]>,
) -> anyhow::Result<T> {
    // Validate that user-provided labels don't have duplicate keys
    validate_no_duplicate_label_keys(labels)?;
    // Note: stored labels functionality has been removed

    let basename = registry.basename();
    let parent_hierarchy = registry.parent_hierarchy();

    // Build hierarchy: parent_hierarchy + [basename]
    let hierarchy = [parent_hierarchy.clone(), vec![basename.clone()]].concat();

    let metric_name = build_component_metric_name(metric_name);

    // Build updated_labels: auto-labels first, then `labels` + stored labels
    let mut updated_labels: Vec<(String, String)> = Vec::new();

    if USE_AUTO_LABELS {
        // Validate that user-provided labels don't conflict with auto-generated labels
        for (key, _) in labels {
            if *key == labels::NAMESPACE || *key == labels::COMPONENT || *key == labels::ENDPOINT {
                return Err(anyhow::anyhow!(
                    "Label '{}' is automatically added by auto_label feature and cannot be manually set",
                    key
                ));
            }
        }

        // Add auto-generated labels with sanitized values
        if hierarchy.len() > 1 {
            let namespace = &hierarchy[1];
            if !namespace.is_empty() {
                let valid_namespace = sanitize_prometheus_label(namespace)?;
                if !valid_namespace.is_empty() {
                    updated_labels.push((labels::NAMESPACE.to_string(), valid_namespace));
                }
            }
        }
        if hierarchy.len() > 2 {
            let component = &hierarchy[2];
            if !component.is_empty() {
                let valid_component = sanitize_prometheus_label(component)?;
                if !valid_component.is_empty() {
                    updated_labels.push((labels::COMPONENT.to_string(), valid_component));
                }
            }
        }
        if hierarchy.len() > 3 {
            let endpoint = &hierarchy[3];
            if !endpoint.is_empty() {
                let valid_endpoint = sanitize_prometheus_label(endpoint)?;
                if !valid_endpoint.is_empty() {
                    updated_labels.push((labels::ENDPOINT.to_string(), valid_endpoint));
                }
            }
        }
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
        == std::any::TypeId::of::<prometheus::Histogram>()
    {
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
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::CounterVec>() {
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

    // Iterate over the DRT's registry and register this metric across all hierarchical levels.
    // The accumulated hierarchy is structured as: ["", "testnamespace", "testnamespace_testcomponent", "testnamespace_testcomponent_testendpoint"]
    // This accumulation is essential to differentiate between the names of children and grandchildren.
    // Build accumulated hierarchy and register metrics in a single loop
    // current_prefix accumulates the hierarchical path as we iterate through hierarchy
    // For example, if hierarchy = ["", "testnamespace", "testcomponent"], then:
    // - Iteration 1: current_prefix = "" (empty string from DRT)
    // - Iteration 2: current_prefix = "testnamespace"
    // - Iteration 3: current_prefix = "testnamespace_testcomponent"
    let mut current_hierarchy = String::new();
    for name in &hierarchy {
        if !current_hierarchy.is_empty() && !name.is_empty() {
            current_hierarchy.push('_');
        }
        current_hierarchy.push_str(name);

        // Register metric at this hierarchical level using the new helper function
        let collector: Box<dyn prometheus::core::Collector> = Box::new(prometheus_metric.clone());
        registry
            .drt()
            .add_prometheus_metric(&current_hierarchy, collector)?;
    }

    Ok(prometheus_metric)
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
use crate::traits::DistributedRuntimeProvider;

pub trait MetricsRegistry: Send + Sync + DistributedRuntimeProvider {
    // Get the name of this registry (without any hierarchy prefix)
    fn basename(&self) -> String;

    /// Retrieve the complete hierarchy and basename for this registry. Currently, the hierarchy for drt is an empty string,
    /// so we must account for the leading underscore. The existing code remains unchanged to accommodate any future
    /// scenarios where drt's prefix might be assigned a value.
    fn hierarchy(&self) -> String {
        [self.parent_hierarchy(), vec![self.basename()]]
            .concat()
            .join("_")
            .trim_start_matches('_')
            .to_string()
    }

    // Get the parent hierarchy for this registry (just the base names, NOT the flattened hierarchy key)
    fn parent_hierarchy(&self) -> Vec<String>;

    // TODO: Add support for additional Prometheus metric types:
    // - Counter: ✅ IMPLEMENTED - create_counter()
    // - CounterVec: ✅ IMPLEMENTED - create_countervec()
    // - Gauge: ✅ IMPLEMENTED - create_gauge()
    // - GaugeHistogram: create_gauge_histogram() - for gauge histograms
    // - Histogram: ✅ IMPLEMENTED - create_histogram()
    // - HistogramVec with custom buckets: create_histogram_with_buckets()
    // - Info: create_info() - for info metrics with labels
    // - IntCounter: ✅ IMPLEMENTED - create_intcounter()
    // - IntCounterVec: ✅ IMPLEMENTED - create_intcountervec()
    // - IntGauge: ✅ IMPLEMENTED - create_intgauge()
    // - IntGaugeVec: ✅ IMPLEMENTED - create_intgaugevec()
    // - Stateset: create_stateset() - for state-based metrics
    // - Summary: create_summary() - for quantiles and sum/count metrics
    // - SummaryVec: create_summary_vec() - for labeled summaries
    // - Untyped: create_untyped() - for untyped metrics

    /// Create a Counter metric
    fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Counter> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create a CounterVec metric with label names (for dynamic labels)
    fn create_countervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::CounterVec> {
        create_metric(
            self,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create a Gauge metric
    fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Gauge> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create a Histogram metric with custom buckets
    fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<prometheus::Histogram> {
        create_metric(self, name, description, labels, buckets, None)
    }

    /// Create an IntCounter metric
    fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounter> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create an IntCounterVec metric with label names (for dynamic labels)
    fn create_intcountervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntCounterVec> {
        create_metric(
            self,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Create an IntGauge metric
    fn create_intgauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGauge> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create an IntGaugeVec metric with label names (for dynamic labels)
    fn create_intgaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::IntGaugeVec> {
        create_metric(
            self,
            name,
            description,
            const_label_values,
            None,
            Some(const_labels),
        )
    }

    /// Get metrics in Prometheus text format
    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        // Execute callbacks first to ensure any new metrics are added to the registry
        let callback_results = self.drt().execute_metrics_callbacks(&self.hierarchy());

        // Log any callback errors but continue
        for result in callback_results {
            if let Err(e) = result {
                tracing::error!("Error executing metrics callback: {}", e);
            }
        }

        // Get the Prometheus registry for this hierarchy
        let prometheus_registry = {
            let mut registry_entry = self.drt().hierarchy_to_metricsregistry.write().unwrap();
            registry_entry
                .entry(self.hierarchy())
                .or_default()
                .prometheus_registry
                .clone()
        };
        let metric_families = prometheus_registry.gather();
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

#[cfg(test)]
mod test_helpers {
    use super::prometheus_names::name_prefix;
    use super::prometheus_names::{nats_client, nats_service};
    use super::*;

    /// Base function to filter Prometheus output lines based on a predicate.
    /// Returns lines that match the predicate, converted to String.
    fn filter_prometheus_lines<F>(input: &str, mut predicate: F) -> Vec<String>
    where
        F: FnMut(&str) -> bool,
    {
        input
            .lines()
            .filter(|line| predicate(line))
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
    }

    /// Filters out all NATS metrics from Prometheus output for test comparisons.
    pub fn remove_nats_lines(input: &str) -> Vec<String> {
        filter_prometheus_lines(input, |line| {
            !line.contains(&format!(
                "{}_{}",
                name_prefix::COMPONENT,
                nats_client::PREFIX
            )) && !line.contains(&format!(
                "{}_{}",
                name_prefix::COMPONENT,
                nats_service::PREFIX
            )) && !line.trim().is_empty()
        })
    }

    /// Filters to only include NATS metrics from Prometheus output for test comparisons.
    pub fn extract_nats_lines(input: &str) -> Vec<String> {
        filter_prometheus_lines(input, |line| {
            line.contains(&format!(
                "{}_{}",
                name_prefix::COMPONENT,
                nats_client::PREFIX
            )) || line.contains(&format!(
                "{}_{}",
                name_prefix::COMPONENT,
                nats_service::PREFIX
            ))
        })
    }

    /// Extracts all component metrics (excluding help text and type definitions).
    /// Returns only the actual metric lines with values.
    pub fn extract_metrics(input: &str) -> Vec<String> {
        filter_prometheus_lines(input, |line| {
            line.starts_with(&format!("{}_", name_prefix::COMPONENT))
                && !line.starts_with("#")
                && !line.trim().is_empty()
        })
    }

    /// Parses a Prometheus metric line and extracts the name, labels, and value.
    /// Used instead of fetching metrics directly to test end-to-end results, not intermediate state.
    ///
    /// # Example
    /// ```
    /// let line = "http_requests_total{method=\"GET\"} 1234";
    /// let (name, labels, value) = parse_prometheus_metric(line).unwrap();
    /// assert_eq!(name, "http_requests_total");
    /// assert_eq!(labels.get("method"), Some(&"GET".to_string()));
    /// assert_eq!(value, 1234.0);
    /// ```
    pub fn parse_prometheus_metric(
        line: &str,
    ) -> Option<(String, std::collections::HashMap<String, String>, f64)> {
        if line.trim().is_empty() || line.starts_with('#') {
            return None;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return None;
        }

        let metric_part = parts[0];
        let value: f64 = parts[1].parse().ok()?;

        let (name, labels) = if metric_part.contains('{') {
            let brace_start = metric_part.find('{').unwrap();
            let brace_end = metric_part.rfind('}').unwrap_or(metric_part.len());
            let name = &metric_part[..brace_start];
            let labels_str = &metric_part[brace_start + 1..brace_end];

            let mut labels = std::collections::HashMap::new();
            for pair in labels_str.split(',') {
                if let Some((k, v)) = pair.split_once('=') {
                    let v = v.trim_matches('"');
                    labels.insert(k.trim().to_string(), v.to_string());
                }
            }
            (name.to_string(), labels)
        } else {
            (metric_part.to_string(), std::collections::HashMap::new())
        };

        Some((name, labels, value))
    }
}

#[cfg(test)]
mod test_metricsregistry_units {
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
    fn test_parse_prometheus_metric() {
        use super::test_helpers::parse_prometheus_metric;
        use std::collections::HashMap;

        // Test parsing a metric with labels
        let line = "http_requests_total{method=\"GET\",status=\"200\"} 1234";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "http_requests_total");

        let mut expected_labels = HashMap::new();
        expected_labels.insert("method".to_string(), "GET".to_string());
        expected_labels.insert("status".to_string(), "200".to_string());
        assert_eq!(labels, expected_labels);

        assert_eq!(value, 1234.0);

        // Test parsing a metric without labels
        let line = "cpu_usage 98.5";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "cpu_usage");
        assert!(labels.is_empty());
        assert_eq!(value, 98.5);

        // Test parsing a metric with float value
        let line = "response_time{service=\"api\"} 0.123";
        let parsed = parse_prometheus_metric(line);
        assert!(parsed.is_some());

        let (name, labels, value) = parsed.unwrap();
        assert_eq!(name, "response_time");

        let mut expected_labels = HashMap::new();
        expected_labels.insert("service".to_string(), "api".to_string());
        assert_eq!(labels, expected_labels);

        assert_eq!(value, 0.123);

        // Test parsing invalid lines
        assert!(parse_prometheus_metric("").is_none()); // Empty line
        assert!(parse_prometheus_metric("# HELP metric description").is_none()); // Help text
        assert!(parse_prometheus_metric("# TYPE metric counter").is_none()); // Type definition
        assert!(parse_prometheus_metric("metric_name").is_none()); // No value

        println!("✓ Prometheus metric parsing works correctly!");
    }

    #[test]
    fn test_metrics_registry_entry_callbacks() {
        use crate::MetricsRegistryEntry;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test 1: Basic callback execution with counter increments
        {
            let mut entry = MetricsRegistryEntry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Add callbacks with different increment values
            for increment in [1, 10, 100] {
                let counter_clone = counter.clone();
                entry.add_callback(Arc::new(move || {
                    counter_clone.fetch_add(increment, Ordering::SeqCst);
                    Ok(())
                }));
            }

            // Verify counter starts at 0
            assert_eq!(counter.load(Ordering::SeqCst), 0);

            // First execution
            let results = entry.execute_callbacks();
            assert_eq!(results.len(), 3);
            assert!(results.iter().all(|r| r.is_ok()));
            assert_eq!(counter.load(Ordering::SeqCst), 111); // 1 + 10 + 100

            // Second execution - callbacks should be reusable
            let results = entry.execute_callbacks();
            assert_eq!(results.len(), 3);
            assert_eq!(counter.load(Ordering::SeqCst), 222); // 111 + 111

            // Test cloning - cloned entry should have no callbacks
            let cloned = entry.clone();
            assert_eq!(cloned.execute_callbacks().len(), 0);
            assert_eq!(counter.load(Ordering::SeqCst), 222); // No change

            // Original still has callbacks
            entry.execute_callbacks();
            assert_eq!(counter.load(Ordering::SeqCst), 333); // 222 + 111
        }

        // Test 2: Mixed success and error callbacks
        {
            let mut entry = MetricsRegistryEntry::new();
            let counter = Arc::new(AtomicUsize::new(0));

            // Successful callback
            let counter_clone = counter.clone();
            entry.add_callback(Arc::new(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }));

            // Error callback
            entry.add_callback(Arc::new(|| Err(anyhow::anyhow!("Simulated error"))));

            // Another successful callback
            let counter_clone = counter.clone();
            entry.add_callback(Arc::new(move || {
                counter_clone.fetch_add(10, Ordering::SeqCst);
                Ok(())
            }));

            // Execute and verify mixed results
            let results = entry.execute_callbacks();
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
            let results = entry.execute_callbacks();
            assert!(results[1].is_err());
            assert_eq!(counter.load(Ordering::SeqCst), 22); // 11 + 11
        }

        // Test 3: Empty registry
        {
            let entry = MetricsRegistryEntry::new();
            let results = entry.execute_callbacks();
            assert_eq!(results.len(), 0);
        }
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod test_metricsregistry_prefixes {
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use prometheus::core::Collector;

    #[tokio::test]
    async fn test_hierarchical_prefixes_and_parent_hierarchies() {
        let drt = create_test_drt_async().await;

        const DRT_NAME: &str = "";
        const NAMESPACE_NAME: &str = "ns901";
        const COMPONENT_NAME: &str = "comp901";
        const ENDPOINT_NAME: &str = "ep901";
        let namespace = drt.namespace(NAMESPACE_NAME).unwrap();
        let component = namespace.component(COMPONENT_NAME).unwrap();
        let endpoint = component.endpoint(ENDPOINT_NAME);

        // DRT
        assert_eq!(drt.basename(), DRT_NAME);
        assert_eq!(drt.parent_hierarchy(), Vec::<String>::new());
        assert_eq!(drt.hierarchy(), DRT_NAME);

        // Namespace
        assert_eq!(namespace.basename(), NAMESPACE_NAME);
        assert_eq!(namespace.parent_hierarchy(), vec!["".to_string()]);
        assert_eq!(namespace.hierarchy(), NAMESPACE_NAME);

        // Component
        assert_eq!(component.basename(), COMPONENT_NAME);
        assert_eq!(
            component.parent_hierarchy(),
            vec!["".to_string(), NAMESPACE_NAME.to_string()]
        );
        assert_eq!(
            component.hierarchy(),
            format!("{}_{}", NAMESPACE_NAME, COMPONENT_NAME)
        );

        // Endpoint
        assert_eq!(endpoint.basename(), ENDPOINT_NAME);
        assert_eq!(
            endpoint.parent_hierarchy(),
            vec![
                "".to_string(),
                NAMESPACE_NAME.to_string(),
                COMPONENT_NAME.to_string(),
            ]
        );
        assert_eq!(
            endpoint.hierarchy(),
            format!("{}_{}_{}", NAMESPACE_NAME, COMPONENT_NAME, ENDPOINT_NAME)
        );

        // Relationships
        assert!(namespace.parent_hierarchy().contains(&drt.basename()));
        assert!(component.parent_hierarchy().contains(&namespace.basename()));
        assert!(endpoint.parent_hierarchy().contains(&component.basename()));

        // Depth
        assert_eq!(drt.parent_hierarchy().len(), 0);
        assert_eq!(namespace.parent_hierarchy().len(), 1);
        assert_eq!(component.parent_hierarchy().len(), 2);
        assert_eq!(endpoint.parent_hierarchy().len(), 3);

        // Invalid namespace behavior - sanitizes to "_123" and succeeds
        // @ryanolson intended to enable validation (see TODO comment in component.rs) but didn't turn it on,
        // so invalid characters are sanitized in MetricsRegistry rather than rejected.
        let invalid_namespace = drt.namespace("@@123").unwrap();
        let result = invalid_namespace.create_counter("test_counter", "A test counter", &[]);
        assert!(result.is_ok());
        if let Ok(counter) = &result {
            // Verify the namespace was sanitized to "_123" in the label
            let desc = counter.desc();
            let namespace_label = desc[0]
                .const_label_pairs
                .iter()
                .find(|l| l.name() == "dynamo_namespace")
                .expect("Should have dynamo_namespace label");
            assert_eq!(namespace_label.value(), "_123");
        }

        // Valid namespace works
        let valid_namespace = drt.namespace("ns567").unwrap();
        assert!(
            valid_namespace
                .create_counter("test_counter", "A test counter", &[])
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_recursive_namespace() {
        // Create a distributed runtime for testing
        let drt = create_test_drt_async().await;

        // Create a deeply chained namespace: ns1.ns2.ns3
        let ns1 = drt.namespace("ns1").unwrap();
        let ns2 = ns1.namespace("ns2").unwrap();
        let ns3 = ns2.namespace("ns3").unwrap();

        // Create a component in the deepest namespace
        let component = ns3.component("test-component").unwrap();

        // Verify the hierarchy structure
        assert_eq!(ns1.basename(), "ns1");
        assert_eq!(ns1.parent_hierarchy(), vec!("".to_string()));
        assert_eq!(ns1.hierarchy(), "ns1");

        assert_eq!(ns2.basename(), "ns2");
        assert_eq!(
            ns2.parent_hierarchy(),
            vec!["".to_string(), "ns1".to_string()]
        );
        assert_eq!(ns2.hierarchy(), "ns1_ns2");

        assert_eq!(ns3.basename(), "ns3");
        assert_eq!(
            ns3.parent_hierarchy(),
            vec!["".to_string(), "ns1".to_string(), "ns2".to_string()]
        );
        assert_eq!(ns3.hierarchy(), "ns1_ns2_ns3");

        assert_eq!(component.basename(), "test-component");
        assert_eq!(
            component.parent_hierarchy(),
            vec![
                "".to_string(),
                "ns1".to_string(),
                "ns2".to_string(),
                "ns3".to_string()
            ]
        );
        assert_eq!(component.hierarchy(), "ns1_ns2_ns3_test-component");

        println!("✓ Chained namespace test passed - all prefixes correct");
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod test_metricsregistry_prometheus_fmt_outputs {
    use super::prometheus_names::name_prefix;
    use super::prometheus_names::{COMPONENT_NATS_METRICS, DRT_NATS_METRICS};
    use super::prometheus_names::{nats_client, nats_service};
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use prometheus::Counter;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_prometheusfactory_using_metrics_registry_trait() {
        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt_async().await;

        // Use a simple constant namespace name
        let namespace_name = "ns345";

        let namespace = drt.namespace(namespace_name).unwrap();
        let component = namespace.component("comp345").unwrap();
        let endpoint = component.endpoint("ep345");

        // Test Counter creation
        let counter = endpoint
            .create_counter("testcounter", "A test counter", &[])
            .unwrap();
        counter.inc_by(123.456789);
        let epsilon = 0.01;
        assert!((counter.get() - 123.456789).abs() < epsilon);

        let endpoint_output_raw = endpoint.prometheus_metrics_fmt().unwrap();
        println!("Endpoint output:");
        println!("{}", endpoint_output_raw);

        // Filter out NATS service metrics for test comparison
        let endpoint_output =
            super::test_helpers::remove_nats_lines(&endpoint_output_raw).join("\n");

        let expected_endpoint_output = r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789"#.to_string();

        assert_eq!(
            endpoint_output, expected_endpoint_output,
            "\n=== ENDPOINT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_endpoint_output, endpoint_output
        );

        // Test Gauge creation
        let gauge = component
            .create_gauge("testgauge", "A test gauge", &[])
            .unwrap();
        gauge.set(50000.0);
        assert_eq!(gauge.get(), 50000.0);

        // Test Prometheus format output for Component (gauge + histogram)
        let component_output_raw = component.prometheus_metrics_fmt().unwrap();
        println!("Component output:");
        println!("{}", component_output_raw);

        // Filter out NATS service metrics for test comparison
        let component_output =
            super::test_helpers::remove_nats_lines(&component_output_raw).join("\n");

        let expected_component_output = r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000"#.to_string();

        assert_eq!(
            component_output, expected_component_output,
            "\n=== COMPONENT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_component_output, component_output
        );

        let intcounter = namespace
            .create_intcounter("testintcounter", "A test int counter", &[])
            .unwrap();
        intcounter.inc_by(12345);
        assert_eq!(intcounter.get(), 12345);

        // Test Prometheus format output for Namespace (int_counter + gauge + histogram)
        let namespace_output_raw = namespace.prometheus_metrics_fmt().unwrap();
        println!("Namespace output:");
        println!("{}", namespace_output_raw);

        // Filter out NATS service metrics for test comparison
        let namespace_output =
            super::test_helpers::remove_nats_lines(&namespace_output_raw).join("\n");

        let expected_namespace_output = r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000
# HELP dynamo_component_testintcounter A test int counter
# TYPE dynamo_component_testintcounter counter
dynamo_component_testintcounter{dynamo_namespace="ns345"} 12345"#.to_string();

        assert_eq!(
            namespace_output, expected_namespace_output,
            "\n=== NAMESPACE COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_namespace_output, namespace_output
        );

        // Test IntGauge creation
        let intgauge = namespace
            .create_intgauge("testintgauge", "A test int gauge", &[])
            .unwrap();
        intgauge.set(42);
        assert_eq!(intgauge.get(), 42);

        // Test IntGaugeVec creation
        let intgaugevec = namespace
            .create_intgaugevec(
                "testintgaugevec",
                "A test int gauge vector",
                &["instance", "status"],
                &[("service", "api")],
            )
            .unwrap();
        intgaugevec
            .with_label_values(&["server1", "active"])
            .set(10);
        intgaugevec
            .with_label_values(&["server2", "inactive"])
            .set(0);

        // Test CounterVec creation
        let countervec = endpoint
            .create_countervec(
                "testcountervec",
                "A test counter vector",
                &["method", "status"],
                &[("service", "api")],
            )
            .unwrap();
        countervec.with_label_values(&["GET", "200"]).inc_by(10.0);
        countervec.with_label_values(&["POST", "201"]).inc_by(5.0);

        // Test Histogram creation
        let histogram = component
            .create_histogram("testhistogram", "A test histogram", &[], None)
            .unwrap();
        histogram.observe(1.0);
        histogram.observe(2.5);
        histogram.observe(4.0);

        // Test Prometheus format output for DRT (all metrics combined)
        let drt_output_raw = drt.prometheus_metrics_fmt().unwrap();
        println!("DRT output:");
        println!("{}", drt_output_raw);

        // Filter out all NATS metrics for comparison
        let filtered_drt_output =
            super::test_helpers::remove_nats_lines(&drt_output_raw).join("\n");

        let expected_drt_output = r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testcountervec A test counter vector
# TYPE dynamo_component_testcountervec counter
dynamo_component_testcountervec{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345",method="GET",service="api",status="200"} 10
dynamo_component_testcountervec{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345",method="POST",service="api",status="201"} 5
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000
# HELP dynamo_component_testhistogram A test histogram
# TYPE dynamo_component_testhistogram histogram
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.005"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.01"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.025"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.05"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.1"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.25"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.5"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="1"} 1
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="2.5"} 2
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="5"} 3
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="10"} 3
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="+Inf"} 3
dynamo_component_testhistogram_sum{dynamo_component="comp345",dynamo_namespace="ns345"} 7.5
dynamo_component_testhistogram_count{dynamo_component="comp345",dynamo_namespace="ns345"} 3
# HELP dynamo_component_testintcounter A test int counter
# TYPE dynamo_component_testintcounter counter
dynamo_component_testintcounter{dynamo_namespace="ns345"} 12345
# HELP dynamo_component_testintgauge A test int gauge
# TYPE dynamo_component_testintgauge gauge
dynamo_component_testintgauge{dynamo_namespace="ns345"} 42
# HELP dynamo_component_testintgaugevec A test int gauge vector
# TYPE dynamo_component_testintgaugevec gauge
dynamo_component_testintgaugevec{dynamo_namespace="ns345",instance="server1",service="api",status="active"} 10
dynamo_component_testintgaugevec{dynamo_namespace="ns345",instance="server2",service="api",status="inactive"} 0
# HELP dynamo_component_uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE dynamo_component_uptime_seconds gauge
dynamo_component_uptime_seconds 0"#.to_string();

        assert_eq!(
            filtered_drt_output, expected_drt_output,
            "\n=== DRT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual (filtered):\n{}\n\
             ==============================",
            expected_drt_output, filtered_drt_output
        );

        println!("✓ All Prometheus format outputs verified successfully!");
    }

    #[test]
    fn test_refactored_filter_functions() {
        // Test data with mixed content
        let test_input = r#"# HELP dynamo_component_requests Total requests
# TYPE dynamo_component_requests counter
dynamo_component_requests 42
# HELP dynamo_component_nats_client_connection_state Connection state
# TYPE dynamo_component_nats_client_connection_state gauge
dynamo_component_nats_client_connection_state 1
# HELP dynamo_component_latency Response latency
# TYPE dynamo_component_latency histogram
dynamo_component_latency_bucket{le="0.1"} 10
dynamo_component_latency_bucket{le="0.5"} 25
dynamo_component_nats_service_requests_total 100
dynamo_component_nats_service_errors_total 5"#;

        // Test remove_nats_lines (excludes NATS lines but keeps help/type)
        let filtered_out = super::test_helpers::remove_nats_lines(test_input);
        assert_eq!(filtered_out.len(), 7); // 7 non-NATS lines
        assert!(!filtered_out.iter().any(|line| line.contains("nats")));

        // Test extract_nats_lines (includes all NATS lines including help/type)
        let filtered_only = super::test_helpers::extract_nats_lines(test_input);
        assert_eq!(filtered_only.len(), 5); // 5 NATS lines
        assert!(filtered_only.iter().all(|line| line.contains("nats")));

        // Test extract_metrics (only actual metric lines, excluding help/type)
        let metrics_only = super::test_helpers::extract_metrics(test_input);
        assert_eq!(metrics_only.len(), 6); // 6 actual metric lines (excluding help/type)
        assert!(
            metrics_only
                .iter()
                .all(|line| line.starts_with("dynamo_component") && !line.starts_with("#"))
        );

        println!("✓ All refactored filter functions work correctly!");
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod test_metricsregistry_nats {
    use super::prometheus_names::name_prefix;
    use super::prometheus_names::{COMPONENT_NATS_METRICS, DRT_NATS_METRICS};
    use super::prometheus_names::{nats_client, nats_service};
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::pipeline::PushRouter;
    use crate::{DistributedRuntime, Runtime};
    use tokio::time::{Duration, sleep};
    #[tokio::test]
    async fn test_drt_nats_metrics() {
        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt_async().await;

        // Get DRT output which should include NATS client metrics
        let drt_output = drt.prometheus_metrics_fmt().unwrap();
        println!("DRT output with NATS metrics:");
        println!("{}", drt_output);

        // Additional checks for NATS client metrics (without checking specific values)
        let drt_nats_metrics = super::test_helpers::extract_nats_lines(&drt_output);

        // Check that NATS client metrics are present
        assert!(
            !drt_nats_metrics.is_empty(),
            "NATS client metrics should be present"
        );

        // Check for specific NATS client metric names (without values)
        // Extract only the metric lines from the already-filtered NATS metrics
        let drt_nats_metric_lines =
            super::test_helpers::extract_metrics(&drt_nats_metrics.join("\n"));
        let actual_drt_nats_metrics_sorted: Vec<&str> = drt_nats_metric_lines
            .iter()
            .map(|line| {
                let without_labels = line.split('{').next().unwrap_or(line);
                // Remove the value part (everything after the last space)
                without_labels.split(' ').next().unwrap_or(without_labels)
            })
            .collect();

        let expect_drt_nats_metrics_sorted = {
            let mut temp = DRT_NATS_METRICS
                .iter()
                .map(|metric| build_component_metric_name(metric))
                .collect::<Vec<_>>();
            temp.sort();
            temp
        };

        // Print both lists for comparison
        println!(
            "actual_drt_nats_metrics_sorted: {:?}",
            actual_drt_nats_metrics_sorted
        );
        println!(
            "expect_drt_nats_metrics_sorted: {:?}",
            expect_drt_nats_metrics_sorted
        );

        // Compare the sorted lists
        assert_eq!(
            actual_drt_nats_metrics_sorted, expect_drt_nats_metrics_sorted,
            "DRT_NATS_METRICS with prefix and expected_nats_metrics should be identical when sorted"
        );

        println!("✓ DistributedRuntime NATS metrics integration test passed!");
    }

    #[tokio::test]
    async fn test_nats_metric_names() {
        // This test only tests the existence of the NATS metrics. It does not check
        // the values of the metrics.

        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt_async().await;

        // Create a namespace and components from the DRT
        let namespace = drt.namespace("ns789").unwrap();
        let components = namespace.component("comp789").unwrap();

        // Create a service to trigger metrics callback registration
        let _service = components.service_builder().create().await.unwrap();

        // Get components output which should include NATS client metrics
        // Additional checks for NATS client metrics (without checking specific values)
        let component_nats_metrics =
            super::test_helpers::extract_nats_lines(&components.prometheus_metrics_fmt().unwrap());
        println!(
            "Component NATS metrics count: {}",
            component_nats_metrics.len()
        );

        // Check that NATS client metrics are present
        assert!(
            !component_nats_metrics.is_empty(),
            "NATS client metrics should be present"
        );

        // Check for specific NATS client metric names (without values)
        let component_metrics =
            super::test_helpers::extract_metrics(&components.prometheus_metrics_fmt().unwrap());
        let actual_component_nats_metrics_sorted: Vec<&str> = component_metrics
            .iter()
            .map(|line| {
                let without_labels = line.split('{').next().unwrap_or(line);
                // Remove the value part (everything after the last space)
                without_labels.split(' ').next().unwrap_or(without_labels)
            })
            .collect();

        let expect_component_nats_metrics_sorted = {
            let mut temp = COMPONENT_NATS_METRICS
                .iter()
                .map(|metric| build_component_metric_name(metric))
                .collect::<Vec<_>>();
            temp.sort();
            temp
        };

        // Print both lists for comparison
        println!(
            "actual_component_nats_metrics_sorted: {:?}",
            actual_component_nats_metrics_sorted
        );
        println!(
            "expect_component_nats_metrics_sorted: {:?}",
            expect_component_nats_metrics_sorted
        );

        // Compare the sorted lists
        assert_eq!(
            actual_component_nats_metrics_sorted, expect_component_nats_metrics_sorted,
            "COMPONENT_NATS_METRICS with prefix and expected_nats_metrics should be identical when sorted"
        );

        // Get both DRT and component output and filter for NATS metrics only
        let drt_output = drt.prometheus_metrics_fmt().unwrap();
        let drt_nats_lines = super::test_helpers::extract_nats_lines(&drt_output);
        let drt_and_component_nats_metrics =
            super::test_helpers::extract_metrics(&drt_nats_lines.join("\n"));
        println!(
            "DRT and component NATS metrics count: {}",
            drt_and_component_nats_metrics.len()
        );

        // Check that the NATS metrics are present in the component output
        assert_eq!(
            drt_and_component_nats_metrics.len(),
            DRT_NATS_METRICS.len() + COMPONENT_NATS_METRICS.len(),
            "DRT at this point should have both the DRT and component NATS metrics"
        );

        // Check that the NATS metrics are present in the component output
        println!("✓ Component NATS metrics integration test passed!");
    }

    /// Tests NATS metrics values before and after endpoint activity with large message processing.
    /// Creates endpoint, sends test messages + 10k byte message, validates metrics (NATS + work handler)
    /// at initial state and post-activity state. Ensures byte thresholds, message counts, and processing
    /// times are within expected ranges. Tests end-to-end client-server communication and metrics collection.
    #[tokio::test]
    async fn test_nats_metrics_values() -> anyhow::Result<()> {
        struct MessageHandler {}
        impl MessageHandler {
            fn new() -> std::sync::Arc<Self> {
                std::sync::Arc::new(Self {})
            }
        }

        #[async_trait]
        impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for MessageHandler {
            async fn generate(
                &self,
                input: SingleIn<String>,
            ) -> Result<ManyOut<Annotated<String>>, Error> {
                let (data, ctx) = input.into_parts();
                let response = data.to_string();
                let stream = stream::iter(vec![Annotated::from_data(response)]);
                Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
            }
        }

        println!("\n=== Initializing DistributedRuntime ===");
        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
        let namespace = drt.namespace("ns123").unwrap();
        let component = namespace.component("comp123").unwrap();
        let ingress = Ingress::for_engine(MessageHandler::new()).unwrap();

        let _backend_handle = tokio::spawn(async move {
            let service = component.service_builder().create().await.unwrap();
            let endpoint = service.endpoint("echo").endpoint_builder().handler(ingress);
            endpoint.start().await.unwrap();
        });

        sleep(Duration::from_millis(500)).await;
        println!("✓ Launched endpoint service in background successfully");

        let drt_output = drt.prometheus_metrics_fmt().unwrap();
        let parsed_metrics: Vec<_> = drt_output
            .lines()
            .filter_map(super::test_helpers::parse_prometheus_metric)
            .collect();

        println!("=== Initial DRT metrics output ===");
        println!("{}", drt_output);

        println!("\n=== Checking Initial Metric Values ===");

        let initial_expected_metric_values = [
            // DRT NATS metrics (ordered to match DRT_NATS_METRICS)
            (
                build_component_metric_name(nats_client::CONNECTION_STATE),
                1.0,
                1.0,
            ), // Should be connected
            (
                build_component_metric_name(nats_client::CURRENT_CONNECTIONS),
                1.0,
                1.0,
            ), // Should have 1 connection
            (
                build_component_metric_name(nats_client::IN_TOTAL_BYTES),
                800.0,
                4000.0,
            ), // Wide range around observed value of 1888
            (
                build_component_metric_name(nats_client::IN_MESSAGES),
                0.0,
                5.0,
            ), // Wide range around 2
            (
                build_component_metric_name(nats_client::OUT_OVERHEAD_BYTES),
                1500.0,
                5000.0,
            ), // Wide range around observed value of 2752
            (
                build_component_metric_name(nats_client::OUT_MESSAGES),
                0.0,
                5.0,
            ), // Wide range around 2
            // Component NATS metrics (ordered to match COMPONENT_NATS_METRICS)
            (
                build_component_metric_name(nats_service::PROCESSING_MS_AVG),
                0.0,
                0.0,
            ), // No processing yet
            (
                build_component_metric_name(nats_service::ERRORS_TOTAL),
                0.0,
                0.0,
            ), // No errors yet
            (
                build_component_metric_name(nats_service::REQUESTS_TOTAL),
                0.0,
                0.0,
            ), // No requests yet
            (
                build_component_metric_name(nats_service::PROCESSING_MS_TOTAL),
                0.0,
                0.0,
            ), // No processing yet
            (
                build_component_metric_name(nats_service::ACTIVE_SERVICES),
                0.0,
                2.0,
            ), // Service may not be fully active yet
            (
                build_component_metric_name(nats_service::ACTIVE_ENDPOINTS),
                0.0,
                2.0,
            ), // Endpoint may not be fully active yet
        ];

        for (metric_name, min_value, max_value) in &initial_expected_metric_values {
            let actual_value = parsed_metrics
                .iter()
                .find(|(name, _, _)| name == metric_name)
                .map(|(_, _, value)| *value)
                .unwrap_or_else(|| panic!("Could not find expected metric: {}", metric_name));

            assert!(
                actual_value >= *min_value && actual_value <= *max_value,
                "Initial metric {} should be between {} and {}, but got {}",
                metric_name,
                min_value,
                max_value,
                actual_value
            );
        }

        println!("\n=== Client Runtime to hit the endpoint ===");
        let client_runtime = Runtime::from_current()?;
        let client_distributed = DistributedRuntime::from_settings(client_runtime.clone()).await?;
        let namespace = client_distributed.namespace("ns123")?;
        let component = namespace.component("comp123")?;
        let client = component.endpoint("echo").client().await?;

        client.wait_for_instances().await?;
        println!("✓ Connected to endpoint, waiting for instances...");

        let router =
            PushRouter::<String, Annotated<String>>::from_client(client, Default::default())
                .await?;

        for i in 0..10 {
            let msg = i.to_string().repeat(2000); // 2k bytes message
            let mut stream = router.random(msg.clone().into()).await?;
            while let Some(resp) = stream.next().await {
                // Check if response matches the original message
                if let Some(data) = &resp.data {
                    let is_same = data == &msg;
                    println!(
                        "Response {}: {} bytes, matches original: {}",
                        i,
                        data.len(),
                        is_same
                    );
                }
            }
        }
        println!("✓ Sent messages and received responses successfully");

        println!("\n=== Waiting 500ms for metrics to update ===");
        sleep(Duration::from_millis(500)).await;
        println!("✓ Wait complete, getting final metrics...");

        let final_drt_output = drt.prometheus_metrics_fmt().unwrap();
        println!("\n=== Final Prometheus DRT output ===");
        println!("{}", final_drt_output);

        let final_drt_nats_output = super::test_helpers::extract_nats_lines(&final_drt_output);
        println!("\n=== Filtered NATS metrics from final DRT output ===");
        for line in &final_drt_nats_output {
            println!("{}", line);
        }

        let final_parsed_metrics: Vec<_> = super::test_helpers::extract_metrics(&final_drt_output)
            .iter()
            .filter_map(|line| super::test_helpers::parse_prometheus_metric(line.as_str()))
            .collect();

        let post_expected_metric_values = [
            // DRT NATS metrics
            (
                build_component_metric_name(nats_client::CONNECTION_STATE),
                1.0,
                1.0,
            ), // Connected
            (
                build_component_metric_name(nats_client::CURRENT_CONNECTIONS),
                1.0,
                1.0,
            ), // 1 connection
            (
                build_component_metric_name(nats_client::IN_TOTAL_BYTES),
                20000.0,
                32000.0,
            ), // Wide range around 26117
            (
                build_component_metric_name(nats_client::IN_MESSAGES),
                8.0,
                20.0,
            ), // Wide range around 16
            (
                build_component_metric_name(nats_client::OUT_OVERHEAD_BYTES),
                2500.0,
                8000.0,
            ), // Wide range around 5524
            (
                build_component_metric_name(nats_client::OUT_MESSAGES),
                8.0,
                20.0,
            ), // Wide range around 16
            // Component NATS metrics
            (
                build_component_metric_name(nats_service::PROCESSING_MS_AVG),
                0.0,
                1.0,
            ), // Low processing time
            (
                build_component_metric_name(nats_service::ERRORS_TOTAL),
                0.0,
                0.0,
            ), // No errors
            (
                build_component_metric_name(nats_service::REQUESTS_TOTAL),
                0.0,
                10.0,
            ), // NATS service stats requests (may differ from work handler count)
            (
                build_component_metric_name(nats_service::PROCESSING_MS_TOTAL),
                0.0,
                5.0,
            ), // Low total processing time
            (
                build_component_metric_name(nats_service::ACTIVE_SERVICES),
                0.0,
                2.0,
            ), // Service may not be fully active
            (
                build_component_metric_name(nats_service::ACTIVE_ENDPOINTS),
                0.0,
                2.0,
            ), // Endpoint may not be fully active
            // Work handler metrics
            (
                build_component_metric_name(work_handler::REQUESTS_TOTAL),
                10.0,
                10.0,
            ), // 10 messages
            (
                build_component_metric_name(work_handler::REQUEST_BYTES_TOTAL),
                21000.0,
                26000.0,
            ), // ~75-125% of 23520
            (
                build_component_metric_name(work_handler::RESPONSE_BYTES_TOTAL),
                18000.0,
                23000.0,
            ), // ~75-125% of 20660
            (
                build_component_metric_name(work_handler::INFLIGHT_REQUESTS),
                0.0,
                1.0,
            ), // 0 or very low
            // Histograms have _{count,sum} suffixes
            (
                format!(
                    "{}_count",
                    build_component_metric_name(work_handler::REQUEST_DURATION_SECONDS)
                ),
                10.0,
                10.0,
            ), // 10 messages
            (
                format!(
                    "{}_sum",
                    build_component_metric_name(work_handler::REQUEST_DURATION_SECONDS)
                ),
                0.0001,
                1.0,
            ), // Processing time sum (wide range)
        ];

        println!("\n=== Checking Post-Activity All Metrics (NATS + Work Handler) ===");
        for (metric_name, min_value, max_value) in &post_expected_metric_values {
            let actual_value = final_parsed_metrics
                .iter()
                .find(|(name, _, _)| name == metric_name)
                .map(|(_, _, value)| *value)
                .unwrap_or_else(|| {
                    panic!(
                        "Could not find expected post-activity metric: {}",
                        metric_name
                    )
                });

            assert!(
                actual_value >= *min_value && actual_value <= *max_value,
                "Post-activity metric {} should be between {} and {}, but got {}",
                metric_name,
                min_value,
                max_value,
                actual_value
            );
            println!(
                "✓ {}: {} (range: {} to {})",
                metric_name, actual_value, min_value, max_value
            );
        }

        println!("✓ All NATS and component metrics parsed successfully!");
        println!("✓ Byte metrics verified to be >= 100 bytes!");
        println!("✓ Post-activity metrics verified with higher thresholds!");
        println!("✓ Work handler metrics reflect increased activity!");

        Ok(())
    }
}
