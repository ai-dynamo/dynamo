// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Metric Registry Framework for Dynamo.
//!
//! This module provides registry classes for Prometheus metrics
//! that auto populates the labels with the namespace-component-endpoint hierarchy.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// This constant determines whether metric names should include the full hierarchy as a prefix.
// If set to true, a hierarchy like ["", "mynamespace", "mycomponent", "myendpoint"]
// results in a metric name of "mynamespace_mycomponent_myendpoint__myendpoint".
// If false, the metric name will be just "myendpoint".
// This setting is applied *universally* to ensure consistent naming conventions.
pub const USE_PREFIXED_METRIC_NAMES: bool = false;

// If set to true, then metrics will be labeled with the namespace, component, and endpoint.
pub const USE_AUTO_LABELS: bool = true;

// Prometheus imports
use prometheus::Encoder;

fn build_metric_name(prefix: &str, metric_name: &str) -> String {
    if !USE_PREFIXED_METRIC_NAMES {
        return metric_name.to_string();
    }

    if prefix.is_empty() {
        metric_name.to_string()
    } else {
        // Double underscore to separate between prefix and actual metric name
        format!("{}__{}", prefix, metric_name)
    }
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
) -> anyhow::Result<Arc<T>> {
    // Validate that user-provided labels don't have duplicate keys
    let mut seen_keys = std::collections::HashSet::new();

    let basename = registry.basename();
    let metric_name = build_metric_name(&registry.prefix(), metric_name);
    let parent_hierarchy = registry.parent_hierarchy();

    // Validate that user-provided labels don't have duplicate keys
    for (key, _) in labels {
        if !seen_keys.insert(*key) {
            return Err(anyhow::anyhow!(
                "Duplicate label key '{}' found in labels",
                key
            ));
        }
    }

    let hierarchy = [parent_hierarchy, vec![basename]].concat();
    // Build updated_labels: auto-labels first, then user labels
    let mut updated_labels: Vec<(String, String)> = Vec::new();

    if USE_AUTO_LABELS {
        // Validate that user-provided labels don't conflict with auto-generated labels
        for (key, _) in labels {
            if *key == "namespace" || *key == "component" || *key == "endpoint" {
                return Err(anyhow::anyhow!(
                    "Label '{}' is automatically added by auto_label feature and cannot be manually set",
                    key
                ));
            }
        }

        // Add auto-generated labels
        if hierarchy.len() > 1 {
            let namespace = &hierarchy[1];
            if !namespace.is_empty() {
                updated_labels.push(("namespace".to_string(), namespace.clone()));
            }
        }
        if hierarchy.len() > 2 {
            let component = &hierarchy[2];
            if !component.is_empty() {
                updated_labels.push(("component".to_string(), component.clone()));
            }
        }
        if hierarchy.len() > 3 {
            let endpoint = &hierarchy[3];
            if !endpoint.is_empty() {
                updated_labels.push(("endpoint".to_string(), endpoint.clone()));
            }
        }
    }

    // Add user labels
    updated_labels.extend(
        labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );

    // Handle different metric types
    let metric = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<prometheus::Histogram>() {
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
    // The prefixed_hierarchy is structured as: ["", "mynamespace", "mynamespace_mycomponent", "mynamespace_mycomponent_myendpoint"]
    // This prefixing is essential to differentiate between the names of children and grandchildren.
    let mut prometheus_registry = registry
        .drt()
        .prometheus_registries_by_prefix
        .lock()
        .unwrap();

    // Build prefixed hierarchy and register metrics in a single loop
    // current_prefix accumulates the hierarchical path as we iterate through hierarchy
    // For example, if hierarchy = ["", "mynamespace", "mycomponent"], then:
    // - Iteration 1: current_prefix = "" (empty string from DRT)
    // - Iteration 2: current_prefix = "mynamespace"
    // - Iteration 3: current_prefix = "mynamespace_mycomponent"
    let mut current_prefix = String::new();
    for name in &hierarchy {
        if !current_prefix.is_empty() && !name.is_empty() {
            current_prefix.push('_');
        }
        current_prefix.push_str(name);

        // Register metric at this hierarchical level
        let collector: Box<dyn prometheus::core::Collector> = Box::new(metric.clone());
        let _ = prometheus_registry
            .entry(current_prefix.clone())
            .or_default()
            .register(collector);
    }

    Ok(Arc::new(metric))
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsRegistry: Send + Sync + crate::traits::DistributedRuntimeProvider {
    // Get the name of this registry (without any prefix)
    fn basename(&self) -> String;

    /// Retrieve the complete hierarchy and basename for this registry. Currently, the prefix for drt is an empty string,
    /// so we must account for the leading underscore. The existing code remains unchanged to accommodate any future
    /// scenarios where drt's prefix might be assigned a value.
    fn prefix(&self) -> String {
        [self.parent_hierarchy(), vec![self.basename()]]
            .concat()
            .join("_")
            .trim_start_matches('_')
            .to_string()
    }

    // Get the parent hierarchy for this registry (just the base names, NOT the prefix)
    fn parent_hierarchy(&self) -> Vec<String>;

    // TODO: Add support for additional Prometheus metric types:
    // - Counter: ✅ IMPLEMENTED - create_counter()
    // - CounterVec: ✅ IMPLEMENTED - create_countervec()
    // - IntCounter: ✅ IMPLEMENTED - create_intcounter()
    // - Gauge: ✅ IMPLEMENTED - create_gauge()
    // - IntGauge/IntGaugeVec: ✅ IMPLEMENTED - create_intgauge() and create_intgaugevec()
    // - Histogram: ✅ IMPLEMENTED - create_histogram()
    // - Summary: create_summary() - for quantiles and sum/count metrics
    // - HistogramVec with custom buckets: create_histogram_with_buckets()
    // - SummaryVec: create_summary_vec() - for labeled summaries
    // - Untyped: create_untyped() - for untyped metrics
    // - Info: create_info() - for info metrics with labels
    // - Stateset: create_stateset() - for state-based metrics
    // - GaugeHistogram: create_gauge_histogram() - for gauge histograms

    /// Create a Counter metric
    fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::Counter>> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create a Gauge metric
    fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::Gauge>> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create an IntCounter metric
    fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::IntCounter>> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create a Histogram metric with custom buckets
    fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
    ) -> anyhow::Result<Arc<prometheus::Histogram>> {
        create_metric(self, name, description, labels, buckets, None)
    }

    /// Create a CounterVec metric with label names (for dynamic labels)
    fn create_countervec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::CounterVec>> {
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
    ) -> anyhow::Result<Arc<prometheus::IntGauge>> {
        create_metric(self, name, description, labels, None, None)
    }

    /// Create an IntGaugeVec metric with label names (for dynamic labels)
    fn create_intgaugevec(
        &self,
        name: &str,
        description: &str,
        const_labels: &[&str],
        const_label_values: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::IntGaugeVec>> {
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
        let prometheus_registry = {
            let mut registry = self.drt().prometheus_registries_by_prefix.lock().unwrap();
            registry.entry(self.prefix()).or_default().clone()
        };
        let metric_families = prometheus_registry.gather();
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

#[cfg(test)]
/// Helper function to create a DRT instance for testing
/// Uses the test-friendly constructor without discovery
pub fn create_test_drt() -> crate::DistributedRuntime {
    let rt = crate::Runtime::single_threaded().unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        crate::DistributedRuntime::from_settings_without_discovery(rt.clone())
            .await
            .unwrap()
    })
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod test_prefixes {
    use super::create_test_drt;
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_hierarchical_prefixes_and_parent_hierarchies() {
        println!("=== Testing Names, Prefixes, and Parent Hierarchies ===");

        // Create a distributed runtime for testing
        let drt = create_test_drt();

        // Generate random namespace name
        let mut hasher = DefaultHasher::new();
        "test_namespace".hash(&mut hasher);
        let random_num = hasher.finish();
        let namespace_name = format!("mynamespace{}", random_num);

        // Create namespace
        let namespace = drt.namespace(&namespace_name).unwrap();

        // Create component
        let component = namespace.component("mycomponent").unwrap();

        // Create endpoint
        let endpoint = component.endpoint("myendpoint");

        // Test DistributedRuntime hierarchy
        println!("\n=== DistributedRuntime ===");
        println!("basename: '{}'", drt.basename());
        println!("parent_hierarchy: {:?}", drt.parent_hierarchy());
        println!("prefix: '{}'", drt.prefix());

        assert_eq!(drt.basename(), "", "DRT basename should be empty");
        assert_eq!(
            drt.parent_hierarchy(),
            Vec::<String>::new(),
            "DRT parent hierarchy should be empty"
        );
        assert_eq!(drt.prefix(), "", "DRT prefix should be empty");

        // Test Namespace hierarchy
        println!("\n=== Namespace ===");
        println!("basename: '{}'", namespace.basename());
        println!("parent_hierarchy: {:?}", namespace.parent_hierarchy());
        println!("prefix: '{}'", namespace.prefix());

        assert_eq!(
            namespace.basename(),
            namespace_name,
            "Namespace basename should match the generated name"
        );
        assert_eq!(
            namespace.parent_hierarchy(),
            vec![""],
            "Namespace parent hierarchy should be [\"\"]"
        );
        assert_eq!(
            namespace.prefix(),
            namespace_name,
            "Namespace prefix should match the generated name, because drt's prefix is empty"
        );

        // Test Component hierarchy
        println!("\n=== Component ===");
        println!("basename: '{}'", component.basename());
        println!("parent_hierarchy: {:?}", component.parent_hierarchy());
        println!("prefix: '{}'", component.prefix());

        assert_eq!(
            component.basename(),
            "mycomponent",
            "Component basename should be 'mycomponent'"
        );
        assert_eq!(
            component.parent_hierarchy(),
            vec!["", &namespace_name],
            "Component parent hierarchy should contain the generated namespace name"
        );
        assert_eq!(
            component.prefix(),
            format!("{}_mycomponent", namespace),
            "Component prefix should be 'namespace_mycomponent'"
        );

        // Test Endpoint hierarchy
        println!("\n=== Endpoint ===");
        println!("basename: '{}'", endpoint.basename());
        println!("parent_hierarchy: {:?}", endpoint.parent_hierarchy());
        println!("prefix: '{}'", endpoint.prefix());

        assert_eq!(
            endpoint.basename(),
            "myendpoint",
            "Endpoint basename should be 'myendpoint'"
        );
        assert_eq!(
            endpoint.parent_hierarchy(),
            vec!["", &namespace_name, "mycomponent"],
            "Endpoint parent hierarchy should contain the generated namespace name"
        );
        assert_eq!(
            endpoint.prefix(),
            format!("{}_mycomponent_myendpoint", namespace),
            "Endpoint prefix should be 'namespace_mycomponent_myendpoint'"
        );

        // Test hierarchy relationships
        println!("\n=== Hierarchy Relationships ===");
        assert!(
            namespace.parent_hierarchy().contains(&drt.basename()),
            "Namespace should have DRT prefix in parent hierarchy"
        );
        assert!(
            component.parent_hierarchy().contains(&namespace.basename()),
            "Component should have Namespace prefix in parent hierarchy"
        );
        assert!(
            endpoint.parent_hierarchy().contains(&component.basename()),
            "Endpoint should have Component prefix in parent hierarchy"
        );
        println!("✓ All parent-child relationships verified");

        // Test hierarchy depth
        println!("\n=== Hierarchy Depth ===");
        assert_eq!(
            drt.parent_hierarchy().len(),
            0,
            "DRT should have 0 parent hierarchy levels"
        );
        assert_eq!(
            namespace.parent_hierarchy().len(),
            1,
            "Namespace should have 1 parent hierarchy level"
        );
        assert_eq!(
            component.parent_hierarchy().len(),
            2,
            "Component should have 2 parent hierarchy levels"
        );
        assert_eq!(
            endpoint.parent_hierarchy().len(),
            3,
            "Endpoint should have 3 parent hierarchy levels"
        );
        println!("✓ All hierarchy depths verified");

        // Summary
        println!("\n=== Summary ===");
        println!("DRT prefix: '{}'", drt.prefix());
        println!("Namespace prefix: '{}'", namespace.prefix());
        println!("Component prefix: '{}'", component.prefix());
        println!("Endpoint prefix: '{}'", endpoint.prefix());
        println!("All hierarchy assertions passed!");
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod test_simple_metricsregistry_trait {
    use super::create_test_drt;
    use super::*;
    use prometheus::Counter;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    #[test]
    fn test_factory_methods_via_registry_trait() {
        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt();

        // Generate random namespace name
        let mut hasher = DefaultHasher::new();
        "test_factory_namespace".hash(&mut hasher);
        let random_num = hasher.finish();
        let namespace_name = format!("mynamespace{}", random_num);

        let namespace = drt.namespace(&namespace_name).unwrap();
        let component = namespace.component("mycomponent").unwrap();
        let endpoint = component.endpoint("myendpoint");

        // Test Counter creation
        let counter = endpoint
            .create_counter("mycounter", "A test counter", &[])
            .unwrap();
        counter.inc_by(123.456789);
        let epsilon = 0.01;
        assert!((counter.get() - 123.456789).abs() < epsilon);

        let endpoint_output = endpoint.prometheus_metrics_fmt().unwrap();
        println!("Endpoint output:");
        println!("{}", endpoint_output);

        let expected_endpoint_output = format!(
            r#"# HELP mycounter A test counter
# TYPE mycounter counter
mycounter{{component="mycomponent",endpoint="myendpoint",namespace="{}"}} 123.456789
"#,
            namespace_name
        );

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
            .create_gauge("mygauge", "A test gauge", &[])
            .unwrap();
        gauge.set(50000.0);
        assert_eq!(gauge.get(), 50000.0);

        // Test Prometheus format output for Component (gauge + histogram)
        let component_output = component.prometheus_metrics_fmt().unwrap();
        println!("Component output:");
        println!("{}", component_output);

        let expected_component_output = format!(
            r#"# HELP mycounter A test counter
# TYPE mycounter counter
mycounter{{component="mycomponent",endpoint="myendpoint",namespace="{}"}} 123.456789
# HELP mygauge A test gauge
# TYPE mygauge gauge
mygauge{{component="mycomponent",namespace="{}"}} 50000
"#,
            namespace_name, namespace_name
        );

        assert_eq!(
            component_output, expected_component_output,
            "\n=== COMPONENT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_component_output, component_output
        );

        let intcounter = namespace
            .create_intcounter("myintcounter", "A test int counter", &[])
            .unwrap();
        intcounter.inc_by(12345);
        assert_eq!(intcounter.get(), 12345);

        // Test Prometheus format output for Namespace (int_counter + gauge + histogram)
        let namespace_output = namespace.prometheus_metrics_fmt().unwrap();
        println!("Namespace output:");
        println!("{}", namespace_output);

        let expected_namespace_output = format!(
            r#"# HELP mycounter A test counter
# TYPE mycounter counter
mycounter{{component="mycomponent",endpoint="myendpoint",namespace="{}"}} 123.456789
# HELP mygauge A test gauge
# TYPE mygauge gauge
mygauge{{component="mycomponent",namespace="{}"}} 50000
# HELP myintcounter A test int counter
# TYPE myintcounter counter
myintcounter{{namespace="{}"}} 12345
"#,
            namespace_name, namespace_name, namespace_name
        );

        assert_eq!(
            namespace_output, expected_namespace_output,
            "\n=== NAMESPACE COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_namespace_output, namespace_output
        );

        // Create a histogram with specified buckets. The Prometheus format output will
        // lack labels since the DistributedRuntime is unnamed.
        let histogram = drt
            .create_histogram(
                "myhistogram",
                "A test histogram",
                &[],
                Some(vec![1.0, 2.5, 5.0, 10.0]),
            )
            .unwrap();
        histogram.observe(1.5);
        histogram.observe(2.5);
        histogram.observe(3.5);

        // Test CounterVec creation
        let countervec = drt
            .create_countervec(
                "mycountervec",
                "A test counter vector",
                &["method", "status"],
                &[("service", "api")],
            )
            .unwrap();
        countervec.with_label_values(&["GET", "200"]).inc_by(10.0);
        countervec.with_label_values(&["POST", "201"]).inc_by(5.0);

        // Test IntGauge creation
        let intgauge = drt
            .create_intgauge("myintgauge", "A test int gauge", &[])
            .unwrap();
        intgauge.set(42);
        assert_eq!(intgauge.get(), 42);

        // Test IntGaugeVec creation
        let intgaugevec = drt
            .create_intgaugevec(
                "myintgaugevec",
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

        // Test Prometheus format output for DRT (which should contain everything)
        let drt_output = drt.prometheus_metrics_fmt().unwrap();
        println!("DRT output:");
        println!("{}", drt_output);

        let expected_drt_output = format!(
            r#"# HELP mycounter A test counter
# TYPE mycounter counter
mycounter{{component="mycomponent",endpoint="myendpoint",namespace="{}"}} 123.456789
# HELP mycountervec A test counter vector
# TYPE mycountervec counter
mycountervec{{method="GET",service="api",status="200"}} 10
mycountervec{{method="POST",service="api",status="201"}} 5
# HELP mygauge A test gauge
# TYPE mygauge gauge
mygauge{{component="mycomponent",namespace="{}"}} 50000
# HELP myhistogram A test histogram
# TYPE myhistogram histogram
myhistogram_bucket{{le="1"}} 0
myhistogram_bucket{{le="2.5"}} 2
myhistogram_bucket{{le="5"}} 3
myhistogram_bucket{{le="10"}} 3
myhistogram_bucket{{le="+Inf"}} 3
myhistogram_sum 7.5
myhistogram_count 3
# HELP myintcounter A test int counter
# TYPE myintcounter counter
myintcounter{{namespace="{}"}} 12345
# HELP myintgauge A test int gauge
# TYPE myintgauge gauge
myintgauge 42
# HELP myintgaugevec A test int gauge vector
# TYPE myintgaugevec gauge
myintgaugevec{{instance="server1",service="api",status="active"}} 10
myintgaugevec{{instance="server2",service="api",status="inactive"}} 0
"#,
            namespace_name, namespace_name, namespace_name
        );

        assert_eq!(
            drt_output, expected_drt_output,
            "\n=== DRT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_drt_output, drt_output
        );

        println!("✓ All Prometheus format outputs verified successfully!");
    }
}
