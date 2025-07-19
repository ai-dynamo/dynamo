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
//! with shared interfaces for easy metric management.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Prometheus imports - using module alias for shorter typing
use prometheus as prom;
use prometheus::Encoder;

// Static constants for metric types
pub const COUNTER_METRIC_TYPE: &str = "counter";
pub const GAUGE_METRIC_TYPE: &str = "gauge";
pub const HISTOGRAM_METRIC_TYPE: &str = "histogram";

#[cfg(test)]
mod test_helpers {
    use super::*;
    use std::sync::Arc;

    /// Helper function to create a DRT instance for testing in sync contexts
    /// Uses the test-friendly constructor without discovery
    pub fn create_test_drt_sync() -> crate::DistributedRuntime {
        let rt = crate::Runtime::from_settings().unwrap();
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            crate::DistributedRuntime::from_settings(rt.clone())
                .await
                .unwrap()
        })
    }
}

// Private macro for internal metric registration within this module
// This macro can only be used within the MetricsRegistry trait methods
//
// ### Why This Macro is Needed
//
// **Problem 1: Generic Functions and dyn Compatibility**
// - We cannot add a generic `register_metric<T>(&self, metric: T)` function directly to the
//   `MetricsRegistry` trait because traits with generic methods are not `dyn`-compatible in Rust.
// - This would break the ability to use `dyn MetricsRegistry` as a trait object, which is
//   required throughout the codebase.
// - The macro provides generic-like functionality while maintaining `dyn` compatibility.
//
// **Problem 2: Box<dyn Collector> Clone Issues**
// - The Prometheus register type parameter **MUST** be `Box<dyn prometheus::core::Collector>`.
// - `Box<dyn Collector>` cannot be cloned because trait objects don't implement `Clone`.
// - To register a metric in multiple registries (one per prefix in the hierarchy), we must:
//   1. Clone the concrete metric type **BEFORE** boxing it
//   2. Create a new `Box<dyn Collector>` for each registry
// - The macro encapsulates this complex logic and ensures consistency.
//
// **Solution Benefits**
// - Maintains `dyn MetricsRegistry` compatibility throughout the codebase
// - Provides generic-like functionality without breaking trait object usage
// - Encapsulates the complex Box/clone logic in one place
// - Ensures consistent registration behavior across all metric types
// - Registers metrics hierarchically at all parent levels for proper aggregation
macro_rules! register_metric_inline {
    ($registry:expr, $metric:expr, $prefix_in_name:expr) => {{
        let mut registry = $registry
            .drt()
            .prometheus_registries_by_prefix
            .lock()
            .unwrap();

        // Register at all hierarchy levels, including the current level
        let mut hierarchy = $registry.parent_hierarchy();
        hierarchy.push($registry.basename());

        let mut prefix_and_name = String::new();
        for prefix in hierarchy {
            if $prefix_in_name {
                if !prefix_and_name.is_empty() {
                    prefix_and_name.push('_');
                }
                prefix_and_name.push_str(&prefix);
            }

            // Always register, even for empty prefixes (for DRT)
            let collector: Box<dyn prometheus::core::Collector> = Box::new($metric.clone());
            let _ = registry
                .entry(if $prefix_in_name { prefix_and_name.clone() } else { $registry.basename() })
                .or_insert(prometheus::Registry::new())
                .register(collector);
        }
    }};
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsRegistry: Send + Sync + crate::traits::DistributedRuntimeProvider {
    // Get the name of this registry (without any prefix)
    fn basename(&self) -> String;

    /// Get the full hierarchy+basename for this registry. Because drt's prefix is an empty string,
    /// we need to handle it separately.
    fn prefix(&self) -> String {
        [self.parent_hierarchy(), vec![self.basename()]]
            .concat()
            .join("_")
            .trim_start_matches('_')
            .to_string()
    }

    // Get the parent hierarchy for this registry (just the base names, NOT the prefix)
    fn parent_hierarchy(&self) -> Vec<String>;

    /// Helper method to build the full metric name with prefix
    fn build_metric_name(&self, metric_name: &str) -> String {
        if self.prefix().is_empty() {
            metric_name.to_string()
        } else {
            // Double underscore to separate between prefix and actual metric name
            format!("{}__{}", self.prefix(), metric_name)
        }
    }

    // TODO: Add support for Prometheus labels using *Vec types (CounterVec, GaugeVec, HistogramVec)
    // - When labels are provided, use the Vec types with label names and values
    // - When no labels are provided, fall back to simple types (Counter, Gauge, Histogram)
    // - This would allow metrics like: create_counter("requests", "Total requests", &[("service", "api"), ("method", "GET")])
    //
    // TODO: Add support for additional Prometheus metric types:
    // - Summary: create_summary() - for quantiles and sum/count metrics
    // - IntCounter/IntCounterVec: create_int_counter() - for integer counters
    // - IntGauge/IntGaugeVec: create_int_gauge() - for integer gauges
    // - HistogramVec with custom buckets: create_histogram_with_buckets()
    // - SummaryVec: create_summary_vec() - for labeled summaries
    // - Untyped: create_untyped() - for untyped metrics
    // - Info: create_info() - for info metrics with labels
    // - Stateset: create_stateset() - for state-based metrics
    // - GaugeHistogram: create_gauge_histogram() - for gauge histograms

    /// Create a new counter metric
    fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        prefix_in_name: bool,
    ) -> anyhow::Result<Arc<prometheus::Counter>> {
        let full_name = self.build_metric_name(name);
        let counter = prometheus::Counter::new(&full_name, description)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, counter.clone(), prefix_in_name);
        Ok(Arc::new(counter.clone()))
    }

    /// Create a new gauge metric
    fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        prefix_in_name: bool,
    ) -> anyhow::Result<Arc<prometheus::Gauge>> {
        let full_name = self.build_metric_name(name);
        let gauge = prometheus::Gauge::new(&full_name, description)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, gauge.clone(), prefix_in_name);
        Ok(Arc::new(gauge))
    }

    /// Create a new histogram metric
    fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
        prefix_in_name: bool,
    ) -> anyhow::Result<Arc<prometheus::Histogram>> {
        let full_name = self.build_metric_name(name);
        let opts = prometheus::HistogramOpts::new(&full_name, description);
        let histogram = prometheus::Histogram::with_opts(opts)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, histogram.clone(), prefix_in_name);
        Ok(Arc::new(histogram))
    }

    /// Get metrics in Prometheus text format
    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        let mut registry = self.drt().prometheus_registries_by_prefix.lock().unwrap();
        let prometheus_registry = registry
            .entry(self.prefix())
            .or_insert(prometheus::Registry::new())
            .clone();
        let metric_families = prometheus_registry.gather();
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

#[cfg(test)]
mod test_simple_registry_trait {
    use super::test_helpers::create_test_drt_sync;
    use super::*;
    use prometheus::Counter;
    use std::sync::Arc;

    struct TestRegistry {
        drt: Arc<crate::DistributedRuntime>,
        prefix: String,
    }

    impl crate::traits::DistributedRuntimeProvider for TestRegistry {
        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }
    }

    impl MetricsRegistry for TestRegistry {
        fn basename(&self) -> String {
            "testprefix".to_string()
        }
        fn parent_hierarchy(&self) -> Vec<String> {
            vec![]
        }
    }

    #[test]
    fn test_factory_methods_via_registry_trait() {
        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt_sync();
        let registry = TestRegistry {
            drt: Arc::new(drt),
            prefix: "testprefix".to_string(),
        };

        // Test Counter creation
        let counter = registry
            .create_counter("my_counter", "A test counter", &[], true)
            .unwrap();
        counter.inc_by(42.0);
        assert_eq!(counter.get() as u64, 42);

        // Test Gauge creation
        let gauge = registry
            .create_gauge("my_gauge", "A test gauge", &[], true)
            .unwrap();
        gauge.set(123.45);
        assert_eq!(gauge.get(), 123.45);

        // Test Histogram creation
        let histogram = registry
            .create_histogram("my_histogram", "A test histogram", &[], true)
            .unwrap();
        histogram.observe(1.5);
        histogram.observe(2.5);
        histogram.observe(3.5);
        // We can't assert the exact histogram buckets, but we can check the output contains the metric name

        // Get Prometheus format output
        let prometheus_output = registry.prometheus_metrics_fmt().unwrap();

        // Print only the histogram section for inspection
        println!("Histogram section:");
        for line in prometheus_output.lines() {
            if line.contains("my_histogram")
                || line.contains("# HELP testprefix_my_histogram")
                || line.contains("# TYPE testprefix_my_histogram")
            {
                println!("{}", line);
            }
        }

        // Verify all metrics are present in the output
        assert!(prometheus_output.contains("testprefix__my_counter"));
        assert!(prometheus_output.contains("testprefix__my_gauge"));
        assert!(prometheus_output.contains("testprefix__my_histogram"));
        assert!(prometheus_output.contains("# HELP testprefix__my_counter A test counter"));
        assert!(prometheus_output.contains("# HELP testprefix__my_gauge A test gauge"));
        assert!(prometheus_output.contains("# HELP testprefix__my_histogram A test histogram"));

        println!("All metric types test passed!");
    }
}

#[cfg(test)]
mod test_runtime_and_namespace_registry_trait {
    use super::test_helpers::create_test_drt_sync;
    use super::*;
    use std::sync::Arc;

    /// Test registry representing a runtime-level metrics registry
    struct MyRuntimeRegistry {
        drt: Arc<crate::DistributedRuntime>,
    }

    impl crate::traits::DistributedRuntimeProvider for MyRuntimeRegistry {
        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }
    }

    impl MetricsRegistry for MyRuntimeRegistry {
        fn basename(&self) -> String {
            "runtime".to_string()
        }
        fn parent_hierarchy(&self) -> Vec<String> {
            vec![]
        }
    }

    /// Test registry representing a namespace-level metrics registry
    struct MyNamespaceRegistry {
        drt: Arc<crate::DistributedRuntime>,
    }

    impl crate::traits::DistributedRuntimeProvider for MyNamespaceRegistry {
        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }
    }

    impl MetricsRegistry for MyNamespaceRegistry {
        fn basename(&self) -> String {
            "namespace".to_string()
        }
        fn parent_hierarchy(&self) -> Vec<String> {
            vec!["runtime".to_string()]
        }
    }

    #[test]
    fn test_runtime_and_namespace() {
        // Setup real DRT
        let drt = create_test_drt_sync();

        // Create runtime-level registry
        let runtime_registry = MyRuntimeRegistry {
            drt: Arc::new(drt.clone()),
        };

        // Create namespace-level registry
        let namespace_registry = MyNamespaceRegistry {
            drt: Arc::new(drt.clone()),
        };

        // Create metrics using factory methods and increment some values
        let runtime_counter = runtime_registry
            .create_counter("total_requests", "Total requests across all runtime", &[], true)
            .unwrap();
        runtime_counter.inc_by(100.0);

        let namespace_gauge = namespace_registry
            .create_gauge(
                "active_connections",
                "Active connections for this namespace",
                &[],
                true,
            )
            .unwrap();
        namespace_gauge.set(25.0);

        // Test Prometheus format output for both registries
        let runtime_output = runtime_registry.prometheus_metrics_fmt().unwrap();
        let namespace_output = namespace_registry.prometheus_metrics_fmt().unwrap();

        // Verify exact content in runtime output (this includes ALL)
        let expected_runtime_output = "\
# HELP runtime__total_requests Total requests across all runtime\n\
# TYPE runtime__total_requests counter\n\
runtime__total_requests 100\n\
# HELP runtime_namespace__active_connections Active connections for this namespace\n\
# TYPE runtime_namespace__active_connections gauge\n\
runtime_namespace__active_connections 25\n";
        assert_eq!(runtime_output, expected_runtime_output);

        // Verify exact content in namespace output (this only includes runtime_namespace))
        let expected_namespace_output = "\
# HELP runtime_namespace__active_connections Active connections for this namespace\n\
# TYPE runtime_namespace__active_connections gauge\n\
runtime_namespace__active_connections 25\n";
        assert_eq!(namespace_output, expected_namespace_output);

        println!("Runtime and Namespace registry tests passed!");
    }
}

#[cfg(test)]
mod test_prefixes {
    use super::test_helpers::create_test_drt_sync;
    use super::*;

    #[test]
    fn test_hierarchical_prefixes_and_parent_hierarchies() {
        println!("=== Testing Names, Prefixes, and Parent Hierarchies ===");

        // Create a distributed runtime for testing
        let drt = create_test_drt_sync();

        // Create namespace
        let ns = drt.namespace("mynamespace").unwrap();

        // Create component
        let component = ns.component("mycomponent").unwrap();

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
        println!("basename: '{}'", ns.basename());
        println!("parent_hierarchy: {:?}", ns.parent_hierarchy());
        println!("prefix: '{}'", ns.prefix());

        assert_eq!(
            ns.basename(),
            "mynamespace",
            "Namespace basename should be 'mynamespace'"
        );
        assert_eq!(
            ns.parent_hierarchy(),
            vec![""],
            "Namespace parent hierarchy should be [\"\"]"
        );
        assert_eq!(
            ns.prefix(),
            "mynamespace",
            "Namespace prefix should be 'mynamespace', because drt's prefix is empty"
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
            vec!["", "mynamespace"],
            "Component parent hierarchy should be [\"\", \"mynamespace\"]"
        );
        assert_eq!(
            component.prefix(),
            "mynamespace_mycomponent",
            "Component prefix should be 'mynamespace_mycomponent'"
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
        assert_eq!(endpoint.parent_hierarchy(), vec!["", "mynamespace", "mycomponent"], "Endpoint parent hierarchy should be [\"\", \"mynamespace\", \"mynamespace_mycomponent\"]");
        assert_eq!(
            endpoint.prefix(),
            "mynamespace_mycomponent_myendpoint",
            "Endpoint prefix should be 'mynamespace_mycomponent_myendpoint'"
        );

        // Test hierarchy relationships
        println!("\n=== Hierarchy Relationships ===");
        assert!(
            ns.parent_hierarchy().contains(&drt.basename()),
            "Namespace should have DRT prefix in parent hierarchy"
        );
        assert!(
            component.parent_hierarchy().contains(&ns.basename()),
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
            ns.parent_hierarchy().len(),
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
        println!("Namespace prefix: '{}'", ns.prefix());
        println!("Component prefix: '{}'", component.prefix());
        println!("Endpoint prefix: '{}'", endpoint.prefix());
        println!("All hierarchy assertions passed!");
    }

    #[test]
    fn test_prometheus_metrics_fmt_in_hierarchical_prefixes() {
        println!("=== MySystemStatsMetrics with Profiling Backend Test ===");

        // Create a distributed runtime for testing
        let drt = create_test_drt_sync();

        // Test Endpoint metrics first (Endpoint implements MetricsRegistry)
        let ns = drt.namespace("mynamespace").unwrap();
        let component = ns.component("mycomponent").unwrap();
        let endpoint = component.endpoint("myendpoint");
        let endpoint_count = endpoint
            .clone()
            .create_counter(
                "count",
                "Total number of requests processed",
                &[("service", "endpoint")],
                true,
            )
            .unwrap();
        endpoint_count.inc_by(4.0);
        let endpoint_output = endpoint.prometheus_metrics_fmt().unwrap();

        // Assert Endpoint metrics output
        let expected_endpoint_output = r#"# HELP mynamespace_mycomponent_myendpoint__count Total number of requests processed
# TYPE mynamespace_mycomponent_myendpoint__count counter
mynamespace_mycomponent_myendpoint__count 4
"#;
        assert_eq!(
            endpoint_output, expected_endpoint_output,
            "\n=== ENDPOINT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==================================",
            expected_endpoint_output, endpoint_output
        );

        // Test Component metrics
        let component_count = component
            .clone()
            .create_counter(
                "count",
                "Total number of requests processed",
                &[("service", "component")],
                true,
            )
            .unwrap();
        component_count.inc_by(3.0);
        let component_output = component.prometheus_metrics_fmt().unwrap();

        // Assert Component metrics output
        let expected_component_output = r#"# HELP mynamespace_mycomponent__count Total number of requests processed
# TYPE mynamespace_mycomponent__count counter
mynamespace_mycomponent__count 3
# HELP mynamespace_mycomponent_myendpoint__count Total number of requests processed
# TYPE mynamespace_mycomponent_myendpoint__count counter
mynamespace_mycomponent_myendpoint__count 4
"#;
        assert_eq!(
            component_output, expected_component_output,
            "\n=== COMPONENT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ====================================",
            expected_component_output, component_output
        );

        // Test Namespace metrics
        let ns_count = ns
            .clone()
            .create_counter(
                "count",
                "Total number of requests processed",
                &[("service", "ns")],
                true,
            )
            .unwrap();
        ns_count.inc_by(2.0);
        let ns_output = ns.prometheus_metrics_fmt().unwrap();

        // Assert Namespace metrics output
        let expected_ns_output = r#"# HELP mynamespace__count Total number of requests processed
# TYPE mynamespace__count counter
mynamespace__count 2
# HELP mynamespace_mycomponent__count Total number of requests processed
# TYPE mynamespace_mycomponent__count counter
mynamespace_mycomponent__count 3
# HELP mynamespace_mycomponent_myendpoint__count Total number of requests processed
# TYPE mynamespace_mycomponent_myendpoint__count counter
mynamespace_mycomponent_myendpoint__count 4
"#;
        assert_eq!(
            ns_output, expected_ns_output,
            "\n=== NAMESPACE COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==================================",
            expected_ns_output, ns_output
        );

        // Test DRT metrics last
        let drt_count = drt
            .clone()
            .create_counter(
                "count",
                "Total number of requests processed",
                &[("service", "drt")],
                true,
            )
            .unwrap();
        drt_count.inc_by(1.0);
        let drt_output = drt.prometheus_metrics_fmt().unwrap();
        println!("drt.prometheus_metrics_fmt(): {}", drt_output);

        // Assert DRT metrics output. This should contain ALL the metrics.
        let expected_drt_output = r#"# HELP count Total number of requests processed
# TYPE count counter
count 1
# HELP mynamespace__count Total number of requests processed
# TYPE mynamespace__count counter
mynamespace__count 2
# HELP mynamespace_mycomponent__count Total number of requests processed
# TYPE mynamespace_mycomponent__count counter
mynamespace_mycomponent__count 3
# HELP mynamespace_mycomponent_myendpoint__count Total number of requests processed
# TYPE mynamespace_mycomponent_myendpoint__count counter
mynamespace_mycomponent_myendpoint__count 4
"#;
        assert_eq!(
            drt_output, expected_drt_output,
            "\n=== DRT COMPARISON FAILED ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_drt_output, drt_output
        );
    }
}
