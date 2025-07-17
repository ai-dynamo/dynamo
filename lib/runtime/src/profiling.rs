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
macro_rules! register_metric_inline {
    ($registry:expr, $metric:expr) => {{
        let mut registry = $registry.drt().metrics_registries_by_prefix.lock().unwrap();
        for prefix in $registry.metrics_hierarchy() {
            let collector: Box<dyn prometheus::core::Collector> = Box::new($metric.clone());
            registry
                .entry(prefix)
                .or_insert(prometheus::Registry::new())
                .register(collector);
        }
    }};
}

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsRegistry: Send + Sync {
    /// Get the prefix for this registry
    fn metrics_prefix(&self) -> String;

    /// Get the hierarchy for this registry (includes its own metrics_prefix)
    fn metrics_hierarchy(&self) -> Vec<String>;

    // Get a reference to the distributed runtime. You cannot call this
    // drt because it'll collide with the DistributedRuntimeProvider trait.
    fn drt(&self) -> &crate::DistributedRuntime;

    fn registry(&self) -> prometheus::Registry {
        let mut registry = self.drt().metrics_registries_by_prefix.lock().unwrap();
        registry
            .entry(self.metrics_prefix())
            .or_insert(prometheus::Registry::new())
            .clone()
    }

    /// Helper method to build the full metric name with prefix
    fn build_metric_name(&self, name: &str) -> String {
        if self.metrics_prefix().is_empty() {
            name.to_string()
        } else {
            format!("{}__{}", self.metrics_prefix(), name)
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
    ) -> anyhow::Result<Arc<prometheus::Counter>> {
        let full_name = self.build_metric_name(name);
        let counter = prometheus::Counter::new(&full_name, description)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, counter.clone());
        Ok(Arc::new(counter.clone()))
    }

    /// Create a new gauge metric
    fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::Gauge>> {
        let full_name = self.build_metric_name(name);
        let gauge = prometheus::Gauge::new(&full_name, description)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, gauge.clone());
        Ok(Arc::new(gauge))
    }

    /// Create a new histogram metric
    fn create_histogram(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Arc<prometheus::Histogram>> {
        let full_name = self.build_metric_name(name);
        let opts = prometheus::HistogramOpts::new(&full_name, description);
        let histogram = prometheus::Histogram::with_opts(opts)?;
        // Use the private macro to register the metric inline
        register_metric_inline!(self, histogram.clone());
        Ok(Arc::new(histogram))
    }

    /// Get metrics in Prometheus text format
    fn encode_prometheus_str(&self) -> anyhow::Result<String> {
        let metric_families = self.registry().gather();
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

#[cfg(test)]
mod test_simple_registry_trait {
    use super::*;
    use prometheus::Counter;
    use std::sync::Arc;

    struct TestRegistry {
        drt: Arc<crate::DistributedRuntime>,
        prefix: String,
    }

    impl MetricsRegistry for TestRegistry {
        fn metrics_prefix(&self) -> String {
            "testprefix".to_string()
        }
        fn metrics_hierarchy(&self) -> Vec<String> {
            vec![self.prefix.clone()]
        }
        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }

        // fn registry(&self) -> prometheus::Registry {
        //     let mut registry = self.drt.metrics_registries_by_prefix.lock().unwrap();
        //     registry
        //         .entry(self.metrics_prefix())
        //         .or_insert(prometheus::Registry::new())
        //         .clone()
        // }
    }

    #[tokio::test]
    async fn test_factory_methods_via_registry_trait() {
        // Setup real DRT and registry using the test-friendly constructor
        let rt = crate::Runtime::from_current().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::from_settings_without_discovery(rt)
                .await
                .unwrap(),
        );
        let registry = TestRegistry {
            drt,
            prefix: "testprefix".to_string(),
        };

        // Test Counter creation
        let counter = registry
            .create_counter("my_counter", "A test counter", &[])
            .unwrap();
        counter.inc_by(42.0);
        assert_eq!(counter.get() as u64, 42);
        println!("Counter value via MetricsRegistry: {}", counter.get());

        // Test Gauge creation
        let gauge = registry
            .create_gauge("my_gauge", "A test gauge", &[])
            .unwrap();
        gauge.set(123.45);
        assert_eq!(gauge.get(), 123.45);
        println!("Gauge value via MetricsRegistry: {}", gauge.get());

        // Test Histogram creation
        let histogram = registry
            .create_histogram("my_histogram", "A test histogram", &[])
            .unwrap();
        histogram.observe(1.5);
        histogram.observe(2.5);
        histogram.observe(3.5);
        // We can't assert the exact histogram buckets, but we can check the output contains the metric name
        println!("Histogram observations: 1.5, 2.5, 3.5");

        // Test Prometheus format output
        let prometheus_output = registry.encode_prometheus_str().unwrap();
        println!("Prometheus format output:");
        println!("{}", prometheus_output);

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
        assert!(prometheus_output.contains("testprefix_my_counter"));
        assert!(prometheus_output.contains("testprefix_my_gauge"));
        assert!(prometheus_output.contains("testprefix_my_histogram"));
        assert!(prometheus_output.contains("# HELP testprefix_my_counter A test counter"));
        assert!(prometheus_output.contains("# HELP testprefix_my_gauge A test gauge"));
        assert!(prometheus_output.contains("# HELP testprefix_my_histogram A test histogram"));

        println!("All metric types test passed!");
    }
}

#[cfg(test)]
mod test_runtime_and_namespace_registry_trait {
    use super::*;
    use std::sync::Arc;

    /// Test registry representing a runtime-level metrics registry
    struct MyRuntimeRegistry {
        drt: Arc<crate::DistributedRuntime>,
    }

    impl MetricsRegistry for MyRuntimeRegistry {
        fn metrics_prefix(&self) -> String {
            "runtime".to_string()
        }

        fn metrics_hierarchy(&self) -> Vec<String> {
            vec![self.metrics_prefix()]
        }

        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }
    }

    /// Test registry representing a namespace-level metrics registry
    struct MyNamespaceRegistry {
        drt: Arc<crate::DistributedRuntime>,
    }

    impl MetricsRegistry for MyNamespaceRegistry {
        fn metrics_prefix(&self) -> String {
            "runtime_namespace".to_string()
        }

        fn metrics_hierarchy(&self) -> Vec<String> {
            vec!["runtime".to_string(), self.metrics_prefix()]
        }

        fn drt(&self) -> &crate::DistributedRuntime {
            &self.drt
        }
    }

    #[tokio::test]
    async fn test_runtime_and_namespace_registry_trait() {
        // Setup real DRT
        let rt = crate::Runtime::from_current().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::from_settings_without_discovery(rt)
                .await
                .unwrap(),
        );

        // Create runtime-level registry
        let runtime_registry = MyRuntimeRegistry { drt: drt.clone() };

        // Create namespace-level registry
        let namespace_registry = MyNamespaceRegistry { drt: drt.clone() };

        // Create metrics using factory methods and increment some values
        let runtime_counter = runtime_registry
            .create_counter("total_requests", "Total requests across all runtime", &[])
            .unwrap();
        runtime_counter.inc_by(100.0);

        let namespace_gauge = namespace_registry
            .create_gauge(
                "active_connections",
                "Active connections for this namespace",
                &[],
            )
            .unwrap();
        namespace_gauge.set(25.0);

        // Test Prometheus format output for both registries
        let runtime_output = runtime_registry.encode_prometheus_str().unwrap();
        let namespace_output = namespace_registry.encode_prometheus_str().unwrap();

        println!("Runtime Prometheus output:");
        println!("{}", runtime_output);

        println!("Namespace Prometheus output:");
        println!("{}", namespace_output);

        // Verify exact content in runtime output (this includes ALL)
        let expected_runtime_output = "\
# HELP runtime_namespace__active_connections Active connections for this namespace\n\
# TYPE runtime_namespace__active_connections gauge\n\
runtime_namespace__active_connections 25\n\
# HELP runtime__total_requests Total requests across all runtime\n\
# TYPE runtime__total_requests counter\n\
runtime__total_requests 100\n";
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
