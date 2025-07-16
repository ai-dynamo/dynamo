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

/// This trait should be implemented by all metric registries, including Prometheus, Envy, OpenTelemetry, and others.
/// It offers a unified interface for creating and managing metrics, organizing sub-registries, and
/// generating output in Prometheus text format.
pub trait MetricsRegistry: Send + Sync {
    /// Get the metric prefix for this registry
    fn metrics_prefix(&self) -> String;
    fn metrics_hierarchy(&self) -> Vec<String>;

    fn registry(&self) -> prometheus::Registry {
        let mut registry = self.drt().metrics_registries_by_prefix.lock().unwrap();
        registry
            .entry(self.metrics_prefix())
            .or_insert(prometheus::Registry::new())
            .clone()
    }

    fn register_metric(&self, metric: Box<dyn prometheus::core::Collector>) {
        let mut registry = self.drt().metrics_registries_by_prefix.lock().unwrap();

        for prefix in self.metrics_hierarchy() {
            registry
                .entry(prefix)
                .or_insert(prometheus::Registry::new())
                .register(metric.clone());
        }
    }

    /// Create a new counter metric
    fn create_counter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<prometheus::Counter> {
        let prefix = if self.metrics_prefix().is_empty() {
            name.to_string()
        } else {
            format!("{}_{}", self.metrics_prefix(), name)
        };

        let counter = prometheus::Counter::new(&prefix, description)?;

        let registry = self.registry();

        let collector = Box::new(counter.clone());

        registry.register(collector.clone());

        Ok(counter)
    }

    // /// Create a new gauge metric
    // fn create_gauge(
    //     &self,
    //     name: &str,
    //     description: &str,
    //     labels: &[(&str, &str)],
    // ) -> Result<Box<dyn MetricGauge>, Box<dyn std::error::Error + Send + Sync>>;

    // /// Create a new histogram metric
    // fn create_histogram(
    //     &self,
    //     name: &str,
    //     description: &str,
    //     labels: &[(&str, &str)],
    // ) -> Result<Box<dyn MetricHistogram>, Box<dyn std::error::Error + Send + Sync>>;

    // /// Get parent metrics only (without children)
    // fn root_prometheus_format_str(
    //     &self,
    // ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;

    // /// Get combined metrics from parent and all children recursively
    // fn all_prometheus_format_str(
    //     &self,
    // ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    //     self.update()?;
    //     let mut result = self.root_prometheus_format_str()?;

    //     // Recursively get metrics from all child registries
    //     for child in self.get_children_registries() {
    //         match child.all_prometheus_format_str() {
    //             Ok(child_metrics) => {
    //                 if !result.is_empty() && !result.ends_with('\n') {
    //                     result.push('\n');
    //                 }
    //                 result.push_str(&child_metrics);
    //             }
    //             Err(e) => {
    //                 eprintln!("Failed to get metrics from child registry: {}", e);
    //             }
    //         }
    //     }

    //     Ok(result)
    // }

    // /// Iterate over all created metrics
    // fn for_each_metric(&self, f: &mut dyn FnMut(&dyn Metric));

    // /// Run all registered update functions
    // fn update(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

// Prometheus Registry
// pub struct PrometheusRegistry {
//     prefix: String,
//     metrics: Arc<Mutex<std::collections::HashMap<String, Box<dyn Metric>>>>,
//     children_registries: Arc<Mutex<Vec<Arc<dyn MetricsRegistry>>>>,
//     update_functions: Arc<Mutex<Vec<Box<dyn Fn() + Send + Sync>>>>,
//     prom_registry: prometheus::Registry,
// }

// impl PrometheusRegistry {
//     /// Create a new Prometheus registry
//     pub fn new(prefix: &str) -> Self {
//         Self {
//             prefix: prefix.to_string(),
//             metrics: Arc::new(Mutex::new(std::collections::HashMap::new())),
//             children_registries: Arc::new(Mutex::new(Vec::new())),
//             update_functions: Arc::new(Mutex::new(Vec::new())),
//             prom_registry: prometheus::Registry::new(),
//         }
//     }
// }

// impl MetricsRegistry for PrometheusRegistry {
//     fn prefix(&self) -> &str {
//         &self.prefix
//     }

//     fn get_children_registries(&self) -> Vec<Arc<dyn MetricsRegistry>> {
//         self.children_registries.lock().unwrap().clone()
//     }

//     fn create_child_registry(&self, prefix: &str) -> Arc<dyn MetricsRegistry> {
//         let child_prefix = format!("{}_{}", self.prefix, prefix);
//         let child = Arc::new(PrometheusRegistry::new(&child_prefix));
//         self.children_registries.lock().unwrap().push(child.clone());
//         child
//     }

//     fn create_counter(
//         &self,
//         name: &str,
//         description: &str,
//         _labels: &[(&str, &str)],
//     ) -> Result<Box<dyn MetricCounter>, Box<dyn std::error::Error + Send + Sync>> {
//         let prefixed_name = format!("{}__{}", self.prefix(), name);

//         // Check if metric name is already registered and add to metrics
//         let mut metrics = self.metrics.lock().unwrap();
//         if metrics.contains_key(&prefixed_name) {
//             return Err(format!("Counter with name '{}' already exists", prefixed_name).into());
//         }

//         let prom_counter = prometheus::Counter::new(&prefixed_name, description)
//             .map_err(|e| format!("Failed to create counter '{}': {}", prefixed_name, e))?;
//         self.prom_registry
//             .register(Box::new(prom_counter.clone()))
//             .map_err(|e| format!("Failed to register counter '{}': {}", prefixed_name, e))?;

//         let metric_counter = PrometheusCounter {
//             prom_counter,
//             name: prefixed_name.clone(),
//             description: description.to_string(),
//         };

//         // Add to our metrics HashMap
//         metrics.insert(
//             prefixed_name,
//             Box::new(metric_counter.clone()) as Box<dyn Metric>,
//         );
//         drop(metrics); // Release lock early

//         Ok(Box::new(metric_counter))
//     }

//     fn create_gauge(
//         &self,
//         name: &str,
//         description: &str,
//         _labels: &[(&str, &str)],
//     ) -> Result<Box<dyn MetricGauge>, Box<dyn std::error::Error + Send + Sync>> {
//         let prefixed_name = format!("{}__{}", self.prefix(), name);

//         // Check if metric name is already registered and add to metrics
//         let mut metrics = self.metrics.lock().unwrap();
//         if metrics.contains_key(&prefixed_name) {
//             return Err(format!("Gauge with name '{}' already exists", prefixed_name).into());
//         }

//         let prom_gauge = prometheus::Gauge::new(&prefixed_name, description)
//             .map_err(|e| format!("Failed to create gauge '{}': {}", prefixed_name, e))?;
//         self.prom_registry
//             .register(Box::new(prom_gauge.clone()))
//             .map_err(|e| format!("Failed to register gauge '{}': {}", prefixed_name, e))?;

//         let metric_gauge = PrometheusGauge {
//             prom_gauge,
//             name: prefixed_name.clone(),
//             description: description.to_string(),
//         };

//         // Add to our metrics HashMap
//         metrics.insert(
//             prefixed_name,
//             Box::new(metric_gauge.clone()) as Box<dyn Metric>,
//         );
//         drop(metrics); // Release lock early

//         Ok(Box::new(metric_gauge))
//     }

//     fn create_histogram(
//         &self,
//         name: &str,
//         description: &str,
//         _labels: &[(&str, &str)],
//     ) -> Result<Box<dyn MetricHistogram>, Box<dyn std::error::Error + Send + Sync>> {
//         let prefixed_name = format!("{}__{}", self.prefix(), name);

//         // Check if metric name is already registered and add to metrics
//         let mut metrics = self.metrics.lock().unwrap();
//         if metrics.contains_key(&prefixed_name) {
//             return Err(format!("Histogram with name '{}' already exists", prefixed_name).into());
//         }

//         let prom_histogram = prometheus::Histogram::with_opts(prometheus::HistogramOpts::new(
//             &prefixed_name,
//             description,
//         ))
//         .map_err(|e| format!("Failed to create histogram '{}': {}", prefixed_name, e))?;
//         self.prom_registry
//             .register(Box::new(prom_histogram.clone()))
//             .map_err(|e| format!("Failed to register histogram '{}': {}", prefixed_name, e))?;

//         let metric_histogram = PrometheusHistogram {
//             prom_histogram,
//             name: prefixed_name.clone(),
//             description: description.to_string(),
//         };

//         // Add to our metrics HashMap
//         metrics.insert(
//             prefixed_name,
//             Box::new(metric_histogram.clone()) as Box<dyn Metric>,
//         );
//         drop(metrics); // Release lock early

//         Ok(Box::new(metric_histogram))
//     }

//     fn root_prometheus_format_str(
//         &self,
//     ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
//         let mut buffer = Vec::new();
//         let encoder = prometheus::TextEncoder::new();
//         let metric_families = self.prom_registry.gather();
//         encoder.encode(&metric_families, &mut buffer)?;
//         Ok(String::from_utf8(buffer)?)
//     }

//     fn for_each_metric(&self, f: &mut dyn FnMut(&dyn Metric)) {
//         let metrics = self.metrics.lock().unwrap();
//         for metric in metrics.values() {
//             f(metric.as_ref());
//         }
//     }

//     fn update(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         let update_functions = self.update_functions.lock().unwrap();
//         for func in update_functions.iter() {
//             func();
//         }
//         Ok(())
//     }
// }

// // ------------------------------------------------------------------------------------------------
// // Metric Traits
// // ------------------------------------------------------------------------------------------------

// /// Generic trait for all metric types
// pub trait Metric: Send + Sync + Any {
//     /// Get the metric name
//     fn get_name(&self) -> &str;

//     /// Get the metric type as a string
//     fn metric_type(&self) -> &'static str;

//     /// Get the metric description
//     fn description(&self) -> &str;
// }

// /// Shared trait for counter operations
// pub trait MetricCounter: Metric {
//     /// Increment the counter by 1
//     fn inc(&self);

//     /// Increment the counter by a specific amount
//     fn inc_by(&self, amount: u64);

//     /// Get the current value
//     fn get_value(&self) -> u64;
// }

// /// Shared trait for gauge operations
// pub trait MetricGauge: Metric {
//     /// Set the gauge value
//     fn set(&self, value: f64);

//     /// Increment the gauge by a value
//     fn inc(&self, value: f64);

//     /// Decrement the gauge by a value
//     fn dec(&self, value: f64);

//     /// Get the current value
//     fn get_value(&self) -> f64;
// }

// /// Shared trait for histogram operations
// pub trait MetricHistogram: Metric {
//     /// Observe a value in the histogram
//     fn observe(&self, value: f64);

//     /// Get the total count of observations
//     fn get_count(&self) -> u64;

//     /// Get the sum of all observed values
//     fn get_sum(&self) -> f64;
// }

// /// Prometheus Counter implementation
// #[derive(Clone)]
// pub struct PrometheusCounter {
//     prom_counter: prometheus::Counter,
//     name: String,
//     description: String,
// }

// impl Metric for PrometheusCounter {
//     fn get_name(&self) -> &str {
//         &self.name
//     }

//     fn metric_type(&self) -> &'static str {
//         COUNTER_METRIC_TYPE
//     }

//     fn description(&self) -> &str {
//         &self.description
//     }
// }

// impl MetricCounter for PrometheusCounter {
//     fn inc(&self) {
//         self.prom_counter.inc();
//     }

//     fn inc_by(&self, amount: u64) {
//         self.prom_counter.inc_by(amount as f64);
//     }

//     fn get_value(&self) -> u64 {
//         self.prom_counter.get() as u64
//     }
// }

// /// Prometheus Gauge implementation
// #[derive(Clone)]
// pub struct PrometheusGauge {
//     prom_gauge: prometheus::Gauge,
//     name: String,
//     description: String,
// }

// impl Metric for PrometheusGauge {
//     fn get_name(&self) -> &str {
//         &self.name
//     }

//     fn metric_type(&self) -> &'static str {
//         GAUGE_METRIC_TYPE
//     }

//     fn description(&self) -> &str {
//         &self.description
//     }
// }

// impl MetricGauge for PrometheusGauge {
//     fn set(&self, value: f64) {
//         self.prom_gauge.set(value);
//     }

//     fn inc(&self, value: f64) {
//         self.prom_gauge.add(value);
//     }

//     fn dec(&self, value: f64) {
//         self.prom_gauge.sub(value);
//     }

//     fn get_value(&self) -> f64 {
//         self.prom_gauge.get()
//     }
// }

// /// Prometheus Histogram implementation
// #[derive(Clone)]
// pub struct PrometheusHistogram {
//     prom_histogram: prometheus::Histogram,
//     name: String,
//     description: String,
// }

// impl Metric for PrometheusHistogram {
//     fn get_name(&self) -> &str {
//         &self.name
//     }

//     fn metric_type(&self) -> &'static str {
//         HISTOGRAM_METRIC_TYPE
//     }

//     fn description(&self) -> &str {
//         &self.description
//     }
// }

// impl MetricHistogram for PrometheusHistogram {
//     fn observe(&self, value: f64) {
//         self.prom_histogram.observe(value);
//     }

//     fn get_count(&self) -> u64 {
//         self.prom_histogram.get_sample_count() as u64
//     }

//     fn get_sum(&self) -> f64 {
//         self.prom_histogram.get_sample_sum()
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_prometheus_metric_registries() {
//         println!("=== Prometheus Registry Tests ===");

//         // Test Prometheus Registry
//         println!("\n--- Prometheus Registry ---");
//         let prom = PrometheusRegistry::new("myapp");
//         assert_eq!(prom.prefix(), "myapp", "The prefix should be 'myapp'");
//         println!("Registry prefix: {}", prom.prefix());

//         // Create and use a counter
//         let prom_counter = prom
//             .create_counter(
//                 "test_prom_counter",
//                 "Test Prometheus counter",
//                 &[("service", "test"), ("protocol", "prometheus")],
//             )
//             .unwrap();
//         prom_counter.inc();
//         prom_counter.inc_by(3);

//         // Add a value check
//         let counter_value = prom_counter.get_value();
//         assert_eq!(
//             counter_value, 4,
//             "Counter value should be 4 after incrementing by 1 and then by 3"
//         );
//         println!(
//             "Prometheus counter '{}': {} (type: {})",
//             prom_counter.get_name(),
//             prom_counter.get_value(),
//             prom_counter.metric_type()
//         );

//         // Create and use a gauge
//         let prom_gauge = prom
//             .create_gauge(
//                 "test_prom_gauge",
//                 "Test Prometheus gauge",
//                 &[("service", "test"), ("protocol", "prometheus")],
//             )
//             .unwrap();
//         prom_gauge.set(15.0);
//         prom_gauge.inc(3.0);
//         prom_gauge.dec(2.0);

//         // Add a value check
//         let gauge_value = prom_gauge.get_value();
//         assert_eq!(gauge_value, 16.0, "Gauge value should be 16.0 after setting to 15.0, incrementing by 3.0, and decrementing by 2.0");
//         println!(
//             "Prometheus gauge '{}': {} (type: {})",
//             prom_gauge.get_name(),
//             prom_gauge.get_value(),
//             prom_gauge.metric_type()
//         );

//         // Create and use a histogram
//         let prom_histogram = prom
//             .create_histogram(
//                 "test_prom_histogram",
//                 "Test Prometheus histogram",
//                 &[("service", "test"), ("protocol", "prometheus")],
//             )
//             .unwrap();
//         prom_histogram.observe(1.5);
//         prom_histogram.observe(2.3);
//         prom_histogram.observe(0.8);
//         prom_histogram.observe(3.1);

//         // Add a value check
//         let histogram_count = prom_histogram.get_count();
//         let histogram_sum = prom_histogram.get_sum();
//         println!(
//             "DEBUG: Expected sum: 7.7, Actual sum: {}, Difference: {}",
//             histogram_sum,
//             (histogram_sum - 7.7).abs()
//         );
//         assert_eq!(
//             histogram_count, 4,
//             "Histogram count should be 4 after observing 4 values"
//         );
//         assert!((histogram_sum - 7.7).abs() < 1e-9, "Histogram sum should be approximately 7.7 after observing values 1.5, 2.3, 0.8, and 3.1");
//         println!(
//             "Prometheus histogram '{}': count={}, sum={} (type: {})",
//             prom_histogram.get_name(),
//             prom_histogram.get_count(),
//             prom_histogram.get_sum(),
//             prom_histogram.metric_type()
//         );

//         println!("\n=== Registry Tests Complete ===");
//     }

//     #[test]
//     fn test_duplicate_metric_names() {
//         println!("=== Duplicate Metric Names Test ===");

//         let registry = PrometheusRegistry::new("test");

//         // Create a counter successfully
//         let counter1 = registry.create_counter("duplicate_test", "Test counter", &[]);
//         assert!(counter1.is_ok(), "First counter creation should succeed");

//         // Try to create another counter with the same name
//         let counter2 = registry.create_counter("duplicate_test", "Test counter", &[]);
//         assert!(counter2.is_err(), "Second counter creation should fail");
//         if let Err(e) = counter2 {
//             println!("Expected error for duplicate counter: {}", e);
//         }

//         // Create a gauge successfully
//         let gauge1 = registry.create_gauge("duplicate_gauge", "Test gauge", &[]);
//         assert!(gauge1.is_ok(), "First gauge creation should succeed");

//         // Try to create another gauge with the same name
//         let gauge2 = registry.create_gauge("duplicate_gauge", "Test gauge", &[]);
//         assert!(gauge2.is_err(), "Second gauge creation should fail");
//         if let Err(e) = gauge2 {
//             println!("Expected error for duplicate gauge: {}", e);
//         }

//         // Create a histogram successfully
//         let histogram1 = registry.create_histogram("duplicate_hist", "Test histogram", &[]);
//         assert!(
//             histogram1.is_ok(),
//             "First histogram creation should succeed"
//         );

//         // Try to create another histogram with the same name
//         let histogram2 = registry.create_histogram("duplicate_hist", "Test histogram", &[]);
//         assert!(histogram2.is_err(), "Second histogram creation should fail");
//         if let Err(e) = histogram2 {
//             println!("Expected error for duplicate histogram: {}", e);
//         }

//         println!("=== Duplicate Metric Names Test Complete ===");
//     }

//     #[test]
//     fn test_service_metrics() {
//         println!("=== Service Metrics with Traits Test ===");

//         // Define TestMetrics struct within the test
//         struct TestMetrics<R: MetricsRegistry> {
//             registry: Arc<R>,
//             pub request_counter: Box<dyn MetricCounter>,
//             pub active_requests_gauge: Box<dyn MetricGauge>,
//             pub request_duration_histogram: Box<dyn MetricHistogram>,
//         }

//         impl<R: MetricsRegistry> TestMetrics<R> {
//             /// Create a new TestMetrics instance using the metric registry
//             fn new(registry: Arc<R>) -> Self {
//                 // Create request counter
//                 let request_counter = registry
//                     .create_counter(
//                         "requests_total",
//                         "Total number of requests processed",
//                         &[("service", "registry")],
//                     )
//                     .unwrap();

//                 // Create active requests gauge
//                 let active_requests_gauge = registry
//                     .create_gauge(
//                         "active_requests",
//                         "Number of requests currently being processed",
//                         &[("service", "registry")],
//                     )
//                     .unwrap();

//                 // Create request duration histogram
//                 let request_duration_histogram = registry
//                     .create_histogram(
//                         "request_duration_seconds",
//                         "Request duration in seconds",
//                         &[("service", "registry")],
//                     )
//                     .unwrap();

//                 TestMetrics {
//                     registry: registry,
//                     request_counter: request_counter,
//                     active_requests_gauge: active_requests_gauge,
//                     request_duration_histogram: request_duration_histogram,
//                 }
//             }
//         }

//         // Create a new Prometheus registry
//         // Create a service metrics struct using the metric traits
//         let test_metrics =
//             TestMetrics::<PrometheusRegistry>::new(Arc::new(PrometheusRegistry::new("service")));
//         println!("Created TestMetrics with trait-based metrics");

//         // Simulate some request processing
//         println!("\n--- Simulating Request Processing ---");

//         // Simulate request start
//         test_metrics.request_counter.inc();
//         test_metrics.active_requests_gauge.inc(1.0);
//         println!(
//             "Request started - Counter: {}, Active: {}",
//             test_metrics.request_counter.get_value(),
//             test_metrics.active_requests_gauge.get_value()
//         );
//         assert_eq!(
//             test_metrics.request_counter.get_value(),
//             1,
//             "Should have processed 1 request"
//         );
//         assert_eq!(
//             test_metrics.active_requests_gauge.get_value(),
//             1.0,
//             "Should have 1 active request"
//         );

//         // Simulate some processing time
//         std::thread::sleep(std::time::Duration::from_millis(100));

//         // Simulate request completion
//         test_metrics.active_requests_gauge.dec(1.0);
//         let duration = 0.1; // Simulated duration
//         test_metrics.request_duration_histogram.observe(duration);
//         println!(
//             "Request completed - Counter: {}, Active: {}, Duration: {}s",
//             test_metrics.request_counter.get_value(),
//             test_metrics.active_requests_gauge.get_value(),
//             duration
//         );
//         assert_eq!(
//             test_metrics.active_requests_gauge.get_value(),
//             0.0,
//             "Should have no active requests"
//         );
//         assert_eq!(
//             test_metrics.request_duration_histogram.get_count(),
//             1,
//             "Should have 1 duration observation"
//         );
//         assert!(
//             (test_metrics.request_duration_histogram.get_sum() - 0.1).abs() < f64::EPSILON,
//             "Sum should be approximately 0.1"
//         );

//         // Simulate another request
//         test_metrics.request_counter.inc();
//         test_metrics.active_requests_gauge.inc(1.0);
//         test_metrics.active_requests_gauge.dec(1.0);
//         test_metrics.request_duration_histogram.observe(0.05);
//         println!(
//             "Second request completed - Counter: {}, Active: {}",
//             test_metrics.request_counter.get_value(),
//             test_metrics.active_requests_gauge.get_value()
//         );
//         assert_eq!(
//             test_metrics.request_counter.get_value(),
//             2,
//             "Should have processed 2 requests"
//         );
//         assert_eq!(
//             test_metrics.active_requests_gauge.get_value(),
//             0.0,
//             "Should have no active requests"
//         );
//         assert_eq!(
//             test_metrics.request_duration_histogram.get_count(),
//             2,
//             "Should have 2 duration observations"
//         );
//         assert!(
//             (test_metrics.request_duration_histogram.get_sum() - 0.15).abs() < f64::EPSILON,
//             "Sum should be approximately 0.15"
//         );

//         // Print final metrics
//         println!("\n--- Final Metrics ---");
//         println!(
//             "Request Counter: {} (type: {})",
//             test_metrics.request_counter.get_value(),
//             test_metrics.request_counter.metric_type()
//         );
//         println!(
//             "Active Requests: {} (type: {})",
//             test_metrics.active_requests_gauge.get_value(),
//             test_metrics.active_requests_gauge.metric_type()
//         );
//         println!(
//             "Request Duration - Count: {}, Sum: {} (type: {})",
//             test_metrics.request_duration_histogram.get_count(),
//             test_metrics.request_duration_histogram.get_sum(),
//             test_metrics.request_duration_histogram.metric_type()
//         );

//         println!("\n=== Service Metrics Test Complete ===");
//     }

//     #[test]
//     fn test_hierarchical_metrics() {
//         println!("=== Hierarchical Metrics Test ===");

//         // Define ParentMetrics struct
//         struct ParentMetrics<R: MetricsRegistry> {
//             registry: Arc<R>,
//             pub parent_counter: Box<dyn MetricCounter>,
//         }

//         impl<R: MetricsRegistry> ParentMetrics<R> {
//             fn new(registry: Arc<R>) -> Self {
//                 let parent_counter = registry
//                     .create_counter(
//                         "requests",
//                         "Total number of parent requests",
//                         &[("service", "parent")],
//                     )
//                     .unwrap();

//                 ParentMetrics {
//                     registry,
//                     parent_counter,
//                 }
//             }
//         }

//         // Define ChildMetrics struct
//         struct ChildMetrics {
//             registry: Arc<dyn MetricsRegistry>,
//             pub child_histogram: Box<dyn MetricHistogram>,
//         }

//         impl ChildMetrics {
//             fn new(registry: Arc<dyn MetricsRegistry>) -> Self {
//                 let child_histogram = registry
//                     .create_histogram(
//                         "requests",
//                         "Total number of child requests",
//                         &[("service", "child")],
//                     )
//                     .unwrap();

//                 ChildMetrics {
//                     registry,
//                     child_histogram,
//                 }
//             }
//         }

//         // Create parent registry
//         let parent_registry = PrometheusRegistry::new("parent");

//         // Create child registry and add it to parent
//         let child_registry = parent_registry.create_child_registry("child");
//         let child_metrics = ChildMetrics::new(child_registry.clone());

//         // Now wrap parent in Arc after adding children
//         let parent_registry = Arc::new(parent_registry);
//         let parent_metrics = ParentMetrics::new(parent_registry.clone());

//         // Simulate some metrics
//         parent_metrics.parent_counter.inc();
//         parent_metrics.parent_counter.inc_by(2);
//         child_metrics.child_histogram.observe(1.5);
//         child_metrics.child_histogram.observe(2.5);

//         // Verify metrics
//         assert_eq!(
//             parent_metrics.parent_counter.get_value(),
//             3,
//             "Parent counter should be 3"
//         );
//         assert_eq!(
//             child_metrics.child_histogram.get_count(),
//             2,
//             "Child histogram should have 2 observations"
//         );

//         println!(
//             "Parent counter: {} (type: {})",
//             parent_metrics.parent_counter.get_value(),
//             parent_metrics.parent_counter.metric_type()
//         );
//         println!(
//             "Child histogram: count={}, sum={} (type: {})",
//             child_metrics.child_histogram.get_count(),
//             child_metrics.child_histogram.get_sum(),
//             child_metrics.child_histogram.metric_type()
//         );

//         // Test hierarchical metrics output
//         match parent_registry.all_prometheus_format_str() {
//             Ok(metrics) => {
//                 println!("\n--- Hierarchical Prometheus Metrics ---");
//                 println!("{}", metrics);

//                 // Check that the output contains expected content
//                 assert!(
//                     metrics.contains("parent__requests 3"),
//                     "Should contain parent counter value"
//                 );
//                 assert!(
//                     metrics.contains("parent_child__requests_bucket"),
//                     "Should contain child histogram bucket"
//                 );
//                 assert!(
//                     metrics.contains("parent_child__requests_sum 4"),
//                     "Should contain child histogram sum"
//                 );
//                 assert!(
//                     metrics.contains("parent_child__requests_count 2"),
//                     "Should contain child histogram count"
//                 );

//                 println!("✓ All Prometheus format checks passed");
//             }
//             Err(e) => {
//                 println!("Failed to get hierarchical metrics: {}", e);
//                 panic!("Failed to get hierarchical metrics: {}", e);
//             }
//         }

//         println!("=== Hierarchical Metrics Test Complete ===");
//     }
// }
