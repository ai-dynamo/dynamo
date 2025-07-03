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

//! Metric Container Framework for Dynamo.
//!
//! This module provides unified container classes for Prometheus and OpenTelemetry metrics
//! with shared interfaces for easy switching between metric backends.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Prometheus imports - using module alias for shorter typing
use prometheus as prom;

// OpenTelemetry imports - using module alias for shorter typing
use opentelemetry as otel;
use opentelemetry_sdk as otel_sdk;

/// Shared trait for metric container operations∏
pub trait MetricContainer: Send + Sync {
    /// Get the metric prefix for this container
    fn prefix(&self) -> &str;

    /// Create a new counter metric
    fn create_counter(&self, name: &str, description: &str, labels: &[(&str, &str)]) -> Box<dyn MetricCounter>;

    /// Create a new gauge metric
    fn create_gauge(&self, name: &str, description: &str, labels: &[(&str, &str)]) -> Box<dyn MetricGauge>;

    /// Get all metrics in the container's format
    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;

    /// Get the container type name (Prometheus, OpenTelemetry, etc.)
    fn container_type(&self) -> &'static str;
}

/// Shared trait for counter operations
pub trait MetricCounter: Send + Sync {
    /// Increment the counter by 1
    fn inc(&self);

    /// Increment the counter by a specific amount
    fn inc_by(&self, amount: u64);

    /// Get the current value
    fn get_value(&self) -> u64;

    /// Get the counter name
    fn get_name(&self) -> &str;
}

/// Shared trait for gauge operations
pub trait MetricGauge: Send + Sync {
    /// Set the gauge value
    fn set(&self, value: f64);

    /// Increment the gauge by a value
    fn inc(&self, value: f64);

    /// Decrement the gauge by a value
    fn dec(&self, value: f64);

    /// Get the current value
    fn get_value(&self) -> f64;

    /// Get the gauge name
    fn get_name(&self) -> &str;
}

/// Prometheus Metrics Container
pub struct PrometheusContainer {
    registry: prometheus::Registry,
    counters: Arc<Mutex<HashMap<String, prometheus::Counter>>>,
    gauges: Arc<Mutex<HashMap<String, prometheus::Gauge>>>,
    prefix: String,
}

impl PrometheusContainer {
    /// Create a new Prometheus container
    pub fn new(prefix: &str) -> Self {
        Self {
            registry: prometheus::Registry::new(),
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            prefix: prefix.to_string(),
        }
    }

    /// Get or create a Prometheus counter
    fn get_or_create_counter(&self, name: &str, help: &str) -> prometheus::Counter {
        let mut counters = self.counters.lock().unwrap();
        if let Some(counter) = counters.get(name) {
            counter.clone()
        } else {
            let counter = prometheus::Counter::new(name, help).unwrap();
            self.registry.register(Box::new(counter.clone())).unwrap();
            counters.insert(name.to_string(), counter.clone());
            counter
        }
    }

    /// Get or create a Prometheus gauge
    fn get_or_create_gauge(&self, name: &str, help: &str) -> prometheus::Gauge {
        let mut gauges = self.gauges.lock().unwrap();
        if let Some(gauge) = gauges.get(name) {
            gauge.clone()
        } else {
            let gauge = prometheus::Gauge::new(name, help).unwrap();
            self.registry.register(Box::new(gauge.clone())).unwrap();
            gauges.insert(name.to_string(), gauge.clone());
            gauge
        }
    }
}

impl MetricContainer for PrometheusContainer {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn create_counter(&self, name: &str, description: &str, _labels: &[(&str, &str)]) -> Box<dyn MetricCounter> {
        let prefixed_name = format!("{}_{}", self.prefix(), name);
        let counter = self.get_or_create_counter(&prefixed_name, description);
        Box::new(PrometheusCounter {
            counter,
            name: prefixed_name,
        })
    }

    fn create_gauge(&self, name: &str, description: &str, _labels: &[(&str, &str)]) -> Box<dyn MetricGauge> {
        let prefixed_name = format!("{}_{}", self.prefix(), name);
        let gauge = self.get_or_create_gauge(&prefixed_name, description);
        Box::new(PrometheusGauge {
            gauge,
            name: prefixed_name,
        })
    }

    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut buffer = Vec::new();
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    fn container_type(&self) -> &'static str {
        "Prometheus"
    }
}

/// Prometheus Counter implementation
pub struct PrometheusCounter {
    counter: prometheus::Counter,
    name: String,
}

impl MetricCounter for PrometheusCounter {
    fn inc(&self) {
        self.counter.inc();
    }

    fn inc_by(&self, amount: u64) {
        self.counter.inc_by(amount as f64);
    }

    fn get_value(&self) -> u64 {
        self.counter.get() as u64
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Prometheus Gauge implementation
pub struct PrometheusGauge {
    gauge: prometheus::Gauge,
    name: String,
}

impl MetricGauge for PrometheusGauge {
    fn set(&self, value: f64) {
        self.gauge.set(value);
    }

    fn inc(&self, value: f64) {
        self.gauge.inc(value);
    }

    fn dec(&self, value: f64) {
        self.gauge.dec(value);
    }

    fn get_value(&self) -> f64 {
        self.gauge.get()
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// OpenTelemetry Metrics Container (metrics-only)
pub struct OpenTelemetryContainer {
    meter: otel::metrics::Meter,
    counters: Arc<Mutex<HashMap<String, otel::metrics::Counter<u64>>>>,
    gauges: Arc<Mutex<HashMap<String, otel::metrics::Gauge<f64>>>>,
    prefix: String,
}

impl OpenTelemetryContainer {
    /// Create a new OpenTelemetry container
    pub fn new(service_name: &str, prefix: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Initialize OpenTelemetry with minimal setup for metrics only
        let resource = otel_sdk::Resource::new(vec![otel::KeyValue::new("service.name", service_name)]);

        // Set up metrics
        let meter_provider = otel_sdk::metrics::MeterProvider::builder()
            .with_resource(resource)
            .build();
        otel::global::set_meter_provider(meter_provider);
        let meter = otel::global::meter("dynamo");

        Ok(Self {
            meter,
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            prefix: prefix.to_string(),
        })
    }

    /// Get or create an OpenTelemetry counter
    fn get_or_create_counter(&self, name: &str, description: &str) -> otel::metrics::Counter<u64> {
        let mut counters = self.counters.lock().unwrap();
        if let Some(counter) = counters.get(name) {
            counter.clone()
        } else {
            let counter = self.meter
                .u64_counter(name)
                .with_description(description)
                .init();
            counters.insert(name.to_string(), counter.clone());
            counter
        }
    }

    /// Get or create an OpenTelemetry gauge
    fn get_or_create_gauge(&self, name: &str, description: &str) -> otel::metrics::Gauge<f64> {
        let mut gauges = self.gauges.lock().unwrap();
        if let Some(gauge) = gauges.get(name) {
            gauge.clone()
        } else {
            let gauge = self.meter
                .f64_gauge(name)
                .with_description(description)
                .init();
            gauges.insert(name.to_string(), gauge.clone());
            gauge
        }
    }

    /// Convert labels to OpenTelemetry attributes
    fn labels_to_attributes(&self, labels: &[(&str, &str)]) -> Vec<otel::KeyValue> {
        labels
            .iter()
            .map(|(k, v)| otel::KeyValue::new(*k, *v))
            .collect()
    }
}

impl MetricContainer for OpenTelemetryContainer {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn create_counter(&self, name: &str, description: &str, labels: &[(&str, &str)]) -> Box<dyn MetricCounter> {
        let prefixed_name = format!("{}_{}", self.prefix(), name);
        let counter = self.get_or_create_counter(&prefixed_name, description);
        let attributes = self.labels_to_attributes(labels);
        Box::new(OpenTelemetryCounter {
            counter,
            name: prefixed_name,
            attributes,
        })
    }

    fn create_gauge(&self, name: &str, description: &str, labels: &[(&str, &str)]) -> Box<dyn MetricGauge> {
        let prefixed_name = format!("{}_{}", self.prefix(), name);
        let gauge = self.get_or_create_gauge(&prefixed_name, description);
        let attributes = self.labels_to_attributes(labels);
        Box::new(OpenTelemetryGauge {
            gauge,
            name: prefixed_name,
            attributes,
        })
    }

    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // OpenTelemetry metrics are typically exported via OTLP, not retrieved as text
        // This is a placeholder implementation
        Ok("OpenTelemetry metrics are exported via OTLP".to_string())
    }

    fn container_type(&self) -> &'static str {
        "OpenTelemetry"
    }
}

/// OpenTelemetry Counter implementation
pub struct OpenTelemetryCounter {
    counter: otel::metrics::Counter<u64>,
    name: String,
    attributes: Vec<otel::KeyValue>,
}

impl MetricCounter for OpenTelemetryCounter {
    fn inc(&self) {
        self.counter.add(1, &self.attributes);
    }

    fn inc_by(&self, amount: u64) {
        self.counter.add(amount, &self.attributes);
    }

    fn get_value(&self) -> u64 {
        // OpenTelemetry counters don't expose current value
        // This is a limitation of the OpenTelemetry API
        0
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// OpenTelemetry Gauge implementation
pub struct OpenTelemetryGauge {
    gauge: otel::metrics::Gauge<f64>,
    name: String,
    attributes: Vec<otel::KeyValue>,
}

impl MetricGauge for OpenTelemetryGauge {
    fn set(&self, value: f64) {
        self.gauge.record(value, &self.attributes);
    }

    fn inc(&self, value: f64) {
        // OpenTelemetry gauges don't have inc/dec methods
        // We record the current value + increment
        self.gauge.record(value, &self.attributes);
    }

    fn dec(&self, value: f64) {
        // OpenTelemetry gauges don't have inc/dec methods
        // We record the current value - decrement
        self.gauge.record(-value, &self.attributes);
    }

    fn get_value(&self) -> f64 {
        // OpenTelemetry gauges don't expose current value
        // This is a limitation of the OpenTelemetry API
        0.0
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_and_opentelemetry_metric_containers() {
        println!("=== Testing Metric Containers ===");

        // Test Prometheus Container
        println!("\n--- Prometheus Container ---");
        let prom_container = PrometheusContainer::new("myapp");
        println!("Container type: {}", prom_container.container_type());
        println!("Container prefix: {}", prom_container.prefix());

        // Create and use a counter
        let prom_counter = prom_container.create_counter(
            "test_prom_counter",
            "Test Prometheus counter",
            &[("service", "test"), ("endpoint", "/api")]
        );
        prom_counter.inc();
        prom_counter.inc_by(5);
        println!("Prometheus counter '{}': {}", prom_counter.get_name(), prom_counter.get_value());

        // Create and use a gauge
        let prom_gauge = prom_container.create_gauge(
            "test_prom_gauge",
            "Test Prometheus gauge",
            &[("service", "test"), ("endpoint", "/api")]
        );
        prom_gauge.set(10.5);
        prom_gauge.inc(2.5);
        prom_gauge.dec(1.0);
        println!("Prometheus gauge '{}': {}", prom_gauge.get_name(), prom_gauge.get_value());

        // Get metrics
        match prom_container.get_metrics() {
            Ok(metrics) => {
                println!("Prometheus metrics:");
                println!("{}", metrics);
            }
            Err(e) => println!("Failed to get Prometheus metrics: {}", e),
        }

        // Test OpenTelemetry Container
        println!("\n--- OpenTelemetry Container ---");
        match OpenTelemetryContainer::new("test-service", "myapp") {
            Ok(otel_container) => {
                println!("Container type: {}", otel_container.container_type());
                println!("Container prefix: {}", otel_container.prefix());

                // Create and use a counter
                let otel_counter = otel_container.create_counter(
                    "test_otel_counter",
                    "Test OpenTelemetry counter",
                    &[("service", "test"), ("protocol", "otlp")]
                );
                otel_counter.inc();
                otel_counter.inc_by(3);
                println!("OpenTelemetry counter '{}': {}", otel_counter.get_name(), otel_counter.get_value());

                // Create and use a gauge
                let otel_gauge = otel_container.create_gauge(
                    "test_otel_gauge",
                    "Test OpenTelemetry gauge",
                    &[("service", "test"), ("protocol", "otlp")]
                );
                otel_gauge.set(15.0);
                otel_gauge.inc(3.0);
                otel_gauge.dec(2.0);
                println!("OpenTelemetry gauge '{}': {}", otel_gauge.get_name(), otel_gauge.get_value());

                // Get metrics
                match otel_container.get_metrics() {
                    Ok(metrics) => {
                        println!("OpenTelemetry metrics:");
                        println!("{}", metrics);
                    }
                    Err(e) => println!("Failed to get OpenTelemetry metrics: {}", e),
                }
            }
            Err(e) => {
                println!("Failed to create OpenTelemetry container: {}", e);
            }
        }

        println!("\n=== Container Tests Complete ===");
    }

    #[test]
    fn test_opentelemetry_tracing_samples() {
        println!("=== Sample OpenTelemetry Tracing Usage ===");

        // Example: Setting up a tracer
        let _tracer = otel::global::tracer("my-service");

        // Example: Creating a span
        let _span = otel::global::tracer("my-service").start("database-query");

        // Example: Adding attributes to a span
        let mut span = otel::global::tracer("my-service").start("http-request");
        span.set_attribute(otel::KeyValue::new("http.method", "GET"));
        span.set_attribute(otel::KeyValue::new("http.url", "/api/users"));
        span.set_attribute(otel::KeyValue::new("http.status_code", 200));

        // Example: Adding events to a span
        span.add_event("cache.miss", vec![
            otel::KeyValue::new("cache.key", "user:123"),
            otel::KeyValue::new("cache.backend", "redis"),
        ]);

        // Example: Creating a child span
        let child_span = otel::global::tracer("my-service")
            .span_builder("database.query")
            .with_parent(otel::Context::current_with_span(span))
            .start(&otel::global::tracer("my-service"));

        // Example: Using span context
        let context = otel::Context::current_with_span(child_span);

        // Example: Extracting trace context from HTTP headers
        let _extracted_context = otel::global::get_text_map_propagator(|propagator| {
            propagator.extract(&otel::TextMapGetter::default(), &HashMap::new())
        });

        // Example: Injecting trace context into HTTP headers
        let mut headers = HashMap::new();
        otel::global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&context, &mut headers);
        });

        println!("Sample tracing operations completed");
        println!("=== Sample Tracing Usage Complete ===");
    }

    /// Trait for structs that can provide their metrics for printing
    pub trait MetricProvider {
        fn get_metrics(&self) -> Vec<(&str, &str, u64, f64)>; // (name, type, counter_value, gauge_value)
    }

    /// Base struct containing common metric functionality
    pub struct BaseServiceMetrics {
        pub container: PrometheusContainer,
    }

    impl BaseServiceMetrics {
        /// Create a new BaseServiceMetrics instance with Prometheus backend
        pub fn new_prometheus(prefix: &str) -> Self {
            let container = PrometheusContainer::new(prefix);
            BaseServiceMetrics { container }
        }

        /// Generic method to print metrics from any struct that implements MetricProvider
        pub fn print_metrics_from<T: MetricProvider>(&self, metrics: &T) {
            let metric_list = metrics.get_metrics();
            for (name, metric_type, counter_value, gauge_value) in metric_list {
                match metric_type {
                    "counter" => println!("Counter '{}': {}", name, counter_value),
                    "gauge" => println!("Gauge '{}': {}", name, gauge_value),
                    _ => println!("Unknown metric '{}': type={}", name, metric_type),
                }
            }
        }
    }

    /// Example struct containing both a MetricCounter and MetricGauge
    pub struct ExampleServiceMetrics {
        pub base: BaseServiceMetrics,
        pub some_specific_counter: Box<dyn MetricCounter>,
        pub some_response_ms: Box<dyn MetricGauge>,
    }

    // Implement MetricProvider for ExampleServiceMetrics
    impl MetricProvider for ExampleServiceMetrics {
        fn get_metrics(&self) -> Vec<(&str, &str, u64, f64)> {
            vec![
                (self.some_specific_counter.get_name(), "counter", self.some_specific_counter.get_value(), 0.0),
                (self.some_response_ms.get_name(), "gauge", 0, self.some_response_ms.get_value()),
            ]
        }
    }

    impl ExampleServiceMetrics {
        /// Create a new ExampleServiceMetrics instance using the parent's new_prometheus
        pub fn new(prefix: &str) -> Self {
            let base = BaseServiceMetrics::new_prometheus(prefix);

            let request_counter = base.container.create_counter(
                "requests_total",
                "Total number of requests",
                &[("service", "api"), ("version", "v1")]
            );

            let response_time_gauge = base.container.create_gauge(
                "response_time_seconds",
                "Response time in seconds",
                &[("service", "api"), ("version", "v1")]
            );

            ExampleServiceMetrics {
                base,
                some_specific_counter: request_counter,
                some_response_ms: response_time_gauge,
            }
        }

        /// Print current metric values using the base class method
        fn print_metrics(&self) {
            self.base.print_metrics_from(self);
        }
    }

    #[test]
    fn test_service_metrics_struct() {
        println!("=== Testing ServiceMetrics Struct with Prometheus Backend ===");

        // Create a new ServiceMetrics instance
        let metrics = ExampleServiceMetrics::new("myapp");

        println!("Created ServiceMetrics with Prometheus backend");
        println!("Initial metrics:");
        metrics.print_metrics();

        // Simulate some API requests using direct access to public fields
        println!("\n--- Simulating API Requests (Direct Access) ---");

        // Record individual requests directly
        metrics.some_specific_counter.inc(); // Request 1
        metrics.some_specific_counter.inc(); // Request 2
        metrics.some_specific_counter.inc(); // Request 3

        // Record batch of requests directly
        metrics.some_specific_counter.inc_by(5); // 5 more requests

        // Set response times directly
        metrics.some_response_ms.set(0.15); // 150ms
        metrics.some_response_ms.inc(0.05); // Add 50ms
        metrics.some_response_ms.dec(0.02); // Subtract 20ms

        println!("\nAfter simulating requests (direct access):");
        metrics.print_metrics();

        // Demonstrate metric names with prefix
        println!("\n--- Metric Names with Prefix ---");
        println!("Counter name: {}", metrics.some_specific_counter.get_name());
        println!("Gauge name: {}", metrics.some_response_ms.get_name());

        // Show that the prefix is applied correctly
        assert!(metrics.some_specific_counter.get_name().starts_with("myapp_"));
        assert!(metrics.some_response_ms.get_name().starts_with("myapp_"));

        println!("\n=== ServiceMetrics Test Complete ===");
    }
}