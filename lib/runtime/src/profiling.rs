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

//! Common Observability Framework for Dynamo.
//!
//! The Common Observability Framework in Dynamo is designed to improve system visibility
//! and support greater fault tolerance. Relying on various different libraries often results
//! in compatibility issues as well as performance and safety issues, which can impact service
//! reliability. By providing a common framework, we simplify implementation, reduce development
//! time, and promote best practices in observability.
//!
//! ## Goals
//!
//! - Provide a common metrics endpoint for each component/process
//! - Provide common APIs for programmers to add metrics
//! - Provide a way to create a common data formats (e.g., structs and metric types)
//! - Provide flexibility – allow different implementations using the same APIs
//! - Improve safety: use tested and well-known APIs instead of disparate libraries
//!
//! ## Configuration
//!
//! The framework can be configured through:
//! 1. Environment variables (highest priority)
//! 2. Configuration files
//! 3. Default values
//!
//! Environment variables:
//! - `DYN_OBSERVABILITY_ENABLED`: Enable/disable observability (default: true)

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

// Prometheus imports - using fully qualified names to avoid conflicts
use prometheus::{
    Registry, IntCounter, IntGauge, Opts, CounterVec, GaugeVec, HistogramVec,
};

// OpenTelemetry imports - using module alias for shorter typing
use opentelemetry as otel;
use opentelemetry::{
    global, metrics::Meter, trace::Tracer,
    KeyValue, Context, trace::Span,
};
use opentelemetry_sdk as otel_sdk;
use opentelemetry_sdk::{
    metrics::MeterProvider, trace::TracerProvider,
    export::metrics::aggregation, Resource,
};
use opentelemetry_otlp::{TonicExporterBuilder, WithExportConfig};

/// Once instance to ensure the observability framework is only initialized once
static INIT: Once = Once::new();

/// Global observability state
static OBSERVABILITY_STATE: once_cell::sync::Lazy<Arc<Mutex<ObservabilityState>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(Mutex::new(ObservabilityState {
            enabled: true,
            metrics_backend: None,
            start_time: Instant::now(),
        }))
    });

/// Configuration for the observability framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Whether observability is enabled
    pub enabled: bool,
    // /// Whether distributed tracing is enabled
    // pub tracing_enabled: bool,
    /// Component name for identification
    pub component_name: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // tracing_enabled: true,
            // TODO: get component name
            component_name: "dynamo-component".to_string(),
        }
    }
}

/// Internal state for the observability framework
struct ObservabilityState {
    enabled: bool,
    metrics_backend: Option<Arc<dyn MetricsBackend>>,
    // tracing_backend: Option<Arc<dyn TracingBackend>>,
    start_time: Instant,
}

/// Supported metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    DynamoCounter(u64),
    DynamoGauge(f64),
    // TODO: add histogram, summary, etc...
}

/// Metric metadata
// TODO: This may not be needed since the DynamoCounter, Gauge, etc. already have the metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMetadata {
    pub name: String,
    pub description: Option<String>,
    pub labels: HashMap<String, String>,
}

/*
/// Trace span for distributed tracing
#[derive(Debug)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub attributes: HashMap<String, String>,
    pub events: Vec<TraceEvent>,
}

impl Clone for TraceSpan {
    fn clone(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
            parent_span_id: self.parent_span_id.clone(),
            name: self.name.clone(),
            start_time: self.start_time,
            end_time: self.end_time,
            attributes: self.attributes.clone(),
            events: self.events.clone(),
        }
    }
}

/// Trace event within a span
#[derive(Debug)]
pub struct TraceEvent {
    pub name: String,
    pub timestamp: Instant,
    pub attributes: HashMap<String, String>,
}

impl Clone for TraceEvent {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            timestamp: self.timestamp,
            attributes: self.attributes.clone(),
        }
    }
}
*/

/// Metrics backend trait - implement this for Prometheus, OpenTelemetry, etc.
pub trait MetricsBackend: Send + Sync {
    /// Record a counter metric
    fn record_counter(&self, metadata: &MetricMetadata, value: u64);

    /// Record a gauge metric
    fn record_gauge(&self, metadata: &MetricMetadata, value: f64);

    /// Get all metrics in the backend's format
    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

/// Tracing backend trait - implement this for OpenTelemetry, Jaeger, etc.
pub trait TracingBackend: Send + Sync {
    /// Start a new span
    fn start_span(&self, name: &str, attributes: HashMap<String, String>) -> Box<dyn ActiveSpan>;

    /// Record a span
    fn record_span(&self, span: &TraceSpan);

    /// Record an event within a span
    fn record_event(&self, span_id: &str, event: &TraceEvent);

    /// Export traces to the backend
    fn export_traces(&self, traces: &[TraceSpan]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Active span trait for tracing backends
pub trait ActiveSpan: Send + Sync {
    /// Get the span ID
    fn span_id(&self) -> &str;

    /// Add an attribute to the span
    fn add_attribute(&mut self, key: String, value: String);

    /// Add an event to the span
    fn add_event(&mut self, name: String, attributes: HashMap<String, String>);

    /// End the span
    fn end(self: Box<Self>);
}

/// Prometheus metrics backend implementation
pub struct PrometheusBackend {
    registry: Registry,
    counters: Arc<Mutex<HashMap<String, prometheus::Counter>>>,
    gauges: Arc<Mutex<HashMap<String, prometheus::Gauge>>>,
    histograms: Arc<Mutex<HashMap<String, prometheus::Histogram>>>,
}

impl PrometheusBackend {
    pub fn new() -> Self {
        Self {
            registry: Registry::new(),
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            histograms: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get or create a Prometheus counter
    pub fn get_or_create_counter(&self, name: &str, help: &str, labels: &[&str]) -> prometheus::Counter {
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
    pub fn get_or_create_gauge(&self, name: &str, help: &str, labels: &[&str]) -> prometheus::Gauge {
        let mut gauges = self.gauges.lock().unwrap();
        if let Some(gauge) = gauges.get(name) {
            gauge.clone()
        } else {
            let gauge = prometheus::Gauge::new(name, help).unwrap();
            self.registry.register(Box::new(gauge.clone())).unwrap();
            let gauge_clone = gauge.clone();
            gauges.insert(name.to_string(), gauge);
            gauge_clone
        }
    }

    /// Get or create a Prometheus histogram
    pub fn get_or_create_histogram(&self, name: &str, help: &str, buckets: Vec<f64>) -> prometheus::Histogram {
        let mut histograms = self.histograms.lock().unwrap();
        if let Some(histogram) = histograms.get(name) {
            histogram.clone()
        } else {
            let histogram_opts = prometheus::HistogramOpts::new(name, help).buckets(buckets);
            let histogram = prometheus::Histogram::with_opts(histogram_opts).unwrap();
            self.registry.register(Box::new(histogram.clone())).unwrap();
            let histogram_clone = histogram.clone();
            histograms.insert(name.to_string(), histogram);
            histogram_clone
        }
    }
}

impl MetricsBackend for PrometheusBackend {
    fn record_counter(&self, metadata: &MetricMetadata, value: u64) {
        let counter = self.get_or_create_counter(
            &metadata.name,
            metadata.description.as_deref().unwrap_or(""),
            &[],
        );
        counter.inc_by(value as f64);
    }

    fn record_gauge(&self, metadata: &MetricMetadata, value: f64) {
        let mut gauge = self.get_or_create_gauge(
            &metadata.name,
            metadata.description.as_deref().unwrap_or(""),
            &[],
        );
        gauge.set(value);
    }

    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut buffer = Vec::new();
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

/// OpenTelemetry metrics backend implementation
pub struct OpenTelemetryBackend {
    meter: otel::metrics::Meter,
    tracer: otel::trace::Tracer,
    counters: Arc<Mutex<HashMap<String, otel::metrics::Counter<u64>>>>,
    gauges: Arc<Mutex<HashMap<String, otel::metrics::Gauge<f64>>>>,
    histograms: Arc<Mutex<HashMap<String, otel::metrics::Histogram<f64>>>>,
}

impl OpenTelemetryBackend {
    pub fn new(service_name: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Initialize OpenTelemetry with OTLP exporter
        let resource = otel_sdk::Resource::new(vec![otel::KeyValue::new("service.name", service_name)]);

        // Set up metrics
        let meter_provider = otel_sdk::metrics::MeterProvider::builder()
            .with_resource(resource.clone())
            .build();
        otel::global::set_meter_provider(meter_provider);
        let meter = otel::global::meter("dynamo");

        // Set up tracing
        let tracer_provider = otel_sdk::trace::TracerProvider::builder()
            .with_resource(resource)
            .build();
        otel::global::set_tracer_provider(tracer_provider);
        let tracer = otel::global::tracer("dynamo");

        Ok(Self {
            meter,
            tracer,
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            histograms: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get or create an OpenTelemetry counter
    pub fn get_or_create_counter(&self, name: &str, description: &str) -> otel::metrics::Counter<u64> {
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
    pub fn get_or_create_gauge(&self, name: &str, description: &str) -> otel::metrics::Gauge<f64> {
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

    /// Get or create an OpenTelemetry histogram
    pub fn get_or_create_histogram(&self, name: &str, description: &str) -> otel::metrics::Histogram<f64> {
        let mut histograms = self.histograms.lock().unwrap();
        if let Some(histogram) = histograms.get(name) {
            histogram.clone()
        } else {
            let histogram = self.meter
                .f64_histogram(name)
                .with_description(description)
                .init();
            histograms.insert(name.to_string(), histogram.clone());
            histogram
        }
    }

    /// Start a new span
    pub fn start_span(&self, name: &str, attributes: HashMap<String, String>) -> otel::trace::Span {
        let mut span_builder = self.tracer.span_builder(name);

        // Convert attributes to OpenTelemetry KeyValue format
        let otel_attributes: Vec<otel::KeyValue> = attributes
            .into_iter()
            .map(|(k, v)| otel::KeyValue::new(k, v))
            .collect();

        span_builder.attributes = otel_attributes.into();
        self.tracer.build(span_builder)
    }
}

impl MetricsBackend for OpenTelemetryBackend {
    fn record_counter(&self, metadata: &MetricMetadata, value: u64) {
        let counter = self.get_or_create_counter(
            &metadata.name,
            metadata.description.as_deref().unwrap_or(""),
        );

        // Convert labels to OpenTelemetry attributes
        let attributes: Vec<otel::KeyValue> = metadata.labels
            .iter()
            .map(|(k, v)| otel::KeyValue::new(k.clone(), v.clone()))
            .collect();

        counter.add(value, &attributes);
    }

    fn record_gauge(&self, metadata: &MetricMetadata, value: f64) {
        let gauge = self.get_or_create_gauge(
            &metadata.name,
            metadata.description.as_deref().unwrap_or(""),
        );

        // Convert labels to OpenTelemetry attributes
        let attributes: Vec<otel::KeyValue> = metadata.labels
            .iter()
            .map(|(k, v)| otel::KeyValue::new(k.clone(), v.clone()))
            .collect();

        gauge.record(value, &attributes);
    }

    fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // OpenTelemetry metrics are typically exported via OTLP, not retrieved as text
        // This is a placeholder implementation
        Ok("OpenTelemetry metrics are exported via OTLP".to_string())
    }
}

/// Metric builder for creating metrics
pub struct MetricBuilder {
    name: String,
    description: Option<String>,
    labels: HashMap<String, String>,
}

impl MetricBuilder {
    /// Create a new metric builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            labels: HashMap::new(),
        }
    }

    /// Add a description to the metric
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a label to the metric
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Build a counter metric
    pub fn counter(self) -> DynamoCounter {
        // For now, create a simple counter without backend
        // This will need to be updated when backends are properly integrated
        DynamoCounter {
            metadata: MetricMetadata {
                name: self.name,
                description: self.description,
                labels: self.labels,
            },
            backend_counter: CounterBackend::Prometheus(
                prometheus::Counter::new("temp", "temporary").unwrap()
            ),
        }
    }

    /// Build a gauge metric
    pub fn gauge(self) -> DynamoGauge {
        DynamoGauge {
            metadata: MetricMetadata {
                name: self.name,
                description: self.description,
                labels: self.labels,
            },
            value: 0.0,
        }
    }


}

/// Counter metric type
pub struct DynamoCounter {
    metadata: MetricMetadata,
    backend_counter: CounterBackend,
}

enum CounterBackend {
    Prometheus(prometheus::Counter),
    OpenTelemetry(otel::metrics::Counter<u64>),
}

impl DynamoCounter {
    /// Increment the counter by 1
    pub fn inc(&mut self) {
        match &self.backend_counter {
            CounterBackend::Prometheus(counter) => {
                counter.inc();
            }
            CounterBackend::OpenTelemetry(counter) => {
                counter.add(1, &[]);  // TODO: add labels
            }
        }
        self.record();
    }

    /// Increment the counter by a specific amount
    pub fn inc_by(&mut self, amount: u64) {
        match &self.backend_counter {
            CounterBackend::Prometheus(counter) => {
                counter.inc_by(amount as f64);
            }
            CounterBackend::OpenTelemetry(counter) => {
                counter.add(amount, &[]);  // TODO: add labels
            }
        }
        self.record();
    }

    /// Get the current value
    pub fn value(&self) -> u64 {
        match &self.backend_counter {
            CounterBackend::Prometheus(counter) => {
                counter.get() as u64
            }
            CounterBackend::OpenTelemetry(_counter) => {
                // OpenTelemetry counters don't expose current value
                0
            }
        }
    }

    /// Record the metric
    fn record(&self) {
        if let Ok(state) = OBSERVABILITY_STATE.lock() {
            if state.enabled {
                if let Some(backend) = &state.metrics_backend {
                    let current_value = self.value();
                    backend.record_counter(&self.metadata, current_value);
                }
            }
        }
    }
}

/// Gauge metric type
pub struct DynamoGauge {
    metadata: MetricMetadata,
    value: f64,
}

impl DynamoGauge {
    /// Set the gauge value
    pub fn set(&mut self, value: f64) {
        self.value = value;
        self.record();
    }

    /// Increment the gauge by a value
    pub fn inc(&mut self, value: f64) {
        self.value += value;
        self.record();
    }

    /// Decrement the gauge by a value
    pub fn dec(&mut self, value: f64) {
        self.value -= value;
        self.record();
    }

    /// Get the current value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Record the metric
    fn record(&self) {
        if let Ok(state) = OBSERVABILITY_STATE.lock() {
            if state.enabled {
                if let Some(backend) = &state.metrics_backend {
                    backend.record_gauge(&self.metadata, self.value);
                }
            }
        }
    }
}



/*
/// Trace span builder
pub struct TraceSpanBuilder {
    name: String,
    trace_id: Option<String>,
    parent_span_id: Option<String>,
    attributes: HashMap<String, String>,
}

impl TraceSpanBuilder {
    /// Create a new trace span builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            trace_id: None,
            parent_span_id: None,
            attributes: HashMap::new(),
        }
    }

    /// Set the trace ID
    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set the parent span ID
    pub fn parent_span_id(mut self, parent_span_id: impl Into<String>) -> Self {
        self.parent_span_id = Some(parent_span_id.into());
        self
    }

    /// Add an attribute to the span
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Build and start the span
    pub fn start(self) -> Box<dyn ActiveSpan> {
        if let Ok(state) = OBSERVABILITY_STATE.lock() {
            if state.enabled {
                if let Some(backend) = &state.tracing_backend {
                    return backend.start_span(&self.name, self.attributes);
                }
            }
        }

        // Fallback to no-op span if no backend is configured
        Box::new(NoOpSpan {
            span_id: generate_span_id(),
        })
    }
}

/// No-op span implementation for when no tracing backend is configured
struct NoOpSpan {
    span_id: String,
}

impl ActiveSpan for NoOpSpan {
    fn span_id(&self) -> &str {
        &self.span_id
    }

    fn add_attribute(&mut self, _key: String, _value: String) {
        // No-op
    }

    fn add_event(&mut self, _name: String, _attributes: HashMap<String, String>) {
        // No-op
    }

    fn end(self: Box<Self>) {
        // No-op
    }
}
*/

/// Initialize the observability framework
pub fn init() {
    INIT.call_once(|| {
        let config = load_config();

        if let Ok(mut state) = OBSERVABILITY_STATE.lock() {
            state.enabled = config.enabled;
        }

        if config.enabled {
            info!("Observability framework initialized for component: {}", config.component_name);
        } else {
            warn!("Observability framework disabled");
        }
    });
}

/// Set the metrics backend
pub fn set_metrics_backend(backend: Arc<dyn MetricsBackend>) {
    if let Ok(mut state) = OBSERVABILITY_STATE.lock() {
        state.metrics_backend = Some(backend);
        info!("Metrics backend set");
    }
}

/*
/// Set the tracing backend
pub fn set_tracing_backend(backend: Arc<dyn TracingBackend>) {
    if let Ok(mut state) = OBSERVABILITY_STATE.lock() {
        state.tracing_backend = Some(backend);
        info!("Tracing backend set");
    }
}
*/

/// Load configuration from environment variables
fn load_config() -> ObservabilityConfig {
    let enabled = std::env::var("DYN_OBSERVABILITY_ENABLED")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .unwrap_or(true);

    // let tracing_enabled = std::env::var("DYN_TRACING_ENABLED")
    //     .unwrap_or_else(|_| "true".to_string())
    //     .parse()
    //     .unwrap_or(true);

    let component_name = std::env::var("DYN_COMPONENT_NAME")
        .unwrap_or_else(|_| "dynamo-component".to_string());

    ObservabilityConfig {
        enabled,
        // tracing_enabled,
        component_name,
    }
}

/*
/// Generate a random trace ID
fn generate_trace_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:032x}", rng.gen::<u128>())
}

/// Generate a random span ID
fn generate_span_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:016x}", rng.gen::<u64>())
}
*/

/// Convenience function to create a counter metric
pub fn counter(name: impl Into<String>) -> DynamoCounter {
    MetricBuilder::new(name).counter()
}

/// Convenience function to create a gauge metric
pub fn gauge(name: impl Into<String>) -> DynamoGauge {
    MetricBuilder::new(name).gauge()
}



/// Convenience function to start a trace span
// pub fn span(name: impl Into<String>) -> TraceSpanBuilder {
//     TraceSpanBuilder::new(name)
// }

/// Get metrics from the configured backend
pub fn get_metrics() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(state) = OBSERVABILITY_STATE.lock() {
        if let Some(backend) = &state.metrics_backend {
            backend.get_metrics()
        } else {
            Err("No metrics backend configured".into())
        }
    } else {
        Err("Failed to access observability state".into())
    }
}

/*
/// Export traces to the configured backend
pub fn export_traces(traces: &[TraceSpan]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(state) = OBSERVABILITY_STATE.lock() {
        if let Some(backend) = &state.tracing_backend {
            backend.export_traces(traces)
        } else {
            Err("No tracing backend configured".into())
        }
    } else {
        Err("Failed to access observability state".into())
    }
}
*/

/// Check if observability is enabled
pub fn is_enabled() -> bool {
    if let Ok(state) = OBSERVABILITY_STATE.lock() {
        state.enabled
    } else {
        false
    }
}

/// Check if metrics backend is configured
pub fn has_metrics_backend() -> bool {
    if let Ok(state) = OBSERVABILITY_STATE.lock() {
        state.metrics_backend.is_some()
    } else {
        false
    }
}

/// Check if tracing backend is configured
pub fn has_tracing_backend() -> bool {
    // if let Ok(state) = OBSERVABILITY_STATE.lock() {
    //     state.tracing_backend.is_some()
    // } else {
    //     false
    // }
    false // Temporarily disabled since tracing_backend was commented out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mock metrics backend for testing
    struct MockMetricsBackend {
        metrics: Arc<Mutex<HashMap<String, MetricValue>>>,
    }

    impl MockMetricsBackend {
        fn new() -> Self {
            Self {
                metrics: Arc::new(Mutex::new(HashMap::new())),
            }
        }
    }

    impl MetricsBackend for MockMetricsBackend {
        fn record_counter(&self, metadata: &MetricMetadata, value: u64) {
            let key = format!("{}", metadata.name);
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.insert(key, MetricValue::DynamoCounter(value));
            }
        }

        fn record_gauge(&self, metadata: &MetricMetadata, value: f64) {
            let key = format!("{}", metadata.name);
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.insert(key, MetricValue::DynamoGauge(value));
            }
        }

        fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            if let Ok(metrics) = self.metrics.lock() {
                Ok(format!("{:?}", *metrics))
            } else {
                Err("Failed to lock metrics".into())
            }
        }
    }

/*
    // Mock tracing backend for testing
    struct MockTracingBackend {
        spans: Arc<Mutex<Vec<TraceSpan>>>,
    }

    impl MockTracingBackend {
        fn new() -> Self {
            Self {
                spans: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl TracingBackend for MockTracingBackend {
        fn start_span(&self, name: &str, attributes: HashMap<String, String>) -> Box<dyn ActiveSpan> {
            let span_id = generate_span_id();
            let span = TraceSpan {
                trace_id: generate_trace_id(),
                span_id: span_id.clone(),
                parent_span_id: None,
                name: name.to_string(),
                start_time: Instant::now(),
                end_time: None,
                attributes,
                events: Vec::new(),
            };

            if let Ok(mut spans) = self.spans.lock() {
                spans.push(span);
            }

            Box::new(MockActiveSpan {
                span_id,
                backend: Arc::clone(&self.spans),
            })
        }

        fn record_span(&self, span: &TraceSpan) {
            if let Ok(mut spans) = self.spans.lock() {
                spans.push(span.clone());
            }
        }

        fn record_event(&self, span_id: &str, event: &TraceEvent) {
            if let Ok(mut spans) = self.spans.lock() {
                if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                    span.events.push(event.clone());
                }
            }
        }

        fn export_traces(&self, _traces: &[TraceSpan]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
    }

    struct MockActiveSpan {
        span_id: String,
        backend: Arc<Mutex<Vec<TraceSpan>>>,
    }

    impl ActiveSpan for MockActiveSpan {
        fn span_id(&self) -> &str {
            &self.span_id
        }

        fn add_attribute(&mut self, _key: String, _value: String) {
            // No-op for mock
        }

        fn add_event(&mut self, _name: String, _attributes: HashMap<String, String>) {
            // No-op for mock
        }

        fn end(self: Box<Self>) {
            if let Ok(mut spans) = self.backend.lock() {
                if let Some(span) = spans.iter_mut().find(|s| s.span_id == self.span_id) {
                    span.end_time = Some(Instant::now());
                }
            }
        }
    }
*/
    #[test]
    fn test_metric_builder() {
        let counter: DynamoCounter = MetricBuilder::new("test_counter")
            .description("A test counter")
            .label("service", "test")
            .counter();

        assert_eq!(counter.value(), 0);
    }

    #[test]
    fn test_counter_increment() {
        let mut counter = counter("test_inc");
        counter.inc();
        assert_eq!(counter.value(), 1);

        counter.inc_by(5);
        assert_eq!(counter.value(), 6);
    }

    #[test]
    fn test_gauge_operations() {
        let mut gauge: DynamoGauge = gauge("test_gauge");
        gauge.set(10.5);
        assert_eq!(gauge.value(), 10.5);

        gauge.inc(2.5);
        assert_eq!(gauge.value(), 13.0);

        gauge.dec(1.0);
        assert_eq!(gauge.value(), 12.0);
    }



/*
    #[test]
    fn test_trace_span() {
        let mut span = span("test_span")
            .attribute("service", "test")
            .start();

        span.add_event("test_event".to_string(), HashMap::new());
        span.end();
    }
*/

    #[test]
    fn test_config_loading() {
        let config = load_config();
        assert!(config.enabled);
        // assert!(config.tracing_enabled); // Commented out since tracing_enabled was removed
    }

    #[test]
    fn test_metrics_backend_integration() {
        let backend = Arc::new(MockMetricsBackend::new());
        set_metrics_backend(backend);

        let mut counter = counter("test_backend_counter");
        counter.inc();

        assert!(has_metrics_backend());
    }

    #[test]
    fn example_prometheus_and_opentelemetry_counters() {
        // Example 1: Using Prometheus Backend directly
        println!("=== Prometheus Counter Example ===");
        let prometheus_backend = PrometheusBackend::new();

        // Create a counter directly using the backend
        let prom_counter: prometheus::Counter = prometheus_backend.get_or_create_counter(
            "prometheus_requests_total",
            "Total number of requests",
            &["service", "endpoint"]
        );
        // Add some numbers
        prom_counter.inc(); // Increment by 1
        prom_counter.inc_by(5.0); // Increment by 5
        prom_counter.inc_by(10.0); // Increment by 10
        println!("Prometheus counter value: {}", prom_counter.get());
        assert_eq!(prom_counter.get(), 16.0);

        let prom_gauge: prometheus::Gauge = prometheus_backend.get_or_create_gauge(
            "prometheus_requests_gauge",
            "Total number of requests",
            &["service", "endpoint"]
        );
        prom_gauge.set(10.0);
        prom_gauge.inc(2.0);
        prom_gauge.dec(100.0);
        println!("Prometheus gauge value: {}", prom_gauge.get());
        assert_eq!(prom_gauge.get(), -90.0);

        // Retrieve metrics from Prometheus backend
        match prometheus_backend.get_metrics() {
            Ok(metrics) => {
                println!("Prometheus metrics output:");
                println!("{}", metrics);
            }
            Err(e) => println!("Failed to get Prometheus metrics: {}", e),
        }

        // Example 2: Using OpenTelemetry Backend directly
        println!("\n=== OpenTelemetry Counter Example ===");
        let otel_backend = OpenTelemetryBackend::new("example-service").unwrap_or_else(|e| {
            println!("Failed to create OpenTelemetry backend: {}", e);
            // This might fail in test environment, so we don't assert
            panic!("OpenTelemetry backend creation failed")
        });

        // Create another counter directly using the backend
        let otel_counter: opentelemetry::metrics::Counter<u64> = otel_backend.get_or_create_counter(
            "opentelemetry_requests_total",
            "Total number of requests via OpenTelemetry"
        );

        // Add some numbers
        otel_counter.add(1, &[otel::KeyValue::new("service", "example"), otel::KeyValue::new("protocol", "otlp")]);
        otel_counter.add(3, &[otel::KeyValue::new("service", "example"), otel::KeyValue::new("protocol", "otlp")]);
        otel_counter.add(7, &[otel::KeyValue::new("service", "example"), otel::KeyValue::new("protocol", "otlp")]);

        println!("OpenTelemetry counter incremented by 11 total");

        // Retrieve metrics from OpenTelemetry backend
        match otel_backend.get_metrics() {
            Ok(metrics) => {
                println!("OpenTelemetry metrics output:");
                println!("{}", metrics);
            }
            Err(e) => println!("Failed to get OpenTelemetry metrics: {}", e),
        }

        println!("\n=== Examples Complete ===");
    }

/*
    #[test]
    fn test_tracing_backend_integration() {
        let backend = Arc::new(MockTracingBackend::new());
        set_tracing_backend(backend);

        let span = span("test_backend_span").start();
        span.end();

        assert!(has_tracing_backend());
    }
*/
}