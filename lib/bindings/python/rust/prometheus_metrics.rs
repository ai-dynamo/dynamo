// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file implements Python bindings for Prometheus metric types.
// It should be kept in sync with:
// - lib/bindings/python/src/dynamo/_metrics.pyi (Python type stubs - method signatures must match)
// - lib/runtime/src/metrics.rs (MetricsRegistry trait - metric types should align)
//
// When adding/modifying metric methods:
// 1. Update the Rust implementation here (#[pymethods])
// 2. Update the Python type stub in _metrics.pyi
// 3. Follow standard Prometheus API conventions (e.g., Counter.inc(), Gauge.set(), etc.)

use prometheus::core::Collector;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::rs;

/// Helper function to order label values according to variable_labels declaration.
/// This ensures labels are passed to with_label_values() in the correct order.
///
/// # Arguments
/// * `variable_labels` - The ordered list of label names as declared in the metric
/// * `labels` - The HashMap of label name-value pairs from Python
///
/// # Returns
/// * `Ok(Vec<&str>)` - Ordered vector of label values matching variable_labels order
/// * `Err(PyErr)` - If a required label is missing
fn collect_ordered_label_values<'a>(
    variable_labels: &[String],
    labels: &'a HashMap<String, String>,
) -> PyResult<Vec<&'a str>> {
    let mut ordered_values = Vec::with_capacity(variable_labels.len());
    for label_name in variable_labels {
        match labels.get(label_name) {
            Some(value) => ordered_values.push(value.as_str()),
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Missing required label '{}'. Expected labels: {:?}, Got: {:?}",
                    label_name,
                    variable_labels,
                    labels.keys().collect::<Vec<_>>()
                )));
            }
        }
    }
    Ok(ordered_values)
}

// Python wrappers for Prometheus metric types.
//
// These wrapper structs are necessary because Prometheus types from the external `prometheus` crate
// cannot be directly exposed to Python via PyO3's #[pyclass] attribute. This is due to:
//
// 1. **Orphan Rule**: PyO3 requires implementing traits on types, but Rust's orphan rule prevents
//    implementing foreign traits (like PyClass) on foreign types (prometheus::Counter, etc.).
//
// 2. **Ownership**: #[pyclass] can only be applied to structs defined in your crate, not external types.
//
// 3. **PyO3 Requirements**: Types exposed to Python must satisfy specific trait bounds (Send, Sync)
//    and implement PyO3's internal traits, which we cannot add to external crate types.
//
// The solution is the newtype wrapper pattern: wrap each Prometheus type in our own struct,
// apply #[pyclass] to our wrapper, and delegate method calls to the inner Prometheus type.

/// Python wrapper for Counter metric
#[pyclass]
pub struct Counter {
    counter: prometheus::Counter,
}

/// Python wrapper for IntCounter metric
#[pyclass]
pub struct IntCounter {
    counter: prometheus::IntCounter,
}

/// Python wrapper for CounterVec metric
#[pyclass]
pub struct CounterVec {
    counter: prometheus::CounterVec,
}

/// Python wrapper for IntCounterVec metric
#[pyclass]
pub struct IntCounterVec {
    counter: prometheus::IntCounterVec,
}

/// Python wrapper for Gauge metric
#[pyclass]
pub struct Gauge {
    gauge: prometheus::Gauge,
}

/// Python wrapper for IntGauge metric
#[pyclass]
pub struct IntGauge {
    gauge: prometheus::IntGauge,
}

/// Python wrapper for GaugeVec metric
#[pyclass]
pub struct GaugeVec {
    gauge: prometheus::GaugeVec,
}

/// Python wrapper for IntGaugeVec metric
#[pyclass]
pub struct IntGaugeVec {
    gauge: prometheus::IntGaugeVec,
}

/// Python wrapper for Histogram metric
#[pyclass]
pub struct Histogram {
    histogram: prometheus::Histogram,
}

// ============================================================================
// Various PyMethod implementations below.
// ============================================================================

#[pymethods]
impl Counter {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.counter.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.counter.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Increment counter by 1
    fn inc(&self) -> PyResult<()> {
        self.counter.inc();
        Ok(())
    }

    /// Increment counter by value
    fn inc_by(&self, value: f64) -> PyResult<()> {
        self.counter.inc_by(value);
        Ok(())
    }

    /// Get counter value
    fn get(&self) -> PyResult<f64> {
        Ok(self.counter.get())
    }
}

impl Counter {
    fn from_prometheus(counter: prometheus::Counter) -> Self {
        Self { counter }
    }
}

#[pymethods]
impl IntCounter {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.counter.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.counter.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Increment counter by 1
    fn inc(&self) -> PyResult<()> {
        self.counter.inc();
        Ok(())
    }

    /// Increment counter by value
    fn inc_by(&self, value: u64) -> PyResult<()> {
        self.counter.inc_by(value);
        Ok(())
    }

    /// Get counter value
    fn get(&self) -> PyResult<u64> {
        Ok(self.counter.get())
    }
}

impl IntCounter {
    fn from_prometheus(counter: prometheus::IntCounter) -> Self {
        Self { counter }
    }
}

#[pymethods]
impl CounterVec {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.counter.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.counter.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Get the variable label names
    fn variable_labels(&self) -> PyResult<Vec<String>> {
        let desc = self.counter.desc();
        Ok(desc[0].variable_labels.clone())
    }

    /// Increment counter by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.counter.with_label_values(&label_values).inc();
        Ok(())
    }

    /// Increment counter by value with labels
    fn inc_by(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.counter.with_label_values(&label_values).inc_by(value);
        Ok(())
    }

    /// Get counter value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<f64> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        Ok(self.counter.with_label_values(&label_values).get())
    }
}

impl CounterVec {
    fn from_prometheus(counter: prometheus::CounterVec) -> Self {
        Self { counter }
    }
}

#[pymethods]
impl IntCounterVec {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.counter.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.counter.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Get the variable label names
    fn variable_labels(&self) -> PyResult<Vec<String>> {
        let desc = self.counter.desc();
        Ok(desc[0].variable_labels.clone())
    }

    /// Increment counter by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.counter.with_label_values(&label_values).inc();
        Ok(())
    }

    /// Increment counter by value with labels
    fn inc_by(&self, labels: HashMap<String, String>, value: u64) -> PyResult<()> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.counter.with_label_values(&label_values).inc_by(value);
        Ok(())
    }

    /// Get counter value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<u64> {
        let desc = self.counter.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        Ok(self.counter.with_label_values(&label_values).get())
    }
}

impl IntCounterVec {
    fn from_prometheus(counter: prometheus::IntCounterVec) -> Self {
        Self { counter }
    }
}

#[pymethods]
impl Gauge {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.gauge.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.gauge.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Set gauge value
    fn set(&self, value: f64) -> PyResult<()> {
        self.gauge.set(value);
        Ok(())
    }

    /// Get gauge value
    fn get(&self) -> PyResult<f64> {
        Ok(self.gauge.get())
    }

    /// Increment gauge by 1
    fn inc(&self) -> PyResult<()> {
        self.gauge.inc();
        Ok(())
    }

    /// Increment gauge by value
    fn inc_by(&self, value: f64) -> PyResult<()> {
        self.gauge.add(value);
        Ok(())
    }

    /// Decrement gauge by 1
    fn dec(&self) -> PyResult<()> {
        self.gauge.dec();
        Ok(())
    }

    /// Decrement gauge by value
    fn dec_by(&self, value: f64) -> PyResult<()> {
        self.gauge.sub(value);
        Ok(())
    }

    /// Add value to gauge
    fn add(&self, value: f64) -> PyResult<()> {
        self.gauge.add(value);
        Ok(())
    }

    /// Subtract value from gauge
    fn sub(&self, value: f64) -> PyResult<()> {
        self.gauge.sub(value);
        Ok(())
    }
}

impl Gauge {
    fn from_prometheus(gauge: prometheus::Gauge) -> Self {
        Self { gauge }
    }
}

#[pymethods]
impl IntGauge {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.gauge.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.gauge.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Set gauge value
    fn set(&self, value: i64) -> PyResult<()> {
        self.gauge.set(value);
        Ok(())
    }

    /// Get gauge value
    fn get(&self) -> PyResult<i64> {
        Ok(self.gauge.get())
    }

    /// Increment gauge by 1
    fn inc(&self) -> PyResult<()> {
        self.gauge.inc();
        Ok(())
    }

    /// Decrement gauge by 1
    fn dec(&self) -> PyResult<()> {
        self.gauge.dec();
        Ok(())
    }

    /// Add value to gauge
    fn add(&self, value: i64) -> PyResult<()> {
        self.gauge.add(value);
        Ok(())
    }

    /// Subtract value from gauge
    fn sub(&self, value: i64) -> PyResult<()> {
        self.gauge.sub(value);
        Ok(())
    }
}

impl IntGauge {
    fn from_prometheus(gauge: prometheus::IntGauge) -> Self {
        Self { gauge }
    }
}

#[pymethods]
impl GaugeVec {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.gauge.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.gauge.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Get the variable label names
    fn variable_labels(&self) -> PyResult<Vec<String>> {
        let desc = self.gauge.desc();
        Ok(desc[0].variable_labels.clone())
    }

    /// Set gauge value with labels
    fn set(&self, value: f64, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).set(value);
        Ok(())
    }

    /// Get gauge value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<f64> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        Ok(self.gauge.with_label_values(&label_values).get())
    }

    /// Increment gauge by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).inc();
        Ok(())
    }

    /// Decrement gauge by 1 with labels
    fn dec(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).dec();
        Ok(())
    }

    /// Add value to gauge with labels
    fn add(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).add(value);
        Ok(())
    }

    /// Subtract value from gauge with labels
    fn sub(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).sub(value);
        Ok(())
    }
}

impl GaugeVec {
    fn from_prometheus(gauge: prometheus::GaugeVec) -> Self {
        Self { gauge }
    }
}

#[pymethods]
impl IntGaugeVec {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.gauge.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.gauge.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Get the variable label names
    fn variable_labels(&self) -> PyResult<Vec<String>> {
        let desc = self.gauge.desc();
        Ok(desc[0].variable_labels.clone())
    }

    /// Set gauge value with labels
    fn set(&self, value: i64, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).set(value);
        Ok(())
    }

    /// Get gauge value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<i64> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        Ok(self.gauge.with_label_values(&label_values).get())
    }

    /// Increment gauge by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).inc();
        Ok(())
    }

    /// Decrement gauge by 1 with labels
    fn dec(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).dec();
        Ok(())
    }

    /// Add value to gauge with labels
    fn add(&self, labels: HashMap<String, String>, value: i64) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).add(value);
        Ok(())
    }

    /// Subtract value from gauge with labels
    fn sub(&self, labels: HashMap<String, String>, value: i64) -> PyResult<()> {
        let desc = self.gauge.desc();
        let label_values = collect_ordered_label_values(&desc[0].variable_labels, &labels)?;
        self.gauge.with_label_values(&label_values).sub(value);
        Ok(())
    }
}

impl IntGaugeVec {
    fn from_prometheus(gauge: prometheus::IntGaugeVec) -> Self {
        Self { gauge }
    }
}

#[pymethods]
impl Histogram {
    /// Get the metric name
    fn name(&self) -> PyResult<String> {
        let desc = self.histogram.desc();
        Ok(desc[0].fq_name.clone())
    }

    /// Get the constant labels
    fn const_labels(&self) -> PyResult<HashMap<String, String>> {
        let desc = self.histogram.desc();
        let labels: HashMap<String, String> = desc[0]
            .const_label_pairs
            .iter()
            .map(|pair| (pair.name().to_string(), pair.value().to_string()))
            .collect();
        Ok(labels)
    }

    /// Observe a value
    fn observe(&self, value: f64) -> PyResult<()> {
        self.histogram.observe(value);
        Ok(())
    }
}

impl Histogram {
    fn from_prometheus(histogram: prometheus::Histogram) -> Self {
        Self { histogram }
    }
}

/// RuntimeMetrics provides factory methods for creating typed Prometheus metrics
/// and utilities for registering metrics callbacks.
/// Exposed as endpoint.metrics, component.metrics, and namespace.metrics in Python.
///
/// NOTE: The create_* methods in RuntimeMetrics must stay in sync with the MetricsRegistry trait
/// in lib/runtime/src/metrics.rs. When adding new metric types, update both locations.
#[pyclass]
#[derive(Clone)]
pub struct RuntimeMetrics {
    hierarchy: Arc<dyn rs::metrics::MetricsHierarchy>,
}

impl RuntimeMetrics {
    /// Create from Endpoint
    pub fn from_endpoint(endpoint: dynamo_runtime::component::Endpoint) -> Self {
        Self {
            hierarchy: Arc::new(endpoint),
        }
    }

    /// Create from Component
    pub fn from_component(component: dynamo_runtime::component::Component) -> Self {
        Self {
            hierarchy: Arc::new(component),
        }
    }

    /// Create from Namespace
    pub fn from_namespace(namespace: dynamo_runtime::component::Namespace) -> Self {
        Self {
            hierarchy: Arc::new(namespace),
        }
    }

    /// Helper to convert Python labels (String, String) to Rust labels (&str, &str)
    fn convert_py_to_rust_labels(labels: &Option<Vec<(String, String)>>) -> Vec<(&str, &str)> {
        labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default()
    }

    /// Helper to convert Python label names Vec<String> to Vec<&str>
    fn convert_py_to_rust_label_names(names: &[String]) -> Vec<&str> {
        names.iter().map(|s| s.as_str()).collect()
    }

    /// Generic helper to register metrics callbacks for any type implementing MetricsHierarchy
    /// This allows Endpoint, Component, and Namespace to share the same callback registration logic
    pub fn register_callback_for<T>(registry_item: &T, callback: PyObject) -> PyResult<()>
    where
        T: rs::metrics::MetricsHierarchy + ?Sized,
    {
        // Get the metrics registry from the hierarchy and register the callback directly
        let metrics_registry = registry_item.get_metrics_registry();
        metrics_registry.add_update_callback(Arc::new(move || {
            // Execute the Python callback in the Python event loop
            Python::with_gil(|py| {
                if let Err(e) = callback.call0(py) {
                    tracing::error!("Metrics callback failed: {}", e);
                }
            });
            Ok(())
        }));

        Ok(())
    }
}

#[pymethods]
impl RuntimeMetrics {
    /// Register a Python callback to be invoked before metrics are scraped
    /// This callback will be called for this endpoint's metrics hierarchy
    fn register_callback(&self, callback: PyObject, _py: Python) -> PyResult<()> {
        Self::register_callback_for(self.hierarchy.as_ref(), callback)
    }

    /// Register a Python callback that returns Prometheus exposition text
    /// The returned text will be appended to the /metrics endpoint output
    /// The callback should return a string in Prometheus text exposition format
    fn register_prometheus_expfmt_callback(&self, callback: PyObject, _py: Python) -> PyResult<()> {
        // Create the callback once (Arc allows sharing across registries)
        let callback_arc = Arc::new(move || {
            // Execute the Python callback in the Python event loop
            Python::with_gil(|py| {
                match callback.call0(py) {
                    Ok(result) => {
                        // Try to extract a string from the result
                        match result.extract::<String>(py) {
                            Ok(text) => Ok(text),
                            Err(e) => {
                                tracing::error!(
                                    "Metrics exposition text callback must return a string: {}",
                                    e
                                );
                                Ok(String::new())
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Metrics exposition text callback failed: {}", e);
                        Ok(String::new())
                    }
                }
            })
        });

        // Register the callback at this hierarchy level
        self.hierarchy
            .get_metrics_registry()
            .add_expfmt_callback(callback_arc.clone());

        // Also register at all parent hierarchy levels so the callback is accessible
        // when prometheus_expfmt() is called on any parent (e.g., DRT)
        let parents = self.hierarchy.parent_hierarchies();
        for parent in parents.iter() {
            parent
                .get_metrics_registry()
                .add_expfmt_callback(callback_arc.clone());
        }

        Ok(())
    }

    // NOTE: The order of create_* methods below matches lib/runtime/src/metrics.rs::MetricsRegistry trait
    // Keep them synchronized when adding new metric types

    /// Create a Counter metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_counter(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Counter>> {
        let labels_vec = Self::convert_py_to_rust_labels(&labels);
        let counter: prometheus::Counter = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &labels_vec,
            None,
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Counter::from_prometheus(counter);
        Py::new(py, metric)
    }

    /// Create a CounterVec metric
    #[pyo3(signature = (name, description, label_names, const_labels=None))]
    fn create_countervec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        const_labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<CounterVec>> {
        let label_names_str = Self::convert_py_to_rust_label_names(&label_names);
        let const_labels_vec = Self::convert_py_to_rust_labels(&const_labels);
        let counter_vec: prometheus::CounterVec = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &const_labels_vec,
            None,
            Some(&label_names_str),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = CounterVec::from_prometheus(counter_vec);
        Py::new(py, metric)
    }

    /// Create a Gauge metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_gauge(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Gauge>> {
        let labels_vec = Self::convert_py_to_rust_labels(&labels);

        let gauge: prometheus::Gauge = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &labels_vec,
            None,
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Gauge::from_prometheus(gauge);
        Py::new(py, metric)
    }

    /// Create a GaugeVec metric
    #[pyo3(signature = (name, description, label_names, const_labels=None))]
    fn create_gaugevec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        const_labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<GaugeVec>> {
        let label_names_str = Self::convert_py_to_rust_label_names(&label_names);
        let const_labels_vec = Self::convert_py_to_rust_labels(&const_labels);
        let gauge_vec: prometheus::GaugeVec = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &const_labels_vec,
            None,
            Some(&label_names_str),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = GaugeVec::from_prometheus(gauge_vec);
        Py::new(py, metric)
    }

    /// Create a Histogram metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_histogram(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Histogram>> {
        let labels_vec = Self::convert_py_to_rust_labels(&labels);

        let histogram: prometheus::Histogram = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &labels_vec,
            None,
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Histogram::from_prometheus(histogram);
        Py::new(py, metric)
    }

    /// Create an IntCounter metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_intcounter(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntCounter>> {
        let labels_vec = Self::convert_py_to_rust_labels(&labels);

        let counter: prometheus::IntCounter = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &labels_vec,
            None,
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntCounter::from_prometheus(counter);
        Py::new(py, metric)
    }

    /// Create an IntCounterVec metric
    #[pyo3(signature = (name, description, label_names, const_labels=None))]
    fn create_intcountervec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        const_labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntCounterVec>> {
        let label_names_str = Self::convert_py_to_rust_label_names(&label_names);
        let const_labels_vec = Self::convert_py_to_rust_labels(&const_labels);
        let counter_vec: prometheus::IntCounterVec = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &const_labels_vec,
            None,
            Some(&label_names_str),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntCounterVec::from_prometheus(counter_vec);
        Py::new(py, metric)
    }

    /// Create an IntGauge metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_intgauge(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntGauge>> {
        let labels_vec = Self::convert_py_to_rust_labels(&labels);

        let gauge: prometheus::IntGauge = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &labels_vec,
            None,
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntGauge::from_prometheus(gauge);
        Py::new(py, metric)
    }

    /// Create an IntGaugeVec metric
    #[pyo3(signature = (name, description, label_names, const_labels=None))]
    fn create_intgaugevec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        const_labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntGaugeVec>> {
        let label_names_str = Self::convert_py_to_rust_label_names(&label_names);
        let const_labels_vec = Self::convert_py_to_rust_labels(&const_labels);
        let gauge_vec: prometheus::IntGaugeVec = rs::metrics::create_metric(
            self.hierarchy.as_ref(),
            &name,
            &description,
            &const_labels_vec,
            None,
            Some(&label_names_str),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntGaugeVec::from_prometheus(gauge_vec);
        Py::new(py, metric)
    }
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add specific metric type classes
    m.add_class::<Counter>()?;
    m.add_class::<IntCounter>()?;
    m.add_class::<CounterVec>()?;
    m.add_class::<IntCounterVec>()?;
    m.add_class::<Gauge>()?;
    m.add_class::<IntGauge>()?;
    m.add_class::<GaugeVec>()?;
    m.add_class::<IntGaugeVec>()?;
    m.add_class::<Histogram>()?;

    Ok(())
}
