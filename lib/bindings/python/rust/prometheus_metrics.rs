// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for registering Prometheus exposition callbacks.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::rs::metrics::{MetricsHierarchy, PrometheusExpositionFormatCallback};

/// Wrap a Python callable into the Rust expfmt callback shape. Logs and
/// returns an empty string on Python-side failures so a single broken
/// callback can't poison the scrape.
fn wrap_py_expfmt_callback(
    callback: PyObject,
    source: &'static str,
) -> PrometheusExpositionFormatCallback {
    Arc::new(move || {
        Python::with_gil(|py| match callback.call0(py) {
            Ok(result) => match result.extract::<String>(py) {
                Ok(text) => Ok(text),
                Err(e) => {
                    tracing::error!(error = %e, source, "expfmt callback must return a string");
                    Ok(String::new())
                }
            },
            Err(e) => {
                tracing::error!(error = %e, source, "expfmt callback raised");
                Ok(String::new())
            }
        })
    })
}

/// Callback-registration handle exposed as `endpoint.metrics` in Python.
#[pyclass]
#[derive(Clone)]
pub struct RuntimeMetrics {
    hierarchy: Arc<dyn MetricsHierarchy>,
}

impl RuntimeMetrics {
    /// Create from Endpoint
    pub fn from_endpoint(endpoint: dynamo_runtime::component::Endpoint) -> Self {
        Self {
            hierarchy: Arc::new(endpoint),
        }
    }
}

#[pymethods]
impl RuntimeMetrics {
    /// Register a callback that returns Prometheus exposition text. The
    /// returned text is appended to the `/metrics` endpoint output.
    fn register_prometheus_expfmt_callback(&self, callback: PyObject) -> PyResult<()> {
        // Register on this hierarchy level only — combined scrapes traverse
        // children, so registering on parents would double-count.
        self.hierarchy
            .get_metrics_registry()
            .add_expfmt_callback(wrap_py_expfmt_callback(callback, "RuntimeMetrics"));
        Ok(())
    }
}

/// Metrics-only handle passed to `LLMEngine.register_prometheus`.
/// Exposes `register_prometheus_expfmt_callback` plus the precomputed
/// `auto_labels` dict. Must not be retained past the hook's return.
#[pyclass(module = "dynamo._core.backend", name = "EngineMetrics")]
pub struct EngineMetrics {
    hierarchy: Arc<dyn MetricsHierarchy>,
    auto_labels: Arc<HashMap<String, String>>,
}

impl EngineMetrics {
    /// Construct from the Rust `EngineMetrics` the Worker holds.
    /// Shares both `Arc`s — no copies.
    pub fn from_rust(inner: &dynamo_backend_common::EngineMetrics) -> Self {
        Self {
            hierarchy: inner.hierarchy().clone(),
            auto_labels: inner.auto_labels().clone(),
        }
    }
}

#[pymethods]
impl EngineMetrics {
    /// Register a callback returning Prometheus exposition text.
    /// Mirrors `RuntimeMetrics.register_prometheus_expfmt_callback`.
    fn register_prometheus_expfmt_callback(&self, callback: PyObject) -> PyResult<()> {
        self.hierarchy
            .get_metrics_registry()
            .add_expfmt_callback(wrap_py_expfmt_callback(callback, "EngineMetrics"));
        Ok(())
    }

    /// Precomputed hierarchy + model labels for the `gather_with_labels` helper.
    #[getter]
    fn auto_labels(&self) -> HashMap<String, String> {
        (*self.auto_labels).clone()
    }
}
