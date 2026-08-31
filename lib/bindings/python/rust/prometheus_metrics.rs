// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for registering Prometheus exposition callbacks.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::rs::metrics::MetricsHierarchy;

/// What a typed callback returns: `[(name, help, type, [(sample, [(k, v)], value)])]`.
///
/// Extracted natively by pyo3 -- deliberately not JSON or any other string, so
/// the structure never round-trips through text on either side of the boundary.
type PyTypedFamilies = Vec<(
    String,
    String,
    String,
    Vec<(String, Vec<(String, String)>, f64)>,
)>;

/// Wrap a Python callable returning typed families into the Rust callback shape.
fn wrap_py_typed_callback(
    callback: PyObject,
    source: &'static str,
) -> crate::rs::metrics::PrometheusTypedCallback {
    Arc::new(move || {
        Python::with_gil(|py| {
            let result = callback
                .call0(py)
                .map_err(|e| anyhow::anyhow!("{source} typed callback raised: {e}"))?;
            let typed: PyTypedFamilies = result.extract(py).map_err(|e| {
                anyhow::anyhow!("{source} typed callback returned an unexpected shape: {e}")
            })?;
            Ok(crate::rs::metrics::prom_typed::build_families(
                typed
                    .into_iter()
                    .map(|(name, help, kind, samples)| {
                        crate::rs::metrics::prom_typed::TypedFamily {
                            name,
                            help,
                            kind,
                            samples: samples
                                .into_iter()
                                .map(|(name, labels, value)| {
                                    crate::rs::metrics::prom_typed::TypedSample {
                                        name,
                                        labels: labels.into_iter().collect(),
                                        value,
                                        timestamp: None,
                                    }
                                })
                                .collect(),
                        }
                    })
                    .collect(),
            ))
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
    /// Register a callback returning typed metric families.
    ///
    /// Preferred over the exposition-text callback: it avoids rendering the
    /// engine's already-typed metrics to text only to parse them back.
    fn register_prometheus_typed_callback(&self, callback: PyObject) -> PyResult<()> {
        self.hierarchy
            .get_metrics_registry()
            .add_typed_callback(wrap_py_typed_callback(callback, "RuntimeMetrics"));
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
    /// Mirrors `RuntimeMetrics.register_prometheus_typed_callback`.
    fn register_prometheus_typed_callback(&self, callback: PyObject) -> PyResult<()> {
        self.hierarchy
            .get_metrics_registry()
            .add_typed_callback(wrap_py_typed_callback(callback, "EngineMetrics"));
        Ok(())
    }

    /// Precomputed hierarchy + model labels for the `gather_with_labels` helper.
    #[getter]
    fn auto_labels(&self) -> HashMap<String, String> {
        (*self.auto_labels).clone()
    }
}
