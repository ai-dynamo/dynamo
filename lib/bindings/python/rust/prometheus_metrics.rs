// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Prometheus metrics callbacks.
//!
//! This module provides minimal bindings for registering callbacks to integrate
//! external Prometheus metrics (e.g., from vLLM, SGLang, TensorRT-LLM) into
//! Dynamo's metrics endpoint.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::rs;

/// RuntimeMetrics provides utilities for registering metrics callbacks.
/// Exposed as endpoint.metrics in Python.
///
/// Note: Metric creation methods have been removed from the public API.
/// This class only provides callback registration for integrating external metrics.
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
}

#[pymethods]
impl RuntimeMetrics {
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
}

pub fn add_to_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No metric type classes to add - only RuntimeMetrics is exposed
    Ok(())
}
