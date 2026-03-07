// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::metrics::performance_metrics::{
    DistributionMetricHandle, PerformanceMetricsRegistry, RateMetricHandle, RatioMetricHandle,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::prometheus_metrics::RuntimeMetrics;

#[pyclass(name = "RateMetric")]
pub struct PyRateMetric {
    inner: RateMetricHandle,
}

#[pymethods]
impl PyRateMetric {
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[pyo3(signature = (count = 1))]
    fn record_count(&self, count: u64) -> PyResult<()> {
        // Keep GIL on hot-path record calls: this path is typically sub-10us/record
        // (Python->Rust boundary + non-blocking channel try_send), so allow_threads
        // overhead is usually not worth it here.
        self.inner
            .record_count(count)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "DistributionMetric")]
pub struct PyDistributionMetric {
    inner: DistributionMetricHandle,
}

#[pymethods]
impl PyDistributionMetric {
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn record_value(&self, value: f64) -> PyResult<()> {
        self.inner
            .record_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "RatioMetric")]
pub struct PyRatioMetric {
    inner: RatioMetricHandle,
}

#[pymethods]
impl PyRatioMetric {
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn record_ratio(&self, numerator: f64, denominator: f64) -> PyResult<()> {
        self.inner
            .record_ratio(numerator, denominator)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "PerformanceMetricsRegistry")]
pub struct PyPerformanceMetricsRegistry {
    registry: PerformanceMetricsRegistry,
}

#[pymethods]
impl PyPerformanceMetricsRegistry {
    #[new]
    fn new(py: Python<'_>, runtime_metrics: &RuntimeMetrics) -> PyResult<Self> {
        let hierarchy = runtime_metrics.hierarchy();
        // Control-path setup can block (thread spawn + registry wiring), so release GIL.
        let registry = py
            .allow_threads(move || PerformanceMetricsRegistry::new_attached_default(hierarchy))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { registry })
    }

    fn window_seconds(&self) -> u64 {
        self.registry.window_duration().as_secs()
    }

    #[pyo3(signature = (name, quantiles = None, sample_period_seconds = 1.0, window_seconds = None))]
    fn new_rate_metric(
        &self,
        py: Python<'_>,
        name: String,
        quantiles: Option<Vec<f64>>,
        sample_period_seconds: f64,
        window_seconds: Option<f64>,
    ) -> PyResult<PyRateMetric> {
        let quantiles = quantiles.unwrap_or_default();
        // Control-path registration can block on worker/registry operations; release GIL.
        let handle = py
            .allow_threads(move || {
                self.registry.new_rate_metric(
                    name,
                    quantiles,
                    Some(sample_period_seconds),
                    window_seconds,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRateMetric { inner: handle })
    }

    #[pyo3(signature = (name, quantiles = None, window_seconds = None))]
    fn new_distribution_metric(
        &self,
        py: Python<'_>,
        name: String,
        quantiles: Option<Vec<f64>>,
        window_seconds: Option<f64>,
    ) -> PyResult<PyDistributionMetric> {
        let quantiles = quantiles.unwrap_or_default();
        // Control-path registration can block on worker/registry operations; release GIL.
        let handle = py
            .allow_threads(move || self.registry.new_distribution_metric(name, quantiles, window_seconds))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDistributionMetric { inner: handle })
    }

    #[pyo3(signature = (name, quantiles = None, window_seconds = None))]
    fn new_ratio_metric(
        &self,
        py: Python<'_>,
        name: String,
        quantiles: Option<Vec<f64>>,
        window_seconds: Option<f64>,
    ) -> PyResult<PyRatioMetric> {
        let quantiles = quantiles.unwrap_or_default();
        // Control-path registration can block on worker/registry operations; release GIL.
        let handle = py
            .allow_threads(move || self.registry.new_ratio_metric(name, quantiles, window_seconds))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRatioMetric { inner: handle })
    }
}
