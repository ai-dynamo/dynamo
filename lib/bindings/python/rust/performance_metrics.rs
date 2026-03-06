// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::metrics::performance_metrics::{
    AttachedPerformanceMetrics, DistributionMetricHandle, MetricSnapshot, PerformanceMetricKind,
    PerformanceMetricsRegistry, RateMetricHandle, RatioMetricHandle,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3::{types::PyDict, Bound};
use std::time::Duration;

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
        self.inner
            .record_count(count)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self
            .inner
            .snapshot()
            .ok_or_else(|| PyValueError::new_err("metric was not found"))?;
        snapshot_to_pydict(py, snapshot)
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

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self
            .inner
            .snapshot()
            .ok_or_else(|| PyValueError::new_err("metric was not found"))?;
        snapshot_to_pydict(py, snapshot)
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

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self
            .inner
            .snapshot()
            .ok_or_else(|| PyValueError::new_err("metric was not found"))?;
        snapshot_to_pydict(py, snapshot)
    }
}

#[pyclass(name = "PerformanceMetricsRegistry")]
pub struct PyPerformanceMetricsRegistry {
    builder: Option<PerformanceMetricsRegistry>,
    attached: Option<AttachedPerformanceMetrics>,
}

#[pymethods]
impl PyPerformanceMetricsRegistry {
    #[new]
    #[pyo3(signature = (window_seconds = 60))]
    fn new(window_seconds: u64) -> Self {
        let seconds = window_seconds.max(1);
        Self {
            builder: Some(PerformanceMetricsRegistry::new(Duration::from_secs(seconds))),
            attached: None,
        }
    }

    fn window_seconds(&self) -> u64 {
        if let Some(builder) = &self.builder {
            return builder.window_duration().as_secs();
        }
        if let Some(attached) = &self.attached {
            return attached.window_duration().as_secs();
        }
        0
    }

    #[pyo3(signature = (name, quantiles = None, sample_period_seconds = 1.0))]
    fn new_rate_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
        sample_period_seconds: f64,
    ) -> PyResult<PyRateMetric> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("metrics already attached"))?;
        let handle = builder
            .new_rate_metric(
                name,
                quantiles.unwrap_or_default(),
                Some(sample_period_seconds),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRateMetric { inner: handle })
    }

    #[pyo3(signature = (name, quantiles = None))]
    fn new_distribution_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
    ) -> PyResult<PyDistributionMetric> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("metrics already attached"))?;
        let handle = builder
            .new_distribution_metric(name, quantiles.unwrap_or_default())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDistributionMetric { inner: handle })
    }

    #[pyo3(signature = (name, quantiles = None))]
    fn new_ratio_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
    ) -> PyResult<PyRatioMetric> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("metrics already attached"))?;
        let handle = builder
            .new_ratio_metric(name, quantiles.unwrap_or_default())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRatioMetric { inner: handle })
    }

    #[pyo3(signature = (runtime_metrics, metric_prefix = "performance", labels = None))]
    fn attach_runtime_metrics(
        &mut self,
        runtime_metrics: &RuntimeMetrics,
        metric_prefix: &str,
        labels: Option<Vec<(String, String)>>,
    ) -> PyResult<()> {
        if self.attached.is_some() {
            return Err(PyValueError::new_err(
                "performance metrics already attached for this registry",
            ));
        }
        let builder = self
            .builder
            .take()
            .ok_or_else(|| PyValueError::new_err("performance metrics already attached"))?;
        let hierarchy = runtime_metrics.hierarchy();
        let label_storage = labels.unwrap_or_default();
        let label_refs: Vec<(&str, &str)> = label_storage
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let attached = builder
            .attach_to_hierarchy(hierarchy.as_ref(), metric_prefix, &label_refs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        attached.publish();
        attached.start_auto_publish(Duration::from_secs(1));

        self.attached = Some(attached);
        Ok(())
    }
}

impl Drop for PyPerformanceMetricsRegistry {
    fn drop(&mut self) {
        if let Some(attached) = self.attached.take() {
            attached.stop_auto_publish();
        }
    }
}

fn snapshot_to_pydict<'py>(py: Python<'py>, snapshot: MetricSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", snapshot.name)?;
    dict.set_item(
        "kind",
        match snapshot.kind {
            PerformanceMetricKind::Rate => "rate",
            PerformanceMetricKind::Distribution => "distribution",
            PerformanceMetricKind::Ratio => "ratio",
        },
    )?;
    dict.set_item("rate_per_second", snapshot.rate_per_second)?;
    dict.set_item("average", snapshot.average)?;
    dict.set_item("quantiles", snapshot.quantiles)?;
    dict.set_item("numerator_sum", snapshot.numerator_sum)?;
    dict.set_item("denominator_sum", snapshot.denominator_sum)?;
    dict.set_item("ratio", snapshot.ratio)?;
    Ok(dict)
}
