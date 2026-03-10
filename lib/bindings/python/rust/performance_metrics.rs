// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Initially contributed by Baseten @michaelfeil, feel free to tag on maintenance.

use dynamo_runtime::metrics::performance_metrics::{
    DistributionMetricHandle as RsDistributionMetricHandle,
    PerformanceMetricsRegistry as RsPerformanceMetricsRegistry,
    RateMetricHandle as RsRateMetricHandle, RatioMetricHandle as RsRatioMetricHandle,
    RequestMetric as RsRequestMetric, RequestMetricsFactory as RsRequestMetricsFactory,
    RequestMetricsOptions as RsRequestMetricsOptions,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

use crate::prometheus_metrics::RuntimeMetrics;

#[pyclass(name = "RateMetric")]
pub struct PyRateMetric {
    inner: RsRateMetricHandle,
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
    inner: RsDistributionMetricHandle,
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
    inner: RsRatioMetricHandle,
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
    registry: RsPerformanceMetricsRegistry,
}

#[pyclass(name = "RequestMetricsFactory")]
pub struct PyRequestMetricsFactory {
    inner: RsRequestMetricsFactory,
}

#[pyclass(name = "RequestMetric")]
pub struct PyRequestMetric {
    inner: RsRequestMetric,
}

#[pymethods]
impl PyPerformanceMetricsRegistry {
    #[new]
    #[pyo3(signature = (runtime_metrics, publish_interval_seconds = 5.0, metric_prefix = "performance".to_string(), labels = None))]
    fn new(
        py: Python<'_>,
        runtime_metrics: &RuntimeMetrics,
        publish_interval_seconds: f64,
        metric_prefix: String,
        labels: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        if !publish_interval_seconds.is_finite() || publish_interval_seconds <= 0.0 {
            return Err(PyValueError::new_err(
                "publish_interval_seconds must be a positive finite number",
            ));
        }

        let hierarchy = runtime_metrics.hierarchy();
        let labels = labels.unwrap_or_default();
        // Control-path setup can block (thread spawn + registry wiring), so release GIL.
        let registry = py
            .allow_threads(move || {
                let label_pairs = labels
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect::<Vec<_>>();
                RsPerformanceMetricsRegistry::new_attached(
                    std::time::Duration::from_secs_f64(publish_interval_seconds),
                    hierarchy,
                    metric_prefix,
                    label_pairs.as_slice(),
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { registry })
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
            .allow_threads(move || {
                self.registry
                    .new_distribution_metric(name, quantiles, window_seconds)
            })
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
            .allow_threads(move || {
                self.registry
                    .new_ratio_metric(name, quantiles, window_seconds)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRatioMetric { inner: handle })
    }
}

#[pymethods]
impl PyRequestMetricsFactory {
    #[new]
    #[pyo3(signature = (
        performance_metrics_registry,
        *,
        metric_prefix = "request_metrics".to_string(),
        request_quantiles = None,
        request_sample_period_seconds = None,
        request_window_seconds = None,
        input_tokens_quantiles = None,
        input_tokens_sample_period_seconds = None,
        input_tokens_window_seconds = None,
        ttft_quantiles = None,
        ttft_window_seconds = None,
        ttft_per_input_token_quantiles = None,
        ttft_per_input_token_window_seconds = None,
        itl_quantiles = None,
        itl_window_seconds = None,
        pre_first_token_cancellation_quantiles = None,
        pre_first_token_cancellation_sample_period_seconds = None,
        pre_first_token_cancellation_window_seconds = None,
        mid_stream_cancellation_quantiles = None,
        mid_stream_cancellation_sample_period_seconds = None,
        mid_stream_cancellation_window_seconds = None,
        successful_request_quantiles = None,
        successful_request_sample_period_seconds = None,
        successful_request_window_seconds = None,
        itl_sample_rate = 0.05
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        performance_metrics_registry: &PyPerformanceMetricsRegistry,
        metric_prefix: String,
        request_quantiles: Option<Vec<f64>>,
        request_sample_period_seconds: Option<f64>,
        request_window_seconds: Option<f64>,
        input_tokens_quantiles: Option<Vec<f64>>,
        input_tokens_sample_period_seconds: Option<f64>,
        input_tokens_window_seconds: Option<f64>,
        ttft_quantiles: Option<Vec<f64>>,
        ttft_window_seconds: Option<f64>,
        ttft_per_input_token_quantiles: Option<Vec<f64>>,
        ttft_per_input_token_window_seconds: Option<f64>,
        itl_quantiles: Option<Vec<f64>>,
        itl_window_seconds: Option<f64>,
        pre_first_token_cancellation_quantiles: Option<Vec<f64>>,
        pre_first_token_cancellation_sample_period_seconds: Option<f64>,
        pre_first_token_cancellation_window_seconds: Option<f64>,
        mid_stream_cancellation_quantiles: Option<Vec<f64>>,
        mid_stream_cancellation_sample_period_seconds: Option<f64>,
        mid_stream_cancellation_window_seconds: Option<f64>,
        successful_request_quantiles: Option<Vec<f64>>,
        successful_request_sample_period_seconds: Option<f64>,
        successful_request_window_seconds: Option<f64>,
        itl_sample_rate: f64,
    ) -> PyResult<Self> {
        let registry = performance_metrics_registry.registry.clone();
        let mut options = RsRequestMetricsOptions::default();
        if let Some(v) = request_quantiles {
            options.request_quantiles = v;
        }
        if let Some(v) = request_sample_period_seconds {
            options.request_sample_period_seconds = Some(v);
        }
        if let Some(v) = request_window_seconds {
            options.request_window_seconds = Some(v);
        }
        if let Some(v) = input_tokens_quantiles {
            options.input_tokens_quantiles = v;
        }
        if let Some(v) = input_tokens_sample_period_seconds {
            options.input_tokens_sample_period_seconds = Some(v);
        }
        if let Some(v) = input_tokens_window_seconds {
            options.input_tokens_window_seconds = Some(v);
        }
        if let Some(v) = ttft_quantiles {
            options.ttft_quantiles = v;
        }
        if let Some(v) = ttft_window_seconds {
            options.ttft_window_seconds = Some(v);
        }
        if let Some(v) = ttft_per_input_token_quantiles {
            options.ttft_per_input_token_quantiles = v;
        }
        if let Some(v) = ttft_per_input_token_window_seconds {
            options.ttft_per_input_token_window_seconds = Some(v);
        }
        if let Some(v) = itl_quantiles {
            options.itl_quantiles = v;
        }
        if let Some(v) = itl_window_seconds {
            options.itl_window_seconds = Some(v);
        }
        if let Some(v) = pre_first_token_cancellation_quantiles {
            options.pre_first_token_cancellation_quantiles = v;
        }
        if let Some(v) = pre_first_token_cancellation_sample_period_seconds {
            options.pre_first_token_cancellation_sample_period_seconds = Some(v);
        }
        if let Some(v) = pre_first_token_cancellation_window_seconds {
            options.pre_first_token_cancellation_window_seconds = Some(v);
        }
        if let Some(v) = mid_stream_cancellation_quantiles {
            options.mid_stream_cancellation_quantiles = v;
        }
        if let Some(v) = mid_stream_cancellation_sample_period_seconds {
            options.mid_stream_cancellation_sample_period_seconds = Some(v);
        }
        if let Some(v) = mid_stream_cancellation_window_seconds {
            options.mid_stream_cancellation_window_seconds = Some(v);
        }
        if let Some(v) = successful_request_quantiles {
            options.successful_request_quantiles = v;
        }
        if let Some(v) = successful_request_sample_period_seconds {
            options.successful_request_sample_period_seconds = Some(v);
        }
        if let Some(v) = successful_request_window_seconds {
            options.successful_request_window_seconds = Some(v);
        }
        options.itl_sample_rate = itl_sample_rate;
        // Control-path setup can block due metric registration calls; release GIL.
        let inner = py
            .allow_threads(move || RsRequestMetricsFactory::new(&registry, metric_prefix, options))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    #[pyo3(signature = (input_tokens = 0))]
    fn new_request(&self, input_tokens: u64) -> PyRequestMetric {
        PyRequestMetric {
            inner: self.inner.new_request(input_tokens),
        }
    }
}

#[pymethods]
impl PyRequestMetric {
    #[pyo3(signature = (total_tokens, cached_tokens = None))]
    fn record_tokens(
        &mut self,
        py: Python<'_>,
        total_tokens: u64,
        cached_tokens: Option<u64>,
    ) -> PyResult<()> {
        // RequestMetric now uses a Send RNG, so this path can run without the GIL.
        py.allow_threads(|| self.inner.record_tokens(total_tokens, cached_tokens))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn success(&mut self) -> PyResult<()> {
        self.inner
            .success()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn cancel(&mut self) -> PyResult<()> {
        self.inner
            .cancel()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
