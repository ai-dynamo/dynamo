// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the mocker AIC FPM engine performance shim.

use std::collections::BTreeMap;
use std::time::Duration;

use aiconfigurator_core::ForwardPassPerfOptions;
use dynamo_mocker::common::engine_perf as rs_engine_perf;
use dynamo_mocker::common::protocols::{ForwardPassSnapshot, WorkerType as RsWorkerType};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyMapping, PyMappingMethods};

use super::replay::MockEngineArgs;

/// AIC model and backend identity used by native forward-pass estimates.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AicEngineConfig {
    inner: rs_engine_perf::AicEngineConfig,
}

#[pymethods]
impl AicEngineConfig {
    #[new]
    #[pyo3(signature = (
        model_name,
        backend,
        system_name="h200_sxm",
        systems_path=None,
        backend_version=None,
        tp_size=1,
        pp_size=1,
        moe_tp_size=None,
        moe_ep_size=None,
        attention_dp_size=None,
        model_arch=None,
        weight_dtype=None,
        moe_dtype=None,
        activation_dtype=None,
        kv_cache_dtype=None,
        kv_block_size=None,
        extra=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_name: String,
        backend: String,
        system_name: &str,
        systems_path: Option<String>,
        backend_version: Option<String>,
        tp_size: u32,
        pp_size: u32,
        moe_tp_size: Option<u32>,
        moe_ep_size: Option<u32>,
        attention_dp_size: Option<u32>,
        model_arch: Option<String>,
        weight_dtype: Option<String>,
        moe_dtype: Option<String>,
        activation_dtype: Option<String>,
        kv_cache_dtype: Option<String>,
        kv_block_size: Option<u32>,
        extra: Option<BTreeMap<String, String>>,
    ) -> Self {
        Self {
            inner: rs_engine_perf::AicEngineConfig {
                model_name,
                model_arch,
                system_name: system_name.to_string(),
                systems_path,
                backend,
                backend_version,
                tp_size,
                pp_size,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                weight_dtype,
                moe_dtype,
                activation_dtype,
                kv_cache_dtype,
                kv_block_size,
                extra: extra.unwrap_or_default(),
            },
        }
    }
}

/// Engine limits used by engine-level helper queries and default AIC correction bounds.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EnginePerfLimits {
    inner: rs_engine_perf::EnginePerfLimits,
}

#[pymethods]
impl EnginePerfLimits {
    #[new]
    #[pyo3(signature = (max_num_batched_tokens=8192, max_num_seqs=512, max_kv_tokens=2_000_000))]
    fn new(max_num_batched_tokens: u32, max_num_seqs: u32, max_kv_tokens: u32) -> PyResult<Self> {
        Ok(Self {
            inner: rs_engine_perf::EnginePerfLimits::new(
                max_num_batched_tokens,
                max_num_seqs,
                max_kv_tokens,
            )
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    #[getter]
    fn max_num_batched_tokens(&self) -> u32 {
        self.inner.max_num_batched_tokens
    }

    #[getter]
    fn max_num_seqs(&self) -> u32 {
        self.inner.max_num_seqs
    }

    #[getter]
    fn max_kv_tokens(&self) -> u32 {
        self.inner.max_kv_tokens
    }
}

/// Online tuning options for the underlying AIC ForwardPassPerfModel.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RustEnginePerfOptions {
    inner: ForwardPassPerfOptions,
}

#[pymethods]
impl RustEnginePerfOptions {
    #[new]
    #[pyo3(signature = (
        max_observations=64,
        min_observations=5,
        bucket_count=16,
        max_num_tokens=8192,
        max_batch_size=512,
        max_kv_tokens=2_000_000,
    ))]
    fn new(
        max_observations: usize,
        min_observations: usize,
        bucket_count: usize,
        max_num_tokens: u32,
        max_batch_size: u32,
        max_kv_tokens: u32,
    ) -> Self {
        Self {
            inner: ForwardPassPerfOptions {
                max_observations,
                min_observations,
                bucket_count,
                max_num_tokens,
                max_batch_size,
                max_kv_tokens,
            },
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationTarget {
    Throughput,
    Latency,
}

impl From<OptimizationTarget> for rs_engine_perf::OptimizationTarget {
    fn from(value: OptimizationTarget) -> Self {
        match value {
            OptimizationTarget::Throughput => Self::Throughput,
            OptimizationTarget::Latency => Self::Latency,
        }
    }
}

/// Request shape and SLA policy for an engine capacity search.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EngineCapacityRequest {
    inner: rs_engine_perf::EngineCapacityRequest,
}

#[pymethods]
impl EngineCapacityRequest {
    #[new]
    #[pyo3(signature = (
        isl,
        osl,
        ttft_sla_ms=None,
        itl_sla_ms=None,
        e2e_latency_sla_ms=None,
        kv_hit_rate=None,
        optimization_target=OptimizationTarget::Throughput,
        accept_length=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        isl: u32,
        osl: u32,
        ttft_sla_ms: Option<f64>,
        itl_sla_ms: Option<f64>,
        e2e_latency_sla_ms: Option<f64>,
        kv_hit_rate: Option<f64>,
        optimization_target: OptimizationTarget,
        accept_length: f64,
    ) -> PyResult<Self> {
        if !accept_length.is_finite() {
            return Err(PyValueError::new_err(format!(
                "accept_length must be finite, got {accept_length}"
            )));
        }
        Ok(Self {
            inner: rs_engine_perf::EngineCapacityRequest {
                isl,
                osl,
                ttft_sla: ttft_sla_ms.map(ms_to_duration).transpose()?,
                itl_sla: itl_sla_ms.map(ms_to_duration).transpose()?,
                e2e_latency_sla: e2e_latency_sla_ms.map(ms_to_duration).transpose()?,
                accept_length: accept_length.max(1.0),
                kv_hit_rate,
                optimization_target: optimization_target.into(),
            },
        })
    }
}

/// Result of a per-engine capacity search.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EngineCapacity {
    inner: rs_engine_perf::EngineCapacity,
}

#[pymethods]
impl EngineCapacity {
    #[getter]
    fn rps(&self) -> f64 {
        self.inner.rps
    }

    #[getter]
    fn ttft_ms(&self) -> Option<f64> {
        self.inner.ttft.map(duration_to_ms)
    }

    #[getter]
    fn itl_ms(&self) -> Option<f64> {
        self.inner.itl.map(duration_to_ms)
    }

    #[getter]
    fn e2e_latency_ms(&self) -> Option<f64> {
        self.inner.e2e_latency.map(duration_to_ms)
    }

    #[getter]
    fn eligible(&self) -> bool {
        self.inner.eligible
    }
}

/// Engine-level performance model backed by AIC forward-pass modeling.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RustEnginePerfModel {
    inner: rs_engine_perf::EnginePerfModel,
}

#[pymethods]
impl RustEnginePerfModel {
    /// Build from all available inputs.
    ///
    /// Explicit AIC config is preferred. If deriving AIC config from engine_args,
    /// engine_args.aic_model_path is required whenever engine_args.aic_backend is set.
    /// Without aic_backend, the model starts in regression-only mode.
    #[staticmethod]
    #[pyo3(signature = (
        *,
        engine_args=None,
        aic_config=None,
        worker_type=None,
        limits=None,
        options=None,
        bootstrap_fpms=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn best_available(
        py: Python<'_>,
        engine_args: Option<PyRef<'_, MockEngineArgs>>,
        aic_config: Option<PyRef<'_, AicEngineConfig>>,
        worker_type: Option<&str>,
        limits: Option<PyRef<'_, EnginePerfLimits>>,
        options: Option<PyRef<'_, RustEnginePerfOptions>>,
        bootstrap_fpms: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let bootstrap_fpms = bootstrap_fpms
            .as_ref()
            .map(|obj| iterations_from_py(obj.bind(py)))
            .transpose()?
            .unwrap_or_default();
        let inputs = rs_engine_perf::EnginePerfModelInputs {
            engine_args: engine_args.as_ref().map(|args| args.inner()),
            aic_config: aic_config.as_ref().map(|config| config.inner.clone()),
            worker_type: worker_type.map(parse_worker_type).transpose()?,
            limits: limits.as_ref().map(|limits| limits.inner.clone()),
            options: options.as_ref().map(|options| options.inner.clone()),
            bootstrap_fpms,
        };
        Ok(Self {
            inner: rs_engine_perf::EnginePerfModel::best_available(inputs).map_err(to_pyerr)?,
        })
    }

    /// Build a regression-only model that learns directly from observed FPM wall times.
    #[staticmethod]
    #[pyo3(signature = (*, worker_type, limits, options=None, bootstrap_fpms=None))]
    fn from_regression(
        py: Python<'_>,
        worker_type: &str,
        limits: PyRef<'_, EnginePerfLimits>,
        options: Option<PyRef<'_, RustEnginePerfOptions>>,
        bootstrap_fpms: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut model = rs_engine_perf::EnginePerfModel::from_regression(
            parse_worker_type(worker_type)?,
            limits.inner.clone(),
            options.as_ref().map(|options| options.inner.clone()),
        )
        .map_err(to_pyerr)?;
        if let Some(fpms) = bootstrap_fpms.as_ref() {
            model
                .tune_with_fpms(&iterations_from_py(fpms.bind(py))?)
                .map_err(to_pyerr)?;
        }
        Ok(Self { inner: model })
    }

    /// Build a strict native AIC model; unsupported AIC configs raise an error.
    #[staticmethod]
    #[pyo3(signature = (*, aic_config, worker_type, limits, options=None, bootstrap_fpms=None))]
    fn from_native(
        py: Python<'_>,
        aic_config: PyRef<'_, AicEngineConfig>,
        worker_type: &str,
        limits: PyRef<'_, EnginePerfLimits>,
        options: Option<PyRef<'_, RustEnginePerfOptions>>,
        bootstrap_fpms: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut model = rs_engine_perf::EnginePerfModel::from_native(
            aic_config.inner.clone(),
            parse_worker_type(worker_type)?,
            limits.inner.clone(),
            options.as_ref().map(|options| options.inner.clone()),
        )
        .map_err(to_pyerr)?;
        if let Some(fpms) = bootstrap_fpms.as_ref() {
            model
                .tune_with_fpms(&iterations_from_py(fpms.bind(py))?)
                .map_err(to_pyerr)?;
        }
        Ok(Self { inner: model })
    }

    /// Estimate one scheduled forward-pass iteration in seconds from current-version FPMs.
    fn estimate_forward_pass_time(
        &self,
        metrics_by_rank: &Bound<'_, PyAny>,
    ) -> PyResult<Option<f64>> {
        Ok(self
            .inner
            .estimate_forward_pass_time(&metrics_by_rank_from_py(metrics_by_rank)?)
            .map_err(to_pyerr)?
            .map(|duration| duration.as_secs_f64()))
    }

    /// Tune with current-version observed FPMs: outer list is iterations, inner list is attention-DP ranks.
    fn tune_with_fpms(&mut self, iterations: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner
            .tune_with_fpms(&iterations_from_py(iterations)?)
            .map_err(to_pyerr)
    }

    /// Return AIC diagnostics as a JSON string.
    fn diagnostics(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.diagnostics()).map_err(to_pyerr)
    }

    /// Return the minimum ready native correction factor, or None if no factor is ready.
    fn get_min_correction_factor(&self) -> Option<f64> {
        self.inner.min_correction_factor()
    }

    /// Return the maximum ready native correction factor, or None if no factor is ready.
    fn get_max_correction_factor(&self) -> Option<f64> {
        self.inner.max_correction_factor()
    }

    /// Return the average ready native correction factor, or None if no factor is ready.
    fn get_avg_correction_factor(&self) -> Option<f64> {
        self.inner.avg_correction_factor()
    }

    /// Estimate queued prefill drain time in seconds.
    ///
    /// Current FPM queued-prefill fields carry raw queued prompt tokens and no
    /// KV-cache reuse estimate. Apply prefix-cache reuse outside the shim by
    /// adjusting queued prefill tokens before calling this helper.
    fn get_queued_prefill_time(
        &self,
        py: Python<'_>,
        metrics_by_rank: &Bound<'_, PyAny>,
    ) -> PyResult<Option<f64>> {
        let metrics_by_rank = metrics_by_rank_from_py(metrics_by_rank)?;
        let model = self.inner.clone();
        Ok(py
            .allow_threads(move || model.get_queued_prefill_time(&metrics_by_rank))
            .map_err(to_pyerr)?
            .map(|duration| duration.as_secs_f64()))
    }

    /// Estimate scheduled decode ITL in seconds; aggregated workers include scheduled or learned average prefill load.
    fn get_scheduled_decode_itl(
        &self,
        metrics_by_rank: &Bound<'_, PyAny>,
    ) -> PyResult<Option<f64>> {
        Ok(self
            .inner
            .get_scheduled_decode_itl(&metrics_by_rank_from_py(metrics_by_rank)?)
            .map_err(to_pyerr)?
            .map(|duration| duration.as_secs_f64()))
    }

    /// Search sustainable per-engine RPS; inspect eligible to see whether eligible SLA metrics passed.
    fn find_engine_capacity_rps(
        &self,
        py: Python<'_>,
        request: PyRef<'_, EngineCapacityRequest>,
    ) -> PyResult<Option<EngineCapacity>> {
        let model = self.inner.clone();
        let request = request.inner.clone();
        Ok(py
            .allow_threads(move || model.find_engine_capacity_rps(request))
            .map_err(to_pyerr)?
            .map(|inner| EngineCapacity { inner }))
    }
}

fn metrics_by_rank_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ForwardPassSnapshot>> {
    if obj.hasattr("scheduled_requests")? {
        return Ok(vec![snapshot_from_py_fpm(obj)?]);
    }

    if let Ok(mapping) = obj.downcast::<PyMapping>() {
        return mapping
            .values()?
            .try_iter()?
            .map(|item| snapshot_from_py_fpm(&item?))
            .collect();
    }

    let iter = obj.try_iter().map_err(|err| {
        PyValueError::new_err(format!(
            "metrics_by_rank must be a ForwardPassMetrics object, a mapping of rank to FPM, or an iterable of FPM objects: {err}"
        ))
    })?;
    iter.map(|item| snapshot_from_py_fpm(&item?)).collect()
}

fn iterations_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<ForwardPassSnapshot>>> {
    if obj.hasattr("scheduled_requests")? {
        return Ok(vec![vec![snapshot_from_py_fpm(obj)?]]);
    }

    if let Ok(mapping) = obj.downcast::<PyMapping>() {
        if mapping_values_are_fpms(mapping)? {
            return Ok(vec![metrics_by_rank_from_py(obj)?]);
        }
        return mapping
            .values()?
            .try_iter()?
            .map(|item| iteration_from_py_item(&item?))
            .collect();
    }

    let iter = obj.try_iter().map_err(|err| {
        PyValueError::new_err(format!(
            "iterations must be a ForwardPassMetrics object, a mapping of iteration to FPM/rank-list, or an iterable of FPM/rank-list items: {err}"
        ))
    })?;
    iter.map(|item| iteration_from_py_item(&item?)).collect()
}

fn iteration_from_py_item(item: &Bound<'_, PyAny>) -> PyResult<Vec<ForwardPassSnapshot>> {
    if item.hasattr("scheduled_requests")? {
        Ok(vec![snapshot_from_py_fpm(item)?])
    } else {
        metrics_by_rank_from_py(item)
    }
}

fn mapping_values_are_fpms(mapping: &Bound<'_, PyMapping>) -> PyResult<bool> {
    for item in mapping.values()?.try_iter()? {
        if !item?.hasattr("scheduled_requests")? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn snapshot_from_py_fpm(fpm: &Bound<'_, PyAny>) -> PyResult<ForwardPassSnapshot> {
    let scheduled = fpm.getattr("scheduled_requests")?;
    let queued = fpm.getattr("queued_requests")?;
    Ok(ForwardPassSnapshot {
        num_prefill_requests: attr_u32(&scheduled, "num_prefill_requests")?,
        sum_prefill_tokens: attr_u64(&scheduled, "sum_prefill_tokens")?,
        var_prefill_length: attr_f64(&scheduled, "var_prefill_length")?,
        sum_prefill_kv_tokens: attr_u64(&scheduled, "sum_prefill_kv_tokens")?,
        num_decode_requests: attr_u32(&scheduled, "num_decode_requests")?,
        sum_decode_kv_tokens: attr_u64(&scheduled, "sum_decode_kv_tokens")?,
        var_decode_kv_tokens: attr_f64(&scheduled, "var_decode_kv_tokens")?,
        num_queued_prefill: attr_u32(&queued, "num_prefill_requests")?,
        sum_queued_prefill_tokens: attr_u64(&queued, "sum_prefill_tokens")?,
        var_queued_prefill_length: attr_f64(&queued, "var_prefill_length")?,
        num_queued_decode: attr_u32(&queued, "num_decode_requests")?,
        sum_queued_decode_kv_tokens: attr_u64(&queued, "sum_decode_kv_tokens")?,
        var_queued_decode_kv_tokens: attr_f64(&queued, "var_decode_kv_tokens")?,
        wall_time_secs: attr_f64(fpm, "wall_time")?,
        version: attr_u32(fpm, "version")?,
        worker_id: attr_string(fpm, "worker_id")?,
        dp_rank: attr_u32(fpm, "dp_rank")?,
        counter_id: attr_u64(fpm, "counter_id")?,
    })
}

fn attr_u32(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<u32> {
    let value = attr_u64(obj, name)?;
    u32::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name}={value} exceeds u32::MAX")))
}

fn attr_u64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<u64> {
    obj.getattr(name)?.extract::<u64>().map_err(|err| {
        PyValueError::new_err(format!("failed to extract integer field {name}: {err}"))
    })
}

fn attr_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    obj.getattr(name)?.extract::<f64>().map_err(|err| {
        PyValueError::new_err(format!("failed to extract float field {name}: {err}"))
    })
}

fn attr_string(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    obj.getattr(name)?.extract::<String>().map_err(|err| {
        PyValueError::new_err(format!("failed to extract string field {name}: {err}"))
    })
}

fn parse_worker_type(value: &str) -> PyResult<RsWorkerType> {
    match value {
        "aggregated" => Ok(RsWorkerType::Aggregated),
        "prefill" => Ok(RsWorkerType::Prefill),
        "decode" => Ok(RsWorkerType::Decode),
        other => Err(PyValueError::new_err(format!(
            "invalid worker_type {other:?}; expected aggregated, prefill, or decode"
        ))),
    }
}

fn ms_to_duration(ms: f64) -> PyResult<Duration> {
    if !ms.is_finite() {
        return Err(PyValueError::new_err(format!(
            "SLA duration must be finite, got {ms}"
        )));
    }
    Duration::try_from_secs_f64(ms.max(0.0) / 1000.0)
        .map_err(|err| PyValueError::new_err(format!("invalid SLA duration {ms} ms: {err}")))
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn to_pyerr<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
