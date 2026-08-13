// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use dynamo_mocker::common::perf_model::PerfModel;
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType as RsMockerEngineType, G1Backend as RsG1Backend,
    MockEngineArgs as RsMockEngineArgs, PreemptionMode as RsPreemptionMode,
    ReasoningConfig as RsReasoningConfig, SglangArgs as RsSglangArgs, TrtllmArgs as RsTrtllmArgs,
    WorkerType as RsWorkerType,
};
use dynamo_mocker::loadgen::{
    ArrivalSpec, DelaySpec, DynamoRequestTrace, LengthSpec, SyntheticTraceSpec, Trace as RsTrace,
};
use dynamo_mocker::replay::{
    CapturedReplayEvent as RsCapturedReplayEvent,
    CapturedReplayEventData as RsCapturedReplayEventData,
    CapturedReplayEventDataView as RsCapturedReplayEventDataView,
    OfflineReplaySession as RsOfflineReplaySession, PoolRouter as RsPoolRouter,
    PoolSpec as RsPoolSpec, ReplayAgenticRequest as RsReplayAgenticRequest,
    ReplayAgenticWorkflow as RsReplayAgenticWorkflow, ReplayArgsMode, ReplayEvent as RsReplayEvent,
    ReplayEventData as RsReplayEventData, ReplayPendingPlacement as RsReplayPendingPlacement,
    ReplayPlacementCandidate as RsReplayPlacementCandidate,
    ReplayRequestSpec as RsReplayRequestSpec,
    ReplayRoutingConstraints as RsReplayRoutingConstraints, ReplayScalingDecision,
    ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySessionOptions as RsReplaySessionOptions,
    ReplaySessionRouter as RsReplaySessionRouter, ReplayStepStatus as RsReplayStepStatus,
    ReplayTerminalStatus as RsReplayTerminalStatus, WorkerSpec as RsWorkerSpec,
    WorkerTarget as RsWorkerTarget,
};
use pyo3::{
    exceptions::{PyException, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyString},
};
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use super::aic_callback::{
    create_aic_callback, create_aic_prefill_load_estimator, estimate_aic_num_gpu_blocks,
};
use super::entrypoint::{AicPerfConfig, KvRouterConfig, to_pyerr};

const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.9;
const DEFAULT_MEM_FRACTION_STATIC: f64 = 0.88;

#[derive(Debug, Serialize)]
struct OfflineReplayCoverage {
    capture_per_request: bool,
    capture_planner_details: bool,
    per_request_records: usize,
}

#[pyclass(name = "_OfflineReplayResult")]
#[derive(Debug)]
pub struct OfflineReplayResult {
    report: dynamo_mocker::replay::TraceSimulationReport,
    lifecycle_operations: Vec<dynamo_mocker::replay::LifecycleOperation>,
    capture_per_request: bool,
    coverage: OfflineReplayCoverage,
}

impl OfflineReplayResult {
    fn new(
        report: dynamo_mocker::replay::TraceSimulationReport,
        capture_per_request: bool,
        capture_planner_details: bool,
        runtime_evidence: dynamo_mocker::replay::OfflineRuntimeEvidence,
    ) -> Self {
        let dynamo_mocker::replay::OfflineRuntimeEvidence {
            lifecycle_operations,
            ..
        } = runtime_evidence;
        let coverage = OfflineReplayCoverage {
            capture_per_request,
            capture_planner_details,
            per_request_records: report.per_request.len(),
        };
        Self {
            report,
            lifecycle_operations,
            capture_per_request,
            coverage,
        }
    }

    fn from_interactive(report: dynamo_mocker::replay::TraceSimulationReport) -> Self {
        let coverage = OfflineReplayCoverage {
            capture_per_request: true,
            capture_planner_details: false,
            per_request_records: report.per_request.len(),
        };
        Self {
            report,
            lifecycle_operations: Vec::new(),
            capture_per_request: true,
            coverage,
        }
    }
}

#[pymethods]
impl OfflineReplayResult {
    #[getter]
    fn summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.report)
            .map(Bound::unbind)
            .map_err(to_pyerr)
    }

    #[getter]
    fn per_request(&self, py: Python<'_>) -> PyResult<PyObject> {
        if !self.capture_per_request {
            return Ok(py.None());
        }
        pythonize(py, &self.report.per_request)
            .map(Bound::unbind)
            .map_err(to_pyerr)
    }

    #[getter]
    fn coverage(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.coverage)
            .map(Bound::unbind)
            .map_err(to_pyerr)
    }

    #[getter]
    fn lifecycle_operations(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.lifecycle_operations)
            .map(Bound::unbind)
            .map_err(to_pyerr)
    }
}

struct ResolvedAicPerfConfig<'a> {
    config: &'a AicPerfConfig,
    backend_version: String,
}

fn resolve_aic_perf_config<'a>(
    py: Python<'_>,
    config: Option<&'a AicPerfConfig>,
) -> PyResult<Option<ResolvedAicPerfConfig<'a>>> {
    config
        .map(|config| {
            Ok(ResolvedAicPerfConfig {
                config,
                backend_version: resolve_aic_backend_version(
                    py,
                    config.backend_name(),
                    config.backend_version(),
                )?,
            })
        })
        .transpose()
}

fn parse_mocker_engine_type(engine_type: &str) -> PyResult<RsMockerEngineType> {
    match engine_type {
        "vllm" => Ok(RsMockerEngineType::Vllm),
        "sglang" => Ok(RsMockerEngineType::Sglang),
        "trtllm" => Ok(RsMockerEngineType::Trtllm),
        other => Err(PyException::new_err(format!(
            "engine_type must be one of 'vllm', 'sglang', or 'trtllm', got '{other}'"
        ))),
    }
}

fn parse_worker_type(worker_type: &str) -> PyResult<RsWorkerType> {
    match worker_type {
        "aggregated" => Ok(RsWorkerType::Aggregated),
        "prefill" => Ok(RsWorkerType::Prefill),
        "decode" => Ok(RsWorkerType::Decode),
        other => Err(PyException::new_err(format!(
            "worker_type must be one of 'aggregated', 'prefill', or 'decode', got '{other}'"
        ))),
    }
}

fn parse_preemption_mode(preemption_mode: &str) -> PyResult<RsPreemptionMode> {
    match preemption_mode {
        "lifo" => Ok(RsPreemptionMode::Lifo),
        "fifo" => Ok(RsPreemptionMode::Fifo),
        other => Err(PyException::new_err(format!(
            "preemption_mode must be either 'lifo' or 'fifo', got '{other}'"
        ))),
    }
}

fn parse_g1_backend(backend: &str) -> PyResult<RsG1Backend> {
    match backend {
        "kvbm" => Ok(RsG1Backend::Kvbm),
        "native" => Ok(RsG1Backend::Native),
        other => Err(PyException::new_err(format!(
            "g1_backend must be either 'kvbm' or 'native', got '{other}'"
        ))),
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ReasoningConfig {
    inner: RsReasoningConfig,
}

impl ReasoningConfig {
    pub fn inner(&self) -> RsReasoningConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl ReasoningConfig {
    #[new]
    fn new(
        start_thinking_token_id: u32,
        end_thinking_token_id: u32,
        thinking_ratio: f64,
    ) -> PyResult<Self> {
        let inner = RsReasoningConfig {
            start_thinking_token_id,
            end_thinking_token_id,
            thinking_ratio,
        };
        Ok(Self { inner })
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct SglangArgs {
    inner: RsSglangArgs,
}

impl SglangArgs {
    pub fn inner(&self) -> RsSglangArgs {
        self.inner.clone()
    }
}

#[pymethods]
impl SglangArgs {
    #[new]
    #[pyo3(signature = (schedule_policy=None, page_size=None, max_prefill_tokens=None, chunked_prefill_size=None, clip_max_new_tokens=None, schedule_conservativeness=None))]
    fn new(
        schedule_policy: Option<String>,
        page_size: Option<usize>,
        max_prefill_tokens: Option<usize>,
        chunked_prefill_size: Option<usize>,
        clip_max_new_tokens: Option<usize>,
        schedule_conservativeness: Option<f64>,
    ) -> PyResult<Self> {
        let inner = RsSglangArgs {
            schedule_policy,
            page_size,
            max_prefill_tokens,
            chunked_prefill_size,
            clip_max_new_tokens,
            schedule_conservativeness,
        };
        Ok(Self { inner })
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct TrtllmArgs {
    inner: RsTrtllmArgs,
}

impl TrtllmArgs {
    pub fn inner(&self) -> RsTrtllmArgs {
        self.inner.clone()
    }
}

#[pymethods]
impl TrtllmArgs {
    #[new]
    #[pyo3(signature = (capacity_scheduler_policy=None))]
    fn new(capacity_scheduler_policy: Option<String>) -> PyResult<Self> {
        let inner = RsTrtllmArgs {
            capacity_scheduler_policy,
        };
        Ok(Self { inner })
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct MockEngineArgs {
    inner: RsMockEngineArgs,
    num_gpu_blocks_explicit: bool,
}

impl MockEngineArgs {
    pub fn inner(&self) -> RsMockEngineArgs {
        self.inner.clone()
    }

    pub(crate) fn num_gpu_blocks_explicit(&self) -> bool {
        self.num_gpu_blocks_explicit
    }
}

#[pymethods]
impl MockEngineArgs {
    #[new]
    #[pyo3(signature = (engine_type="vllm", num_gpu_blocks=None, block_size=0, max_num_seqs=Some(256), max_num_batched_tokens=Some(8192), enable_prefix_caching=true, enable_chunked_prefill=true, speedup_ratio=1.0, decode_speedup_ratio=1.0, dp_size=1, startup_time=None, worker_type="aggregated", planner_profile_data=None, aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, aic_nextn=None, aic_nextn_accept_rates=None, aic_mtp_seed=42, aic_gemm_dtype=None, aic_moe_dtype=None, aic_fmha_dtype=None, aic_kv_cache_dtype=None, aic_comm_dtype=None, gpu_memory_utilization=None, mem_fraction_static=None, free_gpu_memory_fraction=None, enable_local_indexer=false, bootstrap_port=None, handoff_session_timeout_ms=300000, kv_bytes_per_token=None, kv_transfer_bandwidth=None, kv_transfer_timing_mode="full_prompt", reasoning=None, response_replay_trace_path=None, zmq_kv_events_port=None, zmq_replay_port=None, preemption_mode="lifo", router_queue_policy=None, sglang=None, trtllm=None, num_g2_blocks=None, num_g3_blocks=None, offload_batch_size=None, bandwidth_g1_to_g2_gbps=None, bandwidth_g2_to_g1_gbps=None, bandwidth_g2_to_g3_gbps=None, bandwidth_g3_to_g2_gbps=None, enable_g4_storage=false, bandwidth_g2_to_g4_gbps=None, bandwidth_g4_to_g2_gbps=None, max_model_len=None, g1_backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine_type: &str,
        num_gpu_blocks: Option<usize>,
        block_size: usize,
        max_num_seqs: Option<usize>,
        max_num_batched_tokens: Option<usize>,
        enable_prefix_caching: bool,
        enable_chunked_prefill: bool,
        speedup_ratio: f64,
        decode_speedup_ratio: f64,
        dp_size: u32,
        startup_time: Option<f64>,
        worker_type: &str,
        planner_profile_data: Option<PathBuf>,
        aic_backend: Option<String>,
        aic_system: Option<String>,
        aic_backend_version: Option<String>,
        aic_tp_size: Option<usize>,
        aic_model_path: Option<String>,
        aic_moe_tp_size: Option<usize>,
        aic_moe_ep_size: Option<usize>,
        aic_attention_dp_size: Option<usize>,
        aic_nextn: Option<usize>,
        aic_nextn_accept_rates: Option<String>,
        aic_mtp_seed: u64,
        aic_gemm_dtype: Option<String>,
        aic_moe_dtype: Option<String>,
        aic_fmha_dtype: Option<String>,
        aic_kv_cache_dtype: Option<String>,
        aic_comm_dtype: Option<String>,
        gpu_memory_utilization: Option<f64>,
        mem_fraction_static: Option<f64>,
        free_gpu_memory_fraction: Option<f64>,
        enable_local_indexer: bool,
        bootstrap_port: Option<u16>,
        handoff_session_timeout_ms: u64,
        kv_bytes_per_token: Option<usize>,
        kv_transfer_bandwidth: Option<f64>,
        kv_transfer_timing_mode: &str,
        reasoning: Option<ReasoningConfig>,
        response_replay_trace_path: Option<PathBuf>,
        zmq_kv_events_port: Option<u16>,
        zmq_replay_port: Option<u16>,
        preemption_mode: &str,
        router_queue_policy: Option<&str>,
        sglang: Option<SglangArgs>,
        trtllm: Option<TrtllmArgs>,
        num_g2_blocks: Option<usize>,
        num_g3_blocks: Option<usize>,
        offload_batch_size: Option<usize>,
        bandwidth_g1_to_g2_gbps: Option<f64>,
        bandwidth_g2_to_g1_gbps: Option<f64>,
        bandwidth_g2_to_g3_gbps: Option<f64>,
        bandwidth_g3_to_g2_gbps: Option<f64>,
        enable_g4_storage: bool,
        bandwidth_g2_to_g4_gbps: Option<f64>,
        bandwidth_g4_to_g2_gbps: Option<f64>,
        max_model_len: Option<usize>,
        g1_backend: Option<&str>,
    ) -> PyResult<Self> {
        let engine_type = parse_mocker_engine_type(engine_type)?;
        let worker_type = parse_worker_type(worker_type)?;
        let preemption_mode = parse_preemption_mode(preemption_mode)?;
        let g1_backend = g1_backend.map(parse_g1_backend).transpose()?;
        let kv_transfer_timing_mode = kv_transfer_timing_mode
            .parse()
            .map_err(|error: String| PyException::new_err(error))?;
        let router_queue_policy = router_queue_policy
            .map(|value| {
                value.parse().map_err(|e: String| {
                    PyException::new_err(format!("invalid router_queue_policy {value:?}: {e}"))
                })
            })
            .transpose()?;

        let mut builder = RsMockEngineArgs::builder()
            .engine_type(engine_type)
            .block_size(block_size)
            .max_model_len(max_model_len)
            .max_num_seqs(max_num_seqs)
            .max_num_batched_tokens(max_num_batched_tokens)
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .speedup_ratio(speedup_ratio)
            .decode_speedup_ratio(decode_speedup_ratio)
            .dp_size(dp_size)
            .startup_time(startup_time)
            .worker_type(worker_type)
            .planner_profile_data(planner_profile_data.clone())
            .aic_backend(aic_backend)
            .aic_system(aic_system)
            .aic_backend_version(aic_backend_version)
            .aic_tp_size(aic_tp_size)
            .aic_model_path(aic_model_path)
            .aic_moe_tp_size(aic_moe_tp_size)
            .aic_moe_ep_size(aic_moe_ep_size)
            .aic_attention_dp_size(aic_attention_dp_size)
            .aic_gemm_dtype(aic_gemm_dtype)
            .aic_moe_dtype(aic_moe_dtype)
            .aic_fmha_dtype(aic_fmha_dtype)
            .aic_kv_cache_dtype(aic_kv_cache_dtype)
            .aic_comm_dtype(aic_comm_dtype)
            .aic_nextn(aic_nextn)
            .aic_nextn_accept_rates(aic_nextn_accept_rates)
            .aic_mtp_seed(aic_mtp_seed)
            .gpu_memory_utilization(gpu_memory_utilization)
            .mem_fraction_static(mem_fraction_static)
            .free_gpu_memory_fraction(free_gpu_memory_fraction)
            .enable_local_indexer(enable_local_indexer)
            .bootstrap_port(bootstrap_port)
            .handoff_session_timeout_ms(handoff_session_timeout_ms)
            .kv_bytes_per_token(kv_bytes_per_token)
            .kv_transfer_bandwidth(kv_transfer_bandwidth)
            .kv_transfer_timing_mode(kv_transfer_timing_mode)
            .num_g2_blocks(num_g2_blocks)
            .num_g3_blocks(num_g3_blocks)
            .enable_g4_storage(enable_g4_storage)
            .offload_batch_size(offload_batch_size)
            .bandwidth_g1_to_g2_gbps(bandwidth_g1_to_g2_gbps)
            .bandwidth_g2_to_g1_gbps(bandwidth_g2_to_g1_gbps)
            .bandwidth_g2_to_g3_gbps(bandwidth_g2_to_g3_gbps)
            .bandwidth_g3_to_g2_gbps(bandwidth_g3_to_g2_gbps)
            .bandwidth_g2_to_g4_gbps(bandwidth_g2_to_g4_gbps)
            .bandwidth_g4_to_g2_gbps(bandwidth_g4_to_g2_gbps)
            .reasoning(reasoning.map(|config| config.inner()))
            .response_replay_trace_path(response_replay_trace_path)
            .zmq_kv_events_port(zmq_kv_events_port)
            .zmq_replay_port(zmq_replay_port)
            .preemption_mode(preemption_mode)
            .router_queue_policy(router_queue_policy)
            .sglang(sglang.map(|config| config.inner()))
            .trtllm(trtllm.map(|config| config.inner()));
        let num_gpu_blocks_explicit = num_gpu_blocks.is_some();
        if let Some(num_gpu_blocks) = num_gpu_blocks {
            builder = builder.num_gpu_blocks(num_gpu_blocks);
        }
        if let Some(g1_backend) = g1_backend {
            builder = builder.g1_backend(g1_backend);
        }

        if let Some(npz_path) = planner_profile_data {
            let perf_model = PerfModel::from_npz(&npz_path).map_err(|e| {
                PyException::new_err(format!(
                    "Failed to load planner_profile_data from {:?}: {e}",
                    npz_path
                ))
            })?;
            builder = builder.perf_model(Arc::new(perf_model));
        }

        let inner = builder
            .build()
            .map_err(|e| PyException::new_err(format!("Failed to build MockEngineArgs: {e}")))?
            .normalized()
            .map_err(|e| {
                PyException::new_err(format!("Failed to normalize MockEngineArgs: {e}"))
            })?;

        Ok(Self {
            inner,
            num_gpu_blocks_explicit,
        })
    }

    #[staticmethod]
    fn from_json(config_json: &str) -> PyResult<Self> {
        let num_gpu_blocks_explicit = serde_json::from_str::<serde_json::Value>(config_json)
            .ok()
            .and_then(|value| {
                value.as_object().map(|object| {
                    object
                        .get("num_gpu_blocks")
                        .and_then(|value| value.as_u64())
                        .is_some()
                })
            })
            .unwrap_or(false);
        RsMockEngineArgs::from_json_str(config_json)
            .map(|inner| Self {
                inner,
                num_gpu_blocks_explicit,
            })
            .map_err(|e| PyException::new_err(format!("Failed to parse MockEngineArgs JSON: {e}")))
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size
    }

    #[getter]
    fn num_gpu_blocks(&self) -> usize {
        self.inner.num_gpu_blocks
    }

    #[getter]
    fn max_model_len(&self) -> Option<usize> {
        self.inner.max_model_len
    }

    #[getter]
    fn max_num_seqs(&self) -> Option<usize> {
        self.inner.max_num_seqs
    }

    #[getter]
    fn max_num_batched_tokens(&self) -> Option<usize> {
        self.inner.max_num_batched_tokens
    }

    #[getter]
    fn enable_prefix_caching(&self) -> bool {
        self.inner.enable_prefix_caching
    }

    #[getter]
    fn g1_backend(&self) -> &'static str {
        match self.inner.resolved_g1_backend() {
            RsG1Backend::Kvbm => "kvbm",
            RsG1Backend::Native => "native",
        }
    }

    #[setter]
    fn set_enable_prefix_caching(&mut self, value: bool) {
        self.inner.enable_prefix_caching = value;
    }

    #[getter]
    fn enable_local_indexer(&self) -> bool {
        self.inner.enable_local_indexer
    }

    #[getter]
    fn dp_size(&self) -> u32 {
        self.inner.dp_size
    }

    #[getter]
    fn bootstrap_port(&self) -> Option<u16> {
        self.inner.bootstrap_port
    }

    #[getter]
    fn handoff_session_timeout_ms(&self) -> u64 {
        self.inner.handoff_session_timeout_ms
    }

    #[getter]
    fn kv_transfer_timing_mode(&self) -> &'static str {
        match self.inner.kv_transfer_timing_mode {
            dynamo_mocker::common::protocols::KvTransferTimingMode::FullPrompt => "full_prompt",
            dynamo_mocker::common::protocols::KvTransferTimingMode::DestinationMissing => {
                "destination_missing"
            }
        }
    }

    #[getter]
    fn engine_type(&self) -> &'static str {
        match self.inner.engine_type {
            dynamo_mocker::common::protocols::EngineType::Vllm => "vllm",
            dynamo_mocker::common::protocols::EngineType::Sglang => "sglang",
            dynamo_mocker::common::protocols::EngineType::Trtllm => "trtllm",
        }
    }

    #[getter]
    fn kv_bytes_per_token(&self) -> Option<usize> {
        self.inner.kv_bytes_per_token
    }

    #[getter]
    fn response_replay_trace_path(&self) -> Option<PathBuf> {
        self.inner.response_replay_trace_path.clone()
    }

    #[getter]
    fn num_g2_blocks(&self) -> Option<usize> {
        self.inner.num_g2_blocks
    }

    #[getter]
    fn num_g3_blocks(&self) -> Option<usize> {
        self.inner.num_g3_blocks
    }

    #[getter]
    fn enable_g4_storage(&self) -> bool {
        self.inner.enable_g4_storage
    }

    #[getter]
    fn offload_batch_size(&self) -> Option<usize> {
        self.inner.offload_batch_size
    }

    #[getter]
    fn bandwidth_g1_to_g2_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g1_to_g2_gbps
    }

    #[getter]
    fn bandwidth_g2_to_g1_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g2_to_g1_gbps
    }

    #[getter]
    fn bandwidth_g2_to_g3_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g2_to_g3_gbps
    }

    #[getter]
    fn bandwidth_g3_to_g2_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g3_to_g2_gbps
    }

    #[getter]
    fn bandwidth_g2_to_g4_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g2_to_g4_gbps
    }

    #[getter]
    fn bandwidth_g4_to_g2_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_g4_to_g2_gbps
    }

    #[getter]
    fn aic_backend(&self) -> Option<String> {
        self.inner.aic_backend.clone()
    }

    #[setter]
    fn set_aic_backend(&mut self, value: Option<String>) {
        self.inner.aic_backend = value;
    }

    #[getter]
    fn aic_system(&self) -> Option<String> {
        self.inner.aic_system.clone()
    }

    #[setter]
    fn set_aic_system(&mut self, value: Option<String>) {
        self.inner.aic_system = value;
    }

    #[getter]
    fn aic_backend_version(&self) -> Option<String> {
        self.inner.aic_backend_version.clone()
    }

    #[setter]
    fn set_aic_backend_version(&mut self, value: Option<String>) {
        self.inner.aic_backend_version = value;
    }

    #[getter]
    fn aic_tp_size(&self) -> Option<usize> {
        self.inner.aic_tp_size
    }

    #[setter]
    fn set_aic_tp_size(&mut self, value: Option<usize>) {
        self.inner.aic_tp_size = value;
    }

    #[getter]
    fn aic_model_path(&self) -> Option<String> {
        self.inner.aic_model_path.clone()
    }

    #[setter]
    fn set_aic_model_path(&mut self, value: Option<String>) {
        self.inner.aic_model_path = value;
    }

    #[getter]
    fn aic_moe_tp_size(&self) -> Option<usize> {
        self.inner.aic_moe_tp_size
    }

    #[setter]
    fn set_aic_moe_tp_size(&mut self, value: Option<usize>) {
        self.inner.aic_moe_tp_size = value;
    }

    #[getter]
    fn aic_moe_ep_size(&self) -> Option<usize> {
        self.inner.aic_moe_ep_size
    }

    #[setter]
    fn set_aic_moe_ep_size(&mut self, value: Option<usize>) {
        self.inner.aic_moe_ep_size = value;
    }

    #[getter]
    fn aic_attention_dp_size(&self) -> Option<usize> {
        self.inner.aic_attention_dp_size
    }

    #[setter]
    fn set_aic_attention_dp_size(&mut self, value: Option<usize>) {
        self.inner.aic_attention_dp_size = value;
    }

    #[getter]
    fn aic_gemm_dtype(&self) -> Option<String> {
        self.inner.aic_gemm_dtype.clone()
    }

    #[setter]
    fn set_aic_gemm_dtype(&mut self, value: Option<String>) {
        self.inner.aic_gemm_dtype = value;
    }

    #[getter]
    fn aic_moe_dtype(&self) -> Option<String> {
        self.inner.aic_moe_dtype.clone()
    }

    #[setter]
    fn set_aic_moe_dtype(&mut self, value: Option<String>) {
        self.inner.aic_moe_dtype = value;
    }

    #[getter]
    fn aic_fmha_dtype(&self) -> Option<String> {
        self.inner.aic_fmha_dtype.clone()
    }

    #[setter]
    fn set_aic_fmha_dtype(&mut self, value: Option<String>) {
        self.inner.aic_fmha_dtype = value;
    }

    #[getter]
    fn aic_kv_cache_dtype(&self) -> Option<String> {
        self.inner.aic_kv_cache_dtype.clone()
    }

    #[setter]
    fn set_aic_kv_cache_dtype(&mut self, value: Option<String>) {
        self.inner.aic_kv_cache_dtype = value;
    }

    #[getter]
    fn aic_comm_dtype(&self) -> Option<String> {
        self.inner.aic_comm_dtype.clone()
    }

    #[setter]
    fn set_aic_comm_dtype(&mut self, value: Option<String>) {
        self.inner.aic_comm_dtype = value;
    }

    #[getter]
    fn aic_nextn(&self) -> Option<usize> {
        self.inner.aic_nextn
    }

    #[setter]
    fn set_aic_nextn(&mut self, value: Option<usize>) {
        self.inner.aic_nextn = value;
    }

    #[getter]
    fn aic_nextn_accept_rates(&self) -> Option<String> {
        self.inner.aic_nextn_accept_rates.clone()
    }

    #[setter]
    fn set_aic_nextn_accept_rates(&mut self, value: Option<String>) {
        self.inner.aic_nextn_accept_rates = value;
    }

    #[getter]
    fn aic_mtp_seed(&self) -> u64 {
        self.inner.aic_mtp_seed
    }

    #[setter]
    fn set_aic_mtp_seed(&mut self, value: u64) {
        self.inner.aic_mtp_seed = value;
    }

    #[getter]
    fn gpu_memory_utilization(&self) -> Option<f64> {
        self.inner.gpu_memory_utilization
    }

    #[setter]
    fn set_gpu_memory_utilization(&mut self, value: Option<f64>) -> PyResult<()> {
        if let Some(value) = value
            && !(0.0..=1.0).contains(&value)
        {
            return Err(PyValueError::new_err(format!(
                "gpu_memory_utilization must be in [0, 1], got {value}"
            )));
        }
        self.inner.gpu_memory_utilization = value;
        Ok(())
    }

    #[getter]
    fn mem_fraction_static(&self) -> Option<f64> {
        self.inner.mem_fraction_static
    }

    #[setter]
    fn set_mem_fraction_static(&mut self, value: Option<f64>) -> PyResult<()> {
        if let Some(value) = value
            && !(0.0..=1.0).contains(&value)
        {
            return Err(PyValueError::new_err(format!(
                "mem_fraction_static must be in [0, 1], got {value}"
            )));
        }
        self.inner.mem_fraction_static = value;
        Ok(())
    }

    #[getter]
    fn free_gpu_memory_fraction(&self) -> Option<f64> {
        self.inner.free_gpu_memory_fraction
    }

    #[setter]
    fn set_free_gpu_memory_fraction(&mut self, value: Option<f64>) -> PyResult<()> {
        if let Some(value) = value
            && !(0.0..=1.0).contains(&value)
        {
            return Err(PyValueError::new_err(format!(
                "free_gpu_memory_fraction must be in [0, 1], got {value}"
            )));
        }
        self.inner.free_gpu_memory_fraction = value;
        Ok(())
    }

    #[getter]
    fn worker_type(&self) -> &'static str {
        match self.inner.worker_type {
            RsWorkerType::Aggregated => "aggregated",
            RsWorkerType::Prefill => "prefill",
            RsWorkerType::Decode => "decode",
        }
    }

    #[setter]
    fn set_worker_type(&mut self, value: &str) -> PyResult<()> {
        self.inner.worker_type = parse_worker_type(value)?;
        Ok(())
    }

    #[setter]
    fn set_num_gpu_blocks(&mut self, value: usize) {
        self.inner.num_gpu_blocks = value;
        self.num_gpu_blocks_explicit = true;
    }

    fn is_prefill(&self) -> bool {
        self.inner.is_prefill()
    }

    fn is_decode(&self) -> bool {
        self.inner.is_decode()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (bootstrap_port=None, zmq_kv_events_port=None, zmq_replay_port=None, kv_bytes_per_token=None, num_gpu_blocks=None, aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, aic_nextn=None, aic_nextn_accept_rates=None, aic_mtp_seed=None, aic_gemm_dtype=None, aic_moe_dtype=None, aic_fmha_dtype=None, aic_kv_cache_dtype=None, aic_comm_dtype=None, gpu_memory_utilization=None, mem_fraction_static=None, free_gpu_memory_fraction=None, enable_prefix_caching=None, worker_type=None))]
    fn with_overrides(
        &self,
        bootstrap_port: Option<u16>,
        zmq_kv_events_port: Option<u16>,
        zmq_replay_port: Option<u16>,
        kv_bytes_per_token: Option<usize>,
        num_gpu_blocks: Option<usize>,
        aic_backend: Option<String>,
        aic_system: Option<String>,
        aic_backend_version: Option<String>,
        aic_tp_size: Option<usize>,
        aic_model_path: Option<String>,
        aic_moe_tp_size: Option<usize>,
        aic_moe_ep_size: Option<usize>,
        aic_attention_dp_size: Option<usize>,
        aic_nextn: Option<usize>,
        aic_nextn_accept_rates: Option<String>,
        aic_mtp_seed: Option<u64>,
        aic_gemm_dtype: Option<String>,
        aic_moe_dtype: Option<String>,
        aic_fmha_dtype: Option<String>,
        aic_kv_cache_dtype: Option<String>,
        aic_comm_dtype: Option<String>,
        gpu_memory_utilization: Option<f64>,
        mem_fraction_static: Option<f64>,
        free_gpu_memory_fraction: Option<f64>,
        enable_prefix_caching: Option<bool>,
        worker_type: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = self.inner.clone();
        let mut num_gpu_blocks_explicit = self.num_gpu_blocks_explicit;
        if let Some(port) = bootstrap_port {
            inner.bootstrap_port = Some(port);
        }
        if let Some(port) = zmq_kv_events_port {
            inner.zmq_kv_events_port = Some(port);
        }
        if let Some(port) = zmq_replay_port {
            inner.zmq_replay_port = Some(port);
        }
        if let Some(bytes_per_token) = kv_bytes_per_token {
            inner.kv_bytes_per_token = Some(bytes_per_token);
        }
        if let Some(blocks) = num_gpu_blocks {
            inner.num_gpu_blocks = blocks;
            num_gpu_blocks_explicit = true;
        }
        if let Some(backend) = aic_backend {
            inner.aic_backend = Some(backend);
        }
        if let Some(system) = aic_system {
            inner.aic_system = Some(system);
        }
        if let Some(version) = aic_backend_version {
            inner.aic_backend_version = Some(version);
        }
        if let Some(tp_size) = aic_tp_size {
            inner.aic_tp_size = Some(tp_size);
        }
        if let Some(model_path) = aic_model_path {
            inner.aic_model_path = Some(model_path);
        }
        if let Some(moe_tp_size) = aic_moe_tp_size {
            inner.aic_moe_tp_size = Some(moe_tp_size);
        }
        if let Some(moe_ep_size) = aic_moe_ep_size {
            inner.aic_moe_ep_size = Some(moe_ep_size);
        }
        if let Some(attention_dp_size) = aic_attention_dp_size {
            inner.aic_attention_dp_size = Some(attention_dp_size);
        }
        if let Some(dtype) = aic_gemm_dtype {
            inner.aic_gemm_dtype = Some(dtype);
        }
        if let Some(dtype) = aic_moe_dtype {
            inner.aic_moe_dtype = Some(dtype);
        }
        if let Some(dtype) = aic_fmha_dtype {
            inner.aic_fmha_dtype = Some(dtype);
        }
        if let Some(dtype) = aic_kv_cache_dtype {
            inner.aic_kv_cache_dtype = Some(dtype);
        }
        if let Some(dtype) = aic_comm_dtype {
            inner.aic_comm_dtype = Some(dtype);
        }
        if let Some(nextn) = aic_nextn {
            inner.aic_nextn = Some(nextn);
        }
        if let Some(rates) = aic_nextn_accept_rates {
            inner.aic_nextn_accept_rates = Some(rates);
        }
        if let Some(seed) = aic_mtp_seed {
            inner.aic_mtp_seed = seed;
        }
        if let Some(gpu_memory_utilization) = gpu_memory_utilization {
            inner.gpu_memory_utilization = Some(gpu_memory_utilization);
        }
        if let Some(mem_fraction_static) = mem_fraction_static {
            inner.mem_fraction_static = Some(mem_fraction_static);
        }
        if let Some(free_gpu_memory_fraction) = free_gpu_memory_fraction {
            inner.free_gpu_memory_fraction = Some(free_gpu_memory_fraction);
        }
        if let Some(enable_prefix_caching) = enable_prefix_caching {
            inner.enable_prefix_caching = enable_prefix_caching;
        }
        if let Some(worker_type) = worker_type {
            inner.worker_type = parse_worker_type(&worker_type)?;
        }
        inner
            .normalized()
            .map(|inner| Self {
                inner,
                num_gpu_blocks_explicit,
            })
            .map_err(|e| {
                PyException::new_err(format!("Failed to normalize MockEngineArgs overrides: {e}"))
            })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractiveWorkerTarget {
    #[serde(default = "default_interactive_pool_id")]
    pool_id: String,
    worker_id: usize,
    #[serde(default)]
    dp_rank: usize,
}

impl From<InteractiveWorkerTarget> for RsWorkerTarget {
    fn from(target: InteractiveWorkerTarget) -> Self {
        Self {
            pool_id: target.pool_id,
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
        }
    }
}

fn default_interactive_pool_id() -> String {
    "default".to_string()
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractiveRoutingConstraints {
    #[serde(default)]
    required_taints: Vec<String>,
    #[serde(default)]
    preferred_taints: BTreeMap<String, f32>,
}

impl From<InteractiveRoutingConstraints> for RsReplayRoutingConstraints {
    fn from(constraints: InteractiveRoutingConstraints) -> Self {
        Self {
            required_taints: constraints.required_taints,
            preferred_taints: constraints.preferred_taints,
        }
    }
}

#[pyclass(name = "_ReplayWorkerSpec")]
#[derive(Clone, Debug)]
pub struct PyReplayWorkerSpec {
    inner: RsWorkerSpec,
}

#[pymethods]
impl PyReplayWorkerSpec {
    #[new]
    #[pyo3(signature = (worker_id, max_num_seqs=None, tags=Vec::new(), taints=Vec::new(), capabilities=Vec::new(), active=true, draining=false))]
    fn new(
        worker_id: usize,
        max_num_seqs: Option<usize>,
        tags: Vec<String>,
        taints: Vec<String>,
        capabilities: Vec<String>,
        active: bool,
        draining: bool,
    ) -> Self {
        Self {
            inner: RsWorkerSpec {
                worker_id,
                max_num_seqs,
                tags,
                taints,
                capabilities,
                active,
                draining,
            },
        }
    }
}

fn parse_pool_router(router: &str) -> PyResult<RsPoolRouter> {
    match router {
        "round_robin" => Ok(RsPoolRouter::RoundRobin),
        other => Err(PyValueError::new_err(format!(
            "interactive replay pool router must be 'round_robin', got {other:?}"
        ))),
    }
}

#[pyclass(name = "_ReplayPoolSpec")]
#[derive(Clone, Debug)]
pub struct PyReplayPoolSpec {
    inner: RsPoolSpec,
}

#[pymethods]
impl PyReplayPoolSpec {
    #[new]
    #[pyo3(signature = (pool_id, engine_args, workers, router="round_robin"))]
    fn new(
        py: Python<'_>,
        pool_id: String,
        engine_args: MockEngineArgs,
        workers: Vec<Py<PyReplayWorkerSpec>>,
        router: &str,
    ) -> PyResult<Self> {
        if engine_args.inner.aic_backend.is_some() {
            return Err(PyValueError::new_err(
                "interactive replay does not support Python-backed AIC performance callbacks",
            ));
        }
        let engine_args = materialize_replay_mocker_args(py, engine_args)?;
        let workers = workers
            .into_iter()
            .map(|worker| worker.borrow(py).inner.clone())
            .collect();
        Ok(Self {
            inner: RsPoolSpec {
                pool_id,
                engine_args,
                workers,
                router: parse_pool_router(router)?,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractiveRequestSpec {
    logical_request_id: String,
    attempt_id: String,
    group_id: String,
    #[serde(default)]
    internal_uuid: Option<Uuid>,
    session_id: String,
    authored_turn_index: usize,
    #[serde(default)]
    ready_time_ms: f64,
    input_length: usize,
    hash_ids: Vec<u32>,
    trace_block_size: usize,
    output_length: usize,
    #[serde(default)]
    output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    priority: i32,
    #[serde(default)]
    strict_priority: u32,
    #[serde(default)]
    policy_class: Option<String>,
    #[serde(default)]
    routing_constraints: InteractiveRoutingConstraints,
    #[serde(default)]
    target: Option<InteractiveWorkerTarget>,
}

impl From<InteractiveRequestSpec> for RsReplayRequestSpec {
    fn from(request: InteractiveRequestSpec) -> Self {
        Self {
            logical_request_id: request.logical_request_id,
            attempt_id: request.attempt_id,
            group_id: request.group_id,
            internal_uuid: request.internal_uuid,
            session_id: request.session_id,
            authored_turn_index: request.authored_turn_index,
            ready_time_ms: request.ready_time_ms,
            input_length: request.input_length,
            hash_ids: request.hash_ids,
            trace_block_size: request.trace_block_size,
            output_length: request.output_length,
            output_token_ids: request.output_token_ids,
            priority: request.priority,
            strict_priority: request.strict_priority,
            policy_class: request.policy_class,
            routing_constraints: request.routing_constraints.into(),
            target: request.target.map(Into::into),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractiveAgenticRequest {
    request: InteractiveRequestSpec,
    #[serde(default)]
    wait_for: Vec<String>,
    #[serde(default)]
    dependency_delay_ms: f64,
    #[serde(default)]
    prefix_reset: bool,
}

impl From<InteractiveAgenticRequest> for RsReplayAgenticRequest {
    fn from(request: InteractiveAgenticRequest) -> Self {
        Self {
            request: request.request.into(),
            wait_for: request.wait_for,
            dependency_delay_ms: request.dependency_delay_ms,
            prefix_reset: request.prefix_reset,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractiveAgenticWorkflow {
    trace_block_size: usize,
    requests: Vec<InteractiveAgenticRequest>,
}

impl From<InteractiveAgenticWorkflow> for RsReplayAgenticWorkflow {
    fn from(workflow: InteractiveAgenticWorkflow) -> Self {
        Self {
            trace_block_size: workflow.trace_block_size,
            requests: workflow.requests.into_iter().map(Into::into).collect(),
        }
    }
}

fn depythonize_interactive<T>(value: &Bound<'_, PyAny>, kind: &str) -> PyResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    depythonize(value).map_err(|error| {
        PyValueError::new_err(format!("invalid interactive replay {kind}: {error}"))
    })
}

fn has_exact_worker_target_fields(value: &Bound<'_, PyDict>) -> PyResult<bool> {
    let mut has_worker_id = false;
    for (key, _) in value.iter() {
        let Ok(key) = key.downcast::<PyString>() else {
            return Ok(false);
        };
        match key.to_str()? {
            "pool_id" | "dp_rank" => {}
            "worker_id" => has_worker_id = true,
            _ => return Ok(false),
        }
    }
    Ok(has_worker_id)
}

/// Parse the high-volume assignment target without constructing a generic
/// Serde map. Invalid/non-dict inputs deliberately take the existing generic
/// path so its exception type and detailed message remain byte-for-byte
/// compatible. PyO3 integer extraction also preserves Serde's current Python
/// behavior for `bool` and `__index__` values.
fn parse_interactive_worker_target(value: &Bound<'_, PyAny>) -> PyResult<InteractiveWorkerTarget> {
    if let Ok(value) = value.downcast::<PyDict>()
        && has_exact_worker_target_fields(value).unwrap_or(false)
    {
        let pool_id = value.get_item(pyo3::intern!(value.py(), "pool_id"))?;
        let worker_id = value
            .get_item(pyo3::intern!(value.py(), "worker_id"))?
            .expect("validated worker target contains worker_id");
        let dp_rank = value.get_item(pyo3::intern!(value.py(), "dp_rank"))?;
        let direct = (|| -> PyResult<InteractiveWorkerTarget> {
            Ok(InteractiveWorkerTarget {
                pool_id: pool_id
                    .as_ref()
                    .map(Bound::extract)
                    .transpose()?
                    .unwrap_or_else(default_interactive_pool_id),
                worker_id: worker_id.extract()?,
                dp_rank: dp_rank
                    .as_ref()
                    .map(Bound::extract)
                    .transpose()?
                    .unwrap_or_default(),
            })
        })();
        if let Ok(target) = direct {
            return Ok(target);
        }
    }

    depythonize_interactive(value, "worker target")
}

fn pythonize_interactive<T>(py: Python<'_>, value: &T) -> PyResult<PyObject>
where
    T: Serialize,
{
    pythonize(py, value).map(Bound::unbind).map_err(to_pyerr)
}

// Events and pending placements are the high-volume interactive boundary. Keep
// these owned conversions in Serde declaration order: the exhaustive matches
// and destructures make additions to the public Rust schema fail compilation.
fn replay_terminal_status_name(status: RsReplayTerminalStatus) -> &'static str {
    match status {
        RsReplayTerminalStatus::Completed => "completed",
        RsReplayTerminalStatus::Rejected => "rejected",
        RsReplayTerminalStatus::Canceled => "canceled",
        RsReplayTerminalStatus::Failed => "failed",
    }
}

fn replay_step_status_parts(status: RsReplayStepStatus) -> (&'static str, f64) {
    match status {
        RsReplayStepStatus::Advanced { now_ms } => ("advanced", now_ms),
        RsReplayStepStatus::Quiescent { now_ms } => ("quiescent", now_ms),
        RsReplayStepStatus::Drained { now_ms } => ("drained", now_ms),
    }
}

fn replay_step_status_to_python(py: Python<'_>, status: RsReplayStepStatus) -> PyResult<PyObject> {
    let (status, now_ms) = replay_step_status_parts(status);
    let value = PyDict::new(py);
    // Preserve the internally-tagged Serde declaration order exactly.
    value.set_item(pyo3::intern!(py, "status"), status)?;
    value.set_item(pyo3::intern!(py, "now_ms"), now_ms)?;
    Ok(value.into_any().unbind())
}

fn replay_worker_target_to_python<'py>(
    py: Python<'py>,
    target: RsWorkerTarget,
) -> PyResult<Bound<'py, PyDict>> {
    let RsWorkerTarget {
        pool_id,
        worker_id,
        dp_rank,
    } = target;
    let value = PyDict::new(py);
    value.set_item(pyo3::intern!(py, "pool_id"), pool_id)?;
    value.set_item(pyo3::intern!(py, "worker_id"), worker_id)?;
    value.set_item(pyo3::intern!(py, "dp_rank"), dp_rank)?;
    Ok(value)
}

fn replay_routing_constraints_to_python<'py>(
    py: Python<'py>,
    constraints: RsReplayRoutingConstraints,
) -> PyResult<Bound<'py, PyDict>> {
    let RsReplayRoutingConstraints {
        required_taints,
        preferred_taints,
    } = constraints;
    let preferred_taints_value = PyDict::new(py);
    for (taint, weight) in preferred_taints {
        preferred_taints_value.set_item(taint, weight)?;
    }

    let value = PyDict::new(py);
    value.set_item(pyo3::intern!(py, "required_taints"), required_taints)?;
    value.set_item(
        pyo3::intern!(py, "preferred_taints"),
        preferred_taints_value,
    )?;
    Ok(value)
}

fn replay_placement_candidates_to_python<'py>(
    py: Python<'py>,
    candidates: Vec<RsReplayPlacementCandidate>,
) -> PyResult<Bound<'py, PyList>> {
    let values = PyList::empty(py);
    for candidate in candidates {
        values.append(replay_placement_candidate_to_python(py, candidate)?)?;
    }
    Ok(values)
}

fn replay_placement_candidate_to_python<'py>(
    py: Python<'py>,
    candidate: RsReplayPlacementCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let RsReplayPlacementCandidate {
        target,
        active,
        draining,
        eligible,
        constraint_reason,
        in_flight_requests,
        queued_requests,
        running_requests,
        queued_tokens,
        running_tokens,
        max_num_seqs,
        preemption_count,
        kv_prefix_overlap_tokens,
        kv_capacity_blocks,
        kv_occupied_blocks,
        kv_free_blocks,
        tags,
        taints,
        capabilities,
    } = candidate;
    let value = PyDict::new(py);
    value.set_item(
        pyo3::intern!(py, "target"),
        replay_worker_target_to_python(py, target)?,
    )?;
    value.set_item(pyo3::intern!(py, "active"), active)?;
    value.set_item(pyo3::intern!(py, "draining"), draining)?;
    value.set_item(pyo3::intern!(py, "eligible"), eligible)?;
    value.set_item(pyo3::intern!(py, "constraint_reason"), constraint_reason)?;
    value.set_item(pyo3::intern!(py, "in_flight_requests"), in_flight_requests)?;
    value.set_item(pyo3::intern!(py, "queued_requests"), queued_requests)?;
    value.set_item(pyo3::intern!(py, "running_requests"), running_requests)?;
    value.set_item(pyo3::intern!(py, "queued_tokens"), queued_tokens)?;
    value.set_item(pyo3::intern!(py, "running_tokens"), running_tokens)?;
    value.set_item(pyo3::intern!(py, "max_num_seqs"), max_num_seqs)?;
    value.set_item(pyo3::intern!(py, "preemption_count"), preemption_count)?;
    value.set_item(
        pyo3::intern!(py, "kv_prefix_overlap_tokens"),
        kv_prefix_overlap_tokens,
    )?;
    value.set_item(pyo3::intern!(py, "kv_capacity_blocks"), kv_capacity_blocks)?;
    value.set_item(pyo3::intern!(py, "kv_occupied_blocks"), kv_occupied_blocks)?;
    value.set_item(pyo3::intern!(py, "kv_free_blocks"), kv_free_blocks)?;
    value.set_item(pyo3::intern!(py, "tags"), tags)?;
    value.set_item(pyo3::intern!(py, "taints"), taints)?;
    value.set_item(pyo3::intern!(py, "capabilities"), capabilities)?;
    Ok(value)
}

fn replay_event_data_to_python<'py>(
    py: Python<'py>,
    data: RsReplayEventData,
) -> PyResult<Bound<'py, PyDict>> {
    let RsReplayEventData {
        logical_request_id,
        attempt_id,
        group_id,
        internal_uuid,
        session_id,
        authored_turn_index,
        timestamp_ms,
        pool_id,
        worker_id,
        dp_rank,
        terminal_status,
        input_length,
        requested_output_length,
        emitted_output_count,
        reused_input_tokens,
        ttft_ms,
        e2e_latency_ms,
        priority,
        strict_priority,
        policy_class,
        routing_constraints,
        eligible_pool_ids,
        candidates,
    } = data;
    let value = PyDict::new(py);
    value.set_item(pyo3::intern!(py, "logical_request_id"), logical_request_id)?;
    value.set_item(pyo3::intern!(py, "attempt_id"), attempt_id)?;
    value.set_item(pyo3::intern!(py, "group_id"), group_id)?;
    value.set_item(
        pyo3::intern!(py, "internal_uuid"),
        internal_uuid.to_string(),
    )?;
    value.set_item(pyo3::intern!(py, "session_id"), session_id)?;
    value.set_item(
        pyo3::intern!(py, "authored_turn_index"),
        authored_turn_index,
    )?;
    value.set_item(pyo3::intern!(py, "timestamp_ms"), timestamp_ms)?;
    value.set_item(pyo3::intern!(py, "pool_id"), pool_id)?;
    value.set_item(pyo3::intern!(py, "worker_id"), worker_id)?;
    value.set_item(pyo3::intern!(py, "dp_rank"), dp_rank)?;
    value.set_item(
        pyo3::intern!(py, "terminal_status"),
        terminal_status.map(replay_terminal_status_name),
    )?;
    value.set_item(pyo3::intern!(py, "input_length"), input_length)?;
    value.set_item(
        pyo3::intern!(py, "requested_output_length"),
        requested_output_length,
    )?;
    value.set_item(
        pyo3::intern!(py, "emitted_output_count"),
        emitted_output_count,
    )?;
    value.set_item(
        pyo3::intern!(py, "reused_input_tokens"),
        reused_input_tokens,
    )?;
    value.set_item(pyo3::intern!(py, "ttft_ms"), ttft_ms)?;
    value.set_item(pyo3::intern!(py, "e2e_latency_ms"), e2e_latency_ms)?;
    value.set_item(pyo3::intern!(py, "priority"), priority)?;
    value.set_item(pyo3::intern!(py, "strict_priority"), strict_priority)?;
    value.set_item(pyo3::intern!(py, "policy_class"), policy_class)?;
    value.set_item(
        pyo3::intern!(py, "routing_constraints"),
        replay_routing_constraints_to_python(py, routing_constraints)?,
    )?;
    value.set_item(pyo3::intern!(py, "eligible_pool_ids"), eligible_pool_ids)?;
    value.set_item(
        pyo3::intern!(py, "candidates"),
        replay_placement_candidates_to_python(py, candidates)?,
    )?;
    Ok(value)
}

fn replay_event_parts(event: RsReplayEvent) -> (&'static str, RsReplayEventData) {
    match event {
        RsReplayEvent::PlacementNeeded(data) => ("placement_needed", data),
        RsReplayEvent::Routed(data) => ("routed", data),
        RsReplayEvent::Queued(data) => ("queued", data),
        RsReplayEvent::Admitted(data) => ("admitted", data),
        RsReplayEvent::FirstToken(data) => ("first_token", data),
        RsReplayEvent::Terminal(data) => ("terminal", data),
    }
}

fn replay_events_to_python(py: Python<'_>, events: Vec<RsReplayEvent>) -> PyResult<PyObject> {
    let values = PyList::empty(py);
    for event in events {
        let (event_type, data) = replay_event_parts(event);
        let value = PyDict::new(py);
        value.set_item(pyo3::intern!(py, "event_type"), event_type)?;
        value.set_item(
            pyo3::intern!(py, "event"),
            replay_event_data_to_python(py, data)?,
        )?;
        values.append(value)?;
    }
    Ok(values.into_any().unbind())
}

fn replay_strings_to_python<'py>(
    py: Python<'py>,
    values: &[String],
) -> PyResult<Bound<'py, PyList>> {
    let result = PyList::empty(py);
    for value in values {
        result.append(value.as_str())?;
    }
    Ok(result)
}

fn replay_worker_target_ref_to_python<'py>(
    py: Python<'py>,
    target: &RsWorkerTarget,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item(pyo3::intern!(py, "pool_id"), target.pool_id.as_str())?;
    value.set_item(pyo3::intern!(py, "worker_id"), target.worker_id)?;
    value.set_item(pyo3::intern!(py, "dp_rank"), target.dp_rank)?;
    Ok(value)
}

fn replay_routing_constraints_ref_to_python<'py>(
    py: Python<'py>,
    constraints: &RsReplayRoutingConstraints,
) -> PyResult<Bound<'py, PyDict>> {
    let preferred_taints = PyDict::new(py);
    for (taint, weight) in &constraints.preferred_taints {
        preferred_taints.set_item(taint.as_str(), *weight)?;
    }
    let value = PyDict::new(py);
    value.set_item(
        pyo3::intern!(py, "required_taints"),
        replay_strings_to_python(py, &constraints.required_taints)?,
    )?;
    value.set_item(pyo3::intern!(py, "preferred_taints"), preferred_taints)?;
    Ok(value)
}

fn replay_placement_candidate_ref_to_python<'py>(
    py: Python<'py>,
    candidate: &RsReplayPlacementCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item(
        pyo3::intern!(py, "target"),
        replay_worker_target_ref_to_python(py, &candidate.target)?,
    )?;
    value.set_item(pyo3::intern!(py, "active"), candidate.active)?;
    value.set_item(pyo3::intern!(py, "draining"), candidate.draining)?;
    value.set_item(pyo3::intern!(py, "eligible"), candidate.eligible)?;
    value.set_item(
        pyo3::intern!(py, "constraint_reason"),
        candidate.constraint_reason.as_deref(),
    )?;
    value.set_item(
        pyo3::intern!(py, "in_flight_requests"),
        candidate.in_flight_requests,
    )?;
    value.set_item(
        pyo3::intern!(py, "queued_requests"),
        candidate.queued_requests,
    )?;
    value.set_item(
        pyo3::intern!(py, "running_requests"),
        candidate.running_requests,
    )?;
    value.set_item(pyo3::intern!(py, "queued_tokens"), candidate.queued_tokens)?;
    value.set_item(
        pyo3::intern!(py, "running_tokens"),
        candidate.running_tokens,
    )?;
    value.set_item(pyo3::intern!(py, "max_num_seqs"), candidate.max_num_seqs)?;
    value.set_item(
        pyo3::intern!(py, "preemption_count"),
        candidate.preemption_count,
    )?;
    value.set_item(
        pyo3::intern!(py, "kv_prefix_overlap_tokens"),
        candidate.kv_prefix_overlap_tokens,
    )?;
    value.set_item(
        pyo3::intern!(py, "kv_capacity_blocks"),
        candidate.kv_capacity_blocks,
    )?;
    value.set_item(
        pyo3::intern!(py, "kv_occupied_blocks"),
        candidate.kv_occupied_blocks,
    )?;
    value.set_item(
        pyo3::intern!(py, "kv_free_blocks"),
        candidate.kv_free_blocks,
    )?;
    value.set_item(
        pyo3::intern!(py, "tags"),
        replay_strings_to_python(py, &candidate.tags)?,
    )?;
    value.set_item(
        pyo3::intern!(py, "taints"),
        replay_strings_to_python(py, &candidate.taints)?,
    )?;
    value.set_item(
        pyo3::intern!(py, "capabilities"),
        replay_strings_to_python(py, &candidate.capabilities)?,
    )?;
    Ok(value)
}

fn replay_placement_candidates_ref_to_python<'py>(
    py: Python<'py>,
    candidates: &[RsReplayPlacementCandidate],
) -> PyResult<Bound<'py, PyList>> {
    let values = PyList::empty(py);
    for candidate in candidates {
        values.append(replay_placement_candidate_ref_to_python(py, candidate)?)?;
    }
    Ok(values)
}

fn replay_captured_event_data_to_python<'py>(
    py: Python<'py>,
    data: &RsCapturedReplayEventData,
) -> PyResult<Bound<'py, PyDict>> {
    let RsCapturedReplayEventDataView {
        logical_request_id,
        attempt_id,
        group_id,
        internal_uuid,
        session_id,
        authored_turn_index,
        timestamp_ms,
        pool_id,
        worker_id,
        dp_rank,
        terminal_status,
        input_length,
        requested_output_length,
        emitted_output_count,
        reused_input_tokens,
        ttft_ms,
        e2e_latency_ms,
        priority,
        strict_priority,
        policy_class,
        routing_constraints,
        eligible_pool_ids,
        candidates,
    } = data.view();
    let value = PyDict::new(py);
    value.set_item(pyo3::intern!(py, "logical_request_id"), logical_request_id)?;
    value.set_item(pyo3::intern!(py, "attempt_id"), attempt_id)?;
    value.set_item(pyo3::intern!(py, "group_id"), group_id)?;
    value.set_item(
        pyo3::intern!(py, "internal_uuid"),
        internal_uuid.to_string(),
    )?;
    value.set_item(pyo3::intern!(py, "session_id"), session_id)?;
    value.set_item(
        pyo3::intern!(py, "authored_turn_index"),
        authored_turn_index,
    )?;
    value.set_item(pyo3::intern!(py, "timestamp_ms"), timestamp_ms)?;
    value.set_item(pyo3::intern!(py, "pool_id"), pool_id)?;
    value.set_item(pyo3::intern!(py, "worker_id"), worker_id)?;
    value.set_item(pyo3::intern!(py, "dp_rank"), dp_rank)?;
    value.set_item(
        pyo3::intern!(py, "terminal_status"),
        terminal_status.map(replay_terminal_status_name),
    )?;
    value.set_item(pyo3::intern!(py, "input_length"), input_length)?;
    value.set_item(
        pyo3::intern!(py, "requested_output_length"),
        requested_output_length,
    )?;
    value.set_item(
        pyo3::intern!(py, "emitted_output_count"),
        emitted_output_count,
    )?;
    value.set_item(
        pyo3::intern!(py, "reused_input_tokens"),
        reused_input_tokens,
    )?;
    value.set_item(pyo3::intern!(py, "ttft_ms"), ttft_ms)?;
    value.set_item(pyo3::intern!(py, "e2e_latency_ms"), e2e_latency_ms)?;
    value.set_item(pyo3::intern!(py, "priority"), priority)?;
    value.set_item(pyo3::intern!(py, "strict_priority"), strict_priority)?;
    value.set_item(pyo3::intern!(py, "policy_class"), policy_class)?;
    value.set_item(
        pyo3::intern!(py, "routing_constraints"),
        replay_routing_constraints_ref_to_python(py, routing_constraints)?,
    )?;
    value.set_item(
        pyo3::intern!(py, "eligible_pool_ids"),
        replay_strings_to_python(py, eligible_pool_ids)?,
    )?;
    value.set_item(
        pyo3::intern!(py, "candidates"),
        replay_placement_candidates_ref_to_python(py, candidates)?,
    )?;
    Ok(value)
}

fn replay_captured_events_to_python(
    py: Python<'_>,
    events: Vec<RsCapturedReplayEvent>,
) -> PyResult<PyObject> {
    let values = PyList::empty(py);
    for event in events {
        let value = PyDict::new(py);
        value.set_item(pyo3::intern!(py, "event_type"), event.event_type())?;
        value.set_item(
            pyo3::intern!(py, "event"),
            replay_captured_event_data_to_python(py, event.data())?,
        )?;
        values.append(value)?;
    }
    Ok(values.into_any().unbind())
}

fn replay_pending_placements_to_python(
    py: Python<'_>,
    pending: Vec<RsReplayPendingPlacement>,
) -> PyResult<PyObject> {
    let values = PyList::empty(py);
    for placement in pending {
        let RsReplayPendingPlacement {
            logical_request_id,
            attempt_id,
            group_id,
            internal_uuid,
            session_id,
            authored_turn_index,
            ready_at_ms,
            input_length,
            priority,
            strict_priority,
            policy_class,
            routing_constraints,
            eligible_pool_ids,
            candidates,
        } = placement;
        let value = PyDict::new(py);
        value.set_item(pyo3::intern!(py, "logical_request_id"), logical_request_id)?;
        value.set_item(pyo3::intern!(py, "attempt_id"), attempt_id)?;
        value.set_item(pyo3::intern!(py, "group_id"), group_id)?;
        value.set_item(
            pyo3::intern!(py, "internal_uuid"),
            internal_uuid.to_string(),
        )?;
        value.set_item(pyo3::intern!(py, "session_id"), session_id)?;
        value.set_item(
            pyo3::intern!(py, "authored_turn_index"),
            authored_turn_index,
        )?;
        value.set_item(pyo3::intern!(py, "ready_at_ms"), ready_at_ms)?;
        value.set_item(pyo3::intern!(py, "input_length"), input_length)?;
        value.set_item(pyo3::intern!(py, "priority"), priority)?;
        value.set_item(pyo3::intern!(py, "strict_priority"), strict_priority)?;
        value.set_item(pyo3::intern!(py, "policy_class"), policy_class)?;
        value.set_item(
            pyo3::intern!(py, "routing_constraints"),
            replay_routing_constraints_to_python(py, routing_constraints)?,
        )?;
        value.set_item(pyo3::intern!(py, "eligible_pool_ids"), eligible_pool_ids)?;
        value.set_item(
            pyo3::intern!(py, "candidates"),
            replay_placement_candidates_to_python(py, candidates)?,
        )?;
        values.append(value)?;
    }
    Ok(values.into_any().unbind())
}

fn parse_interactive_router(router: &str) -> PyResult<RsReplaySessionRouter> {
    if router == "external" {
        return Ok(RsReplaySessionRouter::External);
    }
    match parse_replay_router_mode(router)? {
        dynamo_mocker::replay::ReplayRouterMode::RoundRobin => {
            Ok(RsReplaySessionRouter::RoundRobin)
        }
        dynamo_mocker::replay::ReplayRouterMode::KvRouter => Ok(RsReplaySessionRouter::KvRouter),
    }
}

/// Synchronous, polling-only Python owner of one causal offline replay.
///
/// The class is deliberately unsendable: the controller calls it on one Python
/// thread and exchanges plain request/event values. Replay never invokes a
/// Python placement callback.
#[pyclass(name = "_OfflineReplaySession", unsendable)]
pub struct PyOfflineReplaySession {
    inner: RsOfflineReplaySession,
}

#[pymethods]
impl PyOfflineReplaySession {
    #[new]
    #[pyo3(signature = (engine_args, trace_block_size, num_workers=1, router="external", session_affinity=false))]
    fn new(
        py: Python<'_>,
        engine_args: MockEngineArgs,
        trace_block_size: usize,
        num_workers: usize,
        router: &str,
        session_affinity: bool,
    ) -> PyResult<Self> {
        if engine_args.inner.aic_backend.is_some() {
            return Err(PyValueError::new_err(
                "interactive replay does not support Python-backed AIC performance callbacks",
            ));
        }
        let args = materialize_replay_mocker_args(py, engine_args)?;
        let router = parse_interactive_router(router)?;
        let inner = RsOfflineReplaySession::new_with_options(
            &args,
            num_workers,
            trace_block_size,
            router,
            RsReplaySessionOptions { session_affinity },
        )
        .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (pools, trace_block_size, router="external", session_affinity=false))]
    fn from_pools(
        py: Python<'_>,
        pools: Vec<Py<PyReplayPoolSpec>>,
        trace_block_size: usize,
        router: &str,
        session_affinity: bool,
    ) -> PyResult<Self> {
        if router != "external" {
            return Err(PyValueError::new_err(format!(
                "pooled interactive replay requires router='external', got {router:?}"
            )));
        }
        let pools = pools
            .into_iter()
            .map(|pool| pool.borrow(py).inner.clone())
            .collect();
        let inner = RsOfflineReplaySession::new_pooled_with_options(
            pools,
            trace_block_size,
            RsReplaySessionOptions { session_affinity },
        )
        .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn submit(&mut self, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request: InteractiveRequestSpec = depythonize_interactive(request, "request")?;
        self.inner.submit_request(request.into()).map_err(to_pyerr)
    }

    fn append_agentic_workflow(
        &mut self,
        workflow: &Bound<'_, PyAny>,
        release_at_ms: f64,
    ) -> PyResult<()> {
        let workflow: InteractiveAgenticWorkflow =
            depythonize_interactive(workflow, "agentic workflow")?;
        self.inner
            .append_agentic_workflow(workflow.into(), release_at_ms)
            .map_err(to_pyerr)
    }

    fn now_ms(&self) -> PyResult<f64> {
        self.inner.now_ms().map_err(to_pyerr)
    }

    fn next_event_time_ms(&mut self) -> PyResult<Option<f64>> {
        self.inner.next_event_time_ms().map_err(to_pyerr)
    }

    fn advance_next(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let status = self.inner.advance_next().map_err(to_pyerr)?;
        replay_step_status_to_python(py, status)
    }

    fn advance_to(&mut self, py: Python<'_>, target_ms: f64) -> PyResult<PyObject> {
        let status = self.inner.advance_to(target_ms).map_err(to_pyerr)?;
        replay_step_status_to_python(py, status)
    }

    fn settle_current_time(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let status = self.inner.settle_current_time().map_err(to_pyerr)?;
        replay_step_status_to_python(py, status)
    }

    fn drain_events(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let events = self.inner.drain_captured_events().map_err(to_pyerr)?;
        replay_captured_events_to_python(py, events)
    }

    fn pending_placements(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let pending = self.inner.pending_placements().map_err(to_pyerr)?;
        replay_pending_placements_to_python(py, pending)
    }

    fn assign(&mut self, logical_request_id: &str, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = parse_interactive_worker_target(target)?;
        self.inner
            .assign(logical_request_id, target.into())
            .map_err(to_pyerr)
    }

    fn assign_pool(&mut self, logical_request_id: &str, pool_id: &str) -> PyResult<()> {
        self.inner
            .assign_pool(logical_request_id, pool_id)
            .map_err(to_pyerr)
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snapshot = self.inner.snapshot().map_err(to_pyerr)?;
        pythonize_interactive(py, &snapshot)
    }

    fn close_admission(&mut self) -> PyResult<()> {
        self.inner.close_admission().map_err(to_pyerr)
    }

    fn is_quiescent(&mut self) -> PyResult<bool> {
        self.inner.is_quiescent().map_err(to_pyerr)
    }

    fn is_drained(&self) -> PyResult<bool> {
        self.inner.is_drained().map_err(to_pyerr)
    }

    fn finalize(&mut self) -> PyResult<OfflineReplayResult> {
        self.inner
            .finalize()
            .map(OfflineReplayResult::from_interactive)
            .map_err(to_pyerr)
    }
}

#[pyfunction]
#[pyo3(signature = (trace_files, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, trace_block_size=None, trace_format="mooncake", trace_shared_prefix_ratio=0.0, trace_num_prefix_groups=0, report_jsonl_path=None, max_sim_time_ms=None, model_name=None, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None, capture_per_request=false, capture_planner_details=true, scaling_policy=None))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_trace_replay(
    py: Python<'_>,
    trace_files: Vec<PathBuf>,
    extra_engine_args: Option<MockEngineArgs>,
    prefill_engine_args: Option<MockEngineArgs>,
    decode_engine_args: Option<MockEngineArgs>,
    router_config: Option<KvRouterConfig>,
    aic_perf_config: Option<&AicPerfConfig>,
    num_workers: usize,
    num_prefill_workers: usize,
    num_decode_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
    trace_block_size: Option<usize>,
    trace_format: &str,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
    report_jsonl_path: Option<PathBuf>,
    max_sim_time_ms: Option<f64>,
    model_name: Option<String>,
    sla_ttft_ms: Option<f64>,
    sla_itl_ms: Option<f64>,
    sla_e2e_ms: Option<f64>,
    capture_per_request: bool,
    capture_planner_details: bool,
    scaling_policy: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    if capture_per_request && replay_mode != "offline" {
        return Err(PyValueError::new_err(
            "capture_per_request only supports replay_mode='offline'",
        ));
    }
    let args_selection = load_replay_args_selection(
        py,
        extra_engine_args,
        prefill_engine_args,
        decode_engine_args,
        num_workers,
        num_prefill_workers,
        num_decode_workers,
    )?;
    let router_mode = parse_replay_router_mode(router_mode)?;
    let trace_format = parse_trace_file_format(trace_format)?;
    dynamo_mocker::loadgen::validate_trace_files(trace_format, &trace_files).map_err(to_pyerr)?;
    let (prefill_load_estimator, _) = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config, model_name)?;
    let replay_mode = replay_mode.to_owned();
    let is_offline = replay_mode == "offline";
    if scaling_policy.is_some() && replay_mode != "offline" {
        return Err(PyValueError::new_err(
            "scaling_policy only supports replay_mode='offline'",
        ));
    }
    let jsonl_path_for_emit = report_jsonl_path.clone();
    let capture_planner_details = scaling_policy.is_some() && capture_planner_details;
    let capture_options = dynamo_mocker::replay::ReplayCaptureOptions {
        capture_per_request: capture_per_request || report_jsonl_path.is_some(),
        capture_planner_details,
        ..Default::default()
    };
    let record_per_request = capture_options.effective_per_request();
    if let Some(ms) = max_sim_time_ms {
        if !ms.is_finite() || ms < 0.0 {
            return Err(PyValueError::new_err(
                "max_sim_time_ms must be a finite, non-negative value",
            ));
        }
        if replay_mode != "offline" {
            return Err(PyValueError::new_err(
                "max_sim_time_ms only supports replay_mode='offline'",
            ));
        }
    }
    validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
    validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
    validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
    let sla = dynamo_mocker::replay::SlaThresholds {
        ttft_ms: sla_ttft_ms,
        itl_ms: sla_itl_ms,
        e2e_ms: sla_e2e_ms,
    };
    let run = move |mut scaling_policy: Option<Box<dyn ReplayScalingPolicy>>| {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;
        if trace_format == dynamo_mocker::loadgen::TraceFileFormat::Dynamo {
            let trace =
                DynamoRequestTrace::from_request_trace_files(&trace_files, trace_block_size)?;
            return run_loaded_dynamo_request_trace(
                args_selection,
                trace,
                router_config,
                prefill_load_estimator,
                num_workers,
                replay_concurrency,
                &replay_mode,
                arrival_speedup_ratio,
                router_mode,
                record_per_request,
                max_sim_time_ms,
                sla,
                scaling_policy,
            );
        }

        let trace_block_size = trace_block_size.unwrap_or(512);
        let trace_file = &trace_files[0];
        if trace_format == dynamo_mocker::loadgen::TraceFileFormat::AppliedComputeAgentic
            && replay_concurrency.is_none()
        {
            anyhow::bail!(
                "trace_format='applied_compute_agentic' requires replay_concurrency because source traces do not contain first-turn timestamps"
            );
        }

        match select_replay_dispatch(args_selection, &replay_mode, replay_concurrency)? {
            ReplayDispatch::AggregatedOfflineConcurrency(args, max_in_flight) => {
                dynamo_mocker::replay::simulate_concurrency_file_with_router_mode_and_format_and_scaling_policy(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    max_in_flight,
                    num_workers,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    max_sim_time_ms,
                    sla,
                    scaling_policy.take(),
                )
            }
            ReplayDispatch::AggregatedOffline(args) => {
                dynamo_mocker::replay::simulate_trace_file_with_router_mode_and_format_and_scaling_policy(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    num_workers,
                    arrival_speedup_ratio,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    max_sim_time_ms,
                    sla,
                    scaling_policy.take(),
                )
            }
            ReplayDispatch::AggregatedOnlineConcurrency(args, max_in_flight) => {
                dynamo_mocker::replay::simulate_concurrency_live_file_with_router_mode_and_format_and_options(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    max_in_flight,
                    num_workers,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    sla,
                )
            }
            ReplayDispatch::AggregatedOnline(args) => {
                dynamo_mocker::replay::simulate_trace_live_file_with_router_mode_and_format_and_options(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    num_workers,
                    arrival_speedup_ratio,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    sla,
                )
            }
            ReplayDispatch::DisaggOfflineConcurrency(config, max_in_flight) => {
                dynamo_mocker::replay::simulate_concurrency_file_disagg_with_router_mode_and_format_and_scaling_policy(
                    *config,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    max_in_flight,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    max_sim_time_ms,
                    sla,
                    scaling_policy.take(),
                )
            }
            ReplayDispatch::DisaggOffline(config) => {
                dynamo_mocker::replay::simulate_trace_file_disagg_with_router_mode_and_format_and_scaling_policy(
                    *config,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    trace_file,
                    trace_block_size,
                    arrival_speedup_ratio,
                    router_mode,
                    trace_format,
                    trace_shared_prefix_ratio,
                    trace_num_prefix_groups,
                    record_per_request,
                    max_sim_time_ms,
                    sla,
                    scaling_policy.take(),
                )
            }
        }
    };
    let (report, runtime_evidence) = if let Some(callback) = scaling_policy {
        let (report, evidence) =
            dynamo_mocker::replay::with_runtime_evidence(capture_options, || {
                dynamo_mocker::replay::with_replay_determinism(capture_options.determinism, || {
                    run(Some(Box::new(PyReplayScalingPolicy { callback })))
                })
            });
        (report.map_err(scaling_run_err_to_pyerr)?, evidence)
    } else {
        let (report, evidence) = py.allow_threads(move || {
            dynamo_mocker::replay::with_runtime_evidence(capture_options, || {
                dynamo_mocker::replay::with_replay_determinism(capture_options.determinism, || {
                    run(None)
                })
            })
        });
        (report.map_err(to_pyerr)?, evidence)
    };
    // Write per-request JSONL from Rust directly if requested, avoiding a
    // potentially-large round trip through pyo3 / pythonize. Each line is one
    // JSON object (matching AIPerf's profile_export.jsonl convention).
    if let Some(path) = jsonl_path_for_emit.as_ref() {
        py.allow_threads(|| write_per_request_jsonl(path, &report.per_request))
            .map_err(to_pyerr)?;
    }
    if is_offline {
        return Py::new(
            py,
            OfflineReplayResult::new(
                report,
                record_per_request,
                capture_planner_details,
                runtime_evidence,
            ),
        )
        .map(Py::into_any);
    }
    pythonize(py, &report).map(Bound::unbind).map_err(to_pyerr)
}

#[allow(clippy::too_many_arguments)]
fn run_loaded_dynamo_request_trace(
    args_selection: ReplayArgsSelection,
    trace: DynamoRequestTrace,
    router_config: Option<dynamo_kv_router::config::KvRouterConfig>,
    prefill_load_estimator: Option<dynamo_mocker::replay::ReplayPrefillLoadEstimator>,
    num_workers: usize,
    replay_concurrency: Option<usize>,
    replay_mode: &str,
    arrival_speedup_ratio: f64,
    router_mode: dynamo_mocker::replay::ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: dynamo_mocker::replay::SlaThresholds,
    mut scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> anyhow::Result<dynamo_mocker::replay::TraceSimulationReport> {
    match trace {
        DynamoRequestTrace::Standard(trace) => {
            match select_replay_dispatch(args_selection, replay_mode, replay_concurrency)? {
                ReplayDispatch::AggregatedOfflineConcurrency(args, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_workload_with_router_mode_and_options_and_scaling_policy(
                            *args,
                            router_config,
                            prefill_load_estimator,
                            trace,
                            max_in_flight,
                            num_workers,
                            router_mode,
                            record_per_request,
                            max_sim_time_ms,
                            sla,
                            scaling_policy.take(),
                        )
                }
                ReplayDispatch::AggregatedOffline(args) => {
                    dynamo_mocker::replay::simulate_loaded_trace_with_router_mode_and_options_and_scaling_policy(
                        *args,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        num_workers,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        max_sim_time_ms,
                        sla,
                        scaling_policy.take(),
                    )
                }
                ReplayDispatch::AggregatedOnlineConcurrency(args, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_live_workload_with_router_mode_and_options(
                        *args,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        max_in_flight,
                        num_workers,
                        router_mode,
                        record_per_request,
                        sla,
                    )
                }
                ReplayDispatch::AggregatedOnline(args) => {
                    dynamo_mocker::replay::simulate_loaded_trace_live_with_router_mode_and_options(
                        *args,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        num_workers,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        sla,
                    )
                }
                ReplayDispatch::DisaggOfflineConcurrency(config, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_workload_disagg_with_router_mode_and_options_and_scaling_policy(
                            *config,
                            router_config,
                            prefill_load_estimator,
                            trace,
                            max_in_flight,
                            router_mode,
                            record_per_request,
                            max_sim_time_ms,
                            sla,
                            scaling_policy.take(),
                        )
                }
                ReplayDispatch::DisaggOffline(config) => {
                    dynamo_mocker::replay::simulate_loaded_trace_disagg_with_router_mode_and_options_and_scaling_policy(
                        *config,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        max_sim_time_ms,
                        sla,
                        scaling_policy.take(),
                    )
                }
            }
        }
        DynamoRequestTrace::Agentic(trace) => {
            anyhow::ensure!(
                scaling_policy.is_none(),
                "scaling_policy replay does not support agentic Dynamo request traces"
            );
            if replay_concurrency.is_some() {
                anyhow::bail!(
                    "agentic Dynamo request traces are not supported with replay_concurrency"
                );
            }
            let ReplayArgsSelection::Aggregated(args) = args_selection else {
                anyhow::bail!(
                    "agentic Dynamo request traces are not supported for disaggregated replay"
                );
            };
            let trace = trace
                .normalize_starts()
                .speed_up_timing(arrival_speedup_ratio)?;
            match replay_mode {
                "offline" => dynamo_mocker::replay::simulate_agentic_trace_workload_with_router_mode(
                    *args,
                    router_config,
                    prefill_load_estimator,
                    trace,
                    num_workers,
                    router_mode,
                    record_per_request,
                    sla,
                ),
                "online" => dynamo_mocker::replay::simulate_agentic_trace_live_workload_with_router_mode_and_options(
                    *args,
                    router_config,
                    prefill_load_estimator,
                    trace,
                    num_workers,
                    router_mode,
                    record_per_request,
                    sla,
                ),
                other => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
            }
        }
    }
}

/// Write per-request records to a JSONL file. One JSON object per line, no
/// outer array wrapper — matches AIPerf's `profile_export.jsonl` convention
/// and is friendlier to streaming consumers (pandas read_json with lines=True,
/// jq -c, etc.).
fn write_per_request_jsonl(
    path: &std::path::Path,
    records: &[dynamo_mocker::replay::PerRequestRecord],
) -> anyhow::Result<()> {
    use std::io::{BufWriter, Write};
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        let line = serde_json::to_string(record)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, request_rate=None, arrival_interval_ms=None, arrival_seed=42, turns_per_session=1, shared_prefix_ratio=0.0, num_prefix_groups=0, inter_turn_delay_ms=0.0, model_name=None, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None, capture_per_request=false, capture_planner_details=true, scaling_policy=None))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_synthetic_trace_replay(
    py: Python<'_>,
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    extra_engine_args: Option<MockEngineArgs>,
    prefill_engine_args: Option<MockEngineArgs>,
    decode_engine_args: Option<MockEngineArgs>,
    router_config: Option<KvRouterConfig>,
    aic_perf_config: Option<&AicPerfConfig>,
    num_workers: usize,
    num_prefill_workers: usize,
    num_decode_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
    request_rate: Option<f64>,
    arrival_interval_ms: Option<f64>,
    arrival_seed: u64,
    turns_per_session: usize,
    shared_prefix_ratio: f64,
    num_prefix_groups: usize,
    inter_turn_delay_ms: f64,
    model_name: Option<String>,
    sla_ttft_ms: Option<f64>,
    sla_itl_ms: Option<f64>,
    sla_e2e_ms: Option<f64>,
    capture_per_request: bool,
    capture_planner_details: bool,
    scaling_policy: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    if capture_per_request && replay_mode != "offline" {
        return Err(PyValueError::new_err(
            "capture_per_request only supports replay_mode='offline'",
        ));
    }
    if scaling_policy.is_some() && replay_mode != "offline" {
        return Err(PyValueError::new_err(
            "scaling_policy only supports replay_mode='offline'",
        ));
    }
    validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
    validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
    validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
    let sla = dynamo_mocker::replay::SlaThresholds {
        ttft_ms: sla_ttft_ms,
        itl_ms: sla_itl_ms,
        e2e_ms: sla_e2e_ms,
    };
    let args_selection = load_replay_args_selection(
        py,
        extra_engine_args,
        prefill_engine_args,
        decode_engine_args,
        num_workers,
        num_prefill_workers,
        num_decode_workers,
    )?;
    let router_mode = parse_replay_router_mode(router_mode)?;
    let (prefill_load_estimator, _) = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config, model_name)?;
    let replay_mode = replay_mode.to_owned();
    let is_offline = replay_mode == "offline";
    let capture_planner_details = scaling_policy.is_some() && capture_planner_details;
    let capture_options = dynamo_mocker::replay::ReplayCaptureOptions {
        capture_per_request,
        capture_planner_details,
        ..Default::default()
    };
    let record_per_request = capture_options.effective_per_request();
    let block_size = match &args_selection {
        ReplayArgsSelection::Aggregated(args) => args.block_size.max(1),
        ReplayArgsSelection::Disagg(config) => config.prefill_args.block_size.max(1),
    };
    let run = move |mut scaling_policy: Option<Box<dyn ReplayScalingPolicy>>| {
        let load_controller =
            parse_synthetic_load_controller(replay_concurrency, request_rate, arrival_interval_ms)?;
        let replay_concurrency = load_controller.replay_concurrency();
        let use_workload = turns_per_session > 1
            || shared_prefix_ratio > 0.0
            || num_prefix_groups > 0
            || inter_turn_delay_ms > 0.0;

        if use_workload {
            let mut trace = build_synthetic_workload(
                block_size,
                input_tokens,
                output_tokens,
                request_count,
                load_controller
                    .arrival_spec()
                    .cloned()
                    .unwrap_or(ArrivalSpec::Burst),
                arrival_seed,
                turns_per_session,
                shared_prefix_ratio,
                num_prefix_groups,
                inter_turn_delay_ms,
            )?;
            if replay_concurrency.is_none() {
                trace = trace.speed_up_timing(arrival_speedup_ratio)?;
            }

            return match args_selection {
                ReplayArgsSelection::Aggregated(args) => match (replay_mode.as_str(), replay_concurrency)
                {
                    ("offline", Some(max_in_flight)) => {
                        dynamo_mocker::replay::simulate_concurrency_workload_with_router_mode_and_options_and_scaling_policy(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            num_workers,
                            router_mode,
                            record_per_request,
                            None,
                            sla,
                            scaling_policy.take(),
                        )
                    }
                    ("offline", None) => {
                        dynamo_mocker::replay::simulate_trace_workload_with_router_mode_and_options_and_scaling_policy(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            num_workers,
                            router_mode,
                            record_per_request,
                            None,
                            sla,
                            scaling_policy.take(),
                        )
                    }
                    ("online", Some(max_in_flight)) => {
                        dynamo_mocker::replay::simulate_concurrency_live_workload_with_router_mode_and_options(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            num_workers,
                            router_mode,
                            record_per_request,
                            sla,
                        )
                    }
                    ("online", None) => {
                        dynamo_mocker::replay::simulate_trace_live_workload_with_router_mode_and_options(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            num_workers,
                            router_mode,
                            record_per_request,
                            sla,
                        )
                    }
                    (other, _) => anyhow::bail!(
                        "replay_mode must be either 'offline' or 'online', got '{}'",
                        other
                    ),
                },
                ReplayArgsSelection::Disagg(config) => {
                    validate_disagg_replay_mode(&replay_mode)?;
                    match (replay_mode.as_str(), replay_concurrency) {
                        ("offline", Some(max_in_flight)) => dynamo_mocker::replay::simulate_concurrency_workload_disagg_with_router_mode_and_options_and_scaling_policy(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            router_mode,
                            record_per_request,
                            None,
                            sla,
                            scaling_policy.take(),
                        ),
                        ("offline", None) => dynamo_mocker::replay::simulate_trace_workload_disagg_with_router_mode_and_options_and_scaling_policy(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            router_mode,
                            record_per_request,
                            None,
                            sla,
                            scaling_policy.take(),
                        ),
                        (other, _) => anyhow::bail!(
                            "replay_mode must be either 'offline' or 'online', got '{}'",
                            other
                        ),
                    }
                }
            };
        }

        let arrival_timestamps_ms = load_controller
            .arrival_spec()
            .map(|spec| spec.timestamps(request_count, arrival_seed))
            .transpose()?;
        let requests = build_synthetic_requests(
            input_tokens,
            output_tokens,
            request_count,
            arrival_timestamps_ms.as_deref(),
        )?;

        match args_selection {
            ReplayArgsSelection::Aggregated(args) => match (replay_mode.as_str(), replay_concurrency)
            {
                ("offline", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_requests_with_router_mode_and_scaling_policy(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        num_workers,
                        router_mode,
                        record_per_request,
                        sla,
                        scaling_policy.take(),
                    )
                }
                ("offline", None) => dynamo_mocker::replay::simulate_trace_requests_with_router_mode_and_scaling_policy(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    requests,
                    num_workers,
                    arrival_speedup_ratio,
                    router_mode,
                    record_per_request,
                    sla,
                    scaling_policy.take(),
                ),
                ("online", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_live_requests_with_router_mode_and_options(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        num_workers,
                        router_mode,
                        record_per_request,
                        sla,
                    )
                }
                ("online", None) => {
                    dynamo_mocker::replay::simulate_trace_live_requests_with_router_mode_and_options(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        num_workers,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        sla,
                    )
                }
                (other, _) => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
            },
            ReplayArgsSelection::Disagg(config) => {
                validate_disagg_replay_mode(&replay_mode)?;
                match (replay_mode.as_str(), replay_concurrency) {
                ("offline", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_requests_disagg_with_router_mode_and_scaling_policy(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        router_mode,
                        record_per_request,
                        sla,
                        scaling_policy.take(),
                    )
                }
                ("offline", None) => {
                    dynamo_mocker::replay::simulate_trace_requests_disagg_with_router_mode_and_scaling_policy(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        sla,
                        scaling_policy.take(),
                    )
                }
                (other, _) => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
                }
            }
        }
    };
    let (report, runtime_evidence) = if let Some(callback) = scaling_policy {
        let (report, evidence) =
            dynamo_mocker::replay::with_runtime_evidence(capture_options, || {
                dynamo_mocker::replay::with_replay_determinism(capture_options.determinism, || {
                    run(Some(Box::new(PyReplayScalingPolicy { callback })))
                })
            });
        (report.map_err(scaling_run_err_to_pyerr)?, evidence)
    } else {
        let (report, evidence) = py.allow_threads(move || {
            dynamo_mocker::replay::with_runtime_evidence(capture_options, || {
                dynamo_mocker::replay::with_replay_determinism(capture_options.determinism, || {
                    run(None)
                })
            })
        });
        (report.map_err(to_pyerr)?, evidence)
    };
    if is_offline {
        return Py::new(
            py,
            OfflineReplayResult::new(
                report,
                record_per_request,
                capture_planner_details,
                runtime_evidence,
            ),
        )
        .map(Py::into_any);
    }
    pythonize(py, &report).map(Bound::unbind).map_err(to_pyerr)
}

enum ReplayArgsSelection {
    Aggregated(Box<RsMockEngineArgs>),
    Disagg(Box<dynamo_mocker::replay::OfflineDisaggReplayConfig>),
}

enum ReplayDispatch {
    AggregatedOfflineConcurrency(Box<RsMockEngineArgs>, usize),
    AggregatedOffline(Box<RsMockEngineArgs>),
    AggregatedOnlineConcurrency(Box<RsMockEngineArgs>, usize),
    AggregatedOnline(Box<RsMockEngineArgs>),
    DisaggOfflineConcurrency(Box<dynamo_mocker::replay::OfflineDisaggReplayConfig>, usize),
    DisaggOffline(Box<dynamo_mocker::replay::OfflineDisaggReplayConfig>),
}

fn select_replay_dispatch(
    args_selection: ReplayArgsSelection,
    replay_mode: &str,
    replay_concurrency: Option<usize>,
) -> anyhow::Result<ReplayDispatch> {
    match (args_selection, replay_mode, replay_concurrency) {
        (ReplayArgsSelection::Aggregated(args), "offline", Some(max_in_flight)) => Ok(
            ReplayDispatch::AggregatedOfflineConcurrency(args, max_in_flight),
        ),
        (ReplayArgsSelection::Aggregated(args), "offline", None) => {
            Ok(ReplayDispatch::AggregatedOffline(args))
        }
        (ReplayArgsSelection::Aggregated(args), "online", Some(max_in_flight)) => Ok(
            ReplayDispatch::AggregatedOnlineConcurrency(args, max_in_flight),
        ),
        (ReplayArgsSelection::Aggregated(args), "online", None) => {
            Ok(ReplayDispatch::AggregatedOnline(args))
        }
        (ReplayArgsSelection::Disagg(config), "offline", Some(max_in_flight)) => Ok(
            ReplayDispatch::DisaggOfflineConcurrency(config, max_in_flight),
        ),
        (ReplayArgsSelection::Disagg(config), "offline", None) => {
            Ok(ReplayDispatch::DisaggOffline(config))
        }
        (ReplayArgsSelection::Disagg(_), other, _) => {
            validate_disagg_replay_mode(other)?;
            anyhow::bail!("replay_mode must be either 'offline' or 'online', got '{other}'")
        }
        (_, other, _) => {
            anyhow::bail!("replay_mode must be either 'offline' or 'online', got '{other}'")
        }
    }
}

fn validate_disagg_replay_mode(replay_mode: &str) -> anyhow::Result<()> {
    if replay_mode == "online" {
        anyhow::bail!("disagg replay only supports replay_mode='offline'");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        RsReplayEvent, RsReplayEventData, RsReplayPendingPlacement, RsReplayPlacementCandidate,
        RsReplayRoutingConstraints, RsReplayStepStatus, RsReplayTerminalStatus, RsWorkerTarget,
        build_synthetic_requests, fpm_snapshots_to_json, reconcile_replay_dp_topology,
        replay_event_parts, replay_step_status_parts, replay_terminal_status_name,
        validate_disagg_replay_mode,
    };
    use dynamo_mocker::common::protocols::{ForwardPassSnapshot, MockEngineArgs};
    use dynamo_mocker::loadgen::ArrivalSpec;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use uuid::Uuid;

    fn direct_worker_target_value(target: RsWorkerTarget) -> Value {
        let RsWorkerTarget {
            pool_id,
            worker_id,
            dp_rank,
        } = target;
        json!({
            "pool_id": pool_id,
            "worker_id": worker_id,
            "dp_rank": dp_rank,
        })
    }

    fn direct_routing_constraints_value(constraints: RsReplayRoutingConstraints) -> Value {
        let RsReplayRoutingConstraints {
            required_taints,
            preferred_taints,
        } = constraints;
        json!({
            "required_taints": required_taints,
            "preferred_taints": preferred_taints,
        })
    }

    fn direct_candidate_value(candidate: RsReplayPlacementCandidate) -> Value {
        let RsReplayPlacementCandidate {
            target,
            active,
            draining,
            eligible,
            constraint_reason,
            in_flight_requests,
            queued_requests,
            running_requests,
            queued_tokens,
            running_tokens,
            max_num_seqs,
            preemption_count,
            kv_prefix_overlap_tokens,
            kv_capacity_blocks,
            kv_occupied_blocks,
            kv_free_blocks,
            tags,
            taints,
            capabilities,
        } = candidate;
        json!({
            "target": direct_worker_target_value(target),
            "active": active,
            "draining": draining,
            "eligible": eligible,
            "constraint_reason": constraint_reason,
            "in_flight_requests": in_flight_requests,
            "queued_requests": queued_requests,
            "running_requests": running_requests,
            "queued_tokens": queued_tokens,
            "running_tokens": running_tokens,
            "max_num_seqs": max_num_seqs,
            "preemption_count": preemption_count,
            "kv_prefix_overlap_tokens": kv_prefix_overlap_tokens,
            "kv_capacity_blocks": kv_capacity_blocks,
            "kv_occupied_blocks": kv_occupied_blocks,
            "kv_free_blocks": kv_free_blocks,
            "tags": tags,
            "taints": taints,
            "capabilities": capabilities,
        })
    }

    fn direct_candidates_value(candidates: Vec<RsReplayPlacementCandidate>) -> Value {
        Value::Array(candidates.into_iter().map(direct_candidate_value).collect())
    }

    fn direct_event_data_value(data: RsReplayEventData) -> Value {
        let RsReplayEventData {
            logical_request_id,
            attempt_id,
            group_id,
            internal_uuid,
            session_id,
            authored_turn_index,
            timestamp_ms,
            pool_id,
            worker_id,
            dp_rank,
            terminal_status,
            input_length,
            requested_output_length,
            emitted_output_count,
            reused_input_tokens,
            ttft_ms,
            e2e_latency_ms,
            priority,
            strict_priority,
            policy_class,
            routing_constraints,
            eligible_pool_ids,
            candidates,
        } = data;
        json!({
            "logical_request_id": logical_request_id,
            "attempt_id": attempt_id,
            "group_id": group_id,
            "internal_uuid": internal_uuid.to_string(),
            "session_id": session_id,
            "authored_turn_index": authored_turn_index,
            "timestamp_ms": timestamp_ms,
            "pool_id": pool_id,
            "worker_id": worker_id,
            "dp_rank": dp_rank,
            "terminal_status": terminal_status.map(replay_terminal_status_name),
            "input_length": input_length,
            "requested_output_length": requested_output_length,
            "emitted_output_count": emitted_output_count,
            "reused_input_tokens": reused_input_tokens,
            "ttft_ms": ttft_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "priority": priority,
            "strict_priority": strict_priority,
            "policy_class": policy_class,
            "routing_constraints": direct_routing_constraints_value(routing_constraints),
            "eligible_pool_ids": eligible_pool_ids,
            "candidates": direct_candidates_value(candidates),
        })
    }

    fn direct_event_value(event: RsReplayEvent) -> Value {
        let (event_type, data) = replay_event_parts(event);
        json!({
            "event_type": event_type,
            "event": direct_event_data_value(data),
        })
    }

    fn direct_pending_placement_value(placement: RsReplayPendingPlacement) -> Value {
        let RsReplayPendingPlacement {
            logical_request_id,
            attempt_id,
            group_id,
            internal_uuid,
            session_id,
            authored_turn_index,
            ready_at_ms,
            input_length,
            priority,
            strict_priority,
            policy_class,
            routing_constraints,
            eligible_pool_ids,
            candidates,
        } = placement;
        json!({
            "logical_request_id": logical_request_id,
            "attempt_id": attempt_id,
            "group_id": group_id,
            "internal_uuid": internal_uuid.to_string(),
            "session_id": session_id,
            "authored_turn_index": authored_turn_index,
            "ready_at_ms": ready_at_ms,
            "input_length": input_length,
            "priority": priority,
            "strict_priority": strict_priority,
            "policy_class": policy_class,
            "routing_constraints": direct_routing_constraints_value(routing_constraints),
            "eligible_pool_ids": eligible_pool_ids,
            "candidates": direct_candidates_value(candidates),
        })
    }

    fn serialization_candidate(with_optional_values: bool) -> RsReplayPlacementCandidate {
        RsReplayPlacementCandidate {
            target: RsWorkerTarget::new("pool-a", 7, 2),
            active: true,
            draining: false,
            eligible: with_optional_values,
            constraint_reason: (!with_optional_values).then(|| "taint mismatch".to_string()),
            in_flight_requests: 3,
            queued_requests: with_optional_values.then_some(4),
            running_requests: with_optional_values.then_some(5),
            queued_tokens: with_optional_values.then_some(6),
            running_tokens: with_optional_values.then_some(7),
            max_num_seqs: with_optional_values.then_some(8),
            preemption_count: with_optional_values.then_some(9),
            kv_prefix_overlap_tokens: with_optional_values.then_some(10),
            kv_capacity_blocks: with_optional_values.then_some(11),
            kv_occupied_blocks: with_optional_values.then_some(12),
            kv_free_blocks: with_optional_values.then_some(13),
            tags: vec!["tag-a".to_string()],
            taints: vec!["taint-a".to_string()],
            capabilities: vec!["chat".to_string()],
        }
    }

    fn serialization_constraints() -> RsReplayRoutingConstraints {
        RsReplayRoutingConstraints {
            required_taints: vec!["taint-a".to_string()],
            preferred_taints: BTreeMap::from([("taint-b".to_string(), 1.25)]),
        }
    }

    fn serialization_event_data(
        ordinal: usize,
        terminal_status: Option<RsReplayTerminalStatus>,
    ) -> RsReplayEventData {
        let populated = ordinal % 2 == 1;
        RsReplayEventData {
            logical_request_id: format!("request-{ordinal}"),
            attempt_id: format!("attempt-{ordinal}"),
            group_id: "group-a".to_string(),
            internal_uuid: Uuid::from_u128(ordinal as u128 + 1),
            session_id: "session-a".to_string(),
            authored_turn_index: ordinal,
            timestamp_ms: ordinal as f64 + 0.5,
            pool_id: populated.then(|| "pool-a".to_string()),
            worker_id: populated.then_some(7),
            dp_rank: populated.then_some(2),
            terminal_status,
            input_length: 16,
            requested_output_length: populated.then_some(8),
            emitted_output_count: ordinal,
            reused_input_tokens: populated.then_some(4),
            ttft_ms: populated.then_some(1.5),
            e2e_latency_ms: populated.then_some(2.5),
            priority: -3,
            strict_priority: 4,
            policy_class: populated.then(|| "latency".to_string()),
            routing_constraints: serialization_constraints(),
            eligible_pool_ids: vec!["pool-a".to_string()],
            candidates: vec![
                serialization_candidate(true),
                serialization_candidate(false),
            ],
        }
    }

    #[test]
    fn online_disaggregation_is_rejected_with_stable_message() {
        assert_eq!(
            validate_disagg_replay_mode("online")
                .unwrap_err()
                .to_string(),
            "disagg replay only supports replay_mode='offline'"
        );
        assert!(validate_disagg_replay_mode("offline").is_ok());
    }

    #[test]
    fn direct_interactive_schema_matches_generic_serde_for_every_variant() {
        let events = vec![
            RsReplayEvent::PlacementNeeded(serialization_event_data(0, None)),
            RsReplayEvent::Routed(serialization_event_data(
                1,
                Some(RsReplayTerminalStatus::Completed),
            )),
            RsReplayEvent::Queued(serialization_event_data(
                2,
                Some(RsReplayTerminalStatus::Rejected),
            )),
            RsReplayEvent::Admitted(serialization_event_data(
                3,
                Some(RsReplayTerminalStatus::Canceled),
            )),
            RsReplayEvent::FirstToken(serialization_event_data(
                4,
                Some(RsReplayTerminalStatus::Failed),
            )),
            RsReplayEvent::Terminal(serialization_event_data(
                5,
                Some(RsReplayTerminalStatus::Completed),
            )),
        ];
        let direct_events =
            Value::Array(events.clone().into_iter().map(direct_event_value).collect());
        assert_eq!(
            direct_events,
            serde_json::to_value(&events).expect("generic event serialization")
        );

        let pending = vec![RsReplayPendingPlacement {
            logical_request_id: "request-pending".to_string(),
            attempt_id: "attempt-pending".to_string(),
            group_id: "group-a".to_string(),
            internal_uuid: Uuid::from_u128(99),
            session_id: "session-a".to_string(),
            authored_turn_index: 6,
            ready_at_ms: 7.5,
            input_length: 16,
            priority: -3,
            strict_priority: 4,
            policy_class: None,
            routing_constraints: serialization_constraints(),
            eligible_pool_ids: vec!["pool-a".to_string()],
            candidates: vec![
                serialization_candidate(true),
                serialization_candidate(false),
            ],
        }];
        let direct_pending = Value::Array(
            pending
                .clone()
                .into_iter()
                .map(direct_pending_placement_value)
                .collect(),
        );
        assert_eq!(
            direct_pending,
            serde_json::to_value(&pending).expect("generic pending-placement serialization")
        );
    }

    #[test]
    fn direct_step_status_schema_matches_generic_serde_for_every_variant() {
        for status in [
            RsReplayStepStatus::Advanced { now_ms: 1.25 },
            RsReplayStepStatus::Quiescent { now_ms: 2.5 },
            RsReplayStepStatus::Drained { now_ms: 3.75 },
        ] {
            let (status_name, now_ms) = replay_step_status_parts(status);
            let direct = json!({
                "status": status_name,
                "now_ms": now_ms,
            });
            assert_eq!(
                direct,
                serde_json::to_value(status).expect("generic step-status serialization")
            );
        }
    }

    #[test]
    fn programmatic_attention_dp_materializes_without_aic_backend() {
        let mut args = MockEngineArgs::builder()
            .dp_size(1)
            .aic_attention_dp_size(Some(4))
            .build()
            .unwrap();

        reconcile_replay_dp_topology(&mut args).unwrap();

        assert_eq!(args.dp_size, 4);
    }

    #[test]
    fn programmatic_attention_dp_rejects_mismatched_topology() {
        let mut args = MockEngineArgs::builder()
            .dp_size(2)
            .aic_attention_dp_size(Some(4))
            .build()
            .unwrap();

        let error = reconcile_replay_dp_topology(&mut args).unwrap_err();

        assert!(error.to_string().contains("dp_size must match"));
    }

    #[test]
    fn fpm_json_preserves_worker_and_dp_rank_identity() {
        let snapshots = fpm_snapshots_to_json(vec![(
            2,
            ForwardPassSnapshot {
                dp_rank: 3,
                ..Default::default()
            },
        )]);

        assert_eq!(snapshots[0]["worker_id"], 2);
        assert_eq!(snapshots[0]["dp_rank"], 3);
    }

    #[test]
    fn simple_synthetic_arrival_mode_changes_timestamps_only() {
        let fixed_timestamps = ArrivalSpec::ConstantQps { qps: 20.0 }
            .timestamps(8, 42)
            .unwrap();
        let poisson_timestamps = ArrivalSpec::PoissonQps { qps: 20.0 }
            .timestamps(8, 42)
            .unwrap();
        let fixed = build_synthetic_requests(16, 4, 8, Some(&fixed_timestamps)).unwrap();
        let poisson = build_synthetic_requests(16, 4, 8, Some(&poisson_timestamps)).unwrap();

        assert_ne!(fixed_timestamps, poisson_timestamps);
        assert_eq!(
            fixed
                .iter()
                .map(|request| request.arrival_timestamp_ms.unwrap())
                .collect::<Vec<_>>(),
            fixed_timestamps
        );
        assert_eq!(
            poisson
                .iter()
                .map(|request| request.arrival_timestamp_ms.unwrap())
                .collect::<Vec<_>>(),
            poisson_timestamps
        );
        for (fixed_request, poisson_request) in fixed.iter().zip(&poisson) {
            assert_eq!(fixed_request.tokens, poisson_request.tokens);
            assert_eq!(
                fixed_request.max_output_tokens,
                poisson_request.max_output_tokens
            );
            assert_eq!(
                fixed_request.output_token_ids,
                poisson_request.output_token_ids
            );
            assert_eq!(fixed_request.uuid, poisson_request.uuid);
            assert_eq!(fixed_request.dp_rank, poisson_request.dp_rank);
            assert_eq!(fixed_request.priority, poisson_request.priority);
            assert_eq!(
                fixed_request.strict_priority,
                poisson_request.strict_priority
            );
            assert_eq!(fixed_request.policy_class, poisson_request.policy_class);
        }
    }
}

fn load_replay_args_selection(
    py: Python<'_>,
    extra_engine_args: Option<MockEngineArgs>,
    prefill_engine_args: Option<MockEngineArgs>,
    decode_engine_args: Option<MockEngineArgs>,
    num_workers: usize,
    num_prefill_workers: usize,
    num_decode_workers: usize,
) -> PyResult<ReplayArgsSelection> {
    let aggregated_args = load_optional_replay_mocker_args(py, extra_engine_args)?;
    let prefill_args = load_optional_replay_mocker_args(py, prefill_engine_args)?;
    let decode_args = load_optional_replay_mocker_args(py, decode_engine_args)?;

    let replay_args_mode = dynamo_mocker::replay::validate_replay_args_mode(
        aggregated_args.as_ref(),
        prefill_args.as_ref(),
        decode_args.as_ref(),
        num_workers,
        num_prefill_workers,
        num_decode_workers,
    )
    .map_err(to_pyerr)?;

    match replay_args_mode {
        ReplayArgsMode::Aggregated => Ok(ReplayArgsSelection::Aggregated(Box::new(
            aggregated_args.unwrap_or_default(),
        ))),
        ReplayArgsMode::Disagg => Ok(ReplayArgsSelection::Disagg(Box::new(
            dynamo_mocker::replay::OfflineDisaggReplayConfig {
                prefill_args: prefill_args.expect("validated disagg prefill args"),
                decode_args: decode_args.expect("validated disagg decode args"),
                num_prefill_workers,
                num_decode_workers,
            },
        ))),
    }
}

fn load_optional_replay_mocker_args(
    py: Python<'_>,
    extra_engine_args: Option<MockEngineArgs>,
) -> PyResult<Option<RsMockEngineArgs>> {
    extra_engine_args
        .map(|extra_args| materialize_replay_mocker_args(py, extra_args))
        .transpose()
}

fn resolve_aic_backend_version(
    py: Python<'_>,
    backend: &str,
    configured_version: Option<&str>,
) -> PyResult<String> {
    py.import("dynamo._internal.aic")?
        .call_method1("resolve_backend_version", (backend, configured_version))?
        .extract()
}

fn materialize_replay_mocker_args(
    py: Python<'_>,
    extra_args: MockEngineArgs,
) -> PyResult<RsMockEngineArgs> {
    let mut args = extra_args.inner();
    populate_missing_offload_kv_bytes_per_token(py, &mut args)?;
    reconcile_replay_dp_topology(&mut args)
        .map_err(|error| PyException::new_err(error.to_string()))?;

    if let Some(ref backend_name) = args.aic_backend.clone() {
        let backend = backend_name.clone();
        let system = args.aic_system.as_deref().unwrap_or("h200_sxm").to_string();
        let model_name = args
            .aic_model_path
            .clone()
            .ok_or_else(|| PyException::new_err("--aic-perf-model requires --model-path"))?;
        let backend_version =
            resolve_aic_backend_version(py, &backend, args.aic_backend_version.as_deref())?;
        args.aic_backend_version = Some(backend_version.clone());
        let backend_version = Some(backend_version);
        let tp_size = args.aic_tp_size.unwrap_or(1);
        let moe_tp_size = args.aic_moe_tp_size;
        let moe_ep_size = args.aic_moe_ep_size;
        let attention_dp_size = args.aic_attention_dp_size;
        let gemm_dtype = args.aic_gemm_dtype.clone();
        let moe_dtype = args.aic_moe_dtype.clone();
        let fmha_dtype = args.aic_fmha_dtype.clone();
        let kv_cache_dtype = args.aic_kv_cache_dtype.clone();
        let comm_dtype = args.aic_comm_dtype.clone();
        let nextn = args.aic_nextn;
        let undiscounted_accept_rates = args.undiscounted_aic_accept_rates();
        // AIC-backed config may intentionally omit num_gpu_blocks. Estimate it
        // here, after candidate TP/backend/model overrides have been applied.
        let num_gpu_blocks_explicit = extra_args.num_gpu_blocks_explicit();
        // Under attention-DP, mirror the live path: one mocker worker owns
        // `dp_size` independent per-rank schedulers, each with a per-rank KV pool.
        // The topology applies whether KV capacity is explicit or estimated.
        if !num_gpu_blocks_explicit {
            let per_rank_blocks = estimate_aic_num_gpu_blocks(
                py,
                &backend,
                &system,
                &model_name,
                tp_size,
                args.block_size,
                args.max_num_batched_tokens.unwrap_or(8192),
                args.gpu_memory_utilization
                    .unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION),
                args.mem_fraction_static
                    .or(Some(DEFAULT_MEM_FRACTION_STATIC)),
                args.free_gpu_memory_fraction,
                backend_version.as_deref(),
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                gemm_dtype.as_deref(),
                moe_dtype.as_deref(),
                fmha_dtype.as_deref(),
                kv_cache_dtype.as_deref(),
                comm_dtype.as_deref(),
            )
            .map_err(|error| {
                PyException::new_err(format!(
                    "Failed to estimate AIC KV cache capacity \
                     (--aic-perf-model was requested): {error}"
                ))
            })?;
            // AIC returns a per-rank (per-GPU) block count. When replicating
            // attention-DP into per-rank workers, each worker owns this per-rank
            // pool (engine-wide capacity stays `per_rank * dp`, now partitioned
            // per rank as on real hardware). With dp == 1 the per-rank pool is
            // the engine-wide pool.
            args.num_gpu_blocks = per_rank_blocks;
        }
        let callback = create_aic_callback(
            py,
            &backend,
            &system,
            &model_name,
            tp_size,
            backend_version.as_deref(),
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            gemm_dtype.as_deref(),
            moe_dtype.as_deref(),
            fmha_dtype.as_deref(),
            kv_cache_dtype.as_deref(),
            comm_dtype.as_deref(),
            nextn,
            undiscounted_accept_rates.as_deref(),
        )
        .map_err(|e| {
            PyException::new_err(format!(
                "Failed to create AIC callback (--aic-perf-model was requested): {}",
                e
            ))
        })?;
        tracing::debug!(
            "AIC perf model: backend={}, gpu={}, model={}, version={:?}",
            backend,
            system,
            model_name,
            backend_version
        );
        // Every scheduler sees its own local batch, including under attention-DP.
        args.perf_model = Arc::new(PerfModel::from_aic_callback(callback));
    }

    Ok(args)
}

/// Reconcile the scheduler topology before optional AIC callback/capacity
/// materialization. This mirrors the JSON loader: an explicit attention-DP
/// size defines rank topology even when the caller supplies KV capacity and no
/// AIC backend.
fn reconcile_replay_dp_topology(args: &mut RsMockEngineArgs) -> anyhow::Result<()> {
    let attention_dp = args.aic_attention_dp_size;
    let dp = attention_dp.unwrap_or(1).max(1);
    let dp = u32::try_from(dp)
        .map_err(|_| anyhow::anyhow!("aic_attention_dp_size does not fit into a u32"))?;
    let has_aic_config = args.aic_backend.is_some() || attention_dp.is_some();
    if has_aic_config && args.dp_size > 1 && args.dp_size != dp {
        anyhow::bail!(
            "dp_size must match aic_attention_dp_size for AIC-backed replay (got dp_size={}, aic_attention_dp_size={dp})",
            args.dp_size
        );
    }
    if attention_dp.is_some() && dp > 1 {
        args.dp_size = dp;
    }
    Ok(())
}

fn populate_missing_offload_kv_bytes_per_token(
    py: Python<'_>,
    args: &mut RsMockEngineArgs,
) -> PyResult<()> {
    if args.kv_bytes_per_token.is_some() {
        return Ok(());
    }
    let offload_requested = args.num_g2_blocks.unwrap_or_default() > 0
        || args.num_g3_blocks.unwrap_or_default() > 0
        || args.enable_g4_storage;
    if !offload_requested {
        return Ok(());
    }
    let Some(model_path) = args.aic_model_path.as_deref() else {
        return Ok(());
    };

    // Match the Python `_resolve_kv_bytes_per_token`: normalize the configured
    // KV-cache dtype (auto/none -> "auto") and forward it so offload KV-byte
    // sizing reflects the quantized KV precision (e.g. fp8 = 1 byte) instead of
    // the model default.
    let kv_cache_dtype: Option<String> = py
        .import("dynamo._internal.aic")?
        .call_method1(
            "_normalize_aic_quant_mode",
            (args.aic_kv_cache_dtype.as_deref(),),
        )?
        .extract()?;
    let kv_cache_dtype = kv_cache_dtype.as_deref().unwrap_or("auto");

    let kv_cache_module = py.import("dynamo.mocker.utils.kv_cache")?;
    let kv_bytes_per_token = kv_cache_module
        .getattr("compute_kv_bytes_per_token")?
        .call1((model_path, kv_cache_dtype))?
        .extract::<Option<usize>>()?;
    if let Some(kv_bytes_per_token) = kv_bytes_per_token {
        args.kv_bytes_per_token = Some(kv_bytes_per_token);
    }
    Ok(())
}

fn load_replay_router_config(
    router_config: Option<KvRouterConfig>,
    model_name: Option<String>,
) -> PyResult<Option<dynamo_kv_router::config::KvRouterConfig>> {
    if model_name.as_ref().is_some_and(|name| name.is_empty()) {
        return Err(PyValueError::new_err("model_name must be non-empty"));
    }

    Ok(router_config.map(|config| config.inner().with_policy_model_name(model_name)))
}

fn load_replay_prefill_load_estimator<'a>(
    py: Python<'_>,
    router_mode: dynamo_mocker::replay::ReplayRouterMode,
    router_config: Option<&KvRouterConfig>,
    aic_perf_config: Option<&'a AicPerfConfig>,
) -> PyResult<(
    Option<dynamo_mocker::replay::ReplayPrefillLoadEstimator>,
    Option<ResolvedAicPerfConfig<'a>>,
)> {
    if router_mode != dynamo_mocker::replay::ReplayRouterMode::KvRouter {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_mode='kv_router'",
            ));
        }
        return Ok((None, None));
    }

    let Some(router_config) = router_config else {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_config with router_prefill_load_model='aic'",
            ));
        }
        return Ok((None, None));
    };

    let router_config = router_config.inner();
    if !router_config.router_prefill_load_model.is_enabled() {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_prefill_load_model='aic'",
            ));
        }
        return Ok((None, None));
    }

    let Some(aic_perf_config) = aic_perf_config else {
        return Err(PyException::new_err(
            "router_prefill_load_model='aic' requires aic_perf_config",
        ));
    };
    let resolved_aic_perf_config = resolve_aic_perf_config(py, Some(aic_perf_config))?
        .expect("AIC perf config resolution must preserve a present config");
    let aic_perf_config = resolved_aic_perf_config.config;

    let estimator = create_aic_prefill_load_estimator(
        py,
        aic_perf_config.backend_name(),
        aic_perf_config.system(),
        aic_perf_config.model_path(),
        aic_perf_config.tp_size(),
        Some(&resolved_aic_perf_config.backend_version),
        aic_perf_config.moe_tp_size(),
        aic_perf_config.moe_ep_size(),
        aic_perf_config.attention_dp_size(),
        aic_perf_config.gemm_dtype(),
        aic_perf_config.moe_dtype(),
        aic_perf_config.fmha_dtype(),
        aic_perf_config.kv_cache_dtype(),
        aic_perf_config.comm_dtype(),
        aic_perf_config.nextn(),
        aic_perf_config.nextn_accept_rates(),
    )?;
    Ok((Some(estimator), Some(resolved_aic_perf_config)))
}

fn parse_replay_router_mode(
    router_mode: &str,
) -> PyResult<dynamo_mocker::replay::ReplayRouterMode> {
    match router_mode {
        "round_robin" => Ok(dynamo_mocker::replay::ReplayRouterMode::RoundRobin),
        "kv_router" => Ok(dynamo_mocker::replay::ReplayRouterMode::KvRouter),
        other => Err(PyException::new_err(format!(
            "router_mode must be either 'round_robin' or 'kv_router', got '{}'",
            other
        ))),
    }
}

fn parse_trace_file_format(
    trace_format: &str,
) -> PyResult<dynamo_mocker::loadgen::TraceFileFormat> {
    match trace_format {
        "mooncake" => Ok(dynamo_mocker::loadgen::TraceFileFormat::Mooncake),
        "mooncake-delta" | "mooncake_delta" => {
            Ok(dynamo_mocker::loadgen::TraceFileFormat::MooncakeDelta)
        }
        "agentic_mooncake" | "agentic-mooncake" => {
            Ok(dynamo_mocker::loadgen::TraceFileFormat::AgenticMooncake)
        }
        "applied_compute_agentic" => {
            Ok(dynamo_mocker::loadgen::TraceFileFormat::AppliedComputeAgentic)
        }
        "dynamo" => Ok(dynamo_mocker::loadgen::TraceFileFormat::Dynamo),
        other => Err(PyException::new_err(format!(
            "trace_format must be 'mooncake', 'mooncake-delta', 'agentic_mooncake'/'agentic-mooncake', 'applied_compute_agentic', or 'dynamo', got '{}'",
            other
        ))),
    }
}

fn parse_replay_concurrency(replay_concurrency: Option<isize>) -> anyhow::Result<Option<usize>> {
    match replay_concurrency {
        Some(value) if value < 1 => anyhow::bail!("replay_concurrency must be at least 1"),
        Some(value) => Ok(Some(value as usize)),
        None => Ok(None),
    }
}

#[derive(Debug, Clone)]
enum SyntheticLoadController {
    Concurrency(usize),
    Arrivals(ArrivalSpec),
}

impl SyntheticLoadController {
    fn replay_concurrency(&self) -> Option<usize> {
        match self {
            Self::Concurrency(value) => Some(*value),
            Self::Arrivals(_) => None,
        }
    }

    fn arrival_spec(&self) -> Option<&ArrivalSpec> {
        match self {
            Self::Concurrency(_) => None,
            Self::Arrivals(spec) => Some(spec),
        }
    }
}

fn parse_synthetic_load_controller(
    replay_concurrency: Option<isize>,
    request_rate: Option<f64>,
    arrival_interval_ms: Option<f64>,
) -> anyhow::Result<SyntheticLoadController> {
    let controller_count = usize::from(replay_concurrency.is_some())
        + usize::from(request_rate.is_some())
        + usize::from(arrival_interval_ms.is_some());
    if controller_count != 1 {
        anyhow::bail!(
            "synthetic replay requires exactly one of replay_concurrency, request_rate, or arrival_interval_ms"
        );
    }

    if let Some(replay_concurrency) = parse_replay_concurrency(replay_concurrency)? {
        return Ok(SyntheticLoadController::Concurrency(replay_concurrency));
    }
    if let Some(qps) = request_rate {
        if !qps.is_finite() || qps <= 0.0 {
            anyhow::bail!("request_rate must be a finite positive number, got {qps}");
        }
        return Ok(SyntheticLoadController::Arrivals(ArrivalSpec::PoissonQps {
            qps,
        }));
    }

    let interval_ms = arrival_interval_ms.expect("controller count validated");
    if !interval_ms.is_finite() || interval_ms < 0.0 {
        anyhow::bail!(
            "arrival_interval_ms must be a finite non-negative number, got {interval_ms}"
        );
    }
    if interval_ms == 0.0 {
        return Ok(SyntheticLoadController::Arrivals(ArrivalSpec::Burst));
    }
    Ok(SyntheticLoadController::Arrivals(
        ArrivalSpec::ConstantQps {
            qps: 1000.0 / interval_ms,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_workload(
    block_size: usize,
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    first_turn_arrivals: ArrivalSpec,
    arrival_seed: u64,
    turns_per_session: usize,
    shared_prefix_ratio: f64,
    num_prefix_groups: usize,
    inter_turn_delay_ms: f64,
) -> anyhow::Result<RsTrace> {
    if input_tokens == 0 {
        anyhow::bail!("input_tokens must be at least 1");
    }
    if output_tokens == 0 {
        anyhow::bail!("output_tokens must be at least 1");
    }
    if request_count == 0 {
        anyhow::bail!("request_count must be at least 1");
    }
    if turns_per_session == 0 {
        anyhow::bail!("turns_per_session must be at least 1");
    }
    if !inter_turn_delay_ms.is_finite() || inter_turn_delay_ms < 0.0 {
        anyhow::bail!("inter_turn_delay_ms must be a finite non-negative number");
    }

    RsTrace::synthetic(SyntheticTraceSpec {
        block_size,
        num_sessions: request_count,
        turns_per_session,
        input_tokens: LengthSpec {
            mean: input_tokens,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: output_tokens,
            stddev: 0.0,
        },
        shared_prefix_ratio,
        num_prefix_groups,
        first_turn_arrivals,
        inter_turn_delays: if inter_turn_delay_ms == 0.0 {
            DelaySpec::None
        } else {
            DelaySpec::ConstantMs(inter_turn_delay_ms)
        },
        seed: 42,
        arrival_seed,
    })
}

fn build_synthetic_requests(
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    arrival_timestamps_ms: Option<&[f64]>,
) -> anyhow::Result<Vec<DirectRequest>> {
    if input_tokens == 0 {
        anyhow::bail!("input_tokens must be at least 1");
    }
    if output_tokens == 0 {
        anyhow::bail!("output_tokens must be at least 1");
    }
    if request_count == 0 {
        anyhow::bail!("request_count must be at least 1");
    }
    if let Some(arrival_timestamps_ms) = arrival_timestamps_ms
        && arrival_timestamps_ms.len() != request_count
    {
        anyhow::bail!(
            "arrival timestamp count {} does not match request_count {request_count}",
            arrival_timestamps_ms.len()
        );
    }

    let mut requests = Vec::with_capacity(request_count);
    for request_idx in 0..request_count {
        let tokens = (0..input_tokens)
            .map(|token_idx| synthetic_token_id(request_idx, token_idx))
            .collect();
        requests.push(DirectRequest {
            tokens,
            max_output_tokens: output_tokens,
            uuid: Some(Uuid::from_u128((request_idx as u128) + 1)),
            dp_rank: 0,
            arrival_timestamp_ms: arrival_timestamps_ms.map(|values| values[request_idx]),
            ..Default::default()
        });
    }

    Ok(requests)
}

fn synthetic_token_id(request_idx: usize, token_idx: usize) -> u32 {
    let mut value =
        (((request_idx as u64) << 32) ^ (token_idx as u64)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let token = value as u32;
    if token == 0 { 1 } else { token }
}

// ---------------------------------------------------------------------------
// Python scaling-policy adapter
// ---------------------------------------------------------------------------

fn fpm_snapshots_to_json(
    snapshots: Vec<(usize, dynamo_mocker::common::protocols::ForwardPassSnapshot)>,
) -> Vec<serde_json::Value> {
    snapshots
        .into_iter()
        .map(|(worker_id, fpm)| {
            json!({
                "worker_id": worker_id,
                "dp_rank": fpm.dp_rank,
                "wall_time": fpm.wall_time_secs,
                "num_prefill_requests": fpm.num_prefill_requests,
                "sum_prefill_tokens": fpm.sum_prefill_tokens,
                "var_prefill_length": fpm.var_prefill_length,
                "sum_prefill_kv_tokens": fpm.sum_prefill_kv_tokens,
                "num_decode_requests": fpm.num_decode_requests,
                "sum_decode_kv_tokens": fpm.sum_decode_kv_tokens,
                "var_decode_kv_tokens": fpm.var_decode_kv_tokens,
                "num_queued_prefill": fpm.num_queued_prefill,
                "sum_queued_prefill_tokens": fpm.sum_queued_prefill_tokens,
                "var_queued_prefill_length": fpm.var_queued_prefill_length,
                "num_queued_decode": fpm.num_queued_decode,
                "sum_queued_decode_kv_tokens": fpm.sum_queued_decode_kv_tokens,
                "var_queued_decode_kv_tokens": fpm.var_queued_decode_kv_tokens,
            })
        })
        .collect()
}

/// Reject a goodput SLA threshold that is not a finite, non-negative value;
/// `None` (unset) is allowed and means "do not gate on this dimension".
fn validate_sla_threshold(name: &str, value: Option<f64>) -> PyResult<()> {
    if let Some(v) = value
        && (!v.is_finite() || v < 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite, non-negative value, got {v}"
        )));
    }
    Ok(())
}

/// Convert a scaling-run error back into a `PyErr`, preserving the original
/// Python exception (its type and traceback) when the failure originated in a
/// scaling callback (`initial_tick_ms` / `on_tick` stash the `PyErr` via
/// `anyhow::Error::new`). Non-Python errors (e.g. a simulation dead-end) fall
/// back to the generic conversion.
fn scaling_run_err_to_pyerr(err: anyhow::Error) -> PyErr {
    match err.downcast::<PyErr>() {
        Ok(py_err) => py_err,
        Err(other) => to_pyerr(other),
    }
}

/// Adapts a Python object to [`ReplayScalingPolicy`]. The high-level replay call
/// keeps the GIL for Python-backed scaling, so each tick is a cheap re-entry.
struct PyReplayScalingPolicy {
    callback: Py<PyAny>,
}

impl ReplayScalingPolicy for PyReplayScalingPolicy {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Python::with_gil(|py| {
            self.callback
                .bind(py)
                .call_method0("initial_tick_ms")?
                .extract::<f64>()
        })
        .map_err(anyhow::Error::new)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        let ReplayScalingSnapshot {
            tick_ordinal,
            now_ms,
            prefill_fpm,
            decode_fpm,
            traffic,
            active_prefill_ids,
            active_decode_ids,
            starting_prefill_ids,
            starting_decode_ids,
            draining_prefill_ids,
            draining_decode_ids,
        } = snapshot;
        Python::with_gil(|py| -> PyResult<ReplayScalingDecision> {
            let non_draining_prefill_count = active_prefill_ids.len() + starting_prefill_ids.len();
            let non_draining_decode_count = active_decode_ids.len() + starting_decode_ids.len();
            let total_prefill_count = non_draining_prefill_count + draining_prefill_ids.len();
            let total_decode_count = non_draining_decode_count + draining_decode_ids.len();
            let metrics_json = json!({
                "tick_ordinal": tick_ordinal,
                "now_ms": now_ms,
                "prefill_fpm_snapshots": fpm_snapshots_to_json(prefill_fpm),
                "decode_fpm_snapshots": fpm_snapshots_to_json(decode_fpm),
                "active_prefill_count": active_prefill_ids.len(),
                "active_decode_count": active_decode_ids.len(),
                "active_prefill_ids": active_prefill_ids,
                "active_decode_ids": active_decode_ids,
                "starting_prefill_count": starting_prefill_ids.len(),
                "starting_decode_count": starting_decode_ids.len(),
                "starting_prefill_ids": starting_prefill_ids,
                "starting_decode_ids": starting_decode_ids,
                "draining_prefill_count": draining_prefill_ids.len(),
                "draining_decode_count": draining_decode_ids.len(),
                "draining_prefill_ids": draining_prefill_ids,
                "draining_decode_ids": draining_decode_ids,
                "non_draining_prefill_count": non_draining_prefill_count,
                "non_draining_decode_count": non_draining_decode_count,
                "total_prefill_count": total_prefill_count,
                "total_decode_count": total_decode_count,
                "traffic": {
                    "duration_s": traffic.duration_s,
                    "num_req": traffic.num_req,
                    "avg_isl": traffic.avg_isl,
                    "avg_osl": traffic.avg_osl,
                    "avg_ttft_ms": traffic.avg_ttft_ms,
                    "avg_itl_ms": traffic.avg_itl_ms,
                    // Native denominators keep completion-derived shape and
                    // latency averages exact even though num_req is offered load.
                    "shape_count": traffic.shape_count,
                    "ttft_count": traffic.ttft_count,
                    "itl_count": traffic.itl_count,
                    "avg_accept_length": traffic.avg_accept_length,
                    "avg_kv_hit_rate": traffic.avg_kv_hit_rate,
                    // Denominators behind the two ratio averages, so the Python
                    // adapter can merge partial windows with exact count weights
                    // instead of approximating with num_req.
                    "hit_rate_count": traffic.hit_rate_count,
                    "accept_length_forward_count": traffic.accept_length_forward_count,
                },
            });
            let metrics_obj = pythonize(py, &metrics_json).map_err(to_pyerr)?;
            let decision = self
                .callback
                .bind(py)
                .call_method1("on_tick", (metrics_obj,))?;
            Ok(ReplayScalingDecision {
                target_prefill: decision.get_item("target_prefill")?.extract()?,
                target_decode: decision.get_item("target_decode")?.extract()?,
                next_tick_ms: decision.get_item("next_tick_ms")?.extract()?,
            })
        })
        .map_err(anyhow::Error::new)
    }
}
