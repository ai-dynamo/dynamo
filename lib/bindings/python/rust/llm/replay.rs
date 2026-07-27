// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use dynamo_mocker::replay::{PlannerHook, PlannerTickDecision, PlannerTickMetrics, ReplayArgsMode};
use pyo3::{
    exceptions::{PyException, PyValueError},
    prelude::*,
};
use pythonize::pythonize;
use serde_json::json;
use uuid::Uuid;

use super::aic_callback::{
    create_aic_callback, create_aic_prefill_load_estimator, estimate_aic_num_gpu_blocks,
};
use super::entrypoint::{AicPerfConfig, KvRouterConfig, to_pyerr};

const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.9;
const DEFAULT_MEM_FRACTION_STATIC: f64 = 0.88;

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
    #[pyo3(signature = (engine_type="vllm", num_gpu_blocks=None, block_size=0, max_num_seqs=Some(256), max_num_batched_tokens=Some(8192), enable_prefix_caching=true, enable_chunked_prefill=true, speedup_ratio=1.0, decode_speedup_ratio=1.0, dp_size=1, startup_time=None, worker_type="aggregated", planner_profile_data=None, aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, aic_nextn=None, aic_nextn_accept_rates=None, aic_mtp_seed=42, aic_gemm_dtype=None, aic_moe_dtype=None, aic_fmha_dtype=None, aic_kv_cache_dtype=None, aic_comm_dtype=None, gpu_memory_utilization=None, mem_fraction_static=None, free_gpu_memory_fraction=None, enable_local_indexer=false, bootstrap_port=None, handoff_session_timeout_ms=300000, kv_bytes_per_token=None, kv_transfer_bandwidth=None, kv_transfer_timing_mode="full_prompt", reasoning=None, response_replay_trace_path=None, zmq_kv_events_port=None, zmq_replay_port=None, preemption_mode="lifo", router_queue_policy=None, sglang=None, trtllm=None, num_g2_blocks=None, num_g3_blocks=None, offload_batch_size=None, bandwidth_g1_to_g2_gbps=None, bandwidth_g2_to_g1_gbps=None, bandwidth_g2_to_g3_gbps=None, bandwidth_g3_to_g2_gbps=None, enable_g4_storage=false, bandwidth_g2_to_g4_gbps=None, bandwidth_g4_to_g2_gbps=None, max_model_len=None, g1_backend="kvbm"))]
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
        g1_backend: &str,
    ) -> PyResult<Self> {
        let engine_type = parse_mocker_engine_type(engine_type)?;
        let worker_type = parse_worker_type(worker_type)?;
        let preemption_mode = parse_preemption_mode(preemption_mode)?;
        let g1_backend = parse_g1_backend(g1_backend)?;
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
            .g1_backend(g1_backend)
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
        match self.inner.g1_backend {
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

#[pyfunction]
#[pyo3(signature = (trace_files, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, trace_block_size=None, trace_format="mooncake", trace_shared_prefix_ratio=0.0, trace_num_prefix_groups=0, report_jsonl_path=None, max_sim_time_ms=None, model_name=None, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None))]
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
) -> PyResult<PyObject> {
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
    let prefill_load_estimator = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config, model_name)?;
    let replay_mode = replay_mode.to_owned();
    if report_jsonl_path.is_some() && replay_mode != "offline" {
        return Err(PyValueError::new_err(
            "report_jsonl_path is only supported for replay_mode='offline'",
        ));
    }
    let jsonl_path_for_emit = report_jsonl_path.clone();
    let record_per_request = report_jsonl_path.is_some();
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
    // Goodput SLA: when set, the collector classifies SLA-satisfying requests and
    // the report carries goodput_* keys. Offline replay only (the online/live
    // entrypoints don't take it) — reject it for non-offline modes rather than
    // silently dropping it, matching report_jsonl_path / max_sim_time_ms. With none
    // set, goodput is omitted as before.
    validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
    validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
    validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
    if replay_mode != "offline"
        && (sla_ttft_ms.is_some() || sla_itl_ms.is_some() || sla_e2e_ms.is_some())
    {
        return Err(PyValueError::new_err(
            "sla_ttft_ms, sla_itl_ms, and sla_e2e_ms only support replay_mode='offline'",
        ));
    }
    let sla = dynamo_mocker::replay::SlaThresholds {
        ttft_ms: sla_ttft_ms,
        itl_ms: sla_itl_ms,
        e2e_ms: sla_e2e_ms,
    };
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;
        if trace_format == dynamo_mocker::loadgen::TraceFileFormat::Dynamo {
            let trace = DynamoRequestTrace::from_request_trace_files(
                &trace_files,
                trace_block_size,
            )?;
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
                dynamo_mocker::replay::simulate_concurrency_file_with_router_mode_and_format(
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
                )
            }
            ReplayDispatch::AggregatedOffline(args) => {
                dynamo_mocker::replay::simulate_trace_file_with_router_mode_and_format(
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
                )
            }
            ReplayDispatch::AggregatedOnlineConcurrency(args, max_in_flight) => {
                dynamo_mocker::replay::simulate_concurrency_live_file_with_router_mode_and_format(
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
                )
            }
            ReplayDispatch::AggregatedOnline(args) => {
                dynamo_mocker::replay::simulate_trace_live_file_with_router_mode_and_format(
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
                )
            }
            ReplayDispatch::DisaggOfflineConcurrency(config, max_in_flight) => {
                dynamo_mocker::replay::simulate_concurrency_file_disagg_with_router_mode_and_format(
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
                )
            }
            ReplayDispatch::DisaggOffline(config) => {
                dynamo_mocker::replay::simulate_trace_file_disagg_with_router_mode_and_format(
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
                )
            }
        }
    });
    let report = report.map_err(to_pyerr)?;
    // Write per-request JSONL from Rust directly if requested, avoiding a
    // potentially-large round trip through pyo3 / pythonize. Each line is one
    // JSON object (matching AIPerf's profile_export.jsonl convention).
    if let Some(path) = jsonl_path_for_emit.as_ref() {
        py.allow_threads(|| write_per_request_jsonl(path, &report.per_request))
            .map_err(to_pyerr)?;
    }
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
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
) -> anyhow::Result<dynamo_mocker::replay::TraceSimulationReport> {
    match trace {
        DynamoRequestTrace::Standard(trace) => {
            match select_replay_dispatch(args_selection, replay_mode, replay_concurrency)? {
                ReplayDispatch::AggregatedOfflineConcurrency(args, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_workload_with_router_mode_and_options(
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
                        )
                }
                ReplayDispatch::AggregatedOffline(args) => {
                    dynamo_mocker::replay::simulate_loaded_trace_with_router_mode_and_options(
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
                    )
                }
                ReplayDispatch::AggregatedOnlineConcurrency(args, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_live_workload_with_router_mode(
                        *args,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        max_in_flight,
                        num_workers,
                        router_mode,
                    )
                }
                ReplayDispatch::AggregatedOnline(args) => {
                    dynamo_mocker::replay::simulate_loaded_trace_live_with_router_mode(
                        *args,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        num_workers,
                        arrival_speedup_ratio,
                        router_mode,
                    )
                }
                ReplayDispatch::DisaggOfflineConcurrency(config, max_in_flight) => {
                    dynamo_mocker::replay::simulate_concurrency_workload_disagg_with_router_mode_and_options(
                            *config,
                            router_config,
                            prefill_load_estimator,
                            trace,
                            max_in_flight,
                            router_mode,
                            record_per_request,
                            max_sim_time_ms,
                            sla,
                        )
                }
                ReplayDispatch::DisaggOffline(config) => {
                    dynamo_mocker::replay::simulate_loaded_trace_disagg_with_router_mode_and_options(
                        *config,
                        router_config,
                        prefill_load_estimator,
                        trace,
                        arrival_speedup_ratio,
                        router_mode,
                        record_per_request,
                        max_sim_time_ms,
                        sla,
                    )
                }
            }
        }
        DynamoRequestTrace::Agentic(trace) => {
            if replay_concurrency.is_some() {
                anyhow::bail!(
                    "agentic Dynamo request traces are not supported with replay_concurrency"
                );
            }
            if replay_mode != "offline" {
                anyhow::bail!("agentic Dynamo request traces are not supported for online replay");
            }
            let ReplayArgsSelection::Aggregated(args) = args_selection else {
                anyhow::bail!(
                    "agentic Dynamo request traces are not supported for disaggregated replay"
                );
            };
            let trace = trace
                .normalize_starts()
                .speed_up_timing(arrival_speedup_ratio)?;
            dynamo_mocker::replay::simulate_agentic_trace_workload_with_router_mode(
                *args,
                router_config,
                prefill_load_estimator,
                trace,
                num_workers,
                router_mode,
                sla,
            )
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
#[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, request_rate=None, arrival_interval_ms=None, arrival_seed=42, turns_per_session=1, shared_prefix_ratio=0.0, num_prefix_groups=0, inter_turn_delay_ms=0.0, model_name=None, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None))]
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
) -> PyResult<PyObject> {
    validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
    validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
    validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
    let sla = dynamo_mocker::replay::SlaThresholds {
        ttft_ms: sla_ttft_ms,
        itl_ms: sla_itl_ms,
        e2e_ms: sla_e2e_ms,
    };
    // The online branches below don't thread `sla`, so reject SLA with a
    // non-offline replay_mode rather than silently dropping goodput
    // (mirrors run_mocker_trace_replay).
    if replay_mode != "offline"
        && (sla_ttft_ms.is_some() || sla_itl_ms.is_some() || sla_e2e_ms.is_some())
    {
        return Err(PyValueError::new_err(
            "sla_ttft_ms, sla_itl_ms, and sla_e2e_ms only support replay_mode='offline'",
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
    let prefill_load_estimator = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config, model_name)?;
    let replay_mode = replay_mode.to_owned();
    let block_size = match &args_selection {
        ReplayArgsSelection::Aggregated(args) => args.block_size.max(1),
        ReplayArgsSelection::Disagg(config) => config.prefill_args.block_size.max(1),
    };
    let report = py.allow_threads(move || {
        let load_controller = parse_synthetic_load_controller(
            replay_concurrency,
            request_rate,
            arrival_interval_ms,
        )?;
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
                        dynamo_mocker::replay::simulate_concurrency_workload_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            num_workers,
                            router_mode,
                            sla,
                        )
                    }
                    ("offline", None) => {
                        dynamo_mocker::replay::simulate_trace_workload_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            num_workers,
                            router_mode,
                            sla,
                        )
                    }
                    ("online", Some(max_in_flight)) => {
                        dynamo_mocker::replay::simulate_concurrency_live_workload_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            num_workers,
                            router_mode,
                        )
                    }
                    ("online", None) => {
                        dynamo_mocker::replay::simulate_trace_live_workload_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            num_workers,
                            router_mode,
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
                        ("offline", Some(max_in_flight)) => dynamo_mocker::replay::simulate_concurrency_workload_disagg_with_router_mode(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            router_mode,
                            sla,
                        ),
                        ("offline", None) => dynamo_mocker::replay::simulate_trace_workload_disagg_with_router_mode(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            router_mode,
                            sla,
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
                    dynamo_mocker::replay::simulate_concurrency_requests_with_router_mode(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        num_workers,
                        router_mode,
                        sla,
                    )
                }
                ("offline", None) => dynamo_mocker::replay::simulate_trace_requests_with_router_mode(
                    *args,
                    router_config.clone(),
                    prefill_load_estimator.clone(),
                    requests,
                    num_workers,
                    arrival_speedup_ratio,
                    router_mode,
                    sla,
                ),
                ("online", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_live_requests_with_router_mode(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        num_workers,
                        router_mode,
                    )
                }
                ("online", None) => {
                    dynamo_mocker::replay::simulate_trace_live_requests_with_router_mode(
                        *args,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        num_workers,
                        arrival_speedup_ratio,
                        router_mode,
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
                    dynamo_mocker::replay::simulate_concurrency_requests_disagg_with_router_mode(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        router_mode,
                        sla,
                    )
                }
                ("offline", None) => {
                    dynamo_mocker::replay::simulate_trace_requests_disagg_with_router_mode(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        arrival_speedup_ratio,
                        router_mode,
                        sla,
                    )
                }
                (other, _) => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
                }
            }
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
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
        build_synthetic_requests, fpm_snapshots_to_json, reconcile_replay_dp_topology,
        validate_disagg_replay_mode,
    };
    use dynamo_mocker::common::protocols::{ForwardPassSnapshot, MockEngineArgs};
    use dynamo_mocker::loadgen::ArrivalSpec;

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
        let backend_version = args.aic_backend_version.clone();
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
            .map_err(|e| {
                PyException::new_err(format!(
                    "Failed to estimate AIC KV cache capacity (--aic-perf-model was requested): {}",
                    e
                ))
            })?;
            // AIC returns a per-rank (per-GPU) block count. When replicating attention-DP
            // into per-rank workers, each worker owns this per-rank pool (engine-wide
            // capacity stays `per_rank * dp`, now partitioned per rank as on real hardware).
            // With dp == 1 the per-rank pool is the engine-wide pool.
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

fn load_replay_prefill_load_estimator(
    py: Python<'_>,
    router_mode: dynamo_mocker::replay::ReplayRouterMode,
    router_config: Option<&KvRouterConfig>,
    aic_perf_config: Option<&AicPerfConfig>,
) -> PyResult<Option<dynamo_mocker::replay::ReplayPrefillLoadEstimator>> {
    if router_mode != dynamo_mocker::replay::ReplayRouterMode::KvRouter {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_mode='kv_router'",
            ));
        }
        return Ok(None);
    }

    let Some(router_config) = router_config else {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_config with router_prefill_load_model='aic'",
            ));
        }
        return Ok(None);
    };

    let router_config = router_config.inner();
    if !router_config.router_prefill_load_model.is_enabled() {
        if aic_perf_config.is_some() {
            return Err(PyException::new_err(
                "aic_perf_config requires router_prefill_load_model='aic'",
            ));
        }
        return Ok(None);
    }

    let Some(aic_perf_config) = aic_perf_config else {
        return Err(PyException::new_err(
            "router_prefill_load_model='aic' requires aic_perf_config",
        ));
    };

    create_aic_prefill_load_estimator(
        py,
        aic_perf_config.backend_name(),
        aic_perf_config.system(),
        aic_perf_config.model_path(),
        aic_perf_config.tp_size(),
        aic_perf_config.backend_version(),
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
    )
    .map(Some)
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
// Planner-in-the-loop replay bridge
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

#[pyclass(unsendable)]
pub struct PlannerReplayBridge {
    handle: Option<dynamo_mocker::replay::PlannerReplayHandle>,
}

#[pymethods]
impl PlannerReplayBridge {
    /// Create a bridge for an aggregated Mooncake-style JSONL trace replay.
    #[new]
    #[pyo3(signature = (trace_file, extra_engine_args, num_workers, router_mode="round_robin", router_config=None, model_name=None, arrival_speedup_ratio=1.0, trace_block_size=512, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None, replay_concurrency=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        trace_file: PathBuf,
        extra_engine_args: &MockEngineArgs,
        num_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        model_name: Option<String>,
        arrival_speedup_ratio: f64,
        trace_block_size: usize,
        sla_ttft_ms: Option<f64>,
        sla_itl_ms: Option<f64>,
        sla_e2e_ms: Option<f64>,
        replay_concurrency: Option<isize>,
    ) -> PyResult<Self> {
        let args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, extra_engine_args.clone()))?;
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config, model_name)?;
        validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
        validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
        validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
        let sla = dynamo_mocker::replay::SlaThresholds {
            ttft_ms: sla_ttft_ms,
            itl_ms: sla_itl_ms,
            e2e_ms: sla_e2e_ms,
        };

        let max_in_flight = parse_replay_concurrency(replay_concurrency).map_err(to_pyerr)?;
        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace_file(
            args,
            router_config,
            None,
            &trace_file,
            trace_block_size,
            num_workers,
            arrival_speedup_ratio,
            max_in_flight,
            router_mode,
            sla,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Create a bridge for a disaggregated Mooncake-style JSONL trace replay.
    #[staticmethod]
    #[pyo3(signature = (trace_file, prefill_engine_args, decode_engine_args, num_prefill_workers, num_decode_workers, router_mode="round_robin", router_config=None, model_name=None, arrival_speedup_ratio=1.0, trace_block_size=512, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None, replay_concurrency=None))]
    #[allow(clippy::too_many_arguments)]
    fn create_disagg(
        trace_file: PathBuf,
        prefill_engine_args: &MockEngineArgs,
        decode_engine_args: &MockEngineArgs,
        num_prefill_workers: usize,
        num_decode_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        model_name: Option<String>,
        arrival_speedup_ratio: f64,
        trace_block_size: usize,
        sla_ttft_ms: Option<f64>,
        sla_itl_ms: Option<f64>,
        sla_e2e_ms: Option<f64>,
        replay_concurrency: Option<isize>,
    ) -> PyResult<Self> {
        let prefill_args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, prefill_engine_args.clone()))?;
        let decode_args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, decode_engine_args.clone()))?;
        let config = dynamo_mocker::replay::OfflineDisaggReplayConfig {
            prefill_args,
            decode_args,
            num_prefill_workers,
            num_decode_workers,
        };
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config, model_name)?;
        validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
        validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
        validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
        let sla = dynamo_mocker::replay::SlaThresholds {
            ttft_ms: sla_ttft_ms,
            itl_ms: sla_itl_ms,
            e2e_ms: sla_e2e_ms,
        };

        let max_in_flight = parse_replay_concurrency(replay_concurrency).map_err(to_pyerr)?;
        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace_file_disagg(
            config,
            router_config,
            None,
            &trace_file,
            trace_block_size,
            arrival_speedup_ratio,
            max_in_flight,
            router_mode,
            sla,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Create a bridge for an aggregated **synthetic** workload: `request_count`
    /// sessions of fixed `input_tokens`/`output_tokens`. `replay_concurrency =
    /// Some(n)` runs closed-loop (cap n in flight); otherwise `request_rate`
    /// selects Poisson open-loop or `arrival_interval_ms` selects fixed-rate
    /// open-loop replay. `shared_prefix_ratio` /
    /// `num_prefix_groups` control prefix-cache sharing; `turns_per_session` > 1
    /// makes each session multi-turn (total requests = request_count * turns).
    #[staticmethod]
    #[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args, num_workers, router_mode="round_robin", router_config=None, model_name=None, replay_concurrency=None, arrival_speedup_ratio=1.0, request_rate=None, arrival_interval_ms=None, arrival_seed=42, turns_per_session=1, shared_prefix_ratio=0.0, num_prefix_groups=0, inter_turn_delay_ms=0.0, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_synthetic(
        input_tokens: usize,
        output_tokens: usize,
        request_count: usize,
        extra_engine_args: &MockEngineArgs,
        num_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        model_name: Option<String>,
        replay_concurrency: Option<isize>,
        arrival_speedup_ratio: f64,
        request_rate: Option<f64>,
        arrival_interval_ms: Option<f64>,
        arrival_seed: u64,
        turns_per_session: usize,
        shared_prefix_ratio: f64,
        num_prefix_groups: usize,
        inter_turn_delay_ms: f64,
        sla_ttft_ms: Option<f64>,
        sla_itl_ms: Option<f64>,
        sla_e2e_ms: Option<f64>,
    ) -> PyResult<Self> {
        let args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, extra_engine_args.clone()))?;
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config, model_name)?;
        validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
        validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
        validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
        let sla = dynamo_mocker::replay::SlaThresholds {
            ttft_ms: sla_ttft_ms,
            itl_ms: sla_itl_ms,
            e2e_ms: sla_e2e_ms,
        };
        let load_controller =
            parse_synthetic_load_controller(replay_concurrency, request_rate, arrival_interval_ms)
                .map_err(to_pyerr)?;
        let max_in_flight = load_controller.replay_concurrency();

        let block_size = args.block_size.max(1);
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
        )
        .map_err(to_pyerr)?;
        // Arrival speedup only applies in arrival mode; closed-loop ignores timing.
        if max_in_flight.is_none() {
            trace = trace
                .speed_up_timing(arrival_speedup_ratio)
                .map_err(to_pyerr)?;
        }

        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace(
            args,
            router_config,
            None,
            trace,
            num_workers,
            max_in_flight,
            router_mode,
            sla,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Create a bridge for a disaggregated **synthetic** workload. See
    /// [`PlannerReplayBridge::from_synthetic`] for the load-shape parameters.
    #[staticmethod]
    #[pyo3(signature = (input_tokens, output_tokens, request_count, prefill_engine_args, decode_engine_args, num_prefill_workers, num_decode_workers, router_mode="round_robin", router_config=None, model_name=None, replay_concurrency=None, arrival_speedup_ratio=1.0, request_rate=None, arrival_interval_ms=None, arrival_seed=42, turns_per_session=1, shared_prefix_ratio=0.0, num_prefix_groups=0, inter_turn_delay_ms=0.0, sla_ttft_ms=None, sla_itl_ms=None, sla_e2e_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_synthetic_disagg(
        input_tokens: usize,
        output_tokens: usize,
        request_count: usize,
        prefill_engine_args: &MockEngineArgs,
        decode_engine_args: &MockEngineArgs,
        num_prefill_workers: usize,
        num_decode_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        model_name: Option<String>,
        replay_concurrency: Option<isize>,
        arrival_speedup_ratio: f64,
        request_rate: Option<f64>,
        arrival_interval_ms: Option<f64>,
        arrival_seed: u64,
        turns_per_session: usize,
        shared_prefix_ratio: f64,
        num_prefix_groups: usize,
        inter_turn_delay_ms: f64,
        sla_ttft_ms: Option<f64>,
        sla_itl_ms: Option<f64>,
        sla_e2e_ms: Option<f64>,
    ) -> PyResult<Self> {
        let prefill_args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, prefill_engine_args.clone()))?;
        let decode_args =
            Python::with_gil(|py| materialize_replay_mocker_args(py, decode_engine_args.clone()))?;
        let config = dynamo_mocker::replay::OfflineDisaggReplayConfig {
            prefill_args,
            decode_args,
            num_prefill_workers,
            num_decode_workers,
        };
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config, model_name)?;
        validate_sla_threshold("sla_ttft_ms", sla_ttft_ms)?;
        validate_sla_threshold("sla_itl_ms", sla_itl_ms)?;
        validate_sla_threshold("sla_e2e_ms", sla_e2e_ms)?;
        let sla = dynamo_mocker::replay::SlaThresholds {
            ttft_ms: sla_ttft_ms,
            itl_ms: sla_itl_ms,
            e2e_ms: sla_e2e_ms,
        };
        let load_controller =
            parse_synthetic_load_controller(replay_concurrency, request_rate, arrival_interval_ms)
                .map_err(to_pyerr)?;
        let max_in_flight = load_controller.replay_concurrency();

        let block_size = config.prefill_args.block_size.max(1);
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
        )
        .map_err(to_pyerr)?;
        if max_in_flight.is_none() {
            trace = trace
                .speed_up_timing(arrival_speedup_ratio)
                .map_err(to_pyerr)?;
        }

        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace_disagg(
            config,
            router_config,
            None,
            trace,
            max_in_flight,
            router_mode,
            sla,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Run the whole replay to completion with the Python planner driving the tick
    /// cadence (the unified replacement for the advance_to/apply_scaling stepping
    /// loop). `planner` must expose `initial_tick_ms() -> float` and
    /// `on_tick(metrics: dict) -> dict` with keys `target_prefill`/`target_decode`
    /// (int | None) and `next_tick_ms` (float | None). The simulation owns the drive
    /// loop and calls back into `planner` once per `PlannerTick`; the GIL is held
    /// throughout, so each callback is a cheap re-entry. Returns the trace report.
    fn run(&mut self, py: Python<'_>, planner: Py<PyAny>) -> PyResult<PyObject> {
        let handle = self
            .handle
            .take()
            .ok_or_else(|| PyException::new_err("bridge has been finalized"))?;
        let hook: Box<dyn PlannerHook> = Box::new(PyPlannerHook { callback: planner });
        let report = handle.run(hook).map_err(planner_run_err_to_pyerr)?;
        pythonize(py, &report)
            .map_err(to_pyerr)
            .map(|obj| obj.unbind())
    }
}

/// Convert a planner-run error back into a `PyErr`, preserving the original
/// Python exception (its type and traceback) when the failure originated in a
/// planner callback (`initial_tick_ms` / `on_tick` stash the `PyErr` via
/// `anyhow::Error::new`). Non-Python errors (e.g. a simulation dead-end) fall
/// back to the generic conversion.
fn planner_run_err_to_pyerr(err: anyhow::Error) -> PyErr {
    match err.downcast::<PyErr>() {
        Ok(py_err) => py_err,
        Err(other) => to_pyerr(other),
    }
}

/// Adapts a Python planner object to the Rust [`PlannerHook`] trait. Invoked once
/// per `PlannerTick` from inside the simulation's `run()` loop; the bridge holds the
/// GIL while running so the per-tick `Python::with_gil` is a cheap re-entry.
struct PyPlannerHook {
    callback: Py<PyAny>,
}

impl PlannerHook for PyPlannerHook {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Python::with_gil(|py| {
            self.callback
                .bind(py)
                .call_method0("initial_tick_ms")?
                .extract::<f64>()
        })
        // Preserve the original `PyErr` (type + traceback) through the anyhow
        // boundary so `PlannerReplayBridge::run` can re-raise it unchanged.
        .map_err(anyhow::Error::new)
    }

    fn on_tick(&mut self, metrics: PlannerTickMetrics) -> anyhow::Result<PlannerTickDecision> {
        let PlannerTickMetrics {
            now_ms,
            prefill_fpm,
            decode_fpm,
            traffic,
            active_prefill_ids,
            active_decode_ids,
            total_prefill,
            total_decode,
        } = metrics;
        Python::with_gil(|py| -> PyResult<PlannerTickDecision> {
            // The metrics dict mirrors the old `advance_to` + `drain_traffic` dicts so
            // the Python adapter's `_build_tick_input` consumes it unchanged.
            let metrics_json = json!({
                "now_ms": now_ms,
                "prefill_fpm_snapshots": fpm_snapshots_to_json(prefill_fpm),
                "decode_fpm_snapshots": fpm_snapshots_to_json(decode_fpm),
                "active_prefill_count": active_prefill_ids.len(),
                "active_decode_count": active_decode_ids.len(),
                "active_prefill_ids": active_prefill_ids,
                "active_decode_ids": active_decode_ids,
                "total_prefill_count": total_prefill,
                "total_decode_count": total_decode,
                "traffic": {
                    "duration_s": traffic.duration_s,
                    "num_req": traffic.num_req,
                    "avg_isl": traffic.avg_isl,
                    "avg_osl": traffic.avg_osl,
                    "avg_ttft_ms": traffic.avg_ttft_ms,
                    "avg_itl_ms": traffic.avg_itl_ms,
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
            // The planner returns a dict with all three keys (values may be None).
            Ok(PlannerTickDecision {
                target_prefill: decision.get_item("target_prefill")?.extract()?,
                target_decode: decision.get_item("target_decode")?.extract()?,
                next_tick_ms: decision.get_item("next_tick_ms")?.extract()?,
            })
        })
        // Preserve the original `PyErr` (type + traceback) through the anyhow
        // boundary so `PlannerReplayBridge::run` can re-raise it unchanged.
        .map_err(anyhow::Error::new)
    }
}
