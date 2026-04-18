// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use dynamo_mocker::common::perf_model::PerfModel;
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType as RsMockerEngineType, MockEngineArgs as RsMockEngineArgs,
    PreemptionMode as RsPreemptionMode, ReasoningConfig as RsReasoningConfig,
    SglangArgs as RsSglangArgs, WorkerType as RsWorkerType,
};
use dynamo_mocker::loadgen::{
    ArrivalSpec, DelaySpec, LengthSpec, SyntheticTraceSpec, Trace as RsTrace,
};
use dynamo_mocker::replay::ReplayArgsMode;
use pyo3::{exceptions::PyException, prelude::*};
use pythonize::pythonize;
use serde_json::json;
use uuid::Uuid;

use super::aic_callback::{create_aic_callback, create_aic_prefill_load_estimator};
use super::entrypoint::{AicPerfConfig, KvRouterConfig, to_pyerr};

fn parse_mocker_engine_type(engine_type: &str) -> PyResult<RsMockerEngineType> {
    match engine_type {
        "vllm" => Ok(RsMockerEngineType::Vllm),
        "sglang" => Ok(RsMockerEngineType::Sglang),
        other => Err(PyException::new_err(format!(
            "engine_type must be either 'vllm' or 'sglang', got '{other}'"
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
pub struct MockEngineArgs {
    inner: RsMockEngineArgs,
}

impl MockEngineArgs {
    pub fn inner(&self) -> RsMockEngineArgs {
        self.inner.clone()
    }
}

#[pymethods]
impl MockEngineArgs {
    #[new]
    #[pyo3(signature = (engine_type="vllm", num_gpu_blocks=16384, block_size=0, max_num_seqs=Some(256), max_num_batched_tokens=Some(2048), enable_prefix_caching=true, enable_chunked_prefill=true, speedup_ratio=1.0, decode_speedup_ratio=1.0, dp_size=1, startup_time=None, worker_type="aggregated", planner_profile_data=None, aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, enable_local_indexer=false, bootstrap_port=None, kv_bytes_per_token=None, kv_transfer_bandwidth=None, reasoning=None, zmq_kv_events_port=None, zmq_replay_port=None, preemption_mode="lifo", router_queue_policy=None, sglang=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine_type: &str,
        num_gpu_blocks: usize,
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
        enable_local_indexer: bool,
        bootstrap_port: Option<u16>,
        kv_bytes_per_token: Option<usize>,
        kv_transfer_bandwidth: Option<f64>,
        reasoning: Option<ReasoningConfig>,
        zmq_kv_events_port: Option<u16>,
        zmq_replay_port: Option<u16>,
        preemption_mode: &str,
        router_queue_policy: Option<&str>,
        sglang: Option<SglangArgs>,
    ) -> PyResult<Self> {
        let engine_type = parse_mocker_engine_type(engine_type)?;
        let worker_type = parse_worker_type(worker_type)?;
        let preemption_mode = parse_preemption_mode(preemption_mode)?;
        let router_queue_policy = router_queue_policy
            .map(|value| {
                value.parse().map_err(|e: String| {
                    PyException::new_err(format!("invalid router_queue_policy {value:?}: {e}"))
                })
            })
            .transpose()?;

        let mut builder = RsMockEngineArgs::builder()
            .engine_type(engine_type)
            .num_gpu_blocks(num_gpu_blocks)
            .block_size(block_size)
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
            .enable_local_indexer(enable_local_indexer)
            .bootstrap_port(bootstrap_port)
            .kv_bytes_per_token(kv_bytes_per_token)
            .kv_transfer_bandwidth(kv_transfer_bandwidth)
            .reasoning(reasoning.map(|config| config.inner()))
            .zmq_kv_events_port(zmq_kv_events_port)
            .zmq_replay_port(zmq_replay_port)
            .preemption_mode(preemption_mode)
            .router_queue_policy(router_queue_policy)
            .sglang(sglang.map(|config| config.inner()));

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

        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_json(config_json: &str) -> PyResult<Self> {
        RsMockEngineArgs::from_json_str(config_json)
            .map(|inner| Self { inner })
            .map_err(|e| PyException::new_err(format!("Failed to parse MockEngineArgs JSON: {e}")))
    }

    fn dump_json(&self) -> PyResult<String> {
        let worker_type = match self.inner.worker_type {
            RsWorkerType::Aggregated => "aggregated",
            RsWorkerType::Prefill => "prefill",
            RsWorkerType::Decode => "decode",
        };
        let engine_type = match self.inner.engine_type {
            RsMockerEngineType::Vllm => "vllm",
            RsMockerEngineType::Sglang => "sglang",
        };
        let preemption_mode = match self.inner.preemption_mode {
            RsPreemptionMode::Lifo => "lifo",
            RsPreemptionMode::Fifo => "fifo",
        };
        let router_queue_policy = self
            .inner
            .router_queue_policy
            .as_ref()
            .map(|policy| policy.to_string());
        let payload = json!({
            "engine_type": engine_type,
            "num_gpu_blocks": self.inner.num_gpu_blocks,
            "block_size": self.inner.block_size,
            "max_num_seqs": self.inner.max_num_seqs,
            "max_num_batched_tokens": self.inner.max_num_batched_tokens,
            "enable_prefix_caching": self.inner.enable_prefix_caching,
            "enable_chunked_prefill": self.inner.enable_chunked_prefill,
            "speedup_ratio": self.inner.speedup_ratio,
            "decode_speedup_ratio": self.inner.decode_speedup_ratio,
            "dp_size": self.inner.dp_size,
            "startup_time": self.inner.startup_time,
            "worker_type": worker_type,
            "planner_profile_data": self.inner.planner_profile_data,
            "aic_backend": self.inner.aic_backend,
            "aic_system": self.inner.aic_system,
            "aic_backend_version": self.inner.aic_backend_version,
            "aic_tp_size": self.inner.aic_tp_size,
            "aic_model_path": self.inner.aic_model_path,
            "aic_moe_tp_size": self.inner.aic_moe_tp_size,
            "aic_moe_ep_size": self.inner.aic_moe_ep_size,
            "aic_attention_dp_size": self.inner.aic_attention_dp_size,
            "enable_local_indexer": self.inner.enable_local_indexer,
            "bootstrap_port": self.inner.bootstrap_port,
            "kv_bytes_per_token": self.inner.kv_bytes_per_token,
            "kv_transfer_bandwidth": self.inner.kv_transfer_bandwidth,
            "reasoning": self.inner.reasoning,
            "zmq_kv_events_port": self.inner.zmq_kv_events_port,
            "zmq_replay_port": self.inner.zmq_replay_port,
            "preemption_mode": preemption_mode,
            "router_queue_policy": router_queue_policy,
            "sglang": self.inner.sglang,
        });
        serde_json::to_string_pretty(&payload)
            .map_err(|e| PyException::new_err(format!("Failed to serialize MockEngineArgs: {e}")))
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
    }

    fn is_prefill(&self) -> bool {
        self.inner.is_prefill()
    }

    fn is_decode(&self) -> bool {
        self.inner.is_decode()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (bootstrap_port=None, zmq_kv_events_port=None, zmq_replay_port=None, kv_bytes_per_token=None, num_gpu_blocks=None, aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, aic_moe_tp_size=None, aic_moe_ep_size=None, aic_attention_dp_size=None, enable_prefix_caching=None, worker_type=None))]
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
        enable_prefix_caching: Option<bool>,
        worker_type: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = self.inner.clone();
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
        if let Some(enable_prefix_caching) = enable_prefix_caching {
            inner.enable_prefix_caching = enable_prefix_caching;
        }
        if let Some(worker_type) = worker_type {
            inner.worker_type = parse_worker_type(&worker_type)?;
        }
        inner.normalized().map(|inner| Self { inner }).map_err(|e| {
            PyException::new_err(format!("Failed to normalize MockEngineArgs overrides: {e}"))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (trace_file, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, trace_block_size=512))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_trace_replay(
    py: Python<'_>,
    trace_file: PathBuf,
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
    trace_block_size: usize,
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
    let prefill_load_estimator = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config);
    let replay_mode = replay_mode.to_owned();
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;

        match args_selection {
            ReplayArgsSelection::Aggregated(args) => {
                match (replay_mode.as_str(), replay_concurrency) {
                    ("offline", Some(max_in_flight)) => {
                        dynamo_mocker::replay::simulate_concurrency_file_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            &trace_file,
                            trace_block_size,
                            max_in_flight,
                            num_workers,
                            router_mode,
                        )
                    }
                    ("offline", None) => {
                        dynamo_mocker::replay::simulate_trace_file_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            &trace_file,
                            trace_block_size,
                            num_workers,
                            arrival_speedup_ratio,
                            router_mode,
                        )
                    }
                    ("online", Some(max_in_flight)) => {
                        dynamo_mocker::replay::simulate_concurrency_live_file_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            &trace_file,
                            trace_block_size,
                            max_in_flight,
                            num_workers,
                            router_mode,
                        )
                    }
                    ("online", None) => {
                        dynamo_mocker::replay::simulate_trace_live_file_with_router_mode(
                            *args,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            &trace_file,
                            trace_block_size,
                            num_workers,
                            arrival_speedup_ratio,
                            router_mode,
                        )
                    }
                    (other, _) => anyhow::bail!(
                        "replay_mode must be either 'offline' or 'online', got '{}'",
                        other
                    ),
                }
            }
            ReplayArgsSelection::Disagg(config) => match (replay_mode.as_str(), replay_concurrency)
            {
                ("offline", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_file_disagg_with_router_mode(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        &trace_file,
                        trace_block_size,
                        max_in_flight,
                        router_mode,
                    )
                }
                ("offline", None) => {
                    dynamo_mocker::replay::simulate_trace_file_disagg_with_router_mode(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        &trace_file,
                        trace_block_size,
                        arrival_speedup_ratio,
                        router_mode,
                    )
                }
                ("online", _) => anyhow::bail!("disagg replay only supports replay_mode='offline'"),
                (other, _) => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
            },
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
}

#[pyfunction]
#[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args=None, prefill_engine_args=None, decode_engine_args=None, router_config=None, aic_perf_config=None, num_workers=1, num_prefill_workers=1, num_decode_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, arrival_interval_ms=1.0, turns_per_session=1, shared_prefix_ratio=0.0, num_prefix_groups=0, inter_turn_delay_ms=0.0))]
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
    arrival_interval_ms: f64,
    turns_per_session: usize,
    shared_prefix_ratio: f64,
    num_prefix_groups: usize,
    inter_turn_delay_ms: f64,
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
    let prefill_load_estimator = load_replay_prefill_load_estimator(
        py,
        router_mode,
        router_config.as_ref(),
        aic_perf_config,
    )?;
    let router_config = load_replay_router_config(router_config);
    let replay_mode = replay_mode.to_owned();
    let block_size = match &args_selection {
        ReplayArgsSelection::Aggregated(args) => args.block_size.max(1),
        ReplayArgsSelection::Disagg(config) => config.prefill_args.block_size.max(1),
    };
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;
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
                arrival_interval_ms,
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
                    match (replay_mode.as_str(), replay_concurrency) {
                        ("offline", Some(max_in_flight)) => dynamo_mocker::replay::simulate_concurrency_workload_disagg_with_router_mode(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            max_in_flight,
                            router_mode,
                        ),
                        ("offline", None) => dynamo_mocker::replay::simulate_trace_workload_disagg_with_router_mode(
                            *config,
                            router_config.clone(),
                            prefill_load_estimator.clone(),
                            trace,
                            router_mode,
                        ),
                        ("online", _) => anyhow::bail!(
                            "disagg replay only supports replay_mode='offline'"
                        ),
                        (other, _) => anyhow::bail!(
                            "replay_mode must be either 'offline' or 'online', got '{}'",
                            other
                        ),
                    }
                }
            };
        }

        let requests = build_synthetic_requests(
            input_tokens,
            output_tokens,
            request_count,
            arrival_interval_ms,
            replay_concurrency.is_none(),
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
            ReplayArgsSelection::Disagg(config) => match (replay_mode.as_str(), replay_concurrency)
            {
                ("offline", Some(max_in_flight)) => {
                    dynamo_mocker::replay::simulate_concurrency_requests_disagg_with_router_mode(
                        *config,
                        router_config.clone(),
                        prefill_load_estimator.clone(),
                        requests,
                        max_in_flight,
                        router_mode,
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
                    )
                }
                ("online", _) => anyhow::bail!("disagg replay only supports replay_mode='offline'"),
                (other, _) => anyhow::bail!(
                    "replay_mode must be either 'offline' or 'online', got '{}'",
                    other
                ),
            },
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
        .map(|extra_args| materialize_replay_mocker_args(py, extra_args.inner()))
        .transpose()
}

fn materialize_replay_mocker_args(
    py: Python<'_>,
    mut args: RsMockEngineArgs,
) -> PyResult<RsMockEngineArgs> {
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
        args.perf_model = Arc::new(PerfModel::from_aic_callback(callback));
    }

    Ok(args)
}

fn load_replay_router_config(
    router_config: Option<KvRouterConfig>,
) -> Option<dynamo_kv_router::config::KvRouterConfig> {
    router_config.map(|config| config.inner())
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

fn parse_replay_concurrency(replay_concurrency: Option<isize>) -> anyhow::Result<Option<usize>> {
    match replay_concurrency {
        Some(value) if value < 1 => anyhow::bail!("replay_concurrency must be at least 1"),
        Some(value) => Ok(Some(value as usize)),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_workload(
    block_size: usize,
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    arrival_interval_ms: f64,
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
    if !arrival_interval_ms.is_finite() || arrival_interval_ms < 0.0 {
        anyhow::bail!("arrival_interval_ms must be a finite non-negative number");
    }
    if !inter_turn_delay_ms.is_finite() || inter_turn_delay_ms < 0.0 {
        anyhow::bail!("inter_turn_delay_ms must be a finite non-negative number");
    }

    let first_turn_arrivals = if arrival_interval_ms == 0.0 {
        ArrivalSpec::Burst
    } else {
        ArrivalSpec::ConstantQps {
            qps: 1000.0 / arrival_interval_ms,
        }
    };

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
    })
}

fn build_synthetic_requests(
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    arrival_interval_ms: f64,
    include_arrival_timestamps: bool,
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
    if !arrival_interval_ms.is_finite() || arrival_interval_ms < 0.0 {
        anyhow::bail!(
            "arrival_interval_ms must be a finite non-negative number, got {}",
            arrival_interval_ms
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
            arrival_timestamp_ms: include_arrival_timestamps
                .then_some(request_idx as f64 * arrival_interval_ms),
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

/// Step-based bridge for driving an offline replay with a Python planner.
///
/// Supports both aggregated and disaggregated topologies. The Python adapter
/// calls `advance_to()` to run the simulation forward, collects FPM/traffic
/// metrics, feeds them to the planner state machine, then calls
/// `apply_scaling()` to resize worker pools.
#[pyclass(unsendable)]
pub struct PlannerReplayBridge {
    handle: Option<dynamo_mocker::replay::PlannerReplayHandle>,
}

#[pymethods]
impl PlannerReplayBridge {
    /// Create a bridge for an aggregated Mooncake-style JSONL trace replay.
    #[new]
    #[pyo3(signature = (trace_file, extra_engine_args, num_workers, router_mode="round_robin", router_config=None, arrival_speedup_ratio=1.0, trace_block_size=512))]
    fn new(
        trace_file: PathBuf,
        extra_engine_args: &MockEngineArgs,
        num_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        arrival_speedup_ratio: f64,
        trace_block_size: usize,
    ) -> PyResult<Self> {
        let args = extra_engine_args.inner();
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config);

        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace_file(
            args,
            router_config,
            None,
            &trace_file,
            trace_block_size,
            num_workers,
            arrival_speedup_ratio,
            router_mode,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Create a bridge for a disaggregated Mooncake-style JSONL trace replay.
    #[staticmethod]
    #[pyo3(signature = (trace_file, prefill_engine_args, decode_engine_args, num_prefill_workers, num_decode_workers, router_mode="round_robin", router_config=None, arrival_speedup_ratio=1.0, trace_block_size=512))]
    #[allow(clippy::too_many_arguments)]
    fn create_disagg(
        trace_file: PathBuf,
        prefill_engine_args: &MockEngineArgs,
        decode_engine_args: &MockEngineArgs,
        num_prefill_workers: usize,
        num_decode_workers: usize,
        router_mode: &str,
        router_config: Option<KvRouterConfig>,
        arrival_speedup_ratio: f64,
        trace_block_size: usize,
    ) -> PyResult<Self> {
        let config = dynamo_mocker::replay::OfflineDisaggReplayConfig {
            prefill_args: prefill_engine_args.inner(),
            decode_args: decode_engine_args.inner(),
            num_prefill_workers,
            num_decode_workers,
        };
        let router_mode = parse_replay_router_mode(router_mode)?;
        let router_config = load_replay_router_config(router_config);

        let handle = dynamo_mocker::replay::PlannerReplayHandle::from_trace_file_disagg(
            config,
            router_config,
            None,
            &trace_file,
            trace_block_size,
            arrival_speedup_ratio,
            router_mode,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Advance the simulation to `until_ms` simulated time.
    ///
    /// Returns a dict with separate prefill/decode worker counts and FPM snapshots.
    /// Traffic metrics are NOT included — call `drain_traffic()` explicitly on
    /// throughput-scaling ticks only.
    fn advance_to(&mut self, py: Python<'_>, until_ms: f64) -> PyResult<PyObject> {
        let handle = self
            .handle
            .as_mut()
            .ok_or_else(|| PyException::new_err("bridge has been finalized"))?;

        let tick_data = handle.advance_to(until_ms).map_err(to_pyerr)?;

        let result = json!({
            "now_ms": tick_data.now_ms,
            "is_done": tick_data.is_done,
            "prefill_fpm_snapshots": fpm_snapshots_to_json(tick_data.prefill_fpm_snapshots),
            "decode_fpm_snapshots": fpm_snapshots_to_json(tick_data.decode_fpm_snapshots),
            "active_prefill_count": tick_data.active_prefill_count,
            "active_decode_count": tick_data.active_decode_count,
            "total_prefill_count": tick_data.total_prefill_count,
            "total_decode_count": tick_data.total_decode_count,
        });

        pythonize(py, &result)
            .map_err(to_pyerr)
            .map(|obj| obj.unbind())
    }

    /// Drain accumulated traffic metrics since the last drain.
    ///
    /// Returns a dict with:
    ///   - `duration_s`   (f64): window length in seconds
    ///   - `num_req`      (usize): completed requests in the window
    ///   - `avg_isl`      (f64): mean input sequence length (tokens)
    ///   - `avg_osl`      (f64): mean output sequence length (tokens)
    ///   - `avg_ttft_ms`  (f64): mean time-to-first-token in milliseconds,
    ///                          averaged only over requests that reported
    ///                          a TTFT sample (0.0 when no samples)
    ///   - `avg_itl_ms`   (f64): mean inter-token latency in milliseconds,
    ///                          averaged only over requests that generated
    ///                          at least one token gap (0.0 when no samples)
    ///
    /// Call this only on throughput-scaling ticks so the observation window
    /// covers the full `throughput_adjustment_interval`.
    fn drain_traffic(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let handle = self
            .handle
            .as_mut()
            .ok_or_else(|| PyException::new_err("bridge has been finalized"))?;

        let stats = handle.drain_traffic();

        let result = json!({
            "duration_s": stats.duration_s,
            "num_req": stats.num_req,
            "avg_isl": stats.avg_isl,
            "avg_osl": stats.avg_osl,
            "avg_ttft_ms": stats.avg_ttft_ms,
            "avg_itl_ms": stats.avg_itl_ms,
        });

        pythonize(py, &result)
            .map_err(to_pyerr)
            .map(|obj| obj.unbind())
    }

    /// Apply a scaling decision with separate prefill and decode targets.
    /// For agg mode, `target_prefill` is ignored (pass 0).
    fn apply_scaling(&mut self, target_prefill: usize, target_decode: usize) -> PyResult<()> {
        let handle = self
            .handle
            .as_mut()
            .ok_or_else(|| PyException::new_err("bridge has been finalized"))?;
        handle
            .apply_scaling(target_prefill, target_decode)
            .map_err(to_pyerr)
    }

    /// Finalize the replay and return the trace simulation report.
    fn finalize(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let handle = self
            .handle
            .take()
            .ok_or_else(|| PyException::new_err("bridge has already been finalized"))?;
        let report = handle.finalize();
        pythonize(py, &report)
            .map_err(to_pyerr)
            .map(|obj| obj.unbind())
    }
}
