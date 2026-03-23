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
use pyo3::{exceptions::PyException, prelude::*};
use pythonize::pythonize;
use uuid::Uuid;

use super::aic_callback::create_aic_callback;
use super::entrypoint::{KvRouterConfig, to_pyerr};

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
    #[pyo3(signature = (engine_type="vllm", num_gpu_blocks=16384, block_size=0, max_num_seqs=Some(256), max_num_batched_tokens=Some(8192), enable_prefix_caching=true, enable_chunked_prefill=true, speedup_ratio=1.0, decode_speedup_ratio=1.0, dp_size=1, startup_time=None, worker_type="aggregated", aic_backend=None, aic_system=None, aic_backend_version=None, aic_tp_size=None, aic_model_path=None, enable_local_indexer=false, bootstrap_port=None, kv_bytes_per_token=None, kv_transfer_bandwidth=None, reasoning=None, zmq_kv_events_port=None, zmq_replay_port=None, preemption_mode="lifo", router_queue_policy=None, sglang=None))]
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
        aic_backend: Option<String>,
        aic_system: Option<String>,
        aic_backend_version: Option<String>,
        aic_tp_size: Option<usize>,
        aic_model_path: Option<String>,
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

        let inner = RsMockEngineArgs::builder()
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
            .aic_backend(aic_backend)
            .aic_system(aic_system)
            .aic_backend_version(aic_backend_version)
            .aic_tp_size(aic_tp_size)
            .aic_model_path(aic_model_path)
            .enable_local_indexer(enable_local_indexer)
            .bootstrap_port(bootstrap_port)
            .kv_bytes_per_token(kv_bytes_per_token)
            .kv_transfer_bandwidth(kv_transfer_bandwidth)
            .reasoning(reasoning.map(|config| config.inner()))
            .zmq_kv_events_port(zmq_kv_events_port)
            .zmq_replay_port(zmq_replay_port)
            .preemption_mode(preemption_mode)
            .router_queue_policy(router_queue_policy)
            .sglang(sglang.map(|config| config.inner()))
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

    fn is_prefill(&self) -> bool {
        self.inner.is_prefill()
    }

    fn is_decode(&self) -> bool {
        self.inner.is_decode()
    }

    #[pyo3(signature = (bootstrap_port=None, zmq_kv_events_port=None, zmq_replay_port=None, kv_bytes_per_token=None))]
    fn with_overrides(
        &self,
        bootstrap_port: Option<u16>,
        zmq_kv_events_port: Option<u16>,
        zmq_replay_port: Option<u16>,
        kv_bytes_per_token: Option<usize>,
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
        inner.normalized().map(|inner| Self { inner }).map_err(|e| {
            PyException::new_err(format!("Failed to normalize MockEngineArgs overrides: {e}"))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (trace_file, extra_engine_args=None, router_config=None, num_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_trace_replay(
    py: Python<'_>,
    trace_file: PathBuf,
    extra_engine_args: Option<MockEngineArgs>,
    router_config: Option<KvRouterConfig>,
    num_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
) -> PyResult<PyObject> {
    let args = load_replay_mocker_args(py, extra_engine_args)?;
    let router_config = load_replay_router_config(router_config);
    let replay_mode = replay_mode.to_owned();
    let router_mode = parse_replay_router_mode(router_mode)?;
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;

        match (replay_mode.as_str(), replay_concurrency) {
            ("offline", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_file_with_router_mode(
                    args,
                    router_config.clone(),
                    &trace_file,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("offline", None) => dynamo_mocker::replay::simulate_trace_file_with_router_mode(
                args,
                router_config.clone(),
                &trace_file,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            ("online", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_live_file_with_router_mode(
                    args,
                    router_config.clone(),
                    &trace_file,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("online", None) => dynamo_mocker::replay::simulate_trace_live_file_with_router_mode(
                args,
                router_config.clone(),
                &trace_file,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            (other, _) => anyhow::bail!(
                "replay_mode must be either 'offline' or 'online', got '{}'",
                other
            ),
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
}

#[pyfunction]
#[pyo3(signature = (input_tokens, output_tokens, request_count, extra_engine_args=None, router_config=None, num_workers=1, replay_concurrency=None, replay_mode="offline", router_mode="round_robin", arrival_speedup_ratio=1.0, arrival_interval_ms=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn run_mocker_synthetic_trace_replay(
    py: Python<'_>,
    input_tokens: usize,
    output_tokens: usize,
    request_count: usize,
    extra_engine_args: Option<MockEngineArgs>,
    router_config: Option<KvRouterConfig>,
    num_workers: usize,
    replay_concurrency: Option<isize>,
    replay_mode: &str,
    router_mode: &str,
    arrival_speedup_ratio: f64,
    arrival_interval_ms: f64,
) -> PyResult<PyObject> {
    let args = load_replay_mocker_args(py, extra_engine_args)?;
    let router_config = load_replay_router_config(router_config);
    let replay_mode = replay_mode.to_owned();
    let router_mode = parse_replay_router_mode(router_mode)?;
    let report = py.allow_threads(move || {
        let replay_concurrency = parse_replay_concurrency(replay_concurrency)?;
        let requests = build_synthetic_requests(
            input_tokens,
            output_tokens,
            request_count,
            arrival_interval_ms,
            replay_concurrency.is_none(),
        )?;

        match (replay_mode.as_str(), replay_concurrency) {
            ("offline", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_requests_with_router_mode(
                    args,
                    router_config.clone(),
                    requests,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("offline", None) => dynamo_mocker::replay::simulate_trace_requests_with_router_mode(
                args,
                router_config.clone(),
                requests,
                num_workers,
                arrival_speedup_ratio,
                router_mode,
            ),
            ("online", Some(max_in_flight)) => {
                dynamo_mocker::replay::simulate_concurrency_live_requests_with_router_mode(
                    args,
                    router_config.clone(),
                    requests,
                    max_in_flight,
                    num_workers,
                    router_mode,
                )
            }
            ("online", None) => {
                dynamo_mocker::replay::simulate_trace_live_requests_with_router_mode(
                    args,
                    router_config.clone(),
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
        }
    });
    let report = report.map_err(to_pyerr)?;
    pythonize(py, &report)
        .map_err(to_pyerr)
        .map(|obj| obj.unbind())
}

fn load_replay_mocker_args(
    py: Python<'_>,
    extra_engine_args: Option<MockEngineArgs>,
) -> PyResult<RsMockEngineArgs> {
    let mut args = match extra_engine_args {
        Some(extra_args) => extra_args.inner(),
        None => RsMockEngineArgs::default(),
    };

    if let Some(ref backend_name) = args.aic_backend.clone() {
        let backend = backend_name.clone();
        let system = args.aic_system.as_deref().unwrap_or("h200_sxm").to_string();
        let model_name = args
            .aic_model_path
            .clone()
            .ok_or_else(|| PyException::new_err("--aic-perf-model requires --model-path"))?;
        let backend_version = args.aic_backend_version.clone();
        let tp_size = args.aic_tp_size.unwrap_or(1);
        let callback = create_aic_callback(
            py,
            &backend,
            &system,
            &model_name,
            tp_size,
            backend_version.as_deref(),
        )
        .map_err(|e| {
            PyException::new_err(format!(
                "Failed to create AIC callback (--aic-perf-model was requested): {}",
                e
            ))
        })?;
        tracing::info!(
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
