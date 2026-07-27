// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result;

#[cfg(test)]
use super::agg::AggRuntimeStats;
use super::agg::{ReplayMode as AggReplayMode, RoundRobinAggRuntime};
#[cfg(test)]
use super::disagg::DisaggRuntimeStats;
use super::disagg::{ReplayMode as DisaggReplayMode, RoundRobinDisaggRuntime};
use super::extensions::kv_events::{self, HandoffDisaggRuntime};
use super::extensions::kv_router::{AggRuntime, DisaggRuntime, ReplayKvRouterConfig};
use super::normalize_trace_requests;
use super::single::{SingleReplayMode, SingleRuntime};
use crate::common::handoff::NormalizedHandoffConformance;
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs, WorkerType};
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::{
    ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayWorkerArtifacts, SlaThresholds,
    TraceCollector, TraceSimulationReport,
};
use crate::scheduler::RouterEventVisibility;

fn finish_with_replay_wall_time(
    collector: TraceCollector,
    started_at: Instant,
    sla: SlaThresholds,
) -> TraceSimulationReport {
    // Capture elapsed time before final report aggregation so bookkeeping such
    // as latency sorting is not counted as replay execution.
    let wall_time_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let mut collector = collector;
    collector.set_sla_thresholds(sla);
    collector.finish().with_wall_time_ms(wall_time_ms)
}

fn use_single_runtime(num_workers: usize, dp_size: u32, router_mode: ReplayRouterMode) -> bool {
    // dp_size>1 needs one scheduler/KV pool per rank. They still share one
    // discrete-event runtime, but require the rank-aware AggRuntime path.
    num_workers == 1 && dp_size <= 1 && router_mode != ReplayRouterMode::KvRouter
}

fn trace_workload_driver(
    trace: Trace,
    engine_block_size: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
) -> Result<WorkloadDriver> {
    match router_mode {
        ReplayRouterMode::RoundRobin => WorkloadDriver::new_trace_without_replay_hashes(
            trace,
            engine_block_size,
            accumulate_session_deltas,
        ),
        ReplayRouterMode::KvRouter if accumulate_session_deltas => {
            trace.into_delta_accumulating_trace_driver_with_block_size(engine_block_size)
        }
        ReplayRouterMode::KvRouter => trace.into_trace_driver_with_block_size(engine_block_size),
    }
}

fn concurrency_workload_driver(
    trace: Trace,
    engine_block_size: usize,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
) -> Result<WorkloadDriver> {
    match router_mode {
        ReplayRouterMode::RoundRobin => WorkloadDriver::new_concurrency_without_replay_hashes(
            trace,
            engine_block_size,
            max_in_flight,
            accumulate_session_deltas,
        ),
        ReplayRouterMode::KvRouter if accumulate_session_deltas => trace
            .into_delta_accumulating_concurrency_driver_with_block_size(
                engine_block_size,
                max_in_flight,
            ),
        ReplayRouterMode::KvRouter => {
            trace.into_concurrency_driver_with_block_size(engine_block_size, max_in_flight)
        }
    }
}

fn agentic_workload_driver(
    trace: AgenticTrace,
    engine_block_size: usize,
    router_mode: ReplayRouterMode,
) -> Result<WorkloadDriver> {
    match router_mode {
        ReplayRouterMode::RoundRobin => {
            WorkloadDriver::new_agentic_trace_without_replay_hashes(trace, engine_block_size)
        }
        ReplayRouterMode::KvRouter => trace.into_trace_driver_with_block_size(engine_block_size),
    }
}

/// Run the deterministic offline half of the live/offline handoff conformance
/// fixture. This is public only for cross-crate conformance tests.
#[doc(hidden)]
pub fn run_offline_handoff_conformance(
    engine_type: EngineType,
    transfer_timing_mode: crate::common::protocols::KvTransferTimingMode,
) -> Result<NormalizedHandoffConformance> {
    if engine_type == EngineType::Trtllm {
        anyhow::bail!("TRT-LLM does not support destination handoff");
    }

    let build_args = |worker_type| {
        let mut builder = MockEngineArgs::builder()
            .engine_type(engine_type)
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(2))
            .worker_type(worker_type)
            .speedup_ratio(1000.0)
            .decode_speedup_ratio(1000.0)
            .kv_transfer_bandwidth(Some(1.0))
            .kv_bytes_per_token(Some(1_000_000))
            .kv_transfer_timing_mode(transfer_timing_mode);
        if engine_type == EngineType::Sglang {
            builder = builder.sglang(Some(SglangArgs {
                page_size: Some(4),
                ..Default::default()
            }));
        }
        builder.build()
    };
    let config = OfflineDisaggReplayConfig {
        prefill_args: build_args(WorkerType::Prefill)?,
        decode_args: build_args(WorkerType::Decode)?,
        num_prefill_workers: 1,
        num_decode_workers: 1,
    }
    .normalized()?;
    let request = DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: 2,
        uuid: Some(uuid::Uuid::from_u128(1)),
        arrival_timestamp_ms: Some(0.0),
        ..Default::default()
    };

    HandoffDisaggRuntime::new_handoff_conformance(&config, VecDeque::from([request]))?
        .run_handoff_conformance(engine_type)
}

pub(crate) fn generate_trace_worker_artifacts(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<ReplayWorkerArtifacts> {
    generate_trace_worker_artifacts_with_visibility(args, trace, None)
}

pub(crate) fn generate_trace_worker_artifacts_with_visibility(
    args: MockEngineArgs,
    trace: Trace,
    router_event_visibility_override: Option<RouterEventVisibility>,
) -> Result<ReplayWorkerArtifacts> {
    kv_events::generate_trace_worker_artifacts_with_visibility(
        args,
        trace,
        router_event_visibility_override,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, args.dp_size, router_mode) {
        simulate_trace_single(
            args,
            requests,
            arrival_speedup_ratio,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        simulate_trace_multi(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            arrival_speedup_ratio,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, args.dp_size, router_mode) {
        simulate_concurrency_single(
            args,
            requests,
            max_in_flight,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        simulate_concurrency_multi(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            max_in_flight,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_delta_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        false,
        true,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_without_session_metadata(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_delta_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        false,
        false,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

pub(crate) fn simulate_agentic_trace_workload(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, args.dp_size, router_mode) {
        simulate_agentic_trace_workload_single(args, trace, sla)
    } else {
        simulate_agentic_trace_workload_multi(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_accumulating_deltas(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_delta_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        true,
        true,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_trace_workload_with_delta_mode(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, args.dp_size, router_mode) {
        simulate_trace_workload_single(
            args,
            trace,
            accumulate_session_deltas,
            emit_session_metadata,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        simulate_trace_workload_multi(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            accumulate_session_deltas,
            emit_session_metadata,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_delta_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        false,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_accumulating_deltas(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_delta_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        true,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_concurrency_workload_with_delta_mode(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, args.dp_size, router_mode) {
        simulate_concurrency_workload_single(
            args,
            trace,
            max_in_flight,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        simulate_concurrency_workload_multi(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => {
            RoundRobinDisaggRuntime::new_round_robin(&config, pending, DisaggReplayMode::Trace)?
                .with_per_request_records(record_per_request)
                .with_max_sim_time_ms(max_sim_time_ms)
                .run()?
        }
        ReplayRouterMode::KvRouter => DisaggRuntime::new(
            &config,
            router_config,
            prefill_load_estimator,
            pending,
            DisaggReplayMode::Trace,
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let pending = VecDeque::from(requests);
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinDisaggRuntime::new_round_robin(
            &config,
            pending,
            DisaggReplayMode::Concurrency { max_in_flight },
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => DisaggRuntime::new(
            &config,
            router_config,
            prefill_load_estimator,
            pending,
            DisaggReplayMode::Concurrency { max_in_flight },
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_disagg_with_session_metadata(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        true,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg_without_session_metadata(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_disagg_with_session_metadata(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        false,
        record_per_request,
        max_sim_time_ms,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_trace_workload_disagg_with_session_metadata(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let mut driver =
        trace_workload_driver(trace, config.prefill_args.block_size, router_mode, false)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinDisaggRuntime::new_round_robin_workload(
            &config,
            driver,
            DisaggReplayMode::Trace,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => DisaggRuntime::new_workload(
            &config,
            router_config,
            prefill_load_estimator,
            driver,
            DisaggReplayMode::Trace,
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let driver = concurrency_workload_driver(
        trace,
        config.prefill_args.block_size,
        max_in_flight,
        router_mode,
        false,
    )?;
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinDisaggRuntime::new_round_robin_workload(
            &config,
            driver,
            DisaggReplayMode::Concurrency { max_in_flight },
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => DisaggRuntime::new_workload(
            &config,
            router_config,
            prefill_load_estimator,
            driver,
            DisaggReplayMode::Concurrency { max_in_flight },
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_trace_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let collector = SingleRuntime::new(args, pending, SingleReplayMode::Trace)
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_concurrency_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let collector = SingleRuntime::new(
        args,
        pending,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_trace_workload_single(
    args: MockEngineArgs,
    trace: Trace,
    accumulate_session_deltas: bool,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let mut driver = WorkloadDriver::new_trace_without_replay_hashes(
        trace,
        engine_block_size,
        accumulate_session_deltas,
    )?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    let collector = SingleRuntime::new_workload(args, driver, SingleReplayMode::Trace)
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_agentic_trace_workload_single(
    args: MockEngineArgs,
    trace: AgenticTrace,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let driver = WorkloadDriver::new_agentic_trace_without_replay_hashes(trace, engine_block_size)?;
    let collector = SingleRuntime::new_workload(args, driver, SingleReplayMode::Trace).run()?;
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_concurrency_workload_single(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let driver = WorkloadDriver::new_concurrency_without_replay_hashes(
        trace,
        engine_block_size,
        max_in_flight,
        accumulate_session_deltas,
    )?;
    let collector = SingleRuntime::new_workload(
        args,
        driver,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_multi(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinAggRuntime::new_round_robin(
            &args,
            pending,
            num_workers,
            AggReplayMode::Trace,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => AggRuntime::new(
            &args,
            router_config,
            prefill_load_estimator,
            pending,
            num_workers,
            AggReplayMode::Trace,
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_multi(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinAggRuntime::new_round_robin(
            &args,
            pending,
            num_workers,
            AggReplayMode::Concurrency { max_in_flight },
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => AggRuntime::new(
            &args,
            router_config,
            prefill_load_estimator,
            pending,
            num_workers,
            AggReplayMode::Concurrency { max_in_flight },
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_multi(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let mut driver = trace_workload_driver(
        trace,
        args.block_size,
        router_mode,
        accumulate_session_deltas,
    )?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinAggRuntime::new_round_robin_workload(
            &args,
            driver,
            num_workers,
            AggReplayMode::Trace,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => AggRuntime::new_workload(
            &args,
            router_config,
            prefill_load_estimator,
            driver,
            num_workers,
            AggReplayMode::Trace,
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

pub(crate) fn simulate_agentic_trace_workload_multi(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let driver = agentic_workload_driver(trace, args.block_size, router_mode)?;
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinAggRuntime::new_round_robin_workload(
            &args,
            driver,
            num_workers,
            AggReplayMode::Trace,
        )?
        .run()?,
        ReplayRouterMode::KvRouter => AggRuntime::new_workload(
            &args,
            router_config,
            prefill_load_estimator,
            driver,
            num_workers,
            AggReplayMode::Trace,
            router_mode,
        )?
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_multi(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let driver = concurrency_workload_driver(
        trace,
        args.block_size,
        max_in_flight,
        router_mode,
        accumulate_session_deltas,
    )?;
    let (collector, _) = match router_mode {
        ReplayRouterMode::RoundRobin => RoundRobinAggRuntime::new_round_robin_workload(
            &args,
            driver,
            num_workers,
            AggReplayMode::Concurrency { max_in_flight },
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
        ReplayRouterMode::KvRouter => AggRuntime::new_workload(
            &args,
            router_config,
            prefill_load_estimator,
            driver,
            num_workers,
            AggReplayMode::Concurrency { max_in_flight },
            router_mode,
        )?
        .with_per_request_records(record_per_request)
        .with_max_sim_time_ms(max_sim_time_ms)
        .run()?,
    };
    Ok(finish_with_replay_wall_time(collector, started_at, sla))
}

#[cfg(test)]
pub(super) fn run_trace_single_collect(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> TraceCollector {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio).unwrap();
    SingleRuntime::new(args, pending, SingleReplayMode::Trace)
        .run()
        .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_single_collect(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> TraceCollector {
    SingleRuntime::new(
        args,
        VecDeque::from(requests),
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_single_collect(
    args: MockEngineArgs,
    trace: Trace,
) -> TraceCollector {
    let engine_block_size = args.block_size;
    SingleRuntime::new_workload(
        args,
        trace
            .into_trace_driver_with_block_size(engine_block_size)
            .unwrap(),
        SingleReplayMode::Trace,
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_single_collect(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
) -> TraceCollector {
    let engine_block_size = args.block_size;
    SingleRuntime::new_workload(
        args,
        trace
            .into_concurrency_driver_with_block_size(engine_block_size, max_in_flight)
            .unwrap(),
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_agentic_trace_single_collect(
    args: MockEngineArgs,
    trace: AgenticTrace,
) -> TraceCollector {
    let engine_block_size = args.block_size;
    SingleRuntime::new_workload(
        args,
        trace
            .into_trace_driver_with_block_size(engine_block_size)
            .unwrap(),
        SingleReplayMode::Trace,
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    let pending = normalize_trace_requests(requests, 1.0).unwrap();
    AggRuntime::new(
        args,
        None,
        None,
        pending,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new(
        args,
        None,
        None,
        VecDeque::from(requests),
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_multi_collect_with_stats(
    args: &MockEngineArgs,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
) -> (TraceCollector, AggRuntimeStats) {
    let driver = if accumulate_session_deltas {
        trace
            .into_delta_accumulating_trace_driver_with_block_size(args.block_size)
            .unwrap()
    } else {
        trace
            .into_trace_driver_with_block_size(args.block_size)
            .unwrap()
    };
    AggRuntime::new_workload(
        args,
        None,
        None,
        driver,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_multi_collect_with_stats(
    args: &MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new_workload(
        args,
        None,
        None,
        trace
            .into_concurrency_driver_with_block_size(args.block_size, max_in_flight)
            .unwrap(),
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_agentic_trace_multi_collect_with_stats(
    args: &MockEngineArgs,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new_workload(
        args,
        None,
        None,
        trace
            .into_trace_driver_with_block_size(args.block_size)
            .unwrap(),
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_collect(
    config: &OfflineDisaggReplayConfig,
    requests: Vec<DirectRequest>,
    router_config: Option<ReplayKvRouterConfig>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio).unwrap();
    DisaggRuntime::new(
        config,
        router_config,
        None,
        pending,
        DisaggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_collect(
    config: &OfflineDisaggReplayConfig,
    requests: Vec<DirectRequest>,
    router_config: Option<ReplayKvRouterConfig>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new(
        config,
        router_config,
        None,
        VecDeque::from(requests),
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_collect(
    config: &OfflineDisaggReplayConfig,
    trace: Trace,
    router_config: Option<ReplayKvRouterConfig>,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new_workload(
        config,
        router_config,
        None,
        trace
            .into_trace_driver_with_block_size(config.prefill_args.block_size)
            .unwrap(),
        DisaggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_collect(
    config: &OfflineDisaggReplayConfig,
    trace: Trace,
    router_config: Option<ReplayKvRouterConfig>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new_workload(
        config,
        router_config,
        None,
        trace
            .into_concurrency_driver_with_block_size(config.prefill_args.block_size, max_in_flight)
            .unwrap(),
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "kvbm-offload")]
    use super::simulate_trace_disagg;
    use super::{generate_trace_worker_artifacts, simulate_trace, use_single_runtime};
    use crate::common::perf_model::{AicCallback, PerfModel};
    #[cfg(feature = "kvbm-offload")]
    use crate::common::protocols::WorkerType;
    use crate::common::protocols::{DirectRequest, MockEngineArgs};
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};
    #[cfg(feature = "kvbm-offload")]
    use crate::replay::OfflineDisaggReplayConfig;
    use crate::replay::{ReplayRouterMode, SlaThresholds};
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn single_runtime_selection_excludes_kv_router() {
        assert!(use_single_runtime(1, 1, ReplayRouterMode::RoundRobin));
        assert!(!use_single_runtime(1, 1, ReplayRouterMode::KvRouter));
        assert!(!use_single_runtime(2, 1, ReplayRouterMode::RoundRobin));
        assert!(!use_single_runtime(2, 1, ReplayRouterMode::KvRouter));
        // dp_size>1 forces the multi (per-rank) path even with a single worker.
        assert!(!use_single_runtime(1, 8, ReplayRouterMode::RoundRobin));
    }

    struct LengthLatency;

    impl AicCallback for LengthLatency {
        fn predict_prefill(&self, _batch_size: usize, effective_isl: usize, _prefix: usize) -> f64 {
            effective_isl as f64
        }

        fn predict_decode(&self, _batch_size: usize, _isl: usize, _osl: usize) -> f64 {
            1.0
        }
    }

    fn rank_timing_args(dp_size: u32) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(false)
            .speedup_ratio(1.0)
            .dp_size(dp_size)
            .perf_model(Arc::new(PerfModel::from_aic_callback(Arc::new(
                LengthLatency,
            ))))
            .build()
            .unwrap()
    }

    fn timed_request(uuid: u128, input_length: usize) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; input_length],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(uuid)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }
    }

    #[test]
    fn attention_dp_trace_entrypoint_aligns_skewed_ranks_to_slowest_boundary() {
        let report = simulate_trace(
            rank_timing_args(2),
            None,
            None,
            vec![timed_request(30, 4), timed_request(31, 8)],
            1,
            1.0,
            ReplayRouterMode::RoundRobin,
            true,
            None,
            SlaThresholds::default(),
        )
        .unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        let mut records = report.per_request;
        records.sort_by_key(|record| record.input_length);
        assert_eq!(records[0].decode_worker_idx, Some(0));
        assert_eq!(records[1].decode_worker_idx, Some(1));
        assert_eq!(records[0].ttft_ms, Some(9.0));
        assert_eq!(records[1].ttft_ms, Some(9.0));

        let dp1 = simulate_trace(
            rank_timing_args(1),
            None,
            None,
            vec![timed_request(32, 4)],
            1,
            1.0,
            ReplayRouterMode::RoundRobin,
            true,
            None,
            SlaThresholds::default(),
        )
        .unwrap();
        assert_eq!(dp1.per_request[0].ttft_ms, Some(5.0));
    }

    #[cfg(feature = "kvbm-offload")]
    fn offload_args(worker_type: WorkerType) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .worker_type(worker_type)
            .num_g2_blocks(Some(8))
            .kv_bytes_per_token(Some(1))
            .offload_batch_size(Some(1))
            .bandwidth_g1_to_g2_gbps(Some(1.0))
            .bandwidth_g2_to_g1_gbps(Some(1.0))
            .build()
            .unwrap()
    }

    #[cfg(feature = "kvbm-offload")]
    fn offload_lifecycle_requests() -> Vec<DirectRequest> {
        [
            (1_u128, 1_u32, 0.0),
            (2_u128, 2_u32, 100.0),
            (3_u128, 1_u32, 200.0),
        ]
        .into_iter()
        .map(|(uuid, token, arrival_timestamp_ms)| DirectRequest {
            // Two blocks ensure vLLM's required final-block recomputation
            // still leaves one router-visible block of restored G2 reuse.
            tokens: vec![token; 8],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(uuid)),
            arrival_timestamp_ms: Some(arrival_timestamp_ms),
            ..Default::default()
        })
        .collect()
    }

    #[cfg(feature = "kvbm-offload")]
    fn assert_g2_restore(report: &crate::replay::TraceSimulationReport) {
        assert_eq!(report.request_counts.completed_requests, 3);
        let restored = report
            .per_request
            .iter()
            .find(|record| record.uuid == Uuid::from_u128(3).to_string())
            .expect("restored request record must be present");
        assert_eq!(
            restored.reused_input_tokens, 4,
            "third request should restore one reusable block from G2 after final-block recomputation: {restored:?}"
        );
    }

    #[cfg(feature = "kvbm-offload")]
    #[test]
    fn aggregated_replay_forces_g1_to_g2_and_g2_to_g1_for_rr_and_kv() {
        for router_mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
            let report = simulate_trace(
                offload_args(WorkerType::Aggregated),
                None,
                None,
                offload_lifecycle_requests(),
                1,
                1.0,
                router_mode,
                true,
                None,
                SlaThresholds::default(),
            )
            .unwrap();
            assert_g2_restore(&report);
        }
    }

    #[cfg(feature = "kvbm-offload")]
    #[test]
    fn disagg_replay_forces_g1_to_g2_and_g2_to_g1_for_rr_and_kv() {
        for router_mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
            let report = simulate_trace_disagg(
                OfflineDisaggReplayConfig {
                    prefill_args: offload_args(WorkerType::Prefill),
                    decode_args: offload_args(WorkerType::Decode),
                    num_prefill_workers: 1,
                    num_decode_workers: 1,
                },
                None,
                None,
                offload_lifecycle_requests(),
                1.0,
                router_mode,
                true,
                None,
                SlaThresholds::default(),
            )
            .unwrap();
            assert_g2_restore(&report);
        }
    }

    #[test]
    fn test_generate_trace_worker_artifacts_emits_monotonic_event_timestamps() {
        let args = MockEngineArgs::builder()
            .block_size(2)
            .num_gpu_blocks(1024)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .enable_prefix_caching(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let trace = Trace {
            block_size: 2,
            sessions: vec![SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![1, 2],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![3, 4],
                        delay_after_previous_ms: 5.0,
                        ..Default::default()
                    },
                ],
            }],
        };

        let artifacts = generate_trace_worker_artifacts(args, trace).unwrap();

        assert_eq!(artifacts.requests.len(), 2);
        assert!(!artifacts.kv_events.is_empty());
        assert!(
            artifacts
                .kv_events
                .windows(2)
                .all(|events| events[0].timestamp_us <= events[1].timestamp_us)
        );

        let first_uuid = artifacts.requests[0].uuid;
        let first_completion_ms = artifacts
            .output_signals
            .iter()
            .find(|signal| signal.signal.uuid == first_uuid && signal.signal.completed)
            .expect("first request must complete")
            .timestamp_us as f64
            / 1000.0;
        assert!(
            artifacts.requests[1].scheduled_ready_at_ms + 0.1 >= first_completion_ms + 5.0,
            "expected second request to wait for completion plus delay"
        );
    }

    #[test]
    fn test_mtp_artifacts_emit_ordered_same_timestamp_bursts() {
        let args = MockEngineArgs::builder()
            .block_size(2)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .enable_prefix_caching(false)
            .speedup_ratio(1000.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .build()
            .unwrap();
        let trace = Trace {
            block_size: 2,
            sessions: vec![SessionTrace {
                session_id: "mtp-session".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 5,
                    hash_ids: vec![1, 2],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            }],
        };

        let artifacts = generate_trace_worker_artifacts(args, trace).unwrap();
        assert_eq!(artifacts.output_signals.len(), 5);
        assert_eq!(
            artifacts.output_signals[0].timestamp_us,
            artifacts.output_signals[1].timestamp_us
        );
        assert_eq!(
            artifacts.output_signals[1].timestamp_us,
            artifacts.output_signals[2].timestamp_us
        );
        assert!(
            artifacts.output_signals[2].timestamp_us < artifacts.output_signals[3].timestamp_us
        );
        assert_eq!(
            artifacts.output_signals[3].timestamp_us,
            artifacts.output_signals[4].timestamp_us
        );
        assert_eq!(
            artifacts
                .output_signals
                .iter()
                .filter(|output| output.signal.completed)
                .count(),
            1
        );
        assert!(artifacts.output_signals.last().unwrap().signal.completed);
    }
}
