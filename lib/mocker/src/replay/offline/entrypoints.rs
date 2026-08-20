// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo compatibility entrypoints over the packaged AISimulate Replayer.
//!
//! This module lowers Dynamo configuration into Replay-owned contracts. It
//! must never include or compile implementation sources from another crate.

use std::collections::VecDeque;

use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ProviderSpec, ReplayAdapters, ReplayCaptureOptions,
    ReplayEngineConfig, ReplayRuntimeInput, ReplayScalingPolicy, ReplaySpec, ReplayTopology,
    Replayer, WorkerPoolSpec,
};
use anyhow::Result;

use super::extensions::kv_events;
use super::extensions::kv_router::{
    KvReplayComposition, ReplayKvRouterConfig, RoundRobinReplayComposition, provider_spec,
};
use super::normalize_trace_requests;
use crate::common::handoff::NormalizedHandoffConformance;
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs, WorkerType};
use crate::engine_adapter::{aggregated_replay_setup, disaggregated_replay_setup};
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayWorkerArtifacts,
    SlaThresholds, TraceSimulationReport,
};
use crate::scheduler::RouterEventVisibility;

fn startup_delay_ms(args: &MockEngineArgs) -> f64 {
    args.startup_time
        .filter(|seconds| *seconds > 0.0)
        .map_or(0.0, |seconds| seconds * 1_000.0)
}

fn worker_pool(initial_workers: usize, args: &MockEngineArgs) -> WorkerPoolSpec {
    WorkerPoolSpec {
        initial_workers,
        startup_delay_ms: startup_delay_ms(args),
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_spec(
    topology: ReplayTopology,
    engine: ReplayEngineConfig,
    router_mode: ReplayRouterMode,
    scaling_enabled: bool,
    max_in_flight: Option<usize>,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<ReplaySpec> {
    Ok(ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology,
        engine: serde_json::to_value(engine)?,
        adapters: ReplayAdapters {
            placement: match router_mode {
                ReplayRouterMode::RoundRobin => ProviderSpec::round_robin(),
                ReplayRouterMode::KvRouter => provider_spec(),
            },
            scaling: if scaling_enabled {
                ProviderSpec {
                    provider: "dynamo_planner".to_string(),
                    config: serde_json::Value::Null,
                }
            } else {
                ProviderSpec::no_scaling()
            },
        },
        max_sim_time_ms,
        max_in_flight,
        record_per_request,
        sla,
        requests: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_aggregated(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    input: ReplayRuntimeInput,
    num_workers: usize,
    max_in_flight: Option<usize>,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let capture_options = ReplayCaptureOptions {
        capture_per_request: record_per_request,
        capture_lifecycle_evidence: scaling_policy
            .as_deref()
            .is_some_and(ReplayScalingPolicy::capture_lifecycle_evidence),
        ..Default::default()
    };
    run_aggregated_with_capture_options(
        args,
        router_config,
        prefill_load_estimator,
        input,
        num_workers,
        max_in_flight,
        router_mode,
        capture_options,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_aggregated_with_capture_options(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    input: ReplayRuntimeInput,
    num_workers: usize,
    max_in_flight: Option<usize>,
    router_mode: ReplayRouterMode,
    capture_options: ReplayCaptureOptions,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let (engine, factory) = aggregated_replay_setup(&args)?;
    let spec = replay_spec(
        ReplayTopology::Aggregated {
            workers: worker_pool(num_workers, &args),
        },
        engine,
        router_mode,
        scaling_policy.is_some(),
        max_in_flight,
        capture_options.effective_per_request(),
        max_sim_time_ms,
        sla,
    )?;

    match router_mode {
        ReplayRouterMode::RoundRobin => Ok(Replayer::with_composition(
            spec,
            factory,
            RoundRobinReplayComposition::new(scaling_policy),
        )?
        .with_capture_options(capture_options)
        .with_runtime_input(input)
        .run()?),
        ReplayRouterMode::KvRouter => Ok(Replayer::with_composition(
            spec,
            factory,
            KvReplayComposition::aggregated(
                args,
                num_workers,
                router_config,
                prefill_load_estimator,
                scaling_policy,
            ),
        )?
        .with_capture_options(capture_options)
        .with_runtime_input(input)
        .run()?),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_disaggregated(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    input: ReplayRuntimeInput,
    max_in_flight: Option<usize>,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let capture_options = ReplayCaptureOptions {
        capture_per_request: record_per_request,
        capture_lifecycle_evidence: scaling_policy
            .as_deref()
            .is_some_and(ReplayScalingPolicy::capture_lifecycle_evidence),
        ..Default::default()
    };
    run_disaggregated_with_capture_options(
        config,
        router_config,
        prefill_load_estimator,
        input,
        max_in_flight,
        router_mode,
        capture_options,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_disaggregated_with_capture_options(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    input: ReplayRuntimeInput,
    max_in_flight: Option<usize>,
    router_mode: ReplayRouterMode,
    capture_options: ReplayCaptureOptions,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let (engine, factory) = disaggregated_replay_setup(&config.prefill_args, &config.decode_args)?;
    let spec = replay_spec(
        ReplayTopology::Disaggregated {
            prefill: worker_pool(config.num_prefill_workers, &config.prefill_args),
            decode: worker_pool(config.num_decode_workers, &config.decode_args),
            handoff_latency_ms: 0.0,
        },
        engine,
        router_mode,
        scaling_policy.is_some(),
        max_in_flight,
        capture_options.effective_per_request(),
        max_sim_time_ms,
        sla,
    )?;

    match router_mode {
        ReplayRouterMode::RoundRobin => Ok(Replayer::with_composition(
            spec,
            factory,
            RoundRobinReplayComposition::new(scaling_policy),
        )?
        .with_capture_options(capture_options)
        .with_runtime_input(input)
        .run()?),
        ReplayRouterMode::KvRouter => Ok(Replayer::with_composition(
            spec,
            factory,
            KvReplayComposition::disaggregated(
                config.prefill_args,
                config.decode_args,
                config.num_prefill_workers,
                config.num_decode_workers,
                router_config,
                prefill_load_estimator,
                scaling_policy,
            ),
        )?
        .with_capture_options(capture_options)
        .with_runtime_input(input)
        .run()?),
    }
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

/// Run the deterministic offline half of the live/offline handoff conformance
/// fixture through the packaged Replay crate.
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
    let prefill_args = build_args(WorkerType::Prefill)?;
    let decode_args = build_args(WorkerType::Decode)?;
    let (engine, factory) = disaggregated_replay_setup(&prefill_args, &decode_args)?;
    let request = DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: 2,
        output_token_ids: Some(vec![7, 8]),
        uuid: Some(uuid::Uuid::from_u128(1)),
        arrival_timestamp_ms: Some(0.0),
        ..Default::default()
    };
    Ok(aisimulate_core::replay::run_engine_handoff_conformance(engine, factory, request)?.into())
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
    visibility: Option<RouterEventVisibility>,
) -> Result<ReplayWorkerArtifacts> {
    kv_events::generate_trace_worker_artifacts_with_visibility(args, trace, visibility)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_with_scaling_policy(
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
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Requests(pending),
        num_workers,
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_with_scaling_policy(
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
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Requests(VecDeque::from(requests)),
        num_workers,
        Some(max_in_flight),
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_with_scaling_policy(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let mut driver = trace_workload_driver(trace, args.block_size, router_mode, false)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_with_capture_options(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    emit_session_metadata: bool,
    capture_options: ReplayCaptureOptions,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let mut driver = trace_workload_driver(trace, args.block_size, router_mode, false)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    run_aggregated_with_capture_options(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        None,
        router_mode,
        capture_options,
        max_sim_time_ms,
        sla,
        None,
    )
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
    let args = args.normalized()?;
    let driver = trace_workload_driver(trace, args.block_size, router_mode, true)?;
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_with_scaling_policy(
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
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let driver =
        concurrency_workload_driver(trace, args.block_size, max_in_flight, router_mode, false)?;
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        Some(max_in_flight),
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
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
    let args = args.normalized()?;
    let driver =
        concurrency_workload_driver(trace, args.block_size, max_in_flight, router_mode, true)?;
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        Some(max_in_flight),
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_agentic_trace_workload(
    args: MockEngineArgs,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    agentic_lanes: Option<usize>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let driver = trace.into_trace_driver_with_options(
        args.block_size,
        router_mode == ReplayRouterMode::KvRouter,
        agentic_lanes,
    )?;
    run_aggregated(
        args,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        num_workers,
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_agentic_trace_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    agentic_lanes: Option<usize>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let driver = trace.into_trace_driver_with_options(
        config.prefill_args.block_size,
        router_mode == ReplayRouterMode::KvRouter,
        agentic_lanes,
    )?;
    run_disaggregated(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_disagg_with_scaling_policy(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    run_disaggregated(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Requests(pending),
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_disagg_with_scaling_policy(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    run_disaggregated(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Requests(VecDeque::from(requests)),
        Some(max_in_flight),
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg_with_scaling_policy(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let mut driver =
        trace_workload_driver(trace, config.prefill_args.block_size, router_mode, false)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    run_disaggregated(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        None,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg_with_capture_options(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    emit_session_metadata: bool,
    capture_options: ReplayCaptureOptions,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let mut driver =
        trace_workload_driver(trace, config.prefill_args.block_size, router_mode, false)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    run_disaggregated_with_capture_options(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        None,
        router_mode,
        capture_options,
        max_sim_time_ms,
        sla,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_disagg_with_scaling_policy(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let driver = concurrency_workload_driver(
        trace,
        config.prefill_args.block_size,
        max_in_flight,
        router_mode,
        false,
    )?;
    run_disaggregated(
        config,
        router_config,
        prefill_load_estimator,
        ReplayRuntimeInput::Workload(driver),
        Some(max_in_flight),
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
        scaling_policy,
    )
}
