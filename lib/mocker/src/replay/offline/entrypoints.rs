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
    WorkerPoolSpec, run_replay_with_composition,
};
use anyhow::Result;

use super::extensions::kv_events;
use super::extensions::kv_router::{
    KvReplayComposition, ReplayAffinityConfig, ReplayKvRouterConfig, RoundRobinReplayComposition,
    RoutingEvidence, provider_spec,
};
use super::normalize_trace_requests;
use crate::common::handoff::NormalizedHandoffConformance;
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs, WorkerType};
use crate::engine_adapter::{aggregated_replay_setup, disaggregated_replay_setup};
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode,
    ReplayTelemetryOptions, ReplayWorkerArtifacts, SlaThresholds, TraceSimulationReport,
    effective_agentic_lanes,
};
use crate::scheduler::RouterEventVisibility;

/// Lower a serialized offline input to the same executor used by the legacy
/// materialized entrypoints, selecting only placement and observers here.
#[cfg(feature = "python-replay")]
#[allow(clippy::too_many_arguments)]
pub fn run_canonical_replay_json(
    payload: &str,
    router_mode: ReplayRouterMode,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    affinity: Option<ReplayAffinityConfig>,
    capture: ReplayCaptureOptions,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<String> {
    anyhow::ensure!(
        affinity.is_none() || router_mode == ReplayRouterMode::KvRouter,
        "conversation affinity requires KV routing"
    );
    anyhow::ensure!(
        affinity.is_none() || scaling_policy.is_none(),
        "conversation affinity requires static worker pools without a Planner"
    );
    if affinity.is_some() {
        super::extensions::kv_router::validate_affinity_router_config(router_config.as_ref())?;
        anyhow::ensure!(
            prefill_load_estimator.is_none(),
            "conversation affinity does not support a custom or AIS router prefill-load estimator"
        );
    }
    let mut payload: serde_json::Value = serde_json::from_str(payload)?;
    let spec = if payload.get("spec").is_some() {
        payload.get_mut("spec").expect("checked above")
    } else {
        &mut payload
    };
    let placement = spec.pointer_mut("/adapters/placement").ok_or_else(|| {
        anyhow::anyhow!("offline replay requires an explicit placement descriptor")
    })?;
    let native_provider = provider_spec();
    let requested_provider = placement.get("provider").and_then(|v| v.as_str());
    anyhow::ensure!(
        (requested_provider == Some("round_robin")
            || requested_provider == Some(native_provider.provider.as_str()))
            && placement
                .get("config")
                .is_none_or(serde_json::Value::is_null),
        "Dynamo replay cannot replace an unknown placement provider or config"
    );
    anyhow::ensure!(
        router_mode == ReplayRouterMode::KvRouter || requested_provider == Some("round_robin"),
        "native KV placement requires router_mode='kv_router'"
    );
    *placement = serde_json::to_value(match router_mode {
        ReplayRouterMode::RoundRobin => ProviderSpec::round_robin(),
        ReplayRouterMode::KvRouter => native_provider,
    })?;
    let scaling = spec
        .pointer_mut("/adapters/scaling")
        .ok_or_else(|| anyhow::anyhow!("offline replay requires an explicit scaling descriptor"))?;
    anyhow::ensure!(
        matches!(
            scaling.get("provider").and_then(|v| v.as_str()),
            Some("none" | "dynamo_planner")
        ) && scaling.get("config").is_none_or(serde_json::Value::is_null),
        "Dynamo replay cannot replace an unknown scaling provider or config"
    );
    anyhow::ensure!(
        scaling_policy.is_some() || scaling["provider"] == "none",
        "dynamo_planner scaling requires a scaling_policy callback"
    );
    let capture_planner_details = scaling_policy.is_some() && capture.capture_lifecycle_evidence;
    *scaling = serde_json::to_value(if scaling_policy.is_some() {
        ProviderSpec {
            provider: "dynamo_planner".into(),
            config: serde_json::Value::Null,
        }
    } else {
        ProviderSpec::no_scaling()
    })?;
    let capture_per_request = capture.effective_per_request()
        || spec
            .get("record_per_request")
            .and_then(|value| value.as_bool())
            // ReplaySpec v1 retains per-request records by default.
            .unwrap_or(true);
    let payload = serde_json::to_string(&payload)?;
    let telemetry = telemetry.map(|options| (options.sample_interval_ms, options.observer));
    let evidence = affinity.as_ref().map(|_| RoutingEvidence::default());
    let mut result = match router_mode {
        ReplayRouterMode::RoundRobin => aisimulate_core::execute_replay_with_composition(
            &payload,
            RoundRobinReplayComposition::new(scaling_policy),
            capture,
            telemetry,
        )?,
        ReplayRouterMode::KvRouter => aisimulate_core::execute_replay_with_composition(
            &payload,
            KvReplayComposition::from_spec(router_config, prefill_load_estimator, scaling_policy)
                .with_affinity(
                    affinity,
                    capture_per_request,
                    evidence.clone().unwrap_or_default(),
                ),
            capture,
            telemetry,
        )?,
    };
    result.report_fields.insert(
        "lifecycle_operations".into(),
        serde_json::to_value(&result.report.runtime_evidence.lifecycle_operations)?,
    );
    result.report_fields.insert(
        "coverage".into(),
        serde_json::json!({
            "capture_per_request": capture_per_request,
            "capture_planner_details": capture_planner_details,
            "per_request_records": result.report.per_request.len(),
        }),
    );
    if let Some(evidence) = evidence {
        result
            .report_fields
            .insert("routing_policy".into(), evidence.snapshot()?);
    }
    result.into_json()
}

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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        ReplayRouterMode::RoundRobin => Ok(run_replay_with_composition(
            spec,
            factory,
            Some(input),
            RoundRobinReplayComposition::new(scaling_policy),
            capture_options,
            telemetry.map(|options| (options.sample_interval_ms, options.observer)),
        )?),
        ReplayRouterMode::KvRouter => Ok(run_replay_with_composition(
            spec,
            factory,
            Some(input),
            KvReplayComposition::aggregated(
                args,
                num_workers,
                router_config,
                prefill_load_estimator,
                scaling_policy,
            ),
            capture_options,
            telemetry.map(|options| (options.sample_interval_ms, options.observer)),
        )?),
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        ReplayRouterMode::RoundRobin => Ok(run_replay_with_composition(
            spec,
            factory,
            Some(input),
            RoundRobinReplayComposition::new(scaling_policy),
            capture_options,
            telemetry.map(|options| (options.sample_interval_ms, options.observer)),
        )?),
        ReplayRouterMode::KvRouter => Ok(run_replay_with_composition(
            spec,
            factory,
            Some(input),
            KvReplayComposition::disaggregated(
                config.prefill_args,
                config.decode_args,
                config.num_prefill_workers,
                config.num_decode_workers,
                router_config,
                prefill_load_estimator,
                scaling_policy,
            ),
            capture_options,
            telemetry.map(|options| (options.sample_interval_ms, options.observer)),
        )?),
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
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
    accumulate_session_deltas: bool,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
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
        telemetry,
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
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let driver = concurrency_workload_driver(
        trace,
        args.block_size,
        max_in_flight,
        router_mode,
        accumulate_session_deltas,
    )?;
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
        telemetry,
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
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let agentic_lanes = effective_agentic_lanes(agentic_lanes, trace.play_count());
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
        scaling_policy,
        telemetry,
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
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let agentic_lanes = effective_agentic_lanes(agentic_lanes, trace.play_count());
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
        scaling_policy,
        telemetry,
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
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
    telemetry: Option<ReplayTelemetryOptions>,
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
        telemetry,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg_with_scaling_policy(
    config: OfflineDisaggReplayConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    emit_session_metadata: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let mut driver = trace_workload_driver(
        trace,
        config.prefill_args.block_size,
        router_mode,
        accumulate_session_deltas,
    )?;
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
        telemetry,
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
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let driver = concurrency_workload_driver(
        trace,
        config.prefill_args.block_size,
        max_in_flight,
        router_mode,
        accumulate_session_deltas,
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
        telemetry,
    )
}

#[cfg(all(test, feature = "python-replay"))]
mod canonical_tests {
    #[test]
    fn serialized_replay_preserves_scaling_capture_and_optional_telemetry() {
        use crate::replay::{
            ReplayCaptureOptions, ReplayRouterMode, ReplayScalingDecision, ReplayScalingPolicy,
            ReplayScalingSnapshot, ReplayTelemetryObserver, ReplayTelemetryOptions,
            ReplayTelemetrySnapshot,
        };
        use serde_json::json;
        use std::sync::{Arc, Mutex};

        struct GrowOnce;
        impl ReplayScalingPolicy for GrowOnce {
            fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
                Ok(1.0)
            }

            fn on_tick(
                &mut self,
                _: ReplayScalingSnapshot,
            ) -> anyhow::Result<ReplayScalingDecision> {
                Ok(ReplayScalingDecision {
                    target_prefill: Some(2),
                    target_decode: Some(2),
                    next_tick_ms: None,
                })
            }
        }

        struct Samples(Arc<Mutex<Vec<ReplayTelemetrySnapshot>>>);
        impl ReplayTelemetryObserver for Samples {
            fn on_sample(&mut self, sample: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
                self.0.lock().unwrap().push(sample);
                Ok(())
            }
        }

        for topology in [
            json!({"kind":"aggregated", "workers":{"initial_workers":1}}),
            json!({"kind":"disaggregated", "prefill":{"initial_workers":1}, "decode":{"initial_workers":1}}),
        ] {
            for mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
                for capture in [false, true] {
                    for telemetry in [false, true] {
                        for authored_capture in [None, Some(false), Some(true)] {
                            let samples = Arc::new(Mutex::new(Vec::new()));
                            let mut payload = json!({
                                "version":1, "topology":topology,
                                "engine":{"rank":{"block_size":16,"num_gpu_blocks":64,
                                    "timing_model":{"type":"fixed","prefill_ms":10.0,"decode_ms":1.0}}},
                                "adapters":{"placement":{"provider":"round_robin"},"scaling":{"provider":"none"}},
                                "requests":[{"id":"first","arrival_time_ms":0.0,"input_tokens":64,"output_tokens":2,"input_token_ids":vec![7;64]}]
                            });
                            if let Some(authored_capture) = authored_capture {
                                payload["record_per_request"] = authored_capture.into();
                            }
                            let effective_capture = capture || authored_capture.unwrap_or(true);
                            let report: serde_json::Value = serde_json::from_str(
                                &crate::replay::run_canonical_replay_json(
                                    &payload.to_string(),
                                    mode,
                                    None,
                                    None,
                                    None,
                                    ReplayCaptureOptions {
                                        capture_per_request: capture,
                                        capture_lifecycle_evidence: capture,
                                        ..Default::default()
                                    },
                                    Some(Box::new(GrowOnce)),
                                    telemetry.then(|| ReplayTelemetryOptions {
                                        sample_interval_ms: 3.0,
                                        observer: Box::new(Samples(Arc::clone(&samples))),
                                    }),
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            assert_eq!(report["completed_requests"], 1);
                            assert_eq!(
                                report["coverage"]["capture_per_request"],
                                effective_capture
                            );
                            assert_eq!(report["coverage"]["capture_planner_details"], capture);
                            assert_eq!(
                                report["coverage"]["per_request_records"],
                                usize::from(effective_capture)
                            );
                            assert_eq!(
                                report["lifecycle_operations"]
                                    .as_array()
                                    .unwrap()
                                    .is_empty(),
                                !capture
                            );
                            let samples = samples.lock().unwrap();
                            assert_eq!(samples.is_empty(), !telemetry);
                            if telemetry {
                                assert!(samples.len() >= 2);
                                assert_eq!(
                                    samples[1].sampled_at_ms - samples[0].sampled_at_ms,
                                    3.0
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
