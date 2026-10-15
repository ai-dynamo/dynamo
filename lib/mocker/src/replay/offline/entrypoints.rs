// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo compatibility entrypoints over the packaged AISimulate Replayer.
//!
//! This module lowers Dynamo configuration into Replay-owned contracts. It
//! must never include or compile implementation sources from another crate.

use std::collections::VecDeque;

use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ProviderSpec, ReplayAdapters, ReplayCaptureOptions,
    ReplayComposition, ReplayEngineConfig, ReplayRuntimeInput, ReplayScalingPolicy, ReplaySpec,
    ReplayTopology, Replayer, WorkerPoolSpec,
};
use anyhow::Result;

use super::extensions::kv_events;
use super::extensions::kv_router::{
    KvReplayComposition, ReplayKvRouterConfig, RoundRobinReplayComposition, provider_spec,
};
use super::normalize_trace_requests;
use crate::common::handoff::NormalizedHandoffConformance;
use crate::common::protocols::{DirectRequest, EngineType, MockerConfig, WorkerType};
use crate::engine_adapter::{aggregated_replay_setup, disaggregated_replay_setup};
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
#[cfg(feature = "python-replay")]
use crate::replay::ReplayAffinityConfig;
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode,
    ReplayTelemetryOptions, ReplayWorkerArtifacts, SlaThresholds, TraceSimulationReport,
};
use crate::scheduler::RouterEventVisibility;

// Dynamo and AISimulate can link separately pinned policy crates. This adapter
// forwards the existing estimator without sharing crate-specific trait objects.
#[cfg(feature = "python-replay")]
struct CompatiblePrefillLoadEstimator(ReplayPrefillLoadEstimator);

#[cfg(feature = "python-replay")]
impl aisimulate_python::native_router::PrefillLoadEstimator for CompatiblePrefillLoadEstimator {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<std::time::Duration> {
        self.0
            .predict_prefill_duration(batch_size, effective_isl, prefix)
    }
}

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
        ReplayRouterMode::KvRouter => ProviderSpec {
            provider: native_provider.provider,
            config: serde_json::json!({"router_config": router_config, "affinity": affinity}),
        },
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
    let mut result = match router_mode {
        ReplayRouterMode::RoundRobin => aisimulate_core::execute_replay_with_composition(
            &payload,
            RoundRobinReplayComposition::new(scaling_policy),
            capture,
            telemetry,
        )?,
        ReplayRouterMode::KvRouter => aisimulate_python::execute_native_router_replay(
            &payload,
            capture,
            scaling_policy,
            telemetry,
            prefill_load_estimator.map(|estimator| {
                std::sync::Arc::new(CompatiblePrefillLoadEstimator(estimator))
                    as aisimulate_python::native_router::ReplayPrefillLoadEstimator
            }),
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
    result.into_json()
}

fn startup_delay_ms(args: &MockerConfig) -> f64 {
    args.startup_time
        .filter(|seconds| *seconds > 0.0)
        .map_or(0.0, |seconds| seconds * 1_000.0)
}

fn worker_pool(initial_workers: usize, args: &MockerConfig) -> WorkerPoolSpec {
    WorkerPoolSpec {
        initial_workers,
        startup_delay_ms: startup_delay_ms(args),
    }
}

fn validate_agentic_host_offload(
    input: &ReplayRuntimeInput,
    roles: &[(usize, &MockerConfig)],
    scaling_enabled: bool,
) -> Result<()> {
    let ReplayRuntimeInput::Workload(driver) = input else {
        return Ok(());
    };
    if !driver.is_agentic() {
        return Ok(());
    }
    anyhow::ensure!(
        roles.iter().all(|(_, args)| args.g3_offload.is_none()),
        "agentic replay does not support G3"
    );
    if !roles
        .iter()
        .any(|(_, args)| args.native_host_offload.is_some())
    {
        return Ok(());
    }

    // These entrypoints are offline. Apply the AgentX G2 deployment contract
    // to every active role, including a P/D role whose own G2 is disabled.
    anyhow::ensure!(
        roles.iter().all(|(workers, _)| *workers == 1),
        "agentic host offload requires one aggregated worker or one prefill and one decode worker"
    );
    anyhow::ensure!(
        !scaling_enabled,
        "agentic host offload requires static worker pools without a scaling policy"
    );
    for (_, args) in roles {
        anyhow::ensure!(
            args.backend == EngineType::Vllm
                && args.ais_perf_config().is_none_or(|config| {
                    config
                        .get("backend")
                        .is_none_or(|backend| backend == "vllm")
                }),
            "agentic host offload requires backend=vllm on every role"
        );
        anyhow::ensure!(
            args.dp_size == 1,
            "agentic host offload requires attention DP=1 on every role"
        );
        anyhow::ensure!(
            args.decode_speedup_ratio == 1.0
                && args.aic_nextn.is_none()
                && args.ais_perf_config().is_none_or(|config| {
                    config
                        .get("speculation")
                        .is_none_or(serde_json::Value::is_null)
                }),
            "agentic host offload requires speculative decoding disabled"
        );
    }
    Ok(())
}

fn with_telemetry<C: ReplayComposition>(
    replayer: Replayer<C>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<Replayer<C>> {
    match telemetry {
        Some(options) => {
            Ok(replayer.with_telemetry_observer(options.sample_interval_ms, options.observer)?)
        }
        None => Ok(replayer),
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
    args: MockerConfig,
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
    args: MockerConfig,
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
    validate_agentic_host_offload(&input, &[(num_workers, &args)], scaling_policy.is_some())?;
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
        ReplayRouterMode::RoundRobin => {
            let replayer = Replayer::with_composition(
                spec,
                factory,
                RoundRobinReplayComposition::new(scaling_policy),
            )?
            .with_capture_options(capture_options)
            .with_runtime_input(input);
            Ok(with_telemetry(replayer, telemetry)?.run()?)
        }
        ReplayRouterMode::KvRouter => {
            let replayer = Replayer::with_composition(
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
            .with_runtime_input(input);
            Ok(with_telemetry(replayer, telemetry)?.run()?)
        }
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
    validate_agentic_host_offload(
        &input,
        &[
            (config.num_prefill_workers, &config.prefill_args),
            (config.num_decode_workers, &config.decode_args),
        ],
        scaling_policy.is_some(),
    )?;
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
        ReplayRouterMode::RoundRobin => {
            let replayer = Replayer::with_composition(
                spec,
                factory,
                RoundRobinReplayComposition::new(scaling_policy),
            )?
            .with_capture_options(capture_options)
            .with_runtime_input(input);
            Ok(with_telemetry(replayer, telemetry)?.run()?)
        }
        ReplayRouterMode::KvRouter => {
            let replayer = Replayer::with_composition(
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
            .with_runtime_input(input);
            Ok(with_telemetry(replayer, telemetry)?.run()?)
        }
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
        MockerConfig::from_value(serde_json::json!({
            "engine": {
                "backend": engine_type,
                "worker_type": worker_type,
                "block_size": 4,
                "num_gpu_blocks": 64,
                "max_num_batched_tokens": 64,
                "max_num_seqs": 2,
                "speedup_ratio": 1000.0,
                "decode_speedup_ratio": 1000.0,
                "kv_transfer_bandwidth": 1.0,
                "kv_transfer_bytes_per_token": 1_000_000,
                "kv_transfer_timing_mode": transfer_timing_mode
            }
        }))
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
    args: MockerConfig,
    trace: Trace,
) -> Result<ReplayWorkerArtifacts> {
    generate_trace_worker_artifacts_with_visibility(args, trace, None)
}

pub(crate) fn generate_trace_worker_artifacts_with_visibility(
    args: MockerConfig,
    trace: Trace,
    visibility: Option<RouterEventVisibility>,
) -> Result<ReplayWorkerArtifacts> {
    kv_events::generate_trace_worker_artifacts_with_visibility(args, trace, visibility)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_with_scaling_policy(
    args: MockerConfig,
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
    args: MockerConfig,
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
    args: MockerConfig,
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
    args: MockerConfig,
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
    args: MockerConfig,
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
    args: MockerConfig,
    router_config: Option<ReplayKvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    agentic_lanes: Option<usize>,
    agentic_options: crate::replay::AgenticReplayOptions,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let driver = agentic_options.into_driver(
        trace,
        args.block_size,
        router_mode == ReplayRouterMode::KvRouter,
        agentic_lanes,
        max_sim_time_ms,
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
    agentic_options: crate::replay::AgenticReplayOptions,
    sla: SlaThresholds,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    telemetry: Option<ReplayTelemetryOptions>,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let driver = agentic_options.into_driver(
        trace,
        config.prefill_args.block_size,
        router_mode == ReplayRouterMode::KvRouter,
        agentic_lanes,
        max_sim_time_ms,
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

#[cfg(test)]
mod agentic_host_offload_tests {
    use super::*;
    use crate::loadgen::{
        AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticHashIdScope,
        AgenticMooncakeHeader, AgenticMooncakeRow, AgenticSourceProvenance,
    };

    fn agentic_input() -> ReplayRuntimeInput {
        let trace = AgenticTrace::from_agentic_mooncake_rows(
            AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: 4,
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "test".to_string(),
                    digest: "agentic-g2-deployment-guard".to_string(),
                },
            },
            vec![AgenticMooncakeRow {
                request_id: "request".to_string(),
                play_id: "play".to_string(),
                session_id: "session".to_string(),
                model: "model".to_string(),
                input_length: Some(4),
                output_length: Some(1),
                hash_ids: Some(vec![1]),
                ..Default::default()
            }],
        )
        .unwrap();
        ReplayRuntimeInput::Workload(trace.into_trace_driver_with_options(4, true, None).unwrap())
    }

    fn args(scope: Option<&str>) -> MockerConfig {
        MockerConfig::from_value(serde_json::json!({
            "engine": {
                "kv_cache_bytes_per_token": 1024,
                "native_host_offload": scope.map(|scope| serde_json::json!({
                    "scope": scope,
                    "num_host_blocks": 16,
                    "kv_layout_id": "agentic-g2-test",
                })),
            },
        }))
        .unwrap()
    }

    #[test]
    fn accepts_single_worker_and_independent_pd_g2_scopes() {
        let input = agentic_input();
        let scopes = [None, Some("dp_rank_local"), Some("cluster_shared")];
        for prefill_scope in scopes {
            let mut prefill = args(prefill_scope);
            prefill.tensor_parallel_size = 4;
            validate_agentic_host_offload(&input, &[(1, &prefill)], false).unwrap();
            for decode_scope in scopes {
                let decode = args(decode_scope);
                validate_agentic_host_offload(&input, &[(1, &prefill), (1, &decode)], false)
                    .unwrap();
            }
        }
    }

    #[test]
    fn rejects_unsupported_deployment_on_every_active_role() {
        let input = agentic_input();
        let offloaded = args(Some("dp_rank_local"));
        // A role without G2 is still part of the restricted AgentX deployment.
        let mut invalid_roles = Vec::new();
        let mut invalid = args(None);
        invalid.dp_size = 2;
        invalid_roles.push((
            invalid,
            "agentic host offload requires attention DP=1 on every role",
        ));
        for backend in [EngineType::Sglang, EngineType::Trtllm] {
            let mut invalid = args(None);
            invalid.backend = backend;
            invalid_roles.push((
                invalid,
                "agentic host offload requires backend=vllm on every role",
            ));
        }
        let mut invalid = args(None);
        invalid.timing_model = aisimulate_core::engine::TimingModelConfig::External {
            provider: "ais".to_string(),
            config: serde_json::json!({"backend": "sglang"}),
        };
        invalid_roles.push((
            invalid,
            "agentic host offload requires backend=vllm on every role",
        ));
        let mut invalid = args(None);
        invalid.aic_nextn = Some(1);
        invalid_roles.push((
            invalid,
            "agentic host offload requires speculative decoding disabled",
        ));
        let mut invalid = args(None);
        invalid.decode_speedup_ratio = 2.0;
        invalid_roles.push((
            invalid,
            "agentic host offload requires speculative decoding disabled",
        ));
        let mut invalid = args(None);
        invalid.timing_model = aisimulate_core::engine::TimingModelConfig::External {
            provider: "ais".to_string(),
            config: serde_json::json!({"speculation": {"nextn": 2}}),
        };
        invalid_roles.push((
            invalid,
            "agentic host offload requires speculative decoding disabled",
        ));
        for (invalid, expected) in invalid_roles {
            for roles in [
                vec![(1, &offloaded), (1, &invalid)],
                vec![(1, &invalid), (1, &offloaded)],
            ] {
                let error = validate_agentic_host_offload(&input, &roles, false).unwrap_err();
                assert_eq!(error.to_string(), expected);
            }
        }
        for roles in [
            vec![(2, &offloaded)],
            vec![(2, &offloaded), (1, &offloaded)],
            vec![(1, &offloaded), (2, &offloaded)],
        ] {
            let error = validate_agentic_host_offload(&input, &roles, false).unwrap_err();
            assert_eq!(
                error.to_string(),
                "agentic host offload requires one aggregated worker or one prefill and one decode worker"
            );
        }
        let error = validate_agentic_host_offload(&input, &[(1, &offloaded)], true).unwrap_err();
        assert_eq!(
            error.to_string(),
            "agentic host offload requires static worker pools without a scaling policy"
        );
    }

    #[test]
    fn native_execution_entrypoints_apply_the_agentic_g2_guard() {
        let error = run_aggregated(
            args(Some("cluster_shared")),
            None,
            None,
            agentic_input(),
            2,
            None,
            ReplayRouterMode::RoundRobin,
            false,
            None,
            SlaThresholds::default(),
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "agentic host offload requires one aggregated worker or one prefill and one decode worker"
        );

        let mut decode = args(None);
        decode.dp_size = 2;
        let error = run_disaggregated(
            OfflineDisaggReplayConfig {
                prefill_args: args(Some("dp_rank_local")),
                decode_args: decode,
                num_prefill_workers: 1,
                num_decode_workers: 1,
            },
            None,
            None,
            agentic_input(),
            None,
            ReplayRouterMode::RoundRobin,
            false,
            None,
            SlaThresholds::default(),
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "agentic host offload requires attention DP=1 on every role"
        );
    }

    #[test]
    fn native_execution_rejects_canonical_g3_for_agentic_replay() {
        for scope in ["worker_local", "cluster_shared"] {
            let mut offloaded = args(Some("cluster_shared"));
            offloaded.g3_offload = Some(
                serde_json::from_value(serde_json::json!({
                    "scope": scope,
                    "num_g3_blocks": 8,
                }))
                .unwrap(),
            );
            // G3 is valid in the canonical engine schema, but remains outside
            // the qualified AgentX support boundary.
            let offloaded = offloaded.normalized().unwrap();
            validate_agentic_host_offload(
                &ReplayRuntimeInput::Requests(VecDeque::new()),
                &[(1, &offloaded)],
                false,
            )
            .unwrap();
            let error = run_aggregated(
                offloaded,
                None,
                None,
                agentic_input(),
                1,
                None,
                ReplayRouterMode::RoundRobin,
                false,
                None,
                SlaThresholds::default(),
                None,
                None,
            )
            .unwrap_err();
            assert_eq!(error.to_string(), "agentic replay does not support G3");
        }
    }

    #[test]
    fn leaves_hbm_and_non_agentic_g2_deployments_unchanged() {
        let mut hbm = args(None);
        hbm.backend = EngineType::Sglang;
        hbm.dp_size = 2;
        hbm.decode_speedup_ratio = 2.0;
        validate_agentic_host_offload(&agentic_input(), &[(2, &hbm)], true).unwrap();
        hbm.decode_speedup_ratio = 1.0;
        hbm.aic_nextn = Some(2);
        validate_agentic_host_offload(&agentic_input(), &[(2, &hbm)], true).unwrap();

        let mut offloaded = args(Some("cluster_shared"));
        offloaded.dp_size = 2;
        offloaded.decode_speedup_ratio = 2.0;
        validate_agentic_host_offload(
            &ReplayRuntimeInput::Requests(VecDeque::new()),
            &[(2, &offloaded)],
            true,
        )
        .unwrap();
        let trace = Trace::from_mooncake_rows(
            vec![crate::loadgen::MooncakeRow {
                input_length: Some(4),
                output_length: Some(1),
                hash_ids: Some(vec![1]),
                timestamp: Some(0.0),
                ..Default::default()
            }],
            4,
        )
        .unwrap();
        let input =
            ReplayRuntimeInput::Workload(trace.into_trace_driver_with_block_size(4).unwrap());
        validate_agentic_host_offload(&input, &[(2, &offloaded)], true).unwrap();
    }
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
