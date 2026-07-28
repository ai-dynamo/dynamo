// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use anyhow::{Result, bail};
use dynamo_kv_router::config::KvRouterConfig;

use super::online;
use super::validate::{
    validate_offline_concurrency_args, validate_offline_disagg_concurrency_args,
    validate_offline_disagg_replay_args, validate_offline_replay_args,
    validate_online_concurrency_args, validate_online_replay_args,
};
use super::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayWorkerArtifacts,
    SlaThresholds, TraceSimulationReport,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::{AgenticTrace, Trace, TraceFileFormat};
use crate::scheduler::RouterEventVisibility;

/// Replay artifact KV-event timestamp visibility override.
///
/// This is intended for parity tests that need to normalize event visibility
/// across mock engines while leaving each engine's production default intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayKvEventVisibility {
    PassStart,
    PassEnd,
}

impl From<ReplayKvEventVisibility> for RouterEventVisibility {
    fn from(visibility: ReplayKvEventVisibility) -> Self {
        match visibility {
            ReplayKvEventVisibility::PassStart => Self::PassStart,
            ReplayKvEventVisibility::PassEnd => Self::PassEnd,
        }
    }
}

fn load_trace_from_file(
    trace_path: &Path,
    trace_block_size: usize,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
) -> Result<Trace> {
    match trace_format {
        TraceFileFormat::Mooncake | TraceFileFormat::MooncakeDelta => {
            Trace::from_mooncake(trace_path, trace_block_size)
        }
        TraceFileFormat::AgenticMooncake => {
            bail!("agentic_mooncake trace format must be loaded as an agentic workload")
        }
        TraceFileFormat::AppliedComputeAgentic => Trace::from_applied_compute_agentic(
            trace_path,
            trace_block_size,
            trace_shared_prefix_ratio,
            trace_num_prefix_groups,
        ),
        TraceFileFormat::Dynamo => {
            bail!("Dynamo request traces must be loaded through the multi-file replay path")
        }
    }
}

fn load_agentic_trace_from_file(
    trace_path: &Path,
    trace_block_size: usize,
    arrival_speedup_ratio: f64,
) -> Result<AgenticTrace> {
    AgenticTrace::from_agentic_mooncake(trace_path, trace_block_size)?
        .normalize_starts()
        .speed_up_timing(arrival_speedup_ratio)
}

fn trace_accumulates_session_deltas(trace_format: TraceFileFormat) -> bool {
    trace_format == TraceFileFormat::MooncakeDelta
}

fn single_turn_trace_requests(
    trace_format: TraceFileFormat,
    trace: &Trace,
) -> Result<Option<Vec<DirectRequest>>> {
    // Dynamo request traces retain compact prompt hashes in WorkloadDriver and
    // materialize only ready requests. The legacy Mooncake path predates that
    // representation and is intentionally unchanged here.
    if matches!(
        trace_format,
        TraceFileFormat::Mooncake | TraceFileFormat::MooncakeDelta
    ) && trace.is_single_turn()
    {
        // The timestamped request path expects every request to carry an
        // arrival timestamp; without this guard a trace missing
        // `first_arrival_timestamp_ms` would panic in
        // `normalize_trace_requests` instead of returning a clear error.
        trace.validate_for_trace_mode()?;
        Ok(Some(trace.to_single_turn_requests()?))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_loaded_trace_with_router_mode_and_options(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    let trace = trace
        .normalize_session_starts()?
        .speed_up_timing(arrival_speedup_ratio)?;
    trace.validate_for_trace_mode()?;
    if trace.is_single_turn() {
        crate::replay::offline::simulate_trace_workload_without_session_metadata(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        crate::replay::offline::simulate_trace_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_loaded_trace_disagg_with_router_mode_and_options(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    let trace = trace
        .normalize_session_starts()?
        .speed_up_timing(arrival_speedup_ratio)?;
    trace.validate_for_trace_mode()?;
    if trace.is_single_turn() {
        crate::replay::offline::simulate_trace_workload_disagg_without_session_metadata(
            config,
            router_config,
            prefill_load_estimator,
            trace,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    } else {
        crate::replay::offline::simulate_trace_workload_disagg(
            config,
            router_config,
            prefill_load_estimator,
            trace,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_loaded_trace_live_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    let trace = trace
        .normalize_session_starts()?
        .speed_up_timing(arrival_speedup_ratio)?;
    trace.validate_for_trace_mode()?;
    online::simulate_trace_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
    )
}

pub fn generate_trace_worker_artifacts_offline(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<ReplayWorkerArtifacts> {
    let args = args.normalized()?;
    crate::replay::offline::generate_trace_worker_artifacts(args, trace)
}

/// Generate offline replay artifacts with a test visibility override for KV events.
pub fn generate_trace_worker_artifacts_offline_with_kv_event_visibility(
    args: MockEngineArgs,
    trace: Trace,
    visibility: ReplayKvEventVisibility,
) -> Result<ReplayWorkerArtifacts> {
    let args = args.normalized()?;
    crate::replay::offline::generate_trace_worker_artifacts_with_visibility(
        args,
        trace,
        Some(visibility.into()),
    )
}

pub fn simulate_trace_file(
    args: MockEngineArgs,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_file_with_router_mode(
        args,
        None,
        None,
        trace_path,
        trace_block_size,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_trace_file_with_router_mode_and_format(
        args,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
        false,
        None,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_file_with_router_mode_and_format(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        let trace =
            load_agentic_trace_from_file(trace_path, trace_block_size, arrival_speedup_ratio)?;
        return crate::replay::offline::simulate_agentic_trace_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            sla,
        );
    }
    if trace_format == TraceFileFormat::AppliedComputeAgentic {
        bail!(
            "applied_compute_agentic trace format requires replay_concurrency because source traces do not contain first-turn timestamps"
        );
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?
    .normalize_session_starts()?
    .speed_up_timing(arrival_speedup_ratio)?;
    let report = if let Some(requests) = single_turn_trace_requests(trace_format, &trace)? {
        crate::replay::offline::simulate_trace(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            1.0,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    } else if trace_accumulates_session_deltas(trace_format) {
        crate::replay::offline::simulate_trace_workload_accumulating_deltas(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    } else {
        crate::replay::offline::simulate_trace_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    };
    Ok(report)
}

pub fn simulate_trace_file_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_trace_file_disagg_with_router_mode_and_format(
        config,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        arrival_speedup_ratio,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
        false,
        None,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_file_disagg_with_router_mode_and_format(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        bail!("agentic_mooncake trace format is not supported for disaggregated replay");
    }
    if trace_format == TraceFileFormat::AppliedComputeAgentic {
        bail!(
            "applied_compute_agentic trace format requires replay_concurrency because source traces do not contain first-turn timestamps"
        );
    }
    if trace_accumulates_session_deltas(trace_format) {
        bail!("mooncake-delta trace format is not supported for disaggregated replay");
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?
    .normalize_session_starts()?
    .speed_up_timing(arrival_speedup_ratio)?;
    let report = if let Some(requests) = single_turn_trace_requests(trace_format, &trace)? {
        crate::replay::offline::simulate_trace_disagg(
            config,
            router_config,
            prefill_load_estimator,
            requests,
            1.0,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    } else {
        crate::replay::offline::simulate_trace_workload_disagg(
            config,
            router_config,
            prefill_load_estimator,
            trace,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    };
    Ok(report)
}

pub fn simulate_trace_live_file(
    args: MockEngineArgs,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_file_with_router_mode(
        args,
        None,
        None,
        trace_path,
        trace_block_size,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_live_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_file_with_router_mode_and_format(
        args,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_live_file_with_router_mode_and_format(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        bail!("agentic_mooncake trace format is not supported for online replay");
    }
    if trace_format == TraceFileFormat::AppliedComputeAgentic {
        bail!(
            "applied_compute_agentic trace format requires replay_concurrency because source traces do not contain first-turn timestamps"
        );
    }
    if trace_accumulates_session_deltas(trace_format) {
        bail!("mooncake-delta trace format is not supported for online replay");
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?
    .normalize_session_starts()?
    .speed_up_timing(arrival_speedup_ratio)?;
    if let Some(requests) = single_turn_trace_requests(trace_format, &trace)? {
        online::simulate_trace_requests(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            1.0,
            router_mode,
        )
    } else {
        online::simulate_trace_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
        )
    }
}

pub fn simulate_trace_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_requests_with_router_mode(
        args,
        None,
        None,
        requests,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trace_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    if requests.is_empty() {
        bail!("trace replay requires at least one request");
    }

    let report = crate::replay::offline::simulate_trace(
        args,
        router_config,
        prefill_load_estimator,
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
        false,
        None,
        sla,
    )?;
    Ok(report)
}

pub fn simulate_trace_requests_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    if requests.is_empty() {
        bail!("trace replay requires at least one request");
    }

    let report = crate::replay::offline::simulate_trace_disagg(
        config,
        router_config,
        prefill_load_estimator,
        requests,
        arrival_speedup_ratio,
        router_mode,
        false,
        None,
        sla,
    )?;
    Ok(report)
}

pub fn simulate_trace_live_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_requests_with_router_mode(
        args,
        None,
        None,
        requests,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_live_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    if requests.is_empty() {
        bail!("trace replay requires at least one request");
    }

    online::simulate_trace_requests(
        args,
        router_config,
        prefill_load_estimator,
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
    )
}

pub fn simulate_concurrency_file(
    args: MockEngineArgs,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_file_with_router_mode(
        args,
        None,
        None,
        trace_path,
        trace_block_size,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_file_with_router_mode_and_format(
        args,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        max_in_flight,
        num_workers,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
        false,
        None,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_file_with_router_mode_and_format(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_concurrency_args(&args, num_workers, max_in_flight, router_mode)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        bail!("agentic_mooncake trace format is not supported with replay_concurrency");
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?;
    let report = if trace_accumulates_session_deltas(trace_format) {
        crate::replay::offline::simulate_concurrency_workload_accumulating_deltas(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    } else {
        crate::replay::offline::simulate_concurrency_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
            record_per_request,
            max_sim_time_ms,
            sla,
        )?
    };
    Ok(report)
}

pub fn simulate_concurrency_file_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_file_disagg_with_router_mode_and_format(
        config,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        max_in_flight,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
        false,
        None,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_file_disagg_with_router_mode_and_format(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_concurrency_args(&config, max_in_flight, router_mode)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        bail!("agentic_mooncake trace format is not supported for disaggregated replay");
    }
    if trace_accumulates_session_deltas(trace_format) {
        bail!("mooncake-delta trace format is not supported for disaggregated replay");
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?;
    let report = crate::replay::offline::simulate_concurrency_workload_disagg(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
    )?;
    Ok(report)
}

pub fn simulate_concurrency_live_file(
    args: MockEngineArgs,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_file_with_router_mode(
        args,
        None,
        None,
        trace_path,
        trace_block_size,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_live_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_file_with_router_mode_and_format(
        args,
        router_config,
        prefill_load_estimator,
        trace_path,
        trace_block_size,
        max_in_flight,
        num_workers,
        router_mode,
        TraceFileFormat::Mooncake,
        0.0,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_live_file_with_router_mode_and_format(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace_path: &Path,
    trace_block_size: usize,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_concurrency_args(&args, num_workers, max_in_flight)?;
    if trace_format == TraceFileFormat::AgenticMooncake {
        bail!("agentic_mooncake trace format is not supported for online replay");
    }
    if trace_accumulates_session_deltas(trace_format) {
        bail!("mooncake-delta trace format is not supported for online replay");
    }
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?;
    online::simulate_concurrency_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
    )
}

pub fn simulate_concurrency_live_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_requests_with_router_mode(
        args,
        None,
        None,
        requests,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_live_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_concurrency_args(&args, num_workers, max_in_flight)?;
    if requests.is_empty() {
        bail!("concurrency replay requires at least one request");
    }

    online::simulate_concurrency_requests(
        args,
        router_config,
        prefill_load_estimator,
        requests,
        max_in_flight,
        num_workers,
        router_mode,
    )
}

pub fn simulate_concurrency_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_requests_with_router_mode(
        args,
        None,
        None,
        requests,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_concurrency_args(&args, num_workers, max_in_flight, router_mode)?;
    if requests.is_empty() {
        bail!("concurrency replay requires at least one request");
    }

    crate::replay::offline::simulate_concurrency(
        args,
        router_config,
        prefill_load_estimator,
        requests,
        max_in_flight,
        num_workers,
        router_mode,
        false,
        None,
        sla,
    )
}

pub fn simulate_concurrency_requests_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_concurrency_args(&config, max_in_flight, router_mode)?;
    if requests.is_empty() {
        bail!("concurrency replay requires at least one request");
    }

    crate::replay::offline::simulate_concurrency_disagg(
        config,
        router_config,
        prefill_load_estimator,
        requests,
        max_in_flight,
        router_mode,
        false,
        None,
        sla,
    )
}

pub fn simulate_trace_workload(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        num_workers,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
    )
}

pub fn simulate_trace_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_router_mode_and_options(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        false,
        None,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_trace_workload_with_router_mode_and_options(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    let report = crate::replay::offline::simulate_trace_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
    )?;
    Ok(report)
}

pub fn simulate_trace_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_disagg_with_router_mode_and_options(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        false,
        None,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_trace_workload_disagg_with_router_mode_and_options(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    let report = crate::replay::offline::simulate_trace_workload_disagg(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        record_per_request,
        max_sim_time_ms,
        sla,
    )?;
    Ok(report)
}

pub fn simulate_trace_live_workload(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_live_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    online::simulate_trace_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
    )
}

pub fn simulate_concurrency_workload(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_router_mode_and_options(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        false,
        None,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_workload_with_router_mode_and_options(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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
    validate_offline_concurrency_args(&args, num_workers, max_in_flight, router_mode)?;
    crate::replay::offline::simulate_concurrency_workload(
        args,
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

pub fn simulate_concurrency_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_disagg_with_router_mode_and_options(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
        false,
        None,
        sla,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_concurrency_workload_disagg_with_router_mode_and_options(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_concurrency_args(&config, max_in_flight, router_mode)?;
    crate::replay::offline::simulate_concurrency_workload_disagg(
        config,
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

#[allow(clippy::too_many_arguments)]
pub fn simulate_agentic_trace_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    sla: SlaThresholds,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    crate::replay::offline::simulate_agentic_trace_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        sla,
    )
}

pub fn simulate_concurrency_live_workload(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_live_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_concurrency_args(&args, num_workers, max_in_flight)?;
    online::simulate_concurrency_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::{EngineType, G1Backend, SglangArgs, WorkerType};
    use crate::loadgen::{SessionTrace, TurnTrace};
    use rstest::rstest;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    fn replay_test_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(8))
            .speedup_ratio(1000.0)
            .build()
            .unwrap()
    }

    fn disagg_test_config() -> OfflineDisaggReplayConfig {
        OfflineDisaggReplayConfig {
            prefill_args: MockEngineArgs {
                worker_type: WorkerType::Prefill,
                block_size: 4,
                ..MockEngineArgs::default()
            },
            decode_args: MockEngineArgs {
                worker_type: WorkerType::Decode,
                block_size: 4,
                ..MockEngineArgs::default()
            },
            num_prefill_workers: 1,
            num_decode_workers: 1,
        }
    }

    fn single_turn_dynamo_trace(first_arrival_timestamp_ms: Option<f64>) -> Trace {
        Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "request_1".to_string(),
                first_arrival_timestamp_ms,
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            }],
        }
    }

    #[test]
    fn loaded_dynamo_trace_preserves_request_metadata_contract() {
        let report = simulate_loaded_trace_with_router_mode_and_options(
            replay_test_args(),
            None,
            None,
            single_turn_dynamo_trace(Some(0.0)),
            2,
            1.0,
            ReplayRouterMode::RoundRobin,
            true,
            None,
            SlaThresholds::default(),
        )
        .unwrap();

        assert_eq!(report.per_request.len(), 1);
        assert_eq!(report.per_request[0].session_id, None);
        assert_eq!(report.per_request[0].turn_index, None);
    }

    #[test]
    fn loaded_dynamo_disagg_trace_validates_timestamps() {
        let error = simulate_loaded_trace_disagg_with_router_mode_and_options(
            disagg_test_config(),
            None,
            None,
            single_turn_dynamo_trace(None),
            1.0,
            ReplayRouterMode::RoundRobin,
            false,
            None,
            SlaThresholds::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("first_arrival_timestamp_ms"));
    }

    #[rstest]
    #[case::vllm(EngineType::Vllm)]
    #[case::trtllm(EngineType::Trtllm)]
    fn native_g1_runs_through_offline_replay_entrypoint(#[case] engine_type: EngineType) {
        let args = MockEngineArgs::builder()
            .engine_type(engine_type)
            .g1_backend(G1Backend::Native)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let requests = [11_u128, 22]
            .into_iter()
            .enumerate()
            .map(|(index, uuid)| DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                output_token_ids: Some(vec![100, 101]),
                uuid: Some(Uuid::from_u128(uuid)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(index as f64 * 100.0),
                ..Default::default()
            })
            .collect();

        // This public API normalizes/validates args and then executes the
        // deterministic single-worker replay core used by offline replay.
        let report = simulate_trace_requests(args, requests, 1, 1.0).unwrap();

        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 4);
        assert!(
            report.first_admission_prefix_cache_reused_ratio > 0.0,
            "second identical prompt should reuse native G1 prefix blocks"
        );
    }

    #[test]
    fn one_worker_sglang_impossible_request_returns_dead_end_error() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(1)
            .speedup_ratio(1000.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let request = DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        };

        let err = simulate_trace_requests_with_router_mode(
            args,
            None,
            None,
            vec![request],
            1,
            1.0,
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap_err();

        assert_eq!(
            err.to_string(),
            "offline replay reached a dead end with 1 in-flight requests remaining"
        );
    }

    #[test]
    fn agentic_mooncake_trace_file_loads_and_scales_timing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{}",
            serde_json::json!({
                "request_id": "r1",
                "timestamp": 100.0,
                "input_length": 4,
                "output_length": 1,
                "hash_ids": [1]
            })
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            serde_json::json!({
                "request_id": "r2",
                "timestamp": 130.0,
                "delay": 10.0,
                "tool_wait_ms": 6.0,
                "wait_for": ["r1"],
                "input_length": 4,
                "output_length": 1,
                "hash_ids": [1]
            })
        )
        .unwrap();

        let trace = load_agentic_trace_from_file(file.path(), 4, 2.0).unwrap();

        assert_eq!(trace.turns[0].first_ready_timestamp_ms, Some(0.0));
        assert_eq!(trace.turns[1].first_ready_timestamp_ms, Some(15.0));
        assert_eq!(trace.turns[1].delay_after_dependencies_ms, 8.0);
    }

    #[test]
    fn single_turn_legacy_trace_formats_use_request_path() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![
                SessionTrace {
                    session_id: "request_1".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
                SessionTrace {
                    session_id: "request_2".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        };

        for trace_format in [TraceFileFormat::Mooncake, TraceFileFormat::MooncakeDelta] {
            let requests = single_turn_trace_requests(trace_format, &trace)
                .unwrap()
                .expect("single-turn traces should become request traces");

            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].arrival_timestamp_ms, Some(0.0));
            assert_eq!(requests[1].arrival_timestamp_ms, Some(0.0));
        }

        assert!(
            single_turn_trace_requests(TraceFileFormat::Dynamo, &trace)
                .unwrap()
                .is_none(),
            "Dynamo traces must retain compact prompts in the workload path"
        );
    }

    #[test]
    fn single_turn_request_trace_formats_without_timestamps_are_rejected() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "request_1".to_string(),
                first_arrival_timestamp_ms: None,
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            }],
        };

        for trace_format in [TraceFileFormat::Mooncake, TraceFileFormat::MooncakeDelta] {
            let err = single_turn_trace_requests(trace_format, &trace)
                .expect_err("missing first_arrival_timestamp_ms must error before reaching the timestamped request path");
            assert!(
                err.to_string().contains("first_arrival_timestamp_ms"),
                "expected validation error to mention first_arrival_timestamp_ms, got {err}",
            );
        }
    }
}
