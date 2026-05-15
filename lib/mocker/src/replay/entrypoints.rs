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
    TraceSimulationReport,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::{Trace, TraceFileFormat};

fn load_trace_from_file(
    trace_path: &Path,
    trace_block_size: usize,
    trace_format: TraceFileFormat,
    trace_shared_prefix_ratio: f64,
    trace_num_prefix_groups: usize,
) -> Result<Trace> {
    match trace_format {
        TraceFileFormat::Mooncake => Trace::from_mooncake(trace_path, trace_block_size),
        TraceFileFormat::AppliedComputeAgentic => Trace::from_applied_compute_agentic(
            trace_path,
            trace_block_size,
            trace_shared_prefix_ratio,
            trace_num_prefix_groups,
        ),
    }
}

fn single_turn_mooncake_requests(
    trace_format: TraceFileFormat,
    trace: &Trace,
) -> Result<Option<Vec<DirectRequest>>> {
    if trace_format == TraceFileFormat::Mooncake && trace.is_single_turn() {
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

pub fn generate_trace_worker_artifacts_offline(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<ReplayWorkerArtifacts> {
    let args = args.normalized()?;
    crate::replay::offline::generate_trace_worker_artifacts(args, trace)
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
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
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
    let report = if let Some(requests) = single_turn_mooncake_requests(trace_format, &trace)? {
        crate::replay::offline::simulate_trace(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            1.0,
            router_mode,
        )?
    } else {
        crate::replay::offline::simulate_trace_workload(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
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
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
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
    let report = if let Some(requests) = single_turn_mooncake_requests(trace_format, &trace)? {
        crate::replay::offline::simulate_trace_disagg(
            config,
            router_config,
            prefill_load_estimator,
            requests,
            1.0,
            router_mode,
        )?
    } else {
        crate::replay::offline::simulate_trace_workload_disagg(
            config,
            router_config,
            prefill_load_estimator,
            trace,
            router_mode,
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
    if let Some(requests) = single_turn_mooncake_requests(trace_format, &trace)? {
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
    )
}

pub fn simulate_trace_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
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
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_concurrency_args(&args, num_workers, max_in_flight, router_mode)?;
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?;
    let report = simulate_concurrency_workload_with_router_mode(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
    )?;
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
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_concurrency_args(&config, max_in_flight, router_mode)?;
    let trace = load_trace_from_file(
        trace_path,
        trace_block_size,
        trace_format,
        trace_shared_prefix_ratio,
        trace_num_prefix_groups,
    )?;
    let report = simulate_concurrency_workload_disagg_with_router_mode(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
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
    )
}

pub fn simulate_concurrency_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
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
    )
}

pub fn simulate_concurrency_requests_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
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
    )
}

pub fn simulate_trace_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
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
    )?;
    Ok(report)
}

pub fn simulate_trace_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    let report = crate::replay::offline::simulate_trace_workload_disagg(
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
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
    )
}

pub fn simulate_concurrency_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
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
    )
}

pub fn simulate_concurrency_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
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
    use crate::loadgen::{SessionTrace, TurnTrace};

    #[test]
    fn single_turn_mooncake_trace_uses_request_path() {
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
                    }],
                },
            ],
        };

        let requests = single_turn_mooncake_requests(TraceFileFormat::Mooncake, &trace)
            .unwrap()
            .expect("single-turn Mooncake traces should become request traces");

        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].arrival_timestamp_ms, Some(0.0));
        assert_eq!(requests[1].arrival_timestamp_ms, Some(0.0));
    }

    #[test]
    fn single_turn_mooncake_trace_without_timestamps_is_rejected() {
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
                }],
            }],
        };

        let err = single_turn_mooncake_requests(TraceFileFormat::Mooncake, &trace)
            .expect_err("missing first_arrival_timestamp_ms must error before reaching the timestamped request path");
        assert!(
            err.to_string().contains("first_arrival_timestamp_ms"),
            "expected validation error to mention first_arrival_timestamp_ms, got {err}",
        );
    }

    #[test]
    fn multi_turn_mooncake_trace_stays_on_workload_path() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 10.0,
                    },
                ],
            }],
        };

        assert!(
            single_turn_mooncake_requests(TraceFileFormat::Mooncake, &trace)
                .unwrap()
                .is_none()
        );
    }
}
