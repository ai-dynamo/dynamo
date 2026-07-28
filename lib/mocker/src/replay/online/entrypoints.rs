// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::{Result, anyhow, bail};
use dynamo_kv_router::config::KvRouterConfig;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
use crate::replay::{
    ReplayPrefillLoadEstimator, ReplayRouterMode, SlaThresholds, TraceSimulationReport,
    normalize_trace_requests,
};

use super::live_runtime::LiveRuntime;
use super::state::{LiveReplayMode, LiveRuntimeStats};

#[derive(Clone, Copy, Default)]
pub(crate) struct OnlineReplayOptions {
    pub(crate) record_per_request: bool,
    pub(crate) sla: SlaThresholds,
}

pub(crate) struct OnlineReplayConfig {
    pub(super) args: MockEngineArgs,
    pub(super) router_config: Option<KvRouterConfig>,
    pub(super) prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    pub(super) num_workers: usize,
    pub(super) router_mode: ReplayRouterMode,
    pub(super) options: OnlineReplayOptions,
}

impl OnlineReplayConfig {
    pub(crate) fn new(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        options: OnlineReplayOptions,
    ) -> Self {
        Self {
            args,
            router_config,
            prefill_load_estimator,
            num_workers,
            router_mode,
            options,
        }
    }

    fn normalized(mut self) -> Result<Self> {
        self.args = self.args.normalized()?;
        Ok(self)
    }
}

fn total_turns(trace: &Trace) -> usize {
    trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum()
}

fn run_live_runtime(
    config: OnlineReplayConfig,
    pending: VecDeque<DirectRequest>,
    mode: LiveReplayMode,
    cancel: CancellationToken,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move { LiveRuntime::new(config, pending, mode, cancel)?.run().await })
}

fn run_live_workload_runtime(
    config: OnlineReplayConfig,
    driver: WorkloadDriver,
    total_turns: usize,
    mode: LiveReplayMode,
    cancel: CancellationToken,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(config, VecDeque::new(), mode, cancel)?
            .run_workload(driver, total_turns)
            .await
    })
}

pub(crate) fn simulate_trace_requests(
    config: OnlineReplayConfig,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (report, _) = run_live_runtime(
        config,
        pending,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_requests(
    config: OnlineReplayConfig,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    if requests.is_empty() {
        bail!("online concurrency replay requires at least one request");
    }

    let pending = VecDeque::from(requests);
    let (report, _) = run_live_runtime(
        config,
        pending,
        LiveReplayMode::Concurrency { max_in_flight },
        CancellationToken::new(),
    )?;
    Ok(report)
}

pub(crate) fn simulate_trace_workload(
    config: OnlineReplayConfig,
    trace: Trace,
    emit_session_metadata: bool,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let engine_block_size = config.args.block_size;
    let total_turns = total_turns(&trace);
    let mut driver = trace.into_trace_driver_with_block_size(engine_block_size)?;
    if !emit_session_metadata {
        driver = driver.without_session_metadata();
    }
    let (report, _) = run_live_workload_runtime(
        config,
        driver,
        total_turns,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_workload(
    config: OnlineReplayConfig,
    trace: Trace,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let engine_block_size = config.args.block_size;
    let total_turns = total_turns(&trace);
    let (report, _) = run_live_workload_runtime(
        config,
        trace.into_concurrency_driver_with_block_size(engine_block_size, max_in_flight)?,
        total_turns,
        LiveReplayMode::Concurrency { max_in_flight },
        CancellationToken::new(),
    )?;
    Ok(report)
}

pub(crate) fn simulate_agentic_trace_workload(
    config: OnlineReplayConfig,
    trace: AgenticTrace,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    let engine_block_size = config.args.block_size;
    let total_turns = trace.turns.len();
    let (report, _) = run_live_workload_runtime(
        config,
        trace.into_trace_driver_with_block_size(engine_block_size)?,
        total_turns,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )?;
    Ok(report)
}

#[cfg(test)]
fn default_test_config(
    args: MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> OnlineReplayConfig {
    OnlineReplayConfig::new(
        args,
        None,
        None,
        num_workers,
        router_mode,
        OnlineReplayOptions::default(),
    )
}

#[cfg(test)]
pub(super) fn simulate_trace_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    run_live_runtime(
        default_test_config(args, num_workers, router_mode),
        pending,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
}

#[cfg(test)]
pub(super) fn simulate_concurrency_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    run_live_runtime(
        default_test_config(args, num_workers, router_mode),
        pending,
        LiveReplayMode::Concurrency { max_in_flight },
        CancellationToken::new(),
    )
}

#[cfg(test)]
pub(super) fn simulate_trace_workload_with_stats(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let total_turns = total_turns(&trace);
    run_live_workload_runtime(
        default_test_config(args, num_workers, router_mode),
        trace.into_trace_driver_with_block_size(engine_block_size)?,
        total_turns,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
}

#[cfg(test)]
pub(super) fn simulate_concurrency_workload_with_stats(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let total_turns = total_turns(&trace);
    run_live_workload_runtime(
        default_test_config(args, num_workers, router_mode),
        trace.into_concurrency_driver_with_block_size(engine_block_size, max_in_flight)?,
        total_turns,
        LiveReplayMode::Concurrency { max_in_flight },
        CancellationToken::new(),
    )
}
