// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::{Result, anyhow, bail};
use dynamo_kv_router::config::KvRouterConfig;

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::{Trace, WorkloadDriver};
use crate::replay::{ReplayRouterMode, TraceSimulationReport, normalize_trace_requests};

use super::live_runtime::LiveRuntime;
use super::state::{LiveReplayMode, LiveRuntimeStats};

fn total_turns(trace: &Trace) -> usize {
    trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum()
}

fn run_live_runtime(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    pending: VecDeque<DirectRequest>,
    num_workers: usize,
    mode: LiveReplayMode,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(args, router_config, pending, num_workers, mode, router_mode)?
            .run()
            .await
    })
}

fn run_live_workload_runtime(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    driver: WorkloadDriver,
    total_turns: usize,
    num_workers: usize,
    mode: LiveReplayMode,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(
            args,
            router_config,
            VecDeque::new(),
            num_workers,
            mode,
            router_mode,
        )?
        .run_workload(driver, total_turns)
        .await
    })
}

pub(crate) fn simulate_trace_requests(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (report, _) = run_live_runtime(
        args,
        router_config,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_requests(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    if requests.is_empty() {
        bail!("online concurrency replay requires at least one request");
    }

    let pending = VecDeque::from(requests);
    let (report, _) = run_live_runtime(
        args,
        router_config,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let total_turns = total_turns(&trace);
    let (report, _) = run_live_workload_runtime(
        args,
        router_config,
        trace.into_trace_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let total_turns = total_turns(&trace);
    let (report, _) = run_live_workload_runtime(
        args,
        router_config,
        trace.into_concurrency_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?;
    Ok(report)
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
        args,
        None,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
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
        args,
        None,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
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
    let total_turns = total_turns(&trace);
    run_live_workload_runtime(
        args,
        None,
        trace.into_trace_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
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
    let total_turns = total_turns(&trace);
    run_live_workload_runtime(
        args,
        None,
        trace.into_concurrency_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
}
