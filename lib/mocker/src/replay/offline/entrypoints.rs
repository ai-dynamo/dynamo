// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;

#[cfg(test)]
use super::agg::AggRuntimeStats;
use super::agg::{AggRuntime, ReplayMode as AggReplayMode};
#[cfg(test)]
use super::disagg::DisaggRuntimeStats;
use super::disagg::{DisaggRuntime, ReplayMode as DisaggReplayMode};
use super::normalize_trace_requests;
use super::single::{SingleReplayMode, SingleRuntime};
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs};
use crate::loadgen::{Trace, WorkloadDriver};
use crate::replay::OfflineDisaggReplayConfig;
#[cfg(test)]
use crate::replay::TraceCollector;
use crate::replay::{ReplayRouterMode, TraceSimulationReport};

pub(crate) fn simulate_trace(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_trace_single(args, requests, arrival_speedup_ratio)
    } else {
        simulate_trace_multi(
            args,
            router_config,
            requests,
            num_workers,
            arrival_speedup_ratio,
            router_mode,
        )
    }
}

pub(crate) fn simulate_concurrency(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_concurrency_single(args, requests, max_in_flight)
    } else {
        simulate_concurrency_multi(
            args,
            router_config,
            requests,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}

pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_trace_workload_single(args, trace)
    } else {
        simulate_trace_workload_multi(args, router_config, trace, num_workers, router_mode)
    }
}

pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_concurrency_workload_single(args, trace, max_in_flight)
    } else {
        simulate_concurrency_workload_multi(
            args,
            router_config,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}

pub(crate) fn simulate_trace_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = DisaggRuntime::new(
        &config,
        router_config,
        pending,
        DisaggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let pending = VecDeque::from(requests);
    let (collector, _) = DisaggRuntime::new(
        &config,
        router_config,
        pending,
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let driver = WorkloadDriver::new_trace(trace)?;
    let (collector, _) = DisaggRuntime::new_workload(
        &config,
        router_config,
        driver,
        DisaggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let driver = WorkloadDriver::new_concurrency(trace)?;
    let (collector, _) = DisaggRuntime::new_workload(
        &config,
        router_config,
        driver,
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let collector = SingleRuntime::new(args, pending, SingleReplayMode::Trace).run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let collector = SingleRuntime::new(
        args,
        pending,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_single(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let collector =
        SingleRuntime::new_workload(args, trace.into_trace_driver()?, SingleReplayMode::Trace)
            .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_single(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let collector = SingleRuntime::new_workload(
        args,
        trace.into_concurrency_driver()?,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = AggRuntime::new(
        &args,
        router_config,
        pending,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let (collector, _) = AggRuntime::new(
        &args,
        router_config,
        pending,
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let (collector, _) = AggRuntime::new_workload(
        &args,
        router_config,
        trace.into_trace_driver()?,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let (collector, _) = AggRuntime::new_workload(
        &args,
        router_config,
        trace.into_concurrency_driver()?,
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
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
    SingleRuntime::new_workload(
        args,
        trace.into_trace_driver().unwrap(),
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
    SingleRuntime::new_workload(
        args,
        trace.into_concurrency_driver().unwrap(),
        SingleReplayMode::Concurrency { max_in_flight },
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
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new_workload(
        args,
        None,
        trace.into_trace_driver().unwrap(),
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
        trace.into_concurrency_driver().unwrap(),
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
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
    router_config: Option<KvRouterConfig>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio).unwrap();
    DisaggRuntime::new(
        config,
        router_config,
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
    router_config: Option<KvRouterConfig>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new(
        config,
        router_config,
        VecDeque::from(requests),
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}
