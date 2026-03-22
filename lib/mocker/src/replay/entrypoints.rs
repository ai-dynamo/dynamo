// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use dynamo_kv_router::config::KvRouterConfig;

use super::loader::load_trace_requests;
use super::online;
use super::validate::{
    validate_offline_concurrency_args, validate_offline_replay_args,
    validate_online_concurrency_args, validate_online_replay_args,
};
use super::{ReplayRouterMode, TraceSimulationReport};
use crate::common::protocols::{DirectRequest, MockEngineArgs};

pub fn simulate_trace_file(
    args: MockEngineArgs,
    trace_path: &Path,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_file_with_router_mode(
        args,
        None,
        trace_path,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace_path: &Path,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    let requests = load_trace_requests(trace_path, args.block_size, true)?;
    let started_at = Instant::now();
    let report = crate::replay::offline::simulate_trace(
        args,
        router_config,
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
    )?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
}

pub fn simulate_trace_live_file(
    args: MockEngineArgs,
    trace_path: &Path,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_file_with_router_mode(
        args,
        None,
        trace_path,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_live_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace_path: &Path,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    let requests = load_trace_requests(trace_path, args.block_size, true)?;
    online::simulate_trace_requests(
        args,
        router_config,
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
    )
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
        requests,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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

    let started_at = Instant::now();
    let report = crate::replay::offline::simulate_trace(
        args,
        router_config,
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
    )?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
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
        requests,
        num_workers,
        arrival_speedup_ratio,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_live_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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
        requests,
        num_workers,
        arrival_speedup_ratio,
        router_mode,
    )
}

pub fn simulate_concurrency_file(
    args: MockEngineArgs,
    trace_path: &Path,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_file_with_router_mode(
        args,
        None,
        trace_path,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace_path: &Path,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let requests = load_trace_requests(trace_path, args.block_size, false)?;
    let started_at = Instant::now();
    let report = simulate_concurrency_requests_with_router_mode(
        args,
        router_config,
        requests,
        max_in_flight,
        num_workers,
        router_mode,
    )?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
}

pub fn simulate_concurrency_live_file(
    args: MockEngineArgs,
    trace_path: &Path,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_file_with_router_mode(
        args,
        None,
        trace_path,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_live_file_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace_path: &Path,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_concurrency_args(&args, num_workers, max_in_flight)?;
    let requests = load_trace_requests(trace_path, args.block_size, false)?;
    online::simulate_concurrency_requests(
        args,
        router_config,
        requests,
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
        requests,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_live_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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
        requests,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_requests_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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
        requests,
        max_in_flight,
        num_workers,
        router_mode,
    )
}
