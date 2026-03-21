// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use super::TraceSimulationReport;
use super::loader::load_trace_requests;
use super::validate::{validate_offline_concurrency_args, validate_offline_replay_args};
use crate::common::protocols::{DirectRequest, MockEngineArgs};

pub fn simulate_trace_file(
    args: MockEngineArgs,
    trace_path: &Path,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    validate_offline_replay_args(&args, num_workers)?;
    let requests = load_trace_requests(trace_path, args.block_size, true)?;
    let started_at = Instant::now();
    let report = crate::replay::runtime::multi::simulate_trace(args, requests, num_workers)?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
}

pub fn simulate_concurrency_file(
    args: MockEngineArgs,
    trace_path: &Path,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    let requests = load_trace_requests(trace_path, args.block_size, false)?;
    let started_at = Instant::now();
    let report = simulate_concurrency_requests(args, requests, max_in_flight, num_workers)?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
}

pub fn simulate_concurrency_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    validate_offline_concurrency_args(&args, num_workers, max_in_flight)?;
    if requests.is_empty() {
        bail!("concurrency replay requires at least one request");
    }

    crate::replay::runtime::multi::simulate_concurrency(args, requests, max_in_flight, num_workers)
}
