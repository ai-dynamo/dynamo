// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public handle for driving an offline aggregated replay with planner-in-the-loop.
//!
//! The [`PlannerReplayHandle`] wraps an `AggRuntime` and exposes a step-based API
//! that allows the Python planner adapter to:
//!
//! 1. Advance the simulation to a given simulated time
//! 2. Collect forward pass metrics and traffic observations
//! 3. Apply scaling decisions (add/remove workers)
//! 4. Finalize and retrieve the trace report

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;

use super::offline::agg::AggRuntime;
use super::offline::components::ReplayMode;
use super::{ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport};
use crate::common::protocols::{ForwardPassSnapshot, MockEngineArgs};
use crate::loadgen::Trace;

/// Snapshot of metrics collected between planner ticks.
pub struct PlannerTickData {
    /// Current simulated time in milliseconds.
    pub now_ms: f64,
    /// Whether the replay has finished (no more work).
    pub is_done: bool,
    /// Forward pass metrics per worker since last tick: (worker_id, snapshot).
    pub fpm_snapshots: Vec<(usize, ForwardPassSnapshot)>,
    /// Traffic observation: (duration_s, num_req, avg_isl, avg_osl).
    pub traffic: (f64, usize, f64, f64),
    /// Number of active workers (not pending removal).
    pub active_worker_count: usize,
    /// Total worker count (including pending removal).
    pub total_worker_count: usize,
}

pub struct PlannerReplayHandle {
    runtime: AggRuntime,
    started_at: Instant,
}

impl PlannerReplayHandle {
    /// Create a handle for a trace-file-based replay with planner integration.
    pub fn from_trace_file(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace_path: &Path,
        trace_block_size: usize,
        num_workers: usize,
        arrival_speedup_ratio: f64,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        let args = args.normalized()?;
        let trace = Trace::from_mooncake(trace_path, trace_block_size)?
            .normalize_session_starts()?
            .speed_up_timing(arrival_speedup_ratio)?;
        let runtime = AggRuntime::new_workload(
            &args,
            router_config,
            prefill_load_estimator,
            trace.into_trace_driver_with_block_size(args.block_size)?,
            num_workers,
            ReplayMode::Trace,
            router_mode,
        )?;
        Ok(Self {
            runtime,
            started_at: Instant::now(),
        })
    }

    /// Create a handle for a synthetic-trace replay with planner integration.
    pub fn from_synthetic(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        requests: VecDeque<crate::common::protocols::DirectRequest>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        let runtime = AggRuntime::new(
            &args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            ReplayMode::Trace,
            router_mode,
        )?;
        Ok(Self {
            runtime,
            started_at: Instant::now(),
        })
    }

    /// Advance the simulation up to `until_ms`, collect metrics, return tick data.
    pub fn advance_to(&mut self, until_ms: f64) -> Result<PlannerTickData> {
        let is_done = self.runtime.advance_to(until_ms)?;
        let fpm_snapshots = self.runtime.drain_fpm();
        let traffic = self.runtime.drain_traffic();
        Ok(PlannerTickData {
            now_ms: self.runtime.now_ms(),
            is_done,
            fpm_snapshots,
            traffic,
            active_worker_count: self.runtime.active_worker_count(),
            total_worker_count: self.runtime.total_worker_count(),
        })
    }

    /// Apply a scaling decision: set the target number of workers.
    pub fn apply_scaling(&mut self, target_workers: usize) -> Result<()> {
        self.runtime.apply_scaling(target_workers)
    }

    /// Finalize the replay and return the report.
    pub fn finalize(self) -> TraceSimulationReport {
        let report: TraceSimulationReport = self.runtime.finalize_report();
        report.with_wall_time_ms(self.started_at.elapsed().as_secs_f64() * 1000.0)
    }
}
