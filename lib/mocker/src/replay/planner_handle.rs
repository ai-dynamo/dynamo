// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public handle for driving an offline replay with planner-in-the-loop.
//!
//! Supports both aggregated and disaggregated topologies via [`RuntimeKind`].
//! The Python planner adapter calls [`PlannerReplayHandle::advance_to`] to
//! step the simulation, collects metrics, and calls [`PlannerReplayHandle::apply_scaling`]
//! to resize worker pools.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;

use super::offline::agg::AggRuntime;
use super::offline::components::{ReplayMode, TrafficStats};
use super::offline::disagg::DisaggRuntime;
use super::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport,
};
use crate::common::protocols::{ForwardPassSnapshot, MockEngineArgs};
use crate::loadgen::Trace;

/// Snapshot of metrics collected between planner ticks.
///
/// For aggregated mode, prefill fields are 0 and all data is in decode fields
/// (matching how the planner treats agg as a single decode-stage engine).
///
/// Traffic metrics are NOT included here — they accumulate across ticks and
/// must be drained explicitly via [`PlannerReplayHandle::drain_traffic`] on
/// throughput-scaling ticks only. Draining on every tick would discard data
/// between the more frequent load-scaling ticks.
pub struct PlannerTickData {
    /// Current simulated time in milliseconds.
    pub now_ms: f64,
    /// Whether the replay has finished (no more work).
    pub is_done: bool,
    /// Prefill FPM snapshots since last tick: (worker_id, snapshot).
    pub prefill_fpm_snapshots: Vec<(usize, ForwardPassSnapshot)>,
    /// Decode (or agg) FPM snapshots since last tick: (worker_id, snapshot).
    pub decode_fpm_snapshots: Vec<(usize, ForwardPassSnapshot)>,
    /// Active prefill workers (0 for agg mode).
    pub active_prefill_count: usize,
    /// Active decode workers (or total active for agg mode).
    pub active_decode_count: usize,
    /// Total prefill workers including pending removal (0 for agg mode).
    pub total_prefill_count: usize,
    /// Total decode workers including pending removal (or total for agg mode).
    pub total_decode_count: usize,
}

#[allow(clippy::large_enum_variant)]
enum RuntimeKind {
    Agg(AggRuntime),
    Disagg(DisaggRuntime),
}

pub struct PlannerReplayHandle {
    runtime: RuntimeKind,
    started_at: Instant,
}

impl PlannerReplayHandle {
    /// Create a handle for an aggregated trace-file replay.
    #[allow(clippy::too_many_arguments)]
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
            runtime: RuntimeKind::Agg(runtime),
            started_at: Instant::now(),
        })
    }

    /// Create a handle for a disaggregated trace-file replay.
    pub fn from_trace_file_disagg(
        config: OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace_path: &Path,
        trace_block_size: usize,
        arrival_speedup_ratio: f64,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        let config = config.normalized()?;
        let trace = Trace::from_mooncake(trace_path, trace_block_size)?
            .normalize_session_starts()?
            .speed_up_timing(arrival_speedup_ratio)?;
        let runtime = DisaggRuntime::new_workload(
            &config,
            router_config,
            prefill_load_estimator,
            trace.into_trace_driver_with_block_size(config.decode_args.block_size)?,
            ReplayMode::Trace,
            router_mode,
        )?;
        Ok(Self {
            runtime: RuntimeKind::Disagg(runtime),
            started_at: Instant::now(),
        })
    }

    /// Advance the simulation up to `until_ms`, collect metrics, return tick data.
    ///
    /// Traffic metrics are NOT drained here — call [`drain_traffic`] explicitly
    /// on throughput-scaling ticks so the accumulator covers the full interval.
    pub fn advance_to(&mut self, until_ms: f64) -> Result<PlannerTickData> {
        match &mut self.runtime {
            RuntimeKind::Agg(rt) => {
                let is_done = rt.advance_to(until_ms)?;
                let fpm = rt.drain_fpm();
                Ok(PlannerTickData {
                    now_ms: rt.now_ms(),
                    is_done,
                    prefill_fpm_snapshots: Vec::new(),
                    decode_fpm_snapshots: fpm,
                    active_prefill_count: 0,
                    active_decode_count: rt.active_worker_count(),
                    total_prefill_count: 0,
                    total_decode_count: rt.total_worker_count(),
                })
            }
            RuntimeKind::Disagg(rt) => {
                let is_done = rt.advance_to(until_ms)?;
                let prefill_fpm = rt.drain_prefill_fpm();
                let decode_fpm = rt.drain_decode_fpm();
                Ok(PlannerTickData {
                    now_ms: rt.now_ms(),
                    is_done,
                    prefill_fpm_snapshots: prefill_fpm,
                    decode_fpm_snapshots: decode_fpm,
                    active_prefill_count: rt.active_prefill_count(),
                    active_decode_count: rt.active_decode_count(),
                    total_prefill_count: rt.total_prefill_count(),
                    total_decode_count: rt.total_decode_count(),
                })
            }
        }
    }

    /// Drain accumulated traffic metrics since the last drain.
    ///
    /// Call this only on throughput-scaling ticks so the window covers the full
    /// `throughput_adjustment_interval`, not just the gap between load ticks.
    pub fn drain_traffic(&mut self) -> TrafficStats {
        match &mut self.runtime {
            RuntimeKind::Agg(rt) => rt.drain_traffic(),
            RuntimeKind::Disagg(rt) => rt.drain_traffic(),
        }
    }

    /// Apply a scaling decision with separate prefill and decode targets.
    /// For agg mode, `target_prefill` is ignored.
    pub fn apply_scaling(&mut self, target_prefill: usize, target_decode: usize) -> Result<()> {
        match &mut self.runtime {
            RuntimeKind::Agg(rt) => rt.apply_scaling(target_decode),
            RuntimeKind::Disagg(rt) => rt.apply_scaling(target_prefill, target_decode),
        }
    }

    /// Finalize the replay and return the report.
    pub fn finalize(self) -> TraceSimulationReport {
        let report = match self.runtime {
            RuntimeKind::Agg(rt) => rt.finalize_report(),
            RuntimeKind::Disagg(rt) => rt.finalize_report(),
        };
        report.with_wall_time_ms(self.started_at.elapsed().as_secs_f64() * 1000.0)
    }
}
