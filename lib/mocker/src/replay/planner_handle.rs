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
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, SlaThresholds,
    TraceSimulationReport,
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

/// An optional in-flight cap -> replay mode. `Some(n)` runs **closed-loop**
/// (cap n requests in flight, trace timestamps ignored); `None` replays at the
/// trace's arrival timestamps. Both work with the planner advance/scaling loop,
/// which is mode-agnostic.
fn replay_mode(max_in_flight: Option<usize>) -> Result<ReplayMode> {
    match max_in_flight {
        Some(0) => anyhow::bail!("max_in_flight must be at least 1"),
        Some(max_in_flight) => Ok(ReplayMode::Concurrency { max_in_flight }),
        None => Ok(ReplayMode::Trace),
    }
}

/// Load + normalize a Mooncake trace. The arrival speedup only matters in
/// arrival mode — closed-loop replay ignores the trace's timestamps.
fn prepare_mooncake_trace(
    trace_path: &Path,
    trace_block_size: usize,
    arrival_speedup_ratio: f64,
    max_in_flight: Option<usize>,
) -> Result<Trace> {
    let trace = Trace::from_mooncake(trace_path, trace_block_size)?.normalize_session_starts()?;
    if max_in_flight.is_none() {
        Ok(trace.speed_up_timing(arrival_speedup_ratio)?)
    } else {
        Ok(trace)
    }
}

impl PlannerReplayHandle {
    /// Build an aggregated handle from an **already-prepared** workload trace.
    ///
    /// Trace preparation is the caller's job: Mooncake callers normalize session
    /// starts and (in arrival mode) speed up timing; synthetic callers build the
    /// trace at the target rate. `max_in_flight = Some(n)` drives the replay
    /// closed-loop (cap n in flight, timestamps ignored); `None` replays at
    /// arrival timestamps. This is the source-agnostic seam used by both the
    /// trace-file and synthetic-workload entrypoints.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trace(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace: Trace,
        num_workers: usize,
        max_in_flight: Option<usize>,
        router_mode: ReplayRouterMode,
        sla: SlaThresholds,
    ) -> Result<Self> {
        let args = args.normalized()?;
        let runtime = AggRuntime::new_workload(
            &args,
            router_config,
            prefill_load_estimator,
            trace.into_trace_driver_with_block_size(args.block_size)?,
            num_workers,
            replay_mode(max_in_flight)?,
            router_mode,
        )?
        .with_sla_thresholds(sla);
        Ok(Self {
            runtime: RuntimeKind::Agg(runtime),
            started_at: Instant::now(),
        })
    }

    /// Create a handle for an aggregated Mooncake-style trace-file replay.
    /// `max_in_flight = Some(n)` runs closed-loop; `None` uses arrival timestamps.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trace_file(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace_path: &Path,
        trace_block_size: usize,
        num_workers: usize,
        arrival_speedup_ratio: f64,
        max_in_flight: Option<usize>,
        router_mode: ReplayRouterMode,
        sla: SlaThresholds,
    ) -> Result<Self> {
        let trace = prepare_mooncake_trace(
            trace_path,
            trace_block_size,
            arrival_speedup_ratio,
            max_in_flight,
        )?;
        Self::from_trace(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            max_in_flight,
            router_mode,
            sla,
        )
    }

    /// Build a disaggregated handle from an **already-prepared** workload trace.
    /// See [`PlannerReplayHandle::from_trace`] for the source-agnostic contract.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trace_disagg(
        config: OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace: Trace,
        max_in_flight: Option<usize>,
        router_mode: ReplayRouterMode,
        sla: SlaThresholds,
    ) -> Result<Self> {
        let config = config.normalized()?;
        let runtime = DisaggRuntime::new_workload(
            &config,
            router_config,
            prefill_load_estimator,
            trace.into_trace_driver_with_block_size(config.decode_args.block_size)?,
            replay_mode(max_in_flight)?,
            router_mode,
        )?
        .with_sla_thresholds(sla);
        Ok(Self {
            runtime: RuntimeKind::Disagg(runtime),
            started_at: Instant::now(),
        })
    }

    /// Create a handle for a disaggregated Mooncake-style trace-file replay.
    /// `max_in_flight = Some(n)` runs closed-loop; `None` uses arrival timestamps.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trace_file_disagg(
        config: OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        trace_path: &Path,
        trace_block_size: usize,
        arrival_speedup_ratio: f64,
        max_in_flight: Option<usize>,
        router_mode: ReplayRouterMode,
        sla: SlaThresholds,
    ) -> Result<Self> {
        let trace = prepare_mooncake_trace(
            trace_path,
            trace_block_size,
            arrival_speedup_ratio,
            max_in_flight,
        )?;
        Self::from_trace_disagg(
            config,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            router_mode,
            sla,
        )
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
    /// Call this only on throughput-scaling ticks so the window covers the
    /// full `throughput_adjustment_interval`, not just the gap between load
    /// ticks. The returned [`TrafficStats::avg_kv_hit_rate`] is the
    /// arithmetic mean of per-request ``overlap / isl`` ratios across
    /// admissions in the window — matching the real router's per-request
    /// Prometheus histogram, where each request contributes one sample
    /// regardless of ISL size.
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
        let wall_time_ms = self.started_at.elapsed().as_secs_f64() * 1000.0;
        let report = match self.runtime {
            RuntimeKind::Agg(rt) => rt.finalize_report(),
            RuntimeKind::Disagg(rt) => rt.finalize_report(),
        };
        report.with_wall_time_ms(wall_time_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::PlannerReplayHandle;
    use crate::common::protocols::MockEngineArgs;
    use crate::loadgen::{ArrivalSpec, DelaySpec, LengthSpec, SyntheticTraceSpec, Trace};
    use crate::replay::{ReplayRouterMode, SlaThresholds};

    const NUM_SESSIONS: usize = 8;

    fn small_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap()
    }

    fn synthetic_trace(first_turn_arrivals: ArrivalSpec) -> Trace {
        Trace::synthetic(SyntheticTraceSpec {
            block_size: 4,
            num_sessions: NUM_SESSIONS,
            turns_per_session: 1,
            input_tokens: LengthSpec {
                mean: 8,
                stddev: 0.0,
            },
            output_tokens: LengthSpec {
                mean: 4,
                stddev: 0.0,
            },
            shared_prefix_ratio: 0.0,
            num_prefix_groups: 0,
            first_turn_arrivals,
            inter_turn_delays: DelaySpec::None,
            seed: 42,
        })
        .unwrap()
    }

    /// One large advance drains every event (arrival timestamps and concurrency
    /// admissions alike), so the planner loop is mode-agnostic.
    fn drive_to_completion(handle: &mut PlannerReplayHandle) {
        let tick = handle.advance_to(1.0e15).unwrap();
        assert!(
            tick.is_done,
            "replay should finish within the advance window"
        );
    }

    #[test]
    fn from_trace_closed_loop_completes_all_requests() {
        // Burst arrivals + an in-flight cap -> closed-loop: trace timestamps are
        // ignored and at most `max_in_flight` run at once, but every request still
        // completes. This is the planner + concurrency path that was previously
        // unreachable (the handle hard-coded ReplayMode::Trace).
        let mut handle = PlannerReplayHandle::from_trace(
            small_args(),
            None,
            None,
            synthetic_trace(ArrivalSpec::Burst),
            1,
            Some(2),
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap();
        drive_to_completion(&mut handle);
        assert_eq!(
            handle.finalize().request_counts.completed_requests,
            NUM_SESSIONS
        );
    }

    #[test]
    fn from_trace_arrival_completes_all_requests() {
        // No cap -> arrival-timestamp (open-loop) replay; every request completes.
        let mut handle = PlannerReplayHandle::from_trace(
            small_args(),
            None,
            None,
            synthetic_trace(ArrivalSpec::ConstantQps { qps: 1000.0 }),
            1,
            None,
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap();
        drive_to_completion(&mut handle);
        assert_eq!(
            handle.finalize().request_counts.completed_requests,
            NUM_SESSIONS
        );
    }
}
