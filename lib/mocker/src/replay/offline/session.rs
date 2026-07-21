// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, resumable control surface for Rush Hour's first Dynamo replay slice.
//!
//! The live runtime is the ordinary aggregated offline event loop. Checkpoints
//! capture active scheduler sequences, logical KV ownership, lifecycle sets,
//! collector state, and the ordered future event heap as values. Restore builds
//! fresh physical KV managers; it never replays from time zero and never shares
//! live block handles with the source timeline.

use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::{Result, ensure};
use serde::Serialize;

use super::agg::{AggReplayCheckpoint, AggRuntime};
use super::components::{EngineWorkerLifecycle, ReplayMode};
use crate::common::perf_model::PerfModel;
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, WorkerType};
use crate::loadgen::Trace;
use crate::replay::{
    ReplayRouterMode, TraceCursorMetricValues, TraceSimulationReport, normalize_trace_requests,
};

/// Hard safety bounds for the first interactive replay kernel.
pub const MAX_REPLAY_SESSION_REQUESTS: usize = 100_000;
pub const MAX_REPLAY_SESSION_SCALE_ACTIONS: usize = 4_096;
pub const MAX_REPLAY_SESSION_REPLICAS: usize = 128;
pub const MAX_REPLAY_SESSION_OBSERVATIONS: usize = 100_000;

/// Construction options for the bounded aggregated-vLLM session.
#[derive(Debug, Clone)]
pub struct ReplaySessionConfig {
    pub engine_args: MockEngineArgs,
    pub initial_replicas: usize,
    pub arrival_speedup_ratio: f64,
    pub capture_per_request: bool,
}

impl ReplaySessionConfig {
    pub fn new(engine_args: MockEngineArgs) -> Self {
        Self {
            engine_args,
            initial_replicas: 1,
            arrival_speedup_ratio: 1.0,
            capture_per_request: false,
        }
    }
}

/// One absolute topology mutation, applied after ordinary events at `at_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReplayScaleAction {
    pub at_ms: f64,
    pub target_replicas: usize,
}

/// Observable state at a paused logical cursor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplaySessionState {
    pub cursor_ms: f64,
    pub revision: u64,
    pub workload_done: bool,
    pub serving_replicas: usize,
    pub target_replicas: usize,
    pub provisioned_replicas: usize,
    pub starting_replicas: usize,
    pub draining_replicas: usize,
    pub admitted_requests: usize,
    pub completed_requests: usize,
    pub in_flight_requests: usize,
    pub pending_arrivals: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayReplicaLifecycle {
    Serving,
    Starting,
    Draining,
    Inactive,
}

/// Traffic and latency values for `(window_start_ms, cursor_ms]` (with time
/// zero included when the window starts at zero), plus cumulative counters at
/// the same cursor. Full-telemetry percentiles are `None` when the window has no
/// sample; [`ReplaySession::traffic_telemetry_since`] deliberately leaves every
/// percentile `None` without collecting its underlying distribution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayTelemetryWindow {
    pub window_start_ms: f64,
    pub cursor_ms: f64,
    pub duration_ms: f64,
    pub cumulative_arrivals: usize,
    pub cumulative_admissions: usize,
    pub cumulative_completions: usize,
    pub cumulative_output_tokens: usize,
    pub arrivals: usize,
    pub admissions: usize,
    pub completions: usize,
    pub output_tokens: usize,
    /// Mean simulated prompt length for requests whose trace arrival is in this
    /// window. `None` means the window contained no arrivals.
    pub avg_isl_tokens: Option<f64>,
    /// Mean requested/planned output length over the same arrival cohort as
    /// `avg_isl_tokens`; this is not the number of tokens emitted by the cursor.
    pub avg_requested_osl_tokens: Option<f64>,
    pub request_rate_rps: f64,
    pub output_throughput_tok_s: f64,
    pub request_admission_ratio: Option<f64>,
    pub p95_ttft_ms: Option<f64>,
    pub p95_itl_ms: Option<f64>,
    pub p95_e2e_ms: Option<f64>,
    pub kv_reuse_rate: Option<f64>,
    pub cumulative_kv_reuse_rate: Option<f64>,
    /// The supported slice has no speculative decoder. Once at least one
    /// decode token is visible, its accepted tokens per forward is exactly 1.
    pub avg_accept_length: Option<f64>,
}

impl ReplayTelemetryWindow {
    fn from_cursor_values(
        values: TraceCursorMetricValues,
        window_start_ms: f64,
        cursor_ms: f64,
    ) -> Self {
        let duration_ms = cursor_ms - window_start_ms;
        let duration_s = duration_ms / 1_000.0;
        let request_rate_rps = if duration_s > 0.0 {
            values.window_arrivals as f64 / duration_s
        } else {
            0.0
        };
        let output_throughput_tok_s = if duration_s > 0.0 {
            values.window_output_tokens as f64 / duration_s
        } else {
            0.0
        };
        let request_admission_ratio = (values.window_arrivals > 0)
            .then(|| values.window_arrivals_admitted as f64 / values.window_arrivals as f64);
        Self {
            window_start_ms,
            cursor_ms,
            duration_ms,
            cumulative_arrivals: values.cumulative_arrivals,
            cumulative_admissions: values.cumulative_admissions,
            cumulative_completions: values.cumulative_completions,
            cumulative_output_tokens: values.cumulative_output_tokens,
            arrivals: values.window_arrivals,
            admissions: values.window_admissions,
            completions: values.window_completions,
            output_tokens: values.window_output_tokens,
            avg_isl_tokens: values.avg_isl_tokens,
            avg_requested_osl_tokens: values.avg_requested_osl_tokens,
            request_rate_rps,
            output_throughput_tok_s,
            request_admission_ratio,
            p95_ttft_ms: values.p95_ttft_ms,
            p95_itl_ms: values.p95_itl_ms,
            p95_e2e_ms: values.p95_e2e_ms,
            kv_reuse_rate: values.window_kv_reuse_rate,
            cumulative_kv_reuse_rate: values.cumulative_kv_reuse_rate,
            avg_accept_length: (values.window_decode_tokens > 0).then_some(1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayReplicaTelemetry {
    pub replica_id: usize,
    pub lifecycle: ReplayReplicaLifecycle,
    pub queued_requests: usize,
    pub running_requests: usize,
    pub in_flight_requests: usize,
    pub busy_ranks: usize,
    pub metrics: ReplayTelemetryWindow,
}

/// Non-destructive telemetry at a paused cursor. `traffic_horizon_ms` is the
/// final trace-arrival timestamp; `terminal_horizon_ms` becomes known only when
/// every request has completed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplaySessionTelemetry {
    pub cursor_ms: f64,
    pub traffic_horizon_ms: f64,
    pub terminal_horizon_ms: Option<f64>,
    pub workload_done: bool,
    pub queue_depth: usize,
    pub running_requests: usize,
    pub active_replicas: usize,
    pub provisioned_replicas: usize,
    pub metrics: ReplayTelemetryWindow,
    pub replicas: Vec<ReplayReplicaTelemetry>,
}

/// One point emitted by [`ReplaySession::advance_sampled`]. The final target
/// always carries telemetry; an interval checkpoint is present only when the
/// point lies on the requested absolute checkpoint grid.
#[derive(Debug, Clone)]
pub struct ReplayTimelineObservation {
    pub at_ms: f64,
    pub telemetry: Option<ReplaySessionTelemetry>,
    pub checkpoint: Option<ReplaySessionCheckpoint>,
}

#[derive(Debug, Clone)]
pub struct ReplaySampledAdvance {
    pub final_state: ReplaySessionState,
    pub observations: Vec<ReplayTimelineObservation>,
}

/// State and one telemetry window captured atomically after an incremental
/// advance. This is the inexpensive retained-future path for consumers that
/// do not need to materialize intermediate timeline observations.
#[derive(Debug, Clone)]
pub struct ReplayAdvanceSnapshot {
    pub state: ReplaySessionState,
    pub telemetry: ReplaySessionTelemetry,
}

/// Restore mechanism selected for a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCheckpointKind {
    /// A deep, value-only active runtime memento.
    DeepRuntimeMemento,
}

/// Reusable checkpoint. Clones share only immutable seed data; restored
/// schedulers and physical KV pools are always freshly constructed.
#[derive(Debug, Clone)]
pub struct ReplaySessionCheckpoint {
    actions: Vec<ReplayScaleAction>,
    cursor_ms: f64,
    traffic_horizon_ms: f64,
    runtime: AggReplayCheckpoint,
}

impl ReplaySessionCheckpoint {
    pub fn cursor_ms(&self) -> f64 {
        self.cursor_ms
    }

    pub fn revision(&self) -> u64 {
        self.actions.len() as u64
    }

    pub fn kind(&self) -> ReplayCheckpointKind {
        ReplayCheckpointKind::DeepRuntimeMemento
    }

    pub fn restore(&self) -> Result<ReplaySession> {
        ReplaySession::restore(self)
    }
}

/// Incrementally driven aggregated replay session for the first Rush Hour
/// integration slice.
struct ReplaySessionCore {
    runtime: AggRuntime,
    actions: Vec<ReplayScaleAction>,
    traffic_horizon_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayTelemetryDetail {
    Full,
    TrafficOnly,
}

impl ReplaySessionCore {
    /// Construct a trace-mode, round-robin session with a bounded initial
    /// replica count.
    ///
    /// The first slice supports aggregated vLLM, DP=1, prefix caching, exact
    /// planned output IDs, no speculative decode, and no KVBM offload. Scaling
    /// may add homogeneous workers through the runtime's existing lifecycle
    /// machinery.
    fn new(config: ReplaySessionConfig, requests: Vec<DirectRequest>) -> Result<Self> {
        ensure!(
            requests.len() <= MAX_REPLAY_SESSION_REQUESTS,
            "replay session has {} requests; limit is {MAX_REPLAY_SESSION_REQUESTS}",
            requests.len()
        );
        ensure!(
            (1..=MAX_REPLAY_SESSION_REPLICAS).contains(&config.initial_replicas),
            "replay session initial replicas must be in 1..={MAX_REPLAY_SESSION_REPLICAS}; got {}",
            config.initial_replicas
        );
        ensure!(
            !requests.is_empty(),
            "replay session requires at least one request"
        );

        for (index, request) in requests.iter().enumerate() {
            let arrival_ms = request.arrival_timestamp_ms.ok_or_else(|| {
                anyhow::anyhow!("replay session request {index} is missing arrival_timestamp_ms")
            })?;
            ensure!(
                arrival_ms.is_finite(),
                "replay session request {index} has non-finite arrival_timestamp_ms"
            );
            ensure!(
                request.uuid.is_some(),
                "replay session request {index} requires an explicit UUID"
            );
            let output_ids = request.output_token_ids.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "replay session request {index} requires exact planned output token IDs"
                )
            })?;
            ensure!(
                output_ids.len() == request.max_output_tokens,
                "replay session request {index} planned {} output IDs but max_output_tokens is {}",
                output_ids.len(),
                request.max_output_tokens
            );
        }
        let unique_ids = requests
            .iter()
            .filter_map(|request| request.uuid)
            .collect::<HashSet<_>>();
        ensure!(
            unique_ids.len() == requests.len(),
            "replay session request UUIDs must be unique"
        );

        let engine_args = config.engine_args.clone().normalized()?;
        Self::validate_engine_args(&engine_args)?;
        let normalized_requests = normalize_trace_requests(requests, config.arrival_speedup_ratio)?;
        let traffic_horizon_ms = normalized_requests
            .back()
            .and_then(|request| request.arrival_timestamp_ms)
            .unwrap_or_default();
        let runtime = AggRuntime::new_checkpointable(
            &engine_args,
            None,
            None,
            normalized_requests,
            config.initial_replicas,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )?
        .with_per_request_records(config.capture_per_request);
        Ok(Self {
            runtime,
            actions: Vec::new(),
            traffic_horizon_ms,
        })
    }

    fn validate_engine_args(args: &MockEngineArgs) -> Result<()> {
        ensure!(
            args.engine_type == EngineType::Vllm,
            "replay session supports only the vLLM scheduler"
        );
        ensure!(
            args.worker_type == WorkerType::Aggregated,
            "replay session supports only aggregated workers"
        );
        ensure!(
            args.dp_size == 1,
            "replay session supports exactly one DP rank per worker"
        );
        ensure!(
            args.enable_prefix_caching,
            "replay session requires prefix caching"
        );
        ensure!(
            args.aic_nextn.is_none(),
            "replay session does not yet checkpoint speculative decoding"
        );
        ensure!(
            !matches!(args.perf_model.as_ref(), PerfModel::Aiconfigurator { .. }),
            "replay session does not checkpoint stateful AI Configurator callbacks"
        );
        ensure!(
            args.num_g2_blocks.is_none() && args.num_g3_blocks.is_none() && !args.enable_g4_storage,
            "replay session does not yet checkpoint KVBM offload tiers"
        );
        Ok(())
    }

    /// Restore a checkpoint into independent scheduler and KV ownership.
    fn restore(checkpoint: &ReplaySessionCheckpoint) -> Result<Self> {
        Ok(Self {
            runtime: AggRuntime::restore_replay(checkpoint.runtime.clone())?,
            actions: checkpoint.actions.clone(),
            traffic_horizon_ms: checkpoint.traffic_horizon_ms,
        })
    }

    /// Advance monotonically to `until_ms` and pause. Ordinary events at the
    /// target timestamp are fully settled before this returns.
    fn advance_to(&mut self, until_ms: f64) -> Result<ReplaySessionState> {
        self.runtime.advance_until_pause(until_ms)?;
        Ok(self.state())
    }

    fn advance_to_with_telemetry(
        &mut self,
        until_ms: f64,
        window_start_ms: f64,
    ) -> Result<ReplayAdvanceSnapshot> {
        self.runtime.advance_until_pause(until_ms)?;
        Ok(ReplayAdvanceSnapshot {
            state: self.state(),
            telemetry: self.telemetry_since(window_start_ms)?,
        })
    }

    /// Apply an absolute replica target at the current paused cursor.
    fn set_target_replicas(&mut self, target_replicas: usize) -> Result<ReplaySessionState> {
        ensure!(
            (1..=MAX_REPLAY_SESSION_REPLICAS).contains(&target_replicas),
            "replay session target replicas must be in 1..={MAX_REPLAY_SESSION_REPLICAS}; got {target_replicas}"
        );
        ensure!(
            self.actions.len() < MAX_REPLAY_SESSION_SCALE_ACTIONS,
            "replay session scale-action limit {MAX_REPLAY_SESSION_SCALE_ACTIONS} reached"
        );
        let at_ms = self.runtime.now_ms();
        // This is idempotent when the cursor was already settled, and ensures a
        // scale action is ordered after ordinary events at the same timestamp.
        self.runtime.advance_until_pause(at_ms)?;
        ensure!(
            !self.runtime.workload_done(),
            "cannot scale a completed replay session"
        );
        ensure!(
            self.runtime.starting_worker_count() == 0 && self.runtime.draining_worker_count() == 0,
            "cannot retarget replicas while a startup or drain transition is in progress"
        );
        ensure!(
            self.actions
                .last()
                .is_none_or(|action| action.at_ms != at_ms),
            "replay session supports at most one scale action at a logical timestamp"
        );
        ensure!(
            target_replicas != self.runtime.target_worker_count(),
            "replay session replica target is already {target_replicas}"
        );

        self.runtime.apply_scaling(target_replicas)?;
        self.actions.push(ReplayScaleAction {
            at_ms,
            target_replicas,
        });
        Ok(self.state())
    }

    fn state(&self) -> ReplaySessionState {
        let (admitted, completed, in_flight, pending_arrivals) = self.runtime.request_counts();
        ReplaySessionState {
            cursor_ms: self.runtime.now_ms(),
            revision: self.actions.len() as u64,
            workload_done: self.runtime.workload_done(),
            serving_replicas: self.runtime.serving_worker_count(),
            target_replicas: self.runtime.target_worker_count(),
            provisioned_replicas: self.runtime.total_worker_count(),
            starting_replicas: self.runtime.starting_worker_count(),
            draining_replicas: self.runtime.draining_worker_count(),
            admitted_requests: admitted,
            completed_requests: completed,
            in_flight_requests: in_flight,
            pending_arrivals,
        }
    }

    fn telemetry_since(&self, window_start_ms: f64) -> Result<ReplaySessionTelemetry> {
        self.telemetry_since_with_detail(window_start_ms, ReplayTelemetryDetail::Full)
    }

    fn traffic_telemetry_since(&self, window_start_ms: f64) -> Result<ReplaySessionTelemetry> {
        self.telemetry_since_with_detail(window_start_ms, ReplayTelemetryDetail::TrafficOnly)
    }

    fn telemetry_since_with_detail(
        &self,
        window_start_ms: f64,
        detail: ReplayTelemetryDetail,
    ) -> Result<ReplaySessionTelemetry> {
        let cursor_ms = self.runtime.now_ms();
        ensure!(
            window_start_ms.is_finite() && window_start_ms >= 0.0,
            "telemetry window start must be a finite non-negative time"
        );
        ensure!(
            window_start_ms <= cursor_ms,
            "telemetry window start {window_start_ms}ms is after cursor {cursor_ms}ms"
        );
        let (cursor_metrics, worker_snapshots) = match detail {
            ReplayTelemetryDetail::Full => self.runtime.cursor_metrics(window_start_ms),
            ReplayTelemetryDetail::TrafficOnly => {
                self.runtime.traffic_cursor_metrics(window_start_ms)
            }
        };
        let queue_depth = self.runtime.router_queue_count()
            + worker_snapshots
                .iter()
                .map(|worker| worker.queued_requests)
                .sum::<usize>();
        let running_requests = worker_snapshots
            .iter()
            .map(|worker| worker.running_requests)
            .sum();
        let replicas = worker_snapshots
            .into_iter()
            .map(|worker| {
                let values = cursor_metrics
                    .per_worker
                    .get(&worker.worker_id)
                    .cloned()
                    .unwrap_or_default();
                ReplayReplicaTelemetry {
                    replica_id: worker.worker_id,
                    lifecycle: match worker.lifecycle {
                        EngineWorkerLifecycle::Serving => ReplayReplicaLifecycle::Serving,
                        EngineWorkerLifecycle::Starting => ReplayReplicaLifecycle::Starting,
                        EngineWorkerLifecycle::Draining => ReplayReplicaLifecycle::Draining,
                        EngineWorkerLifecycle::Inactive => ReplayReplicaLifecycle::Inactive,
                    },
                    queued_requests: worker.queued_requests,
                    running_requests: worker.running_requests,
                    in_flight_requests: worker.in_flight_requests,
                    busy_ranks: worker.busy_ranks,
                    metrics: ReplayTelemetryWindow::from_cursor_values(
                        values,
                        window_start_ms,
                        cursor_ms,
                    ),
                }
            })
            .collect();
        let workload_done = self.runtime.workload_done();
        Ok(ReplaySessionTelemetry {
            cursor_ms,
            traffic_horizon_ms: self.traffic_horizon_ms,
            terminal_horizon_ms: self.runtime.workload_terminal_ms(),
            workload_done,
            queue_depth,
            running_requests,
            active_replicas: self.runtime.serving_worker_count(),
            provisioned_replicas: self.runtime.total_worker_count(),
            metrics: ReplayTelemetryWindow::from_cursor_values(
                cursor_metrics.aggregate,
                window_start_ms,
                cursor_ms,
            ),
            replicas,
        })
    }

    fn advance_sampled(
        &mut self,
        until_ms: f64,
        telemetry_interval_ms: f64,
        checkpoint_interval_ms: Option<f64>,
    ) -> Result<ReplaySampledAdvance> {
        ensure!(
            until_ms.is_finite() && until_ms >= self.runtime.now_ms(),
            "sampled replay target must be finite and at or after the current cursor"
        );
        ensure!(
            telemetry_interval_ms.is_finite() && telemetry_interval_ms > 0.0,
            "telemetry interval must be a finite positive duration"
        );
        if let Some(interval_ms) = checkpoint_interval_ms {
            ensure!(
                interval_ms.is_finite() && interval_ms > 0.0,
                "checkpoint interval must be a finite positive duration"
            );
        }

        let start_ms = self.runtime.now_ms();
        if until_ms == start_ms {
            let window_start_ms =
                (start_ms / telemetry_interval_ms).floor() * telemetry_interval_ms;
            return Ok(ReplaySampledAdvance {
                final_state: self.state(),
                observations: vec![ReplayTimelineObservation {
                    at_ms: start_ms,
                    telemetry: Some(self.telemetry_since(window_start_ms)?),
                    checkpoint: None,
                }],
            });
        }
        let span_ms = until_ms - start_ms;
        let estimated_telemetry = (span_ms / telemetry_interval_ms).ceil() as usize + 1;
        let estimated_checkpoints = checkpoint_interval_ms
            .map(|interval_ms| (span_ms / interval_ms).ceil() as usize)
            .unwrap_or_default();
        ensure!(
            estimated_telemetry.saturating_add(estimated_checkpoints)
                <= MAX_REPLAY_SESSION_OBSERVATIONS,
            "sampled replay would exceed observation limit {MAX_REPLAY_SESSION_OBSERVATIONS}"
        );

        let mut observations = Vec::new();
        // Telemetry windows stay on the absolute cadence even when callers
        // seek in smaller actor ticks (for example 950ms -> 1000ms). The first
        // grid sample must still describe [0ms, 1000ms], not only the final
        // 50ms of the most recent command.
        let mut window_start_ms =
            (start_ms / telemetry_interval_ms).floor() * telemetry_interval_ms;
        let mut next_telemetry_ms = next_absolute_boundary(start_ms, telemetry_interval_ms);
        let mut next_checkpoint_ms =
            checkpoint_interval_ms.map(|interval_ms| next_absolute_boundary(start_ms, interval_ms));

        while self.runtime.now_ms() < until_ms {
            let boundary_ms = next_checkpoint_ms
                .into_iter()
                .chain(std::iter::once(next_telemetry_ms))
                .fold(until_ms, f64::min)
                .min(until_ms);
            ensure!(
                boundary_ms > self.runtime.now_ms(),
                "sampled replay interval no longer advances at this cursor"
            );
            self.runtime.advance_until_pause(boundary_ms)?;

            let telemetry_due = next_telemetry_ms <= boundary_ms;
            let checkpoint_due = next_checkpoint_ms.is_some_and(|at_ms| at_ms <= boundary_ms);
            let final_boundary = boundary_ms == until_ms;
            if telemetry_due || checkpoint_due || final_boundary {
                ensure!(
                    observations.len() < MAX_REPLAY_SESSION_OBSERVATIONS,
                    "sampled replay observation limit {MAX_REPLAY_SESSION_OBSERVATIONS} reached"
                );
                let telemetry = if telemetry_due || final_boundary {
                    let telemetry = self.telemetry_since(window_start_ms)?;
                    window_start_ms = boundary_ms;
                    Some(telemetry)
                } else {
                    None
                };
                let checkpoint = checkpoint_due.then(|| self.checkpoint()).transpose()?;
                observations.push(ReplayTimelineObservation {
                    at_ms: boundary_ms,
                    telemetry,
                    checkpoint,
                });
            }

            while next_telemetry_ms <= boundary_ms {
                next_telemetry_ms += telemetry_interval_ms;
            }
            if let (Some(interval_ms), Some(next_ms)) =
                (checkpoint_interval_ms, next_checkpoint_ms.as_mut())
            {
                while *next_ms <= boundary_ms {
                    *next_ms += interval_ms;
                }
            }
        }

        Ok(ReplaySampledAdvance {
            final_state: self.state(),
            observations,
        })
    }

    /// Capture a reusable checkpoint at the current cursor.
    fn checkpoint(&self) -> Result<ReplaySessionCheckpoint> {
        Ok(ReplaySessionCheckpoint {
            actions: self.actions.clone(),
            cursor_ms: self.runtime.now_ms(),
            traffic_horizon_ms: self.traffic_horizon_ms,
            runtime: self.runtime.checkpoint_replay()?,
        })
    }

    /// Run the retained future to completion and produce the ordinary replay
    /// report. Any scale actions already applied remain part of this timeline.
    fn finish(self) -> Result<TraceSimulationReport> {
        let (collector, _) = self.runtime.run()?;
        Ok(collector.finish())
    }
}

enum ReplaySessionCommand {
    State(mpsc::SyncSender<ReplaySessionState>),
    Telemetry {
        window_start_ms: f64,
        reply: mpsc::SyncSender<Result<ReplaySessionTelemetry>>,
    },
    TrafficTelemetry {
        window_start_ms: f64,
        reply: mpsc::SyncSender<Result<ReplaySessionTelemetry>>,
    },
    AdvanceTo {
        until_ms: f64,
        reply: mpsc::SyncSender<Result<ReplaySessionState>>,
    },
    AdvanceToWithTelemetry {
        until_ms: f64,
        window_start_ms: f64,
        reply: mpsc::SyncSender<Result<ReplayAdvanceSnapshot>>,
    },
    AdvanceSampled {
        until_ms: f64,
        telemetry_interval_ms: f64,
        checkpoint_interval_ms: Option<f64>,
        reply: mpsc::SyncSender<Result<ReplaySampledAdvance>>,
    },
    SetTargetReplicas {
        target_replicas: usize,
        reply: mpsc::SyncSender<Result<ReplaySessionState>>,
    },
    Checkpoint(mpsc::SyncSender<Result<ReplaySessionCheckpoint>>),
    Finish(mpsc::SyncSender<Result<TraceSimulationReport>>),
    Shutdown,
}

/// Sendable handle for a replay core confined to one owner thread.
///
/// The underlying aggregated runtime intentionally contains non-`Send`
/// planner/router types even though this session configures neither. Keeping
/// it on an owner thread preserves that contract while allowing an Axum actor
/// or another `SimulationDriver: Send` implementation to own this handle.
pub struct ReplaySession {
    command_tx: Option<mpsc::Sender<ReplaySessionCommand>>,
    owner: Option<JoinHandle<()>>,
}

impl ReplaySession {
    /// Materialize an existing single-turn Dynamo trace into the bounded
    /// interactive session. UUIDs are assigned deterministically by trace
    /// position so separately constructed sessions produce identical reports.
    pub fn from_trace(config: ReplaySessionConfig, trace: Trace) -> Result<Self> {
        ensure!(
            trace.is_single_turn(),
            "interactive replay currently supports exactly one turn per trace session"
        );
        ensure!(
            config.engine_args.block_size == trace.block_size,
            "engine block size {} does not match trace block size {}",
            config.engine_args.block_size,
            trace.block_size
        );
        let trace = trace.normalize_session_starts()?;
        let mut requests = trace.to_single_turn_requests()?;
        const RUSH_HOUR_UUID_PREFIX: u128 = 0x5255_5348_484f_5552_0000_0000_0000_0000u128;
        for (index, request) in requests.iter_mut().enumerate() {
            request.uuid = Some(uuid::Uuid::from_u128(
                RUSH_HOUR_UUID_PREFIX | (index as u128 + 1),
            ));
        }
        Self::new(config, requests)
    }

    /// Load the ordinary Dynamo Mooncake JSONL format, then construct a
    /// single-turn interactive session. Loading happens before the owner
    /// thread starts, so path and parse errors are returned synchronously.
    pub fn from_mooncake_file(
        config: ReplaySessionConfig,
        path: impl AsRef<Path>,
        trace_block_size: usize,
    ) -> Result<Self> {
        let trace = Trace::from_mooncake(path.as_ref(), trace_block_size)?;
        Self::from_trace(config, trace)
    }

    pub fn new(config: ReplaySessionConfig, requests: Vec<DirectRequest>) -> Result<Self> {
        Self::spawn_owner(move || ReplaySessionCore::new(config, requests))
    }

    pub fn restore(checkpoint: &ReplaySessionCheckpoint) -> Result<Self> {
        let checkpoint = checkpoint.clone();
        Self::spawn_owner(move || ReplaySessionCore::restore(&checkpoint))
    }

    fn spawn_owner<F>(build: F) -> Result<Self>
    where
        F: FnOnce() -> Result<ReplaySessionCore> + Send + 'static,
    {
        let (command_tx, command_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let owner = thread::Builder::new()
            .name("dynamo-replay-session".to_string())
            .spawn(move || {
                let core = match build() {
                    Ok(core) => {
                        let _ = ready_tx.send(Ok(()));
                        core
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                Self::run_owner(core, command_rx);
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                command_tx: Some(command_tx),
                owner: Some(owner),
            }),
            Ok(Err(error)) => {
                let _ = owner.join();
                Err(error)
            }
            Err(error) => {
                let _ = owner.join();
                Err(anyhow::anyhow!(
                    "replay session owner exited during startup: {error}"
                ))
            }
        }
    }

    fn run_owner(mut core: ReplaySessionCore, command_rx: mpsc::Receiver<ReplaySessionCommand>) {
        while let Ok(command) = command_rx.recv() {
            match command {
                ReplaySessionCommand::State(reply) => {
                    let _ = reply.send(core.state());
                }
                ReplaySessionCommand::Telemetry {
                    window_start_ms,
                    reply,
                } => {
                    let _ = reply.send(core.telemetry_since(window_start_ms));
                }
                ReplaySessionCommand::TrafficTelemetry {
                    window_start_ms,
                    reply,
                } => {
                    let _ = reply.send(core.traffic_telemetry_since(window_start_ms));
                }
                ReplaySessionCommand::AdvanceTo { until_ms, reply } => {
                    let result = Self::transaction(&mut core, |core| core.advance_to(until_ms));
                    let _ = reply.send(result);
                }
                ReplaySessionCommand::AdvanceToWithTelemetry {
                    until_ms,
                    window_start_ms,
                    reply,
                } => {
                    let result = Self::transaction(&mut core, |core| {
                        core.advance_to_with_telemetry(until_ms, window_start_ms)
                    });
                    let _ = reply.send(result);
                }
                ReplaySessionCommand::AdvanceSampled {
                    until_ms,
                    telemetry_interval_ms,
                    checkpoint_interval_ms,
                    reply,
                } => {
                    let result = Self::transaction(&mut core, |core| {
                        core.advance_sampled(
                            until_ms,
                            telemetry_interval_ms,
                            checkpoint_interval_ms,
                        )
                    });
                    let _ = reply.send(result);
                }
                ReplaySessionCommand::SetTargetReplicas {
                    target_replicas,
                    reply,
                } => {
                    let result = Self::transaction(&mut core, |core| {
                        core.set_target_replicas(target_replicas)
                    });
                    let _ = reply.send(result);
                }
                ReplaySessionCommand::Checkpoint(reply) => {
                    let _ = reply.send(core.checkpoint());
                }
                ReplaySessionCommand::Finish(reply) => {
                    let _ = reply.send(core.finish());
                    return;
                }
                ReplaySessionCommand::Shutdown => return,
            }
        }
    }

    fn transaction<T>(
        core: &mut ReplaySessionCore,
        operation: impl FnOnce(&mut ReplaySessionCore) -> Result<T>,
    ) -> Result<T> {
        let rollback = core.checkpoint()?;
        match operation(core) {
            Ok(value) => Ok(value),
            Err(operation_error) => match ReplaySessionCore::restore(&rollback) {
                Ok(restored) => {
                    *core = restored;
                    Err(operation_error)
                }
                Err(rollback_error) => Err(anyhow::anyhow!(
                    "replay session command failed ({operation_error:#}) and rollback failed ({rollback_error:#})"
                )),
            },
        }
    }

    fn request<T>(
        &self,
        build: impl FnOnce(mpsc::SyncSender<T>) -> ReplaySessionCommand,
    ) -> Result<T> {
        let command_tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("replay session is already closed"))?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        command_tx
            .send(build(reply_tx))
            .map_err(|_| anyhow::anyhow!("replay session owner is unavailable"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("replay session owner dropped its reply"))
    }

    pub fn state(&self) -> Result<ReplaySessionState> {
        self.request(ReplaySessionCommand::State)
    }

    pub fn telemetry(&self) -> Result<ReplaySessionTelemetry> {
        self.telemetry_since(0.0)
    }

    pub fn telemetry_since(&self, window_start_ms: f64) -> Result<ReplaySessionTelemetry> {
        self.request(|reply| ReplaySessionCommand::Telemetry {
            window_start_ms,
            reply,
        })?
    }

    /// Return exact traffic, admission, KV-reuse, sequence-length, and
    /// accept-length values for `(window_start_ms, cursor_ms]`, together with
    /// the ordinary point-in-time topology/queue gauges. This read-only query
    /// deliberately skips TTFT/ITL/E2E sample collection and leaves their
    /// percentile fields `None`.
    pub fn traffic_telemetry_since(&self, window_start_ms: f64) -> Result<ReplaySessionTelemetry> {
        self.request(|reply| ReplaySessionCommand::TrafficTelemetry {
            window_start_ms,
            reply,
        })?
    }

    pub fn advance_to(&mut self, until_ms: f64) -> Result<ReplaySessionState> {
        self.request(|reply| ReplaySessionCommand::AdvanceTo { until_ms, reply })?
    }

    /// Advance and capture a telemetry window under one rollback transaction.
    /// On error both the logical cursor and simulator internals remain at the
    /// pre-call state.
    pub fn advance_to_with_telemetry(
        &mut self,
        until_ms: f64,
        window_start_ms: f64,
    ) -> Result<ReplayAdvanceSnapshot> {
        self.request(|reply| ReplaySessionCommand::AdvanceToWithTelemetry {
            until_ms,
            window_start_ms,
            reply,
        })?
    }

    /// Advance through many telemetry/checkpoint boundaries under one rollback
    /// transaction. Boundaries use absolute multiples of each interval, so
    /// chunking a long seek produces the same grid as one call. The final
    /// target is always returned as a telemetry observation.
    pub fn advance_sampled(
        &mut self,
        until_ms: f64,
        telemetry_interval_ms: f64,
        checkpoint_interval_ms: Option<f64>,
    ) -> Result<ReplaySampledAdvance> {
        self.request(|reply| ReplaySessionCommand::AdvanceSampled {
            until_ms,
            telemetry_interval_ms,
            checkpoint_interval_ms,
            reply,
        })?
    }

    pub fn set_target_replicas(&mut self, target_replicas: usize) -> Result<ReplaySessionState> {
        self.request(|reply| ReplaySessionCommand::SetTargetReplicas {
            target_replicas,
            reply,
        })?
    }

    pub fn checkpoint(&self) -> Result<ReplaySessionCheckpoint> {
        self.request(ReplaySessionCommand::Checkpoint)?
    }

    pub fn finish(mut self) -> Result<TraceSimulationReport> {
        let command_tx = self
            .command_tx
            .take()
            .ok_or_else(|| anyhow::anyhow!("replay session is already closed"))?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        command_tx
            .send(ReplaySessionCommand::Finish(reply_tx))
            .map_err(|_| anyhow::anyhow!("replay session owner is unavailable"))?;
        drop(command_tx);
        let result = reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("replay session owner dropped its finish reply"))?;
        if let Some(owner) = self.owner.take() {
            owner
                .join()
                .map_err(|_| anyhow::anyhow!("replay session owner thread panicked"))?;
        }
        result
    }
}

fn next_absolute_boundary(cursor_ms: f64, interval_ms: f64) -> f64 {
    let mut boundary_ms = ((cursor_ms / interval_ms).floor() + 1.0) * interval_ms;
    if boundary_ms <= cursor_ms {
        boundary_ms = cursor_ms + interval_ms;
    }
    boundary_ms
}

impl Drop for ReplaySession {
    fn drop(&mut self) {
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(ReplaySessionCommand::Shutdown);
        }
        if let Some(owner) = self.owner.take() {
            let _ = owner.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::perf_model::AicCallback;
    use std::io::Write;
    use std::sync::Arc;
    use uuid::Uuid;

    struct FixedAicLatency;

    impl AicCallback for FixedAicLatency {
        fn predict_prefill(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> f64 {
            1.0
        }

        fn predict_decode(&self, _batch_size: usize, _isl: usize, _osl: usize) -> f64 {
            1.0
        }
    }

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1.0)
            .startup_time(Some(0.010))
            .build()
            .unwrap()
    }

    fn request(id: u128, arrival_ms: f64, prompt: u32) -> DirectRequest {
        DirectRequest {
            tokens: vec![prompt; 12],
            max_output_tokens: 4,
            output_token_ids: Some(vec![101, 102, 103, 104]),
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: Some(arrival_ms),
            ..Default::default()
        }
    }

    fn session(requests: Vec<DirectRequest>) -> ReplaySession {
        let mut config = ReplaySessionConfig::new(args());
        config.capture_per_request = true;
        ReplaySession::new(config, requests).unwrap()
    }

    fn assert_same_report(left: TraceSimulationReport, right: TraceSimulationReport) {
        assert_eq!(
            serde_json::to_value(&left).unwrap(),
            serde_json::to_value(&right).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&left.per_request).unwrap(),
            serde_json::to_value(&right.per_request).unwrap()
        );
    }

    #[test]
    fn busy_checkpoint_restores_and_continues_deterministically() {
        let requests = vec![request(1, 0.0, 7), request(2, 100.0, 9)];
        let mut source = session(requests);
        let paused = source.advance_to(0.001).unwrap();
        assert_eq!(paused.cursor_ms, 0.001);
        assert_eq!(paused.in_flight_requests, 1);

        let checkpoint = source.checkpoint().unwrap();
        assert_eq!(checkpoint.kind(), ReplayCheckpointKind::DeepRuntimeMemento);
        let restored = checkpoint.restore().unwrap();
        assert_eq!(source.state().unwrap(), restored.state().unwrap());
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn non_block_aligned_busy_checkpoint_restores_and_continues_deterministically() {
        let mut non_aligned = request(3, 0.0, 7);
        non_aligned.tokens.truncate(10);
        let mut source = session(vec![non_aligned]);
        let paused = source.advance_to(0.001).unwrap();
        assert_eq!(paused.in_flight_requests, 1);

        let restored = source.checkpoint().unwrap().restore().unwrap();
        assert_eq!(source.state().unwrap(), restored.state().unwrap());
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn scale_action_uses_lifecycle_and_survives_busy_restore() {
        let requests = vec![
            request(11, 0.0, 1),
            request(12, 5.0, 2),
            request(13, 20.0, 3),
            request(14, 40.0, 4),
        ];
        let mut source = session(requests);
        source.advance_to(0.001).unwrap();
        let scaled = source.set_target_replicas(2).unwrap();
        assert_eq!(scaled.revision, 1);
        assert_eq!(scaled.target_replicas, 2);
        assert_eq!(scaled.starting_replicas, 1);
        assert_eq!(scaled.provisioned_replicas, 2);

        let checkpoint = source.checkpoint().unwrap();
        let mut restored = checkpoint.restore().unwrap();
        assert_eq!(source.state().unwrap(), restored.state().unwrap());
        let source_ready = source.advance_to(10.001).unwrap();
        let restored_ready = restored.advance_to(10.001).unwrap();
        assert_eq!(source_ready, restored_ready);
        assert_eq!(source_ready.starting_replicas, 0);
        assert_eq!(source_ready.serving_replicas, 2);
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn interactive_advance_drains_startup_after_request_workload_finishes() {
        let slow_start_args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1.0)
            .startup_time(Some(30.0))
            .build()
            .unwrap();
        let mut source = ReplaySession::new(
            ReplaySessionConfig::new(slow_start_args),
            vec![request(10, 0.0, 7)],
        )
        .unwrap();

        source.advance_to(0.001).unwrap();
        source.set_target_replicas(2).unwrap();
        let request_complete = source.advance_to(10_000.0).unwrap();
        assert!(request_complete.workload_done);
        assert_eq!(request_complete.starting_replicas, 1);
        assert_eq!(request_complete.serving_replicas, 1);

        let lifecycle_complete = source.advance_to(30_001.0).unwrap();
        assert!(lifecycle_complete.workload_done);
        assert_eq!(lifecycle_complete.starting_replicas, 0);
        assert_eq!(lifecycle_complete.serving_replicas, 2);
    }

    #[test]
    fn advance_with_telemetry_is_atomic_when_the_window_is_invalid() {
        let mut source = session(vec![request(10_001, 0.0, 7)]);
        let before = source.state().unwrap();

        let error = source.advance_to_with_telemetry(5.0, 6.0).unwrap_err();
        assert!(error.to_string().contains("after cursor"));
        assert_eq!(source.state().unwrap(), before);

        let snapshot = source.advance_to_with_telemetry(5.0, 0.0).unwrap();
        assert_eq!(snapshot.state.cursor_ms, 5.0);
        assert_eq!(snapshot.telemetry.cursor_ms, 5.0);
    }

    #[test]
    fn busy_scale_down_drain_survives_restore() {
        let mut config = ReplaySessionConfig::new(args());
        config.initial_replicas = 2;
        config.capture_per_request = true;
        let mut source = ReplaySession::new(
            config,
            vec![
                request(15, 0.0, 1),
                request(16, 0.0, 2),
                request(17, 100.0, 3),
            ],
        )
        .unwrap();
        source.advance_to(0.001).unwrap();
        let draining = source.set_target_replicas(1).unwrap();
        assert_eq!(draining.target_replicas, 1);
        assert_eq!(draining.provisioned_replicas, 2);
        assert_eq!(draining.draining_replicas, 1);

        let restored = source.checkpoint().unwrap().restore().unwrap();
        assert_eq!(source.state().unwrap(), restored.state().unwrap());
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn restored_checkpoint_branches_without_retaining_future_actions() {
        let requests = vec![request(21, 0.0, 1), request(22, 200.0, 2)];
        let mut original = session(requests);
        original.advance_to(0.001).unwrap();
        let checkpoint = original.checkpoint().unwrap();

        original.set_target_replicas(2).unwrap();
        let mut branch = checkpoint.restore().unwrap();
        branch.set_target_replicas(3).unwrap();

        assert_eq!(original.state().unwrap().target_replicas, 2);
        assert_eq!(branch.state().unwrap().target_replicas, 3);
        assert_eq!(original.state().unwrap().revision, 1);
        assert_eq!(branch.state().unwrap().revision, 1);
    }

    #[test]
    fn quiescent_checkpoint_uses_the_same_deep_memento() {
        let mut source = session(vec![request(31, 0.0, 5), request(32, 10_000.0, 5)]);
        source.advance_to(5_000.0).unwrap();
        let checkpoint = source.checkpoint().unwrap();
        assert_eq!(checkpoint.kind(), ReplayCheckpointKind::DeepRuntimeMemento);
        let restored = checkpoint.restore().unwrap();
        assert_eq!(source.state().unwrap(), restored.state().unwrap());
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn session_rejects_nondeterministic_or_unsupported_inputs() {
        let mut missing_outputs = request(41, 0.0, 1);
        missing_outputs.output_token_ids = None;
        let error = ReplaySession::new(ReplaySessionConfig::new(args()), vec![missing_outputs])
            .err()
            .expect("missing planned outputs must be rejected");
        assert!(error.to_string().contains("planned output token IDs"));

        let unsupported_args = MockEngineArgs::builder()
            .block_size(4)
            .dp_size(2)
            .enable_prefix_caching(true)
            .build()
            .unwrap();
        let error = ReplaySession::new(
            ReplaySessionConfig::new(unsupported_args),
            vec![request(42, 0.0, 1)],
        )
        .err()
        .expect("DP>1 must be rejected");
        assert!(error.to_string().contains("exactly one DP rank"));

        let callback_args = MockEngineArgs::builder()
            .block_size(4)
            .enable_prefix_caching(true)
            .perf_model(Arc::new(PerfModel::from_aic_callback(Arc::new(
                FixedAicLatency,
            ))))
            .build()
            .unwrap();
        let error = ReplaySession::new(
            ReplaySessionConfig::new(callback_args),
            vec![request(43, 0.0, 1)],
        )
        .err()
        .expect("stateful timing callbacks must be rejected");
        assert!(error.to_string().contains("AI Configurator callbacks"));
    }

    #[test]
    fn advance_is_monotonic_and_scale_is_absolute() {
        let mut replay = session(vec![request(51, 0.0, 1), request(52, 100.0, 2)]);
        replay.advance_to(0.001).unwrap();
        assert!(replay.advance_to(0.0).is_err());
        replay.set_target_replicas(2).unwrap();
        assert!(replay.set_target_replicas(2).is_err());
        assert!(replay.set_target_replicas(3).is_err());
    }

    #[test]
    fn session_handle_can_move_to_an_owner_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<ReplaySession>();
    }

    #[test]
    fn telemetry_hides_modeled_future_until_completion_is_visible() {
        let mut replay = session(vec![request(61, 0.0, 1)]);
        replay.advance_to(0.0).unwrap();
        let at_arrival = replay.telemetry().unwrap();
        assert_eq!(at_arrival.cursor_ms, 0.0);
        assert_eq!(at_arrival.metrics.cumulative_arrivals, 1);
        assert_eq!(at_arrival.metrics.cumulative_output_tokens, 0);
        assert_eq!(at_arrival.queue_depth, 1);
        assert_eq!(at_arrival.running_requests, 0);
        assert_eq!(
            at_arrival
                .replicas
                .iter()
                .map(|replica| replica.running_requests)
                .sum::<usize>(),
            at_arrival.running_requests
        );

        // Resuming starts a pass at the prior pause. The scheduler records the
        // modeled completion timestamp immediately, but telemetry must not
        // expose that token before the completion event reaches the cursor.
        replay.advance_to(0.001).unwrap();
        let busy = replay.telemetry_since(0.0).unwrap();
        assert_eq!(busy.metrics.cumulative_output_tokens, 0);
        assert_eq!(busy.metrics.p95_ttft_ms, None);

        let checkpoint = replay.checkpoint().unwrap();
        let restored = checkpoint.restore().unwrap();
        assert_eq!(busy, restored.telemetry_since(0.0).unwrap());

        replay.advance_to(10_000.0).unwrap();
        let finished = replay.telemetry_since(0.001).unwrap();
        assert_eq!(finished.cursor_ms, 10_000.0);
        assert!(finished.workload_done);
        let terminal_horizon_ms = finished.terminal_horizon_ms.unwrap();
        assert!(terminal_horizon_ms < finished.cursor_ms);
        assert_eq!(finished.metrics.cumulative_output_tokens, 4);
        assert_eq!(finished.metrics.cumulative_completions, 1);
        assert!(finished.metrics.p95_ttft_ms.is_some());
        assert!(finished.metrics.p95_e2e_ms.is_some());

        replay.advance_to(11_000.0).unwrap();
        assert_eq!(
            replay.telemetry().unwrap().terminal_horizon_ms,
            Some(terminal_horizon_ms)
        );
    }

    #[test]
    fn mooncake_constructor_honors_initial_replica_count() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{{\"timestamp\": 10, \"input_length\": 8, \"output_length\": 2, \"hash_ids\": [1, 2]}}"
        )
        .unwrap();
        writeln!(
            file,
            "{{\"timestamp\": 20, \"input_length\": 8, \"output_length\": 2, \"hash_ids\": [1, 3]}}"
        )
        .unwrap();

        let mut config = ReplaySessionConfig::new(args());
        config.initial_replicas = 4;
        let replay = ReplaySession::from_mooncake_file(config, file.path(), 4).unwrap();
        let state = replay.state().unwrap();
        assert_eq!(state.serving_replicas, 4);
        assert_eq!(state.provisioned_replicas, 4);
        assert_eq!(state.pending_arrivals, 2);
        let telemetry = replay.telemetry().unwrap();
        assert_eq!(telemetry.replicas.len(), 4);
        assert_eq!(
            telemetry
                .replicas
                .iter()
                .map(|replica| replica.replica_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn busy_restore_preserves_future_eviction_behavior() {
        let eviction_args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(4))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1.0)
            .build()
            .unwrap();
        let make = |id, arrival_ms, prompt| DirectRequest {
            tokens: vec![prompt; 12],
            max_output_tokens: 4,
            output_token_ids: Some(vec![101, 102, 103, 104]),
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: Some(arrival_ms),
            ..Default::default()
        };
        let mut config = ReplaySessionConfig::new(eviction_args);
        config.capture_per_request = true;
        let mut source = ReplaySession::new(
            config,
            vec![
                make(71, 0.0, 1),
                make(72, 1_000.0, 2),
                make(73, 2_000.0, 3),
                make(74, 3_000.0, 1),
            ],
        )
        .unwrap();

        source.advance_to(1_000.001).unwrap();
        assert_eq!(source.state().unwrap().in_flight_requests, 1);
        let restored = source.checkpoint().unwrap().restore().unwrap();
        assert_same_report(source.finish().unwrap(), restored.finish().unwrap());
    }

    #[test]
    fn sampled_advance_matches_sequential_boundaries_with_one_command() {
        let requests = vec![
            request(81, 0.0, 1),
            request(82, 15.0, 2),
            request(83, 45.0, 3),
            request(84, 100.0, 4),
        ];
        let mut sampled = session(requests.clone());
        let mut sequential = session(requests);

        let result = sampled.advance_sampled(55.0, 10.0, Some(20.0)).unwrap();
        assert_eq!(
            result
                .observations
                .iter()
                .map(|observation| observation.at_ms)
                .collect::<Vec<_>>(),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 55.0]
        );
        assert_eq!(
            result
                .observations
                .iter()
                .filter(|observation| observation.checkpoint.is_some())
                .map(|observation| observation.at_ms)
                .collect::<Vec<_>>(),
            vec![20.0, 40.0]
        );

        let mut previous_ms = 0.0;
        for observation in &result.observations {
            let state = sequential.advance_to(observation.at_ms).unwrap();
            let expected = sequential.telemetry_since(previous_ms).unwrap();
            assert_eq!(observation.telemetry.as_ref(), Some(&expected));
            if let Some(checkpoint) = &observation.checkpoint {
                assert_eq!(checkpoint.cursor_ms(), observation.at_ms);
                assert_eq!(checkpoint.restore().unwrap().state().unwrap(), state);
            }
            previous_ms = observation.at_ms;
        }
        assert_eq!(result.final_state, sequential.state().unwrap());
        assert_same_report(sampled.finish().unwrap(), sequential.finish().unwrap());
    }

    #[test]
    fn sampled_advance_keeps_absolute_windows_across_sub_grid_calls() {
        let requests = vec![request(91, 0.0, 1), request(92, 500.0, 2)];
        let mut whole = session(requests.clone());
        let mut sliced = session(requests);

        let whole_result = whole.advance_sampled(1_000.0, 1_000.0, None).unwrap();
        sliced.advance_sampled(950.0, 1_000.0, None).unwrap();
        let sliced_result = sliced.advance_sampled(1_000.0, 1_000.0, None).unwrap();

        let whole_telemetry = whole_result.observations[0].telemetry.as_ref().unwrap();
        let sliced_telemetry = sliced_result.observations[0].telemetry.as_ref().unwrap();
        assert_eq!(whole_telemetry, sliced_telemetry);
        assert_eq!(sliced_telemetry.metrics.window_start_ms, 0.0);
        assert_eq!(sliced_telemetry.metrics.duration_ms, 1_000.0);
        assert_eq!(whole_result.final_state, sliced_result.final_state);
    }

    #[test]
    fn zero_span_sampled_advance_still_emits_final_telemetry() {
        let mut replay = session(vec![request(93, 0.0, 1)]);
        let result = replay.advance_sampled(0.0, 1_000.0, Some(5_000.0)).unwrap();

        assert_eq!(result.observations.len(), 1);
        let observation = &result.observations[0];
        assert_eq!(observation.at_ms, 0.0);
        assert!(observation.telemetry.is_some());
        assert!(observation.checkpoint.is_none());
        assert_eq!(result.final_state, replay.state().unwrap());
    }

    #[test]
    fn traffic_telemetry_matches_full_traffic_fields_and_point_gauges() {
        let mut replay = session(vec![request(94, 0.0, 1), request(95, 5.0, 2)]);
        replay.advance_to(10_000.0).unwrap();
        let state_before = replay.state().unwrap();

        let full = replay.telemetry_since(0.0).unwrap();
        assert!(full.metrics.p95_ttft_ms.is_some());
        assert!(full.metrics.p95_itl_ms.is_some());
        assert!(full.metrics.p95_e2e_ms.is_some());

        let traffic = replay.traffic_telemetry_since(0.0).unwrap();
        assert_eq!(replay.state().unwrap(), state_before);

        let mut expected = full;
        expected.metrics.p95_ttft_ms = None;
        expected.metrics.p95_itl_ms = None;
        expected.metrics.p95_e2e_ms = None;
        for replica in &mut expected.replicas {
            replica.metrics.p95_ttft_ms = None;
            replica.metrics.p95_itl_ms = None;
            replica.metrics.p95_e2e_ms = None;
        }

        assert_eq!(traffic, expected);
        assert_eq!(traffic.metrics.avg_isl_tokens, Some(12.0));
        assert_eq!(traffic.metrics.avg_requested_osl_tokens, Some(4.0));
        assert_eq!(traffic.metrics.avg_accept_length, Some(1.0));
    }

    #[test]
    fn retired_assigned_replica_keeps_long_window_metrics_in_full_and_traffic_views() {
        let mut config = ReplaySessionConfig::new(args());
        config.initial_replicas = 2;
        let mut replay = ReplaySession::new(
            config,
            vec![
                request(96, 0.0, 1),
                request(97, 0.0, 2),
                request(98, 50_000.0, 3),
            ],
        )
        .unwrap();

        replay.advance_to(10_000.0).unwrap();
        let scaled = replay.set_target_replicas(1).unwrap();
        assert_eq!(scaled.provisioned_replicas, 1);
        assert_eq!(scaled.serving_replicas, 1);
        let state_before = replay.state().unwrap();

        let full = replay.telemetry_since(0.0).unwrap();
        assert_eq!(full.provisioned_replicas, 1);
        assert_eq!(full.replicas.len(), 2);
        let retired = &full.replicas[1];
        assert_eq!(retired.replica_id, 1);
        assert_eq!(retired.lifecycle, ReplayReplicaLifecycle::Inactive);
        assert_eq!(retired.queued_requests, 0);
        assert_eq!(retired.running_requests, 0);
        assert_eq!(retired.in_flight_requests, 0);
        assert_eq!(retired.busy_ranks, 0);
        assert_eq!(retired.metrics.cumulative_arrivals, 1);
        assert_eq!(retired.metrics.arrivals, 1);
        assert_eq!(retired.metrics.cumulative_completions, 1);
        assert_eq!(retired.metrics.cumulative_output_tokens, 4);
        assert_eq!(retired.metrics.avg_isl_tokens, Some(12.0));
        assert_eq!(retired.metrics.avg_requested_osl_tokens, Some(4.0));
        assert_eq!(retired.metrics.kv_reuse_rate, Some(0.0));
        assert_eq!(retired.metrics.avg_accept_length, Some(1.0));
        assert!(retired.metrics.p95_ttft_ms.is_some());
        assert!(retired.metrics.p95_itl_ms.is_some());
        assert!(retired.metrics.p95_e2e_ms.is_some());

        let traffic = replay.traffic_telemetry_since(0.0).unwrap();
        let traffic_retired = &traffic.replicas[1];
        assert_eq!(traffic_retired.lifecycle, ReplayReplicaLifecycle::Inactive);
        assert_eq!(traffic_retired.metrics.cumulative_arrivals, 1);
        assert_eq!(traffic_retired.metrics.cumulative_completions, 1);
        assert_eq!(traffic_retired.metrics.cumulative_output_tokens, 4);
        assert_eq!(traffic_retired.metrics.kv_reuse_rate, Some(0.0));
        assert_eq!(traffic_retired.metrics.avg_accept_length, Some(1.0));
        assert_eq!(traffic_retired.metrics.p95_ttft_ms, None);
        assert_eq!(traffic_retired.metrics.p95_itl_ms, None);
        assert_eq!(traffic_retired.metrics.p95_e2e_ms, None);
        assert_eq!(replay.state().unwrap(), state_before);
    }

    #[test]
    fn never_assigned_retired_replica_is_inactive_with_empty_metrics() {
        let mut config = ReplaySessionConfig::new(args());
        config.initial_replicas = 2;
        let mut replay =
            ReplaySession::new(config, vec![request(99, 0.0, 1), request(100, 50_000.0, 2)])
                .unwrap();

        replay.advance_to(10_000.0).unwrap();
        replay.set_target_replicas(1).unwrap();
        let state_before = replay.state().unwrap();
        let telemetry = replay.telemetry_since(0.0).unwrap();

        assert_eq!(telemetry.provisioned_replicas, 1);
        assert_eq!(telemetry.active_replicas, 1);
        assert_eq!(telemetry.replicas.len(), 2);
        let retired = &telemetry.replicas[1];
        assert_eq!(retired.replica_id, 1);
        assert_eq!(retired.lifecycle, ReplayReplicaLifecycle::Inactive);
        assert_eq!(retired.queued_requests, 0);
        assert_eq!(retired.running_requests, 0);
        assert_eq!(retired.in_flight_requests, 0);
        assert_eq!(retired.busy_ranks, 0);
        assert_eq!(retired.metrics.cumulative_arrivals, 0);
        assert_eq!(retired.metrics.cumulative_admissions, 0);
        assert_eq!(retired.metrics.cumulative_completions, 0);
        assert_eq!(retired.metrics.cumulative_output_tokens, 0);
        assert_eq!(retired.metrics.arrivals, 0);
        assert_eq!(retired.metrics.admissions, 0);
        assert_eq!(retired.metrics.completions, 0);
        assert_eq!(retired.metrics.output_tokens, 0);
        assert_eq!(retired.metrics.request_rate_rps, 0.0);
        assert_eq!(retired.metrics.output_throughput_tok_s, 0.0);
        assert_eq!(retired.metrics.avg_isl_tokens, None);
        assert_eq!(retired.metrics.avg_requested_osl_tokens, None);
        assert_eq!(retired.metrics.request_admission_ratio, None);
        assert_eq!(retired.metrics.p95_ttft_ms, None);
        assert_eq!(retired.metrics.p95_itl_ms, None);
        assert_eq!(retired.metrics.p95_e2e_ms, None);
        assert_eq!(retired.metrics.kv_reuse_rate, None);
        assert_eq!(retired.metrics.cumulative_kv_reuse_rate, None);
        assert_eq!(retired.metrics.avg_accept_length, None);
        assert_eq!(replay.state().unwrap(), state_before);
    }

    #[test]
    fn accept_length_requires_a_visible_decode_forward() {
        let prefill_only = ReplayTelemetryWindow::from_cursor_values(
            TraceCursorMetricValues {
                window_output_tokens: 1,
                ..Default::default()
            },
            0.0,
            1.0,
        );
        assert_eq!(prefill_only.avg_accept_length, None);

        let with_decode = ReplayTelemetryWindow::from_cursor_values(
            TraceCursorMetricValues {
                window_output_tokens: 2,
                window_decode_tokens: 1,
                ..Default::default()
            },
            0.0,
            2.0,
        );
        assert_eq!(with_decode.avg_accept_length, Some(1.0));
    }
}
