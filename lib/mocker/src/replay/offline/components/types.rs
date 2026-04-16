// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::RouterEvent;
use uuid::Uuid;

use super::super::runtime_utils::WorkerCompletionPayload;
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot};
use crate::loadgen::ReplayRequestHashes;
use crate::scheduler::AdmissionEvent;

#[derive(Debug, Clone, Copy)]
pub(in crate::replay) enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::replay::offline) enum EnginePassMode {
    Visible,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerAdmission {
    pub(crate) uuid: Uuid,
    pub(crate) worker_idx: usize,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct ScheduledWorkerCompletion {
    pub(in crate::replay::offline) at_ms: f64,
    pub(in crate::replay::offline) payload: WorkerCompletionPayload,
}

#[derive(Debug, Default)]
pub(in crate::replay::offline) struct EngineEffects {
    pub(in crate::replay::offline) admissions: Vec<AdmissionEvent>,
    pub(in crate::replay::offline) pass_start_kv_events: Vec<RouterEvent>,
    pub(in crate::replay::offline) immediate_completions: Vec<WorkerCompletionPayload>,
    pub(in crate::replay::offline) scheduled_completions: Vec<ScheduledWorkerCompletion>,
    /// Forward pass metrics snapshots emitted by workers during this drive cycle,
    /// keyed by worker index. Collected for planner integration.
    pub(in crate::replay::offline) fpm_snapshots: Vec<(usize, ForwardPassSnapshot)>,
}

impl EngineEffects {
    pub(in crate::replay::offline) fn is_empty(&self) -> bool {
        self.admissions.is_empty()
            && self.pass_start_kv_events.is_empty()
            && self.immediate_completions.is_empty()
            && self.scheduled_completions.is_empty()
    }
}

#[derive(Debug, Default)]
pub(crate) struct RouterEffects {
    pub(crate) admissions: Vec<WorkerAdmission>,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct ReadyArrival {
    pub(in crate::replay::offline) request: DirectRequest,
    pub(in crate::replay::offline) arrival_time_ms: f64,
    pub(in crate::replay::offline) replay_hashes: Option<ReplayRequestHashes>,
}

/// Accumulated traffic statistics returned by [`TrafficAccumulator::drain`].
///
/// IMPORTANT: When fields here are added or renamed, update the PyO3
/// binding in ``lib/bindings/python/rust/llm/replay.rs`` (drain_traffic
/// method) so the exported JSON dict matches.  The Python adapter in
/// ``replay_adapter.py`` reads these keys by name.
#[derive(Debug, Clone)]
pub struct TrafficStats {
    pub duration_s: f64,
    pub num_req: usize,
    pub avg_isl: f64,
    pub avg_osl: f64,
    pub avg_ttft_ms: f64,
    pub avg_itl_ms: f64,
}

/// Accumulates traffic statistics between planner ticks for deriving
/// `TrafficObservation` (num_req, avg ISL, avg OSL over a window).
#[derive(Debug)]
pub(in crate::replay::offline) struct TrafficAccumulator {
    window_start_ms: f64,
    num_req: usize,
    total_isl: usize,
    total_osl: usize,
    total_ttft_ms: f64,
    total_itl_ms: f64,
    itl_count: usize,
}

impl TrafficAccumulator {
    pub(in crate::replay::offline) fn new() -> Self {
        Self {
            window_start_ms: 0.0,
            num_req: 0,
            total_isl: 0,
            total_osl: 0,
            total_ttft_ms: 0.0,
            total_itl_ms: 0.0,
            itl_count: 0,
        }
    }

    /// Record one completed request with optional latency data.
    pub(in crate::replay::offline) fn on_request(
        &mut self,
        input_tokens: usize,
        output_tokens: usize,
        latencies: Option<(f64, f64)>,
    ) {
        self.num_req += 1;
        self.total_isl += input_tokens;
        self.total_osl += output_tokens;
        if let Some((ttft_ms, mean_itl_ms)) = latencies {
            self.total_ttft_ms += ttft_ms;
            if mean_itl_ms > 0.0 {
                self.total_itl_ms += mean_itl_ms;
                self.itl_count += 1;
            }
        }
    }

    /// Drain the accumulator at the given simulated time, resetting counters.
    pub(in crate::replay::offline) fn drain(&mut self, now_ms: f64) -> TrafficStats {
        let duration_s = (now_ms - self.window_start_ms) / 1000.0;
        let num_req = self.num_req;
        let avg_isl = if num_req > 0 {
            self.total_isl as f64 / num_req as f64
        } else {
            0.0
        };
        let avg_osl = if num_req > 0 {
            self.total_osl as f64 / num_req as f64
        } else {
            0.0
        };
        let avg_ttft_ms = if num_req > 0 {
            self.total_ttft_ms / num_req as f64
        } else {
            0.0
        };
        let avg_itl_ms = if self.itl_count > 0 {
            self.total_itl_ms / self.itl_count as f64
        } else {
            0.0
        };
        self.window_start_ms = now_ms;
        self.num_req = 0;
        self.total_isl = 0;
        self.total_osl = 0;
        self.total_ttft_ms = 0.0;
        self.total_itl_ms = 0.0;
        self.itl_count = 0;
        TrafficStats {
            duration_s,
            num_req,
            avg_isl,
            avg_osl,
            avg_ttft_ms,
            avg_itl_ms,
        }
    }
}
