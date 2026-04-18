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
    /// Number of blocks the router matched against the prefix cache at
    /// admission time. Used by the traffic accumulator to derive an
    /// average KV hit rate for the planner.
    pub(crate) overlap_blocks: u32,
    /// Total ISL expressed in blocks (ceil(isl_tokens / block_size)),
    /// paired with ``overlap_blocks`` for the hit-rate ratio.
    pub(crate) isl_blocks: u32,
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
    /// Mean prefix-cache hit rate (0.0-1.0) across router admissions in
    /// the window, computed as ``mean(overlap_blocks / isl_blocks)`` over
    /// admitted requests (i.e. the arithmetic mean of per-request
    /// ratios). Matches the semantics of the real router's
    /// ``dynamo_component_router_kv_hit_rate`` Prometheus histogram,
    /// which observes one ``overlap/isl`` sample per request; the
    /// PromQL query ``sum(increase(_sum)) / sum(increase(_count))``
    /// returns the arithmetic mean of those samples, independent of
    /// per-request ISL size.
    pub avg_kv_hit_rate: f64,
}

/// Accumulates traffic statistics between planner ticks for deriving
/// `TrafficObservation` (num_req, avg ISL, avg OSL, avg latencies, avg
/// KV hit rate over a window).
///
/// Latency samples are tracked independently of request counts: a request
/// only contributes to ``total_ttft_ms`` / ``ttft_count`` if a positive TTFT
/// was recorded, and similarly for ITL.  This means ``avg_ttft_ms`` and
/// ``avg_itl_ms`` reflect only requests that actually produced the sample,
/// rather than silently underestimating when some requests lack latency
/// data (e.g. requests that fail before emitting a token).
///
/// KV hit-rate observations come from the router at admission time (not
/// completion) and are recorded as per-request ratios, matching the real
/// router's per-request histogram: each admission contributes one
/// ``overlap_blocks / isl_blocks`` sample to the running mean, so large
/// requests don't get weighted more heavily than small ones.
#[derive(Debug)]
pub(in crate::replay::offline) struct TrafficAccumulator {
    window_start_ms: f64,
    num_req: usize,
    total_isl: usize,
    total_osl: usize,
    total_ttft_ms: f64,
    total_itl_ms: f64,
    ttft_count: usize,
    itl_count: usize,
    /// Running sum of per-request hit-rate ratios (``overlap / isl``);
    /// divided by ``hit_rate_count`` at drain time to give the mean.
    total_hit_rate: f64,
    /// Number of admissions with non-zero ISL blocks in the current window.
    hit_rate_count: usize,
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
            ttft_count: 0,
            itl_count: 0,
            total_hit_rate: 0.0,
            hit_rate_count: 0,
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
            if ttft_ms > 0.0 {
                self.total_ttft_ms += ttft_ms;
                self.ttft_count += 1;
            }
            if mean_itl_ms > 0.0 {
                self.total_itl_ms += mean_itl_ms;
                self.itl_count += 1;
            }
        }
    }

    /// Record one router admission's prefix-cache overlap as a
    /// per-request ratio. Called at admission time (not completion) so
    /// the mean hit rate reflects the router's view at routing decision
    /// — matching the real router's per-request histogram, where each
    /// request contributes exactly one ``overlap/isl`` sample.
    /// Admissions with ``isl_blocks == 0`` are skipped (no meaningful
    /// ratio), mirroring ``RequestTracker::kv_hit_rate()`` returning
    /// ``None`` in that case.
    pub(in crate::replay::offline) fn on_admission(
        &mut self,
        overlap_blocks: u32,
        isl_blocks: u32,
    ) {
        if isl_blocks == 0 {
            return;
        }
        self.total_hit_rate += f64::from(overlap_blocks) / f64::from(isl_blocks);
        self.hit_rate_count += 1;
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
        let avg_ttft_ms = if self.ttft_count > 0 {
            self.total_ttft_ms / self.ttft_count as f64
        } else {
            0.0
        };
        let avg_itl_ms = if self.itl_count > 0 {
            self.total_itl_ms / self.itl_count as f64
        } else {
            0.0
        };
        let avg_kv_hit_rate = if self.hit_rate_count > 0 {
            self.total_hit_rate / self.hit_rate_count as f64
        } else {
            0.0
        };
        self.window_start_ms = now_ms;
        self.num_req = 0;
        self.total_isl = 0;
        self.total_osl = 0;
        self.total_ttft_ms = 0.0;
        self.total_itl_ms = 0.0;
        self.ttft_count = 0;
        self.itl_count = 0;
        self.total_hit_rate = 0.0;
        self.hit_rate_count = 0;
        TrafficStats {
            duration_s,
            num_req,
            avg_isl,
            avg_osl,
            avg_ttft_ms,
            avg_itl_ms,
            avg_kv_hit_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traffic_accumulator_drain_with_no_admissions_reports_zero_hit_rate() {
        let mut acc = TrafficAccumulator::new();
        acc.on_request(100, 50, None);
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.num_req, 1);
        assert!((stats.avg_isl - 100.0).abs() < 1e-9);
        assert!((stats.avg_osl - 50.0).abs() < 1e-9);
        assert_eq!(stats.avg_kv_hit_rate, 0.0);
    }

    #[test]
    fn traffic_accumulator_hit_rate_is_mean_of_per_request_ratios() {
        let mut acc = TrafficAccumulator::new();
        // Small request: mostly hit. Big request: no hit.
        acc.on_admission(3, 4); // per-request ratio: 0.75
        acc.on_admission(0, 12); // per-request ratio: 0.0
        acc.on_request(256, 32, None);
        acc.on_request(768, 32, None);
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.num_req, 2);
        // Per-request mean matches the real router's Prometheus histogram:
        // (0.75 + 0.0) / 2 = 0.375. Every request contributes one sample
        // regardless of ISL size, so large requests don't dominate.
        assert!((stats.avg_kv_hit_rate - 0.375).abs() < 1e-9);
    }

    #[test]
    fn traffic_accumulator_skips_admissions_with_zero_isl_blocks() {
        let mut acc = TrafficAccumulator::new();
        acc.on_admission(0, 0); // skipped -- no meaningful ratio
        acc.on_admission(2, 4); // ratio = 0.5
        let stats = acc.drain(1_000.0);
        // Only the non-zero-ISL sample counts toward the mean.
        assert!((stats.avg_kv_hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn traffic_accumulator_resets_counters_on_drain() {
        let mut acc = TrafficAccumulator::new();
        acc.on_admission(5, 10);
        acc.on_request(100, 50, None);
        let _ = acc.drain(1_000.0);
        // Second drain on the same accumulator should see no state carried over.
        let stats = acc.drain(2_000.0);
        assert!((stats.duration_s - 1.0).abs() < 1e-9);
        assert_eq!(stats.num_req, 0);
        assert_eq!(stats.avg_isl, 0.0);
        assert_eq!(stats.avg_osl, 0.0);
        assert_eq!(stats.avg_kv_hit_rate, 0.0);
    }
}
