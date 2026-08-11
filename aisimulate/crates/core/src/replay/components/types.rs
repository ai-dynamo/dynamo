// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use uuid::Uuid;

use super::super::core::{EngineEventBatch, EngineProgress, NoEngineEvents};
use super::super::events::{EnginePassCompletion, WorkerCompletionPayload};
use super::super::evidence::KvIngestEventEncoder;
use crate::engine::{CommandResult, KvEvent, LifecycleEvent, PressureEvent as EnginePressureEvent};
use crate::replay::core::RequestIdentity;
use crate::replay::loadgen::ReplayRequestPayload;
use crate::replay::protocol::DirectRequest;

impl RequestIdentity for DirectRequest {
    fn request_id(&self) -> Option<Uuid> {
        self.uuid
    }

    fn preferred_dp_rank(&self) -> Option<u32> {
        self.preferred_dp_rank
    }
}

impl RequestIdentity for ReplayRequestPayload {
    fn request_id(&self) -> Option<Uuid> {
        self.metadata().uuid
    }

    fn preferred_dp_rank(&self) -> Option<u32> {
        self.metadata().preferred_dp_rank
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnginePassMode {
    Visible,
    Hidden,
}

#[doc(hidden)]
pub trait ReplayEngineObservation {
    type Batch: EngineEventBatch;

    /// Whether the engine should retain and publish neutral KV events.
    const CAPTURE_ENGINE_KV_EVENTS: bool;

    /// Whether this observation adapter needs engine KV events for one worker
    /// stage. Most adapters use the same setting for every stage; adapters
    /// whose prefill and decode policies observe different event streams may
    /// override this method.
    fn capture_engine_kv_events(_stage: crate::replay::WorkerStage) -> bool {
        Self::CAPTURE_ENGINE_KV_EVENTS
    }

    /// Convert native G1 observations at their original visibility boundary.
    /// Dynamo Router adapters implement this without leaking RouterEvent into
    /// the generalized engine crate.
    fn observe_engine_events(
        stage: crate::replay::WorkerStage,
        worker_id: usize,
        dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> Self::Batch;

    fn stored_hashes(_events: &Self::Batch) -> Vec<u64> {
        Vec::new()
    }

    /// Return the number of events that should contribute to canonical
    /// ingestion evidence. `None` means this observation adapter does not
    /// expose an encodable KV stream.
    fn kv_ingest_event_count(_events: &Self::Batch) -> Option<usize> {
        None
    }

    /// Encode one batch into Replay's provider-neutral canonical digest.
    /// Implementations live beside the concrete event type (for example the
    /// Dynamo Router adapter), while Replay owns the accumulator and timing.
    fn encode_kv_ingest(
        _events: &Self::Batch,
        _encoder: &mut KvIngestEventEncoder<'_>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl ReplayEngineObservation for NoEngineEvents {
    type Batch = ();

    const CAPTURE_ENGINE_KV_EVENTS: bool = false;

    #[inline]
    fn observe_engine_events(
        _stage: crate::replay::WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        _events: Vec<KvEvent>,
    ) -> Self::Batch {
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AdmissionEvent {
    pub(crate) uuid: Uuid,
    pub(crate) reused_input_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PressureEvent {
    pub(crate) worker_id: u64,
    pub(crate) dp_rank: u32,
    pub(crate) event: EnginePressureEvent,
}

pub(crate) struct ObservedCommandEffects<Events: EngineEventBatch> {
    pub(crate) result: CommandResult,
    pub(crate) lifecycle_events: Vec<LifecycleEvent>,
    pub(crate) engine_events: Events,
}

#[derive(Debug)]
pub(crate) struct ScheduledEngineCompletion<Events: EngineEventBatch = ()> {
    pub(crate) at_ms: f64,
    pub(crate) completion: EnginePassCompletion<Events>,
}

#[derive(Debug)]
pub(crate) struct ArtifactPassStart {
    pub(crate) at_ms: f64,
    pub(crate) kv_events: Box<[KvEvent]>,
}

#[derive(Debug, Default)]
pub(crate) struct EngineEffects<Events: EngineEventBatch = ()> {
    pub(crate) admissions: Vec<AdmissionEvent>,
    pub(crate) pressure_events: Vec<PressureEvent>,
    pub(crate) pass_start_events: Events,
    pub(crate) artifact_pass_start: Option<Box<ArtifactPassStart>>,
    pub(crate) immediate_completions: Vec<WorkerCompletionPayload<Events>>,
    pub(crate) scheduled_completion: Option<ScheduledEngineCompletion<Events>>,
    pub(crate) progress: EngineProgress,
}

impl<Events: EngineEventBatch> EngineEffects<Events> {
    pub(crate) fn schedule_completion(
        &mut self,
        at_ms: f64,
        completion: EnginePassCompletion<Events>,
    ) {
        assert!(
            self.scheduled_completion.is_none(),
            "one drive may schedule at most one logical-engine pass"
        );
        self.scheduled_completion = Some(ScheduledEngineCompletion { at_ms, completion });
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.admissions.is_empty()
            && self.pressure_events.is_empty()
            && self.pass_start_events.is_empty()
            && self.artifact_pass_start.is_none()
            && self.immediate_completions.is_empty()
            && self.scheduled_completion.is_none()
            && !self.progress.made_progress
    }
}

/// Accumulated traffic statistics returned when the replay traffic accumulator is drained.
///
/// IMPORTANT: When fields here are added or renamed, update the PyO3
/// binding in ``lib/bindings/python/rust/llm/replay.rs`` (drain_traffic
/// method) so the exported JSON dict matches.  The Python adapter in
/// ``replay_adapter.py`` reads these keys by name.
#[derive(Debug, Clone)]
pub struct TrafficStats {
    pub duration_s: f64,
    /// Requests offered to the replay runtime during the window. This matches
    /// the live planner's `requests_started_total` demand signal.
    pub num_req: usize,
    pub avg_isl: f64,
    pub avg_osl: f64,
    pub avg_ttft_ms: f64,
    pub avg_itl_ms: f64,
    /// Completed, non-rejected requests behind `avg_isl` and `avg_osl`.
    pub shape_count: usize,
    /// Completed requests behind `avg_ttft_ms`.
    pub ttft_count: usize,
    /// Completed requests behind `avg_itl_ms`.
    pub itl_count: usize,
    /// Mean visible tokens produced per decode request-forward, including the
    /// base token. ``None`` means the window had no decode forwards.
    pub avg_accept_length: Option<f64>,
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
    /// Number of samples behind `avg_kv_hit_rate` (its denominator: router
    /// admissions with `isl_blocks > 0`). Carried so a consumer that merges
    /// several drained windows can reconstruct the exact sample-weighted mean
    /// rather than approximating it with `num_req`.
    pub hit_rate_count: usize,
    /// Number of decode request-forwards behind `avg_accept_length` (its
    /// denominator). Same purpose as `hit_rate_count` for cross-window merges.
    pub accept_length_forward_count: usize,
}

/// Accumulates traffic statistics between planner ticks for deriving
/// `TrafficObservation` (num_req, avg ISL, avg OSL, avg latencies, avg
/// KV hit rate over a window).
///
/// Offered request counts are recorded at arrival, matching the live
/// planner's `requests_started_total` signal. Shape and latency samples are
/// recorded independently at completion: a completed, non-rejected request
/// contributes to ISL/OSL, and only contributes to ``total_ttft_ms`` /
/// ``ttft_count`` if a positive TTFT was recorded (similarly for ITL). This
/// keeps demand independent of deployment capacity while preserving actual
/// output lengths and completed-request latency semantics.
///
/// KV hit-rate observations come from the router at admission time (not
/// completion) and are recorded as per-request ratios, matching the real
/// router's per-request histogram: each admission contributes one
/// ``overlap_blocks / isl_blocks`` sample to the running mean, so large
/// requests don't get weighted more heavily than small ones.
#[derive(Debug)]
pub(crate) struct TrafficAccumulator {
    window_start_ms: f64,
    offered_count: usize,
    total_isl: usize,
    total_osl: usize,
    shape_count: usize,
    total_ttft_ms: f64,
    total_itl_ms: f64,
    ttft_count: usize,
    itl_count: usize,
    /// Running sum of per-request hit-rate ratios (``overlap / isl``);
    /// divided by ``hit_rate_count`` at drain time to give the mean.
    total_hit_rate: f64,
    /// Number of admissions with non-zero ISL blocks in the current window.
    hit_rate_count: usize,
    /// Visible output tokens emitted by decode forwards in the current window.
    total_accept_length_tokens: usize,
    /// Number of request decode-forwards in the current window.
    accept_length_forward_count: usize,
}

impl TrafficAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            window_start_ms: 0.0,
            offered_count: 0,
            total_isl: 0,
            total_osl: 0,
            shape_count: 0,
            total_ttft_ms: 0.0,
            total_itl_ms: 0.0,
            ttft_count: 0,
            itl_count: 0,
            total_hit_rate: 0.0,
            hit_rate_count: 0,
            total_accept_length_tokens: 0,
            accept_length_forward_count: 0,
        }
    }

    /// Record one request offered to the replay runtime.
    pub(crate) fn on_arrival(&mut self) {
        self.offered_count += 1;
    }

    /// Record one completed, non-rejected request with optional latency data.
    pub(crate) fn on_completion(
        &mut self,
        input_tokens: usize,
        output_tokens: usize,
        latencies: Option<(f64, f64)>,
    ) {
        self.total_isl += input_tokens;
        self.total_osl += output_tokens;
        self.shape_count += 1;
        if let Some((ttft_ms, mean_itl_ms)) = latencies {
            if ttft_ms > 0.0 {
                self.total_ttft_ms += ttft_ms;
                self.ttft_count += 1;
            }
            if output_tokens > 1 && mean_itl_ms.is_finite() && mean_itl_ms >= 0.0 {
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
    pub(crate) fn on_admission(&mut self, overlap_blocks: u32, isl_blocks: u32) {
        if isl_blocks == 0 {
            return;
        }
        self.total_hit_rate += f64::from(overlap_blocks) / f64::from(isl_blocks);
        self.hit_rate_count += 1;
    }

    /// Record visible token bursts from decode forwards for accept-length
    /// scaling. ``visible_output_tokens`` is the numerator and
    /// ``decode_forwards`` is the number of requests that participated in the
    /// decode forward.
    pub(crate) fn on_accept_length_sample(
        &mut self,
        visible_output_tokens: usize,
        decode_forwards: usize,
    ) {
        if visible_output_tokens == 0 || decode_forwards == 0 {
            return;
        }
        self.total_accept_length_tokens += visible_output_tokens;
        self.accept_length_forward_count += decode_forwards;
    }

    /// Drain the accumulator at the given simulated time, resetting counters.
    pub(crate) fn drain(&mut self, now_ms: f64) -> TrafficStats {
        let duration_s = (now_ms - self.window_start_ms) / 1000.0;
        let num_req = self.offered_count;
        let avg_isl = if self.shape_count > 0 {
            self.total_isl as f64 / self.shape_count as f64
        } else {
            0.0
        };
        let avg_osl = if self.shape_count > 0 {
            self.total_osl as f64 / self.shape_count as f64
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
        let avg_accept_length = if self.accept_length_forward_count > 0 {
            Some(self.total_accept_length_tokens as f64 / self.accept_length_forward_count as f64)
        } else {
            None
        };
        // Capture the sample counts before the reset so a consumer that merges
        // several drained windows can reconstruct exact count-weighted means.
        let shape_count = self.shape_count;
        let ttft_count = self.ttft_count;
        let itl_count = self.itl_count;
        let hit_rate_count = self.hit_rate_count;
        let accept_length_forward_count = self.accept_length_forward_count;
        self.window_start_ms = now_ms;
        self.offered_count = 0;
        self.total_isl = 0;
        self.total_osl = 0;
        self.shape_count = 0;
        self.total_ttft_ms = 0.0;
        self.total_itl_ms = 0.0;
        self.ttft_count = 0;
        self.itl_count = 0;
        self.total_hit_rate = 0.0;
        self.hit_rate_count = 0;
        self.total_accept_length_tokens = 0;
        self.accept_length_forward_count = 0;
        TrafficStats {
            duration_s,
            num_req,
            avg_isl,
            avg_osl,
            avg_ttft_ms,
            avg_itl_ms,
            shape_count,
            ttft_count,
            itl_count,
            avg_accept_length,
            avg_kv_hit_rate,
            hit_rate_count,
            accept_length_forward_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn no_engine_events_and_batch_are_zero_sized() {
        type Batch = <NoEngineEvents as ReplayEngineObservation>::Batch;

        assert_eq!(size_of::<NoEngineEvents>(), 0);
        assert_eq!(size_of::<Batch>(), 0);
    }

    #[test]
    fn traffic_accumulator_drain_with_no_admissions_reports_zero_hit_rate() {
        let mut acc = TrafficAccumulator::new();
        acc.on_arrival();
        acc.on_completion(100, 50, None);
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.num_req, 1);
        assert_eq!(stats.shape_count, 1);
        assert!((stats.avg_isl - 100.0).abs() < 1e-9);
        assert!((stats.avg_osl - 50.0).abs() < 1e-9);
        assert_eq!(stats.avg_kv_hit_rate, 0.0);
        assert_eq!(stats.avg_accept_length, None);
    }

    #[test]
    fn traffic_accumulator_hit_rate_is_mean_of_per_request_ratios() {
        let mut acc = TrafficAccumulator::new();
        // Small request: mostly hit. Big request: no hit.
        acc.on_admission(3, 4); // per-request ratio: 0.75
        acc.on_admission(0, 12); // per-request ratio: 0.0
        acc.on_arrival();
        acc.on_arrival();
        acc.on_completion(256, 32, None);
        acc.on_completion(768, 32, None);
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.num_req, 2);
        // Per-request mean matches the real router's Prometheus histogram:
        // (0.75 + 0.0) / 2 = 0.375. Every request contributes one sample
        // regardless of ISL size, so large requests don't dominate.
        assert!((stats.avg_kv_hit_rate - 0.375).abs() < 1e-9);
    }

    #[test]
    fn traffic_accumulator_accept_length_is_weighted_by_decode_forwards() {
        let mut acc = TrafficAccumulator::new();
        acc.on_accept_length_sample(6, 2); // two requests accepted three tokens each
        acc.on_accept_length_sample(2, 2); // two requests accepted one token each
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.avg_accept_length, Some(2.0));
    }

    #[test]
    fn traffic_accumulator_accept_length_missing_without_decode_forwards() {
        let mut acc = TrafficAccumulator::new();
        acc.on_accept_length_sample(4, 0);
        acc.on_accept_length_sample(0, 4);
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.avg_accept_length, None);
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
        acc.on_arrival();
        acc.on_completion(100, 50, None);
        let _ = acc.drain(1_000.0);
        // Second drain on the same accumulator should see no state carried over.
        let stats = acc.drain(2_000.0);
        assert!((stats.duration_s - 1.0).abs() < 1e-9);
        assert_eq!(stats.num_req, 0);
        assert_eq!(stats.avg_isl, 0.0);
        assert_eq!(stats.avg_osl, 0.0);
        assert_eq!(stats.avg_kv_hit_rate, 0.0);
        assert_eq!(stats.avg_accept_length, None);
    }

    #[test]
    fn traffic_accumulator_retains_zero_millisecond_itl_samples() {
        let mut acc = TrafficAccumulator::new();
        acc.on_arrival();
        acc.on_arrival();
        acc.on_completion(10, 3, Some((1.0, 0.0)));
        acc.on_completion(10, 3, Some((1.0, 10.0)));
        let stats = acc.drain(1_000.0);
        assert_eq!(stats.avg_itl_ms, 5.0);
    }

    #[test]
    fn traffic_accumulator_reports_offered_demand_before_completion() {
        let mut acc = TrafficAccumulator::new();
        acc.on_arrival();

        let offered = acc.drain(1_000.0);
        assert_eq!(offered.num_req, 1);
        assert_eq!(offered.shape_count, 0);
        assert_eq!(offered.avg_isl, 0.0);
        assert_eq!(offered.avg_osl, 0.0);

        acc.on_completion(100, 50, Some((2_000.0, 20.0)));
        let completed = acc.drain(2_000.0);
        assert_eq!(completed.num_req, 0);
        assert_eq!(completed.shape_count, 1);
        assert_eq!(completed.avg_isl, 100.0);
        assert_eq!(completed.avg_osl, 50.0);
        assert_eq!(completed.ttft_count, 1);
        assert_eq!(completed.itl_count, 1);
    }
}
