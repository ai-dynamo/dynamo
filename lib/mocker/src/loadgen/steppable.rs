// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! [`SteppableReplay`]: the real mocker offline runtimes (single-engine,
//! aggregated multi-worker + router, and disaggregated prefill/decode) exposed
//! as **passive, externally-clocked, dynamic-admission** engines — the Tier-2
//! boundary from the engine-boundary spec, uniform across every topology.
//!
//! The native offline runtimes drive themselves over a static arrival queue.
//! This trait inverts that: the caller owns the clock and the loop, submits
//! requests as they emerge (closed-loop dataflow), and steps the runtime one
//! logical timestamp at a time. Each step advances the runtime's own time,
//! writes `on_admit`/`on_token`/`on_terminal` into its collector, and reports
//! per-request events so the caller can wake its own futures. Batching, KV
//! cache, prefix caching, chunked prefill, routing, and disagg handoff are the
//! runtimes' — this adds nothing but the seam.

use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::replay::ReplayTerminalStatus;
use crate::replay::SlaThresholds;
use crate::replay::TraceCollector;
use crate::replay::TraceSimulationReport;
use crate::replay::offline::core::ReplayWorkerCore;

/// One per-request event produced by an engine [`step`](SteppableReplay::step).
#[derive(Debug, Clone)]
pub struct EngineEvent {
    /// The request this event belongs to.
    pub uuid: Uuid,
    /// True when this pass emitted an output token for the request (the caller
    /// gates first-token off the first such event). Always equals
    /// `token_id.is_some()`.
    pub emitted_token: bool,
    /// Exact output token ID when this is a client-visible token event.
    /// Delta-accumulating trace drivers consume this value to reconstruct the
    /// next turn's cumulative prompt without inventing token identities.
    pub token_id: Option<u32>,
    /// Authoritative terminal classification, or `None` for a token-only event;
    /// distinguishes cancellation and failure from successful completion and
    /// rejection.
    pub terminal_status: Option<ReplayTerminalStatus>,
}

impl EngineEvent {
    /// The single construction point, so `emitted_token` cannot diverge from
    /// `token_id` and the terminal classification stays authoritative.
    pub(crate) fn new(
        uuid: Uuid,
        token_id: Option<u32>,
        terminal_status: Option<ReplayTerminalStatus>,
    ) -> Self {
        Self {
            uuid,
            emitted_token: token_id.is_some(),
            token_id,
            terminal_status,
        }
    }

    /// Construct a terminal event that carries no output token.
    pub fn terminal(uuid: Uuid, status: ReplayTerminalStatus) -> Self {
        Self::new(uuid, None, Some(status))
    }
}

/// The result of one runtime step.
#[derive(Debug, Clone, Default)]
pub struct StepOutcome {
    /// Runtime time (ms) after this step — the caller advances its clock here.
    pub end_ms: f64,
    /// Per-request events emitted during this step.
    pub events: Vec<EngineEvent>,
}

/// A real mocker offline runtime as a steppable, clock-injected, dynamically-fed
/// engine. Implemented by every topology (single / aggregated / disaggregated),
/// so a caller drives any of them through one uniform loop.
pub trait SteppableReplay {
    /// Current runtime time (ms). Submissions arrive here; the collector's unit.
    fn now_ms(&self) -> f64;
    /// Advance the runtime clock forward to `now_ms` (used when the caller
    /// advances virtual time past an idle gap, e.g. an inter-turn delay, with no
    /// engine work). Monotonic: a `now_ms` at or before the current time is a
    /// no-op.
    fn advance_now_ms(&mut self, now_ms: f64);
    /// Admit `req` at the current time, recording `on_arrival`. Returns the id
    /// used to correlate later [`EngineEvent`]s.
    fn submit(&mut self, req: DirectRequest) -> anyhow::Result<Uuid>;
    /// Cancel an admitted request at the current runtime time. Returns the
    /// terminal event when cancellation won, or `None` if the request was
    /// already terminal or unknown. Cleanup may continue in later steps when
    /// an engine pass was already in flight.
    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>>;
    /// Toggle retention of the backend's complete per-request causality
    /// records for JSONL/raw-record exporters.
    fn set_capture_per_request(&mut self, capture: bool);
    /// Configure canonical goodput thresholds on the backend collector.
    fn set_sla_thresholds(&mut self, sla: SlaThresholds);
    /// Advance one logical timestamp of work, advancing the runtime clock and
    /// returning the new time plus per-request events to route.
    fn step(&mut self) -> anyhow::Result<StepOutcome>;
    /// Advance at most through `until_ms`, emitting events at or before that
    /// deadline. Routed runtimes stop at the deadline when their next internal
    /// event is later, allowing an external DES driver to interleave arrivals
    /// and firing gates without changing batch composition.
    ///
    /// The direct single-worker wrapper cannot interrupt an already executing
    /// scheduler pass and therefore rejects finite deadlines; external mixed-
    /// event drivers should use the one-worker aggregated runtime.
    fn step_until(&mut self, until_ms: f64) -> anyhow::Result<StepOutcome> {
        if until_ms.is_finite() {
            anyhow::bail!("this steppable engine cannot stop at finite deadline {until_ms}ms");
        }
        self.step()
    }
    /// True when no pending or in-flight request work remains.
    fn is_idle(&self) -> bool;
    /// Number of in-flight (admitted, not terminal) requests.
    fn in_flight(&self) -> usize;
    /// Drain the accumulated measurements into a report stamped with `wall_ms`.
    /// Leaves the runtime's collector empty.
    fn take_report(&mut self, wall_ms: f64) -> TraceSimulationReport;

    /// Next sim time (ms) at which the engine can make progress, for the
    /// caller's discrete-event pump: aggregated/disaggregated surface their
    /// event-heap `next_timestamp` (`min(next_arrival, next_event, next_offload)`);
    /// single-worker returns the current time while in-flight work remains,
    /// `None` when idle. The caller advances virtual time to
    /// `min(next_pacer_deadline, next_event_ms())` and then steps.
    fn next_event_ms(&mut self) -> Option<f64>;

    /// Measured `(ttft_ms, mean_itl_ms)` for `uuid` once it has a first token,
    /// else `None`. Read on completion to build the caller's own per-request
    /// record; the caller stamps arrival/e2e from its own clock (identical to
    /// the engine's, since admission happens at the caller's `advance_now_ms`).
    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)>;

    /// First scheduler admission `(at_ms, reused_input_tokens)` once known.
    fn request_admission(&self, _uuid: Uuid) -> Option<(f64, usize)> {
        None
    }

    /// Output tokens actually emitted for `uuid` (OSL), or `None` if unknown.
    fn actual_output_length(&self, uuid: Uuid) -> Option<usize>;
}

/// Build an `EngineEvent` from a scheduler output signal.
pub(crate) fn event_from_signal(signal: &crate::common::protocols::OutputSignal) -> EngineEvent {
    let terminal_status = signal.completed.then_some(if signal.rejected {
        ReplayTerminalStatus::Rejected
    } else {
        ReplayTerminalStatus::Completed
    });
    EngineEvent::new(signal.uuid, signal.token_id, terminal_status)
}

/// Single-engine topology: one `VllmCore`/`SglangCore` worker, no router.
pub struct SteppableEngine {
    worker: ReplayWorkerCore,
    collector: TraceCollector,
    now_ms: f64,
}

impl SteppableEngine {
    /// Build a single-worker engine from `MockEngineArgs`.
    pub fn new(args: MockEngineArgs) -> Self {
        // The steppable single-worker engine has exactly one (decode) worker for
        // the whole run and no event loop to integrate, so declare a static
        // count and its GPU parallelism up front — `finish()` then derives
        // `decode_worker_seconds` as 1 × duration_s and `gpu_hours` from the
        // engine's own parallelism. Mirrors `offline/single.rs`; without it the
        // report leaves both fields at 0 (the aggregated/disagg steppable paths
        // declare their own).
        let decode_gpus_per_worker = args.aic_gpus_per_worker();
        let mut collector = TraceCollector::default();
        collector.set_static_worker_count(0, 1);
        collector.set_gpus_per_worker(0, decode_gpus_per_worker);
        SteppableEngine {
            worker: ReplayWorkerCore::new(args),
            collector,
            now_ms: 0.0,
        }
    }

    /// Toggle per-request record capture (for raw-record export).
    pub fn set_capture_per_request(&mut self, capture: bool) {
        self.collector.set_capture_per_request(capture);
    }
}

impl SteppableReplay for SteppableEngine {
    fn now_ms(&self) -> f64 {
        self.now_ms
    }

    fn advance_now_ms(&mut self, now_ms: f64) {
        if now_ms > self.now_ms {
            self.now_ms = now_ms;
        }
    }

    fn submit(&mut self, req: DirectRequest) -> anyhow::Result<Uuid> {
        let input_length = req.tokens.len();
        let output_length = req.max_output_tokens;
        let uuid = self.worker.receive(req);
        self.collector
            .on_arrival(uuid, self.now_ms, input_length, output_length);
        Ok(uuid)
    }

    fn cancel(&mut self, uuid: Uuid) -> anyhow::Result<Option<EngineEvent>> {
        if !self.worker.cancel(uuid)? {
            return Ok(None);
        }
        self.collector
            .on_terminal(uuid, ReplayTerminalStatus::Canceled);
        Ok(Some(EngineEvent::terminal(
            uuid,
            ReplayTerminalStatus::Canceled,
        )))
    }

    fn set_capture_per_request(&mut self, capture: bool) {
        self.collector.set_capture_per_request(capture);
    }

    fn set_sla_thresholds(&mut self, sla: SlaThresholds) {
        self.collector.set_sla_thresholds(sla);
    }

    fn step(&mut self) -> anyhow::Result<StepOutcome> {
        let pass = self.worker.execute_pass(&mut self.collector, self.now_ms);
        let mut events = Vec::with_capacity(pass.output_signals.len());
        for signal in &pass.output_signals {
            if signal.completed {
                let status = if signal.rejected {
                    ReplayTerminalStatus::Rejected
                } else {
                    ReplayTerminalStatus::Completed
                };
                self.collector.on_terminal(signal.uuid, status);
            }
            events.push(event_from_signal(signal));
        }
        self.now_ms = pass.end_ms;
        Ok(StepOutcome {
            end_ms: pass.end_ms,
            events,
        })
    }

    fn is_idle(&self) -> bool {
        self.worker.is_empty()
    }

    fn in_flight(&self) -> usize {
        self.worker.num_requests()
    }

    fn take_report(&mut self, wall_ms: f64) -> TraceSimulationReport {
        std::mem::take(&mut self.collector)
            .finish()
            .with_wall_time_ms(wall_ms)
    }

    fn next_event_ms(&mut self) -> Option<f64> {
        // Single worker: no router/event heap, so the only "engine event" is
        // "make progress on the in-flight batch now". Idle -> None.
        if self.worker.is_empty() {
            None
        } else {
            Some(self.now_ms)
        }
    }

    fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)> {
        self.collector.request_latencies(uuid)
    }

    fn request_admission(&self, uuid: Uuid) -> Option<(f64, usize)> {
        self.collector.request_admission(uuid)
    }

    fn actual_output_length(&self, uuid: Uuid) -> Option<usize> {
        self.collector.actual_output_length(uuid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::ReplayRouterMode;

    fn request(uuid: u128) -> DirectRequest {
        DirectRequest {
            tokens: (0..128).collect(),
            max_output_tokens: 16,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
            priority: 0,
            strict_priority: 0,
            policy_class: None,
        }
    }

    #[test]
    fn aggregate_step_until_never_crosses_external_deadline() {
        let mut engine = crate::loadgen::SteppableAgg::new(
            MockEngineArgs::default(),
            1,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();
        engine.submit(request(1)).unwrap();

        let deadline_ms = 0.000_001;
        let outcome = engine.step_until(deadline_ms).unwrap();

        assert!(outcome.end_ms <= deadline_ms);
        assert_eq!(engine.now_ms(), deadline_ms);
        assert!(!engine.is_idle());
    }

    #[test]
    fn aggregate_cancellation_is_typed_and_suppresses_stale_pass_events() {
        for router_mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
            let mut engine =
                crate::loadgen::SteppableAgg::new(MockEngineArgs::default(), 1, router_mode)
                    .unwrap();
            engine.set_capture_per_request(true);
            let uuid = engine.submit(request(10)).unwrap();
            engine.step_until(0.000_001).unwrap();

            let canceled = engine.cancel(uuid).unwrap().unwrap();
            assert_eq!(
                canceled.terminal_status,
                Some(ReplayTerminalStatus::Canceled)
            );
            let mut stale_events = Vec::new();
            for _ in 0..1_000 {
                if engine.is_idle() {
                    break;
                }
                stale_events.extend(engine.step().unwrap().events);
            }
            assert!(engine.is_idle(), "canceled aggregate engine did not drain");
            assert!(stale_events.iter().all(|event| event.uuid != uuid));

            let report = engine.take_report(engine.now_ms());
            assert_eq!(report.request_counts.num_requests, 1);
            assert_eq!(report.request_counts.completed_requests, 0);
            assert_eq!(report.per_request.len(), 1);
            assert_eq!(
                report.per_request[0].terminal_status,
                ReplayTerminalStatus::Canceled
            );
        }
    }

    #[test]
    fn direct_single_worker_cancellation_is_idempotent() {
        let mut engine = SteppableEngine::new(MockEngineArgs::default());
        engine.set_capture_per_request(true);
        let uuid = engine.submit(request(11)).unwrap();

        assert_eq!(
            engine.cancel(uuid).unwrap().unwrap().terminal_status,
            Some(ReplayTerminalStatus::Canceled)
        );
        assert!(engine.cancel(uuid).unwrap().is_none());
        assert!(engine.is_idle());
        let report = engine.take_report(engine.now_ms());
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(
            report.per_request[0].terminal_status,
            ReplayTerminalStatus::Canceled
        );
    }

    #[test]
    fn disaggregated_events_expose_only_collector_visible_decode_tokens() {
        let mut engine = crate::loadgen::SteppableDisagg::new(
            MockEngineArgs::default(),
            MockEngineArgs::default(),
            1,
            1,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();
        engine.submit(request(7)).unwrap();

        let mut emitted_tokens = 0_usize;
        let mut steps = 0_usize;
        while !engine.is_idle() {
            let outcome = engine.step().unwrap();
            emitted_tokens += outcome
                .events
                .iter()
                .filter(|event| event.emitted_token)
                .count();
            steps += 1;
            assert!(
                steps < 1_000,
                "disaggregated steppable replay did not drain"
            );
        }

        let report = engine.take_report(engine.now_ms());
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.request_counts.total_output_tokens, 16);
        assert_eq!(emitted_tokens, report.request_counts.total_output_tokens);
    }

    #[test]
    fn disaggregate_step_until_never_crosses_external_deadline() {
        let args = MockEngineArgs::default();
        let mut engine = crate::loadgen::SteppableDisagg::new(
            args.clone(),
            args,
            1,
            1,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();
        engine.submit(request(3)).unwrap();

        let deadline_ms = 0.000_001;
        let outcome = engine.step_until(deadline_ms).unwrap();

        assert!(outcome.end_ms <= deadline_ms);
        assert_eq!(engine.now_ms(), deadline_ms);
        assert!(!engine.is_idle());
    }

    #[test]
    fn direct_single_worker_rejects_finite_step_deadline() {
        let mut engine = SteppableEngine::new(MockEngineArgs::default());
        engine.submit(request(2)).unwrap();
        assert!(engine.step_until(1.0).is_err());
        assert_eq!(engine.now_ms(), 0.0);
    }

    #[test]
    fn deadline_bounded_steps_preserve_request_measurements() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let _guard = runtime.enter();
        let build = || {
            let mut engine = crate::loadgen::SteppableAgg::new(
                MockEngineArgs::default(),
                2,
                ReplayRouterMode::RoundRobin,
            )
            .unwrap();
            engine.set_capture_per_request(true);
            engine
        };
        let mut unbounded = build();
        let mut bounded = build();
        for uuid in 0..12 {
            unbounded.submit(request(uuid)).unwrap();
            bounded.submit(request(uuid)).unwrap();
        }

        while !unbounded.is_idle() {
            unbounded.step().unwrap();
        }
        while !bounded.is_idle() {
            let deadline_ms = bounded.now_ms() + 1.0;
            bounded.step_until(deadline_ms).unwrap();
        }

        let unbounded_wall_ms = unbounded.now_ms();
        let bounded_wall_ms = bounded.now_ms();
        let unbounded_report = unbounded.take_report(unbounded_wall_ms);
        let bounded_report = bounded.take_report(bounded_wall_ms);
        assert_eq!(
            unbounded_report.request_counts.num_requests,
            bounded_report.request_counts.num_requests
        );
        assert_eq!(
            unbounded_report.request_counts.completed_requests,
            bounded_report.request_counts.completed_requests
        );
        assert_eq!(
            unbounded_report.request_counts.total_input_tokens,
            bounded_report.request_counts.total_input_tokens
        );
        assert_eq!(
            unbounded_report.request_counts.total_output_tokens,
            bounded_report.request_counts.total_output_tokens
        );
        assert_eq!(
            serde_json::to_vec(&unbounded_report.per_request).unwrap(),
            serde_json::to_vec(&bounded_report.per_request).unwrap(),
            "external idle checkpoints must not change per-request perf timestamps"
        );
        assert_eq!(
            unbounded_report.latency.ttft.mean_ms,
            bounded_report.latency.ttft.mean_ms
        );
        assert_eq!(
            unbounded_report.latency.itl.distribution.mean_ms,
            bounded_report.latency.itl.distribution.mean_ms
        );
    }
}
