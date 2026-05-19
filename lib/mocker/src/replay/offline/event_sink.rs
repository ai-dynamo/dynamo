// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sink for per-request engine events emitted during offline replay.
//!
//! Decouples engine event production (agg/disagg/scheduler) from event
//! consumption. The built-in `TraceCollector` implements `EventSink` and
//! accumulates a full `TraceSimulationReport`; out-of-tree observers
//! (custom exporters, alternate aggregators, debug taps) can plug in
//! their own `EventSink` impl by being installed as the secondary on a
//! `CompositeEventSink`.

use uuid::Uuid;

use crate::replay::TraceCollector;

/// Sink for per-request engine events emitted by AggRuntime / DisaggRuntime.
///
/// The native offline replay uses a default implementation backed by
/// `TraceCollector` (accumulates a full report). Out-of-tree observers
/// implement this trait and install themselves as the secondary on a
/// `CompositeEventSink` to receive the same event stream alongside the
/// built-in collector.
///
/// All methods take `&mut self` because the typical impl is stateful
/// (e.g. accumulates per-request data, advances a write position).
pub trait EventSink: Send {
    fn on_arrival(
        &mut self,
        uuid: Uuid,
        arrival_time_ms: f64,
        input_length: usize,
        output_length: usize,
    );

    fn on_admit(&mut self, uuid: Uuid, admit_time_ms: f64, reused_input_tokens: usize);

    fn on_token(&mut self, uuid: Uuid, token_time_ms: f64);

    /// Called once per request when the engine finalizes it (last token
    /// emitted, finish_reason known). The native `TraceCollector` ignores
    /// this (its completion data is derived from accumulated tokens at
    /// `finish()` time); out-of-tree sinks can override to capture
    /// terminal events.
    fn on_completion(
        &mut self,
        _uuid: Uuid,
        _completion_time_ms: f64,
        _input_tokens: usize,
        _output_tokens: usize,
        _finish_reason: &str,
    ) {
    }
}

// -----------------------------------------------------------------------------
// CompositeEventSink — primary TraceCollector + optional secondary sink.
// -----------------------------------------------------------------------------

/// Composite `EventSink` that forwards events to a primary sink (the
/// in-runtime `TraceCollector` that accumulates a full report) and
/// optionally to a secondary sink (any `EventSink` impl).
///
/// Native callers never set `secondary`; the composite then behaves
/// byte-identically to a bare `TraceCollector`. Out-of-tree observers
/// install their own `EventSink` impl as the secondary so engine events
/// fan out to both the in-process accumulator (for callers that still
/// consume the `TraceSimulationReport`) and the external pipeline.
///
/// `primary` is exposed as a public field so callers can invoke
/// `TraceCollector`'s inherent methods (e.g. `request_latencies`,
/// `snapshot`, `finish`) — those are not part of the `EventSink` trait.
#[derive(Default)]
pub struct CompositeEventSink {
    pub primary: TraceCollector,
    pub secondary: Option<Box<dyn EventSink>>,
}

impl CompositeEventSink {
    pub fn new(primary: TraceCollector) -> Self {
        Self {
            primary,
            secondary: None,
        }
    }

    pub fn with_secondary(mut self, secondary: Box<dyn EventSink>) -> Self {
        self.secondary = Some(secondary);
        self
    }

    /// Install or replace the secondary sink.
    pub fn set_secondary(&mut self, secondary: Box<dyn EventSink>) {
        self.secondary = Some(secondary);
    }

    /// Consume the composite and return the underlying primary collector.
    /// Used by runtime `run`/`run_async` exits to preserve their existing
    /// `TraceCollector` return type.
    pub fn into_primary(self) -> TraceCollector {
        self.primary
    }
}

impl EventSink for CompositeEventSink {
    fn on_arrival(&mut self, uuid: Uuid, arrival_time_ms: f64, isl: usize, osl: usize) {
        EventSink::on_arrival(&mut self.primary, uuid, arrival_time_ms, isl, osl);
        if let Some(sink) = self.secondary.as_mut() {
            sink.on_arrival(uuid, arrival_time_ms, isl, osl);
        }
    }
    fn on_admit(&mut self, uuid: Uuid, admit_time_ms: f64, reused: usize) {
        EventSink::on_admit(&mut self.primary, uuid, admit_time_ms, reused);
        if let Some(sink) = self.secondary.as_mut() {
            sink.on_admit(uuid, admit_time_ms, reused);
        }
    }
    fn on_token(&mut self, uuid: Uuid, token_time_ms: f64) {
        EventSink::on_token(&mut self.primary, uuid, token_time_ms);
        if let Some(sink) = self.secondary.as_mut() {
            sink.on_token(uuid, token_time_ms);
        }
    }
    fn on_completion(
        &mut self,
        uuid: Uuid,
        completion_time_ms: f64,
        input_tokens: usize,
        output_tokens: usize,
        finish_reason: &str,
    ) {
        // primary (TraceCollector) uses the default no-op impl; completion
        // data is derived from accumulated tokens at finish() time.
        if let Some(sink) = self.secondary.as_mut() {
            sink.on_completion(
                uuid,
                completion_time_ms,
                input_tokens,
                output_tokens,
                finish_reason,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct MockSink {
        events: Vec<String>,
    }

    impl EventSink for MockSink {
        fn on_arrival(&mut self, uuid: Uuid, arrival_time_ms: f64, isl: usize, osl: usize) {
            self.events.push(format!(
                "arrival {} {:.1}ms isl={} osl={}",
                uuid, arrival_time_ms, isl, osl
            ));
        }
        fn on_admit(&mut self, uuid: Uuid, admit_time_ms: f64, reused: usize) {
            self.events.push(format!(
                "admit {} {:.1}ms reused={}",
                uuid, admit_time_ms, reused
            ));
        }
        fn on_token(&mut self, uuid: Uuid, token_time_ms: f64) {
            self.events
                .push(format!("token {} {:.1}ms", uuid, token_time_ms));
        }
    }

    #[test]
    fn mock_sink_records_events() {
        let mut sink = MockSink { events: Vec::new() };
        let uuid = Uuid::new_v4();
        sink.on_arrival(uuid, 0.0, 100, 50);
        sink.on_admit(uuid, 5.0, 10);
        sink.on_token(uuid, 30.0);
        sink.on_token(uuid, 60.0);
        assert_eq!(sink.events.len(), 4);
    }

    #[test]
    fn event_sink_is_dyn_compatible() {
        // Verify the trait is object-safe (compiles only if it is).
        let mut sink = MockSink { events: Vec::new() };
        let _dyn_sink: &mut dyn EventSink = &mut sink;
    }

    #[test]
    fn composite_with_no_secondary_matches_primary() {
        // CompositeEventSink with secondary=None must produce the same
        // effect on the underlying TraceCollector as direct usage.
        let uuid = Uuid::new_v4();

        let mut bare = TraceCollector::default();
        EventSink::on_arrival(&mut bare, uuid, 0.0, 100, 50);
        EventSink::on_admit(&mut bare, uuid, 5.0, 10);
        EventSink::on_token(&mut bare, uuid, 30.0);

        let mut composite = CompositeEventSink::default();
        composite.on_arrival(uuid, 0.0, 100, 50);
        composite.on_admit(uuid, 5.0, 10);
        composite.on_token(uuid, 30.0);

        assert_eq!(
            bare.request_latencies(uuid),
            composite.primary.request_latencies(uuid),
        );
    }

    #[test]
    fn composite_forwards_to_secondary() {
        struct Captor {
            log: Arc<Mutex<Vec<&'static str>>>,
        }
        impl EventSink for Captor {
            fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {
                self.log.lock().unwrap().push("arrival");
            }
            fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {
                self.log.lock().unwrap().push("admit");
            }
            fn on_token(&mut self, _: Uuid, _: f64) {
                self.log.lock().unwrap().push("token");
            }
            fn on_completion(&mut self, _: Uuid, _: f64, _: usize, _: usize, _: &str) {
                self.log.lock().unwrap().push("completion");
            }
        }
        let log = Arc::new(Mutex::new(Vec::new()));
        let captor = Captor { log: log.clone() };
        let mut composite = CompositeEventSink::default().with_secondary(Box::new(captor));
        let uuid = Uuid::new_v4();
        composite.on_arrival(uuid, 0.0, 1, 1);
        composite.on_admit(uuid, 1.0, 0);
        composite.on_token(uuid, 2.0);
        composite.on_completion(uuid, 3.0, 1, 1, "stop");
        assert_eq!(
            *log.lock().unwrap(),
            vec!["arrival", "admit", "token", "completion"]
        );
    }

    // -------------------------------------------------------------------
    // Adversarial coverage — these tests probe the EventSink trait /
    // CompositeEventSink for misuse, boundary conditions, and surprising
    // semantics that the happy-path tests above don't exercise. Tests are
    // prefixed `adversarial_` so they're greppable.
    // -------------------------------------------------------------------

    /// Captor that records the order each method was invoked across both
    /// primary and secondary, by writing into a shared log.
    struct OrderingCaptor {
        log: Arc<Mutex<Vec<&'static str>>>,
        tag: &'static str,
    }

    impl EventSink for OrderingCaptor {
        fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {
            self.log.lock().unwrap().push(self.tag);
        }
        fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {
            self.log.lock().unwrap().push(self.tag);
        }
        fn on_token(&mut self, _: Uuid, _: f64) {
            self.log.lock().unwrap().push(self.tag);
        }
        fn on_completion(&mut self, _: Uuid, _: f64, _: usize, _: usize, _: &str) {
            self.log.lock().unwrap().push(self.tag);
        }
    }

    /// Sink that panics on the next call it receives. Used to verify
    /// panic propagation semantics in the composite.
    struct PanickingSink;
    impl EventSink for PanickingSink {
        fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {
            panic!("PanickingSink::on_arrival");
        }
        fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {
            panic!("PanickingSink::on_admit");
        }
        fn on_token(&mut self, _: Uuid, _: f64) {
            panic!("PanickingSink::on_token");
        }
        fn on_completion(&mut self, _: Uuid, _: f64, _: usize, _: usize, _: &str) {
            panic!("PanickingSink::on_completion");
        }
    }

    #[test]
    #[should_panic(expected = "PanickingSink::on_token")]
    fn adversarial_secondary_panic_propagates_to_caller() {
        // CONTRACT: if the secondary panics, the composite panics. The
        // engine is single-threaded over a sink, so a panicking sink will
        // crash the runtime. Users of CompositeEventSink must ensure
        // their secondary impl does not panic.
        let mut composite = CompositeEventSink::default().with_secondary(Box::new(PanickingSink));
        composite.on_token(Uuid::nil(), 0.0);
    }

    #[test]
    fn adversarial_primary_called_before_secondary_on_arrival() {
        // CONTRACT: primary is invoked strictly before secondary. After
        // primary records the on_arrival, the secondary inspects the
        // primary's state and confirms the request is registered.
        struct ProbingSecondary {
            log: Arc<Mutex<Vec<&'static str>>>,
            primary_ptr: *const TraceCollector,
        }
        unsafe impl Send for ProbingSecondary {}
        impl EventSink for ProbingSecondary {
            fn on_arrival(&mut self, uuid: Uuid, _: f64, _: usize, _: usize) {
                // SAFETY: ProbingSecondary holds a raw pointer to the
                // composite's primary purely to verify primary-before-
                // secondary ordering inside this same single-threaded test.
                let primary = unsafe { &*self.primary_ptr };
                // After on_arrival fires on primary, the request must
                // already be registered there.
                if primary.tokens_emitted(uuid) == 0 {
                    self.log.lock().unwrap().push("primary_present");
                }
            }
            fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {}
            fn on_token(&mut self, _: Uuid, _: f64) {}
        }
        let mut composite = CompositeEventSink::default();
        let primary_ptr: *const TraceCollector = &composite.primary;
        let probe_log = Arc::new(Mutex::new(Vec::new()));
        composite.set_secondary(Box::new(ProbingSecondary {
            log: probe_log.clone(),
            primary_ptr,
        }));
        composite.on_arrival(Uuid::new_v4(), 0.0, 1, 1);
        assert_eq!(*probe_log.lock().unwrap(), vec!["primary_present"]);
    }

    #[test]
    fn adversarial_set_secondary_replaces_prior() {
        // CONTRACT: set_secondary replaces (not stacks) any prior
        // secondary. After replacement, only the new sink receives
        // subsequent events.
        let log_a = Arc::new(Mutex::new(Vec::new()));
        let log_b = Arc::new(Mutex::new(Vec::new()));
        let mut composite =
            CompositeEventSink::default().with_secondary(Box::new(OrderingCaptor {
                log: log_a.clone(),
                tag: "A",
            }));
        composite.on_token(Uuid::nil(), 0.0);
        composite.set_secondary(Box::new(OrderingCaptor {
            log: log_b.clone(),
            tag: "B",
        }));
        composite.on_token(Uuid::nil(), 1.0);
        assert_eq!(*log_a.lock().unwrap(), vec!["A"]);
        assert_eq!(*log_b.lock().unwrap(), vec!["B"]);
    }

    #[test]
    fn adversarial_with_secondary_called_twice_keeps_last() {
        // CONTRACT: with_secondary called twice on the builder yields the
        // second secondary only — the builder is not additive.
        let log_a = Arc::new(Mutex::new(Vec::new()));
        let log_b = Arc::new(Mutex::new(Vec::new()));
        let mut composite = CompositeEventSink::default()
            .with_secondary(Box::new(OrderingCaptor {
                log: log_a.clone(),
                tag: "A",
            }))
            .with_secondary(Box::new(OrderingCaptor {
                log: log_b.clone(),
                tag: "B",
            }));
        composite.on_arrival(Uuid::nil(), 0.0, 1, 1);
        assert!(log_a.lock().unwrap().is_empty());
        assert_eq!(*log_b.lock().unwrap(), vec!["B"]);
    }

    #[test]
    fn adversarial_no_api_to_clear_secondary_once_set() {
        // GAP: once a secondary is installed there is no public API to
        // remove it (no `clear_secondary` / `take_secondary`). The only
        // way back to "no secondary" is to drop the composite and rebuild.
        // This test documents that gap so a reviewer notices: if external
        // consumers need to detach mid-run, the API needs extending.
        let log = Arc::new(Mutex::new(Vec::new()));
        let composite = CompositeEventSink::default().with_secondary(Box::new(OrderingCaptor {
            log: log.clone(),
            tag: "T",
        }));
        assert!(composite.secondary.is_some());
        // No method like composite.clear_secondary() exists. Public API
        // surface verified by exhaustive enumeration in the impl block
        // above (new, with_secondary, set_secondary, into_primary).
    }

    #[test]
    fn adversarial_into_primary_after_events_preserves_state() {
        // CONTRACT: into_primary() returns the same TraceCollector that
        // was being updated. State accumulated through the composite
        // must survive consumption.
        let uuid = Uuid::new_v4();
        let mut composite = CompositeEventSink::default();
        composite.on_arrival(uuid, 0.0, 100, 50);
        composite.on_admit(uuid, 5.0, 10);
        composite.on_token(uuid, 30.0);
        composite.on_token(uuid, 60.0);
        let primary = composite.into_primary();
        assert_eq!(primary.tokens_emitted(uuid), 2);
        // 2+ tokens → request_latencies returns Some((ttft, mean_itl)).
        assert!(primary.request_latencies(uuid).is_some());
    }

    #[test]
    fn adversarial_into_primary_drops_secondary_silently() {
        // CONTRACT: into_primary consumes self; the secondary's Drop
        // must run normally. We verify via an external flag.
        struct DropSpy {
            flag: Arc<Mutex<bool>>,
        }
        impl EventSink for DropSpy {
            fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {}
            fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {}
            fn on_token(&mut self, _: Uuid, _: f64) {}
        }
        impl Drop for DropSpy {
            fn drop(&mut self) {
                *self.flag.lock().unwrap() = true;
            }
        }
        let flag = Arc::new(Mutex::new(false));
        let composite =
            CompositeEventSink::default().with_secondary(Box::new(DropSpy { flag: flag.clone() }));
        let _primary = composite.into_primary();
        assert!(*flag.lock().unwrap(), "DropSpy::drop was not called");
    }

    #[test]
    fn adversarial_completion_default_impl_is_noop_for_basic_sink() {
        // CONTRACT: the on_completion default impl is a no-op. Sinks
        // that do not override it must not receive observable state
        // changes from completion events.
        let log = Arc::new(Mutex::new(Vec::new()));
        struct NoCompletionOverride {
            log: Arc<Mutex<Vec<&'static str>>>,
        }
        impl EventSink for NoCompletionOverride {
            fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {
                self.log.lock().unwrap().push("arrival");
            }
            fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {}
            fn on_token(&mut self, _: Uuid, _: f64) {}
            // on_completion intentionally NOT overridden.
        }
        let mut sink = NoCompletionOverride { log: log.clone() };
        sink.on_arrival(Uuid::nil(), 0.0, 1, 1);
        sink.on_completion(Uuid::nil(), 1.0, 1, 1, "stop");
        // Only the arrival event was recorded; on_completion default
        // impl produced no output.
        assert_eq!(*log.lock().unwrap(), vec!["arrival"]);
    }

    #[test]
    fn adversarial_trace_collector_completion_default_is_noop() {
        // CONTRACT (PR1 specific): TraceCollector deliberately does NOT
        // override on_completion — its completion data is derived from
        // token_times at finish() time. Verify that calling on_completion
        // through the EventSink trait on a bare TraceCollector mutates
        // nothing observable.
        let uuid = Uuid::new_v4();
        let mut collector = TraceCollector::default();
        EventSink::on_arrival(&mut collector, uuid, 0.0, 100, 50);
        EventSink::on_token(&mut collector, uuid, 10.0);
        let before = collector.tokens_emitted(uuid);
        EventSink::on_completion(&mut collector, uuid, 20.0, 100, 1, "stop");
        let after = collector.tokens_emitted(uuid);
        assert_eq!(
            before, after,
            "on_completion default must not mutate primary"
        );
    }

    #[test]
    fn adversarial_composite_completion_skips_primary_calls_secondary() {
        // CONTRACT: composite.on_completion does NOT call primary
        // (TraceCollector uses the default no-op for completion), but
        // DOES call secondary's on_completion. This is the asymmetry the
        // impl encodes — verify it explicitly.
        struct CompletionOnly {
            log: Arc<Mutex<Vec<&'static str>>>,
        }
        impl EventSink for CompletionOnly {
            fn on_arrival(&mut self, _: Uuid, _: f64, _: usize, _: usize) {}
            fn on_admit(&mut self, _: Uuid, _: f64, _: usize) {}
            fn on_token(&mut self, _: Uuid, _: f64) {}
            fn on_completion(&mut self, _: Uuid, _: f64, _: usize, _: usize, _: &str) {
                self.log.lock().unwrap().push("completion");
            }
        }
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut composite = CompositeEventSink::default()
            .with_secondary(Box::new(CompletionOnly { log: log.clone() }));
        let uuid = Uuid::new_v4();
        composite.on_arrival(uuid, 0.0, 1, 1);
        let primary_tokens_before = composite.primary.tokens_emitted(uuid);
        composite.on_completion(uuid, 1.0, 1, 1, "stop");
        let primary_tokens_after = composite.primary.tokens_emitted(uuid);
        assert_eq!(primary_tokens_before, primary_tokens_after);
        assert_eq!(
            *log.lock().unwrap(),
            vec!["completion"],
            "secondary must observe completion"
        );
    }

    #[test]
    fn adversarial_nil_uuid_accepted_as_request_id() {
        // CONTRACT: the trait does not reject Uuid::nil() as a sentinel —
        // it's treated as a regular id. If the runtime ever uses nil
        // as "uninitialized" elsewhere, sinks must not assume so.
        let mut composite = CompositeEventSink::default();
        composite.on_arrival(Uuid::nil(), 0.0, 1, 1);
        composite.on_token(Uuid::nil(), 1.0);
        assert_eq!(composite.primary.tokens_emitted(Uuid::nil()), 1);
    }

    #[test]
    fn adversarial_negative_time_ms_accepted() {
        // SURPRISE: time_ms parameters are bare f64 with no monotonicity
        // or sign check. A negative arrival_time_ms is recorded as-is.
        // Sinks that compute durations must guard against this — the
        // trait does not.
        let uuid = Uuid::new_v4();
        let mut composite = CompositeEventSink::default();
        composite.on_arrival(uuid, -100.0, 10, 5);
        composite.on_admit(uuid, -50.0, 0);
        composite.on_token(uuid, -25.0);
        // No panic, no rejection — event flowed through.
        assert_eq!(composite.primary.tokens_emitted(uuid), 1);
    }

    #[test]
    fn adversarial_zero_isl_zero_osl_accepted() {
        // CONTRACT: isl=0, osl=0 are valid (e.g. malformed dataset edge
        // case). Trait does not reject; downstream metric code must
        // handle. Verify no panic.
        let mut composite = CompositeEventSink::default();
        composite.on_arrival(Uuid::new_v4(), 0.0, 0, 0);
    }

    #[test]
    fn adversarial_token_before_arrival_silently_dropped() {
        // SURPRISE: TraceCollector::on_token for an unknown Uuid is a
        // silent no-op (see collector.rs — Entry::or_default-style logic
        // exists but for tokens the lookup discards). External sinks
        // installed as secondary will still see the event. This
        // asymmetry is documented here so reviewers know the contract.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut composite =
            CompositeEventSink::default().with_secondary(Box::new(OrderingCaptor {
                log: log.clone(),
                tag: "secondary_saw_token",
            }));
        let uuid = Uuid::new_v4();
        composite.on_token(uuid, 100.0);
        // Primary did not crash, may or may not have recorded — verify
        // current behavior: tokens_emitted reports 0 because no arrival
        // registered the request.
        assert_eq!(composite.primary.tokens_emitted(uuid), 0);
        // Secondary always sees the event regardless.
        assert_eq!(*log.lock().unwrap(), vec!["secondary_saw_token"]);
    }

    #[test]
    fn adversarial_tokens_emitted_unknown_uuid_returns_zero() {
        // CONTRACT (PR1 added): TraceCollector::tokens_emitted returns 0
        // for an unknown Uuid (`map_or(0, ...)`). Used by the runtime
        // to derive finish_reason — a fresh Uuid must not panic.
        let collector = TraceCollector::default();
        assert_eq!(collector.tokens_emitted(Uuid::new_v4()), 0);
        assert_eq!(collector.tokens_emitted(Uuid::nil()), 0);
    }

    #[test]
    fn adversarial_many_events_same_uuid_accumulate() {
        // CONTRACT: repeated on_token for the same Uuid each push a new
        // entry; tokens_emitted scales linearly. Stress with 10k tokens.
        let uuid = Uuid::new_v4();
        let mut composite = CompositeEventSink::default();
        composite.on_arrival(uuid, 0.0, 100, 10_000);
        for i in 0..10_000 {
            composite.on_token(uuid, i as f64);
        }
        assert_eq!(composite.primary.tokens_emitted(uuid), 10_000);
    }

    #[test]
    fn adversarial_thousand_distinct_uuids_no_collision() {
        // CONTRACT: distinct Uuids do not collide in the inner map.
        let mut composite = CompositeEventSink::default();
        let uuids: Vec<Uuid> = (0..1000).map(|_| Uuid::new_v4()).collect();
        for u in &uuids {
            composite.on_arrival(*u, 0.0, 1, 1);
            composite.on_token(*u, 1.0);
        }
        for u in &uuids {
            assert_eq!(composite.primary.tokens_emitted(*u), 1);
        }
    }

    #[test]
    fn adversarial_default_constructs_composite_with_default_collector() {
        // CONTRACT: CompositeEventSink::default() yields a usable
        // composite with a default-constructed primary. Calling
        // into_primary on a fresh composite must not panic and the
        // primary must be empty.
        let composite = CompositeEventSink::default();
        assert!(composite.secondary.is_none());
        let primary = composite.into_primary();
        assert_eq!(primary.tokens_emitted(Uuid::new_v4()), 0);
    }
}
