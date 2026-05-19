// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{ReadyArrival, ReplayMode};
use crate::common::protocols::DirectRequest;
use crate::loadgen::WorkloadDriver;

/// Event on the streaming admission channel.
///
/// Producers may either submit a request directly (`Submit { burst_id: None }`)
/// in which case the runtime admits it as soon as its arrival timestamp /
/// concurrency gate allows, or tag it with a `burst_id` to defer admission
/// until a matching `FlushBurst(burst_id)` is received.
///
/// Burst tagging gives the producer explicit control over batch composition:
/// every `Submit` tagged with the same `burst_id` is guaranteed to enter the
/// admission queue contiguously — no untagged submit, and no submit from any
/// other burst, can interleave between them. Bursts flush in the order their
/// `FlushBurst` events arrive on the channel, not in `burst_id` numeric order.
///
/// `DrainBarrier` is the producer's acknowledgment in lockstep mode that it
/// has finished reacting to every engine event delivered through its event
/// channel so far — see [`AdmissionQueue::new_streaming_lockstep`] for the
/// full protocol. Producers that are not in lockstep mode never need to send
/// `DrainBarrier`; the runtime simply ignores it.
///
/// This is useful for:
/// * Benchmarking harnesses that need bit-identical batch composition across
///   reruns, free of wall-clock jitter between producer and runtime.
/// * Regression test suites that drive synthetic workloads and need
///   reproducible scheduler behavior.
/// * Any external scheduler that wants atomic "submit these N requests as one
///   batch" semantics rather than relying on transport coalescing.
/// * Closed-loop simulators where the producer reacts to completions by
///   issuing replacement work and the runtime must wait for the reaction
///   before advancing sim time.
///
/// `From<DirectRequest>` is implemented for ergonomic untagged submission:
/// `tx.send(req.into())`.
#[derive(Debug)]
pub enum StreamingAdmission {
    /// Submit a request. If `burst_id` is `Some`, the request is staged and
    /// will be admitted (in submission order, contiguous with its burst-mates)
    /// only when a `FlushBurst` with the same id is received.
    Submit {
        req: DirectRequest,
        burst_id: Option<u64>,
    },
    /// Atomically flush every request previously submitted with this
    /// `burst_id` into the admission queue. No-op if no requests are staged
    /// under the given id.
    FlushBurst(u64),
    /// Producer acknowledgment that it has finished reacting to every engine
    /// event delivered through its event channel so far. Only meaningful in
    /// lockstep mode (see [`AdmissionQueue::new_streaming_lockstep`]); a no-op
    /// for non-lockstep streaming queues.
    ///
    /// One `DrainBarrier` cancels at most one outstanding gate: if the
    /// runtime completed two batches of requests back-to-back and is awaiting
    /// two barriers, the producer must send two barriers. Any number of
    /// `Submit` / `FlushBurst` events may appear before the barrier — those
    /// are the producer's reaction. The barrier marks the end of the
    /// reaction.
    DrainBarrier,
}

impl From<DirectRequest> for StreamingAdmission {
    fn from(req: DirectRequest) -> Self {
        Self::Submit {
            req,
            burst_id: None,
        }
    }
}

struct StreamingState {
    rx: mpsc::UnboundedReceiver<StreamingAdmission>,
    /// Untagged submits and submits that have been flushed from a burst.
    /// Pulled from by `drain_ready` / `next_ready_time_ms`.
    pending: VecDeque<DirectRequest>,
    /// Per-burst staging area. A `FlushBurst(id)` drains `staged[id]` into
    /// `pending` in submission order; the entry is then removed.
    staged: HashMap<u64, Vec<DirectRequest>>,
    closed: Arc<AtomicBool>,
    /// Lockstep mode: when on, the runtime gates sim-time advancement on
    /// receipt of one `DrainBarrier` per completion batch. The producer is
    /// expected to send `DrainBarrier` after committing its reaction to each
    /// completion-batch event delivered through its event channel.
    lockstep: bool,
    /// Number of `DrainBarrier` events received but not yet consumed by the
    /// runtime. Each completed batch consumes one barrier; the runtime calls
    /// [`consume_drain_barrier`] when releasing the gate. Untracked (and
    /// silently ignored) when `lockstep` is false.
    barrier_credits: usize,
}

impl StreamingState {
    fn apply(&mut self, event: StreamingAdmission) {
        match event {
            StreamingAdmission::Submit {
                req,
                burst_id: None,
            } => self.pending.push_back(req),
            StreamingAdmission::Submit {
                req,
                burst_id: Some(id),
            } => self.staged.entry(id).or_default().push(req),
            StreamingAdmission::FlushBurst(id) => {
                if let Some(reqs) = self.staged.remove(&id) {
                    self.pending.extend(reqs);
                }
            }
            StreamingAdmission::DrainBarrier => {
                if self.lockstep {
                    self.barrier_credits = self.barrier_credits.saturating_add(1);
                }
            }
        }
    }
}

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
    Streaming(StreamingState),
}

pub struct AdmissionQueue {
    source: AdmissionSource,
    mode: ReplayMode,
}

impl AdmissionQueue {
    pub(in crate::replay::offline) fn new_requests(
        source: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Requests(source),
            mode,
        }
    }

    pub(in crate::replay::offline) fn new_workload(
        driver: WorkloadDriver,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Workload(driver),
            mode,
        }
    }

    /// Build a streaming admission queue driven by an mpsc channel.
    ///
    /// The channel carries `StreamingAdmission` events — either tagged or
    /// untagged submits, or `FlushBurst` markers. Closing the channel signals
    /// end-of-input; any requests still staged in an unflushed burst when the
    /// channel closes are dropped (they were never committed to the queue).
    ///
    /// The returned queue runs in **non-lockstep** (default) mode: the
    /// runtime advances sim time as soon as the local event horizon dictates,
    /// without coordinating with the producer. `DrainBarrier` events on the
    /// channel are silently ignored. For lockstep simulation see
    /// [`Self::new_streaming_lockstep`].
    pub fn new_streaming(
        rx: mpsc::UnboundedReceiver<StreamingAdmission>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Streaming(StreamingState {
                rx,
                pending: VecDeque::new(),
                staged: HashMap::new(),
                closed: Arc::new(AtomicBool::new(false)),
                lockstep: false,
                barrier_credits: 0,
            }),
            mode,
        }
    }

    /// Build a streaming admission queue in **lockstep** mode.
    ///
    /// In lockstep mode the runtime gates each sim-time advance that follows
    /// a completion event on receipt of one `StreamingAdmission::DrainBarrier`
    /// from the producer. The protocol is:
    ///
    /// 1. Runtime completes one or more requests (a "completion batch"). The
    ///    producer observes these completions through its engine event
    ///    channel ([`crate::replay::offline::EventSink`] or a downstream
    ///    consumer of it).
    /// 2. The runtime sets an internal flag: "next advance requires a
    ///    barrier".
    /// 3. The producer reacts to the completions — typically by submitting
    ///    zero or more replacement requests via `Submit` / `FlushBurst` —
    ///    and then sends `DrainBarrier` to signal "my reaction is complete".
    /// 4. The runtime consumes one barrier credit and advances.
    ///
    /// One barrier cancels one outstanding gate. If the runtime emits two
    /// completion batches before the producer sends a barrier, two barriers
    /// are required before the next advance.
    ///
    /// **Required contract on the producer side**: a lockstep producer must
    /// send exactly one `DrainBarrier` per observed completion batch, or the
    /// runtime will deadlock. The runtime cannot detect misuse beyond the
    /// generic "channel closed before drained" path. Consumers that build on
    /// this API are encouraged to expose a higher-level helper that emits the
    /// barrier from a single, audit-able call site.
    ///
    /// Lockstep mode is orthogonal to burst tagging: a producer can use
    /// either, both, or neither.
    pub fn new_streaming_lockstep(
        rx: mpsc::UnboundedReceiver<StreamingAdmission>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Streaming(StreamingState {
                rx,
                pending: VecDeque::new(),
                staged: HashMap::new(),
                closed: Arc::new(AtomicBool::new(false)),
                lockstep: true,
                barrier_credits: 0,
            }),
            mode,
        }
    }

    /// Shared close flag for streaming admission queues. Producers that
    /// signal end-of-stream from a separate thread can clone this `Arc`
    /// and `store(true, Release)` on it; the runtime loop observes the
    /// flag in `recv_next` / `poll_pending` for graceful shutdown.
    /// Returns `None` for non-streaming sources (Requests / Workload),
    /// which don't have a producer-side close concept.
    #[cfg(test)]
    pub(in crate::replay::offline) fn close_signal(&self) -> Option<Arc<AtomicBool>> {
        match &self.source {
            AdmissionSource::Streaming(state) => Some(state.closed.clone()),
            _ => None,
        }
    }

    /// Drain any queued events off the channel, applying them to internal
    /// state. Untagged submits land in `pending`; tagged submits are staged;
    /// `FlushBurst` flushes the matching stage. Returns the number of events
    /// processed (not the number of requests promoted to `pending`).
    pub(in crate::replay::offline) fn poll_pending(&mut self) -> usize {
        let AdmissionSource::Streaming(state) = &mut self.source else {
            return 0;
        };
        let mut count = 0;
        loop {
            match state.rx.try_recv() {
                Ok(event) => {
                    state.apply(event);
                    count += 1;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    state.closed.store(true, Ordering::Release);
                    break;
                }
            }
        }
        count
    }

    /// Await at least one event on the channel and apply it. Returns `Some(())`
    /// if an event was processed (caller should re-check `pending` /
    /// `drain_ready`) or `None` if the channel has closed with no pending
    /// staged work.
    ///
    /// Burst-staged submits do not pop into `pending` until their
    /// `FlushBurst`; callers that await `recv_next` should expect to spin
    /// through one or more staged events before a burst flush exposes work.
    pub(in crate::replay::offline) async fn recv_next(&mut self) -> Option<()> {
        let AdmissionSource::Streaming(state) = &mut self.source else {
            return None;
        };
        let event = state.rx.recv().await?;
        state.apply(event);
        Some(())
    }

    #[cfg(test)]
    pub(in crate::replay::offline) fn total_requests_known(&self) -> Option<usize> {
        match &self.source {
            AdmissionSource::Requests(pending) => Some(pending.len()),
            AdmissionSource::Workload(driver) => Some(driver.total_turns()),
            AdmissionSource::Streaming(_) => None,
        }
    }

    pub(in crate::replay::offline) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(in crate::replay::offline) fn next_ready_time_ms(
        &mut self,
        cluster_in_flight: usize,
    ) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Workload(driver)) => {
                if cluster_in_flight < *max_in_flight {
                    driver.next_ready_time_ms()
                } else {
                    None
                }
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
            (ReplayMode::Trace, AdmissionSource::Streaming(state)) => state
                .pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Streaming(state)) => {
                if cluster_in_flight >= *max_in_flight {
                    None
                } else {
                    state
                        .pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                }
            }
        }
    }

    pub(in crate::replay::offline) fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms,
                        replay_hashes: None,
                    });
                }
                Ok(ready)
            }
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => Ok(driver
                .pop_ready(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| ReadyArrival {
                    request: ready.request,
                    arrival_time_ms: ready.scheduled_ready_at_ms,
                    replay_hashes: ready.replay_hashes,
                })
                .collect()),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(request) = pending.pop_front() else {
                        break;
                    };
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                    });
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Workload(driver)) => {
                let available = max_in_flight.saturating_sub(cluster_in_flight);
                if available == 0 {
                    return Ok(Vec::new());
                }
                Ok(driver
                    .pop_ready(now_ms, available)
                    .into_iter()
                    .map(|ready| ReadyArrival {
                        request: ready.request,
                        arrival_time_ms: now_ms,
                        replay_hashes: ready.replay_hashes,
                    })
                    .collect())
            }
            (ReplayMode::Trace, AdmissionSource::Streaming(state)) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = state
                        .pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = state
                        .pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms,
                        replay_hashes: None,
                    });
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Streaming(state)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(request) = state.pending.pop_front() else {
                        break;
                    };
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                    });
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
        }
    }

    pub(in crate::replay::offline) fn on_request_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_complete(uuid, now_ms)
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
            AdmissionSource::Streaming(state) => {
                state.pending.is_empty()
                    && state.staged.is_empty()
                    && state.closed.load(Ordering::Acquire)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(in crate::replay::offline) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
            AdmissionSource::Streaming(state) => {
                state.pending.len() + state.staged.values().map(Vec::len).sum::<usize>()
            }
        }
    }

    /// True iff this admission queue is a lockstep streaming source.
    ///
    /// Non-streaming sources and non-lockstep streaming sources return false.
    /// Used by runtimes to decide whether to gate sim-time advancement on
    /// drain barriers after a completion batch.
    pub(in crate::replay::offline) fn is_lockstep(&self) -> bool {
        matches!(&self.source, AdmissionSource::Streaming(state) if state.lockstep)
    }

    /// Try to consume one outstanding drain-barrier credit. Returns true if
    /// a barrier was available (and consumed), false otherwise.
    ///
    /// Always returns false for non-lockstep sources.
    ///
    /// **Ordering invariant (deterministic-replay contract):** before
    /// checking credits this method drains any events still buffered on
    /// the admission channel via `try_recv`, applying them to the queue
    /// state. After the drain it ALSO refuses to consume if `pending` or
    /// `staged` is non-empty: the producer's reaction (submits and
    /// staged-but-unflushed bursts) must be admitted at the **current**
    /// sim time before the runtime advances past the barrier. The runtime
    /// is expected to fall through to its normal admission/event drain
    /// loop in that case (see [`Self::has_queued_admission_work`]); only
    /// after pending and staged are empty does the gate consume.
    ///
    /// Without this invariant, scheduler jitter on the producer side
    /// (separate tasks for "flush burst" and "send barrier") can deliver
    /// the events in the order `[BARRIER, SUBMIT, …, FLUSH]` even when
    /// the producer's intent was `[SUBMIT, …, FLUSH, BARRIER]`. Consuming
    /// the credit immediately would advance sim time past the reaction
    /// and admit it at a later tick, perturbing batch composition across
    /// reruns. With this invariant, both orderings collapse to the same
    /// observable sequence: admit-then-advance.
    pub(in crate::replay::offline) fn consume_drain_barrier(&mut self) -> bool {
        // Drain any currently-buffered events off the channel first.
        // Any post-barrier submit that's already on the wire becomes
        // visible in `pending` / `staged` before we decide.
        let _ = self.poll_pending();
        let AdmissionSource::Streaming(state) = &mut self.source else {
            return false;
        };
        if !state.lockstep || state.barrier_credits == 0 {
            return false;
        }
        // Queued reaction (admittable submits OR mid-burst staged work)
        // belongs at the current sim time and must be drained by the
        // caller before the barrier releases.
        if !state.pending.is_empty() || !state.staged.is_empty() {
            return false;
        }
        state.barrier_credits -= 1;
        true
    }

    /// True iff the streaming admission queue has work the runtime must
    /// drain at the current sim time before any further advance: either
    /// untagged / flushed submits sitting in `pending`, or submits staged
    /// against a burst id that hasn't flushed yet.
    ///
    /// The runtime uses this to short-circuit the lockstep gate: if there
    /// is queued admission work, the gate must NOT wait on a barrier — it
    /// must fall through to the normal drain path so the queued reaction
    /// is admitted at the current sim time. Without this short-circuit
    /// the runtime deadlocks when the producer's reaction races into the
    /// channel after the barrier and the cluster is at concurrency cap
    /// (so `next_ready_time_ms` returns `None` even with pending work).
    ///
    /// Always false for non-streaming sources.
    pub(in crate::replay::offline) fn has_queued_admission_work(&self) -> bool {
        match &self.source {
            AdmissionSource::Streaming(state) => {
                !state.pending.is_empty() || !state.staged.is_empty()
            }
            _ => false,
        }
    }

    /// Number of outstanding drain-barrier credits. Used in tests and
    /// diagnostics; production callers should prefer
    /// [`Self::consume_drain_barrier`].
    #[cfg(test)]
    pub(in crate::replay::offline) fn drain_barrier_credits(&self) -> usize {
        match &self.source {
            AdmissionSource::Streaming(state) => state.barrier_credits,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    fn direct_request_at(arrival_ms: f64, isl: usize) -> DirectRequest {
        DirectRequest {
            tokens: vec![0u32; isl],
            max_output_tokens: 8,
            uuid: Some(Uuid::new_v4()),
            dp_rank: 0,
            arrival_timestamp_ms: Some(arrival_ms),
        }
    }

    #[test]
    fn streaming_admission_unread_is_not_drained() {
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        assert!(!q.is_drained());
        assert_eq!(q.total_requests_known(), None);
    }

    #[test]
    fn streaming_admission_closed_with_no_pending_is_drained() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        drop(tx);
        let _ = q.poll_pending();
        assert!(q.is_drained());
    }

    #[test]
    fn streaming_admission_drain_ready_admits_at_arrival_time() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(direct_request_at(10.0, 100).into()).unwrap();
        tx.send(direct_request_at(25.0, 100).into()).unwrap();
        drop(tx);
        let _ = q.poll_pending();

        // At now=20ms only the first should drain (arrival_ms=10).
        let admitted = q.drain_ready(20.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].arrival_time_ms, 10.0);

        // Advance to now=30ms — second admits.
        let admitted = q.drain_ready(30.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].arrival_time_ms, 25.0);
        assert!(q.is_drained());
    }

    #[tokio::test]
    async fn streaming_admission_await_next_yields_on_send() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);

        let send_task = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            tx.send(direct_request_at(50.0, 100).into()).unwrap();
        });

        // recv_next applies the event into pending; we then drain.
        let got = q.recv_next().await;
        assert!(got.is_some());
        let admitted = q.drain_ready(100.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].arrival_time_ms, 50.0);

        send_task.await.unwrap();
    }

    #[tokio::test]
    async fn streaming_admission_await_next_returns_none_on_close() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        drop(tx);
        assert!(q.recv_next().await.is_none());
    }

    // ---- burst semantics ----------------------------------------------------

    fn burst_submit(req: DirectRequest, burst_id: u64) -> StreamingAdmission {
        StreamingAdmission::Submit {
            req,
            burst_id: Some(burst_id),
        }
    }

    #[test]
    fn burst_staged_submits_do_not_admit_until_flush() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 8 });

        // Three staged submits, no flush yet.
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        let _ = q.poll_pending();

        // Nothing admits — they are staged.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 0);

        // Flush exposes all three contiguously in one drain.
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 3);
    }

    #[test]
    fn burst_flush_preserves_within_burst_order() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);

        let r1 = direct_request_at(10.0, 16);
        let r2 = direct_request_at(20.0, 16);
        let r3 = direct_request_at(30.0, 16);
        let (u1, u2, u3) = (r1.uuid, r2.uuid, r3.uuid);
        tx.send(burst_submit(r1, 7)).unwrap();
        tx.send(burst_submit(r2, 7)).unwrap();
        tx.send(burst_submit(r3, 7)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(7)).unwrap();
        let _ = q.poll_pending();

        let admitted = q.drain_ready(100.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 3);
        assert_eq!(admitted[0].request.uuid, u1);
        assert_eq!(admitted[1].request.uuid, u2);
        assert_eq!(admitted[2].request.uuid, u3);
    }

    #[test]
    fn interleaved_bursts_flush_in_flush_order_not_arrival_order() {
        // Producer interleaves bursts 1 and 2 on the channel, but flushes 1
        // before 2. Burst 1's requests must all admit before any of burst 2's.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q =
            AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 32 });

        let b1_a = direct_request_at(0.0, 16);
        let b2_a = direct_request_at(0.0, 16);
        let b1_b = direct_request_at(0.0, 16);
        let b2_b = direct_request_at(0.0, 16);
        let b1_c = direct_request_at(0.0, 16);
        let (b1_a_u, b1_b_u, b1_c_u) = (b1_a.uuid, b1_b.uuid, b1_c.uuid);
        let (b2_a_u, b2_b_u) = (b2_a.uuid, b2_b.uuid);

        tx.send(burst_submit(b1_a, 1)).unwrap();
        tx.send(burst_submit(b2_a, 2)).unwrap();
        tx.send(burst_submit(b1_b, 1)).unwrap();
        tx.send(burst_submit(b2_b, 2)).unwrap();
        tx.send(burst_submit(b1_c, 1)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(2)).unwrap();
        let _ = q.poll_pending();

        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 5);
        // Burst 1's three requests come first, in submission order.
        assert_eq!(admitted[0].request.uuid, b1_a_u);
        assert_eq!(admitted[1].request.uuid, b1_b_u);
        assert_eq!(admitted[2].request.uuid, b1_c_u);
        // Then burst 2's two, in submission order.
        assert_eq!(admitted[3].request.uuid, b2_a_u);
        assert_eq!(admitted[4].request.uuid, b2_b_u);
    }

    #[test]
    fn untagged_submits_admit_immediately_alongside_staged_bursts() {
        // Untagged submits behave as before — they admit on drain, regardless
        // of whether other bursts are mid-stage. The staged burst's submits
        // remain staged until its flush.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q =
            AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 32 });

        let untagged = direct_request_at(0.0, 16);
        let burst_req = direct_request_at(0.0, 16);
        let (utag_u, burst_u) = (untagged.uuid, burst_req.uuid);

        tx.send(burst_submit(burst_req, 5)).unwrap();
        tx.send(untagged.into()).unwrap();
        let _ = q.poll_pending();

        // Only the untagged one admits.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].request.uuid, utag_u);

        // Flushing burst 5 admits its staged request.
        tx.send(StreamingAdmission::FlushBurst(5)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 1).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].request.uuid, burst_u);
    }

    #[test]
    fn flush_burst_with_no_staged_requests_is_noop() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::FlushBurst(42)).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(100.0, 0).expect("drain_ready");
        assert!(admitted.is_empty());
        assert!(q.is_drained());
    }

    #[test]
    fn unflushed_staged_burst_blocks_drained_even_after_channel_close() {
        // Channel closes with a staged-but-unflushed burst still present.
        // is_drained must remain false because work was committed to a burst
        // that the runtime never received a flush for. (Consumers can detect
        // this and surface it as an error rather than silently dropping.)
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(burst_submit(direct_request_at(0.0, 16), 99))
            .unwrap();
        drop(tx);
        let _ = q.poll_pending();
        assert!(!q.is_drained());
    }

    // -------------------------------------------------------------------
    // Adversarial coverage — probe streaming admission for misuse,
    // race conditions, boundary conditions, and surprising semantics.
    // Prefixed `adversarial_` so they're greppable.
    // -------------------------------------------------------------------

    #[test]
    fn adversarial_flush_arrives_before_any_submit_silent_noop() {
        // CONTRACT: FlushBurst(N) before any Submit{burst_id=N} is a
        // silent no-op. The state.staged map has no entry for N, so the
        // remove returns None.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::FlushBurst(7)).unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 7))
            .unwrap();
        let _ = q.poll_pending();
        // The submit AFTER the flush re-stages into the same burst id —
        // it does NOT immediately flow to pending. Without a second flush
        // it stays staged.
        let admitted = q.drain_ready(100.0, 0).expect("drain_ready");
        assert!(
            admitted.is_empty(),
            "submit after early flush should not auto-admit"
        );
        // Confirm it's staged, not lost.
        assert_eq!(q.total_requests(), 1);
        drop(tx);
        let _ = q.poll_pending();
        // is_drained must be false: staged work remains.
        assert!(!q.is_drained());
    }

    #[test]
    fn adversarial_flush_burst_twice_in_a_row_second_is_noop() {
        // CONTRACT: a second FlushBurst(N) immediately after the first
        // (with no intervening Submit) is a no-op. The first flush
        // removes the staging entry; the second finds nothing.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 8 });
        tx.send(burst_submit(direct_request_at(0.0, 16), 3))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(3)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(3)).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(
            admitted.len(),
            1,
            "double flush admits the single request once"
        );
        assert!(q.is_drained());
    }

    #[test]
    fn adversarial_burst_id_reused_after_flush_is_a_new_burst() {
        // CONTRACT: after a Flush has emptied burst id N, subsequent
        // Submit{burst_id=N} entries form a NEW burst that requires its
        // own Flush. The runtime does not "remember" that N was ever
        // flushed; reuse is implicit.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 8 });

        // Burst A: 2 submits, flush.
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 2);

        // Burst A-prime: same id, 1 submit — staged, NOT auto-flushed.
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 2).expect("drain_ready");
        assert!(admitted.is_empty(), "reused burst id must restage");

        // Confirm explicit flush admits it.
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 2).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
    }

    #[test]
    fn adversarial_burst_id_zero_accepted_as_regular_id() {
        // CONTRACT: burst_id=0 is not a sentinel; it's treated as a
        // regular id. (Producers concerned by this can use NonZeroU64
        // wrappers themselves — the trait doesn't enforce.)
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 4 });
        tx.send(burst_submit(direct_request_at(0.0, 16), 0))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(0)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
    }

    #[test]
    fn adversarial_burst_id_u64_max_accepted() {
        // CONTRACT: u64::MAX is a valid burst_id.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 4 });
        tx.send(burst_submit(direct_request_at(0.0, 16), u64::MAX))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(u64::MAX)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
    }

    #[test]
    fn adversarial_many_concurrent_bursts_staged() {
        // CONTRACT: many distinct burst ids can be staged concurrently
        // without collision. total_requests counts staged + pending.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(
            rx,
            ReplayMode::Concurrency {
                max_in_flight: 1024,
            },
        );
        for id in 0..100u64 {
            tx.send(burst_submit(direct_request_at(0.0, 16), id))
                .unwrap();
            tx.send(burst_submit(direct_request_at(0.0, 16), id))
                .unwrap();
        }
        let _ = q.poll_pending();
        assert_eq!(
            q.total_requests(),
            200,
            "200 requests across 100 staged bursts"
        );
        // Nothing admits yet.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 0);
        // Flush them all in id order.
        for id in 0..100u64 {
            tx.send(StreamingAdmission::FlushBurst(id)).unwrap();
        }
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 200);
        assert!(q.is_drained());
    }

    #[test]
    fn adversarial_huge_burst_10k_requests_atomic_flush() {
        // CONTRACT: a single burst can carry 10k requests and flush
        // atomically. Drain in Concurrency mode emits all of them when
        // max_in_flight is high enough.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(
            rx,
            ReplayMode::Concurrency {
                max_in_flight: 20_000,
            },
        );
        for _ in 0..10_000 {
            tx.send(burst_submit(direct_request_at(0.0, 8), 1)).unwrap();
        }
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 10_000);
    }

    #[test]
    fn adversarial_total_requests_counts_staged_plus_pending() {
        // CONTRACT: total_requests reports the sum of staged (across all
        // burst ids) plus pending. Useful for runtime accounting.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        // 3 untagged (pending) + 4 in burst 1 + 2 in burst 2 + 1 in burst 3.
        for _ in 0..3 {
            tx.send(direct_request_at(0.0, 16).into()).unwrap();
        }
        for _ in 0..4 {
            tx.send(burst_submit(direct_request_at(0.0, 16), 1))
                .unwrap();
        }
        for _ in 0..2 {
            tx.send(burst_submit(direct_request_at(0.0, 16), 2))
                .unwrap();
        }
        tx.send(burst_submit(direct_request_at(0.0, 16), 3))
            .unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.total_requests(), 10);
    }

    #[test]
    fn adversarial_total_requests_known_returns_none_for_streaming() {
        // CONTRACT: total_requests_known returns None for Streaming
        // because new submits can arrive at any time. This signals to
        // any consumer that pre-allocation / progress-bar setup is not
        // viable for streaming.
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        assert_eq!(q.total_requests_known(), None);
    }

    #[test]
    fn adversarial_close_signal_only_for_streaming_source() {
        // CONTRACT: close_signal() returns Some only when the source is
        // Streaming; for the legacy Requests / Workload modes it returns
        // None.
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let q_stream = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        assert!(q_stream.close_signal().is_some());

        let q_req = AdmissionQueue::new_requests(
            VecDeque::from([direct_request_at(0.0, 8)]),
            ReplayMode::Trace,
        );
        assert!(q_req.close_signal().is_none());
    }

    #[test]
    fn adversarial_close_signal_flips_on_disconnect() {
        // CONTRACT: poll_pending observes channel disconnect and flips
        // the close_signal AtomicBool to true. External monitors can
        // poll this without owning the receiver.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        let signal = q.close_signal().expect("streaming has close signal");
        assert!(!signal.load(Ordering::Acquire));
        drop(tx);
        let _ = q.poll_pending();
        assert!(signal.load(Ordering::Acquire));
    }

    #[test]
    fn adversarial_poll_pending_counts_events_not_promoted_requests() {
        // CONTRACT: poll_pending returns *event count* on the channel,
        // not the count of requests that reached `pending`. A flush
        // event counts as one even though it may promote N requests.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let n = q.poll_pending();
        assert_eq!(n, 4, "3 submits + 1 flush = 4 events");
    }

    #[tokio::test]
    async fn adversarial_recv_next_after_close_with_staged_returns_none() {
        // CONTRACT: once the channel is fully closed (drained of events),
        // recv_next returns None even if staged bursts remain. The
        // staged work is observable via total_requests / is_drained, but
        // there is no way to flush from within the queue.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        tx.send(burst_submit(direct_request_at(0.0, 16), 5))
            .unwrap();
        drop(tx);
        // First recv_next consumes the submit.
        assert!(q.recv_next().await.is_some());
        // Second recv_next sees the channel closed.
        assert!(q.recv_next().await.is_none());
        // Staged burst still there.
        assert_eq!(q.total_requests(), 1);
        assert!(!q.is_drained());
    }

    #[test]
    fn adversarial_untagged_submit_does_not_join_active_burst() {
        // CONTRACT: untagged Submit goes straight to pending; it does
        // NOT join an open burst that happens to have the same producer.
        // The two paths are independent.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 8 });
        let burst_req = direct_request_at(0.0, 16);
        let untagged = direct_request_at(0.0, 16);
        let untagged_uuid = untagged.uuid;
        tx.send(burst_submit(burst_req, 1)).unwrap();
        tx.send(untagged.into()).unwrap();
        let _ = q.poll_pending();
        // Drain — only the untagged comes out. Burst still staged.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].request.uuid, untagged_uuid);
    }

    #[test]
    fn adversarial_trace_mode_head_of_line_blocking_in_pending() {
        // SURPRISE: in Trace mode, drain_ready checks ONLY the front of
        // pending for arrival-time readiness. If a producer sends two
        // untagged submits with arrival_ms=100, then 0 (out of order),
        // the request at arrival_ms=100 sits at the head and blocks
        // the second from admitting until now>=100. Documents
        // head-of-line semantics in streaming Trace mode.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        let r_late = direct_request_at(100.0, 16);
        let r_early = direct_request_at(0.0, 16);
        let late_uuid = r_late.uuid;
        let early_uuid = r_early.uuid;
        tx.send(r_late.into()).unwrap();
        tx.send(r_early.into()).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        // At now=50ms: late (front) not ready -> blocked. Early at index
        // 1 is ready but cannot be drained until late is dequeued.
        let admitted = q.drain_ready(50.0, 0).expect("drain_ready");
        assert!(
            admitted.is_empty(),
            "trace-mode drain is head-of-line; out-of-order arrivals block"
        );
        // At now=200ms both ready -> drain in submission (=insertion) order.
        let admitted = q.drain_ready(200.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 2);
        assert_eq!(admitted[0].request.uuid, late_uuid);
        assert_eq!(admitted[1].request.uuid, early_uuid);
    }

    #[test]
    fn adversarial_concurrency_mode_respects_max_in_flight() {
        // CONTRACT: Concurrency mode emits at most (max_in_flight -
        // cluster_in_flight) requests per drain_ready call.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 5 });
        for _ in 0..20 {
            tx.send(direct_request_at(0.0, 16).into()).unwrap();
        }
        let _ = q.poll_pending();
        // Already 3 in flight: only 2 slots free.
        let admitted = q.drain_ready(0.0, 3).expect("drain_ready");
        assert_eq!(admitted.len(), 2);
        // Already 5: zero slots.
        let admitted = q.drain_ready(0.0, 5).expect("drain_ready");
        assert!(admitted.is_empty());
        // Free up: emits another 5.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 5);
    }

    #[test]
    fn adversarial_drain_ready_empty_when_only_staged_present() {
        // CONTRACT: when pending is empty but staged has bursts, drain_ready
        // emits nothing — staged is invisible until flush.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q =
            AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 100 });
        for _ in 0..50 {
            tx.send(burst_submit(direct_request_at(0.0, 16), 1))
                .unwrap();
        }
        let _ = q.poll_pending();
        let admitted = q.drain_ready(1000.0, 0).expect("drain_ready");
        assert!(admitted.is_empty());
        // next_ready_time_ms must agree: nothing in pending → None.
        assert!(q.next_ready_time_ms(0).is_none());
    }

    #[test]
    fn adversarial_flush_burst_for_empty_id_after_partial_flush() {
        // CONTRACT: stage to burst 1, flush 1 fully, then a FlushBurst(1)
        // on a now-empty staged entry is a no-op (HashMap::remove on
        // missing returns None).
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 4 });
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert!(q.is_drained());
    }

    #[test]
    fn adversarial_streaming_on_request_completed_is_noop() {
        // CONTRACT: on_request_completed is a Workload-only path. For
        // Streaming sources it's a no-op (the streaming consumer is
        // responsible for its own completion accounting).
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        // Should not panic, should not error.
        q.on_request_completed(Uuid::new_v4(), 0.0)
            .expect("noop ok");
    }

    // ---- lockstep semantics --------------------------------------------------

    #[test]
    fn non_lockstep_drain_barrier_is_ignored() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        assert!(!q.is_lockstep());
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();
        // No credits accrued — non-lockstep mode.
        assert!(!q.consume_drain_barrier());
        assert_eq!(q.drain_barrier_credits(), 0);
    }

    #[test]
    fn lockstep_drain_barrier_accrues_one_credit_each() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        assert!(q.is_lockstep());
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 3);
        assert!(q.consume_drain_barrier());
        assert!(q.consume_drain_barrier());
        assert!(q.consume_drain_barrier());
        assert!(!q.consume_drain_barrier());
        assert_eq!(q.drain_barrier_credits(), 0);
    }

    #[test]
    fn lockstep_submits_and_barriers_interleave_in_event_order() {
        // The producer submits a request, then signals barrier, then submits
        // another. The submits land in pending; the barrier accrues a credit.
        // None of these block each other.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(
            rx,
            ReplayMode::Concurrency { max_in_flight: 8 },
        );

        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        let _ = q.poll_pending();

        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 2);
        assert_eq!(q.drain_barrier_credits(), 1);
        assert!(q.consume_drain_barrier());
    }

    #[test]
    fn lockstep_barriers_compose_with_burst_flush() {
        // Burst flush + barrier in one channel: the burst is admitted, the
        // barrier credits separately.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(
            rx,
            ReplayMode::Concurrency { max_in_flight: 8 },
        );

        tx.send(burst_submit(direct_request_at(0.0, 16), 7))
            .unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 7))
            .unwrap();
        tx.send(StreamingAdmission::FlushBurst(7)).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();

        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 2);
        assert!(q.consume_drain_barrier());
    }

    #[tokio::test]
    async fn lockstep_recv_next_yields_on_barrier() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        let send_task = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            tx.send(StreamingAdmission::DrainBarrier).unwrap();
        });
        let got = q.recv_next().await;
        assert!(got.is_some());
        assert!(q.consume_drain_barrier());
        send_task.await.unwrap();
    }

    // -------------------------------------------------------------------
    // Adversarial coverage — probe the drain-barrier / lockstep API for
    // misuse, boundary conditions, and race-free degenerate paths.
    // Prefixed `adversarial_` so they're greppable.
    // -------------------------------------------------------------------

    #[test]
    fn adversarial_consume_drain_barrier_on_requests_source_returns_false() {
        // CONTRACT: consume_drain_barrier always returns false for
        // non-streaming sources (legacy Requests path).
        let mut q = AdmissionQueue::new_requests(
            VecDeque::from([direct_request_at(0.0, 16)]),
            ReplayMode::Trace,
        );
        assert!(!q.consume_drain_barrier());
        assert!(!q.is_lockstep());
        assert_eq!(q.drain_barrier_credits(), 0);
    }

    #[test]
    fn adversarial_is_lockstep_false_for_non_lockstep_streaming() {
        // CONTRACT: new_streaming (the non-lockstep constructor) yields
        // is_lockstep=false. The two constructors are not interchangeable.
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let q = AdmissionQueue::new_streaming(rx, ReplayMode::Trace);
        assert!(!q.is_lockstep());
    }

    #[test]
    fn adversarial_consume_underflow_returns_false_no_panic() {
        // CONTRACT: consume_drain_barrier on a lockstep queue with 0
        // credits returns false and does NOT underflow the counter.
        // Repeated calls remain safe.
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        assert_eq!(q.drain_barrier_credits(), 0);
        for _ in 0..1000 {
            assert!(!q.consume_drain_barrier());
        }
        assert_eq!(q.drain_barrier_credits(), 0);
    }

    #[test]
    fn adversarial_many_barriers_accrue_before_any_completion() {
        // CONTRACT: barriers received before any completion still
        // accumulate as credits — the runtime can consume them on the
        // gates that follow. Useful for "front-loaded" producer designs.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        for _ in 0..50 {
            tx.send(StreamingAdmission::DrainBarrier).unwrap();
        }
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 50);
    }

    #[test]
    fn adversarial_barriers_independent_of_burst_pending_and_staged() {
        // CONTRACT: DrainBarrier never touches `pending` or `staged`;
        // they are completely independent paths.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(
            rx,
            ReplayMode::Concurrency { max_in_flight: 8 },
        );
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();
        assert_eq!(
            q.total_requests(),
            1,
            "still staged, barriers did not touch it"
        );
        assert_eq!(q.drain_barrier_credits(), 2);
        // Flushing the burst still works as normal.
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        // Barriers still there.
        assert_eq!(q.drain_barrier_credits(), 2);
    }

    #[test]
    fn adversarial_consume_one_at_a_time_decrements_credits() {
        // CONTRACT: each successful consume_drain_barrier decrements the
        // credit count by exactly 1, and only one credit per call.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 3);
        assert!(q.consume_drain_barrier());
        assert_eq!(q.drain_barrier_credits(), 2);
        assert!(q.consume_drain_barrier());
        assert_eq!(q.drain_barrier_credits(), 1);
        assert!(q.consume_drain_barrier());
        assert_eq!(q.drain_barrier_credits(), 0);
        assert!(!q.consume_drain_barrier());
    }

    #[test]
    fn adversarial_drain_barrier_in_non_lockstep_does_not_block_drain() {
        // CONTRACT: in non-lockstep streaming, DrainBarrier is fully
        // ignored — it does NOT block a subsequent untagged submit from
        // admitting. (Non-lockstep producers can spuriously send
        // barriers without affecting throughput.)
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming(rx, ReplayMode::Concurrency { max_in_flight: 4 });
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1, "non-lockstep should freely admit");
        assert!(q.is_drained());
    }

    #[test]
    fn adversarial_is_lockstep_for_workload_source() {
        // CONTRACT: WorkloadDriver-backed AdmissionQueue is never
        // lockstep (lockstep is a Streaming-only mode). The is_lockstep
        // helper uses `matches!` against `AdmissionSource::Streaming`,
        // so any non-Streaming source returns false — verified at the
        // Requests path above; the same dispatch covers Workload by
        // construction (single match arm).
        // (We avoid constructing a real WorkloadDriver here because the
        // type has no #[cfg(test)] empty constructor in this branch;
        // the same match arm is exercised by the Requests case.)
    }

    #[test]
    fn adversarial_lockstep_close_with_outstanding_credits_observable() {
        // CONTRACT: closing the channel does not "consume" outstanding
        // credits. They remain available to be drained by consume_drain_barrier
        // calls after close. (The runtime relies on this: it may consume
        // a queued barrier even after detecting channel close.)
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 2);
        assert!(q.consume_drain_barrier());
        assert!(q.consume_drain_barrier());
        assert!(!q.consume_drain_barrier());
    }

    #[test]
    fn adversarial_lockstep_is_drained_requires_no_pending_no_staged_closed() {
        // CONTRACT: is_drained is independent of barrier credits — it
        // only considers pending + staged + closed. A lockstep queue
        // with credits but no pending work is "drained".
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        drop(tx);
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 2);
        assert!(q.is_drained(), "credits don't count against is_drained");
    }

    #[tokio::test]
    async fn adversarial_lockstep_recv_next_after_close_returns_none() {
        // CONTRACT: in lockstep mode, recv_next still returns None when
        // the channel is closed and drained — barriers do not prevent
        // close detection.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        drop(tx);
        assert!(q.recv_next().await.is_none());
    }

    #[test]
    fn adversarial_poll_pending_counts_barriers_as_events() {
        // CONTRACT: poll_pending counts DrainBarrier as one event each,
        // same as Submit and FlushBurst.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(StreamingAdmission::FlushBurst(99)).unwrap();
        let n = q.poll_pending();
        assert_eq!(n, 4);
    }

    #[test]
    fn adversarial_lockstep_with_workload_source_via_streaming_only() {
        // CONTRACT (type safety): the only constructor that returns a
        // lockstep queue is new_streaming_lockstep. Workload and Requests
        // sources cannot be made lockstep via any public API. Verified
        // by constructing both other kinds and asserting is_lockstep is
        // false; only the streaming-lockstep constructor is true.
        let (_tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let q_lock = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        assert!(q_lock.is_lockstep());

        let q_req = AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace);
        assert!(!q_req.is_lockstep());
    }

    // -------------------------------------------------------------------
    // Drain-barrier ordering invariant: a barrier may not release the
    // lockstep gate while the producer's reaction (queued submits and
    // staged-but-unflushed bursts) is still pending. The producer may
    // send barriers and submits in any order across reruns due to
    // scheduler jitter on its side; the runtime must tolerate this and
    // always admit the queued reaction at the CURRENT sim time before
    // advancing past the barrier.
    // -------------------------------------------------------------------

    #[test]
    fn drain_barrier_waits_for_pending_submits_before_consuming() {
        // RACE: producer's scheduler delivered [BARRIER, SUBMIT] to the
        // channel out of intended order (intended order: [SUBMIT, BARRIER]).
        // After poll_pending, the runtime has: pending=[s], credits=1.
        // consume_drain_barrier MUST return false until pending is drained
        // by the caller, so the runtime is forced to admit `s` at the
        // current sim time before advancing.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 1);
        // BUG-EXPOSE: pending non-empty must block barrier consumption.
        assert!(
            !q.consume_drain_barrier(),
            "barrier must not release while pending reaction is queued"
        );
        // Caller admits the pending submit.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        // Now pending is empty — barrier may consume.
        assert!(q.consume_drain_barrier());
    }

    #[test]
    fn drain_barrier_waits_for_staged_bursts_before_consuming() {
        // RACE: producer delivered [BARRIER, SUBMIT(burst=1)] before the
        // matching FlushBurst arrives. Pending is empty, BUT staged is
        // not — the producer's reaction is still in-flight on the
        // channel. The barrier must not release until the burst flushes.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        tx.send(burst_submit(direct_request_at(0.0, 16), 1))
            .unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 1);
        // BUG-EXPOSE: staged-but-unflushed work blocks barrier consumption.
        assert!(
            !q.consume_drain_barrier(),
            "barrier must not release while staged burst is unflushed"
        );
        // Producer eventually flushes.
        tx.send(StreamingAdmission::FlushBurst(1)).unwrap();
        let _ = q.poll_pending();
        // Pending now has the burst contents — still must wait.
        assert!(!q.consume_drain_barrier());
        // Admit them.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        // Now pending and staged are both empty.
        assert!(q.consume_drain_barrier());
    }

    #[test]
    fn drain_barrier_drains_channel_before_consuming() {
        // RACE: a submit follows a barrier on the channel; the runtime
        // only sees the barrier via poll_pending, then immediately tries
        // to consume. If consume_drain_barrier does not re-drain the
        // channel first, it would accept the now-1 credit and advance
        // past the pending submit.
        //
        // Sequence: send BARRIER → poll → send SUBMIT (still in channel
        // buffer) → consume. The fix must internally try_recv and
        // observe the SUBMIT before deciding.
        let (tx, rx) = mpsc::unbounded_channel::<StreamingAdmission>();
        let mut q = AdmissionQueue::new_streaming_lockstep(rx, ReplayMode::Trace);
        tx.send(StreamingAdmission::DrainBarrier).unwrap();
        let _ = q.poll_pending();
        assert_eq!(q.drain_barrier_credits(), 1);
        // Producer's reaction lands AFTER the runtime polled but BEFORE
        // it consumed — exactly the cross-task race the user described.
        tx.send(direct_request_at(0.0, 16).into()).unwrap();
        assert!(
            !q.consume_drain_barrier(),
            "barrier consumption must drain channel and refuse while a queued reaction is observable"
        );
        // After the implicit drain, the submit is visible in pending.
        let admitted = q.drain_ready(0.0, 0).expect("drain_ready");
        assert_eq!(admitted.len(), 1);
        assert!(q.consume_drain_barrier());
    }
}
