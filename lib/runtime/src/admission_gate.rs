// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent backend admission gate.
//!
//! There is exactly one admission point in the runtime:
//! `Ingress::handle_payload_shared` in
//! [`crate::pipeline::network::ingress::push_handler`], which every request
//! plane already funnels through. The TCP and NATS ingress paths carry no
//! admission logic of their own and no admission sizing.
//!
//! ```text
//! TCP request plane  ─┐
//!                     ├─> Ingress::handle_payload_shared ─> gate ─> backend worker
//! NATS request plane ─┘
//! ```
//!
//! # Engine limits
//!
//! Two independent limits govern entry to the engine, and a request enters only
//! when both have room:
//!
//! - The **engine request limit** bounds admitted requests that have not
//!   finished. A request holds request capacity from admission until its
//!   response stream ends or is dropped, however long it then runs.
//! - The **engine wait limit** bounds admitted requests still awaiting their
//!   first engine response. A request holds wait capacity from admission until
//!   the gate observes the first item of the engine's response stream, whatever
//!   that item carries. Obtaining the stream is not a response.
//!
//! Admission reserves both together, under one lock, so no request ever holds
//! one while it waits for the other. A request that never produces a response —
//! `generate` fails, the stream ends empty, or the request is cancelled or
//! dropped — releases both on the way out. Every release hands capacity to the
//! queue, so queued work proceeds as soon as both limits have room.
//!
//! # Queue
//!
//! A request that cannot enter immediately waits in one bounded FIFO queue,
//! holding no engine capacity while it does. Queued requests are admitted oldest
//! first, and a new arrival is admitted directly only when nothing is queued, so
//! it can never bypass an older request. When the queue is full the request is
//! rejected before it reaches the engine; a queue limit of zero disables
//! queueing, so a request that cannot enter immediately is rejected. A request
//! leaves the queue only by admission or cancellation: queue residence has no
//! age limit.
//!
//! # Configuration
//!
//! Each setting is read once, when the process-global gate is created, from the
//! first of its environment names that holds a valid value:
//!
//! | Setting | Environment names, in precedence order | Default |
//! |---|---|---|
//! | Engine request limit | `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT`, `DYN_ENGINE_REQUEST_LIMIT` | 10000 |
//! | Engine wait limit | `DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT` | 10000 |
//! | Request queue limit | `DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT`, `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` | 40000 |
//!
//! Engine limits must be positive integers, and the queue limit a non-negative
//! one. A value that is set but invalid is warned about and passed over. A
//! Python worker that accepts `--engine-request-limit` exports its resolved
//! value to the canonical engine request limit name before the gate is created,
//! so an explicit flag takes precedence over both environment names. The limits
//! are fixed for the life of the process: engine capacity reports never resize
//! them.
//!
//! TCP requests pass through the TCP work queue and worker pool before reaching
//! this gate, then keep their worker-pool permit while waiting here. The TCP
//! pool can therefore limit how many requests reach the gate at once. Its
//! `DYN_TCP_WORKER_POOL_SIZE` / `DYN_TCP_WORK_QUEUE_SIZE` controls remain
//! separately configured from this gate; NATS has no corresponding TCP-side
//! bound.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, OnceLock};
use std::task::Poll;

use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::engine::{
    AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data, EngineStream,
};
use crate::error::{DynamoError, ErrorType};
use crate::metrics::backend_admission::{AdmissionSource, BackendAdmissionMetrics};

/// Maximum admitted requests that have not finished. `--engine-request-limit`
/// reaches the gate through this name.
const DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT: &str =
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT";

/// Legacy alias of [`DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT`].
const DYN_ENGINE_REQUEST_LIMIT: &str = "DYN_ENGINE_REQUEST_LIMIT";

/// Maximum admitted requests still awaiting their first engine response.
const DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT: &str = "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT";

/// Maximum requests waiting in the gate queue.
const DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT: &str = "DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT";

/// Legacy alias of [`DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT`].
const DYN_DYNAMO_REQUEST_QUEUE_LIMIT: &str = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT";

/// One configuration setting: its environment names in precedence order, the
/// smallest value it accepts, and the value it takes when no name holds a valid
/// one.
struct Setting {
    names: &'static [&'static str],
    min: usize,
    default: usize,
}

const ENGINE_REQUEST_LIMIT: Setting = Setting {
    names: &[
        DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
        DYN_ENGINE_REQUEST_LIMIT,
    ],
    min: 1,
    default: 10_000,
};

const ENGINE_WAIT_LIMIT: Setting = Setting {
    names: &[DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT],
    min: 1,
    default: 10_000,
};

/// Zero is valid: it disables queueing.
const REQUEST_QUEUE_LIMIT: Setting = Setting {
    names: &[
        DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
        DYN_DYNAMO_REQUEST_QUEUE_LIMIT,
    ],
    min: 0,
    default: 40_000,
};

impl Setting {
    /// The value of the first name `lookup` finds valid, or the default.
    ///
    /// A value that is set but is not an integer of at least `min` is warned
    /// about and passed over, so it cannot hide a valid value behind it.
    fn resolve(&self, lookup: &impl Fn(&str) -> Option<String>) -> usize {
        for &name in self.names {
            let Some(raw) = lookup(name) else {
                continue;
            };
            match raw.trim().parse::<usize>() {
                Ok(value) if value >= self.min => return value,
                _ => tracing::warn!(
                    env = name,
                    value = %raw,
                    minimum = self.min,
                    "Ignoring invalid backend admission setting; expected an integer no less \
                     than the minimum"
                ),
            }
        }
        self.default
    }
}

/// The gate's three limits, fixed for its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmissionLimits {
    /// Admitted requests that have not finished.
    engine_request_limit: usize,
    /// Admitted requests still awaiting their first engine response.
    engine_wait_limit: usize,
    /// Requests waiting in the queue.
    request_queue_limit: usize,
}

impl AdmissionLimits {
    fn from_environment() -> Self {
        Self::resolve(|name| std::env::var(name).ok())
    }

    /// Resolve every setting through `lookup`. Pure, so the names, their
    /// precedence and their defaults are testable without touching the process
    /// environment.
    fn resolve(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            engine_request_limit: ENGINE_REQUEST_LIMIT.resolve(&lookup),
            engine_wait_limit: ENGINE_WAIT_LIMIT.resolve(&lookup),
            request_queue_limit: REQUEST_QUEUE_LIMIT.resolve(&lookup),
        }
    }
}

/// The message a shed request is refused with.
const OVERLOADED_MESSAGE: &str = "Server overloaded: worker at capacity";

/// The message a request cancelled while queued is refused with. It is
/// deliberately not an overload: the caller went away, which is not
/// backpressure.
const CANCELLED_MESSAGE: &str = "Request cancelled while queued for backend admission";

/// The process-global gate, constructed on first touch.
static GATE: LazyLock<Arc<BackendAdmissionGate>> =
    LazyLock::new(BackendAdmissionGate::from_environment);

/// The one gate every request in this process admits through.
pub(crate) fn global() -> &'static Arc<BackendAdmissionGate> {
    &GATE
}

/// Accept the capacity a registered model card reports.
///
/// The engine limits come only from configuration and stay fixed for the life
/// of the process, so a capacity report never changes either of them, however
/// early or late it arrives.
pub fn record_engine_capacity(_max_num_seqs: Option<u64>, _data_parallel_size: Option<u32>) {}

/// Expose the process-global gate's metrics for scraping. Idempotent.
pub(crate) fn register_metrics(registry: &crate::MetricsRegistry) {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| global().metrics.register(registry));
}

/// Why the gate refused a request. Private: callers see only the typed error
/// [`reject`] builds from it, so no admission state escapes this module.
enum Rejection {
    /// The request could not be admitted and the queue had no room for it.
    QueueFull,
    /// The request was cancelled before it was admitted.
    Cancelled,
}

/// Trace a refusal against the request it refused, and build the error the
/// caller fails it with.
///
/// Shedding is backpressure on this one worker, so it carries
/// [`ErrorType::WorkerOverloaded`]. A cancellation is not backpressure: the
/// caller went away, so classifying it as an overload would misreport a worker
/// that never refused anything.
fn reject(rejection: Rejection, context: Option<&dyn AsyncEngineContext>) -> DynamoError {
    let request_id = context.map(|context| context.id()).unwrap_or_default();
    let (error_type, message) = match rejection {
        Rejection::QueueFull => {
            tracing::warn!(
                request_id,
                "Worker at capacity (an engine limit and the queue are both full), rejecting \
                 request"
            );
            (ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE)
        }
        Rejection::Cancelled => {
            tracing::debug!(request_id, "{CANCELLED_MESSAGE}");
            (ErrorType::Cancelled, CANCELLED_MESSAGE)
        }
    };
    DynamoError::builder()
        .error_type(error_type)
        .message(message)
        .build()
}

/// A queued request, oldest first. Sending on `tx` hands it engine capacity
/// already reserved under both limits.
struct Waiter {
    id: u64,
    tx: oneshot::Sender<()>,
}

/// The gate's state, behind its one lock.
struct GateState {
    limits: AdmissionLimits,
    /// Admitted requests that have not finished, including those that have
    /// already produced their first response.
    engine_requests: usize,
    /// Admitted requests still awaiting their first engine response; each is
    /// also counted in `engine_requests`.
    engine_waits: usize,
    /// Live waiters, oldest first. Its length is exactly the queue occupancy.
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
}

impl GateState {
    /// Whether one more request can enter the engine: both limits need room.
    fn has_engine_capacity(&self) -> bool {
        self.engine_requests < self.limits.engine_request_limit
            && self.engine_waits < self.limits.engine_wait_limit
    }

    /// Reserve both kinds of capacity for one request entering the engine.
    /// Always together, so no request holds one while it waits for the other.
    fn take_engine_capacity(&mut self) {
        self.engine_requests += 1;
        self.engine_waits += 1;
    }

    /// Hand engine capacity to the oldest waiters for as long as both limits
    /// have room, skipping any that went away between enqueue and wake-up so
    /// the capacity passes on.
    fn wake_waiters(&mut self) {
        while self.has_engine_capacity() {
            let Some(waiter) = self.waiters.pop_front() else {
                return;
            };
            // A successful send means the ticket is still there to receive the
            // offer, not that it will take it: a request cancelled while queued
            // keeps its receiver open and may still refuse the capacity,
            // refunding it as it settles. The capacity moves here because the
            // ticket now owns it; the admission is counted only once the
            // request actually reaches the engine.
            if waiter.tx.send(()).is_ok() {
                self.take_engine_capacity();
            }
            // Otherwise the waiter is gone; popping it already reclaimed its
            // queue place, so continue to the next one.
        }
    }
}

/// Two engine limits plus one bounded FIFO queue, shared by every endpoint in
/// the process.
pub(crate) struct BackendAdmissionGate {
    state: Mutex<GateState>,
    metrics: BackendAdmissionMetrics,
}

impl BackendAdmissionGate {
    fn from_environment() -> Arc<Self> {
        let limits = AdmissionLimits::from_environment();
        // Logged once per process, as the effective configuration: the limits
        // are not exported as metrics.
        tracing::info!(
            engine_request_limit = limits.engine_request_limit,
            engine_wait_limit = limits.engine_wait_limit,
            request_queue_limit = limits.request_queue_limit,
            "Backend admission gate created"
        );
        Self::new(limits)
    }

    /// Build a standalone gate. Production uses [`global`]; tests build their
    /// own so they never contend for process-global capacity.
    fn new(limits: AdmissionLimits) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState {
                limits,
                engine_requests: 0,
                engine_waits: 0,
                waiters: VecDeque::new(),
                next_ticket: 0,
            }),
            metrics: BackendAdmissionMetrics::new(),
        })
    }

    /// Apply one state transition under the lock, then publish the occupancy
    /// it left from the authoritative counts. Every mutation goes through here,
    /// so no transition can drift a gauge.
    fn transition<T>(&self, apply: impl FnOnce(&mut GateState) -> T) -> T {
        let mut state = self.state.lock();
        let result = apply(&mut state);
        self.metrics.set_occupancy(
            state.engine_requests,
            state.engine_waits,
            state.waiters.len(),
        );
        result
    }

    /// Run `generate` under engine capacity reserved by both limits.
    ///
    /// This is the whole admission interface, and it is exactly
    /// [`AsyncEngine::generate`]'s own result type: capacity is reserved
    /// *before* `generate` is polled, so a refused request never reaches the
    /// engine, and what comes back is the engine's stream — no permit, no
    /// admission outcome, nothing to match on. The stream holds request
    /// capacity until it ends or is dropped, and wait capacity until the
    /// engine's first response; a `generate` that fails after admission
    /// releases both at once.
    ///
    /// Every request entering here is counted as received, and one that
    /// reaches `generate` as admitted. A refusal is traced and classified here
    /// and surfaces as the standardized [`DynamoError`]; an engine failure keeps
    /// its own error untouched.
    ///
    /// [`AsyncEngine::generate`]: crate::engine::AsyncEngine::generate
    pub(crate) async fn admit<R, F>(
        self: &Arc<Self>,
        context: Option<&dyn AsyncEngineContext>,
        generate: F,
    ) -> anyhow::Result<EngineStream<R>>
    where
        R: Data,
        F: Future<Output = anyhow::Result<EngineStream<R>>>,
    {
        self.metrics.received();
        let (permit, source) = self.acquire(context).await.map_err(anyhow::Error::new)?;
        // Polling `generate` is the engine handoff, and nothing can intervene
        // between here and that first poll, so this — not reserving capacity —
        // is where the request is counted as admitted, exactly once. Whatever
        // the engine then does with it does not undo that.
        self.metrics.admitted(source);
        let stream = generate.await?;
        Ok(Box::pin(AdmittedStream {
            inner: stream,
            permit: Some(permit),
        }))
    }

    /// Reserve engine capacity under both limits, waiting in FIFO order when
    /// either is exhausted, and report whether the request had to queue for it.
    ///
    /// The lock is never held across an await: the decision is made under the
    /// lock, and only the handoff is awaited.
    async fn acquire(
        self: &Arc<Self>,
        context: Option<&dyn AsyncEngineContext>,
    ) -> Result<(EnginePermit, AdmissionSource), DynamoError> {
        enum Decision {
            Admitted,
            Queued(u64, oneshot::Receiver<()>),
            Rejected,
        }

        // A caller that has already gone away must not enter the engine. Direct
        // admission never awaits, so this is the only point at which that can
        // be caught before `generate` is polled; the queued path re-checks in
        // `AdmissionTicket::wait`, where the context can also stop while the
        // request waits.
        if context.is_some_and(|context| context.is_stopped()) {
            // Counted as a cancellation, which a rejection never is.
            self.metrics.cancelled();
            return Err(reject(Rejection::Cancelled, context));
        }

        let decision = self.transition(|state| {
            // Direct admission only when nothing is waiting, so a new request
            // can never bypass an older one.
            if state.waiters.is_empty() && state.has_engine_capacity() {
                state.take_engine_capacity();
                Decision::Admitted
            } else if state.waiters.len() < state.limits.request_queue_limit {
                let id = state.next_ticket;
                state.next_ticket = state.next_ticket.wrapping_add(1);
                let (tx, rx) = oneshot::channel();
                state.waiters.push_back(Waiter { id, tx });
                Decision::Queued(id, rx)
            } else {
                Decision::Rejected
            }
        });

        match decision {
            Decision::Admitted => {
                Ok((EnginePermit::new(Arc::clone(self)), AdmissionSource::Direct))
            }
            // Counted here rather than under the lock.
            Decision::Rejected => {
                self.metrics.rejected();
                Err(reject(Rejection::QueueFull, context))
            }
            Decision::Queued(id, rx) => {
                let permit = AdmissionTicket {
                    gate: Arc::clone(self),
                    id,
                    rx,
                    settled: false,
                }
                .wait(context)
                .await?;
                Ok((permit, AdmissionSource::Queue))
            }
        }
    }

    /// Release the wait capacity of a request that has produced its first
    /// response, and pass it on to the queue.
    fn release_wait(&self) {
        self.transition(|state| {
            state.engine_waits = state.engine_waits.saturating_sub(1);
            state.wake_waiters();
        });
    }

    /// Release the request capacity of a request that finished, failed or went
    /// away, or of an offer a queued request refused — with its wait capacity
    /// too when it still holds that — and pass it on to the queue.
    fn release_request(&self, holds_wait: bool) {
        self.transition(|state| {
            state.engine_requests = state.engine_requests.saturating_sub(1);
            if holds_wait {
                state.engine_waits = state.engine_waits.saturating_sub(1);
            }
            state.wake_waiters();
        });
    }

    /// Drop an abandoned waiter so it stops consuming queue capacity.
    ///
    /// Cancellation can land anywhere in the FIFO, so this is a linear search.
    fn remove_waiter(&self, id: u64) {
        self.transition(|state| {
            if let Some(index) = state.waiters.iter().position(|waiter| waiter.id == id) {
                state.waiters.remove(index);
            }
        });
    }
}

/// Engine capacity held by one admitted request: request capacity until it
/// finishes, and wait capacity until its first response. Private, and never
/// named in any signature a caller can reach: only `admit` and the
/// [`AdmittedStream`] it returns ever own one, so no caller can hold, forget or
/// forge one.
struct EnginePermit {
    gate: Arc<BackendAdmissionGate>,
    /// Still awaiting the first engine response, so still holding wait capacity.
    holds_wait: bool,
}

impl EnginePermit {
    fn new(gate: Arc<BackendAdmissionGate>) -> Self {
        Self {
            gate,
            holds_wait: true,
        }
    }

    /// The first engine response: release wait capacity, exactly once, and keep
    /// request capacity while the request continues.
    fn first_response(&mut self) {
        if std::mem::take(&mut self.holds_wait) {
            self.gate.release_wait();
        }
    }
}

impl Drop for EnginePermit {
    fn drop(&mut self) {
        // The request finished, failed or went away: release everything it
        // still holds.
        self.gate.release_request(self.holds_wait);
    }
}

/// The engine's own response stream, holding the capacity it was admitted on.
///
/// It is an [`EngineStream`] like any other — item type, items and context are
/// all the engine's — so the caller sees no admission type at all.
struct AdmittedStream<R: Data> {
    inner: EngineStream<R>,
    /// Taken when the stream ends, so its request capacity is released exactly
    /// once; `Drop` covers a stream abandoned before that.
    permit: Option<EnginePermit>,
}

impl<R: Data> Stream for AdmittedStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match &polled {
            // An engine response, whatever it carries. The first releases wait
            // capacity to the next queued request before the item is forwarded;
            // every later one finds it already released.
            Poll::Ready(Some(_)) => {
                if let Some(permit) = self.permit.as_mut() {
                    permit.first_response();
                }
            }
            // The stream finished: release everything this request still holds,
            // including wait capacity if it never produced a response.
            Poll::Ready(None) => drop(self.permit.take()),
            Poll::Pending => {}
        }
        polled
    }
}

impl<R: Data> AsyncEngineContextProvider for AdmittedStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.context()
    }
}

impl<R: Data> AsyncEngineStream<R> for AdmittedStream<R> {}

impl<R: Data> std::fmt::Debug for AdmittedStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmittedStream")
            .field("inner", &self.inner)
            .finish()
    }
}

/// A place in the FIFO. Dropping it before an offer arrives unregisters the
/// waiter; dropping it after one was handed over refunds that capacity so the
/// next waiter is woken instead of it being lost.
struct AdmissionTicket {
    gate: Arc<BackendAdmissionGate>,
    id: u64,
    rx: oneshot::Receiver<()>,
    /// Set once the gate's offer has been settled — converted into an
    /// [`EnginePermit`], or refused and refunded on the spot — so `Drop` has
    /// nothing left to account for.
    settled: bool,
}

impl AdmissionTicket {
    /// Resolve a cancellation against an offer the gate may already have sent.
    ///
    /// Closing first makes this single-winner: nothing can arrive afterwards.
    /// Cancellation wins an offer that is already there, because honoring it
    /// would admit a caller that has already gone and hold capacity ahead of
    /// live waiters. Taking the offer out of the channel makes it this
    /// function's to settle — `Drop` could no longer find it — so the
    /// cancellation is counted and the capacity refunded here and now. Without
    /// an offer nothing is settled here, and `Drop` counts the cancellation and
    /// unregisters the waiter.
    fn refuse_offer(&mut self) {
        self.rx.close();
        if self.rx.try_recv().is_ok() {
            self.settled = true;
            self.gate.metrics.cancelled();
            self.gate.release_request(true);
        }
    }

    async fn wait(
        mut self,
        context: Option<&dyn AsyncEngineContext>,
    ) -> Result<EnginePermit, DynamoError> {
        // Cancellation while queued must be prompt, and it must win a
        // simultaneous offer: one can be sent to this waiter after the context
        // stopped but before this future is polled again, and admitting then
        // would run a request the caller already abandoned. The precheck plus
        // the biased ordering give the stop strictly higher priority; an offer
        // that had already arrived is then refused in `refuse_offer`.
        let offered = match context {
            Some(context) if context.is_stopped() => false,
            Some(context) => tokio::select! {
                biased;
                _ = context.stopped() => false,
                offer = &mut self.rx => offer.is_ok(),
            },
            None => (&mut self.rx).await.is_ok(),
        };
        if !offered {
            // A cancellation is the caller leaving: never a rejection, and
            // never an admission either. It is counted exactly once, either by
            // `refuse_offer` or by the `Drop` this return runs.
            self.refuse_offer();
            return Err(reject(Rejection::Cancelled, context));
        }
        // The one place a queued request takes the capacity it was offered:
        // `admit` counts the admission when it hands the request to the engine.
        self.settled = true;
        Ok(EnginePermit::new(Arc::clone(&self.gate)))
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        // Reaching here unsettled means this request left the queue without
        // becoming an admission — because `wait` resolved a cancellation, or
        // because the whole future was dropped before it could. Everything left
        // to account for is settled from here, exactly once.
        if self.settled {
            return;
        }
        self.gate.metrics.cancelled();
        // Closing first makes the handoff race single-winner: either the gate's
        // send already succeeded and `try_recv` yields capacity that must be
        // refunded, or the send fails and the gate moves on to the next waiter.
        self.rx.close();
        match self.rx.try_recv() {
            // An offer this request never consumed: the capacity goes back to
            // the next waiter, since this candidate is absent, not admitted.
            Ok(()) => self.gate.release_request(true),
            Err(_) => self.gate.remove_waiter(self.id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{FutureExt, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::pipeline::context::Controller;

    fn gate(
        engine_request_limit: usize,
        engine_wait_limit: usize,
        request_queue_limit: usize,
    ) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(AdmissionLimits {
            engine_request_limit,
            engine_wait_limit,
            request_queue_limit,
        })
    }

    /// Published as (engine requests, engine waits, queued).
    fn published(gate: &Arc<BackendAdmissionGate>) -> (i64, i64, i64) {
        gate.metrics.published()
    }

    fn queued(gate: &Arc<BackendAdmissionGate>) -> usize {
        gate.state.lock().waiters.len()
    }

    /// The scheduling tests drive [`BackendAdmissionGate::acquire`] directly:
    /// FIFO order, refunds and both limits are decided before an engine is ever
    /// involved, and a permit is what those transitions move around. Calling
    /// [`EnginePermit::first_response`] and dropping the permit stand in for the
    /// engine's first response and the end of its stream. The public
    /// stream-returning [`BackendAdmissionGate::admit`], which does both from
    /// the engine's own stream, is covered separately.
    type Acquired = Result<(EnginePermit, AdmissionSource), DynamoError>;

    fn permit(acquired: Acquired) -> EnginePermit {
        acquired.expect("expected admission").0
    }

    /// Yield, never sleep, until the queue holds `expected` requests.
    async fn await_queued(gate: &Arc<BackendAdmissionGate>, expected: usize) {
        for _ in 0..1_000 {
            if queued(gate) == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "the queue holds {} requests, never {expected}",
            queued(gate)
        );
    }

    /// Spawn an acquisition — it only resolves once the gate decides, so a
    /// queueing request can never be awaited on the test task — and return once
    /// it has joined the FIFO.
    async fn spawn_queued_admit(
        gate: &Arc<BackendAdmissionGate>,
    ) -> tokio::task::JoinHandle<Acquired> {
        let before = queued(gate);
        let handle = tokio::spawn({
            let gate = Arc::clone(gate);
            async move { gate.acquire(None).await }
        });
        await_queued(gate, before + 1).await;
        handle
    }

    /// The outcome of a spawned request, failing rather than hanging if the gate
    /// never decides it.
    async fn outcome<T>(handle: tokio::task::JoinHandle<T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("the gate must decide promptly")
            .expect("the request task completes")
    }

    /// The `(type, message)` pair a refusal carries, or `None` once admitted.
    fn refusal(acquired: &Acquired) -> Option<(ErrorType, &str)> {
        acquired
            .as_ref()
            .err()
            .map(|error| (error.error_type(), error.message()))
    }

    fn is_queue_full(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE))
    }

    fn is_cancelled(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
    }

    ///////////////////// CONFIGURATION /////////////////////

    /// Limits resolved from exactly these environment values.
    fn limits_from(env: &[(&str, &str)]) -> AdmissionLimits {
        AdmissionLimits::resolve(|name| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    const DEFAULTS: AdmissionLimits = AdmissionLimits {
        engine_request_limit: 10_000,
        engine_wait_limit: 10_000,
        request_queue_limit: 40_000,
    };

    #[test]
    fn unset_settings_take_their_fixed_defaults() {
        assert_eq!(limits_from(&[]), DEFAULTS);
    }

    /// The canonical name outranks its legacy alias, and a value that is not a
    /// positive integer is passed over rather than hiding the next source. It
    /// configures the full-request limit and nothing else.
    #[test]
    fn the_engine_request_limit_reads_the_canonical_name_then_its_legacy_alias() {
        const CANONICAL: &str = DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT;
        const LEGACY: &str = DYN_ENGINE_REQUEST_LIMIT;
        for (env, expected) in [
            (vec![(CANONICAL, "7")], 7),
            (vec![(LEGACY, "5")], 5),
            (vec![(CANONICAL, "7"), (LEGACY, "5")], 7),
            (vec![(CANONICAL, "7"), (LEGACY, "invalid")], 7),
            (vec![(CANONICAL, "7"), (LEGACY, "0")], 7),
            (vec![(CANONICAL, " 7 ")], 7),
            (vec![(CANONICAL, "0"), (LEGACY, "5")], 5),
            (vec![(CANONICAL, "-1"), (LEGACY, "5")], 5),
            (vec![(CANONICAL, "invalid"), (LEGACY, "5")], 5),
            (vec![(CANONICAL, ""), (LEGACY, "5")], 5),
            (vec![(CANONICAL, "0"), (LEGACY, "0")], 10_000),
            (
                vec![(CANONICAL, "99999999999999999999999999999999")],
                10_000,
            ),
        ] {
            assert_eq!(
                limits_from(&env),
                AdmissionLimits {
                    engine_request_limit: expected,
                    ..DEFAULTS
                },
                "{env:?}"
            );
        }
    }

    /// The first-response limit has one name of its own: neither name of the
    /// full-request limit reaches it.
    #[test]
    fn the_engine_wait_limit_reads_only_its_own_name() {
        const WAIT: &str = DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT;
        for (env, expected) in [
            (vec![(WAIT, "3")], 3),
            (vec![(WAIT, " 3 ")], 3),
            (vec![(WAIT, "0")], 10_000),
            (vec![(WAIT, "-1")], 10_000),
            (vec![(WAIT, "invalid")], 10_000),
        ] {
            assert_eq!(
                limits_from(&env),
                AdmissionLimits {
                    engine_wait_limit: expected,
                    ..DEFAULTS
                },
                "{env:?}"
            );
        }
        for name in [
            DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
            DYN_ENGINE_REQUEST_LIMIT,
        ] {
            assert_eq!(
                limits_from(&[(name, "5")]).engine_wait_limit,
                10_000,
                "{name}"
            );
        }
    }

    /// Zero is a valid queue limit under either name: it disables queueing.
    #[test]
    fn the_queue_limit_accepts_zero_and_reads_the_canonical_name_then_its_legacy_alias() {
        const CANONICAL: &str = DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT;
        const LEGACY: &str = DYN_DYNAMO_REQUEST_QUEUE_LIMIT;
        for (env, expected) in [
            (vec![(CANONICAL, "17")], 17),
            (vec![(LEGACY, "9")], 9),
            (vec![(CANONICAL, "17"), (LEGACY, "9")], 17),
            (vec![(CANONICAL, "17"), (LEGACY, "invalid")], 17),
            (vec![(CANONICAL, "0")], 0),
            (vec![(LEGACY, "0")], 0),
            (vec![(CANONICAL, " 0 "), (LEGACY, "9")], 0),
            (vec![(CANONICAL, "0"), (LEGACY, "-1")], 0),
            (vec![(CANONICAL, "-1"), (LEGACY, "9")], 9),
            (vec![(CANONICAL, "invalid")], 40_000),
            (vec![(CANONICAL, "")], 40_000),
        ] {
            assert_eq!(
                limits_from(&env),
                AdmissionLimits {
                    request_queue_limit: expected,
                    ..DEFAULTS
                },
                "{env:?}"
            );
        }
    }

    /// A valid canonical value ends the search: the legacy alias it shadows is
    /// never read, so an invalid alias can neither replace it nor be warned
    /// about.
    #[test]
    fn a_valid_canonical_value_never_reads_its_shadowed_legacy_alias() {
        let env = [
            (DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT, "7"),
            (DYN_ENGINE_REQUEST_LIMIT, "invalid"),
            (DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT, "0"),
            (DYN_DYNAMO_REQUEST_QUEUE_LIMIT, "-1"),
        ];
        let read = std::cell::RefCell::new(Vec::new());
        let limits = AdmissionLimits::resolve(|name| {
            read.borrow_mut().push(name.to_string());
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        });
        assert_eq!(
            limits,
            AdmissionLimits {
                engine_request_limit: 7,
                request_queue_limit: 0,
                ..DEFAULTS
            }
        );
        assert_eq!(
            *read.borrow(),
            [
                DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
                DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT,
                DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
            ]
        );
    }

    /// Model registration reports capacity through the process-global hook, and
    /// neither engine limit moves however many reports arrive.
    #[test]
    fn capacity_reports_never_change_the_limits() {
        let before = global().state.lock().limits;
        for (max_num_seqs, data_parallel_size) in
            [(Some(1), Some(1)), (Some(256), Some(8)), (None, None)]
        {
            record_engine_capacity(max_num_seqs, data_parallel_size);
        }
        assert_eq!(global().state.lock().limits, before);
    }

    ///////////////////// ENGINE LIMITS /////////////////////

    /// Request capacity is free, but the one request awaiting its first
    /// response holds all the wait capacity, so the next request queues until
    /// that response. The request that answered keeps its request capacity.
    #[tokio::test]
    async fn the_wait_limit_alone_holds_admission_until_a_first_response() {
        let gate = gate(10, 1, 4);
        let mut waiting = permit(gate.acquire(None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1));

        waiting.first_response();
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the answering request still holds request capacity"
        );

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Wait capacity is free, because the one admitted request has answered,
    /// but it keeps its request capacity until it finishes.
    #[tokio::test]
    async fn the_request_limit_alone_holds_admission_until_a_request_finishes() {
        let gate = gate(1, 10, 4);
        let mut running = permit(gate.acquire(None).await);
        running.first_response();
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 0, 1));

        drop(running);
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(published(&gate), (1, 1, 0));

        drop(admitted);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// With both limits exhausted, a first response frees wait capacity but
    /// admits nothing while request capacity is still full.
    #[tokio::test]
    async fn a_first_response_admits_only_when_the_request_limit_has_room() {
        let gate = gate(2, 1, 4);
        let mut running = permit(gate.acquire(None).await);
        running.first_response();
        let mut waiting = permit(gate.acquire(None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (2, 1, 1), "both limits are exhausted");

        waiting.first_response();
        assert_eq!(
            published(&gate),
            (2, 0, 1),
            "wait capacity alone admits nothing"
        );

        drop(running);
        let (admitted, _) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(published(&gate), (2, 1, 0));

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// The mirror case: a request that finishes frees request capacity but
    /// admits nothing while wait capacity is still full.
    #[tokio::test]
    async fn a_completion_admits_only_when_the_wait_limit_has_room() {
        let gate = gate(2, 1, 4);
        let mut running = permit(gate.acquire(None).await);
        running.first_response();
        let mut waiting = permit(gate.acquire(None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (2, 1, 1), "both limits are exhausted");

        drop(running);
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "request capacity alone admits nothing"
        );

        waiting.first_response();
        let (admitted, _) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(published(&gate), (2, 1, 0));

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Wait capacity is released once, at the first response, however often
    /// that is reported; request capacity is released once, at the end. A
    /// second request holds both kinds throughout, so a repeated release would
    /// show as a count below it rather than vanish at zero.
    #[tokio::test]
    async fn each_kind_of_capacity_is_released_exactly_once() {
        let gate = gate(10, 10, 0);
        let _held = permit(gate.acquire(None).await);
        let mut permit = permit(gate.acquire(None).await);
        assert_eq!(published(&gate), (2, 2, 0));

        permit.first_response();
        permit.first_response();
        assert_eq!(published(&gate), (2, 1, 0));

        drop(permit);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    ///////////////////// QUEUE /////////////////////

    #[tokio::test]
    async fn zero_queue_rejects_without_capacity_and_admits_after_release() {
        let gate = gate(1, 1, 0);
        let held = permit(gate.acquire(None).await);
        let overflow = gate.acquire(None).now_or_never().expect("must not wait");
        assert!(is_queue_full(&overflow));
        assert_eq!(published(&gate), (1, 1, 0));
        assert_eq!(gate.metrics.rejections(), 1);

        drop(held);
        let next = permit(gate.acquire(None).now_or_never().expect("capacity is free"));
        assert_eq!(published(&gate), (1, 1, 0));
        drop(next);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn n_direct_admissions_then_exactly_q_queued_then_reject() {
        let gate = gate(2, 2, 3);
        let held = vec![
            permit(gate.acquire(None).await),
            permit(gate.acquire(None).await),
        ];
        let mut waiters = Vec::new();
        for _ in 0..3 {
            waiters.push(spawn_queued_admit(&gate).await);
        }
        assert_eq!(
            published(&gate),
            (2, 2, 3),
            "queued requests hold no engine capacity"
        );

        assert!(
            is_queue_full(&gate.acquire(None).await),
            "the Q+1th request must be shed"
        );
        assert_eq!(queued(&gate), 3, "a rejection must not consume queue space");

        drop(held);
        for waiter in waiters {
            drop(permit(outcome(waiter).await));
        }
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn queue_is_fifo_and_a_new_request_never_bypasses_an_older_waiter() {
        let gate = gate(1, 1, 8);
        let held = permit(gate.acquire(None).await);

        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        // Each arrival joins behind the waiters already queued, so the last one
        // in must also be the last one served.
        for index in 0..5usize {
            let before = queued(&gate);
            handles.push(tokio::spawn({
                let gate = Arc::clone(&gate);
                let order = Arc::clone(&order);
                async move {
                    let permit = permit(gate.acquire(None).await);
                    order.lock().push(index);
                    drop(permit);
                }
            }));
            await_queued(&gate, before + 1).await;
        }

        drop(held);
        for handle in handles {
            outcome(handle).await;
        }
        assert_eq!(
            *order.lock(),
            vec![0, 1, 2, 3, 4],
            "queued requests must be admitted oldest first"
        );
        assert_eq!(published(&gate), (0, 0, 0));
    }

    ///////////////////// CANCELLATION AND REFUND /////////////////////

    #[tokio::test]
    async fn a_dropped_queued_waiter_frees_its_queue_place_immediately() {
        let gate = gate(1, 1, 1);
        let held = permit(gate.acquire(None).await);

        let waiter = spawn_queued_admit(&gate).await;
        assert!(is_queue_full(&gate.acquire(None).await), "queue is full");

        waiter.abort();
        let _ = waiter.await;
        await_queued(&gate, 0).await;
        assert_eq!(gate.metrics.cancellations(), 1);

        // The freed queue place is reusable: this request queues rather than
        // being shed.
        let reuse = spawn_queued_admit(&gate).await;
        drop(held);
        drop(permit(outcome(reuse).await));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn cancellation_beats_a_simultaneous_offer_and_refunds_both_kinds() {
        let gate = gate(1, 1, 4);
        let held = permit(gate.acquire(None).await);

        // Waiter A is cancellable; poll it exactly once so it registers in the
        // FIFO and then stops being polled.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.acquire(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());

        // Waiter B queues behind it and is the one that must inherit the
        // capacity.
        let survivor = spawn_queued_admit(&gate).await;

        // Make both outcomes ready for waiter A before it is polled again: the
        // context is stopped, and then the released capacity is offered to it.
        controller.stop_generating();
        drop(held);
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "the capacity was offered to the doomed waiter"
        );

        // Cancellation must win even though the offer is also ready.
        assert!(
            is_cancelled(&doomed.await),
            "a cancelled request must not be admitted by a racing offer"
        );

        // The refund covers both kinds, so the next live waiter is admitted.
        let survivor = permit(outcome(survivor).await);
        assert_eq!(published(&gate), (1, 1, 0));
        drop(survivor);
        assert_eq!(published(&gate), (0, 0, 0));
        assert_eq!(gate.metrics.cancellations(), 1);
        assert_eq!(
            Arc::strong_count(&gate),
            1,
            "tickets and permits must not retain the gate"
        );
    }

    /// A queued request that leaves without being admitted is a cancellation
    /// exactly once, whether its own future observed the stop or the whole task
    /// was dropped first. Neither shape is a rejection.
    #[tokio::test]
    async fn a_queued_cancellation_counts_once_whether_observed_or_dropped() {
        let gate = gate(1, 1, 4);
        let _held = permit(gate.acquire(None).await);

        // Observed: the waiting request is polled again and sees the stop.
        let controller = Arc::new(Controller::default());
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            let context: Arc<dyn AsyncEngineContext> = controller.clone();
            async move { gate.acquire(Some(context.as_ref())).await }
        });
        await_queued(&gate, 1).await;
        controller.stop_generating();
        assert!(is_cancelled(&outcome(waiting).await));
        assert_eq!(queued(&gate), 0, "cancellation frees the queue place");
        assert_eq!(gate.metrics.cancellations(), 1, "counted where observed");

        // Dropped: the task is aborted, so only the ticket's own `Drop` can
        // account for it.
        let abandoned = spawn_queued_admit(&gate).await;
        abandoned.abort();
        let _ = abandoned.await;
        await_queued(&gate, 0).await;
        assert_eq!(gate.metrics.cancellations(), 2);

        assert_eq!(gate.metrics.rejections(), 0);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    #[test]
    fn shedding_is_a_worker_scoped_overload() {
        // Backpressure on this one worker. A pool-scoped `ResourceExhausted`
        // would claim the whole pool is out of room, and a `Backend` error
        // would report a fault that did not happen.
        let error = reject(Rejection::QueueFull, None);
        assert_eq!(error.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(error.message(), OVERLOADED_MESSAGE);
    }

    #[test]
    fn a_cancellation_is_not_an_overload() {
        // The caller went away. Reporting that as backpressure would migrate or
        // shed against a worker that never refused the request.
        let error = reject(Rejection::Cancelled, None);
        assert_eq!(error.error_type(), ErrorType::Cancelled);
        assert_eq!(error.message(), CANCELLED_MESSAGE);
    }

    ///////////////////// THE ADMITTED STREAM /////////////////////

    /// A stand-in engine: it records that it ran, then returns the two chunks
    /// its stream yields, on its own context.
    async fn generate(
        ran: Arc<AtomicUsize>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> anyhow::Result<EngineStream<usize>> {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(crate::engine::ResponseStream::new(
            Box::pin(futures::stream::iter([1usize, 2])),
            context,
        ))
    }

    /// A stand-in engine whose stream ends without a single response.
    async fn generate_nothing() -> anyhow::Result<EngineStream<usize>> {
        Ok(crate::engine::ResponseStream::new(
            Box::pin(futures::stream::empty()),
            engine_context(),
        ))
    }

    fn engine_context() -> Arc<dyn AsyncEngineContext> {
        Arc::new(Controller::default())
    }

    async fn admit_two(
        gate: &Arc<BackendAdmissionGate>,
        ran: &Arc<AtomicUsize>,
    ) -> EngineStream<usize> {
        gate.admit(None, generate(Arc::clone(ran), engine_context()))
            .await
            .expect("admitted")
    }

    /// The stream is the engine's own. It holds wait capacity until its first
    /// response and request capacity until it ends, each released exactly once:
    /// a second stream holds both kinds throughout, so a repeated release would
    /// show as a count below it.
    #[tokio::test]
    async fn an_admitted_stream_holds_each_kind_of_capacity_for_its_own_lifetime() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let context = engine_context();
        let mut stream = gate
            .admit(None, generate(Arc::clone(&ran), Arc::clone(&context)))
            .await
            .expect("admitted");
        assert_eq!(ran.load(Ordering::SeqCst), 2);
        assert_eq!(
            published(&gate),
            (2, 2, 0),
            "obtaining the stream is not a response"
        );
        assert!(
            Arc::ptr_eq(&stream.context(), &context),
            "the engine's own context must be delegated, not replaced"
        );

        // The items are the engine's, unchanged.
        assert_eq!(stream.next().await, Some(1));
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the first response releases wait capacity only"
        );
        assert_eq!(stream.next().await, Some(2));
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "a later response releases nothing"
        );
        assert_eq!(stream.next().await, None);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "the end of the stream releases request capacity"
        );
        drop(stream);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "and dropping it releases nothing more"
        );
    }

    /// A first response admits the next queued request while the stream that
    /// answered keeps running and keeps its request capacity until it ends.
    #[tokio::test]
    async fn the_first_response_admits_a_queued_request_while_its_stream_continues() {
        let gate = gate(2, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let mut first = admit_two(&gate, &ran).await;
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "no response yet, so the next waits"
        );

        assert_eq!(first.next().await, Some(1));
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the queued request awaits its own first response"
        );

        assert_eq!(first.next().await, Some(2));
        assert_eq!(first.next().await, None);
        assert_eq!(published(&gate), (1, 1, 0));
        drop(admitted);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn a_stream_that_ends_without_a_response_releases_both_kinds_once() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let mut empty = gate
            .admit(None, generate_nothing())
            .await
            .expect("admitted");
        assert_eq!(published(&gate), (2, 2, 0));

        assert_eq!(
            empty.next().await,
            None,
            "the engine ended without answering"
        );
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "an empty end releases both kinds"
        );
        drop(empty);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "and dropping the ended stream releases nothing more"
        );
    }

    /// Cancellation, task abort and a client that goes away all end in a drop.
    #[tokio::test]
    async fn a_stream_dropped_before_its_first_response_releases_both_kinds() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let stream = admit_two(&gate, &ran).await;
        assert_eq!(published(&gate), (1, 1, 0));

        drop(stream);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn a_stream_dropped_after_its_first_response_releases_only_request_capacity() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let mut stream = admit_two(&gate, &ran).await;
        assert_eq!(stream.next().await, Some(1));
        assert_eq!(published(&gate), (2, 1, 0));

        drop(stream);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "wait capacity is not released a second time"
        );
    }

    #[tokio::test]
    async fn aborting_admit_while_generate_is_pending_releases_both_kinds() {
        // Capacity is reserved before `generate` is polled, so the window
        // between admission and a stream existing has no wrapper to release it
        // — only dropping the `admit` future itself can.
        let gate = gate(1, 1, 4);
        let running = Arc::new(AtomicUsize::new(0));
        let task = tokio::spawn({
            let gate = Arc::clone(&gate);
            let running = Arc::clone(&running);
            async move {
                gate.admit(None, async move {
                    running.fetch_add(1, Ordering::SeqCst);
                    std::future::pending::<anyhow::Result<EngineStream<usize>>>().await
                })
                .await
            }
        });
        while running.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(published(&gate), (1, 1, 0), "held while generate runs");

        task.abort();
        let _ = task.await;
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "aborting mid-generate releases both"
        );
    }

    #[tokio::test]
    async fn a_failed_generate_releases_both_kinds_and_keeps_its_own_error() {
        let gate = gate(1, 1, 0);

        let error = gate
            .admit(None, async {
                Err::<EngineStream<usize>, _>(anyhow::anyhow!("engine failed to start"))
            })
            .await
            .expect_err("the engine failed");
        assert_eq!(error.to_string(), "engine failed to start");
        assert!(
            error.downcast_ref::<DynamoError>().is_none(),
            "an engine failure must pass through untouched"
        );
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "a failed generate frees both at once"
        );
    }

    #[tokio::test]
    async fn a_refused_request_never_reaches_the_engine() {
        let gate = gate(1, 1, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;

        let error = gate
            .admit(None, generate(Arc::clone(&ran), engine_context()))
            .await
            .expect_err("the second request is refused");
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "a refused request must not run the engine"
        );

        // The refusal reaches the caller as the standardized error, through the
        // engine's own result type.
        let rejection = error
            .downcast_ref::<DynamoError>()
            .expect("a refusal is a DynamoError");
        assert_eq!(rejection.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(rejection.message(), OVERLOADED_MESSAGE);
    }

    /// An already-cancelled request must not reach the engine even when the gate
    /// is idle. Direct admission never awaits, so nothing downstream of the
    /// decision would ever notice the caller had gone.
    #[tokio::test]
    async fn an_already_cancelled_request_never_reaches_the_engine() {
        let gate = gate(1, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let controller = Arc::new(Controller::default());
        controller.stop_generating();
        let context: Arc<dyn AsyncEngineContext> = controller;

        let error = gate
            .admit(
                Some(context.as_ref()),
                generate(Arc::clone(&ran), engine_context()),
            )
            .await
            .expect_err("a cancelled request must be refused");

        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "generate must never be polled"
        );
        let rejection = error
            .downcast_ref::<DynamoError>()
            .expect("a refusal is a DynamoError");
        assert_eq!(rejection.error_type(), ErrorType::Cancelled);
        assert_eq!(rejection.message(), CANCELLED_MESSAGE);
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "nothing was reserved or queued"
        );
    }

    ///////////////////// METRICS /////////////////////

    /// Requests received, counted by [`BackendAdmissionGate::admit`].
    fn receives(gate: &Arc<BackendAdmissionGate>) -> u64 {
        gate.metrics.receives()
    }

    /// Requests passed into the engine as (direct, queue), counted by
    /// [`BackendAdmissionGate::admit`].
    fn admissions(gate: &Arc<BackendAdmissionGate>) -> (u64, u64) {
        gate.metrics.admissions()
    }

    /// [`spawn_queued_admit`] through the public interface, with the
    /// [`generate`] stand-in as the engine.
    async fn spawn_queued_generate(
        gate: &Arc<BackendAdmissionGate>,
        ran: &Arc<AtomicUsize>,
        context: Option<Arc<dyn AsyncEngineContext>>,
    ) -> tokio::task::JoinHandle<anyhow::Result<EngineStream<usize>>> {
        let before = queued(gate);
        let handle = tokio::spawn({
            let gate = Arc::clone(gate);
            let ran = Arc::clone(ran);
            async move {
                gate.admit(context.as_deref(), generate(ran, engine_context()))
                    .await
            }
        });
        await_queued(gate, before + 1).await;
        handle
    }

    /// The typed refusal an `admit` error carries.
    fn admit_refusal(admitted: &anyhow::Result<EngineStream<usize>>) -> Option<(ErrorType, &str)> {
        admitted
            .as_ref()
            .err()
            .and_then(|error| error.downcast_ref::<DynamoError>())
            .map(|error| (error.error_type(), error.message()))
    }

    /// The gauges come from the gate's own counts, so they follow every
    /// transition: direct admission, enqueue, first response with a queued
    /// handoff, completion, and drain.
    #[tokio::test]
    async fn the_gauges_follow_every_transition() {
        let gate = gate(2, 1, 4);
        assert_eq!(published(&gate), (0, 0, 0));
        let mut held = permit(gate.acquire(None).await);
        assert_eq!(published(&gate), (1, 1, 0), "direct admission");
        let waiter = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1), "enqueue");
        held.first_response();
        let mut granted = permit(outcome(waiter).await);
        assert_eq!(published(&gate), (2, 1, 0), "first response and handoff");
        drop(held);
        assert_eq!(published(&gate), (1, 1, 0), "completion");
        granted.first_response();
        assert_eq!(
            published(&gate),
            (1, 0, 0),
            "the handed-off request answers"
        );
        drop(granted);
        assert_eq!(published(&gate), (0, 0, 0), "drain");
    }

    /// Every request the gate receives is counted once on entry, and once more
    /// as admitted only when it is handed to the engine — directly, or after
    /// waiting in the queue. Waiting is not an admission, a shed or
    /// already-cancelled request is never one, and an engine that fails after
    /// the handoff does not undo one.
    #[tokio::test]
    async fn every_request_is_received_once_and_admitted_only_at_the_engine_handoff() {
        let gate = gate(2, 1, 1);
        let ran = Arc::new(AtomicUsize::new(0));
        assert_eq!((receives(&gate), admissions(&gate)), (0, (0, 0)));

        let mut held = admit_two(&gate, &ran).await;
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (1, (1, 0)),
            "free capacity is a direct admission"
        );

        let queued = spawn_queued_generate(&gate, &ran, None).await;
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (2, (1, 0)),
            "waiting in the queue is not an admission"
        );

        let shed = gate
            .admit(None, generate(Arc::clone(&ran), engine_context()))
            .await;
        assert_eq!(
            admit_refusal(&shed),
            Some((ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE))
        );
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (3, (1, 0)),
            "a shed request is received but never admitted"
        );
        assert_eq!(gate.metrics.rejections(), 1);

        let controller = Arc::new(Controller::default());
        controller.stop_generating();
        let context: Arc<dyn AsyncEngineContext> = controller;
        let cancelled = gate
            .admit(
                Some(context.as_ref()),
                generate(Arc::clone(&ran), engine_context()),
            )
            .await;
        assert_eq!(
            admit_refusal(&cancelled),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
        );
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (4, (1, 0)),
            "nor is one already cancelled on arrival"
        );
        assert_eq!(gate.metrics.cancellations(), 1);
        assert_eq!(gate.metrics.rejections(), 1, "which is not a rejection");

        // The holder's first response hands wait capacity to the queued
        // request, which is admitted as it is handed to the engine.
        assert_eq!(held.next().await, Some(1));
        let queued = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (4, (1, 1)),
            "a queued request is admitted at the handoff"
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "only the admitted requests ran"
        );

        // Never answered, so dropping it frees both kinds for the next one,
        // whose engine then fails.
        drop(queued);
        let failed = gate
            .admit(None, async {
                Err::<EngineStream<usize>, _>(anyhow::anyhow!("engine failed to start"))
            })
            .await;
        assert!(failed.is_err());
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (5, (2, 1)),
            "an engine failure after the handoff stays an admission"
        );

        // Every request received is accounted for exactly once.
        let (direct, queue) = admissions(&gate);
        assert_eq!(
            receives(&gate),
            direct + queue + gate.metrics.rejections() + gate.metrics.cancellations()
        );
    }

    /// A request cancelled while queued, whether its own future observed the
    /// stop or the whole task was dropped, is received and cancelled but never
    /// admitted, and never reaches the engine.
    #[tokio::test]
    async fn a_request_cancelled_while_queued_is_received_but_never_admitted() {
        let gate = gate(1, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;

        // Observed: the waiting request is polled again and sees the stop.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let waiting = spawn_queued_generate(&gate, &ran, Some(context)).await;
        controller.stop_generating();
        assert_eq!(
            admit_refusal(&outcome(waiting).await),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
        );

        // Dropped: the task is aborted while it waits.
        let abandoned = spawn_queued_generate(&gate, &ran, None).await;
        abandoned.abort();
        let _ = abandoned.await;
        await_queued(&gate, 0).await;

        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "only the holder reached the engine"
        );
        assert_eq!(receives(&gate), 3, "both cancelled requests were received");
        assert_eq!(
            admissions(&gate),
            (1, 0),
            "neither cancelled request was admitted"
        );
        assert_eq!(gate.metrics.cancellations(), 2);
        assert_eq!(gate.metrics.rejections(), 0);
    }

    /// A candidate whose caller has already gone is absent, not admitted. Its
    /// receiver stays open, so the gate's offer succeeds and only the ticket can
    /// refuse it: the capacity must not count as an admission or reach the
    /// engine, and must move on to the next request in FIFO order.
    #[tokio::test]
    async fn a_cancelled_candidate_is_never_admitted_and_its_capacity_moves_on() {
        let gate = gate(10, 1, 8);
        let ran = Arc::new(AtomicUsize::new(0));
        let mut held = admit_two(&gate, &ran).await;

        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.admit(
            Some(context.as_ref()),
            generate(Arc::clone(&ran), engine_context()),
        ));
        assert!(futures::poll!(&mut doomed).is_pending());
        let next = spawn_queued_generate(&gate, &ran, None).await;
        assert_eq!(published(&gate), (1, 1, 2));

        // Stopped before the offer reaches it, but its receiver stays open.
        controller.stop_generating();
        assert_eq!(held.next().await, Some(1));
        assert_eq!(
            published(&gate),
            (2, 1, 1),
            "the capacity was offered to the candidate"
        );

        assert_eq!(
            admit_refusal(&doomed.await),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE)),
            "cancellation wins the offer"
        );

        let _next = outcome(next).await.expect("admitted from the queue");
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the refund went to the next request in FIFO order"
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "the candidate never reached the engine"
        );
        assert_eq!(
            admissions(&gate),
            (1, 1),
            "the candidate was never admitted"
        );
        assert_eq!(gate.metrics.cancellations(), 1);
        assert_eq!(gate.metrics.rejections(), 0);
        assert_eq!(receives(&gate), 3);
    }

    /// Collectors are per gate: one gate's counts are invisible to another. The
    /// family a gate registers is covered in
    /// [`crate::metrics::backend_admission`].
    #[tokio::test]
    async fn each_gate_owns_its_collectors() {
        let one = gate(1, 1, 0);
        let two = gate(1, 1, 0);
        let _held = permit(one.acquire(None).await);
        assert!(is_queue_full(&one.acquire(None).await));
        assert_eq!((one.metrics.rejections(), two.metrics.rejections()), (1, 0));
        assert_eq!((published(&one), published(&two)), ((1, 1, 0), (0, 0, 0)));
    }
}
