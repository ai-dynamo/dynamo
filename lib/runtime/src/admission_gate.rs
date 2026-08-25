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
//! # Sizing
//!
//! The concurrent-request limit resolves in this order:
//!
//! 1. a positive [`DYN_ENGINE_REQUEST_LIMIT`];
//! 2. `ceil(3/2 * max_num_seqs * data_parallel_size)` from the capacity reported
//!    through [`record_engine_capacity`];
//! 3. exactly [`DEFAULT_CONCURRENCY_LIMIT`] — never multiplied.
//!
//! The FIFO queue length resolves independently:
//!
//! 1. a positive [`DYN_DYNAMO_REQUEST_QUEUE_LIMIT`];
//! 2. [`DEFAULT_QUEUE_CAPACITY`].
//!
//! # Controlled delay
//!
//! Queueing is bounded in time as well as in length. Every entry is stamped
//! with `enqueue time + queue delay` under the state lock, so FIFO order is
//! also nondecreasing deadline order. The delay is one process-wide budget:
//! a positive [`DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS`], otherwise
//! [`DEFAULT_QUEUE_DELAY`]. It bounds queue residence only — an admitted
//! request's execution time is not limited by it, and a request that takes a
//! free slot directly never carries a deadline at all.
//!
//! Expiry is deadline-driven rather than swept or timed per request. The gate
//! owns exactly one Tokio timer, armed on the oldest live deadline and re-armed
//! whenever the head changes; expired entries leave the FIFO by `pop_front`
//! alone, so the unexpired tail is never inspected. Because a slot can also
//! free up before that timer runs, every grant-producing path re-checks the head
//! deadline under the same lock and rejects the due prefix before granting.
//! An expired request is refused as a worker-scoped
//! [`ErrorType::WorkerOverloaded`] naming the queue delay — an overload, not a
//! cancellation and not a backend fault.
//!
//! # Where the hint comes from
//!
//! The implemented rule is exactly this: **the first usable capacity report from
//! any non-LoRA base model card registered in this process wins.** "Usable"
//! means [`automatic_concurrency_limit`] returns `Some` — both `max_num_seqs`
//! and `data_parallel_size` present, non-zero, and their scaled product in
//! range. A later conflicting report from any other base card in the same
//! process is warned about and ignored. The environment override always wins, so
//! a late report can never override it.
//!
//! The gate does not check what kind of component published the card, and there
//! is no marker or provenance test that could make it do so. `register_model` is
//! reached by routers as well as engines — `global_router`,
//! `vllm.omni.stage_router` and `thunderagent_router` all call it — so a router
//! card carrying a usable `max_num_seqs` would size the gate for its process. In
//! tree those router cards leave `max_num_seqs` unset today, so they report
//! nothing usable and the limit falls through to the next rule; that is a
//! property of the current router configs, not a guarantee this module enforces.
//!
//! Registration can complete after the gate has already admitted requests, so
//! the limit stays adjustable rather than being frozen at construction.
//!
//! The TCP request plane keeps its own unrelated `DYN_TCP_WORKER_POOL_SIZE` /
//! `DYN_TCP_WORK_QUEUE_SIZE` pool sizing.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, OnceLock, Weak};
use std::task::Poll;
use std::time::Duration;

use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};
use tokio::time::Instant;

use crate::engine::{
    AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data, EngineStream,
};
use crate::error::{DynamoError, ErrorType};

/// Overrides the gate's concurrent-request limit. Surfaced to operators as
/// `--engine-request-limit`.
const DYN_ENGINE_REQUEST_LIMIT: &str = "DYN_ENGINE_REQUEST_LIMIT";

/// Overrides the gate's maximum FIFO queue length.
const DYN_DYNAMO_REQUEST_QUEUE_LIMIT: &str = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT";

/// Overrides, in whole milliseconds, how long a request may stay in the FIFO
/// before it is no longer worth admitting. Environment-only: there is no flag.
const DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS: &str = "DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS";

/// Final fallback concurrent-request limit. This is the limit itself, not a
/// `max_num_seqs` stand-in: the 3/2 factor is never applied to it.
const DEFAULT_CONCURRENCY_LIMIT: usize = 10_000;

/// Default FIFO queue length.
const DEFAULT_QUEUE_CAPACITY: usize = 40_000;

/// Default maximum queue residence before a queued request is given up on.
const DEFAULT_QUEUE_DELAY: Duration = Duration::from_millis(5_000);

/// The message a shed request is refused with.
const OVERLOADED_MESSAGE: &str = "Server overloaded: worker at capacity";

/// The message a request that outlived the queue delay is refused with. It
/// carries the same [`ErrorType::WorkerOverloaded`] as a full queue and names
/// the queue delay as the reason rather than a full queue.
const EXPIRED_MESSAGE: &str =
    "Server overloaded: request rejected after exceeding the backend admission queue delay";

/// The message a request cancelled while queued is refused with. It is
/// deliberately not an overload: the caller went away, which is not
/// backpressure.
const CANCELLED_MESSAGE: &str = "Request cancelled while queued for backend admission";

/// The process-global gate. Constructed on first touch; its limit stays
/// adjustable so a late capacity hint is not lost.
static GATE: LazyLock<Arc<BackendAdmissionGate>> =
    LazyLock::new(BackendAdmissionGate::from_environment);

/// The one gate every request in this process admits through.
pub(crate) fn global() -> &'static Arc<BackendAdmissionGate> {
    &GATE
}

/// Record the capacity a model card publishes. Called for every non-LoRA base
/// model card registered in this process, whatever component registered it — see
/// the module docs on hint provenance. Sizing is advisory, so an absent or
/// unusable report simply leaves the previous limit in place.
pub fn record_engine_capacity(max_num_seqs: Option<u64>, data_parallel_size: Option<u32>) {
    global().record_capacity_report(max_num_seqs, data_parallel_size);
}

/// `ceil(3/2 * max_num_seqs * data_parallel_size)` in integer arithmetic, or
/// `None` when either input is missing, zero, or the product overflows.
fn automatic_concurrency_limit(
    max_num_seqs: Option<u64>,
    data_parallel_size: Option<u32>,
) -> Option<usize> {
    let sequences = max_num_seqs?.checked_mul(u64::from(data_parallel_size?))?;
    let limit = sequences.checked_mul(3)?.div_ceil(2);
    usize::try_from(limit).ok().filter(|limit| *limit > 0)
}

/// Resolve the concurrent-request limit from already-parsed inputs. Pure, so the
/// precedence rules are testable without touching the process environment.
///
/// Each rule is checked for validity on its own: a non-positive override does
/// not swallow a usable hint, it simply falls through to it.
fn resolve_concurrency_limit(env_override: Option<usize>, hint: Option<usize>) -> usize {
    if let Some(limit) = env_override.filter(|limit| *limit > 0) {
        return limit;
    }
    if let Some(limit) = hint.filter(|limit| *limit > 0) {
        return limit;
    }
    DEFAULT_CONCURRENCY_LIMIT
}

/// Resolve the maximum queue delay from an already-parsed override, in whole
/// milliseconds. Pure, so the precedence is testable without touching the
/// process environment.
///
/// A positive override replaces the default outright — one millisecond is a
/// valid setting — and anything else leaves [`DEFAULT_QUEUE_DELAY`] in place.
fn resolve_queue_delay(env_override_ms: Option<usize>) -> Duration {
    match env_override_ms
        .and_then(|ms| u64::try_from(ms).ok())
        .filter(|ms| *ms > 0)
    {
        Some(ms) => Duration::from_millis(ms),
        None => DEFAULT_QUEUE_DELAY,
    }
}

/// Parse a positive `usize` from `name`. Unset, unparseable, zero, or negative
/// all fall through to the next sizing rule.
fn positive_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().parse::<usize>() {
        Ok(value) if value > 0 => Some(value),
        _ => {
            tracing::warn!(
                env = name,
                value = %raw,
                "Ignoring invalid backend admission override; expected a positive integer"
            );
            None
        }
    }
}

/// The outcome the gate hands a queued ticket.
///
/// `Slot` transfers one unit of `active` capacity, so a ticket that has gone
/// away must refund it. `Expired` carries no capacity and its FIFO entry is
/// already removed, so it is never refunded and never unregistered.
enum Handoff {
    Slot,
    Expired,
}

/// Why the gate refused a request. Private: callers see only the typed error
/// [`reject`] builds from it, so no admission state escapes this module.
enum Rejection {
    /// The limit and the queue are both full.
    QueueFull,
    /// The request sat in the FIFO past its queue delay, so it is no longer
    /// worth admitting. Distinct from both a cancellation and a full queue.
    Expired,
    /// The request was cancelled while queued.
    Cancelled,
}

/// Trace a refusal against the request it refused, and build the error the
/// caller fails it with.
///
/// Shedding and queue-delay expiry are both backpressure on this one worker, so
/// both carry [`ErrorType::WorkerOverloaded`] and are told apart by their
/// message alone. A cancellation is not backpressure: the caller went away, so
/// classifying it as an overload would misreport a worker that never refused
/// anything.
fn reject(rejection: Rejection, context: Option<&dyn AsyncEngineContext>) -> DynamoError {
    let request_id = context.map(|context| context.id()).unwrap_or_default();
    let (error_type, message) = match rejection {
        Rejection::QueueFull => {
            tracing::warn!(
                request_id,
                "Worker at capacity (engine limit and queue both full), rejecting request"
            );
            (ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE)
        }
        Rejection::Expired => {
            tracing::warn!(
                request_id,
                "Request exceeded the backend admission queue delay, rejecting request"
            );
            (ErrorType::WorkerOverloaded, EXPIRED_MESSAGE)
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

/// A queued request, oldest first. The sender is the outcome handoff.
struct Waiter {
    id: u64,
    /// Absolute deadline, stamped under the state lock at enqueue. One
    /// process-wide delay budget makes these nondecreasing along the FIFO.
    due: Instant,
    tx: oneshot::Sender<Handoff>,
}

struct GateState {
    /// Positive `DYN_ENGINE_REQUEST_LIMIT`, read once. Always wins.
    env_override: Option<usize>,
    /// The adopted automatic hint.
    hint: Option<usize>,
    /// Effective limit; recomputed when a hint lands.
    limit: usize,
    /// Fixed FIFO length.
    queue_capacity: usize,
    /// Fixed maximum queue residence, applied to every entry at enqueue.
    queue_delay: Duration,
    /// Slots held by admitted requests. May briefly exceed `limit` after a
    /// shrink; a shrink never revokes a permit that is already held.
    active: usize,
    /// Live waiters, oldest first. Its length is exactly the queue occupancy.
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
    /// Bumped under this lock whenever the oldest entry changes, so the one
    /// expiry driver re-reads the head and re-arms its single timer.
    head_generation: watch::Sender<u64>,
}

impl GateState {
    /// Identity of the oldest live entry: which ticket it is and when it is due.
    fn head(&self) -> Option<(u64, Instant)> {
        self.waiters.front().map(|waiter| (waiter.id, waiter.due))
    }

    /// Deadline for a request enqueued at `now`. Constructed checked, so an
    /// operator-supplied delay too large for the monotonic clock degrades to the
    /// default budget instead of panicking here.
    fn due_at(&self, now: Instant) -> Instant {
        now.checked_add(self.queue_delay)
            .unwrap_or_else(|| now + DEFAULT_QUEUE_DELAY)
    }

    /// Apply `mutate`, then wake the expiry driver if the oldest entry changed.
    ///
    /// The bump happens under the state lock, and the driver marks the
    /// generation seen under that same lock before reading the head, so no
    /// mutation can slip between its read and its wait. An enqueue behind an
    /// unchanged head leaves the generation alone: the fixed delay budget makes
    /// later deadlines nondecreasing, so the armed timer is still the earliest.
    fn with_head_watch<R>(&mut self, mutate: impl FnOnce(&mut Self) -> R) -> R {
        let before = self.head();
        let result = mutate(self);
        if self.head() != before {
            self.head_generation
                .send_modify(|generation| *generation = generation.wrapping_add(1));
        }
        result
    }

    /// Remove the due prefix of the FIFO, oldest first, and return it so the
    /// tickets can be told.
    ///
    /// Deadlines are nondecreasing along the FIFO, so the first live entry ends
    /// the scan: this is one `pop_front` per expired request and never looks at
    /// the unexpired tail. Reaching the deadline exactly counts as expired.
    fn drain_expired(&mut self, now: Instant) -> Vec<Waiter> {
        let mut expired = Vec::new();
        while self.waiters.front().is_some_and(|waiter| waiter.due <= now) {
            expired.push(self.waiters.pop_front().expect("front was just observed"));
        }
        expired
    }

    /// Reject every entry due at `now`, freeing its queue capacity immediately,
    /// and report how many left the FIFO.
    ///
    /// Callers already hold the state lock; delivering here is safe because
    /// `oneshot::Sender::send` only stores the outcome and wakes a task.
    fn expire_due(&mut self, now: Instant) -> usize {
        let expired = self.with_head_watch(|state| state.drain_expired(now));
        let count = expired.len();
        for waiter in expired {
            // A ticket that already went away simply drops the outcome.
            let _ = waiter.tx.send(Handoff::Expired);
        }
        count
    }

    /// Expire against a freshly sampled clock until the head is live or the FIFO
    /// is empty.
    ///
    /// Draining a long prefix takes time, so the head is re-compared each pass:
    /// an entry that crosses its deadline during the cleanup must not be left
    /// holding queue capacity. The caller's decision linearizes at the final,
    /// live comparison.
    fn expire_due_now(&mut self) {
        while self.expire_due(Instant::now()) > 0 {}
    }

    /// Take `limit` as the process-global automatic hint if nothing has claimed
    /// that slot yet.
    fn adopt_hint(&mut self, limit: usize) {
        match self.hint {
            None => {
                self.hint = Some(limit);
                self.recompute_limit();
            }
            Some(existing) if existing != limit => {
                tracing::warn!(
                    kept = existing,
                    ignored = limit,
                    "Another model card in this process reports a different capacity; \
                     keeping the first one"
                );
            }
            Some(_) => {}
        }
    }

    /// Hand freed capacity to the oldest live waiters, skipping any that went
    /// away between enqueue and wake-up so their slot is refunded to the next.
    ///
    /// The due prefix is rejected before every candidate, under this same lock
    /// and against a freshly sampled clock, so a slot is never handed to a
    /// request whose deadline has elapsed — including one that reaches it while
    /// this loop skips past departed waiters ahead of it. The one gate timer
    /// makes expiry prompt; this makes it correct when a slot frees up first, or
    /// before the timer has been scheduled at all.
    fn wake_waiters(&mut self) {
        self.with_head_watch(|state| {
            loop {
                for waiter in state.drain_expired(Instant::now()) {
                    // A ticket that already went away simply drops the outcome.
                    let _ = waiter.tx.send(Handoff::Expired);
                }
                if state.active >= state.limit {
                    return;
                }
                let Some(waiter) = state.waiters.pop_front() else {
                    return;
                };
                if waiter.tx.send(Handoff::Slot).is_ok() {
                    state.active += 1;
                }
                // Otherwise the waiter is gone; popping it already reclaimed its
                // queue slot, so continue to the next one.
            }
        });
    }

    fn recompute_limit(&mut self) {
        let resolved = resolve_concurrency_limit(self.env_override, self.hint);
        if resolved == self.limit {
            return;
        }
        let previous = self.limit;
        self.limit = resolved;
        if resolved > previous {
            // Growth releases queued work immediately.
            self.wake_waiters();
        }
        tracing::info!(
            previous,
            limit = resolved,
            active = self.active,
            "Backend admission limit updated"
        );
    }
}

/// Drive the one expiry timer a gate ever owns.
///
/// There is no periodic sweep and no timer per queued request: this task pins a
/// single [`tokio::time::Sleep`] and `reset`s it onto the oldest live deadline
/// whenever the head moves, so the same timer serves the whole queue. It holds
/// the state through a [`Weak`], so it neither keeps a finished gate alive nor
/// outlives one, and it never holds the state lock across an await.
async fn drive_expiry(state: Weak<Mutex<GateState>>, mut head_changed: watch::Receiver<u64>) {
    let timer = tokio::time::sleep(Duration::ZERO);
    tokio::pin!(timer);

    loop {
        let head_due = {
            let Some(state) = state.upgrade() else {
                return;
            };
            let state = state.lock();
            // Marking the generation seen under the state lock is what makes the
            // notification lossless: any mutation from here on bumps it
            // afterwards, so `changed()` below cannot miss it.
            head_changed.mark_unchanged();
            state.head().map(|(_, due)| due)
        };

        let Some(due) = head_due else {
            // Nothing queued: leave the timer disarmed and wait on the signal
            // alone until a head exists again.
            if head_changed.changed().await.is_err() {
                return;
            }
            continue;
        };

        // Reset, never replace: one timer tracks whichever entry is oldest.
        timer.as_mut().reset(due);

        tokio::select! {
            _ = timer.as_mut() => {}
            changed = head_changed.changed() => {
                // Enqueue into an empty queue, cancellation, expiry or a permit
                // handoff moved the head; re-read it and reset the same timer.
                if changed.is_err() {
                    return;
                }
                continue;
            }
        }

        let expired = {
            let Some(state) = state.upgrade() else {
                return;
            };
            let mut state = state.lock();
            state.with_head_watch(|state| state.drain_expired(Instant::now()))
        };
        // Delivered with the state unlocked; the next loop pass re-reads the
        // head and re-arms.
        for waiter in expired {
            let _ = waiter.tx.send(Handoff::Expired);
        }
    }
}

/// One concurrency limit plus one bounded FIFO queue, shared by every endpoint
/// in the process.
pub(crate) struct BackendAdmissionGate {
    /// Held behind its own `Arc` so [`drive_expiry`] can watch it through a
    /// `Weak` without counting as a reference to the gate itself.
    state: Arc<Mutex<GateState>>,
    /// Guards the one-time start of the expiry driver.
    expiry_driver: OnceLock<()>,
}

impl BackendAdmissionGate {
    fn from_environment() -> Arc<Self> {
        let env_override = positive_env(DYN_ENGINE_REQUEST_LIMIT);
        let queue_capacity =
            positive_env(DYN_DYNAMO_REQUEST_QUEUE_LIMIT).unwrap_or(DEFAULT_QUEUE_CAPACITY);
        let queue_delay = resolve_queue_delay(positive_env(DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS));
        let gate = Self::new(env_override, queue_capacity, queue_delay);
        tracing::debug!(
            limit = gate.limit(),
            queue_capacity,
            // The gate's own value, so a delay it rejected as unrepresentable is
            // reported as the default it fell back to.
            queue_delay_ms = gate.queue_delay().as_millis(),
            has_env_override = env_override.is_some(),
            "Backend admission gate created"
        );
        gate
    }

    /// Build a standalone gate. Production uses [`global`]; tests build their
    /// own so they never contend for process-global capacity.
    ///
    /// No task is spawned here: [`global`] and standalone gates are both
    /// constructed outside a Tokio runtime, so the expiry driver starts on the
    /// first queued admission instead.
    fn new(env_override: Option<usize>, queue_capacity: usize, queue_delay: Duration) -> Arc<Self> {
        let limit = resolve_concurrency_limit(env_override, None);
        // Checked once here rather than per request: a delay the monotonic clock
        // cannot represent has no usable deadline, so fall back to the default.
        let queue_delay = match Instant::now().checked_add(queue_delay) {
            Some(_) => queue_delay,
            None => {
                tracing::warn!(
                    requested_ms = queue_delay.as_millis(),
                    "Backend admission queue delay is too large to represent; using the default"
                );
                DEFAULT_QUEUE_DELAY
            }
        };
        let (head_generation, _) = watch::channel(0);
        Arc::new(Self {
            state: Arc::new(Mutex::new(GateState {
                env_override,
                hint: None,
                limit,
                queue_capacity,
                queue_delay,
                active: 0,
                waiters: VecDeque::new(),
                next_ticket: 0,
                head_generation,
            })),
            expiry_driver: OnceLock::new(),
        })
    }

    /// Effective concurrent-request limit.
    fn limit(&self) -> usize {
        self.state.lock().limit
    }

    /// Configured maximum queue residence.
    fn queue_delay(&self) -> Duration {
        self.state.lock().queue_delay
    }

    /// Start the single expiry driver this gate will ever own. Called only from
    /// the queued admission path, which is always inside a Tokio runtime.
    fn ensure_expiry_driver(&self) {
        self.expiry_driver.get_or_init(|| {
            let head_changed = self.state.lock().head_generation.subscribe();
            tokio::spawn(drive_expiry(Arc::downgrade(&self.state), head_changed));
        });
    }

    /// Requests holding a slot.
    fn active(&self) -> usize {
        self.state.lock().active
    }

    /// Requests waiting in the FIFO.
    fn queued(&self) -> usize {
        self.state.lock().waiters.len()
    }

    /// Record a published capacity. Only the first usable report becomes the
    /// process-global hint, regardless of which component published it; a later
    /// conflicting one warns and is ignored. The environment override always
    /// takes precedence over any report.
    fn record_capacity_report(&self, max_num_seqs: Option<u64>, data_parallel_size: Option<u32>) {
        let Some(limit) = automatic_concurrency_limit(max_num_seqs, data_parallel_size) else {
            return;
        };
        self.state.lock().adopt_hint(limit);
    }

    /// Run `generate` under one admitted slot.
    ///
    /// This is the whole admission interface, and it is exactly
    /// [`AsyncEngine::generate`]'s own result type: capacity is acquired
    /// *before* `generate` is polled, so a refused request never reaches the
    /// engine, and what comes back is the engine's stream — no permit, no
    /// admission outcome, nothing to match on. The slot lives in that stream, so
    /// it is released when the stream completes, is dropped, is cancelled, or
    /// its task is aborted; a `generate` that fails after admission drops the
    /// slot immediately.
    ///
    /// A refusal is traced and classified here and surfaces as the standardized
    /// [`DynamoError`]; an engine failure keeps its own error untouched.
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
        let slot = self.acquire(context).await.map_err(anyhow::Error::new)?;
        let stream = generate.await?;
        Ok(Box::pin(AdmittedStream {
            inner: stream,
            slot: Some(slot),
        }))
    }

    /// Take one slot, waiting in FIFO order when the limit is busy.
    ///
    /// The lock is never held across an await: the decision is made under the
    /// lock, and only the handoff is awaited.
    async fn acquire(
        self: &Arc<Self>,
        context: Option<&dyn AsyncEngineContext>,
    ) -> Result<ActiveSlot, DynamoError> {
        enum Decision {
            Granted,
            Queued(u64, oneshot::Receiver<Handoff>),
            Rejected,
        }

        // A caller that has already gone away must not take a slot. Direct
        // admission grants without ever awaiting, so this is the only point at
        // which that can be caught before `generate` is polled; the queued path
        // re-checks in `AdmissionTicket::wait`, where the context can also stop
        // while the request waits.
        if context.is_some_and(|context| context.is_stopped()) {
            return Err(reject(Rejection::Cancelled, context));
        }

        let decision = {
            let mut state = self.state.lock();
            // Entries the queue delay has already given up on must not hold
            // capacity against this request, or a stale prefix would shed it as
            // overloaded and a stale head would be dispatched ahead of it.
            state.expire_due_now();
            // Direct admission only when nothing is waiting, so a new request
            // can never bypass an older one. A directly admitted request never
            // carries a queue deadline.
            if state.active < state.limit && state.waiters.is_empty() {
                state.active += 1;
                Decision::Granted
            } else if state.waiters.len() < state.queue_capacity {
                let id = state.next_ticket;
                state.next_ticket = state.next_ticket.wrapping_add(1);
                let (tx, rx) = oneshot::channel();
                // Sampled at the actual enqueue point and stamped under this
                // lock, so the deadline is this request's own queue-entry time
                // plus the budget, and FIFO order is also nondecreasing
                // deadline order.
                let due = state.due_at(Instant::now());
                state.with_head_watch(|state| state.waiters.push_back(Waiter { id, due, tx }));
                Decision::Queued(id, rx)
            } else {
                Decision::Rejected
            }
        };

        match decision {
            Decision::Granted => Ok(ActiveSlot {
                gate: Arc::clone(self),
            }),
            Decision::Rejected => Err(reject(Rejection::QueueFull, context)),
            Decision::Queued(id, rx) => {
                // The queue is only reachable from an async caller, so this is
                // the first point at which a runtime is guaranteed to exist.
                self.ensure_expiry_driver();
                AdmissionTicket {
                    gate: Arc::clone(self),
                    id,
                    rx: Some(rx),
                    taken: false,
                }
                .wait(context)
                .await
            }
        }
    }

    /// Release one held slot back to the oldest live waiter, or to the limit.
    fn release(&self) {
        let mut state = self.state.lock();
        state.active = state.active.saturating_sub(1);
        state.wake_waiters();
    }

    /// Drop an abandoned waiter so it stops consuming queue capacity.
    ///
    /// Cancellation can land anywhere in the FIFO, so this stays a linear
    /// search; the expiry path never uses it.
    fn remove_waiter(&self, id: u64) {
        let mut state = self.state.lock();
        state.with_head_watch(|state| {
            if let Some(index) = state.waiters.iter().position(|waiter| waiter.id == id) {
                state.waiters.remove(index);
            }
        });
    }

    /// Seed a starting limit that a later hint can still resize. Production
    /// sizing always comes from the environment or a hint.
    #[cfg(test)]
    fn set_limit_for_test(&self, limit: usize) {
        let mut state = self.state.lock();
        state.limit = limit;
        state.wake_waiters();
    }

    /// Enqueue a waiter with an explicit deadline, bypassing [`Self::acquire`], so
    /// a test can build a large or deliberately out-of-order queue without a
    /// task per entry and without starting the expiry driver.
    #[cfg(test)]
    fn push_waiter_for_test(&self, due: Instant) -> oneshot::Receiver<Handoff> {
        let mut state = self.state.lock();
        let id = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let (tx, rx) = oneshot::channel();
        state.with_head_watch(|state| state.waiters.push_back(Waiter { id, due, tx }));
        rx
    }

    /// Run the locked expiry transition at an explicit instant, so a test can
    /// place the deadline boundary exactly.
    #[cfg(test)]
    fn expire_due_for_test(&self, now: Instant) -> usize {
        self.state.lock().expire_due(now)
    }
}

/// One unit of live concurrency. Private, and never named in any signature a
/// caller can reach: it is only ever owned by an [`AdmittedStream`], so no
/// caller can hold, forget or forge one. Dropping it — on end-of-stream, a
/// generate error, an encode or publish error, task abort, or cancellation —
/// releases the slot to the oldest queued request.
struct ActiveSlot {
    gate: Arc<BackendAdmissionGate>,
}

impl Drop for ActiveSlot {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// The engine's own response stream, holding the slot it was admitted on.
///
/// It is an [`EngineStream`] like any other — item type, items and context are
/// all the engine's — so the caller sees no admission type at all.
struct AdmittedStream<R: Data> {
    inner: EngineStream<R>,
    /// Taken at end-of-stream so a caller that keeps an exhausted stream around
    /// does not keep the slot with it; `Drop` covers every other exit.
    slot: Option<ActiveSlot>,
}

impl<R: Data> Stream for AdmittedStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None)) {
            drop(self.slot.take());
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

/// A place in the FIFO. Dropping it before the slot arrives unregisters the
/// waiter; dropping it after the slot was handed over refunds that slot so the
/// next waiter is woken instead of it being lost.
struct AdmissionTicket {
    gate: Arc<BackendAdmissionGate>,
    id: u64,
    rx: Option<oneshot::Receiver<Handoff>>,
    /// Set once the gate's outcome has been consumed — a slot converted into an
    /// [`ActiveSlot`], or an expiry that already removed this entry — so `Drop`
    /// has nothing left to refund or unregister.
    taken: bool,
}

impl AdmissionTicket {
    async fn wait(
        mut self,
        context: Option<&dyn AsyncEngineContext>,
    ) -> Result<ActiveSlot, DynamoError> {
        let outcome = {
            let rx = self.rx.as_mut().expect("ticket always holds its receiver");
            match context {
                // Cancellation while queued must be prompt, and it must win a
                // simultaneous handoff. A slot can be sent to this waiter after
                // the context was stopped but before this future is polled
                // again; admitting then would run a request the caller already
                // abandoned and would hold capacity ahead of live waiters. The
                // precheck plus the biased ordering make cancellation strictly
                // higher priority; `Drop` returns the already-sent slot to the
                // next waiter.
                Some(context) if context.is_stopped() => None,
                Some(context) => tokio::select! {
                    biased;
                    _ = context.stopped() => None,
                    handoff = &mut *rx => handoff.ok(),
                },
                None => rx.await.ok(),
            }
        };

        match outcome {
            Some(Handoff::Slot) => {
                self.taken = true;
                Ok(ActiveSlot {
                    gate: Arc::clone(&self.gate),
                })
            }
            Some(Handoff::Expired) => {
                // The gate popped this entry before sending, so there is no
                // queue slot to release and no capacity to refund.
                self.taken = true;
                Err(reject(Rejection::Expired, context))
            }
            None => Err(reject(Rejection::Cancelled, context)),
        }
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        if self.taken {
            return;
        }
        let Some(mut rx) = self.rx.take() else {
            return;
        };
        // Closing first makes the handoff race single-winner: either the sender
        // already succeeded and `try_recv` yields the slot we must refund, or
        // the send fails and the releaser moves on to the next waiter.
        rx.close();
        match rx.try_recv() {
            Ok(Handoff::Slot) => self.gate.release(),
            // Cancellation beat a ready expiry. The entry is already out of the
            // FIFO and no slot was ever allocated, so `active` must not move.
            Ok(Handoff::Expired) => {}
            Err(_) => self.gate.remove_waiter(self.id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A gate whose queue delay is far longer than any test that is not about
    /// expiry, so those tests see the pre-Controlled-Delay behaviour exactly.
    fn gate(limit: usize, queue: usize) -> Arc<BackendAdmissionGate> {
        gate_with_delay(limit, queue, Duration::from_secs(3_600))
    }

    fn gate_with_delay(limit: usize, queue: usize, delay: Duration) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(Some(limit), queue, delay)
    }

    /// A gate with no environment override, so its limit comes from a hint or
    /// the fallback.
    fn hint_sized_gate(queue: usize) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(None, queue, Duration::from_secs(3_600))
    }

    /// The scheduling tests drive [`BackendAdmissionGate::acquire`] directly:
    /// FIFO order, refunds and expiry are all decided before an engine is ever
    /// involved, and a slot is what those transitions move around. The public
    /// stream-returning [`BackendAdmissionGate::admit`] is covered separately,
    /// at the end.
    type Acquired = Result<ActiveSlot, DynamoError>;

    /// Spawn an acquisition — it only resolves once the gate decides, so a
    /// queueing request can never be awaited on the test task — and return once
    /// it has joined the FIFO. Yields rather than sleeps, so a paused clock does
    /// not drift while the spawned task registers.
    async fn spawn_queued_admit(
        gate: &Arc<BackendAdmissionGate>,
    ) -> tokio::task::JoinHandle<Acquired> {
        let before = gate.queued();
        let handle = tokio::spawn({
            let gate = Arc::clone(gate);
            async move { gate.acquire(None).await }
        });
        for _ in 0..1_000 {
            if gate.queued() != before {
                return handle;
            }
            tokio::task::yield_now().await;
        }
        panic!("spawned request never joined the queue");
    }

    fn permit(acquired: Acquired) -> ActiveSlot {
        acquired.expect("expected admission")
    }

    /// The `(type, message)` pair a refusal carries, or `None` once a slot is
    /// granted. Both halves matter: shedding and expiry share one error type and
    /// are told apart by their message alone.
    fn refusal(acquired: &Acquired) -> Option<(ErrorType, &str)> {
        acquired
            .as_ref()
            .err()
            .map(|error| (error.error_type(), error.message()))
    }

    fn is_queue_full(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE))
    }

    fn is_expired(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::WorkerOverloaded, EXPIRED_MESSAGE))
    }

    fn is_cancelled(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
    }

    ///////////////////// SIZING: PURE RESOLVER INPUTS /////////////////////

    /// Precedence, and the validity of each rule on its own: a non-positive
    /// override must fall through to a usable hint rather than swallow it.
    #[test]
    fn the_limit_resolves_by_precedence_with_each_rule_validated() {
        const FALLBACK: usize = DEFAULT_CONCURRENCY_LIMIT;
        for (env_override, hint, expected) in [
            (Some(7), Some(384), 7),
            (Some(7), None, 7),
            (None, Some(384), 384),
            (Some(0), Some(384), 384),
            (Some(0), Some(0), FALLBACK),
            (None, Some(0), FALLBACK),
            (Some(0), None, FALLBACK),
            (None, None, FALLBACK),
        ] {
            assert_eq!(
                resolve_concurrency_limit(env_override, hint),
                expected,
                "override={env_override:?} hint={hint:?}"
            );
        }
        // The fallback is the limit itself, never routed through the 3/2 factor.
        assert_eq!(FALLBACK, 10_000);
    }

    /// `ceil(3/2 * max_num_seqs * dp)`, and `None` for every input that has no
    /// usable product: missing, zero, or overflowing.
    #[test]
    fn the_automatic_limit_is_integer_ceil_of_three_halves() {
        for (max_num_seqs, dp, expected) in [
            (Some(1), Some(1), Some(2)),
            (Some(3), Some(1), Some(5)),
            (Some(4), Some(1), Some(6)),
            (Some(256), Some(1), Some(384)),
            (Some(2), Some(2), Some(6)),
            (Some(5), Some(3), Some(23)),
            (None, Some(1), None),
            (Some(256), None, None),
            (Some(0), Some(1), None),
            (Some(256), Some(0), None),
            (Some(u64::MAX), Some(1), None),
            (Some(u64::MAX), Some(2), None),
        ] {
            assert_eq!(
                automatic_concurrency_limit(max_num_seqs, dp),
                expected,
                "max_num_seqs={max_num_seqs:?} dp={dp:?}"
            );
        }
    }

    ///////////////////// ADMISSION: N + EXACT Q /////////////////////

    #[tokio::test]
    async fn n_direct_admissions_then_exactly_q_queue_then_reject() {
        let gate = gate(2, 3);

        let held: Vec<_> = vec![
            permit(gate.acquire(None).await),
            permit(gate.acquire(None).await),
        ];
        assert_eq!(gate.active(), 2);
        assert_eq!(gate.queued(), 0);

        // Exactly Q queue, and no dispatcher hides a Q+1th.
        let mut waiters = Vec::new();
        for _ in 0..3 {
            waiters.push(tokio::spawn({
                let gate = Arc::clone(&gate);
                async move { gate.acquire(None).await }
            }));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(gate.queued(), 3, "queue holds exactly Q waiters");

        assert!(
            is_queue_full(&gate.acquire(None).await),
            "the Q+1th request must be shed"
        );
        assert_eq!(gate.queued(), 3, "a rejection must not consume queue space");

        drop(held);
        for waiter in waiters {
            let admission = waiter.await.expect("waiter completes");
            assert!(admission.is_ok());
        }
    }

    #[tokio::test]
    async fn queue_is_fifo_and_a_new_request_never_bypasses_an_older_waiter() {
        // `admit` resolves only once the request is admitted, so the queueing
        // decision is observable through `queued()` and the admission order,
        // never by awaiting `admit` on the test task.
        let gate = gate(1, 8);
        let held = permit(gate.acquire(None).await);

        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        // Each arrival joins behind the waiters already queued, so the last one
        // in must also be the last one served.
        for index in 0..5usize {
            let before = gate.queued();
            handles.push(tokio::spawn({
                let gate = Arc::clone(&gate);
                let order = Arc::clone(&order);
                async move {
                    let permit = permit(gate.acquire(None).await);
                    order.lock().push(index);
                    drop(permit);
                }
            }));
            for _ in 0..1_000 {
                if gate.queued() != before {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(gate.queued(), before + 1, "arrival {index} joined the FIFO");
        }
        assert_eq!(gate.queued(), 5, "every arrival joined the FIFO");

        drop(held);
        for handle in handles {
            handle.await.expect("waiter completes");
        }

        assert_eq!(
            *order.lock(),
            vec![0, 1, 2, 3, 4],
            "queued requests must be admitted oldest first"
        );
        assert_eq!(gate.queued(), 0);
        assert_eq!(gate.active(), 0);
    }

    ///////////////////// CANCELLATION AND REFUND /////////////////////

    #[tokio::test]
    async fn a_dropped_queued_waiter_frees_its_queue_slot_immediately() {
        let gate = gate(1, 1);
        let _held = permit(gate.acquire(None).await);

        let waiter = spawn_queued_admit(&gate).await;
        assert_eq!(gate.queued(), 1);
        assert!(is_queue_full(&gate.acquire(None).await), "queue is full");

        waiter.abort();
        let _ = waiter.await;
        for _ in 0..1_000 {
            if gate.queued() == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(gate.queued(), 0, "an abandoned waiter must unregister");

        // The freed queue slot is reusable: this request queues rather than
        // being shed.
        let reuse = spawn_queued_admit(&gate).await;
        assert_eq!(gate.queued(), 1, "the freed queue slot must be reusable");

        drop(_held);
        let admission = tokio::time::timeout(Duration::from_secs(5), reuse)
            .await
            .expect("the requeued request must be admitted")
            .expect("waiter completes");
        drop(permit(admission));
    }

    #[tokio::test]
    async fn cancellation_beats_a_simultaneous_handoff_and_refunds_the_slot() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 4);
        let held = permit(gate.acquire(None).await);

        // Waiter A is cancellable; poll it exactly once so it registers in the
        // FIFO and then stops being polled.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.acquire(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());
        assert_eq!(gate.queued(), 1);

        // Waiter B queues behind it and is the one that must inherit the slot.
        let survivor = spawn_queued_admit(&gate).await;
        assert_eq!(gate.queued(), 2);

        // Make both outcomes ready for waiter A before it is polled again: the
        // context is stopped, and then the released slot is handed to it.
        controller.stop_generating();
        drop(held);
        assert_eq!(gate.active(), 1, "the slot was handed to the doomed waiter");
        assert_eq!(gate.queued(), 1, "only the survivor is still queued");

        // Cancellation must win even though the handoff is also ready.
        let admission = doomed.await;
        assert!(
            is_cancelled(&admission),
            "a cancelled request must not be admitted by a racing handoff"
        );
        drop(admission);

        // The refunded slot goes to the next live waiter.
        let admission = tokio::time::timeout(Duration::from_secs(5), survivor)
            .await
            .expect("the surviving waiter must inherit the refunded slot")
            .expect("waiter completes");
        let permit = permit(admission);
        assert_eq!(gate.active(), 1);
        assert_eq!(gate.queued(), 0);

        drop(permit);
        assert_eq!(gate.active(), 0);
        assert_eq!(gate.queued(), 0);
        assert_eq!(
            Arc::strong_count(&gate),
            1,
            "tickets and permits must not retain the gate"
        );
    }

    #[tokio::test]
    async fn queued_requests_wake_on_context_cancellation() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 4);
        let _held = permit(gate.acquire(None).await);

        let controller = Arc::new(Controller::default());
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            let controller: Arc<dyn AsyncEngineContext> = controller.clone();
            async move { gate.acquire(Some(controller.as_ref())).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1);

        controller.stop_generating();

        let admission = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("a cancelled waiter must return promptly")
            .expect("waiter completes");
        assert!(is_cancelled(&admission));
        assert_eq!(gate.queued(), 0, "cancellation frees the queue slot");
    }

    ///////////////////// LATE HINTS AND RESIZING /////////////////////

    #[tokio::test]
    async fn a_late_report_still_applies() {
        let gate = hint_sized_gate(4);
        // Model-card registration lands after the gate has already served
        // requests on the fallback limit.
        drop(permit(gate.acquire(None).await));

        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 384, "a late report must not be lost");
    }

    #[tokio::test]
    async fn conflicting_reports_keep_the_first() {
        let gate = hint_sized_gate(4);

        gate.record_capacity_report(Some(2), Some(1));
        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 3, "the first usable report wins");

        // Re-reporting the same value is not a conflict.
        gate.record_capacity_report(Some(2), Some(1));
        assert_eq!(gate.limit(), 3);
    }

    #[tokio::test]
    async fn no_hint_leaves_the_fallback_in_place() {
        let gate = hint_sized_gate(4);
        gate.record_capacity_report(None, Some(1));
        gate.record_capacity_report(Some(0), Some(1));
        assert_eq!(gate.limit(), DEFAULT_CONCURRENCY_LIMIT);
    }

    #[tokio::test]
    async fn the_env_override_survives_any_later_hint() {
        let gate = gate(5, 4);
        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 5);
    }

    #[tokio::test]
    async fn growth_wakes_queued_waiters() {
        let gate = hint_sized_gate(4);
        gate.set_limit_for_test(1);
        let _held = permit(gate.acquire(None).await);

        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.acquire(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1);

        // A hint that raises the limit must release the queued request.
        gate.record_capacity_report(Some(4), Some(1));
        assert_eq!(gate.limit(), 6);

        let admission = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("growth must wake the FIFO")
            .expect("waiter completes");
        assert!(admission.is_ok());
    }

    #[tokio::test]
    async fn shrinking_never_revokes_a_held_permit() {
        // Starts on the fallback limit, so four permits are admitted freely.
        let gate = hint_sized_gate(4);
        let permits = vec![
            permit(gate.acquire(None).await),
            permit(gate.acquire(None).await),
            permit(gate.acquire(None).await),
            permit(gate.acquire(None).await),
        ];
        assert_eq!(gate.active(), 4);

        // ceil(3/2 * 1 * 1) = 2, below the four slots already held.
        gate.record_capacity_report(Some(1), Some(1));
        assert_eq!(gate.limit(), 2);
        assert_eq!(gate.active(), 4, "held permits are never revoked");

        // New work waits until `active` drains below the smaller limit.
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.acquire(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1, "the shrunken limit holds new work back");

        drop(permits);
        let admission = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("draining below the new limit must admit the waiter")
            .expect("waiter completes");
        drop(permit(admission));
        assert_eq!(gate.active(), 0);
    }

    ///////////////////// CONTROLLED DELAY: SIZING /////////////////////

    #[test]
    fn the_queue_delay_comes_from_a_positive_millisecond_override() {
        assert_eq!(DEFAULT_QUEUE_DELAY, Duration::from_millis(5_000));
        // `positive_env` already maps unset and unparseable to `None`, so zero
        // is the only other value that can reach here. One millisecond is valid,
        // and needs no fractional parsing.
        for (override_ms, expected) in [
            (None, DEFAULT_QUEUE_DELAY),
            (Some(0), DEFAULT_QUEUE_DELAY),
            (Some(1), Duration::from_millis(1)),
            (Some(250), Duration::from_millis(250)),
            (Some(60_000), Duration::from_secs(60)),
        ] {
            let delay = resolve_queue_delay(override_ms);
            assert_eq!(delay, expected, "override {override_ms:?}");
            assert_eq!(
                BackendAdmissionGate::new(Some(1), 1, delay).queue_delay(),
                expected,
                "the resolved delay must reach the gate"
            );
        }
    }

    #[test]
    fn a_delay_too_large_to_represent_falls_back_to_the_default() {
        // Enqueue stamps `now + delay`, so an unbounded override must degrade
        // rather than panic there.
        assert_eq!(
            BackendAdmissionGate::new(Some(1), 1, Duration::MAX).queue_delay(),
            DEFAULT_QUEUE_DELAY
        );
    }

    ///////////////////// CONTROLLED DELAY: EXPIRY /////////////////////

    #[tokio::test(start_paused = true)]
    async fn a_queued_request_expires_and_gives_its_queue_place_back() {
        // One slot and one queue place. The holder never finishes, so nothing
        // releases a slot: only the gate's own timer can resolve the waiter.
        let gate = gate_with_delay(1, 1, Duration::from_millis(200));
        let _held = permit(gate.acquire(None).await);
        let waiting = spawn_queued_admit(&gate).await;
        assert!(is_queue_full(&gate.acquire(None).await), "queue is full");

        assert!(
            is_expired(&waiting.await.expect("waiter completes")),
            "the queue delay must resolve a request no slot was ever freed for"
        );
        assert_eq!(gate.queued(), 0, "an expired request leaves the FIFO");
        assert_eq!(gate.active(), 1, "expiry allocates no capacity");

        // The reclaimed queue place is usable straight away.
        let reuse = spawn_queued_admit(&gate).await;
        assert_eq!(gate.queued(), 1);
        reuse.abort();
        let _ = reuse.await;
    }

    /// Expiry pops the due head and stops. Three oracles in one queue: the
    /// boundary is inclusive, a live head ends the scan, and the cost is the
    /// number of entries that actually expired — a long tail is never inspected
    /// and survives intact in FIFO order.
    #[tokio::test]
    async fn expiry_removes_only_the_due_head_prefix_and_never_scans_the_tail() {
        const TAIL: usize = 10_000;
        let gate = gate_with_delay(1, TAIL + 8, Duration::from_millis(50));
        let now = Instant::now();
        let mut overdue = gate.push_waiter_for_test(now - Duration::from_millis(1));
        let mut exact = gate.push_waiter_for_test(now);
        let mut live = gate.push_waiter_for_test(now + Duration::from_secs(60));
        // Deliberately out of order and long overdue, behind a live head: only a
        // search of the tail could reach it.
        let mut buried = gate.push_waiter_for_test(now - Duration::from_secs(60));
        let _tail: Vec<_> = (0..TAIL)
            .map(|_| gate.push_waiter_for_test(now + Duration::from_secs(60)))
            .collect();

        gate.expire_due_for_test(now);

        assert!(matches!(overdue.try_recv(), Ok(Handoff::Expired)));
        assert!(
            matches!(exact.try_recv(), Ok(Handoff::Expired)),
            "reaching the deadline exactly counts as expired"
        );
        assert_eq!(
            gate.queued(),
            TAIL + 2,
            "exactly the due prefix was removed and a live head ended the scan"
        );
        for rx in [&mut live, &mut buried] {
            assert!(
                matches!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty)),
                "the unexpired tail is never inspected"
            );
        }
        assert!(
            gate.state
                .lock()
                .waiters
                .iter()
                .map(|waiter| waiter.id)
                .eq(2..(2 + TAIL as u64 + 2)),
            "the surviving tail kept its entries and their FIFO order"
        );
    }

    ///////////////////// CONTROLLED DELAY: EXPIRY VS HANDOFF /////////////////////

    #[tokio::test]
    async fn no_grant_path_ever_dispatches_an_expired_head() {
        /// An already-due entry ahead of a live one, pushed directly so no
        /// expiry driver exists: the locked check on each grant-producing path
        /// is the only thing that can reject the first.
        fn due_then_live(
            gate: &Arc<BackendAdmissionGate>,
        ) -> (oneshot::Receiver<Handoff>, oneshot::Receiver<Handoff>) {
            let now = Instant::now();
            (
                gate.push_waiter_for_test(now - Duration::from_millis(1)),
                gate.push_waiter_for_test(now + Duration::from_secs(60)),
            )
        }

        let gate = BackendAdmissionGate::new(None, 8, Duration::from_millis(50));
        gate.set_limit_for_test(1);
        let held = permit(gate.acquire(None).await);

        // Releasing a permit.
        let (mut overdue, mut live) = due_then_live(&gate);
        drop(held);
        assert!(
            matches!(overdue.try_recv(), Ok(Handoff::Expired)),
            "a slot must never be granted to an entry that was already due"
        );
        assert!(
            matches!(live.try_recv(), Ok(Handoff::Slot)),
            "the handoff must advance to the oldest unexpired request"
        );
        assert_eq!(gate.active(), 1, "exactly one slot was handed over");

        // Growing the limit: ceil(3/2 * 4 * 1) = 6, above the slot in use.
        let (mut overdue, mut live) = due_then_live(&gate);
        gate.record_capacity_report(Some(4), Some(1));
        assert!(matches!(overdue.try_recv(), Ok(Handoff::Expired)));
        assert!(matches!(live.try_recv(), Ok(Handoff::Slot)));
        assert_eq!(gate.active(), 2);
        assert_eq!(gate.queued(), 0);

        // A departed waiter ahead of an already-due one. The overdue entry is
        // only reachable after the skip, so it is caught only if the deadline is
        // re-checked per candidate rather than sampled once per handoff.
        let departed = gate.push_waiter_for_test(Instant::now() + Duration::from_secs(60));
        let (mut overdue, mut live) = due_then_live(&gate);
        drop(departed);

        gate.set_limit_for_test(8);
        assert!(
            matches!(overdue.try_recv(), Ok(Handoff::Expired)),
            "skipping a departed waiter must not hand its slot to an expired one"
        );
        assert!(matches!(live.try_recv(), Ok(Handoff::Slot)));
        assert_eq!(gate.active(), 3);
        assert_eq!(gate.queued(), 0);
    }

    #[tokio::test]
    async fn an_expired_head_is_pruned_before_the_queue_full_decision() {
        // The single queue slot is held by an entry the delay budget has already
        // given up on, so an arriving request must inherit it, not be shed.
        let gate = gate_with_delay(1, 1, Duration::from_millis(50));
        let _held = permit(gate.acquire(None).await);
        let mut stale = gate.push_waiter_for_test(Instant::now() - Duration::from_millis(1));
        assert_eq!(gate.queued(), 1);

        let mut arriving = Box::pin(gate.acquire(None));
        assert!(futures::poll!(&mut arriving).is_pending());

        assert!(matches!(stale.try_recv(), Ok(Handoff::Expired)));
        assert_eq!(
            gate.queued(),
            1,
            "the arriving request took the queue slot the expired entry vacated"
        );
    }

    #[tokio::test]
    async fn cancellation_stays_distinct_from_expiry_and_never_refunds_capacity() {
        use crate::pipeline::context::Controller;

        let gate = gate_with_delay(1, 4, Duration::from_millis(50));
        let held = permit(gate.acquire(None).await);

        // Poll once so the ticket registers, then stop being polled.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.acquire(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());
        assert_eq!(gate.queued(), 1);

        // Make both outcomes ready before the ticket is polled again.
        controller.stop_generating();
        gate.expire_due_for_test(Instant::now() + Duration::from_secs(1));
        assert_eq!(gate.queued(), 0, "expiry removed the entry from the FIFO");

        let admission = doomed.await;
        assert!(
            is_cancelled(&admission),
            "cancellation must win a simultaneous expiry, and stay distinct from it"
        );
        drop(admission);
        assert_eq!(
            gate.active(),
            1,
            "an expiry carries no slot, so there is nothing to refund"
        );

        drop(held);
        assert_eq!(gate.active(), 0);
        assert_eq!(gate.queued(), 0);
    }

    ///////////////////// CONTROLLED DELAY: ONE TIMER /////////////////////

    #[tokio::test(start_paused = true)]
    async fn one_timer_re_arms_as_the_head_moves_and_stops_with_the_gate() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(400));
        let held = permit(gate.acquire(None).await);

        // One timer, three entries, two distinct deadlines: 400 ms and 700 ms.
        let start = Instant::now();
        let cancelled = spawn_queued_admit(&gate).await;
        let first = spawn_queued_admit(&gate).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        let second = spawn_queued_admit(&gate).await;

        // Cancellation moves the head; the same timer must re-arm behind it.
        cancelled.abort();
        let _ = cancelled.await;
        while gate.queued() != 2 {
            tokio::task::yield_now().await;
        }

        assert!(is_expired(&first.await.expect("waiter completes")));
        let first_at = start.elapsed();
        assert!(
            (Duration::from_millis(400)..Duration::from_millis(700)).contains(&first_at),
            "the head must expire on its own deadline, not the tail's: {first_at:?}"
        );
        assert_eq!(gate.queued(), 1, "only the due head was removed");

        assert!(is_expired(&second.await.expect("waiter completes")));
        let second_at = start.elapsed();
        assert!(
            second_at >= Duration::from_millis(700),
            "the same timer must be reset onto the later deadline: {second_at:?}"
        );
        assert_eq!(gate.queued(), 0);

        // The driver retains only a `Weak` reference and never holds the state
        // lock across an await, so dropping the gate must release the state
        // rather than leak a task and its state per gate.
        let weak_state = Arc::downgrade(&gate.state);
        drop(held);
        drop(gate);
        assert!(
            weak_state.upgrade().is_none(),
            "the expiry driver kept the dropped gate's state alive"
        );
    }

    ///////////////////// REFUSAL: STANDARDIZED TYPED ERRORS /////////////////////

    #[test]
    fn shedding_and_expiry_are_worker_scoped_overloads() {
        // Both are backpressure on this one worker, so both carry the
        // worker-scoped overload and only the message tells them apart. A
        // pool-scoped `ResourceExhausted` would claim the whole pool is out of
        // room, and a `Backend` error would report a fault that did not happen.
        for (rejection, expected) in [
            (Rejection::QueueFull, OVERLOADED_MESSAGE),
            (Rejection::Expired, EXPIRED_MESSAGE),
        ] {
            let error = reject(rejection, None);
            assert_eq!(error.error_type(), ErrorType::WorkerOverloaded);
            assert_eq!(error.message(), expected);
        }
        assert_ne!(OVERLOADED_MESSAGE, EXPIRED_MESSAGE);
        assert!(EXPIRED_MESSAGE.contains("queue delay"));
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

    fn engine_context() -> Arc<dyn AsyncEngineContext> {
        Arc::new(crate::pipeline::context::Controller::default())
    }

    #[tokio::test]
    async fn an_admitted_stream_is_the_engines_own_and_holds_the_slot_to_its_end() {
        let gate = gate(1, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let context = engine_context();

        let mut stream = gate
            .admit(None, generate(Arc::clone(&ran), Arc::clone(&context)))
            .await
            .expect("the first request is admitted");
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        assert_eq!(gate.active(), 1, "the returned stream carries the slot");
        assert!(
            Arc::ptr_eq(&stream.context(), &context),
            "the engine's own context must be delegated, not replaced"
        );

        // The items are the engine's, unchanged.
        assert_eq!(stream.next().await, Some(1));
        assert_eq!(stream.next().await, Some(2));
        assert_eq!(gate.active(), 1, "the slot is held for the whole stream");

        // End-of-stream releases it, without waiting for the caller to drop an
        // exhausted stream.
        assert_eq!(stream.next().await, None);
        assert_eq!(gate.active(), 0, "completion releases the slot");
        drop(stream);
        assert_eq!(gate.active(), 0, "and dropping it does not double-release");
    }

    #[tokio::test]
    async fn an_unfinished_stream_releases_its_slot_when_dropped() {
        let gate = gate(1, 0);
        let ran = Arc::new(AtomicUsize::new(0));

        let stream = gate
            .admit(None, generate(Arc::clone(&ran), engine_context()))
            .await
            .expect("the first request is admitted");
        assert_eq!(gate.active(), 1);

        // Abandoned mid-stream: cancellation, task abort and a client that goes
        // away all end here.
        drop(stream);
        assert_eq!(gate.active(), 0);
    }

    #[tokio::test]
    async fn aborting_admit_while_generate_is_pending_releases_the_slot() {
        // The slot is taken before `generate` is polled, so the window between
        // admission and a stream existing has no wrapper to release it — only
        // dropping the `admit` future itself can.
        let gate = gate(1, 4);
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
        assert_eq!(gate.active(), 1, "the slot is held while generate runs");

        task.abort();
        let _ = task.await;
        assert_eq!(gate.active(), 0, "aborting mid-generate releases the slot");
    }

    #[tokio::test]
    async fn a_failed_generate_releases_its_slot_and_keeps_its_own_error() {
        let gate = gate(1, 0);

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
        assert_eq!(gate.active(), 0, "a failed generate frees the slot at once");
    }

    #[tokio::test]
    async fn a_refused_request_never_reaches_the_engine() {
        // Limit 1, queue 0: the first request holds the only slot.
        let gate = gate(1, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = gate
            .admit(None, generate(Arc::clone(&ran), engine_context()))
            .await
            .expect("the first request is admitted");

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
    /// is completely idle. Direct admission grants without awaiting, so nothing
    /// downstream of the decision would ever notice the caller had gone.
    #[tokio::test]
    async fn an_already_cancelled_request_never_reaches_the_engine() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 4);
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
            "the generate future must never be polled"
        );
        let rejection = error
            .downcast_ref::<DynamoError>()
            .expect("a refusal is a DynamoError");
        assert_eq!(rejection.error_type(), ErrorType::Cancelled);
        assert_eq!(rejection.message(), CANCELLED_MESSAGE);
        assert_eq!(gate.active(), 0, "no slot was taken");
        assert_eq!(gate.queued(), 0, "and nothing was queued");
    }
}
