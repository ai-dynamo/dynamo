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
//! # Controlled delay and adaptive LIFO
//!
//! Two separate policies act on the queue, each with its own switch. Controlled
//! delay decides *whether* a request that has waited too long is rejected
//! instead of eventually entering the backend; adaptive LIFO decides *which*
//! still-eligible request the next freed slot goes to once controlled delay has
//! rejected one. Adaptive LIFO does nothing until controlled delay has actually
//! rejected a live waiter, but neither switch changes the other's behaviour.
//!
//! ## Controlled delay
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
//! [`DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY`] turns that rejection
//! off, leaving the bounded FIFO by itself: nothing is stamped out of the queue
//! for age, a request may wait longer than the delay and still be admitted in
//! FIFO order, and no timer is armed at all. Such a request is not an expired
//! one — expiry is simply not in force — so it is never counted as a rejection.
//! The queue length bound is the only backpressure left.
//!
//! ## Adaptive LIFO
//!
//! Which request the freed capacity then goes to is the other policy. Rejection
//! is always from the front; selection is not. An admission that rejected
//! nothing takes the oldest request, as it always has; one preceded by a
//! rejection takes the *newest* instead, that being the request with the most of
//! its delay budget left. One round only: the admission after it starts at the
//! front again, however many requests the rejection removed. Only a refusal that
//! reached a request counts — one whose ticket had already gone away is an
//! absent request, not a rejected one. Selecting from the back needs no deadline
//! check of its own, since the due prefix is removed immediately beforehand and
//! the uniform budget makes deadlines nondecreasing along the FIFO, so a live
//! front implies a live back.
//! [`DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO`] turns the back selection
//! off and leaves the delay, the expiry and the front rejection untouched.
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
use crate::metrics::backend_admission::BackendAdmissionMetrics;

/// Overrides the gate's concurrent-request limit. Surfaced to operators as
/// `--engine-request-limit`.
const DYN_ENGINE_REQUEST_LIMIT: &str = "DYN_ENGINE_REQUEST_LIMIT";

/// Overrides the gate's maximum FIFO queue length.
const DYN_DYNAMO_REQUEST_QUEUE_LIMIT: &str = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT";

/// Overrides, in whole milliseconds, how long a request may stay in the FIFO
/// before it is no longer worth admitting. Environment-only: there is no flag.
const DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS: &str = "DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS";

/// Turns queue-delay expiry on or off. Off, a queued request is never rejected
/// for age and the bounded FIFO is the only backpressure left.
/// Environment-only: there is no flag.
const DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY: &str =
    "DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY";

/// Turns the back selection that follows a rejection on or off. It does not
/// enable or disable the queue delay, and has no effect until controlled delay
/// has rejected a live waiter. Environment-only: there is no flag.
const DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO: &str =
    "DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO";

/// Final fallback concurrent-request limit. This is the limit itself, not a
/// `max_num_seqs` stand-in: the 3/2 factor is never applied to it.
const DEFAULT_CONCURRENCY_LIMIT: usize = 10_000;

/// Default FIFO queue length.
const DEFAULT_QUEUE_CAPACITY: usize = 40_000;

/// Default maximum queue residence before a queued request is given up on.
const DEFAULT_QUEUE_DELAY: Duration = Duration::from_millis(5_000);

/// Both queue policies are on unless an operator deliberately turns one off.
const DEFAULT_POLICY_ENABLED: bool = true;

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

/// Resolve one queue-policy switch from an already-read raw value. Pure, so the
/// vocabulary and the default are testable without touching the process
/// environment, and shared by both switches so they cannot drift apart.
///
/// The vocabulary is the canonical Dynamo one, parsed by its single owner so
/// `on` and `yes` cannot mean something here that they do not mean elsewhere.
/// Unset and declared-empty are not choices, so only an unrecognized value is
/// warned about; all three keep [`DEFAULT_POLICY_ENABLED`].
fn resolve_policy_switch(env: &str, raw: Option<&str>) -> bool {
    let Some(raw) = raw.filter(|raw| !raw.trim().is_empty()) else {
        return DEFAULT_POLICY_ENABLED;
    };
    crate::config::parse_bool(raw).unwrap_or_else(|_| {
        tracing::warn!(
            env,
            value = %raw,
            default = DEFAULT_POLICY_ENABLED,
            "Ignoring invalid backend admission queue-policy switch; expected \
             one of: true/false, 1/0, on/off, yes/no"
        );
        DEFAULT_POLICY_ENABLED
    })
}

/// Read one queue-policy switch from the process environment.
fn policy_switch_from_env(env: &str) -> bool {
    resolve_policy_switch(env, std::env::var(env).ok().as_deref())
}

/// The two independent queue policies, resolved once at construction.
///
/// Kept together so the pair is passed as one named value: they are both plain
/// booleans, and a positional pair of those is easy to transpose.
#[derive(Clone, Copy, Debug)]
struct QueuePolicy {
    /// Whether a queued request that outlives its deadline is rejected.
    controlled_delay: bool,
    /// Whether the admission that follows such a rejection takes the newest
    /// queued request rather than the oldest.
    adaptive_lifo: bool,
}

impl QueuePolicy {
    fn from_environment() -> Self {
        Self {
            controlled_delay: policy_switch_from_env(
                DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY,
            ),
            adaptive_lifo: policy_switch_from_env(DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO),
        }
    }
}

#[cfg(test)]
impl Default for QueuePolicy {
    fn default() -> Self {
        Self {
            controlled_delay: DEFAULT_POLICY_ENABLED,
            adaptive_lifo: DEFAULT_POLICY_ENABLED,
        }
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

/// Which end of the FIFO an admission was selected from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DequeueSource {
    Fifo,
    AdaptiveLifo,
}

impl DequeueSource {
    fn is_tail(self) -> bool {
        matches!(self, Self::AdaptiveLifo)
    }
}

/// The outcome the gate hands a queued ticket.
///
/// `Slot` transfers one unit of `active` capacity, so a ticket that has gone
/// away must refund it. `Expired` carries no capacity and its FIFO entry is
/// already removed, so it is never refunded and never unregistered.
///
/// A `Slot` carries the end it was selected from, because the gate cannot know
/// at send time whether the ticket will consume it: a queued request whose
/// context stopped keeps its receiver open, so the send succeeds and the ticket
/// still refuses the slot. The source has to travel with the offer so the ticket
/// can count the dequeue it actually took, and so a refund can put back a tail
/// selection that was never spent.
enum Handoff {
    Slot(DequeueSource),
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

/// Expose the process-global gate's metrics for scraping. Idempotent.
pub(crate) fn register_metrics(registry: &crate::MetricsRegistry) {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| global().metrics.register(registry));
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
    /// The two queue policies, read once at construction.
    policy: QueuePolicy,
    /// Set by a rejection and cleared by the admission that follows it, so
    /// however many there were they buy exactly one admission from the back.
    admit_from_back: bool,
    /// Slots held by admitted requests. May briefly exceed `limit` after a
    /// shrink; a shrink never revokes a permit that is already held.
    active: usize,
    /// Live waiters, oldest first. Its length is exactly the queue occupancy.
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
    /// Bumped under this lock whenever the oldest entry changes, so the one
    /// expiry driver re-reads the head and re-arms its single timer.
    head_generation: watch::Sender<u64>,
    metrics: Arc<BackendAdmissionMetrics>,
}

impl GateState {
    /// Identity of the oldest live entry: which ticket it is and when it is due.
    fn head(&self) -> Option<(u64, Instant)> {
        self.waiters.front().map(|waiter| (waiter.id, waiter.due))
    }

    /// Publish the occupancy gauges from the authoritative counts, rather than
    /// stepping them per transition: direct admission, enqueue, grant, removal,
    /// expiry, release and resizing then cannot drift them.
    fn publish_occupancy(&self) {
        self.metrics.set_occupancy(self.active, self.waiters.len());
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
    /// Every FIFO mutation — and `wake_waiters`, which also moves `active` —
    /// runs through here, so republishing the gauges once at the end covers all
    /// of them.
    fn with_head_watch<R>(&mut self, mutate: impl FnOnce(&mut Self) -> R) -> R {
        let before = self.head();
        let result = mutate(self);
        if self.head() != before {
            self.head_generation
                .send_modify(|generation| *generation = generation.wrapping_add(1));
        }
        self.publish_occupancy();
        result
    }

    /// Remove the due prefix of the FIFO, oldest first, and return it so the
    /// tickets can be told.
    ///
    /// Deadlines are nondecreasing along the FIFO, so the first live entry ends
    /// the scan: this is one `pop_front` per expired request and never looks at
    /// the unexpired tail. Reaching the deadline exactly counts as expired.
    ///
    /// With controlled delay off nothing is ever due: this is the one place
    /// deadlines are acted on, so disabling it here is what leaves the bounded
    /// FIFO alone and lets a long-waiting request still be admitted in order.
    fn drain_expired(&mut self, now: Instant) -> Vec<Waiter> {
        let mut expired = Vec::new();
        if !self.policy.controlled_delay {
            return expired;
        }
        while self.waiters.front().is_some_and(|waiter| waiter.due <= now) {
            expired.push(self.waiters.pop_front().expect("front was just observed"));
        }
        expired
    }

    /// Tell each drained entry it expired, and send the next admission to the
    /// back if any of those refusals reached a request.
    ///
    /// A ticket that has already gone away drops the outcome: that request left
    /// of its own accord, so it is absent rather than rejected — it is neither
    /// counted nor allowed to buy an admission from the back.
    ///
    /// A refusal that does reach a request is committed here, under the state
    /// lock that also arms the tail selection, so the two are one transition: a
    /// scrape can never see the tail admission a rejection bought without also
    /// seeing the rejection that bought it. The request learns its fate later,
    /// and whether it ever reads that outcome no longer changes the count.
    fn reject_expired(&mut self, expired: Vec<Waiter>) {
        let mut rejected = false;
        for waiter in expired {
            if waiter.tx.send(Handoff::Expired).is_ok() {
                rejected = true;
                self.metrics.rejected_request_expired();
            }
        }
        self.admit_from_back |= self.policy.adaptive_lifo && rejected;
    }

    /// Take the next candidate from whichever end the selection points at. The
    /// due prefix is drained immediately before this on every path, so a
    /// surviving front entry is live and the uniform budget puts the back's
    /// deadline no earlier — taking from the back needs no deadline test.
    fn take_next_waiter(&mut self) -> Option<Waiter> {
        if self.admit_from_back {
            self.waiters.pop_back()
        } else {
            self.waiters.pop_front()
        }
    }

    /// Account one request taking a slot, and restart at the front: however many
    /// rejections preceded it, they buy this one admission. A request admitted
    /// directly into an emptied queue spends it too — it is the newest request
    /// the gate has, which is what the back selection asks for.
    fn take_slot(&mut self) {
        self.active += 1;
        self.admit_from_back = false;
    }

    /// Reject every entry due at `now`, freeing its queue capacity immediately,
    /// and report how many left the FIFO.
    ///
    /// Callers already hold the state lock; delivering here is safe because
    /// `oneshot::Sender::send` only stores the outcome and wakes a task.
    fn expire_due(&mut self, now: Instant) -> usize {
        let expired = self.with_head_watch(|state| state.drain_expired(now));
        let count = expired.len();
        self.reject_expired(expired);
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

    /// Hand freed capacity to the waiters the selection points at — the front
    /// ordinarily, the back for the one round a rejection buys — skipping any
    /// that went away between enqueue and wake-up so their slot passes on.
    ///
    /// The due prefix is rejected before every candidate, under this same lock
    /// and against a freshly sampled clock, so a slot is never handed to a
    /// request whose deadline has elapsed — including one that reaches it while
    /// this loop skips past departed waiters ahead of it. The one gate timer
    /// makes expiry prompt; this makes it correct when a slot frees up first, or
    /// before the timer has been scheduled at all.
    ///
    /// Each pass rejects and then selects, so a rejection this loop performs
    /// itself sends that same pass to the back and the pass after it is back at
    /// the front.
    fn wake_waiters(&mut self) {
        self.with_head_watch(|state| {
            loop {
                let expired = state.drain_expired(Instant::now());
                state.reject_expired(expired);
                if state.active >= state.limit {
                    return;
                }
                // Read before `take_slot` clears it, so the offer names the end
                // this waiter was actually selected from.
                let source = if state.admit_from_back {
                    DequeueSource::AdaptiveLifo
                } else {
                    DequeueSource::Fifo
                };
                let Some(waiter) = state.take_next_waiter() else {
                    return;
                };
                // A successful send means the ticket is still there to receive
                // the offer, not that it will take it: a request cancelled while
                // queued keeps its receiver open and may still refuse the slot,
                // refunding it as it settles. The capacity moves here because
                // the ticket now owns it; the dequeue is counted where it is
                // consumed.
                if waiter.tx.send(Handoff::Slot(source)).is_ok() {
                    state.take_slot();
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
        self.metrics.set_engine_request_limit(resolved);
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

        {
            let Some(state) = state.upgrade() else {
                return;
            };
            // Rejecting and arming the back selection under one lock is what
            // keeps the timer's rejection indistinguishable from a handoff's.
            state.lock().expire_due(Instant::now());
        }
        // The next loop pass re-reads the head and re-arms the same timer.
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
    /// Shared with [`GateState`], which publishes the occupancy gauges.
    metrics: Arc<BackendAdmissionMetrics>,
}

impl BackendAdmissionGate {
    fn from_environment() -> Arc<Self> {
        let env_override = positive_env(DYN_ENGINE_REQUEST_LIMIT);
        let queue_capacity =
            positive_env(DYN_DYNAMO_REQUEST_QUEUE_LIMIT).unwrap_or(DEFAULT_QUEUE_CAPACITY);
        let queue_delay = resolve_queue_delay(positive_env(DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS));
        let policy = QueuePolicy::from_environment();
        let gate = Self::new(env_override, queue_capacity, queue_delay, policy);
        tracing::debug!(
            limit = gate.limit(),
            queue_capacity,
            // The gate's own value, so a delay it rejected as unrepresentable is
            // reported as the default it fell back to.
            queue_delay_ms = gate.queue_delay().as_millis(),
            controlled_delay = policy.controlled_delay,
            adaptive_lifo = policy.adaptive_lifo,
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
    fn new(
        env_override: Option<usize>,
        queue_capacity: usize,
        queue_delay: Duration,
        policy: QueuePolicy,
    ) -> Arc<Self> {
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
        // Sizing is published as the metrics are constructed; the concurrency
        // limit is republished by `recompute_limit` when a capacity hint lands.
        // Occupancy starts at zero and every transition republishes it.
        let metrics = Arc::new(BackendAdmissionMetrics::new(limit, queue_capacity));
        Arc::new(Self {
            state: Arc::new(Mutex::new(GateState {
                env_override,
                hint: None,
                limit,
                queue_capacity,
                queue_delay,
                policy,
                admit_from_back: false,
                active: 0,
                waiters: VecDeque::new(),
                next_ticket: 0,
                head_generation,
                metrics: Arc::clone(&metrics),
            })),
            expiry_driver: OnceLock::new(),
            metrics,
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
    ///
    /// With controlled delay off there is nothing for it to do: no entry is ever
    /// due, so a timer armed on a head deadline would fire, reject nothing and
    /// immediately re-arm on the same elapsed instant.
    fn ensure_expiry_driver(&self) {
        let controlled_delay = self.state.lock().policy.controlled_delay;
        if !controlled_delay {
            return;
        }
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
            // Classified as its own path — neither live path was selected — and
            // counted as a cancellation, which a rejection never is.
            self.metrics.received_cancelled();
            self.metrics.cancelled();
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
                state.take_slot();
                // The only `active` change outside `with_head_watch`.
                state.publish_occupancy();
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

        // Counted here rather than under the lock: the decision is what
        // classifies the request, and every request reaches exactly one arm.
        match decision {
            Decision::Granted => {
                self.metrics.received_direct();
                Ok(ActiveSlot {
                    gate: Arc::clone(self),
                })
            }
            // Shed for a full queue is how the queue path ended for this
            // request, not a path of its own.
            Decision::Rejected => {
                self.metrics.received_queue();
                self.metrics.rejected_queue_full();
                Err(reject(Rejection::QueueFull, context))
            }
            Decision::Queued(id, rx) => {
                self.metrics.received_queue();
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

    /// Release one held slot to whichever waiter the selection points at, or
    /// back to the limit when nothing is queued.
    fn release(&self) {
        self.give_back_slot(false);
    }

    /// Return a slot that was offered to a ticket which never consumed it.
    ///
    /// The admission it was selected for never happened, so a tail selection it
    /// spent is put back: the rejection that bought that one round has still not
    /// been paid out, and without this the refunded slot would restart at the
    /// front and the round would be lost. A front selection owes nothing, so it
    /// restores nothing.
    fn refund(&self, source: DequeueSource) {
        self.give_back_slot(source.is_tail());
    }

    fn give_back_slot(&self, restore_tail_selection: bool) {
        let mut state = self.state.lock();
        state.active = state.active.saturating_sub(1);
        state.admit_from_back |= restore_tail_selection;
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
/// releases the slot to the next queued request the selection points at.
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
    /// Set once the gate's outcome has been settled — a slot converted into an
    /// [`ActiveSlot`], an expiry that already removed this entry, or a slot
    /// refused and refunded on the spot — so `Drop` has nothing left to account
    /// for, refund or unregister.
    taken: bool,
}

impl AdmissionTicket {
    /// Resolve a cancellation against whatever the gate has already sent.
    ///
    /// Closing first makes this single-winner: nothing can arrive afterwards.
    /// The two handoffs are then deliberately treated differently.
    ///
    /// An expiry the gate committed to is reported as the expiry it is. Sending
    /// it already removed the entry, counted the rejection and armed the
    /// one-round tail selection, all under one lock; a cancellation noticed
    /// afterwards is later news about the same request and must not reclassify
    /// it or add a second outcome to the counts.
    ///
    /// A slot is the opposite case, and cancellation still wins it: honoring it
    /// would admit a caller that has already gone and hold capacity ahead of
    /// live waiters. Taking it out of the channel is what makes it this
    /// function's to settle — `Drop` could no longer find it — so the
    /// cancellation is counted and the capacity returned here and now, and the
    /// ticket is marked settled so `Drop` does neither again.
    fn resolve_cancelled(&mut self) -> Option<Handoff> {
        let handoff = {
            let rx = self.rx.as_mut()?;
            rx.close();
            rx.try_recv()
        };
        match handoff {
            Ok(Handoff::Expired) => Some(Handoff::Expired),
            Ok(Handoff::Slot(source)) => {
                self.taken = true;
                self.gate.metrics.cancelled();
                self.gate.refund(source);
                None
            }
            Err(_) => None,
        }
    }

    async fn wait(
        mut self,
        context: Option<&dyn AsyncEngineContext>,
    ) -> Result<ActiveSlot, DynamoError> {
        // Cancellation while queued must be prompt, and it must win a
        // simultaneous *slot*: one can be sent to this waiter after the context
        // stopped but before this future is polled again, and admitting then
        // would run a request the caller already abandoned. The precheck plus
        // the biased ordering give the stop strictly higher priority; what the
        // gate had already committed to is then settled in `resolve_cancelled`.
        let outcome = match context {
            Some(context) if context.is_stopped() => self.resolve_cancelled(),
            Some(context) => {
                let handoff = {
                    let rx = self.rx.as_mut().expect("ticket always holds its receiver");
                    tokio::select! {
                        biased;
                        _ = context.stopped() => None,
                        handoff = &mut *rx => Some(handoff.ok()),
                    }
                };
                match handoff {
                    Some(handoff) => handoff,
                    None => self.resolve_cancelled(),
                }
            }
            None => {
                let rx = self.rx.as_mut().expect("ticket always holds its receiver");
                rx.await.ok()
            }
        };

        match outcome {
            Some(Handoff::Slot(source)) => {
                // The one place a queued request actually becomes an admission,
                // so the one place a dequeue is counted: only an offer that is
                // consumed counts. An offer this ticket refuses never reaches
                // here — it is settled by the cancellation path, or by `Drop`
                // when the ticket went away without resolving anything.
                self.taken = true;
                self.gate.metrics.dequeued(source.is_tail());
                Ok(ActiveSlot {
                    gate: Arc::clone(&self.gate),
                })
            }
            Some(Handoff::Expired) => {
                // The gate popped this entry before sending, so there is no
                // queue slot to release and no capacity to refund — and the
                // rejection was already counted where the gate committed to it,
                // so reporting it here is metric-neutral.
                self.taken = true;
                Err(reject(Rejection::Expired, context))
            }
            // A cancellation is the caller leaving: never a rejection, and it
            // keeps the queue path it was classified under. It is counted
            // exactly once either way. Where a slot had to be refused,
            // `resolve_cancelled` already counted and refunded it and set
            // `taken`, so the `Drop` this return runs immediately is a no-op;
            // otherwise `taken` is still unset and that `Drop` is what counts
            // the cancellation and unregisters the waiter — the same path a
            // ticket dropped without ever resolving anything takes.
            None => Err(reject(Rejection::Cancelled, context)),
        }
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        // `taken` is set only where `wait` already settled an outcome, so
        // reaching here with it unset means this request left the queue without
        // becoming an admission — because `wait` resolved a cancellation, or
        // because the whole future was dropped before it could. Everything left
        // to account for is settled from here, exactly once.
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
            // The gate committed to this rejection, and counted it, when it sent
            // the outcome. The request going away before reading it changes
            // neither: it is that rejection, not a cancellation, and the entry
            // is already out of the FIFO with no slot ever allocated, so
            // `active` must not move and nothing is unregistered.
            Ok(Handoff::Expired) => {}
            // An offer this request never consumed. The slot goes back and, if
            // it was selected from the tail, so does the one-round tail
            // selection: this candidate is absent, not admitted, so the
            // rejection that armed it is still owed an admission from the back.
            Ok(Handoff::Slot(source)) => {
                self.gate.metrics.cancelled();
                self.gate.refund(source);
            }
            Err(_) => {
                self.gate.metrics.cancelled();
                self.gate.remove_waiter(self.id);
            }
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
    /// expiry, so those tests see the pre-Controlled-Delay behaviour exactly:
    /// nothing is ever due, so nothing arms the back selection either.
    fn gate(limit: usize, queue: usize) -> Arc<BackendAdmissionGate> {
        gate_with_delay(limit, queue, Duration::from_secs(3_600))
    }

    fn gate_with_delay(limit: usize, queue: usize, delay: Duration) -> Arc<BackendAdmissionGate> {
        gate_with_policy(limit, queue, delay, QueuePolicy::default())
    }

    fn gate_with_policy(
        limit: usize,
        queue: usize,
        delay: Duration,
        policy: QueuePolicy,
    ) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(Some(limit), queue, delay, policy)
    }

    /// Adaptive LIFO off, controlled delay untouched.
    fn without_adaptive_lifo() -> QueuePolicy {
        QueuePolicy {
            adaptive_lifo: false,
            ..QueuePolicy::default()
        }
    }

    /// Controlled delay off, adaptive LIFO untouched.
    fn without_controlled_delay() -> QueuePolicy {
        QueuePolicy {
            controlled_delay: false,
            ..QueuePolicy::default()
        }
    }

    /// A gate with no environment override, so its limit comes from a hint or
    /// the fallback.
    fn hint_sized_gate(queue: usize) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(
            None,
            queue,
            Duration::from_secs(3_600),
            QueuePolicy::default(),
        )
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
        assert_eq!(refusals(&gate), (0, 0));
        assert_eq!(cancellations(&gate), 1);
        assert_eq!(published(&gate), (1, 0, 1, 4));
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
                gate_with_policy(1, 1, delay, QueuePolicy::default()).queue_delay(),
                expected,
                "the resolved delay must reach the gate"
            );
        }
    }

    /// Both switches speak the canonical Dynamo boolean vocabulary, and every
    /// way of not making a choice — unset, declared empty, or a spelling outside
    /// it — keeps the enabled default.
    #[test]
    fn both_queue_policy_switches_default_on_and_read_the_canonical_vocabulary() {
        const { assert!(DEFAULT_POLICY_ENABLED) };
        for env in [
            DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY,
            DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO,
        ] {
            for (raw, expected) in [
                (None, true),
                (Some(""), true),
                (Some("   "), true),
                (Some("enabled"), true),
                (Some("1"), true),
                (Some("TRUE"), true),
                (Some("On"), true),
                (Some(" yes "), true),
                (Some("0"), false),
                (Some("false"), false),
                (Some("OFF"), false),
                (Some("no"), false),
            ] {
                assert_eq!(resolve_policy_switch(env, raw), expected, "{env} {raw:?}");
            }
        }
        // Two names, not one renamed: an operator can set either independently.
        assert_ne!(
            DYN_DYNAMO_REQUEST_QUEUE_ENABLE_CONTROLLED_DELAY,
            DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO
        );
        assert!(DYN_DYNAMO_REQUEST_QUEUE_ENABLE_ADAPTIVE_LIFO.ends_with("ADAPTIVE_LIFO"));
    }

    /// The two switches are independent: each reaches its own field, and
    /// neither disturbs the other.
    #[test]
    fn the_two_policies_are_configured_independently() {
        for (controlled_delay, adaptive_lifo) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let policy = QueuePolicy {
                controlled_delay,
                adaptive_lifo,
            };
            let gate = gate_with_policy(1, 1, Duration::from_millis(50), policy);
            let state = gate.state.lock();
            assert_eq!(state.policy.controlled_delay, controlled_delay);
            assert_eq!(state.policy.adaptive_lifo, adaptive_lifo);
        }
    }

    #[test]
    fn a_delay_too_large_to_represent_falls_back_to_the_default() {
        // Enqueue stamps `now + delay`, so an unbounded override must degrade
        // rather than panic there.
        assert_eq!(
            gate_with_policy(1, 1, Duration::MAX, QueuePolicy::default()).queue_delay(),
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

        let gate =
            BackendAdmissionGate::new(None, 8, Duration::from_millis(50), QueuePolicy::default());
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
            matches!(live.try_recv(), Ok(Handoff::Slot(_))),
            "the handoff must advance to the oldest unexpired request"
        );
        assert_eq!(gate.active(), 1, "exactly one slot was handed over");

        // Growing the limit: ceil(3/2 * 4 * 1) = 6, above the slot in use.
        let (mut overdue, mut live) = due_then_live(&gate);
        gate.record_capacity_report(Some(4), Some(1));
        assert!(matches!(overdue.try_recv(), Ok(Handoff::Expired)));
        assert!(matches!(live.try_recv(), Ok(Handoff::Slot(_))));
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
        assert!(matches!(live.try_recv(), Ok(Handoff::Slot(_))));
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

    /// A rejection the gate has already committed to is reported as the
    /// rejection it is, even though the caller stopped before collecting it.
    ///
    /// Cancellation wins a simultaneous *slot*, because honoring one would admit
    /// an abandoned request; it does not win a simultaneous *expiry*, because
    /// sending that expiry already removed the entry, counted the rejection and
    /// armed the one-round tail selection. A cancellation observed afterwards is
    /// later news about the same request: reclassifying it would contradict
    /// counts the gate had already published.
    #[tokio::test]
    async fn a_committed_expiry_outranks_a_later_observed_cancellation() {
        use crate::pipeline::context::Controller;

        let gate = gate_with_delay(1, 4, Duration::from_millis(50));
        let held = permit(gate.acquire(None).await);

        // Poll once so the ticket registers, then stop being polled.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.acquire(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());
        assert_eq!(gate.queued(), 1);

        // A live request behind it, to show where the tail round the rejection
        // armed actually goes.
        let mut behind_it = live(&gate);
        let mut newest = live(&gate);

        // Make both outcomes ready before the ticket is polled again: the
        // context stops, and only then does the gate expire the entry.
        controller.stop_generating();
        assert_eq!(
            gate.expire_due_for_test(Instant::now() + Duration::from_secs(1)),
            1,
            "expiry removed the entry from the FIFO"
        );

        let admission = doomed.await;
        assert!(
            is_expired(&admission),
            "the rejection the gate committed to is what the request is told"
        );
        drop(admission);
        assert_eq!(
            refusals(&gate),
            (0, 1),
            "and it is counted, so the armed tail round has a rejection behind it"
        );
        assert_eq!(
            cancellations(&gate),
            0,
            "an expiry is not also a cancellation"
        );
        assert_eq!(
            gate.active(),
            1,
            "an expiry carries no slot, so there is nothing to refund"
        );

        // The tail round that rejection armed is spent on the newest request.
        drop(held);
        assert_eq!(
            granted_from(&mut newest),
            Some(DequeueSource::AdaptiveLifo),
            "the rejection sends the freed slot to the back"
        );
        assert!(still_queued(&mut behind_it), "the front keeps its place");
        assert_eq!(gate.active(), 1);
    }

    ///////////////////// CONTROLLED DELAY: SELECTION /////////////////////

    /// A queue entry with an exact deadline, pushed directly so no expiry driver
    /// is involved and the FIFO can be built one entry at a time. `due` is
    /// already past its deadline; `live` is nowhere near its own.
    fn due(gate: &Arc<BackendAdmissionGate>) -> oneshot::Receiver<Handoff> {
        gate.push_waiter_for_test(Instant::now() - Duration::from_millis(1))
    }

    fn live(gate: &Arc<BackendAdmissionGate>) -> oneshot::Receiver<Handoff> {
        gate.push_waiter_for_test(Instant::now() + Duration::from_secs(60))
    }

    /// What the gate has told a queued entry so far. A raw entry has no ticket
    /// to consume the offer, so these read the handoff itself — which is also
    /// where the selected end is now carried.
    fn granted_from(rx: &mut oneshot::Receiver<Handoff>) -> Option<DequeueSource> {
        match rx.try_recv() {
            Ok(Handoff::Slot(source)) => Some(source),
            _ => None,
        }
    }

    fn granted(rx: &mut oneshot::Receiver<Handoff>) -> bool {
        granted_from(rx).is_some()
    }

    fn rejected(rx: &mut oneshot::Receiver<Handoff>) -> bool {
        matches!(rx.try_recv(), Ok(Handoff::Expired))
    }

    fn still_queued(rx: &mut oneshot::Receiver<Handoff>) -> bool {
        matches!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty))
    }

    /// Rejections buy one admission from the back and no more: two of them do
    /// not owe two, a departed newest request is passed over rather than ending
    /// the search, and the admission after it is oldest first again. Ordinary
    /// oldest-first service with nothing rejected is covered by the FIFO test
    /// above.
    #[tokio::test]
    async fn rejections_buy_exactly_one_admission_from_the_back() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(50));
        let held = permit(gate.acquire(None).await);
        let (mut stale, mut also_stale) = (due(&gate), due(&gate));
        let (mut oldest, mut middle, mut newest) = (live(&gate), live(&gate), live(&gate));
        // Behind them all, and gone before the handoff reaches it.
        drop(live(&gate));

        // Releasing the permit rejects the due prefix and selects, in one pass.
        drop(held);
        assert!(rejected(&mut stale) && rejected(&mut also_stale));
        assert!(
            granted(&mut newest),
            "a rejection sends that admission to the newest eligible request"
        );
        assert!(
            still_queued(&mut oldest),
            "the front keeps its place; it is passed over, not rejected"
        );

        // Rejects nothing, so it must not inherit a LIFO order or a second back
        // admission from the two rejections above.
        gate.set_limit_for_test(2);
        assert!(
            granted(&mut oldest),
            "the admission after the back one restarts at the front"
        );
        assert!(still_queued(&mut middle));
        assert_eq!((gate.active(), gate.queued()), (2, 1));
    }

    /// A queued request that went away is absent, not rejected: draining it must
    /// not buy an admission from the back.
    #[tokio::test]
    async fn a_departed_entry_leaving_on_its_deadline_is_not_a_rejection() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(50));
        let held = permit(gate.acquire(None).await);
        drop(due(&gate));
        let (mut oldest, mut newest) = (live(&gate), live(&gate));

        drop(held);
        assert!(
            granted(&mut oldest),
            "an expiry no request received must leave the admission at the front"
        );
        assert!(still_queued(&mut newest));
        assert_eq!(
            refusals(&gate),
            (0, 0),
            "and a refusal that reached nobody is not counted either: the send \
             that fails is the same event that declines to arm the tail"
        );
    }

    /// Adaptive LIFO off governs selection alone: the same request is still
    /// rejected by controlled delay, and the freed slot goes to the front.
    #[tokio::test]
    async fn adaptive_lifo_off_rejects_from_the_front_and_admits_from_the_front() {
        let gate = gate_with_policy(1, 8, Duration::from_millis(50), without_adaptive_lifo());
        let held = permit(gate.acquire(None).await);
        let (mut stale, mut oldest, mut newest) = (due(&gate), live(&gate), live(&gate));

        drop(held);
        assert!(rejected(&mut stale), "the same request is still rejected");
        assert_eq!(
            granted_from(&mut oldest),
            Some(DequeueSource::Fifo),
            "with the back selection off the freed slot goes to the front"
        );
        assert!(still_queued(&mut newest));
        assert_eq!((gate.active(), gate.queued()), (1, 1));
    }

    ///////////////////// CONTROLLED DELAY: DISABLED /////////////////////

    /// Controlled delay off leaves the bounded FIFO by itself: a request that
    /// has waited well past the delay is still admitted, in FIFO order, and is
    /// never counted as expired.
    #[tokio::test(start_paused = true)]
    async fn controlled_delay_off_admits_a_request_that_outwaited_the_delay() {
        let delay = Duration::from_millis(200);
        let gate = gate_with_policy(1, 8, delay, without_controlled_delay());
        let held = permit(gate.acquire(None).await);

        let first = spawn_queued_admit(&gate).await;
        let second = spawn_queued_admit(&gate).await;

        // Far past the deadline both entries would have carried. Nothing may
        // remove them: with expiry off there is no timer to do it, and the
        // grant-path check must find nothing due either.
        tokio::time::sleep(delay * 10).await;
        assert_eq!(gate.queued(), 2, "no entry may leave the queue for age");

        drop(held);
        // Held, not dropped in place: releasing it here would pass the slot
        // straight on and make the counts below the pair's, not the first's.
        let admitted = first.await.expect("waiter completes");
        assert!(
            admitted.is_ok(),
            "a long-waiting request is admitted rather than rejected"
        );
        assert_eq!(
            refusals(&gate),
            (0, 0),
            "nothing expired, so nothing counted"
        );
        assert_eq!(dequeues(&gate), (1, 0), "and it came from the FIFO front");
        assert_eq!(gate.queued(), 1, "the one behind it kept its place");

        drop(admitted);
        second.abort();
        let _ = second.await;
    }

    /// With expiry off, an entry that is already past its deadline is neither
    /// rejected nor skipped on any grant-producing path.
    #[tokio::test]
    async fn controlled_delay_off_never_rejects_an_overdue_head() {
        let gate = gate_with_policy(1, 8, Duration::from_millis(50), without_controlled_delay());
        let held = permit(gate.acquire(None).await);
        let (mut overdue, mut behind) = (due(&gate), live(&gate));

        drop(held);
        assert_eq!(
            granted_from(&mut overdue),
            Some(DequeueSource::Fifo),
            "an overdue head keeps its place at the front and is admitted"
        );
        assert!(still_queued(&mut behind));
        assert_eq!(refusals(&gate), (0, 0));
    }

    /// With expiry off there is nothing for a timer to do, so none is started:
    /// arming one on a deadline that is never acted on would fire, reject
    /// nothing and immediately re-arm on the same elapsed instant.
    #[tokio::test]
    async fn controlled_delay_off_starts_no_expiry_driver() {
        let gate = gate_with_policy(1, 1, Duration::from_millis(50), without_controlled_delay());
        let _held = permit(gate.acquire(None).await);
        let waiting = spawn_queued_admit(&gate).await;

        assert!(
            gate.expiry_driver.get().is_none(),
            "no expiry driver may be started when nothing can expire"
        );

        waiting.abort();
        let _ = waiting.await;
    }

    /// The gate's own timer arms the same one-round selection a handoff does.
    /// Nothing frees a slot until this test does, so the timer is the only thing
    /// that can resolve the first waiter.
    #[tokio::test(start_paused = true)]
    async fn a_timer_rejection_sends_the_next_admission_to_the_back() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(400));
        let held = permit(gate.acquire(None).await);

        let doomed = spawn_queued_admit(&gate).await;
        // Far enough behind the doomed request that its rejection lands while
        // these two are still comfortably live.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let oldest = spawn_queued_admit(&gate).await;
        let newest = spawn_queued_admit(&gate).await;

        assert!(is_expired(&doomed.await.expect("waiter completes")));
        assert_eq!(gate.queued(), 2, "only the due request left the queue");

        drop(held);
        let admitted = tokio::time::timeout(Duration::from_millis(50), newest)
            .await
            .expect("the newest queued request must inherit the freed slot")
            .expect("waiter completes");
        assert!(
            !oldest.is_finished(),
            "the older request keeps its place rather than being served or rejected"
        );
        drop(permit(admitted));
        oldest.abort();
        let _ = oldest.await;
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

    ///////////////////// METRICS /////////////////////

    /// Published as (engine requests, queued, engine limit, queue limit).
    fn published(gate: &Arc<BackendAdmissionGate>) -> (i64, i64, i64, i64) {
        gate.metrics.published()
    }

    /// Refusals as (queue_full, request_expired).
    fn refusals(gate: &Arc<BackendAdmissionGate>) -> (u64, u64) {
        gate.metrics.refusals()
    }

    /// Requests received as (direct, queue, cancelled).
    fn received(gate: &Arc<BackendAdmissionGate>) -> (u64, u64, u64) {
        gate.metrics.received_paths()
    }

    /// Queued requests admitted as (fifo, adaptive_lifo).
    fn dequeues(gate: &Arc<BackendAdmissionGate>) -> (u64, u64) {
        gate.metrics.dequeues()
    }

    fn cancellations(gate: &Arc<BackendAdmissionGate>) -> u64 {
        gate.metrics.cancellations()
    }

    /// Gauges come from the gate's own counts, so they follow every transition,
    /// and the limit follows a late hint.
    #[tokio::test]
    async fn the_gauges_follow_occupancy_and_sizing() {
        let gate = gate(1, 4);
        assert_eq!(published(&gate), (0, 0, 1, 4));
        let held = permit(gate.acquire(None).await);
        assert_eq!(published(&gate), (1, 0, 1, 4), "direct admission");
        let waiter = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1, 4), "enqueue");
        drop(held);
        let granted = permit(waiter.await.expect("waiter completes"));
        assert_eq!(published(&gate), (1, 0, 1, 4), "grant");
        drop(granted);
        assert_eq!(published(&gate), (0, 0, 1, 4), "release");

        let hinted = hint_sized_gate(4);
        hinted.record_capacity_report(Some(2), Some(1));
        assert_eq!(published(&hinted), (0, 0, 3, 4), "late hint");
    }

    /// Every request the gate receives lands on exactly one path, and a shed
    /// request is the queue path ending badly rather than a path of its own.
    #[tokio::test]
    async fn every_request_is_classified_exactly_once() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 1);
        assert_eq!(received(&gate), (0, 0, 0));

        let held = permit(gate.acquire(None).await);
        assert_eq!(received(&gate), (1, 0, 0), "a free slot is the direct path");

        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(received(&gate), (1, 1, 0), "no free slot is the queue path");

        assert!(is_queue_full(&gate.acquire(None).await));
        assert_eq!(
            received(&gate),
            (1, 2, 0),
            "a shed request took the queue path too"
        );
        assert_eq!(refusals(&gate), (1, 0));

        let controller = Arc::new(Controller::default());
        controller.stop_generating();
        let context: Arc<dyn AsyncEngineContext> = controller;
        assert!(is_cancelled(&gate.acquire(Some(context.as_ref())).await));
        assert_eq!(
            received(&gate),
            (1, 2, 1),
            "already cancelled is neither live path"
        );
        assert_eq!(cancellations(&gate), 1);
        assert_eq!(refusals(&gate), (1, 0), "which is not a refusal");

        drop(held);
        drop(permit(queued.await.expect("waiter completes")));
    }

    /// Only the two refusal reasons count, and an expiry counts once, where the
    /// gate commits to it.
    #[tokio::test]
    async fn rejections_are_counted_by_reason_only() {
        let expiring = gate_with_delay(1, 4, Duration::from_millis(1));
        let _busy = permit(expiring.acquire(None).await);
        let waiter = spawn_queued_admit(&expiring).await;
        assert!(is_expired(&waiter.await.expect("waiter completes")));
        assert_eq!(refusals(&expiring), (0, 1));
        assert_eq!(published(&expiring), (1, 0, 1, 4));
        assert_eq!(dequeues(&expiring), (0, 0), "an expiry is not a dequeue");
        assert_eq!(cancellations(&expiring), 0, "nor a cancellation");
    }

    /// A queued request aborted after the gate expired it, but before it read
    /// the outcome. The expiry is the gate's committed rejection either side of
    /// that race: it is counted as one, it is not a cancellation, and the tail
    /// round it armed is still spent on the newest queued request.
    #[tokio::test]
    async fn an_expiry_the_request_never_collected_is_still_the_committed_rejection() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(50));
        let held = permit(gate.acquire(None).await);

        // A real queued admission, polled once so its ticket registers and then
        // never polled again — the shape a task abort leaves behind.
        let mut abandoned = Box::pin(gate.acquire(None));
        assert!(futures::poll!(&mut abandoned).is_pending());
        let mut front = live(&gate);
        let mut newest = live(&gate);
        assert_eq!(gate.queued(), 3);

        // Only the abandoned entry is due; the two behind it are nowhere near.
        assert_eq!(
            gate.expire_due_for_test(Instant::now() + Duration::from_secs(1)),
            1,
            "exactly the abandoned entry expired"
        );
        assert_eq!(
            refusals(&gate),
            (0, 1),
            "the rejection is counted at the moment the gate commits to it, not \
             when some request gets around to reading it"
        );
        drop(abandoned);
        assert_eq!(
            refusals(&gate),
            (0, 1),
            "abandoning it neither double-counts nor withdraws the rejection"
        );
        assert_eq!(cancellations(&gate), 0, "and it is not a cancellation");

        drop(held);
        assert_eq!(
            granted_from(&mut newest),
            Some(DequeueSource::AdaptiveLifo),
            "the tail round that rejection armed is still honored"
        );
        assert!(still_queued(&mut front), "the front keeps its place");
    }

    /// A queued request that leaves without being admitted is a cancellation
    /// exactly once, whether its own future observed the stop or the whole task
    /// was dropped first. Neither shape is a rejection or a dequeue.
    #[tokio::test]
    async fn a_queued_cancellation_counts_once_whether_observed_or_dropped() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 4);
        let _held = permit(gate.acquire(None).await);

        // Observed: the waiting request is polled again and sees the stop.
        let controller = Arc::new(Controller::default());
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            let context: Arc<dyn AsyncEngineContext> = controller.clone();
            async move { gate.acquire(Some(context.as_ref())).await }
        });
        while gate.queued() == 0 {
            tokio::task::yield_now().await;
        }
        controller.stop_generating();
        assert!(is_cancelled(&waiting.await.expect("waiter completes")));
        assert_eq!(cancellations(&gate), 1, "counted where it was observed");

        // Dropped: the task is aborted, so nothing ever reaches the arm above
        // and only the ticket's own `Drop` can account for it.
        let abandoned = spawn_queued_admit(&gate).await;
        abandoned.abort();
        let _ = abandoned.await;
        while gate.queued() != 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            cancellations(&gate),
            2,
            "an abandoned ticket is a cancellation too"
        );

        assert_eq!(
            refusals(&gate),
            (0, 0),
            "a cancellation is never a rejection"
        );
        assert_eq!(dequeues(&gate), (0, 0), "and never a dequeue");
        assert_eq!(
            received(&gate),
            (1, 2, 0),
            "both kept the queue path they arrived on"
        );
    }

    /// A dequeue is counted where a real request consumes the offer, and named
    /// by the end it was selected from. Driven through `acquire` rather than raw
    /// FIFO entries, because it is the ticket, not the send, that takes the slot.
    #[tokio::test(start_paused = true)]
    async fn dequeues_are_counted_by_the_end_the_request_came_from() {
        let gate = gate_with_delay(1, 8, Duration::from_millis(400));
        let held = permit(gate.acquire(None).await);

        let doomed = spawn_queued_admit(&gate).await;
        // Far enough behind that these two are still live when it is rejected.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let oldest = spawn_queued_admit(&gate).await;
        let newest = spawn_queued_admit(&gate).await;

        assert!(is_expired(&doomed.await.expect("waiter completes")));
        assert_eq!(dequeues(&gate), (0, 0), "a rejection is not a dequeue");

        // That rejection sends this one admission to the tail.
        drop(held);
        let admitted = tokio::time::timeout(Duration::from_millis(50), newest)
            .await
            .expect("the newest queued request inherits the freed slot")
            .expect("waiter completes");
        assert_eq!(
            dequeues(&gate),
            (0, 1),
            "a tail admission counts as its own"
        );

        // Releasing it rejects nothing, so the next starts at the front again.
        drop(permit(admitted));
        let admitted = tokio::time::timeout(Duration::from_millis(50), oldest)
            .await
            .expect("the front is served next")
            .expect("waiter completes");
        assert_eq!(dequeues(&gate), (1, 1));

        drop(permit(admitted));
        assert_eq!(refusals(&gate), (0, 1));
        assert_eq!(cancellations(&gate), 0);
        assert_eq!(received(&gate), (1, 3, 0));
    }

    /// A tail candidate whose caller has already gone is absent, not admitted.
    /// It must not be counted as a dequeue, and the one-round tail selection the
    /// rejection bought must survive its refusal — otherwise a cancellation
    /// silently converts that round back into a front admission.
    #[tokio::test(start_paused = true)]
    async fn a_cancelled_tail_candidate_is_absent_and_keeps_the_tail_selection_owed() {
        use crate::pipeline::context::Controller;

        let gate = gate_with_delay(1, 8, Duration::from_millis(400));
        let held = permit(gate.acquire(None).await);

        // Front to back: an entry that will come due, two live entries, and a
        // cancellable newest that the tail selection will pick. The first is
        // still live while the queue is built, because `acquire` drains the due
        // prefix and would otherwise reject it before the rest arrived.
        let mut stale = gate.push_waiter_for_test(Instant::now() + Duration::from_millis(100));
        let mut front = live(&gate);
        let mut middle = live(&gate);
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.acquire(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());
        assert_eq!(gate.queued(), 4);

        // Stopped before the handoff reaches it, but its receiver stays open, so
        // the gate's send succeeds and only the ticket can refuse the slot.
        tokio::time::sleep(Duration::from_millis(150)).await;
        controller.stop_generating();
        drop(held);
        assert!(rejected(&mut stale), "the due entry is rejected as ever");
        assert_eq!(gate.active(), 1, "the slot was handed to the newest entry");

        let admission = doomed.await;
        assert!(is_cancelled(&admission), "cancellation wins the handoff");
        drop(admission);

        assert_eq!(
            dequeues(&gate),
            (0, 0),
            "a candidate that refused the offer never dequeued"
        );
        assert_eq!(
            granted_from(&mut middle),
            Some(DequeueSource::AdaptiveLifo),
            "the refunded slot still owes the tail admission the rejection bought"
        );
        assert!(
            still_queued(&mut front),
            "so the front is passed over exactly as it would have been"
        );
        assert_eq!(gate.active(), 1, "the refund handed the slot straight on");
        assert_eq!(
            cancellations(&gate),
            1,
            "the tail candidate that went away is the one cancellation here"
        );
        assert_eq!(
            refusals(&gate),
            (0, 1),
            "and the only rejection is the due entry that armed the round, not \
             the candidate that refused the slot it bought"
        );
    }

    /// Collectors are per gate: one gate's counts are invisible to another. The
    /// family a gate registers is covered in
    /// [`crate::metrics::backend_admission`].
    #[tokio::test]
    async fn each_gate_owns_its_collectors() {
        let one = gate(1, 0);
        let two = gate(1, 0);
        let _held = permit(one.acquire(None).await);
        assert!(is_queue_full(&one.acquire(None).await));
        assert_eq!((refusals(&one), refusals(&two)), ((1, 0), (0, 0)));
        assert_eq!(published(&two), (0, 0, 1, 0));
    }
}
