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
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::engine::AsyncEngineContext;

/// Overrides the gate's concurrent-request limit. Surfaced to operators as
/// `--engine-request-limit`.
pub const DYN_ENGINE_REQUEST_LIMIT: &str = "DYN_ENGINE_REQUEST_LIMIT";

/// Overrides the gate's maximum FIFO queue length.
pub const DYN_DYNAMO_REQUEST_QUEUE_LIMIT: &str = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT";

/// Final fallback concurrent-request limit. This is the limit itself, not a
/// `max_num_seqs` stand-in: the 3/2 factor is never applied to it.
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 10_000;

/// Default FIFO queue length.
pub const DEFAULT_QUEUE_CAPACITY: usize = 40_000;

/// Prefix the egress side matches to classify a response-stream prologue error
/// as `ErrorType::WorkerOverloaded`.
pub const OVERLOADED_PREFIX: &str = "Server overloaded:";

/// The prologue error a shed request receives.
pub const OVERLOADED_MESSAGE: &str = "Server overloaded: worker at capacity";

/// The process-global gate. Constructed on first touch; its limit stays
/// adjustable so a late capacity hint is not lost.
static GATE: LazyLock<Arc<BackendAdmissionGate>> =
    LazyLock::new(BackendAdmissionGate::from_environment);

/// The one gate every request in this process admits through.
pub fn global() -> &'static Arc<BackendAdmissionGate> {
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
pub fn automatic_concurrency_limit(
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
pub fn resolve_concurrency_limit(env_override: Option<usize>, hint: Option<usize>) -> usize {
    if let Some(limit) = env_override.filter(|limit| *limit > 0) {
        return limit;
    }
    if let Some(limit) = hint.filter(|limit| *limit > 0) {
        return limit;
    }
    DEFAULT_CONCURRENCY_LIMIT
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

/// A queued request, oldest first. The sender is the slot handoff.
struct Waiter {
    id: u64,
    tx: oneshot::Sender<()>,
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
    /// Slots held by admitted requests. May briefly exceed `limit` after a
    /// shrink; a shrink never revokes a permit that is already held.
    active: usize,
    /// Live waiters, oldest first. Its length is exactly the queue occupancy.
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
}

impl GateState {
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
    fn wake_waiters(&mut self) {
        while self.active < self.limit {
            let mut woke = false;
            while let Some(waiter) = self.waiters.pop_front() {
                if waiter.tx.send(()).is_ok() {
                    self.active += 1;
                    woke = true;
                    break;
                }
                // The waiter is gone; popping it already reclaimed its queue
                // slot, so continue to the next one.
            }
            if !woke {
                break;
            }
        }
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

/// One concurrency limit plus one bounded FIFO queue, shared by every endpoint
/// in the process.
pub struct BackendAdmissionGate {
    state: Mutex<GateState>,
}

impl BackendAdmissionGate {
    fn from_environment() -> Arc<Self> {
        let env_override = positive_env(DYN_ENGINE_REQUEST_LIMIT);
        let queue_capacity =
            positive_env(DYN_DYNAMO_REQUEST_QUEUE_LIMIT).unwrap_or(DEFAULT_QUEUE_CAPACITY);
        let gate = Self::new(env_override, queue_capacity);
        tracing::debug!(
            limit = gate.limit(),
            queue_capacity,
            has_env_override = env_override.is_some(),
            "Backend admission gate created"
        );
        gate
    }

    /// Build a standalone gate. Production uses [`global`]; tests build their
    /// own so they never contend for process-global capacity.
    pub fn new(env_override: Option<usize>, queue_capacity: usize) -> Arc<Self> {
        let limit = resolve_concurrency_limit(env_override, None);
        Arc::new(Self {
            state: Mutex::new(GateState {
                env_override,
                hint: None,
                limit,
                queue_capacity,
                active: 0,
                waiters: VecDeque::new(),
                next_ticket: 0,
            }),
        })
    }

    /// Effective concurrent-request limit.
    pub fn limit(&self) -> usize {
        self.state.lock().limit
    }

    /// Configured FIFO queue length.
    pub fn queue_capacity(&self) -> usize {
        self.state.lock().queue_capacity
    }

    /// Requests holding a slot.
    pub fn active(&self) -> usize {
        self.state.lock().active
    }

    /// Requests waiting in the FIFO.
    pub fn queued(&self) -> usize {
        self.state.lock().waiters.len()
    }

    /// Record a published capacity. Only the first usable report becomes the
    /// process-global hint, regardless of which component published it; a later
    /// conflicting one warns and is ignored. The environment override always
    /// takes precedence over any report.
    pub fn record_capacity_report(
        &self,
        max_num_seqs: Option<u64>,
        data_parallel_size: Option<u32>,
    ) {
        let Some(limit) = automatic_concurrency_limit(max_num_seqs, data_parallel_size) else {
            return;
        };
        self.state.lock().adopt_hint(limit);
    }

    /// Admit one request, waiting in FIFO order when the limit is busy.
    ///
    /// The lock is never held across an await: the decision is made under the
    /// lock, and only the handoff is awaited.
    pub async fn admit(self: &Arc<Self>, context: Option<&dyn AsyncEngineContext>) -> Admission {
        enum Decision {
            Granted,
            Queued(u64, oneshot::Receiver<()>),
            Rejected,
        }

        let decision = {
            let mut state = self.state.lock();
            // Direct admission only when nothing is waiting, so a new request
            // can never bypass an older one.
            if state.active < state.limit && state.waiters.is_empty() {
                state.active += 1;
                Decision::Granted
            } else if state.waiters.len() < state.queue_capacity {
                let id = state.next_ticket;
                state.next_ticket = state.next_ticket.wrapping_add(1);
                let (tx, rx) = oneshot::channel();
                state.waiters.push_back(Waiter { id, tx });
                Decision::Queued(id, rx)
            } else {
                Decision::Rejected
            }
        };

        match decision {
            Decision::Granted => Admission::Granted(AdmissionPermit {
                gate: Arc::clone(self),
            }),
            Decision::Rejected => Admission::Overloaded,
            Decision::Queued(id, rx) => {
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
    fn remove_waiter(&self, id: u64) {
        let mut state = self.state.lock();
        if let Some(index) = state.waiters.iter().position(|waiter| waiter.id == id) {
            state.waiters.remove(index);
        }
    }

    /// Seed a starting limit that a later hint can still resize. Production
    /// sizing always comes from the environment or a hint.
    #[cfg(test)]
    fn set_limit_for_test(&self, limit: usize) {
        let mut state = self.state.lock();
        state.limit = limit;
        state.wake_waiters();
    }
}

/// The result of asking the gate for a slot.
pub enum Admission {
    /// A slot is held for as long as the returned permit lives.
    Granted(AdmissionPermit),
    /// The limit and the queue are both full.
    Overloaded,
    /// The request was cancelled while queued.
    Cancelled,
}

/// A held concurrency slot. Dropping it — on normal end-of-stream, a generate
/// error, an encode or publish error, task abort, or cancellation — releases the
/// slot to the oldest queued request.
pub struct AdmissionPermit {
    gate: Arc<BackendAdmissionGate>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// A place in the FIFO. Dropping it before the slot arrives unregisters the
/// waiter; dropping it after the slot was handed over refunds that slot so the
/// next waiter is woken instead of it being lost.
struct AdmissionTicket {
    gate: Arc<BackendAdmissionGate>,
    id: u64,
    rx: Option<oneshot::Receiver<()>>,
    /// Set once the slot has been converted into an [`AdmissionPermit`].
    taken: bool,
}

impl AdmissionTicket {
    async fn wait(mut self, context: Option<&dyn AsyncEngineContext>) -> Admission {
        let granted = {
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
                Some(context) if context.is_stopped() => false,
                Some(context) => tokio::select! {
                    biased;
                    _ = context.stopped() => false,
                    handoff = &mut *rx => handoff.is_ok(),
                },
                None => rx.await.is_ok(),
            }
        };

        if granted {
            self.taken = true;
            return Admission::Granted(AdmissionPermit {
                gate: Arc::clone(&self.gate),
            });
        }
        Admission::Cancelled
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
            Ok(()) => self.gate.release(),
            Err(_) => self.gate.remove_waiter(self.id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn gate(limit: usize, queue: usize) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(Some(limit), queue)
    }

    fn permit(admission: Admission) -> AdmissionPermit {
        match admission {
            Admission::Granted(permit) => permit,
            Admission::Overloaded => panic!("expected admission, got overloaded"),
            Admission::Cancelled => panic!("expected admission, got cancelled"),
        }
    }

    fn is_overloaded(admission: &Admission) -> bool {
        matches!(admission, Admission::Overloaded)
    }

    ///////////////////// SIZING: PURE RESOLVER INPUTS /////////////////////

    #[test]
    fn env_override_wins_over_hint_and_fallback() {
        assert_eq!(resolve_concurrency_limit(Some(7), Some(384)), 7);
        assert_eq!(resolve_concurrency_limit(Some(7), None), 7);
    }

    #[test]
    fn hint_wins_over_fallback() {
        assert_eq!(resolve_concurrency_limit(None, Some(384)), 384);
    }

    #[test]
    fn fallback_is_exactly_ten_thousand_and_never_multiplied() {
        assert_eq!(
            resolve_concurrency_limit(None, None),
            DEFAULT_CONCURRENCY_LIMIT
        );
        assert_eq!(DEFAULT_CONCURRENCY_LIMIT, 10_000);
        // A missing hint must not be routed through the 3/2 factor.
        assert_ne!(
            resolve_concurrency_limit(None, None),
            DEFAULT_CONCURRENCY_LIMIT * 3 / 2
        );
    }

    #[test]
    fn a_non_positive_override_falls_through_to_a_valid_hint() {
        // Each rule is validated on its own: a zero override must not swallow a
        // usable hint and drop the limit all the way to the fallback.
        assert_eq!(resolve_concurrency_limit(Some(0), Some(384)), 384);
    }

    #[test]
    fn zero_inputs_fall_through() {
        assert_eq!(
            resolve_concurrency_limit(Some(0), Some(0)),
            DEFAULT_CONCURRENCY_LIMIT
        );
        assert_eq!(
            resolve_concurrency_limit(None, Some(0)),
            DEFAULT_CONCURRENCY_LIMIT
        );
        assert_eq!(
            resolve_concurrency_limit(Some(0), None),
            DEFAULT_CONCURRENCY_LIMIT
        );
        assert_eq!(
            resolve_concurrency_limit(None, None),
            DEFAULT_CONCURRENCY_LIMIT
        );
    }

    #[test]
    fn automatic_limit_is_integer_ceil_of_three_halves() {
        assert_eq!(automatic_concurrency_limit(Some(1), Some(1)), Some(2));
        assert_eq!(automatic_concurrency_limit(Some(3), Some(1)), Some(5));
        assert_eq!(automatic_concurrency_limit(Some(4), Some(1)), Some(6));
        assert_eq!(automatic_concurrency_limit(Some(256), Some(1)), Some(384));
        assert_eq!(automatic_concurrency_limit(Some(2), Some(2)), Some(6));
        assert_eq!(automatic_concurrency_limit(Some(5), Some(3)), Some(23));
    }

    #[test]
    fn automatic_limit_rejects_missing_zero_and_overflow() {
        assert_eq!(automatic_concurrency_limit(None, Some(1)), None);
        assert_eq!(automatic_concurrency_limit(Some(256), None), None);
        assert_eq!(automatic_concurrency_limit(Some(0), Some(1)), None);
        assert_eq!(automatic_concurrency_limit(Some(256), Some(0)), None);
        assert_eq!(automatic_concurrency_limit(Some(u64::MAX), Some(2)), None);
        assert_eq!(automatic_concurrency_limit(Some(u64::MAX), Some(1)), None);
    }

    ///////////////////// ADMISSION: N + EXACT Q /////////////////////

    #[tokio::test]
    async fn n_direct_admissions_then_exactly_q_queue_then_reject() {
        let gate = gate(2, 3);

        let held: Vec<_> = vec![
            permit(gate.admit(None).await),
            permit(gate.admit(None).await),
        ];
        assert_eq!(gate.active(), 2);
        assert_eq!(gate.queued(), 0);

        // Exactly Q queue, and no dispatcher hides a Q+1th.
        let mut waiters = Vec::new();
        for _ in 0..3 {
            waiters.push(tokio::spawn({
                let gate = Arc::clone(&gate);
                async move { gate.admit(None).await }
            }));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(gate.queued(), 3, "queue holds exactly Q waiters");

        assert!(
            is_overloaded(&gate.admit(None).await),
            "the Q+1th request must be shed"
        );
        assert_eq!(gate.queued(), 3, "a rejection must not consume queue space");

        drop(held);
        for waiter in waiters {
            let admission = waiter.await.expect("waiter completes");
            assert!(matches!(admission, Admission::Granted(_)));
        }
    }

    #[tokio::test]
    async fn queue_is_fifo_and_a_new_request_never_bypasses_an_older_waiter() {
        // `admit` resolves only once the request is admitted, so the queueing
        // decision is observable through `queued()` and the admission order,
        // never by awaiting `admit` on the test task.
        let gate = gate(1, 8);
        let held = permit(gate.admit(None).await);

        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        // The fifth request arrives last and must be served last, behind every
        // waiter already in the queue.
        for index in 0..5usize {
            let gate = Arc::clone(&gate);
            let order = Arc::clone(&order);
            handles.push(tokio::spawn(async move {
                let permit = permit(gate.admit(None).await);
                order.lock().push(index);
                tokio::time::sleep(Duration::from_millis(10)).await;
                drop(permit);
            }));
            tokio::time::sleep(Duration::from_millis(20)).await;
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

    #[tokio::test]
    async fn a_late_arrival_queues_while_waiters_remain() {
        let gate = gate(1, 4);
        let held = permit(gate.admit(None).await);

        let queued = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1);

        // Still one waiter ahead, so this must not be admitted directly.
        let late = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            gate.queued(),
            2,
            "the late arrival joined the back of the FIFO"
        );

        drop(held);
        let first = permit(queued.await.expect("waiter completes"));
        drop(first);
        let second = permit(late.await.expect("waiter completes"));
        drop(second);
    }

    ///////////////////// CANCELLATION AND REFUND /////////////////////

    #[tokio::test]
    async fn a_dropped_queued_waiter_frees_its_queue_slot_immediately() {
        let gate = gate(1, 1);
        let _held = permit(gate.admit(None).await);

        let waiter = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1);
        assert!(is_overloaded(&gate.admit(None).await), "queue is full");

        waiter.abort();
        let _ = waiter.await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert_eq!(gate.queued(), 0, "an abandoned waiter must unregister");

        // The freed queue slot is reusable: this request queues rather than
        // being shed. It is spawned because `admit` only returns once admitted.
        let reuse = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1, "the freed queue slot must be reusable");

        drop(_held);
        let admission = tokio::time::timeout(Duration::from_secs(5), reuse)
            .await
            .expect("the requeued request must be admitted")
            .expect("waiter completes");
        drop(permit(admission));
    }

    #[tokio::test]
    async fn a_handoff_to_a_departed_waiter_is_refunded_to_the_next() {
        let gate = gate(1, 4);
        let held = permit(gate.admit(None).await);

        // Two waiters; the first is torn down at the same time as the release,
        // so the slot has to fall through to the second.
        let doomed = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move {
                let admission = gate.admit(None).await;
                // Drop the permit-or-ticket without using it.
                drop(admission);
            }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        let survivor = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 2);

        doomed.abort();
        let _ = doomed.await;
        drop(held);

        let admission = tokio::time::timeout(Duration::from_secs(5), survivor)
            .await
            .expect("the surviving waiter must inherit the refunded slot")
            .expect("waiter completes");
        let permit = permit(admission);
        assert_eq!(gate.active(), 1);
        drop(permit);
        assert_eq!(gate.active(), 0);
    }

    #[tokio::test]
    async fn cancellation_beats_a_simultaneous_handoff_and_refunds_the_slot() {
        use crate::pipeline::context::Controller;

        let gate = gate(1, 4);
        let held = permit(gate.admit(None).await);

        // Waiter A is cancellable; poll it exactly once so it registers in the
        // FIFO and then stops being polled.
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        let mut doomed = Box::pin(gate.admit(Some(context.as_ref())));
        assert!(futures::poll!(&mut doomed).is_pending());
        assert_eq!(gate.queued(), 1);

        // Waiter B queues behind it and is the one that must inherit the slot.
        let survivor = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
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
            matches!(admission, Admission::Cancelled),
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
        let _held = permit(gate.admit(None).await);

        let controller = Arc::new(Controller::default());
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            let controller: Arc<dyn AsyncEngineContext> = controller.clone();
            async move { gate.admit(Some(controller.as_ref())).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(gate.queued(), 1);

        controller.stop_generating();

        let admission = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("a cancelled waiter must return promptly")
            .expect("waiter completes");
        assert!(matches!(admission, Admission::Cancelled));
        assert_eq!(gate.queued(), 0, "cancellation frees the queue slot");
    }

    ///////////////////// SLOT RELEASE ON EVERY EXIT /////////////////////

    #[tokio::test]
    async fn slots_are_released_on_success_error_and_abort() {
        let gate = gate(1, 4);

        // Success.
        drop(permit(gate.admit(None).await));
        assert_eq!(gate.active(), 0);

        // Error path.
        async fn fails(_permit: AdmissionPermit) -> Result<(), &'static str> {
            Err("backend error")
        }
        assert!(fails(permit(gate.admit(None).await)).await.is_err());
        assert_eq!(gate.active(), 0);

        // Task abort mid-flight.
        let running = Arc::new(AtomicUsize::new(0));
        let task = tokio::spawn({
            let gate = Arc::clone(&gate);
            let running = Arc::clone(&running);
            async move {
                let _permit = permit(gate.admit(None).await);
                running.fetch_add(1, Ordering::SeqCst);
                std::future::pending::<()>().await;
            }
        });
        while running.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(gate.active(), 1);
        task.abort();
        let _ = task.await;
        assert_eq!(gate.active(), 0);
    }

    ///////////////////// LATE HINTS AND RESIZING /////////////////////

    #[tokio::test]
    async fn a_late_report_still_applies() {
        let gate = BackendAdmissionGate::new(None, 4);
        // Model-card registration lands after the gate has already served
        // requests on the fallback limit.
        drop(permit(gate.admit(None).await));

        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 384, "a late report must not be lost");
    }

    #[tokio::test]
    async fn conflicting_reports_keep_the_first() {
        let gate = BackendAdmissionGate::new(None, 4);

        gate.record_capacity_report(Some(2), Some(1));
        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 3, "the first usable report wins");

        // Re-reporting the same value is not a conflict.
        gate.record_capacity_report(Some(2), Some(1));
        assert_eq!(gate.limit(), 3);
    }

    #[tokio::test]
    async fn no_hint_leaves_the_fallback_in_place() {
        let gate = BackendAdmissionGate::new(None, 4);
        gate.record_capacity_report(None, Some(1));
        gate.record_capacity_report(Some(0), Some(1));
        assert_eq!(gate.limit(), DEFAULT_CONCURRENCY_LIMIT);
    }

    #[tokio::test]
    async fn the_env_override_survives_any_later_hint() {
        let gate = BackendAdmissionGate::new(Some(5), 4);
        gate.record_capacity_report(Some(256), Some(1));
        assert_eq!(gate.limit(), 5);
    }

    #[tokio::test]
    async fn growth_wakes_queued_waiters() {
        let gate = BackendAdmissionGate::new(None, 4);
        gate.set_limit_for_test(1);
        let _held = permit(gate.admit(None).await);

        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
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
        assert!(matches!(admission, Admission::Granted(_)));
    }

    #[tokio::test]
    async fn shrinking_never_revokes_a_held_permit() {
        // Starts on the fallback limit, so four permits are admitted freely.
        let gate = BackendAdmissionGate::new(None, 4);
        let permits = vec![
            permit(gate.admit(None).await),
            permit(gate.admit(None).await),
            permit(gate.admit(None).await),
            permit(gate.admit(None).await),
        ];
        assert_eq!(gate.active(), 4);

        // ceil(3/2 * 1 * 1) = 2, below the four slots already held.
        gate.record_capacity_report(Some(1), Some(1));
        assert_eq!(gate.limit(), 2);
        assert_eq!(gate.active(), 4, "held permits are never revoked");

        // New work waits until `active` drains below the smaller limit.
        let waiting = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.admit(None).await }
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

    ///////////////////// ONE GLOBAL MANAGER /////////////////////

    #[test]
    fn every_entry_path_reaches_one_manager() {
        // `handle_payload_shared` is the single admission point, and it always
        // resolves the gate through `global()`, so TCP and NATS requests and
        // every endpoint share this one instance.
        assert!(Arc::ptr_eq(global(), global()));
    }
}
