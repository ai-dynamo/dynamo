// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A request's arrival at the actor, which admits it at once or places it in its class queue,
//! and the expiry of queued requests.

use super::*;

/// How an arrival's attempt to enter its class queue ended.
enum Insertion {
    /// The arrival entered the queue.
    Queued,
    /// The arrival was answered with a rejection, counted once, or dropped uncounted because
    /// its receiver was already closed. `exposed` reports whether expiry during the attempt
    /// removed an entry from a class that kept others, which can expose a head that runs.
    Refused { exposed: bool },
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Returns whether ready work may have changed: the request entered the queue or was booked,
    /// or the expiry it ran exposed a class head.
    pub(super) fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        attempt_tx: Option<oneshot::Sender<AttemptId>>,
        lifecycle_transfer: Option<Arc<AdmissionLifecycleTransfer>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        prepared: Option<Box<PreparedAdmission>>,
        queue_metadata: QueueMetadata,
    ) -> bool {
        let class_index = queue_metadata.class_index;
        self.class_counters[class_index]
            .received_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        let decay_now = Instant::now();
        let due_at = self.pending.enforced_due_at(&queue_metadata);
        if due_at.is_some_and(|due_at| due_at <= decay_now) {
            self.reject_due_time_passed(class_index, &mut request);
            return false;
        }
        let context = AdmissionContext {
            class_index,
            due_at,
            attempt_tx,
            lifecycle_transfer,
        };
        let snapshot = queue_metadata.snapshot;
        let class = self.profile.class(class_index);
        let mut limits_now = decay_now;
        // With zero discovered workers every per-worker limit scales to zero and
        // `queue_rejection` would reject with `limit: 0`. Admit directly instead
        // so selection reports the real condition, `NoEndpoints`.
        if self.workers_with_configs.borrow().is_empty() {
            return self.admit_one(context, request, decay_now, prepared) == Admission::Booked;
        }
        // A class with any zero per-worker limit never queues: keep the direct
        // admission semantics by admitting while a worker has prefill capacity,
        // and otherwise fall through so `queue_rejection` produces the same
        // limit-zero rejection the queued path always has.
        if class.never_queues() {
            let should_queue = class.queueing_enabled()
                && (self.pending.has_backlog(class_index) || {
                    let reject = (self.eligibility.exclusive && request.affinity_target.is_some())
                        .then_some(self.reject_thresholds.as_ref())
                        .flatten();
                    let full_ranks =
                        reset_full_ranks(&mut self.gate_full_ranks, reject_limits(reject));
                    if let Some(full_ranks) = full_ranks {
                        self.slots
                            .record_full_worker_ranks_at(decay_now, full_ranks);
                    }
                    self.eligibility.with(
                        &request,
                        &self.workers_with_configs,
                        &self.gate_full_ranks,
                        false,
                        |configs, eligibility, _, _, _| {
                            all_eligible_workers_full(
                                &self.slots.worker_load_reader(decay_now),
                                configs,
                                class.busy_thresholds,
                                eligibility,
                            )
                        },
                    )
                });
            if !should_queue {
                return self.admit_one(context, request, decay_now, prepared) == Admission::Booked;
            }
            // The availability check and load projection can take time. Judge this request's
            // due time, and the queued entries' below, at a fresh instant before the limits.
            limits_now = Instant::now();
            if due_at.is_some_and(|due_at| due_at <= limits_now) {
                self.reject_due_time_passed(class_index, &mut request);
                return false;
            }
        }
        tracing::trace!(
            policy_class = self.profile.class(class_index).name,
            "ordering request"
        );
        let AdmissionContext {
            attempt_tx,
            lifecycle_transfer,
            ..
        } = context;
        let placement = self.eligibility.lane(&request, queue_metadata.lane_key);
        let queued = QueuedRequest {
            request,
            attempt_tx,
            lifecycle_transfer: lifecycle_transfer.clone(),
            enqueue_at: decay_now,
            due_at,
            block_hashes,
            enqueue_turn: self.turn_prepared.turn(),
            gate_target: Cell::new(None),
        };
        self.turn_prepared.stash(prepared);
        if let Insertion::Refused { exposed } =
            self.insert_arrival(queue_metadata, queued, limits_now)
        {
            return exposed;
        }
        if let Some(due_at) = due_at {
            self.class_timers.arm_if_earlier(class_index, due_at);
        }
        if let QueueLane::Affinity(target) = placement {
            self.pending.recheck_affinity_target(target);
        }
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_pending();
        }
        self.class_counters[class_index].add(snapshot);
        true
    }

    /// Inserts an arrival once the entries due by `now` have left, so the class limits do not
    /// count them. The arrival's own due time is judged at a fresh instant right before each
    /// insertion attempt and, ahead of its class's, after a limit refusal. A refusal stands only
    /// once no entry of the class was due; each retry removes an existing entry and nothing joins
    /// the class meanwhile, so this ends.
    ///
    /// A refused arrival is answered and counted here, or dropped uncounted when its receiver is
    /// already closed. The caller counts a queued arrival as pending and arms its class timer.
    fn insert_arrival(
        &mut self,
        queue_metadata: QueueMetadata,
        mut queued: QueuedRequest,
        now: Instant,
    ) -> Insertion {
        let class_index = queue_metadata.class_index;
        let due_at = queued.due_at;
        // A head that expiry exposes may be able to run, so every refusal below still reports it.
        let mut exposed = self.expire_due_classes(now);
        let priority_jump = queued.request.priority_jump;
        let strict_priority = queued.request.strict_priority;
        let placement = self
            .eligibility
            .lane(&queued.request, queue_metadata.lane_key);
        let worker_count = self.workers_with_configs.borrow().len();
        loop {
            if due_at.is_some_and(|due_at| due_at <= Instant::now()) {
                self.reject_due_time_passed(class_index, &mut queued.request);
                return Insertion::Refused { exposed };
            }
            let (rejection, rejected) = match self.pending.enqueue_with_due_at(
                queue_metadata,
                worker_count,
                priority_jump,
                strict_priority,
                placement,
                queued,
            ) {
                Ok(()) => return Insertion::Queued,
                Err(rejected) => rejected,
            };
            let mut request = rejected.request;
            let now = Instant::now();
            if due_at.is_some_and(|due_at| due_at <= now) {
                self.reject_due_time_passed(class_index, &mut request);
                return Insertion::Refused { exposed };
            }
            // As with a due time, a receiver already closed at this check is not counted.
            if request.response_is_closed() {
                return Insertion::Refused { exposed };
            }
            if self
                .pending
                .next_due_at(class_index)
                .is_some_and(|due_at| due_at <= now)
            {
                exposed |= self.expire_class(class_index, now);
                queued = QueuedRequest {
                    request,
                    ..rejected
                };
                continue;
            }
            self.class_counters[class_index]
                .rejected_queue_size_exceeded_total
                .fetch_add(1, AtomicOrdering::Relaxed);
            request.respond(Err(KvSchedulerError::QueueRejected(rejection)));
            return Insertion::Refused { exposed };
        }
    }

    /// Rejects a request whose due time passed and counts it against its class. A request whose
    /// response receiver is already closed at this check is dropped uncounted, as booking drops
    /// it; a receiver closed after the check can still be counted.
    pub(super) fn reject_due_time_passed(
        &self,
        class_index: usize,
        request: &mut SchedulingRequest,
    ) {
        if request.response_is_closed() {
            return;
        }
        self.class_counters[class_index]
            .rejected_due_time_passed_total
            .fetch_add(1, AtomicOrdering::Relaxed);
        request.respond(Err(KvSchedulerError::DeadlineExceeded));
    }

    /// Rejects the queued requests of every class whose timer is due at `now`. Each such class
    /// loses its overdue entries, without a DRR charge, and rearms at its next due time. Returns
    /// whether a class kept entries after losing one, as [`Self::expire_class`] does.
    pub(super) fn expire_due_classes(&mut self, now: Instant) -> bool {
        let mut exposed = false;
        while let Some(class_index) = self.class_timers.pop_due(now) {
            exposed |= self.expire_class(class_index, now);
        }
        exposed
    }

    /// Rejects the class's entries that expire by `now`, without a DRR charge, and moves its
    /// timer to its next due time. Returns whether it removed an entry and the class kept others:
    /// a lane root may then have changed to one that can run.
    fn expire_class(&mut self, class_index: usize, now: Instant) -> bool {
        let expired = self.pending.take_expired(class_index, now);
        let removed = !expired.is_empty();
        for entry in expired {
            self.class_counters[class_index].sub(entry.snapshot());
            let mut request = entry.into_payload().request;
            self.reject_due_time_passed(class_index, &mut request);
        }
        let due_at = self.pending.next_due_at(class_index);
        self.class_timers.arm(class_index, due_at);
        removed && self.pending.has_backlog(class_index)
    }

    /// Follows a removal from the class: an unarmed timer means the class had no entry that
    /// expires, and removing one cannot change that.
    pub(super) fn rearm_class_timer(&mut self, class_index: usize) {
        if self.class_timers.armed[class_index].is_some() {
            let due_at = self.pending.next_due_at(class_index);
            self.class_timers.arm(class_index, due_at);
        }
    }
}
