// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dispatching queued requests: which lane heads may go now, and the drain that admits them.

use super::*;

/// Whether a head's eligible ranks are the class's: no pin, caller allow-list, LoRA, required
/// taint, or, under an exclusive selector, affinity target. Such heads share one room answer
/// per class.
fn unconstrained(request: &SchedulingRequest, exclusive: bool) -> bool {
    request.pinned_worker.is_none()
        && request.allowed_worker_ids.is_none()
        && request.lora_name.is_none()
        && !request.routing_constraints.has_hard_constraints()
        && !(exclusive && request.affinity_target.is_some())
}

/// Asks the dispatch gate about one head inside a dispatch predicate: `false` holds the head
/// and records the earliest requested recheck still in the future; `true` stores the gate's
/// target on the head.
fn gate_allows(
    gate: &dyn DispatchGate,
    policy_class: &str,
    queued: &QueuedRequest,
    recheck_at: &Cell<Option<Instant>>,
) -> bool {
    match gate.check(&queued.dispatch_head(policy_class)) {
        DispatchDecision::Go { target } => {
            queued.gate_target.set(target);
            true
        }
        DispatchDecision::Hold { until } => {
            // A deadline that already passed arms nothing: re-arming it would wake the actor
            // again at once while the gate still holds.
            if let Some(until) = until.filter(|until| *until > Instant::now()) {
                recheck_at.set(Some(recheck_at.get().map_or(until, |at| at.min(until))));
            }
            false
        }
    }
}

/// One pass over the ready lane heads, asking whether each may dispatch now: the gate
/// first, then capacity for a class that queues on it. Load is read per head, so a head
/// sees the bookings made earlier in the drain. The ranks past the host's reject
/// thresholds are recorded at most once per pass, when a head first needs them.
struct DispatchPass<'a, P: SequencePublisher, C> {
    slots: &'a ActorSlots<P>,
    eligibility: &'a QueueEligibility,
    workers: &'a watch::Receiver<HashMap<WorkerId, C>>,
    gate: Option<&'a dyn DispatchGate>,
    /// The earliest recheck a held head asked for.
    gate_recheck_at: &'a Cell<Option<Instant>>,
    /// The host's reject thresholds, judged only while an affinity lane waits.
    reject: Option<&'a RejectThresholdsProvider>,
    full_ranks: &'a mut FullRankCounts,
    /// Whether this pass has recorded `full_ranks` yet.
    full_ranks_fresh: bool,
    /// Per class, the room answer its unconstrained heads share; all None as the pass starts.
    class_room: &'a mut [Option<bool>],
    decay_now: Instant,
}

impl<P: SequencePublisher + 'static, C: WorkerConfigLike> DispatchPass<'_, P, C> {
    fn check(
        &mut self,
        class_index: usize,
        class: &PolicyClassConfig,
        queued: &QueuedRequest,
    ) -> LaneCheck {
        if let Some(gate) = self.gate
            && !gate_allows(gate, &class.name, queued, self.gate_recheck_at)
        {
            return LaneCheck::Blocked(BlockReason::Gate);
        }
        if !class.queueing_enabled() {
            return LaneCheck::Dispatch;
        }
        let shared_answer = unconstrained(&queued.request, self.eligibility.exclusive);
        if shared_answer && let Some(room) = self.class_room[class_index] {
            return if room {
                LaneCheck::Dispatch
            } else {
                LaneCheck::Blocked(BlockReason::SharedCapacity)
            };
        }
        if !self.full_ranks_fresh {
            let rejecting = reset_full_ranks(self.full_ranks, reject_limits(self.reject));
            if let Some(full_ranks) = rejecting {
                self.slots
                    .record_full_worker_ranks_at(self.decay_now, full_ranks);
            }
            self.full_ranks_fresh = true;
        }
        let request = &queued.request;
        let check = self.eligibility.with(
            request,
            self.workers,
            self.full_ranks,
            false,
            |configs, routing, _, _, _| {
                if !all_eligible_workers_full(
                    &self.slots.worker_load_reader(self.decay_now),
                    configs,
                    class.busy_thresholds,
                    routing,
                ) {
                    LaneCheck::Dispatch
                } else if self.eligibility.exclusive
                    && request.pinned_worker.is_none()
                    && request.affinity_target.is_some()
                    && !routing.has_affinity_target()
                {
                    // `QueueEligibility::with` keeps an exclusive target only while it is
                    // eligible. It is gone: the head may go anywhere with room, so it
                    // competes on every pass.
                    LaneCheck::Blocked(BlockReason::Fallback)
                } else {
                    LaneCheck::Blocked(BlockReason::Capacity)
                }
            },
        );
        if shared_answer {
            self.class_room[class_index] = Some(check == LaneCheck::Dispatch);
            if check == LaneCheck::Blocked(BlockReason::Capacity) {
                // Every unconstrained head of the class shares this refusal, so the queue
                // probes one of them on the next capacity change.
                return LaneCheck::Blocked(BlockReason::SharedCapacity);
            }
        }
        check
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Whether a lane head can dispatch now. A hold this asks about still arms the gate's
    /// recheck, so a newly exposed root is woken at its own deadline.
    pub(super) fn has_dispatchable_ready_head(&mut self) -> bool {
        let decay_now = Instant::now();
        let gate_recheck_at = Cell::new(self.gate_recheck_at);
        self.class_room.clear();
        self.class_room.resize(self.pending.class_count(), None);
        let mut pass = DispatchPass {
            slots: &self.slots,
            eligibility: &self.eligibility,
            workers: &self.workers_with_configs,
            gate: self.dispatch_gate.get().map(|gate| gate.as_ref()),
            gate_recheck_at: &gate_recheck_at,
            reject: self
                .pending
                .has_affinity_backlog()
                .then_some(self.reject_thresholds.as_ref())
                .flatten(),
            full_ranks: &mut self.gate_full_ranks,
            full_ranks_fresh: false,
            class_room: &mut self.class_room,
            decay_now,
        };
        let dispatchable = self
            .pending
            .any_ready_head(|class_index, class, queued| pass.check(class_index, class, queued));
        self.gate_recheck_at = gate_recheck_at.get();
        dispatchable
    }

    /// Consume recorded capacity changes before draining. A worker free re-asks the lanes
    /// waiting on that worker and every session lane blocked on capacity; a full recheck
    /// re-asks every capacity-blocked lane. Lanes the gate holds are re-asked only when the
    /// gate wakes the actor or its hold timer fires, since capacity alone cannot free them.
    /// `recheck` widens the scope for the caller's own reason.
    pub(super) fn recheck_capacity_updates(&mut self, recheck: Option<RecheckScope>) {
        let drained = self.capacity_updates.drain_into(&mut self.capacity_workers);
        let mut scope = recheck;
        if drained.gate_recheck {
            scope = Some(RecheckScope::All);
        }
        // Consume hints even when there is no pending work.
        if self.pending.pending_count() == 0 {
            return;
        }
        let mut capacity_all = drained.full_recheck;
        if !capacity_all && scope.is_none() {
            // A changed worker without a config is gone: its lanes must look elsewhere.
            let configs = self.workers_with_configs.borrow();
            capacity_all = self
                .capacity_workers
                .iter()
                .any(|&worker| worker_config_for_rank(&configs, worker).is_err());
        }
        if capacity_all {
            scope = Some(scope.map_or(RecheckScope::Capacity, |scope| {
                scope.max(RecheckScope::Capacity)
            }));
        }
        match scope {
            Some(scope) => self.pending.recheck_blocked_lanes(scope),
            // Named workers freed: re-ask the lanes pinned or bound to them, and the session
            // lanes, which may run on any of them.
            None => {
                for &worker in &self.capacity_workers {
                    self.pending.recheck_worker(worker);
                }
            }
        }
    }

    pub(super) async fn handle_capacity_updates(&mut self, drain_cleanup: bool) {
        self.expire_due_classes(Instant::now());
        let cleanup_ready = drain_cleanup && self.drain_cleanup();
        self.recheck_capacity_updates(cleanup_ready.then_some(RecheckScope::All));
        if self.pending.pending_count() == 0 {
            return;
        }
        self.drain_ready(None).await;
    }

    /// Drains after a change that may free blocked lanes: `recheck` names the lanes to ask
    /// again beyond what the recorded capacity changes call for.
    pub(super) async fn handle_update(
        &mut self,
        reply: Option<CommandReply>,
        recheck: Option<RecheckScope>,
    ) {
        self.expire_due_classes(Instant::now());
        self.recheck_capacity_updates(recheck);
        self.drain_ready(reply).await;
    }

    /// Admits ready work until no lane head can dispatch. `reply`, from the command that started
    /// the drain, is sent before the first refresh await or when the drain ends, never held across
    /// that await.
    async fn drain_ready(&mut self, mut reply: Option<CommandReply>) {
        // The gate is installed once, before the actor starts.
        let gate = self.dispatch_gate.get().cloned();
        // Continuation draining stays actor-local; never self-send through the
        // bounded command channel while processing an update.
        loop {
            if self.pending.pending_count() == 0 {
                break;
            }
            let decay_now = Instant::now();
            // An expired head is rejected here rather than dispatched.
            self.expire_due_classes(decay_now);
            // Load is read after the previous admission booked, so this one sees that
            // reservation. Frees and peer bookings from other tasks can land between
            // two reads, as they can right after the gate.
            let gate_recheck_at = Cell::new(self.gate_recheck_at);
            let extracted = {
                self.class_room.clear();
                self.class_room.resize(self.pending.class_count(), None);
                let mut pass = DispatchPass {
                    slots: &self.slots,
                    eligibility: &self.eligibility,
                    workers: &self.workers_with_configs,
                    gate: gate.as_deref(),
                    gate_recheck_at: &gate_recheck_at,
                    reject: self
                        .pending
                        .has_affinity_backlog()
                        .then_some(self.reject_thresholds.as_ref())
                        .flatten(),
                    full_ranks: &mut self.gate_full_ranks,
                    full_ranks_fresh: false,
                    class_room: &mut self.class_room,
                    decay_now,
                };
                self.pending.extract_next(|class_index, class, queued| {
                    pass.check(class_index, class, queued)
                })
            };
            self.gate_recheck_at = gate_recheck_at.get();
            let Some((mut candidate, charge)) = extracted else {
                break;
            };
            let snapshot = candidate.snapshot();
            let class_index = candidate.class_index();
            self.class_counters[class_index].sub(snapshot);
            let queued = candidate.payload_mut();
            let mut refreshed = false;
            let mut released = false;
            let mut stopping = false;
            let refreshing = should_refresh_overlap(
                self.overlap_scores_refresh.is_some(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.enqueue_at,
                decay_now,
            );
            if refreshing {
                if let Some(reply) = reply.take() {
                    reply.send();
                }
                // NOTE: Overlap refresh is bounded by `OVERLAP_REFRESH_TIMEOUT`. We intentionally
                // accept load crossing the class threshold during this await: busy
                // thresholds guide admission, not reservation. The candidate's DRR
                // charge stays unpaid until its admission outcome is known.
                match self.refresh_candidate(queued, decay_now).await {
                    RefreshEnd::Finished(Some(snapshot)) => {
                        refreshed = true;
                        tracing::debug!(
                            request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                            wait_ms = queued.enqueue_at.elapsed().as_millis() as u64,
                            "refreshed overlap scores after long queue wait"
                        );
                        queued.request.overlap = snapshot.overlap;
                        queued.request.kv_transfer_candidates =
                            if queued.request.retain_kv_transfer_chain {
                                snapshot.kv_transfer_candidates
                            } else {
                                None
                            };
                    }
                    RefreshEnd::Finished(None) => {}
                    end => {
                        tracing::debug!(
                            request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                            wait_ms = queued.enqueue_at.elapsed().as_millis() as u64,
                            outcome = ?end,
                            "overlap refresh ended unfinished"
                        );
                        released = matches!(end, RefreshEnd::Released);
                        stopping = matches!(end, RefreshEnd::Shutdown);
                    }
                }
                self.turn_prepared.drop_judged_load();
            }
            let mut queued = candidate.into_payload();
            // The gate's `Go` target replaces the soft affinity target unless the caller
            // pinned a worker; the gate learns the head left the queue once, here.
            if let Some(gate) = &gate
                && !released
            {
                let target = queued.gate_target.take();
                if queued.request.pinned_worker.is_none()
                    && let Some(target) = target
                {
                    queued.request.affinity_target = Some(target.into());
                }
                let policy_class = &self.profile.class(class_index).name;
                gate.dispatched(
                    &queued.dispatch_head(policy_class),
                    queued.request.pinned_worker.or(target),
                );
            }
            let prepared = self.turn_prepared.take_for(queued.enqueue_turn, refreshed);
            // The due time can pass during the refresh. Admission checks it again before
            // selection and before booking or delivery.
            let admission = if released {
                Admission::Refused
            } else if refreshing && queued.due_at.is_some_and(|due_at| due_at <= Instant::now()) {
                let mut request = queued.request;
                self.reject_due_time_passed(class_index, &mut request);
                Admission::Refused
            } else {
                tracing::trace!(
                    policy_class = self.profile.class(class_index).name,
                    "scheduling request from pending queue"
                );
                let context = AdmissionContext {
                    class_index,
                    due_at: queued.due_at,
                    attempt_tx: queued.attempt_tx,
                    lifecycle_transfer: queued.lifecycle_transfer,
                };
                self.admit_one(context, queued.request, Instant::now(), prepared)
            };
            self.pending.settle(charge, admission != Admission::Refused);
            self.rearm_class_timer(class_index);
            // Every sender is gone, so the actor stops next.
            if stopping {
                break;
            }
        }
        if let Some(reply) = reply {
            reply.send();
        }
    }
}
