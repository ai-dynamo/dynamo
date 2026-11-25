// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::Poll;

use rustc_hash::FxHashMap;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::multi_worker::{
    ActiveSequencesMultiWorker, ReplicaWorkerPolicy, SequencePublisher, SequenceSubscriber,
};
use super::prompt_registry::WorkerLoadSnapshot;
use super::request_maps::RequestBooking;
use super::single::{PrefillCompletion, RequestId};
use crate::booking::SchedulerBookingDescriptor;
use crate::wire::{
    ActiveSequenceEvent, ActiveSequenceEventData, ActiveSequenceEventOrigin,
    MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS,
};
use dynamo_router_protocol::{attempt::AttemptId, worker::WorkerWithDpRank};

/// How long a worker event waits for its replicated booking.
const PENDING_WORKER_EVENT_TTL: Duration = Duration::from_secs(30);
/// Most pending worker events kept; the oldest are dropped first.
const PENDING_WORKER_EVENT_CAPACITY: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingWorkerEvent {
    MarkPrefillCompleted,
    Free,
}

/// Worker events for requests not booked here yet.
///
/// Workers publish to every router directly, so their events can overtake the
/// replicated `AddRequest`. The booking takes the event when it arrives: a freed
/// request is not booked, and a prefill-completed one is booked as decoding.
/// Each event records the booking it found here (`None` for none), so only the
/// Add that replaces that booking takes it.
#[derive(Default)]
pub(super) struct PendingWorkerEvents {
    events: FxHashMap<
        String,
        (
            WorkerWithDpRank,
            PendingWorkerEvent,
            Option<AttemptId>,
            Instant,
        ),
    >,
    /// Recording order, for expiry. May hold ids already taken.
    order: VecDeque<(Instant, String)>,
}

impl PendingWorkerEvents {
    fn record(
        &mut self,
        request_id: String,
        worker: WorkerWithDpRank,
        event: PendingWorkerEvent,
        held_against: Option<AttemptId>,
        now: Instant,
    ) {
        while let Some((recorded_at, _)) = self.order.front()
            && (now.duration_since(*recorded_at) >= PENDING_WORKER_EVENT_TTL
                || self.order.len() >= PENDING_WORKER_EVENT_CAPACITY)
        {
            let Some((recorded_at, request_id)) = self.order.pop_front() else {
                break;
            };
            if self
                .events
                .get(&request_id)
                .is_some_and(|(_, _, _, at)| *at == recorded_at)
            {
                self.events.remove(&request_id);
            }
        }
        match self.events.get_mut(&request_id) {
            // A free supersedes an earlier prefill mark for the same booking.
            Some(existing) if existing.0 == worker => {
                if event == PendingWorkerEvent::Free {
                    existing.1 = event;
                    existing.2 = held_against;
                }
            }
            _ => {
                self.order.push_back((now, request_id.clone()));
                self.events
                    .insert(request_id, (worker, event, held_against, now));
            }
        }
    }

    /// Take the event a booking of `request_id` on `worker` must honor, given the
    /// booking its Add replaces (`None` for none).
    fn take(
        &mut self,
        request_id: &str,
        worker: WorkerWithDpRank,
        replaced: Option<AttemptId>,
        now: Instant,
    ) -> Option<PendingWorkerEvent> {
        if self.events.is_empty() {
            return None;
        }
        let (event_worker, event, held_against, recorded_at) = self.events.remove(request_id)?;
        (event_worker == worker
            && held_against == replaced
            && now.duration_since(recorded_at) < PENDING_WORKER_EVENT_TTL)
            .then_some(event)
    }
}

#[derive(Default)]
pub(super) struct ReplicaBatchEffects {
    worker_loads: FxHashMap<WorkerWithDpRank, PendingLoadPublication>,
    wake_scheduler: bool,
    cleanup_prompt_trie: bool,
}

struct PendingLoadPublication {
    latest_load: WorkerLoadSnapshot,
    publish: bool,
}

impl ReplicaBatchEffects {
    fn record_worker_load(
        &mut self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        publish: bool,
    ) {
        self.worker_loads
            .entry(worker)
            .and_modify(|pending| {
                pending.latest_load = load;
                pending.publish |= publish;
            })
            .or_insert(PendingLoadPublication {
                latest_load: load,
                publish,
            });
    }
}

impl<P: SequencePublisher + 'static> ActiveSequencesMultiWorker<P> {
    /// Test hook: apply one decoded replica-sync batch and flush its deferred effects once.
    #[cfg(any(test, feature = "testing"))]
    pub fn apply_replica_batch(&self, events: Vec<ActiveSequenceEvent>) {
        let mut effects = ReplicaBatchEffects::default();
        for event in events {
            self.apply_replica_event(event, &mut effects);
        }
        self.flush_replica_batch_effects(&mut effects);
    }

    /// Spawn a background task that subscribes to replica-sync events from peer routers
    /// and applies them to the local state.
    pub fn start_replica_sync<S: SequenceSubscriber + 'static>(
        self: &Arc<Self>,
        subscriber: S,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = this.run_replica_sync(subscriber, cancel_token).await {
                tracing::error!("Error in active sequences events subscription: {error}");
            }
        });
    }

    pub(super) async fn run_replica_sync<S: SequenceSubscriber>(
        &self,
        mut subscriber: S,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let mut effects = ReplicaBatchEffects::default();

        loop {
            let result = tokio::select! {
                result = subscriber.next_event() => result,
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            };

            let Some(result) = result else {
                break;
            };
            let Ok(event) = result else {
                tracing::error!(
                    "Error receiving active sequence event: {}",
                    result.unwrap_err()
                );
                continue;
            };

            let batch_start = Instant::now();
            let mut batch_events = 0;
            let mut exit_after_flush = false;
            let mut yield_after_flush = false;
            let mut next_event = Some(event);

            loop {
                let event = next_event
                    .take()
                    .expect("replica batch event must be present");
                self.apply_replica_event(event, &mut effects);
                batch_events += 1;

                if cancel_token.is_cancelled() {
                    exit_after_flush = true;
                    break;
                }

                if batch_events >= MAX_REPLICA_BATCH_EVENTS
                    || batch_start.elapsed() >= MAX_REPLICA_BATCH_DURATION
                {
                    yield_after_flush = true;
                    break;
                }

                match poll_fn(|cx| Poll::Ready(subscriber.poll_next_event(cx))).await {
                    Poll::Ready(Some(Ok(event))) => next_event = Some(event),
                    Poll::Ready(Some(Err(error))) => {
                        tracing::error!("Error receiving active sequence event: {error}");
                        break;
                    }
                    Poll::Ready(None) => {
                        exit_after_flush = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }

            self.flush_replica_batch_effects(&mut effects);
            subscriber.record_drain(batch_events);

            if exit_after_flush {
                break;
            }
            if yield_after_flush {
                tokio::task::yield_now().await;
            }
        }

        Ok(())
    }

    fn apply_replica_event(&self, event: ActiveSequenceEvent, effects: &mut ReplicaBatchEffects) {
        let ActiveSequenceEvent {
            request_id,
            worker: event_worker,
            data,
            router_id,
            lora_name,
            origin,
        } = event;

        if origin == ActiveSequenceEventOrigin::Router && router_id == self.router_id {
            return;
        }
        if !self.replica_sync && !matches!(data, ActiveSequenceEventData::MarkPrefillCompleted) {
            return;
        }
        let can_defer_worker_event =
            self.replica_sync && origin == ActiveSequenceEventOrigin::Worker;

        // ActiveSequenceEvent does not carry prompt-load decay timestamps yet.
        // Peer routers still approximate decay anchoring with local receive time.
        let decay_now = Instant::now();

        match data {
            ActiveSequenceEventData::AddRequest {
                token_sequence,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
            } => {
                // The same router only re-books an id after its own copy is gone, so its
                // Add replaces the mirror left here (its `Free` was lost or never sent),
                // even when the new attempt is already over and is not booked below.
                // An Add from another router, or onto a local booking, is still ignored.
                let stale = self
                    .request_index
                    .booking_and_replica_router(&request_id)
                    .and_then(|(booking, replica_router)| {
                        (replica_router == Some(router_id)).then_some(booking)
                    });
                let pending_event = self.pending_worker_events.lock().take(
                    &request_id,
                    event_worker,
                    stale.map(|stale| stale.attempt_id),
                    decay_now,
                );
                if let Some(stale) = stale {
                    self.free_replica_booking(&request_id, stale, decay_now, effects);
                }
                if pending_event == Some(PendingWorkerEvent::Free) {
                    return;
                }
                let Ok(attempt_id) = self.request_index.try_insert_request(
                    request_id.clone(),
                    event_worker,
                    lora_name,
                    Some(router_id),
                ) else {
                    return;
                };
                let booking = SchedulerBookingDescriptor {
                    request_id: request_id.clone(),
                    worker: event_worker,
                    attempt_id,
                };
                if self.replica_worker_policy == ReplicaWorkerPolicy::LazyRegister {
                    self.ensure_worker_registered(event_worker);
                }
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&event_worker) else {
                    self.request_index.remove_request_if_booking(
                        &request_id,
                        event_worker,
                        attempt_id,
                    );
                    tracing::debug!(
                        worker = ?event_worker,
                        "Dropping replica AddRequest for unregistered worker"
                    );
                    return;
                };
                let (expired_request_ids, load) = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let outcome = seq.add_request_with_prefill_tracking(
                        request_id,
                        token_sequence,
                        expected_output_tokens,
                        track_prefill_tokens,
                        prefill_load_hint,
                        decay_now,
                    );
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            event_worker,
                            outcome.membership_delta,
                            load,
                        );
                    (outcome.expired_request_ids, load)
                };
                drop(table);
                self.request_index
                    .remove_requests(expired_request_ids.iter());
                let prefilled_booking = (pending_event
                    == Some(PendingWorkerEvent::MarkPrefillCompleted))
                .then(|| booking.clone());
                if let Some(observer) = self.replica_request_lease_observer() {
                    observer.admitted(booking);
                }
                effects.record_worker_load(event_worker, load, true);
                effects.cleanup_prompt_trie = true;
                if let Some(booking) = prefilled_booking {
                    self.apply_prefill_completed(&booking, decay_now, false, effects);
                }
            }
            // In both arms below, a worker's event for a worker other than the one booked
            // waits like one with no booking: it may belong to a retry whose Add has yet
            // to replace the stale mirror here.
            ActiveSequenceEventData::Free => {
                let found = self.request_index.booking_for(&request_id);
                let Some(current) = found.filter(|current| current.worker == event_worker) else {
                    if can_defer_worker_event {
                        self.pending_worker_events.lock().record(
                            request_id,
                            event_worker,
                            PendingWorkerEvent::Free,
                            found.map(|found| found.attempt_id),
                            decay_now,
                        );
                    }
                    return;
                };
                self.free_replica_booking(&request_id, current, decay_now, effects);
            }
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let found = self.request_index.booking_and_replica_router(&request_id);
                let Some((current, replica_router)) =
                    found.filter(|(current, _)| current.worker == event_worker)
                else {
                    if can_defer_worker_event {
                        self.pending_worker_events.lock().record(
                            request_id,
                            event_worker,
                            PendingWorkerEvent::MarkPrefillCompleted,
                            found.map(|(found, _)| found.attempt_id),
                            decay_now,
                        );
                    }
                    return;
                };
                // A router's mark is for its own booking, so it applies only to the mirror
                // of that router's booking, never to a local one or another router's.
                // Workers before the `origin` field send marks with no origin and their
                // own id as `router_id`; those are still worker marks.
                let from_owner = origin == ActiveSequenceEventOrigin::Router
                    && router_id != event_worker.worker_id;
                if from_owner && replica_router != Some(router_id) {
                    return;
                }
                let booking = SchedulerBookingDescriptor {
                    request_id,
                    worker: current.worker,
                    attempt_id: current.attempt_id,
                };
                self.apply_prefill_completed(&booking, decay_now, from_owner, effects);
            }
        }
    }

    /// Free `current` if it is still the live booking of `request_id`. Returns with no
    /// table guard held.
    pub(super) fn free_replica_booking(
        &self,
        request_id: &RequestId,
        current: RequestBooking,
        decay_now: Instant,
        effects: &mut ReplicaBatchEffects,
    ) {
        let booking = SchedulerBookingDescriptor {
            request_id: request_id.clone(),
            worker: current.worker,
            attempt_id: current.attempt_id,
        };
        let table = self.workers.read();
        let Some(&idx) = table.index.get(&current.worker) else {
            return;
        };
        let load = {
            let slot = &table.slots[idx];
            let mut seq = slot.sequences.write();
            // Recheck under the sequence lock: since the caller read `current`, it may
            // have expired and the id been booked again on this worker.
            if self.request_index.booking_for(request_id) != Some(current) {
                return;
            }
            let Some(delta) = seq.free(request_id, decay_now) else {
                return;
            };
            let load = seq.worker_load_snapshot();
            self.prompt_registry
                .apply_membership_delta_and_load_without_cleanup(current.worker, delta, load);
            load
        };
        drop(table);
        self.request_index.remove_request_if_booking(
            request_id,
            current.worker,
            current.attempt_id,
        );
        if let Some(observer) = self.replica_request_lease_observer() {
            observer.completed(&booking);
        }
        effects.record_worker_load(current.worker, load, true);
        effects.wake_scheduler = true;
        effects.cleanup_prompt_trie = true;
    }

    pub(super) fn apply_prefill_completed(
        &self,
        booking: &SchedulerBookingDescriptor,
        now: Instant,
        from_owner: bool,
        effects: &mut ReplicaBatchEffects,
    ) {
        let table = self.workers.read();
        let Some(&idx) = table.index.get(&booking.worker) else {
            return;
        };
        let (completion, load) = {
            let mut seq = table.slots[idx].sequences.write();
            // Recheck under the sequence lock: since the caller read `booking`, it may
            // have been freed and the id booked again on this worker.
            if !self.has_booking(booking) {
                return;
            }
            let completion = seq.mark_prefill_completed(&booking.request_id, now);
            let load = (completion != PrefillCompletion::Unchanged).then(|| {
                let load = seq.worker_load_snapshot();
                self.prompt_registry
                    .replace_worker_load_state(booking.worker, load);
                load
            });
            (completion, load)
        };
        drop(table);
        if let Some(load) = load {
            effects.record_worker_load(booking.worker, load, false);
        }
        // A mark from the router this booking mirrors is progress even when it changes
        // nothing: that router re-sends it as a keepalive while it decodes.
        if (from_owner || completion == PrefillCompletion::LoadReleased)
            && let Some(observer) = self.replica_request_lease_observer()
        {
            observer.progressed(booking);
        }
        if completion == PrefillCompletion::LoadReleased {
            effects.wake_scheduler = true;
        }
    }

    fn flush_replica_batch_effects(&self, effects: &mut ReplicaBatchEffects) {
        let decay_now = Instant::now();
        let mut scheduler_loads = Vec::with_capacity(effects.worker_loads.len());
        for (worker, pending) in effects.worker_loads.drain() {
            // Every applied change refreshes the load gauges; only shared load is republished.
            let snapshot =
                self.observe_worker_load_snapshot(worker, pending.latest_load, decay_now);
            if pending.publish {
                scheduler_loads.push(snapshot);
            }
        }
        if !scheduler_loads.is_empty() {
            self.publisher.publish_scheduler_load_batch(scheduler_loads);
        }

        if std::mem::take(&mut effects.wake_scheduler) {
            self.notify_remote_state_update();
        }
        if std::mem::take(&mut effects.cleanup_prompt_trie) {
            self.prompt_registry.maybe_cleanup();
        }
    }
}
