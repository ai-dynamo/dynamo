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
use super::single::PrefillCompletion;
use crate::booking::SchedulerBookingDescriptor;
use crate::wire::{
    ActiveSequenceEvent, ActiveSequenceEventData, ActiveSequenceEventOrigin,
    MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;

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
#[derive(Default)]
pub(super) struct PendingWorkerEvents {
    events: FxHashMap<String, (WorkerWithDpRank, PendingWorkerEvent, Instant)>,
    /// Recording order, for expiry. May hold ids already taken.
    order: VecDeque<(Instant, String)>,
}

impl PendingWorkerEvents {
    fn record(
        &mut self,
        request_id: String,
        worker: WorkerWithDpRank,
        event: PendingWorkerEvent,
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
                .is_some_and(|(_, _, at)| *at == recorded_at)
            {
                self.events.remove(&request_id);
            }
        }
        match self.events.get_mut(&request_id) {
            // A free supersedes an earlier prefill mark for the same booking.
            Some(existing) if existing.0 == worker => {
                if event == PendingWorkerEvent::Free {
                    existing.1 = event;
                }
            }
            _ => {
                self.order.push_back((now, request_id.clone()));
                self.events.insert(request_id, (worker, event, now));
            }
        }
    }

    /// Take the event a booking of `request_id` on `worker` must honor.
    fn take(
        &mut self,
        request_id: &str,
        worker: WorkerWithDpRank,
        now: Instant,
    ) -> Option<PendingWorkerEvent> {
        if self.events.is_empty() {
            return None;
        }
        let (event_worker, event, recorded_at) = self.events.remove(request_id)?;
        (event_worker == worker && now.duration_since(recorded_at) < PENDING_WORKER_EVENT_TTL)
            .then_some(event)
    }
}

#[derive(Default)]
struct ReplicaBatchEffects {
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
                let pending_event =
                    self.pending_worker_events
                        .lock()
                        .take(&request_id, event_worker, decay_now);
                if pending_event == Some(PendingWorkerEvent::Free) {
                    return;
                }
                let Ok(attempt_id) = self.request_index.try_insert_request(
                    request_id.clone(),
                    event_worker,
                    lora_name,
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
                    self.apply_prefill_completed(&booking, decay_now, effects);
                }
            }
            ActiveSequenceEventData::Free => {
                let Some(current) = self.request_index.booking_for(&request_id) else {
                    if can_defer_worker_event {
                        self.pending_worker_events.lock().record(
                            request_id,
                            event_worker,
                            PendingWorkerEvent::Free,
                            decay_now,
                        );
                    }
                    return;
                };
                if current.worker != event_worker {
                    return;
                }
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
                    let Some(delta) = seq.free(&request_id, decay_now) else {
                        return;
                    };
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            current.worker,
                            delta,
                            load,
                        );
                    load
                };
                drop(table);
                self.request_index.remove_request_if_booking(
                    &request_id,
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
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let Some(current) = self.request_index.booking_for(&request_id) else {
                    if can_defer_worker_event {
                        self.pending_worker_events.lock().record(
                            request_id,
                            event_worker,
                            PendingWorkerEvent::MarkPrefillCompleted,
                            decay_now,
                        );
                    }
                    return;
                };
                if current.worker != event_worker {
                    return;
                }
                let booking = SchedulerBookingDescriptor {
                    request_id,
                    worker: current.worker,
                    attempt_id: current.attempt_id,
                };
                self.apply_prefill_completed(&booking, decay_now, effects);
            }
        }
    }

    fn apply_prefill_completed(
        &self,
        booking: &SchedulerBookingDescriptor,
        now: Instant,
        effects: &mut ReplicaBatchEffects,
    ) {
        let table = self.workers.read();
        let Some(&idx) = table.index.get(&booking.worker) else {
            return;
        };
        let (completion, load) = {
            let mut seq = table.slots[idx].sequences.write();
            let completion = seq.mark_prefill_completed(&booking.request_id, now);
            if completion == PrefillCompletion::Unchanged {
                return;
            }
            let load = seq.worker_load_snapshot();
            self.prompt_registry
                .replace_worker_load_state(booking.worker, load);
            (completion, load)
        };
        drop(table);
        effects.record_worker_load(booking.worker, load, false);
        if completion == PrefillCompletion::PhaseChanged {
            return;
        }
        if let Some(observer) = self.replica_request_lease_observer() {
            observer.progressed(booking);
        }
        effects.wake_scheduler = true;
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
