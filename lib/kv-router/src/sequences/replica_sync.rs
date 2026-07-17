// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use crate::active_sequence::TrackedSequenceHashes;
use crate::protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank};

const MAX_REPLICA_BATCH_EVENTS: usize = 256;
const MAX_REPLICA_BATCH_DURATION: Duration = Duration::from_millis(1);

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
        if event.router_id == self.router_id {
            return;
        }

        let received_stride = event.normalized_stride();
        if received_stride != Ok(self.active_sequence_stride) {
            if self
                .warned_replica_stride_mismatches
                .lock()
                .insert((event.router_id, received_stride))
            {
                let received_stride_label = received_stride
                    .map(|stride| stride.to_string())
                    .unwrap_or_else(|_| "malformed".to_string());
                tracing::warn!(
                    source_router_id = event.router_id,
                    request_id = %event.request_id,
                    expected_stride = %self.active_sequence_stride,
                    received_stride = %received_stride_label,
                    "Dropping active-sequence replica event with incompatible stride"
                );
            }
            self.publisher.observe_replica_stride_mismatch(
                self.active_sequence_stride,
                received_stride,
                self.worker_type(),
            );
            return;
        }

        let ActiveSequenceEvent {
            request_id,
            worker: event_worker,
            data,
            router_id,
            stride: _,
            lora_name,
        } = event;
        debug_assert_ne!(router_id, self.router_id);

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
                let token_sequence =
                    token_sequence.map(TrackedSequenceHashes::from_validated_event);
                if self.replica_worker_policy == ReplicaWorkerPolicy::LazyRegister {
                    self.ensure_worker_registered(event_worker);
                }
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&event_worker) else {
                    tracing::debug!(
                        worker = ?event_worker,
                        "Dropping replica AddRequest for unregistered worker"
                    );
                    return;
                };

                self.request_index
                    .set_request(request_id.clone(), event_worker, lora_name);
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
                effects.record_worker_load(event_worker, load, true);
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::Free => {
                let Some(worker) = self.request_index.remove_request(&request_id) else {
                    return;
                };
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&worker) else {
                    return;
                };
                let load = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let delta = seq.free(&request_id, decay_now);
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(worker, delta, load);
                    load
                };
                drop(table);
                effects.record_worker_load(worker, load, true);
                effects.wake_scheduler = true;
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let Some(worker) = self.request_index.worker_for(&request_id) else {
                    return;
                };
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&worker) else {
                    return;
                };
                let load = {
                    let mut seq = table.slots[idx].sequences.write();
                    seq.mark_prefill_completed(&request_id, decay_now);
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry.replace_worker_load_state(worker, load);
                    load
                };
                drop(table);
                effects.record_worker_load(worker, load, false);
                effects.wake_scheduler = true;
            }
        }
    }

    fn flush_replica_batch_effects(&self, effects: &mut ReplicaBatchEffects) {
        let decay_now = Instant::now();
        let mut active_loads = Vec::with_capacity(effects.worker_loads.len());
        for (worker, pending) in effects.worker_loads.drain() {
            if pending.publish {
                active_loads.push(self.observe_worker_load_snapshot(
                    worker,
                    pending.latest_load,
                    decay_now,
                ));
            }
        }
        if !active_loads.is_empty() {
            self.publisher.publish_load_batch(active_loads);
        }

        if std::mem::take(&mut effects.wake_scheduler) {
            self.notify_remote_state_update();
        }
        if std::mem::take(&mut effects.cleanup_prompt_trie) {
            self.prompt_registry.maybe_cleanup();
        }
    }
}
