// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use dashmap::DashMap;
use dynamo_kv_router::{
    indexer::WorkerKvQueryResponse, protocols::RouterEvent, recovery::CursorState,
};
use dynamo_runtime::component::{Component, Instance};
use rand::Rng;
use tokio::{
    sync::{Mutex, Semaphore, watch},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

use super::worker_query_state::{LiveEventAction, PendingDrainPlan, RankState, RecoveryKey};
use super::worker_query_transport::{RuntimeWorkerQueryTransport, WorkerQueryTransport};
use crate::{
    discovery::{KvEventSource, KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::Indexer,
};

const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;
const RECOVERY_CONCURRENCY_LIMIT: usize = 16;
#[cfg(test)]
const KV_EVENT_TOPIC: &str = dynamo_kv_router::protocols::KV_EVENT_SUBJECT;

#[derive(Debug)]
struct SourceBinding {
    source: KvEventSource,
    lifetime: CancellationToken,
}

struct RecoveryTask {
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

impl SourceBinding {
    fn recovery_target(&self) -> Option<&Instance> {
        self.source.recovery_target.as_ref()
    }
}

#[derive(Debug, Default)]
struct SourceSlot {
    active: Option<Arc<SourceBinding>>,
    rank: RankState,
    lifecycle_generation: u64,
    ever_activated: bool,
    was_ambiguous: bool,
    reset_pending: bool,
}

/// Coordinates KV recovery for sources advertised under one exact KV-state endpoint.
///
/// The discovery advertisement is the sole authority for the relationship between a logical
/// rank, its event publisher incarnation, and its optional callable recovery target. Runtime
/// configs only constrain which logical ranks are currently expected by the serving endpoint.
pub(crate) struct WorkerQueryClient {
    transport: Arc<dyn WorkerQueryTransport>,
    indexer: Indexer,
    membership_rx: watch::Receiver<KvSourceMembershipView>,
    _membership_guard: Option<KvSourceMembershipWatch>,
    membership_sync: Mutex<()>,
    slots: DashMap<RecoveryKey, Arc<Mutex<SourceSlot>>>,
    /// Immutable publisher binding lookup performed once per event envelope.
    publisher_bindings: DashMap<u64, Arc<SourceBinding>>,
    recovery_tasks: DashMap<RecoveryKey, RecoveryTask>,
    recovery_semaphore: Arc<Semaphore>,
    cancellation_token: CancellationToken,
}

impl WorkerQueryClient {
    pub(crate) async fn spawn(
        component: Component,
        indexer: Indexer,
        membership_watch: KvSourceMembershipWatch,
        cancellation_token: CancellationToken,
    ) -> Result<Arc<Self>> {
        let transport = Arc::new(RuntimeWorkerQueryTransport::new(&component).await?);
        let membership_rx = watch::Receiver::clone(&membership_watch);
        let client = Arc::new(Self {
            transport,
            indexer,
            membership_rx,
            _membership_guard: Some(membership_watch),
            membership_sync: Mutex::new(()),
            slots: DashMap::new(),
            publisher_bindings: DashMap::new(),
            recovery_tasks: DashMap::new(),
            recovery_semaphore: Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)),
            cancellation_token,
        });

        client.sync_membership().await;

        let background = client.clone();
        tokio::spawn(async move {
            background.clone().run_membership_loop().await;
        });
        Ok(client)
    }

    #[cfg(test)]
    fn new_for_test(
        indexer: Indexer,
        membership_rx: watch::Receiver<KvSourceMembershipView>,
        transport: Arc<dyn WorkerQueryTransport>,
    ) -> Arc<Self> {
        Arc::new(Self {
            transport,
            indexer,
            membership_rx,
            _membership_guard: None,
            membership_sync: Mutex::new(()),
            slots: DashMap::new(),
            publisher_bindings: DashMap::new(),
            recovery_tasks: DashMap::new(),
            recovery_semaphore: Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)),
            cancellation_token: CancellationToken::new(),
        })
    }

    async fn run_membership_loop(self: Arc<Self>) {
        let mut membership_rx = self.membership_rx.clone();

        loop {
            tokio::select! {
                biased;
                _ = self.cancellation_token.cancelled() => break,
                result = membership_rx.changed() => {
                    if result.is_err() {
                        tracing::error!("KV source membership watch closed unexpectedly");
                        break;
                    }
                    membership_rx.borrow_and_update();
                    self.sync_membership().await;
                }
            }
        }

        self.deactivate_all().await;
    }

    /// Apply the latest shared membership snapshot before the event subscriber consumes its
    /// corresponding scope. Re-reading after acquiring the lock prevents a delayed reconciler
    /// from applying an older watch value after a newer one.
    pub(crate) async fn sync_membership(self: &Arc<Self>) -> KvSourceMembershipView {
        let _sync = self.membership_sync.lock().await;
        let view = self.membership_rx.borrow().clone();
        self.reconcile_view(view.clone()).await;
        view
    }

    async fn reconcile_view(self: &Arc<Self>, view: KvSourceMembershipView) {
        let generations = view.lifecycle_generations;
        let mut expected: HashMap<RecoveryKey, (KvSourceStatus, u64)> = view
            .sources
            .into_iter()
            .map(|(worker, status)| {
                let generation = generations.get(&worker).copied().unwrap_or(0);
                ((worker.worker_id, worker.dp_rank), (status, generation))
            })
            .collect();
        let existing: Vec<_> = self.slots.iter().map(|entry| *entry.key()).collect();
        for key in existing {
            if !expected.contains_key(&key) {
                self.remove_unexpected_key(key).await;
            }
        }

        for (key, (status, generation)) in expected.drain() {
            self.reconcile_key(key, status, generation).await;
        }
    }

    async fn reconcile_key(
        self: &Arc<Self>,
        key: RecoveryKey,
        status: KvSourceStatus,
        lifecycle_generation: u64,
    ) {
        let slot = self
            .slots
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(SourceSlot::default())))
            .clone();
        let mut slot = slot.lock().await;

        let selected = status.active_source().cloned();
        let generation_changed = slot.lifecycle_generation != lifecycle_generation;
        if let (Some(active), Some(selected)) = (&slot.active, &selected)
            && active.source.publisher_id == selected.publisher_id
            && !generation_changed
            && !slot.reset_pending
        {
            return;
        }

        let had_active = self.deactivate_locked(key, &mut slot).await;
        match selected {
            None => {
                if slot.reset_pending || had_active || generation_changed {
                    if let Err(error) = self.reset_rank(key).await {
                        slot.reset_pending = true;
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear inactive KV source state; reset remains pending");
                        return;
                    }
                    slot.reset_pending = false;
                }
                slot.lifecycle_generation = lifecycle_generation;
                slot.rank = RankState::default();
                slot.was_ambiguous = matches!(status, KvSourceStatus::Ambiguous(_));
            }
            Some(source) => {
                if slot.reset_pending
                    || generation_changed
                    || slot.ever_activated
                    || slot.was_ambiguous
                {
                    if let Err(error) = self.reset_rank(key).await {
                        slot.reset_pending = true;
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = source.publisher_id, "KV source activation remains disabled because cold reset failed");
                        return;
                    }
                    slot.reset_pending = false;
                }
                slot.lifecycle_generation = lifecycle_generation;
                let binding = Arc::new(SourceBinding {
                    lifetime: self.cancellation_token.child_token(),
                    source,
                });
                if let Some(target) = binding.recovery_target() {
                    self.transport
                        .clear_instance_tombstone(&target.endpoint_instance_id())
                        .await;
                }
                slot.rank.activate(binding.recovery_target().is_some());
                slot.active = Some(binding.clone());
                slot.ever_activated = true;
                slot.was_ambiguous = false;
                self.publisher_bindings
                    .insert(binding.source.publisher_id, binding.clone());
                if binding.recovery_target().is_some() {
                    self.spawn_recovery(key, binding, None, None).await;
                } else {
                    tracing::warn!(
                        kv_state_endpoint = %binding.source.kv_state_endpoint,
                        worker_id = key.0,
                        dp_rank = key.1,
                        publisher_id = binding.source.publisher_id,
                        "KV source is live-only; serving and best-effort KV routing continue without recovery"
                    );
                }
            }
        }
    }

    async fn deactivate_locked(&self, key: RecoveryKey, slot: &mut SourceSlot) -> bool {
        let Some(binding) = slot.active.take() else {
            return false;
        };
        self.publisher_bindings
            .remove_if(&binding.source.publisher_id, |_, current| {
                Arc::ptr_eq(current, &binding)
            });
        binding.lifetime.cancel();
        self.cancel_recovery(key).await;
        if let Some(target) = binding.recovery_target() {
            let target = target.endpoint_instance_id();
            self.transport.cancel_instance_streams(&target).await;
        }
        true
    }

    async fn deactivate_all(self: &Arc<Self>) {
        let keys: Vec<_> = self.slots.iter().map(|entry| *entry.key()).collect();
        for key in keys {
            self.remove_unexpected_key(key).await;
        }
    }

    async fn remove_unexpected_key(&self, key: RecoveryKey) {
        let Some(slot_handle) = self.slots.get(&key).map(|entry| entry.clone()) else {
            return;
        };
        let mut slot = slot_handle.lock().await;
        let had_active = self.deactivate_locked(key, &mut slot).await;
        if slot.reset_pending || had_active {
            if let Err(error) = self.reset_rank(key).await {
                slot.reset_pending = true;
                tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear KV state for a worker removed from serving membership; retaining reset-pending slot");
                return;
            }
            slot.reset_pending = false;
        }
        drop(slot);
        self.slots
            .remove_if(&key, |_, current| Arc::ptr_eq(current, &slot_handle));
    }

    async fn reset_rank(&self, key: RecoveryKey) -> Result<()> {
        self.indexer
            .reset_worker_dp_rank_and_wait(key.0, key.1)
            .await
            .with_context(|| {
                format!(
                    "failed to reset KV state for worker {} dp_rank {}",
                    key.0, key.1
                )
            })
    }

    pub(crate) async fn shutdown(self: &Arc<Self>) {
        self.cancellation_token.cancel();
        self.deactivate_all().await;
    }

    /// Handle one event envelope after a single immutable publisher lookup.
    pub(crate) async fn handle_live_batch(
        self: &Arc<Self>,
        publisher_id: u64,
        events: Vec<RouterEvent>,
    ) {
        let Some(binding) = self
            .publisher_bindings
            .get(&publisher_id)
            .map(|entry| entry.clone())
        else {
            tracing::debug!(
                publisher_id,
                "Dropping KV event batch from an inactive or ambiguous source"
            );
            return;
        };
        let expected = binding.source.worker;
        if let Some(event) = events.iter().find(|event| {
            event.worker_id != expected.worker_id || event.event.dp_rank != expected.dp_rank
        }) {
            tracing::error!(
                publisher_id,
                expected_worker_id = expected.worker_id,
                expected_dp_rank = expected.dp_rank,
                event_worker_id = event.worker_id,
                event_dp_rank = event.event.dp_rank,
                "Dropping KV event batch whose payload disagrees with its source advertisement"
            );
            return;
        }

        if events.is_empty() {
            return;
        }
        let key = (expected.worker_id, expected.dp_rank);
        let Some(slot_handle) = self.slots.get(&key).map(|entry| entry.clone()) else {
            return;
        };
        let mut slot = slot_handle.lock().await;
        if !slot
            .active
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &binding))
            || slot.reset_pending
        {
            return;
        }

        for event in events {
            let clear_event = matches!(
                &event.event.data,
                dynamo_kv_router::protocols::KvCacheEventData::Cleared
            )
            .then(|| event.clone());
            let recoverable = binding.recovery_target().is_some();
            match slot.rank.observe_live_event(event, recoverable) {
                LiveEventAction::Ignore => {}
                LiveEventAction::Apply { event_id, event } => {
                    if let Err(error) = self.indexer.try_apply_event(event).await {
                        slot.reset_pending = true;
                        slot.rank.finish_failed_recovery();
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, event_id, "KV event queue rejected a live event; rank remains fenced pending reset");
                        return;
                    }
                    slot.rank.commit_live_event(event_id);
                }
                LiveEventAction::Clear { event_id } => {
                    drop(slot);
                    if let Err(error) = self
                        .apply_live_worker_clear(
                            key,
                            event_id,
                            binding.clone(),
                            clear_event.expect("clear action preserves the clear event"),
                        )
                        .await
                    {
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, event_id, "Failed to acknowledge live clear; affected ranks remain fenced");
                        return;
                    }
                    slot = slot_handle.lock().await;
                    if !slot
                        .active
                        .as_ref()
                        .is_some_and(|active| Arc::ptr_eq(active, &binding))
                    {
                        return;
                    }
                }
                LiveEventAction::Recover {
                    start_event_id,
                    end_event_id,
                    reset,
                } => {
                    if reset {
                        self.cancel_recovery(key).await;
                        if let Err(error) = self.reset_rank(key).await {
                            slot.reset_pending = true;
                            slot.rank.finish_failed_recovery();
                            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear KV state before gap recovery; rank remains fenced");
                            return;
                        }
                    }
                    self.spawn_recovery(key, binding.clone(), start_event_id, end_event_id)
                        .await
                }
                LiveEventAction::ResetDegraded { event } => {
                    self.cancel_recovery(key).await;
                    if let Err(error) = self.reset_rank(key).await {
                        slot.reset_pending = true;
                        slot.rank.finish_failed_recovery();
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear KV state after an event sequence gap; rank remains fenced");
                        return;
                    }
                    let event_id = event.event.event_id;
                    if let Err(error) = self.apply_events_and_flush([event]).await {
                        slot.reset_pending = true;
                        slot.rank.finish_failed_recovery();
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, event_id, "Failed to acknowledge degraded gap event; rank remains fenced");
                        return;
                    }
                    slot.rank.commit_live_event(event_id);
                }
            }
        }
    }

    async fn apply_live_worker_clear(
        &self,
        emitter: RecoveryKey,
        event_id: u64,
        binding: Arc<SourceBinding>,
        event: RouterEvent,
    ) -> Result<()> {
        let mut worker_slots: Vec<_> = self
            .slots
            .iter()
            .filter(|entry| entry.key().0 == emitter.0)
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        worker_slots.sort_by_key(|(key, _)| key.1);
        let mut guards = Vec::with_capacity(worker_slots.len());
        for (key, slot) in worker_slots {
            guards.push((key, slot.lock_owned().await));
        }
        let Some(emitter_slot) = guards
            .iter()
            .find(|(slot_key, _)| *slot_key == emitter)
            .map(|(_, slot)| slot)
        else {
            return Ok(());
        };
        if !emitter_slot
            .active
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return Ok(());
        }
        for (key, _) in &mut guards {
            self.cancel_recovery(*key).await;
        }
        if let Err(error) = self.apply_events_and_flush([event]).await {
            for (_, slot) in &mut guards {
                slot.reset_pending = true;
                slot.rank.finish_failed_recovery();
            }
            return Err(error);
        }
        for (key, slot) in &mut guards {
            slot.rank
                .apply_worker_clear_barrier(event_id, *key == emitter);
        }
        Ok(())
    }

    async fn apply_events_and_flush(
        &self,
        events: impl IntoIterator<Item = RouterEvent>,
    ) -> Result<()> {
        for event in events {
            self.indexer
                .try_apply_event(event)
                .await
                .context("KV indexer rejected event queue admission")?;
        }
        self.indexer
            .flush_and_wait()
            .await
            .context("KV indexer failed to acknowledge cold-path mutations")
    }

    async fn cancel_recovery(&self, key: RecoveryKey) {
        if let Some((_, task)) = self.recovery_tasks.remove(&key) {
            task.cancel.cancel();
            task.handle.abort();
            if let Err(error) = task.handle.await
                && !error.is_cancelled()
            {
                tracing::warn!(%error, worker_id = key.0, dp_rank = key.1, "KV recovery task failed while joining cancellation");
            }
        }
    }

    async fn spawn_recovery(
        self: &Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) {
        let Some(target) = binding.recovery_target().cloned() else {
            return;
        };
        self.cancel_recovery(key).await;
        if binding.lifetime.is_cancelled() {
            return;
        }
        self.launch_recovery(key, binding, target, start_event_id, end_event_id);
    }

    fn launch_recovery(
        self: &Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) {
        let cancel = binding.lifetime.child_token();
        let task_cancel = cancel.clone();
        let client = self.clone();
        let handle = tokio::spawn(async move {
            if start_event_id.is_none() {
                let jitter_us = rand::rng().random_range(0..3000u64);
                tokio::time::sleep(Duration::from_micros(jitter_us)).await;
            }
            let recovery = async {
                let _permit = client
                    .recovery_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .context("recovery semaphore closed")?;
                client
                    .fetch_recovery_response(key, target, start_event_id, end_event_id)
                    .await
            };
            let result = tokio::select! {
                biased;
                _ = task_cancel.cancelled() => return,
                result = recovery => result,
            };
            if task_cancel.is_cancelled() {
                return;
            }
            client
                .finish_recovery(key, binding, task_cancel, result)
                .await;
        });
        self.recovery_tasks
            .insert(key, RecoveryTask { cancel, handle });
    }

    fn schedule_recovery_after_current(
        self: &Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        start_event_id: u64,
    ) {
        let Some(target) = binding.recovery_target().cloned() else {
            return;
        };
        let client = self.clone();
        tokio::spawn(async move {
            let Some(slot) = client.slots.get(&key).map(|entry| entry.clone()) else {
                return;
            };
            let slot = slot.lock().await;
            if binding.lifetime.is_cancelled()
                || !slot.rank.recovery_inflight
                || !slot
                    .active
                    .as_ref()
                    .is_some_and(|active| Arc::ptr_eq(active, &binding))
            {
                return;
            }
            client.cancel_recovery(key).await;
            if binding.lifetime.is_cancelled() {
                return;
            }
            client.launch_recovery(key, binding, target, Some(start_event_id), None);
        });
    }

    async fn finish_recovery(
        self: Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        cancel: CancellationToken,
        result: Result<WorkerKvQueryResponse>,
    ) {
        if cancel.is_cancelled() {
            return;
        }
        let contains_clear = match &result {
            Ok(WorkerKvQueryResponse::Events { events, .. })
            | Ok(WorkerKvQueryResponse::TreeDump { events, .. }) => events.iter().any(|event| {
                matches!(
                    &event.event.data,
                    dynamo_kv_router::protocols::KvCacheEventData::Cleared
                )
            }),
            _ => false,
        };
        if contains_clear {
            match result {
                Ok(WorkerKvQueryResponse::Events {
                    events,
                    last_event_id,
                }) => {
                    self.finish_recovery_with_clear(
                        key,
                        binding,
                        cancel,
                        events,
                        last_event_id,
                        false,
                    )
                    .await;
                }
                Ok(WorkerKvQueryResponse::TreeDump {
                    events,
                    last_event_id,
                }) => {
                    self.finish_recovery_with_clear(
                        key,
                        binding,
                        cancel,
                        events,
                        last_event_id,
                        true,
                    )
                    .await;
                }
                _ => unreachable!("contains_clear only matches event-bearing recovery responses"),
            }
            return;
        }
        let Some(slot) = self.slots.get(&key).map(|entry| entry.clone()) else {
            return;
        };
        let mut slot = slot.lock().await;
        if cancel.is_cancelled()
            || !slot
                .active
                .as_ref()
                .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return;
        }

        let recovered_cursor = match result {
            Ok(WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            }) => {
                if !recovery_events_match_source(key, &events) {
                    tracing::error!(
                        worker_id = key.0,
                        dp_rank = key.1,
                        publisher_id = binding.source.publisher_id,
                        "Discarding recovery events for another logical source"
                    );
                    None
                } else {
                    if let Err(error) = self.apply_events_and_flush(events).await {
                        slot.reset_pending = true;
                        slot.rank.finish_failed_recovery();
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = binding.source.publisher_id, "Failed to acknowledge recovered events; rank remains fenced");
                        return;
                    }
                    Some(
                        slot.rank.cursor.advance_to(
                            slot.rank
                                .cursor
                                .last_applied_id()
                                .unwrap_or(0)
                                .max(last_event_id),
                        ),
                    )
                }
            }
            Ok(WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            }) => {
                if !recovery_events_match_source(key, &events) {
                    tracing::error!(
                        worker_id = key.0,
                        dp_rank = key.1,
                        publisher_id = binding.source.publisher_id,
                        "Discarding recovery tree dump for another logical source"
                    );
                    None
                } else if let Err(error) = self.reset_rank(key).await {
                    slot.reset_pending = true;
                    slot.rank.finish_failed_recovery();
                    tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to reset rank before applying recovery tree dump; rank remains fenced");
                    return;
                } else {
                    if let Err(error) = self.apply_events_and_flush(events).await {
                        slot.reset_pending = true;
                        slot.rank.finish_failed_recovery();
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = binding.source.publisher_id, "Failed to acknowledge recovery tree dump; rank remains fenced");
                        return;
                    }
                    Some(CursorState::Initial.advance_to(last_event_id))
                }
            }
            Ok(response) => {
                tracing::warn!(
                    worker_id = key.0,
                    dp_rank = key.1,
                    ?response,
                    "KV recovery returned no applicable state"
                );
                None
            }
            Err(error) => {
                tracing::warn!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = binding.source.publisher_id, "KV recovery failed; continuing with degraded live events");
                None
            }
        };

        let Some(recovered_cursor) = recovered_cursor else {
            self.finish_degraded_locked(key, &mut slot).await;
            return;
        };

        slot.rank.begin_successful_recovery_drain(recovered_cursor);
        let PendingDrainPlan {
            events,
            cursor,
            next_recovery_start,
        } = slot.rank.plan_pending_drain();
        if !events.is_empty()
            && let Err(error) = self.apply_events_and_flush(events).await
        {
            slot.reset_pending = true;
            slot.rank.finish_failed_recovery();
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to acknowledge buffered live events after recovery; rank remains fenced");
            return;
        }
        slot.rank.commit_pending_drain(cursor, next_recovery_start);
        drop(slot);
        if let Some(start_event_id) = next_recovery_start {
            self.schedule_recovery_after_current(key, binding, start_event_id);
        }
    }

    async fn finish_degraded_locked(&self, key: RecoveryKey, slot: &mut SourceSlot) {
        let pending = slot.rank.take_failed_recovery_degraded();
        let last_event_id = pending.last().map(|event| event.event.event_id);
        if !pending.is_empty()
            && let Err(error) = self.apply_events_and_flush(pending).await
        {
            slot.reset_pending = true;
            slot.rank.finish_failed_recovery();
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to acknowledge degraded live events; rank remains fenced");
            return;
        }
        slot.rank.commit_failed_recovery_degraded(last_event_id);
    }

    async fn finish_failed_recovery_degraded(
        &self,
        key: RecoveryKey,
        binding: &Arc<SourceBinding>,
        cancel: &CancellationToken,
    ) {
        let Some(slot) = self.slots.get(&key).map(|entry| entry.clone()) else {
            return;
        };
        let mut slot = slot.lock().await;
        if cancel.is_cancelled()
            || !slot
                .active
                .as_ref()
                .is_some_and(|active| Arc::ptr_eq(active, binding))
        {
            return;
        }
        self.finish_degraded_locked(key, &mut slot).await;
    }

    async fn finish_recovery_with_clear(
        self: &Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        cancel: CancellationToken,
        events: Vec<RouterEvent>,
        last_event_id: u64,
        tree_dump: bool,
    ) {
        if !recovery_events_match_source(key, &events) {
            tracing::error!(
                worker_id = key.0,
                dp_rank = key.1,
                "Discarding recovery batch containing a foreign clear barrier"
            );
            self.finish_failed_recovery_degraded(key, &binding, &cancel)
                .await;
            return;
        }

        let mut worker_slots: Vec<_> = self
            .slots
            .iter()
            .filter(|entry| entry.key().0 == key.0)
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        worker_slots.sort_by_key(|(key, _)| key.1);
        let mut guards = Vec::with_capacity(worker_slots.len());
        for (slot_key, slot) in worker_slots {
            guards.push((slot_key, slot.lock_owned().await));
        }
        let Some(emitter_index) = guards.iter().position(|(slot_key, _)| *slot_key == key) else {
            return;
        };
        if cancel.is_cancelled()
            || !guards[emitter_index]
                .1
                .active
                .as_ref()
                .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return;
        }

        if tree_dump && let Err(error) = self.reset_rank(key).await {
            guards[emitter_index].1.reset_pending = true;
            guards[emitter_index].1.rank.finish_failed_recovery();
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to reset rank before recovery tree dump; rank remains fenced");
            return;
        }

        let mut cursor = if tree_dump {
            CursorState::Initial
        } else {
            guards[emitter_index].1.rank.cursor
        };
        let mut clear_event_ids = Vec::new();
        for event in &events {
            let event_id = event.event.event_id;
            if matches!(
                &event.event.data,
                dynamo_kv_router::protocols::KvCacheEventData::Cleared
            ) {
                for (slot_key, _) in &mut guards {
                    if *slot_key != key {
                        self.cancel_recovery(*slot_key).await;
                    }
                }
                clear_event_ids.push(event_id);
                cursor = cursor.apply_barrier(event_id);
            } else {
                cursor = cursor.advance_to(event_id);
            }
        }
        if let Err(error) = self.apply_events_and_flush(events).await {
            for (_, slot) in &mut guards {
                slot.reset_pending = true;
                slot.rank.finish_failed_recovery();
            }
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to acknowledge recovery batch containing clear; affected ranks remain fenced");
            return;
        }
        if cancel.is_cancelled()
            || !guards[emitter_index]
                .1
                .active
                .as_ref()
                .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return;
        }
        for event_id in clear_event_ids {
            for (slot_key, slot) in &mut guards {
                slot.rank
                    .apply_worker_clear_barrier(event_id, *slot_key == key);
            }
        }
        guards[emitter_index]
            .1
            .rank
            .begin_successful_recovery_drain(cursor.advance_to(last_event_id));
        let PendingDrainPlan {
            events,
            cursor,
            next_recovery_start,
        } = guards[emitter_index].1.rank.plan_pending_drain();
        if !events.is_empty()
            && let Err(error) = self.apply_events_and_flush(events).await
        {
            guards[emitter_index].1.reset_pending = true;
            guards[emitter_index].1.rank.finish_failed_recovery();
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to acknowledge buffered events after clear recovery; rank remains fenced");
            return;
        }
        guards[emitter_index]
            .1
            .rank
            .commit_pending_drain(cursor, next_recovery_start);
        drop(guards);
        if let Some(start_event_id) = next_recovery_start {
            self.schedule_recovery_after_current(key, binding, start_event_id);
        }
    }

    async fn fetch_recovery_response(
        &self,
        key: RecoveryKey,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let mut last_error = None;
        for attempt in 0..RECOVERY_MAX_RETRIES {
            match self
                .transport
                .query_worker(key.0, key.1, target.clone(), start_event_id, end_event_id)
                .await
            {
                Ok(response) => return Ok(response),
                Err(error) => {
                    last_error = Some(error);
                    if attempt + 1 < RECOVERY_MAX_RETRIES {
                        let backoff_ms = RECOVERY_INITIAL_BACKOFF_MS * 2_u64.pow(attempt);
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("KV recovery returned no response")))
    }
}

fn recovery_events_match_source(key: RecoveryKey, events: &[RouterEvent]) -> bool {
    events
        .iter()
        .all(|event| event.worker_id == key.0 && event.event.dp_rank == key.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dynamo_kv_router::{
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, WorkerKvQueryRequest},
        protocols::{
            DpRank, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
            KvCacheStoredBlockData, LocalBlockHash, WorkerId, WorkerWithDpRank,
        },
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::TransportType,
        discovery::{
            Discovery, DiscoveryInstance, DiscoverySpec, EventTransportKind, MockDiscovery,
            SharedMockRegistry,
        },
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
            network::Ingress,
        },
        protocols::EndpointId,
        storage::kv::Selector,
        stream,
        transports::event_plane::{EventPublisher, EventScope},
    };
    use std::{
        collections::HashSet,
        path::Path,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use tokio::sync::{Notify, watch};

    use crate::{
        discovery::{
            KvSourceAmbiguity, KvSourceMembershipCoordinator, KvSourceMembershipView,
            KvStateEndpointResolution, runtime_config_watch,
        },
        kv_router::indexer::LowerTierIndexers,
        local_model::runtime_config::ModelRuntimeConfig,
        model_card::ModelDeploymentCard,
    };

    #[derive(Default)]
    struct MockTransport {
        responses: Mutex<Vec<WorkerKvQueryResponse>>,
        release: Mutex<Option<Arc<Notify>>>,
    }

    #[async_trait]
    impl WorkerQueryTransport for MockTransport {
        async fn query_worker(
            &self,
            _worker_id: WorkerId,
            _dp_rank: DpRank,
            _target: Instance,
            _start_event_id: Option<u64>,
            _end_event_id: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            if let Some(release) = self.release.lock().await.clone() {
                release.notified().await;
            }
            self.responses
                .lock()
                .await
                .pop()
                .context("missing mock recovery response")
        }
    }

    #[derive(Default)]
    struct OrderedCancellationTransport {
        query_started: Notify,
        query_dropped: AtomicBool,
        cancel_started: Notify,
        cancel_release: Notify,
        tombstone_clears: AtomicUsize,
    }

    struct QueryDropFlag<'a>(&'a AtomicBool);

    impl Drop for QueryDropFlag<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl WorkerQueryTransport for OrderedCancellationTransport {
        async fn query_worker(
            &self,
            _worker_id: WorkerId,
            _dp_rank: DpRank,
            _target: Instance,
            _start_event_id: Option<u64>,
            _end_event_id: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            let _drop_flag = QueryDropFlag(&self.query_dropped);
            self.query_started.notify_one();
            std::future::pending().await
        }

        async fn cancel_instance_streams(
            &self,
            _endpoint_id: &dynamo_runtime::discovery::EndpointInstanceId,
        ) -> usize {
            assert!(
                self.query_dropped.load(Ordering::SeqCst),
                "old recovery task must be joined before transport cancellation"
            );
            self.cancel_started.notify_one();
            self.cancel_release.notified().await;
            1
        }

        async fn clear_instance_tombstone(
            &self,
            _endpoint_id: &dynamo_runtime::discovery::EndpointInstanceId,
        ) {
            self.tombstone_clears.fetch_add(1, Ordering::SeqCst);
        }
    }

    async fn shared_drt(store_path: &Path) -> DistributedRuntime {
        DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig {
                discovery_backend: DiscoveryBackend::KvStore(Selector::File(
                    store_path.to_path_buf(),
                )),
                nats_config: None,
                request_plane: RequestPlaneMode::Tcp,
                event_transport_kind: EventTransportKind::Zmq,
            },
        )
        .await
        .unwrap()
    }

    fn shared_component(drt: &DistributedRuntime, namespace: &str) -> Component {
        drt.namespace(namespace)
            .unwrap()
            .component("router")
            .unwrap()
    }

    fn indexer() -> (KvIndexer, Indexer) {
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let indexer = KvIndexer::new(CancellationToken::new(), 4, metrics);
        (
            indexer.clone(),
            Indexer::KvIndexer {
                primary: indexer,
                lower_tier: LowerTierIndexers::new(1, 4),
                approx: None,
                primary_records_routing_decisions: false,
            },
        )
    }

    fn source_for(
        endpoint: &EndpointId,
        worker: WorkerWithDpRank,
        publisher_id: u64,
        recovery_target: Option<Instance>,
    ) -> KvEventSource {
        KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker,
            publisher_id,
            recovery_target,
        }
    }

    fn source(endpoint: &EndpointId, publisher_id: u64) -> KvEventSource {
        source_for(
            endpoint,
            WorkerWithDpRank::new(42, 4),
            publisher_id,
            Some(Instance {
                namespace: endpoint.namespace.clone(),
                component: endpoint.component.clone(),
                endpoint: format!("query-{publisher_id}"),
                instance_id: publisher_id,
                transport: TransportType::Nats(String::new()),
                device_type: None,
            }),
        )
    }

    fn store(event_id: u64) -> RouterEvent {
        RouterEvent::new(
            42,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(event_id),
                        tokens_hash: LocalBlockHash(event_id),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 4,
            },
        )
    }

    fn clear_for(worker: WorkerWithDpRank, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: worker.dp_rank,
            },
        )
    }

    #[tokio::test]
    async fn exact_removal_and_stale_recovery_are_fenced_by_publisher() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let (kv_indexer, indexer) = indexer();
        let transport = Arc::new(MockTransport::default());
        let worker = WorkerWithDpRank::new(42, 4);
        let old = source(&kv_endpoint, 100);
        let new = source_for(&kv_endpoint, worker, 205, None);
        let initial = membership_view(
            &serving,
            &kv_endpoint,
            [(worker, KvSourceStatus::ActiveRecoverable(old.clone()), 0)],
        );
        let (_tx, rx) = watch::channel(initial.clone());
        let client = WorkerQueryClient::new_for_test(indexer, rx, transport.clone());
        let release = Arc::new(Notify::new());
        *transport.release.lock().await = Some(release.clone());
        transport
            .responses
            .lock()
            .await
            .push(WorkerKvQueryResponse::TreeDump {
                events: vec![store(100)],
                last_event_id: 100,
            });

        client.reconcile_view(initial).await;
        let old_binding = client
            .publisher_bindings
            .get(&100)
            .expect("source A should be active")
            .clone();
        client
            .reconcile_view(membership_view(
                &serving,
                &kv_endpoint,
                [(
                    worker,
                    KvSourceStatus::Ambiguous(KvSourceAmbiguity::Incarnations {
                        publisher_ids: vec![100, 205],
                    }),
                    1,
                )],
            ))
            .await;
        assert!(!client.publisher_bindings.contains_key(&100));
        assert!(!client.publisher_bindings.contains_key(&205));

        client
            .reconcile_view(membership_view(
                &serving,
                &kv_endpoint,
                [(worker, KvSourceStatus::ActiveLiveOnly(new), 2)],
            ))
            .await;
        assert!(!client.publisher_bindings.contains_key(&100));
        assert!(client.publisher_bindings.contains_key(&205));

        client
            .clone()
            .finish_recovery(
                (worker.worker_id, worker.dp_rank),
                old_binding,
                CancellationToken::new(),
                Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![store(100)],
                    last_event_id: 100,
                }),
            )
            .await;
        release.notify_waiters();
        client.handle_live_batch(100, vec![store(101)]).await;
        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(events.iter().all(|event| event.event.event_id != 100));
        assert!(events.iter().all(|event| event.event.event_id != 101));
    }

    #[tokio::test]
    async fn coalesced_overlap_resets_even_when_the_same_publisher_remains() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let source = source_for(&kv_endpoint, worker, 100, None);
        let initial = membership_view(
            &serving,
            &kv_endpoint,
            [(worker, KvSourceStatus::ActiveLiveOnly(source.clone()), 0)],
        );
        let (_tx, rx) = watch::channel(initial.clone());
        let (kv_indexer, indexer) = indexer();
        let client =
            WorkerQueryClient::new_for_test(indexer, rx, Arc::new(MockTransport::default()));

        client.reconcile_view(initial).await;
        client
            .handle_live_batch(100, vec![store_for(worker, 1)])
            .await;
        kv_indexer.flush().await;
        assert!(contains_block(&kv_indexer.dump_events().await.unwrap(), 1));

        // The watch may coalesce A -> ambiguous(A, B) -> A. The cumulative lifecycle
        // generation still requires the consumer to fence and cold-reset A.
        client
            .reconcile_view(membership_view(
                &serving,
                &kv_endpoint,
                [(worker, KvSourceStatus::ActiveLiveOnly(source), 2)],
            ))
            .await;
        kv_indexer.flush().await;
        assert!(kv_indexer.dump_events().await.unwrap().is_empty());
        assert!(client.publisher_bindings.contains_key(&100));
    }

    #[tokio::test]
    async fn failed_reset_retains_generation_fence_and_slot() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let initial = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveLiveOnly(source_for(&kv_endpoint, worker, 100, None)),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(initial.clone());
        let (kv_indexer, indexer) = indexer();
        let client =
            WorkerQueryClient::new_for_test(indexer, rx, Arc::new(MockTransport::default()));
        client.reconcile_view(initial).await;

        kv_indexer.shutdown();
        kv_indexer.event_sender().closed().await;
        let replacement = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveLiveOnly(source_for(&kv_endpoint, worker, 205, None)),
                1,
            )],
        );
        client.reconcile_view(replacement.clone()).await;
        client.reconcile_view(replacement).await;

        let key = (worker.worker_id, worker.dp_rank);
        let slot = client.slots.get(&key).unwrap().clone();
        let slot = slot.lock().await;
        assert!(slot.active.is_none());
        assert!(slot.reset_pending);
        assert_eq!(slot.lifecycle_generation, 0);
        drop(slot);
        assert!(!client.publisher_bindings.contains_key(&205));

        client
            .reconcile_view(membership_view(&serving, &kv_endpoint, std::iter::empty()))
            .await;
        assert!(client.slots.contains_key(&key));
        assert!(client.slots.get(&key).unwrap().lock().await.reset_pending);
    }

    #[tokio::test]
    async fn live_enqueue_failure_does_not_advance_cursor_and_fences_rank() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveLiveOnly(source_for(&kv_endpoint, worker, 100, None)),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let (kv_indexer, indexer) = indexer();
        let client =
            WorkerQueryClient::new_for_test(indexer, rx, Arc::new(MockTransport::default()));
        client.reconcile_view(view).await;
        kv_indexer.shutdown();
        kv_indexer.event_sender().closed().await;

        client.handle_live_batch(100, vec![store(1)]).await;

        let slot = client
            .slots
            .get(&(worker.worker_id, worker.dp_rank))
            .unwrap()
            .clone();
        let slot = slot.lock().await;
        assert_eq!(slot.rank.last_applied_id(), None);
        assert!(slot.reset_pending);
    }

    #[tokio::test]
    async fn replacement_waits_for_old_target_stream_cancellation() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let shared_target = source(&kv_endpoint, 100).recovery_target.unwrap();
        let initial = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveRecoverable(source_for(
                    &kv_endpoint,
                    worker,
                    100,
                    Some(shared_target.clone()),
                )),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(initial.clone());
        let (_, indexer) = indexer();
        let transport = Arc::new(OrderedCancellationTransport::default());
        let client = WorkerQueryClient::new_for_test(indexer, rx, transport.clone());
        client.reconcile_view(initial).await;
        assert_eq!(transport.tombstone_clears.load(Ordering::SeqCst), 1);
        transport.query_started.notified().await;

        let replacement = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveRecoverable(source_for(
                    &kv_endpoint,
                    worker,
                    205,
                    Some(shared_target),
                )),
                1,
            )],
        );
        let reconcile = tokio::spawn({
            let client = client.clone();
            async move { client.reconcile_view(replacement).await }
        });
        transport.cancel_started.notified().await;
        assert_eq!(transport.tombstone_clears.load(Ordering::SeqCst), 1);
        transport.cancel_release.notify_waiters();
        reconcile.await.unwrap();
        assert_eq!(transport.tombstone_clears.load(Ordering::SeqCst), 2);
        assert!(client.publisher_bindings.contains_key(&205));
    }

    #[tokio::test]
    async fn foreign_clear_recovery_failure_drains_buffered_live_events() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveRecoverable(source(&kv_endpoint, 100)),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let (kv_indexer, indexer) = indexer();
        let transport = Arc::new(MockTransport::default());
        *transport.release.lock().await = Some(Arc::new(Notify::new()));
        let client = WorkerQueryClient::new_for_test(indexer, rx, transport);
        client.reconcile_view(view).await;
        client.handle_live_batch(100, vec![store(1)]).await;
        let binding = client.publisher_bindings.get(&100).unwrap().clone();

        client
            .clone()
            .finish_recovery(
                (worker.worker_id, worker.dp_rank),
                binding,
                CancellationToken::new(),
                Ok(WorkerKvQueryResponse::Events {
                    events: vec![clear_for(WorkerWithDpRank::new(99, 4), 2)],
                    last_event_id: 2,
                }),
            )
            .await;

        let events = kv_indexer.dump_events().await.unwrap();
        assert!(contains_block(&events, 1));
        let slot = client
            .slots
            .get(&(worker.worker_id, worker.dp_rank))
            .unwrap()
            .clone();
        assert!(!slot.lock().await.rank.recovery_inflight);
    }

    #[tokio::test]
    async fn clear_tree_dump_reset_failure_keeps_rank_fenced() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveRecoverable(source(&kv_endpoint, 100)),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let (kv_indexer, indexer) = indexer();
        let transport = Arc::new(MockTransport::default());
        *transport.release.lock().await = Some(Arc::new(Notify::new()));
        let client = WorkerQueryClient::new_for_test(indexer, rx, transport);
        client.reconcile_view(view).await;
        client.handle_live_batch(100, vec![store(1)]).await;
        let binding = client.publisher_bindings.get(&100).unwrap().clone();
        kv_indexer.shutdown();
        kv_indexer.event_sender().closed().await;

        client
            .clone()
            .finish_recovery(
                (worker.worker_id, worker.dp_rank),
                binding,
                CancellationToken::new(),
                Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![clear_for(worker, 2)],
                    last_event_id: 2,
                }),
            )
            .await;

        let slot = client
            .slots
            .get(&(worker.worker_id, worker.dp_rank))
            .unwrap()
            .clone();
        let slot = slot.lock().await;
        assert!(!slot.rank.recovery_inflight);
        assert_eq!(slot.rank.last_applied_id(), None);
        assert!(slot.reset_pending);
    }

    #[tokio::test]
    async fn gap_recovery_resets_before_full_snapshot_and_ordered_live_drain() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [(
                worker,
                KvSourceStatus::ActiveRecoverable(source(&kv_endpoint, 100)),
                0,
            )],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let (kv_indexer, indexer) = indexer();
        let transport = Arc::new(MockTransport::default());
        *transport.release.lock().await = Some(Arc::new(Notify::new()));
        let client = WorkerQueryClient::new_for_test(indexer, rx, transport);
        client.reconcile_view(view).await;
        let binding = client.publisher_bindings.get(&100).unwrap().clone();

        client.handle_live_batch(100, vec![store(1)]).await;
        client
            .clone()
            .finish_recovery(
                (worker.worker_id, worker.dp_rank),
                binding.clone(),
                CancellationToken::new(),
                Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![store(1)],
                    last_event_id: 1,
                }),
            )
            .await;
        assert!(contains_block(&kv_indexer.dump_events().await.unwrap(), 1));

        client.handle_live_batch(100, vec![store(3)]).await;
        client.handle_live_batch(100, vec![store(4)]).await;
        let slot = client
            .slots
            .get(&(worker.worker_id, worker.dp_rank))
            .unwrap()
            .clone();
        assert_eq!(slot.lock().await.rank.last_applied_id(), Some(1));
        assert!(kv_indexer.dump_events().await.unwrap().is_empty());

        client
            .clone()
            .finish_recovery(
                (worker.worker_id, worker.dp_rank),
                binding,
                CancellationToken::new(),
                Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![store(1), store(2)],
                    last_event_id: 2,
                }),
            )
            .await;

        let events = kv_indexer.dump_events().await.unwrap();
        for block in 1..=4 {
            assert!(contains_block(&events, block));
        }
        let slot = slot.lock().await;
        assert_eq!(slot.rank.last_applied_id(), Some(4));
        assert!(!slot.rank.recovery_inflight);
        drop(slot);
        client.shutdown().await;
    }

    #[tokio::test]
    async fn foreign_event_rejects_the_entire_envelope_before_index_mutation() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let source = source_for(&kv_endpoint, worker, 100, None);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [(worker, KvSourceStatus::ActiveLiveOnly(source), 0)],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let (kv_indexer, indexer) = indexer();
        let client =
            WorkerQueryClient::new_for_test(indexer, rx, Arc::new(MockTransport::default()));
        client.reconcile_view(view).await;

        let foreign = store_for(WorkerWithDpRank::new(99, 4), 2);
        client
            .handle_live_batch(100, vec![store_for(worker, 1), foreign])
            .await;
        kv_indexer.flush().await;

        assert!(kv_indexer.dump_events().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn live_clear_preserves_worker_wide_barrier_semantics() {
        let serving = EndpointId::from("test.router.generate");
        let kv_endpoint = EndpointId::from("test.router.kv");
        let (kv_indexer, indexer) = indexer();
        let rank_4 = WorkerWithDpRank::new(42, 4);
        let rank_5 = WorkerWithDpRank::new(42, 5);
        let source_4 = source_for(&kv_endpoint, rank_4, 100, None);
        let source_5 = source_for(&kv_endpoint, rank_5, 205, None);
        let view = membership_view(
            &serving,
            &kv_endpoint,
            [
                (rank_4, KvSourceStatus::ActiveLiveOnly(source_4), 0),
                (rank_5, KvSourceStatus::ActiveLiveOnly(source_5), 0),
            ],
        );
        let (_tx, rx) = watch::channel(view.clone());
        let client =
            WorkerQueryClient::new_for_test(indexer, rx, Arc::new(MockTransport::default()));
        client.reconcile_view(view).await;

        client
            .handle_live_batch(100, vec![store_for(rank_4, 1)])
            .await;
        client
            .handle_live_batch(205, vec![store_for(rank_5, 1)])
            .await;
        kv_indexer.flush().await;
        assert_eq!(kv_indexer.dump_events().await.unwrap().len(), 2);

        client
            .handle_live_batch(100, vec![clear_for(rank_4, 2)])
            .await;
        kv_indexer.flush().await;
        assert!(kv_indexer.dump_events().await.unwrap().is_empty());
    }

    struct ControlledRecoveryEngine {
        worker: WorkerWithDpRank,
        calls: AtomicUsize,
        delayed_started: Notify,
        delayed_release: Notify,
        delayed_finished: Notify,
    }

    struct NotifyOnDrop<'a>(&'a Notify);

    impl Drop for NotifyOnDrop<'_> {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<WorkerKvQueryRequest>, ManyOut<WorkerKvQueryResponse>, anyhow::Error>
        for ControlledRecoveryEngine
    {
        async fn generate(
            &self,
            request: SingleIn<WorkerKvQueryRequest>,
        ) -> Result<ManyOut<WorkerKvQueryResponse>> {
            let (request, context) = request.into_parts();
            assert_eq!(request.worker_id, self.worker.worker_id);
            assert_eq!(request.dp_rank, self.worker.dp_rank);
            let response = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                WorkerKvQueryResponse::TreeDump {
                    events: Vec::new(),
                    last_event_id: 0,
                }
            } else {
                let _finished = NotifyOnDrop(&self.delayed_finished);
                self.delayed_started.notify_waiters();
                self.delayed_release.notified().await;
                WorkerKvQueryResponse::Events {
                    events: vec![store_for(self.worker, 2)],
                    last_event_id: 2,
                }
            };
            Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![response])),
                context.context(),
            ))
        }
    }

    fn store_for(worker: WorkerWithDpRank, event_id: u64) -> RouterEvent {
        let mut event = store(event_id);
        event.worker_id = worker.worker_id;
        event.event.dp_rank = worker.dp_rank;
        event
    }

    fn store_block_for(worker: WorkerWithDpRank, event_id: u64, block_hash: u64) -> RouterEvent {
        let mut event = store_for(worker, event_id);
        let KvCacheEventData::Stored(data) = &mut event.event.data else {
            unreachable!("store_for always returns a stored event");
        };
        data.blocks[0].block_hash = ExternalSequenceBlockHash(block_hash);
        data.blocks[0].tokens_hash = LocalBlockHash(block_hash);
        event
    }

    fn contains_block(events: &[RouterEvent], block: u64) -> bool {
        events.iter().any(|event| match &event.event.data {
            KvCacheEventData::Stored(data) => data
                .blocks
                .iter()
                .any(|stored| stored.block_hash == ExternalSequenceBlockHash(block)),
            _ => false,
        })
    }

    fn contains_rank_block(events: &[RouterEvent], worker: WorkerWithDpRank, block: u64) -> bool {
        events.iter().any(|event| {
            event.worker_id == worker.worker_id
                && event.event.dp_rank == worker.dp_rank
                && match &event.event.data {
                    KvCacheEventData::Stored(data) => data
                        .blocks
                        .iter()
                        .any(|stored| stored.block_hash == ExternalSequenceBlockHash(block)),
                    _ => false,
                }
        })
    }

    struct RegisteredTestSource {
        worker: WorkerWithDpRank,
        publisher: EventPublisher,
        instance: DiscoveryInstance,
    }

    struct TestSourcePublisher {
        worker: WorkerWithDpRank,
        publisher: EventPublisher,
    }

    async fn create_test_source_publisher(
        drt: &DistributedRuntime,
        kv_endpoint: &EndpointId,
        worker: WorkerWithDpRank,
    ) -> TestSourcePublisher {
        let publisher = EventPublisher::for_endpoint_id_with_transport(
            drt,
            kv_endpoint,
            KV_EVENT_TOPIC,
            EventTransportKind::Zmq,
        )
        .await
        .unwrap();
        TestSourcePublisher { worker, publisher }
    }

    async fn advertise_test_source(
        discovery: &dyn Discovery,
        kv_endpoint: &EndpointId,
        source_publisher: TestSourcePublisher,
        recovery_target: Option<Instance>,
    ) -> RegisteredTestSource {
        let TestSourcePublisher { worker, publisher } = source_publisher;
        let source = source_for(
            kv_endpoint,
            worker,
            publisher.publisher_id(),
            recovery_target,
        );
        let instance = discovery
            .register(DiscoverySpec::EventSource {
                scope: EventScope::Endpoint {
                    endpoint: kv_endpoint.clone(),
                },
                topic: KV_EVENT_TOPIC.to_string(),
                publisher_id: source.publisher_id,
                metadata: serde_json::to_value(&source).unwrap(),
            })
            .await
            .unwrap();
        RegisteredTestSource {
            worker,
            publisher,
            instance,
        }
    }

    async fn register_test_source(
        source_drt: &DistributedRuntime,
        discovery: &dyn Discovery,
        kv_endpoint: &EndpointId,
        worker: WorkerWithDpRank,
        recovery_target: Option<Instance>,
    ) -> RegisteredTestSource {
        let publisher = create_test_source_publisher(source_drt, kv_endpoint, worker).await;
        advertise_test_source(discovery, kv_endpoint, publisher, recovery_target).await
    }

    async fn publish_rank_blocks(sources: &[RegisteredTestSource], event_id: u64, block_base: u64) {
        for source in sources {
            source
                .publisher
                .publish(&vec![store_block_for(
                    source.worker,
                    event_id,
                    block_base + u64::from(source.worker.dp_rank),
                )])
                .await
                .unwrap();
        }
    }

    async fn wait_for_index_state(
        kv_indexer: &KvIndexer,
        predicate: impl Fn(&[RouterEvent]) -> bool,
        failure: &'static str,
    ) -> Vec<RouterEvent> {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                kv_indexer.flush().await;
                let events = kv_indexer.dump_events().await.unwrap();
                if predicate(&events) {
                    return events;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect(failure)
    }

    fn membership_view(
        serving_endpoint: &EndpointId,
        kv_state_endpoint: &EndpointId,
        sources: impl IntoIterator<Item = (WorkerWithDpRank, KvSourceStatus, u64)>,
    ) -> KvSourceMembershipView {
        let mut statuses = HashMap::new();
        let mut generations = HashMap::new();
        for (worker, status, generation) in sources {
            statuses.insert(worker, status);
            generations.insert(worker, generation);
        }
        KvSourceMembershipView {
            serving_endpoint: serving_endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(kv_state_endpoint.clone()),
            sources: statuses,
            lifecycle_generations: generations,
            recovery_expected: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn direct_zmq_multi_node_replacement_isolated_by_global_rank() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let store = tempfile::tempdir().unwrap();
            let frontend_drt = shared_drt(store.path()).await;
            let leader_drt = shared_drt(store.path()).await;
            let node_0_drt = shared_drt(store.path()).await;
            let old_node_1_drt = shared_drt(store.path()).await;
            let namespace = "test-direct-zmq-multi-node";
            let frontend = shared_component(&frontend_drt, namespace);
            let old_node_1 = shared_component(&old_node_1_drt, namespace);
            let serving = frontend.endpoint("generate");
            let serving_id = serving.id();
            let kv_endpoint = EndpointId {
                namespace: namespace.to_string(),
                component: "router".to_string(),
                name: "kv-state".to_string(),
            };
            let logical_worker_id = leader_drt.connection_id();
            let source_discovery: Arc<dyn Discovery> = Arc::new(MockDiscovery::new(
                Some(frontend_drt.connection_id()),
                SharedMockRegistry::new(),
            ));
            assert_eq!(
                HashSet::from([
                    frontend_drt.connection_id(),
                    logical_worker_id,
                    node_0_drt.connection_id(),
                    old_node_1_drt.connection_id(),
                ])
                .len(),
                4
            );

            let discovery = leader_drt.discovery();
            let serving_instance = discovery
                .register(DiscoverySpec::Endpoint {
                    namespace: serving_id.namespace.clone(),
                    component: serving_id.component.clone(),
                    endpoint: serving_id.name.clone(),
                    transport: TransportType::Tcp("tcp://127.0.0.1:1".to_string()),
                    device_type: None,
                })
                .await
                .unwrap();
            let mut card = ModelDeploymentCard::with_name_only("test-model");
            card.runtime_config = ModelRuntimeConfig {
                data_parallel_start_rank: 0,
                data_parallel_size: 8,
                enable_local_indexer: true,
                kv_state_endpoint: Some(kv_endpoint.clone()),
                ..Default::default()
            };
            let model_instance = discovery
                .register(
                    DiscoverySpec::from_model(
                        serving_id.namespace.clone(),
                        serving_id.component.clone(),
                        serving_id.name.clone(),
                        &card,
                    )
                    .unwrap(),
                )
                .await
                .unwrap();
            let mut configs = runtime_config_watch(&serving).await.unwrap();
            configs
                .wait_for(|configs| configs.contains_key(&logical_worker_id))
                .await
                .unwrap();

            let delayed_rank = WorkerWithDpRank::new(logical_worker_id, 4);
            let recovery_engine = Arc::new(ControlledRecoveryEngine {
                worker: delayed_rank,
                calls: AtomicUsize::new(0),
                delayed_started: Notify::new(),
                delayed_release: Notify::new(),
                delayed_finished: Notify::new(),
            });
            let recovery_endpoint = old_node_1
                .endpoint("controlled-kv-recovery")
                .endpoint_builder()
                .handler(Ingress::for_engine(recovery_engine.clone()).unwrap())
                .start_with_registration()
                .await
                .unwrap();

            let mut node_0_sources = Vec::new();
            for dp_rank in 0..4 {
                node_0_sources.push(
                    register_test_source(
                        &node_0_drt,
                        source_discovery.as_ref(),
                        &kv_endpoint,
                        WorkerWithDpRank::new(logical_worker_id, dp_rank),
                        None,
                    )
                    .await,
                );
            }
            let mut old_node_1_sources = Vec::new();
            for dp_rank in 4..8 {
                let worker = WorkerWithDpRank::new(logical_worker_id, dp_rank);
                let recovery_target =
                    (worker == delayed_rank).then(|| recovery_endpoint.instance().clone());
                old_node_1_sources.push(
                    register_test_source(
                        &old_node_1_drt,
                        source_discovery.as_ref(),
                        &kv_endpoint,
                        worker,
                        recovery_target,
                    )
                    .await,
                );
            }

            let replacement_node_1_drt = shared_drt(store.path()).await;
            let _replacement_node_1 = shared_component(&replacement_node_1_drt, namespace);
            assert!(
                ![
                    frontend_drt.connection_id(),
                    logical_worker_id,
                    node_0_drt.connection_id(),
                    old_node_1_drt.connection_id(),
                ]
                .contains(&replacement_node_1_drt.connection_id())
            );
            let mut pending_replacement_sources = Vec::new();
            for dp_rank in 4..8 {
                pending_replacement_sources.push(
                    create_test_source_publisher(
                        &replacement_node_1_drt,
                        &kv_endpoint,
                        WorkerWithDpRank::new(logical_worker_id, dp_rank),
                    )
                    .await,
                );
            }

            let (kv_indexer, indexer) = indexer();
            let cancel = CancellationToken::new();
            let membership_coordinator = KvSourceMembershipCoordinator::start(
                serving_id.clone(),
                configs.clone(),
                source_discovery.clone(),
            );
            let membership_watch = membership_coordinator.subscribe();
            let mut membership_observer = membership_watch.clone();
            crate::kv_router::indexer::recovery::subscriber::start_subscriber(
                serving.clone(),
                indexer,
                membership_watch,
                "test-model".to_string(),
                "decode",
                cancel.child_token(),
            )
            .await
            .unwrap();

            tokio::time::timeout(
                Duration::from_secs(5),
                membership_observer.wait_for(|view| {
                    (0..8).all(|dp_rank| {
                        view.sources
                            .get(&WorkerWithDpRank::new(logical_worker_id, dp_rank))
                            .is_some_and(|status| status.active_source().is_some())
                    })
                }),
            )
            .await
            .expect("all eight logical ranks did not become active")
            .unwrap();

            let initial_events = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publish_rank_blocks(&node_0_sources, 1, 100).await;
                    publish_rank_blocks(&old_node_1_sources, 1, 100).await;
                    kv_indexer.flush().await;
                    let events = kv_indexer.dump_events().await.unwrap();
                    if (0..8).all(|dp_rank| {
                        contains_rank_block(
                            &events,
                            WorkerWithDpRank::new(logical_worker_id, dp_rank),
                            100 + u64::from(dp_rank),
                        )
                    }) {
                        break events;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("distinct state for all eight global ranks was not indexed");
            assert_eq!(
                initial_events
                    .iter()
                    .map(|event| WorkerWithDpRank::new(event.worker_id, event.event.dp_rank))
                    .collect::<HashSet<_>>(),
                (0..8)
                    .map(|dp_rank| WorkerWithDpRank::new(logical_worker_id, dp_rank))
                    .collect()
            );

            let old_rank_4 = old_node_1_sources
                .iter()
                .find(|source| source.worker == delayed_rank)
                .unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    old_rank_4
                        .publisher
                        .publish(&vec![store_block_for(delayed_rank, 3, 903)])
                        .await
                        .unwrap();
                    if recovery_engine.calls.load(Ordering::SeqCst) >= 2 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("old node-1 recovery did not become in flight");

            let mut replacement_node_1_sources = Vec::new();
            for source_publisher in pending_replacement_sources {
                let dp_rank = source_publisher.worker.dp_rank;
                replacement_node_1_sources.push(
                    advertise_test_source(
                        source_discovery.as_ref(),
                        &kv_endpoint,
                        source_publisher,
                        None,
                    )
                    .await,
                );
                tokio::time::timeout(
                    Duration::from_secs(5),
                    membership_observer.wait_for(|view| {
                        matches!(
                            view.sources
                                .get(&WorkerWithDpRank::new(logical_worker_id, dp_rank)),
                            Some(KvSourceStatus::Ambiguous(_))
                        )
                    }),
                )
                .await
                .unwrap_or_else(|_| panic!("rank {dp_rank} did not observe source overlap"))
                .unwrap();
            }
            assert!(old_node_1_sources.iter().all(|old| {
                replacement_node_1_sources
                    .iter()
                    .all(|new| old.publisher.publisher_id() != new.publisher.publisher_id())
            }));

            let ambiguity = tokio::time::timeout(Duration::from_secs(5), async {
                membership_observer
                    .wait_for(|view| {
                        (0..4).all(|dp_rank| {
                            view.sources
                                .get(&WorkerWithDpRank::new(logical_worker_id, dp_rank))
                                .is_some_and(|status| status.active_source().is_some())
                        }) && (4..8).all(|dp_rank| {
                            matches!(
                                view.sources
                                    .get(&WorkerWithDpRank::new(logical_worker_id, dp_rank)),
                                Some(KvSourceStatus::Ambiguous(_))
                            )
                        })
                    })
                    .await
                    .map(|view| view.clone())
            })
            .await;
            match ambiguity {
                Ok(result) => {
                    result.unwrap();
                }
                Err(_) => panic!(
                    "node-1 publisher overlap did not become rank-local ambiguity: {:?}",
                    membership_observer.borrow()
                ),
            }
            let ambiguous_events = wait_for_index_state(
                &kv_indexer,
                |events| {
                    (0..4).all(|dp_rank| {
                        contains_rank_block(
                            events,
                            WorkerWithDpRank::new(logical_worker_id, dp_rank),
                            100 + u64::from(dp_rank),
                        )
                    }) && events.iter().all(|event| event.event.dp_rank < 4)
                },
                "only the overlapping node-1 rank slice should fail KV closed",
            )
            .await;
            assert_eq!(
                ambiguous_events
                    .iter()
                    .map(|event| event.event.dp_rank)
                    .collect::<HashSet<_>>(),
                HashSet::from([0, 1, 2, 3])
            );
            assert!(configs.borrow().contains_key(&logical_worker_id));

            for source in &old_node_1_sources {
                source_discovery
                    .unregister(source.instance.clone())
                    .await
                    .unwrap();
            }
            tokio::time::timeout(
                Duration::from_secs(5),
                membership_observer.wait_for(|view| {
                    (0..8).all(|dp_rank| {
                        view.sources
                            .get(&WorkerWithDpRank::new(logical_worker_id, dp_rank))
                            .is_some_and(|status| status.active_source().is_some())
                    })
                }),
            )
            .await
            .expect("replacement node-1 rank slice did not become selectable")
            .unwrap();

            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publish_rank_blocks(&replacement_node_1_sources, 1, 200).await;
                    kv_indexer.flush().await;
                    let events = kv_indexer.dump_events().await.unwrap();
                    if (4..8).all(|dp_rank| {
                        contains_rank_block(
                            &events,
                            WorkerWithDpRank::new(logical_worker_id, dp_rank),
                            200 + u64::from(dp_rank),
                        )
                    }) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("replacement node-1 state was not activated after the cold reset");

            recovery_engine.delayed_release.notify_waiters();
            tokio::time::timeout(
                Duration::from_secs(5),
                recovery_engine.delayed_finished.notified(),
            )
            .await
            .expect("old node-1 recovery did not finish or cancel after release");
            publish_rank_blocks(&old_node_1_sources, 4, 400).await;
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publish_rank_blocks(&replacement_node_1_sources, 2, 300).await;
                    kv_indexer.flush().await;
                    let events = kv_indexer.dump_events().await.unwrap();
                    if (4..8).all(|dp_rank| {
                        contains_rank_block(
                            &events,
                            WorkerWithDpRank::new(logical_worker_id, dp_rank),
                            300 + u64::from(dp_rank),
                        )
                    }) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("replacement node-1 publishers stopped applying live state");
            let final_events = kv_indexer.dump_events().await.unwrap();
            for dp_rank in 0..4 {
                assert!(contains_rank_block(
                    &final_events,
                    WorkerWithDpRank::new(logical_worker_id, dp_rank),
                    100 + u64::from(dp_rank),
                ));
            }
            for dp_rank in 4..8 {
                let worker = WorkerWithDpRank::new(logical_worker_id, dp_rank);
                assert!(contains_rank_block(
                    &final_events,
                    worker,
                    200 + u64::from(dp_rank),
                ));
                assert!(contains_rank_block(
                    &final_events,
                    worker,
                    300 + u64::from(dp_rank),
                ));
                assert!(!contains_rank_block(
                    &final_events,
                    worker,
                    100 + u64::from(dp_rank),
                ));
                assert!(!contains_rank_block(
                    &final_events,
                    worker,
                    400 + u64::from(dp_rank),
                ));
            }
            assert!(!contains_rank_block(&final_events, delayed_rank, 2));
            assert!(configs.borrow().contains_key(&logical_worker_id));

            cancel.cancel();
            for source in &replacement_node_1_sources {
                source_discovery
                    .unregister(source.instance.clone())
                    .await
                    .unwrap();
            }
            for source in &node_0_sources {
                source_discovery
                    .unregister(source.instance.clone())
                    .await
                    .unwrap();
            }
            discovery.unregister(model_instance).await.unwrap();
            discovery.unregister(serving_instance).await.unwrap();
            recovery_endpoint.shutdown().await.unwrap();
        })
        .await
        .expect("direct ZMQ multi-node KV source lifecycle test timed out");
    }
}
