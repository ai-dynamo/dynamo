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
use tokio::sync::{Mutex, Semaphore, watch};
use tokio_util::sync::CancellationToken;

use super::worker_query_state::{LiveEventAction, PendingDrainAction, RankState, RecoveryKey};
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
    recovery_cancels: DashMap<RecoveryKey, CancellationToken>,
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
            recovery_cancels: DashMap::new(),
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
            recovery_cancels: DashMap::new(),
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
        {
            return;
        }

        let had_active = self.deactivate_locked(key, &mut slot).await;
        slot.lifecycle_generation = lifecycle_generation;
        match selected {
            None => {
                if (had_active || generation_changed)
                    && let Err(error) = self.reset_rank(key).await
                {
                    tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear inactive KV source state");
                }
                slot.rank = RankState::default();
                slot.was_ambiguous = matches!(status, KvSourceStatus::Ambiguous(_));
            }
            Some(source) => {
                if (generation_changed || slot.ever_activated || slot.was_ambiguous)
                    && let Err(error) = self.reset_rank(key).await
                {
                    tracing::error!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = source.publisher_id, "KV source activation remains disabled because cold reset failed");
                    return;
                }
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
                    self.spawn_recovery(key, binding, None, None);
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
        if let Some((_, cancel)) = self.recovery_cancels.remove(&key) {
            cancel.cancel();
        }
        if let Some(target) = binding.recovery_target() {
            let transport = self.transport.clone();
            let target = target.endpoint_instance_id();
            tokio::spawn(async move {
                transport.cancel_instance_streams(&target).await;
            });
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
        if had_active && let Err(error) = self.reset_rank(key).await {
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear KV state for a worker removed from serving membership");
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
                LiveEventAction::Apply(event) => self.indexer.apply_event(event).await,
                LiveEventAction::Clear { event_id } => {
                    drop(slot);
                    self.apply_live_worker_clear(
                        key,
                        event_id,
                        binding.clone(),
                        clear_event.expect("clear action preserves the clear event"),
                    )
                    .await;
                    slot = slot_handle.lock().await;
                    if !slot
                        .active
                        .as_ref()
                        .is_some_and(|active| Arc::ptr_eq(active, &binding))
                    {
                        return;
                    }
                }
                LiveEventAction::Recover { start_event_id } => {
                    self.spawn_recovery(key, binding.clone(), start_event_id, None)
                }
                LiveEventAction::ResetDegraded {
                    event,
                    recover_from,
                } => {
                    self.cancel_recovery(key);
                    if let Err(error) = self.reset_rank(key).await {
                        tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to clear KV state after an event sequence gap");
                        slot.rank.finish_failed_recovery();
                        return;
                    }
                    let event_id = event.event.event_id;
                    self.indexer.apply_event(event).await;
                    self.indexer.flush_and_wait().await;
                    if let Some(start_event_id) = recover_from {
                        self.spawn_recovery(
                            key,
                            binding.clone(),
                            Some(start_event_id),
                            event_id.checked_sub(1),
                        );
                    }
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
    ) {
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
            return;
        };
        if !emitter_slot
            .active
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return;
        }
        for (key, slot) in &mut guards {
            self.cancel_recovery(*key);
            slot.rank
                .apply_worker_clear_barrier(event_id, *key == emitter);
        }
        self.indexer.apply_event(event).await;
        self.indexer.flush_and_wait().await;
    }

    fn cancel_recovery(&self, key: RecoveryKey) {
        if let Some((_, cancel)) = self.recovery_cancels.remove(&key) {
            cancel.cancel();
        }
    }

    fn spawn_recovery(
        self: &Arc<Self>,
        key: RecoveryKey,
        binding: Arc<SourceBinding>,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) {
        let Some(target) = binding.recovery_target().cloned() else {
            return;
        };
        self.cancel_recovery(key);
        let cancel = binding.lifetime.child_token();
        self.recovery_cancels.insert(key, cancel.clone());
        let client = self.clone();
        tokio::spawn(async move {
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
                _ = cancel.cancelled() => return,
                result = recovery => result,
            };
            if cancel.is_cancelled() {
                return;
            }
            client.finish_recovery(key, binding, cancel, result).await;
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

        let current_cursor = slot.rank.cursor;
        let successful = match result {
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
                    false
                } else {
                    for event in events {
                        self.indexer.apply_event(event).await;
                    }
                    self.indexer.flush_and_wait().await;
                    let cursor = current_cursor.advance_to(
                        current_cursor
                            .last_applied_id()
                            .unwrap_or(0)
                            .max(last_event_id),
                    );
                    slot.rank.begin_successful_recovery_drain(cursor);
                    true
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
                    false
                } else if let Err(error) = self.reset_rank(key).await {
                    tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to reset rank before applying recovery tree dump");
                    false
                } else {
                    for event in events {
                        self.indexer.apply_event(event).await;
                    }
                    self.indexer.flush_and_wait().await;
                    slot.rank.begin_successful_recovery_drain(
                        CursorState::Initial.advance_to(last_event_id),
                    );
                    true
                }
            }
            Ok(response) => {
                tracing::warn!(
                    worker_id = key.0,
                    dp_rank = key.1,
                    ?response,
                    "KV recovery returned no applicable state"
                );
                false
            }
            Err(error) => {
                tracing::warn!(%error, worker_id = key.0, dp_rank = key.1, publisher_id = binding.source.publisher_id, "KV recovery failed; continuing with degraded live events");
                false
            }
        };

        if !successful {
            let pending = slot.rank.finish_failed_recovery_degraded();
            if !pending.is_empty() {
                for event in pending {
                    self.indexer.apply_event(event).await;
                }
                self.indexer.flush_and_wait().await;
            }
            return;
        }

        let mut follow_up = None;
        loop {
            match slot.rank.next_pending_drain_action() {
                PendingDrainAction::Apply(event) => self.indexer.apply_event(event).await,
                PendingDrainAction::RecoverFrom(start_event_id) => {
                    follow_up = Some(start_event_id);
                    break;
                }
                PendingDrainAction::Complete => break,
            }
        }
        self.indexer.flush_and_wait().await;
        drop(slot);
        if let Some(start_event_id) = follow_up {
            self.spawn_recovery(key, binding, Some(start_event_id), None);
        }
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
            tracing::error!(%error, worker_id = key.0, dp_rank = key.1, "Failed to reset rank before recovery tree dump");
            return;
        }

        let mut cursor = guards[emitter_index].1.rank.cursor;
        for event in events {
            let event_id = event.event.event_id;
            if matches!(
                &event.event.data,
                dynamo_kv_router::protocols::KvCacheEventData::Cleared
            ) {
                for (slot_key, slot) in &mut guards {
                    if *slot_key != key {
                        self.cancel_recovery(*slot_key);
                    }
                    slot.rank
                        .apply_worker_clear_barrier(event_id, *slot_key == key);
                }
                cursor = cursor.apply_barrier(event_id);
            } else {
                cursor = cursor.advance_to(event_id);
            }
            self.indexer.apply_event(event).await;
        }
        self.indexer.flush_and_wait().await;
        if cancel.is_cancelled()
            || !guards[emitter_index]
                .1
                .active
                .as_ref()
                .is_some_and(|active| Arc::ptr_eq(active, &binding))
        {
            return;
        }
        guards[emitter_index]
            .1
            .rank
            .begin_successful_recovery_drain(cursor.advance_to(last_event_id));
        let mut follow_up = None;
        loop {
            match guards[emitter_index].1.rank.next_pending_drain_action() {
                PendingDrainAction::Apply(event) => self.indexer.apply_event(event).await,
                PendingDrainAction::RecoverFrom(start_event_id) => {
                    follow_up = Some(start_event_id);
                    break;
                }
                PendingDrainAction::Complete => break,
            }
        }
        self.indexer.flush_and_wait().await;
        drop(guards);
        if let Some(start_event_id) = follow_up {
            self.spawn_recovery(key, binding, Some(start_event_id), None);
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
        discovery::{DiscoverySpec, EventTransportKind},
        distributed::DistributedConfig,
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
            network::Ingress,
        },
        protocols::EndpointId,
        stream,
        traits::DistributedRuntimeProvider,
        transports::event_plane::{EventPublisher, EventScope},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Notify, watch};

    use crate::{
        discovery::{
            KvSourceAmbiguity, KvSourceMembershipView, KvStateEndpointResolution, ModelManager,
            runtime_config_watch,
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

    async fn component(name: &str) -> Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        drt.namespace(format!("test-{name}"))
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

    fn contains_block(events: &[RouterEvent], block: u64) -> bool {
        events.iter().any(|event| match &event.event.data {
            KvCacheEventData::Stored(data) => data
                .blocks
                .iter()
                .any(|stored| stored.block_hash == ExternalSequenceBlockHash(block)),
            _ => false,
        })
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
    async fn direct_zmq_source_overlap_fences_stale_recovery_and_events() {
        tokio::time::timeout(Duration::from_secs(20), async {
            let component = component("direct-zmq-source-lifecycle").await;
            let drt = component.drt().clone();
            let serving = component.endpoint("generate");
            let kv_endpoint = EndpointId {
                namespace: component.namespace().name().to_string(),
                component: component.name().to_string(),
                name: "kv-state".to_string(),
            };
            let worker = WorkerWithDpRank::new(drt.connection_id(), 4);
            let discovery = drt.discovery();

            let serving_id = serving.id();
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
                data_parallel_start_rank: worker.dp_rank,
                data_parallel_size: 1,
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
                .wait_for(|configs| configs.contains_key(&worker.worker_id))
                .await
                .unwrap();

            let recovery_engine = Arc::new(ControlledRecoveryEngine {
                worker,
                calls: AtomicUsize::new(0),
                delayed_started: Notify::new(),
                delayed_release: Notify::new(),
            });
            let recovery_endpoint = component
                .endpoint("controlled-kv-recovery")
                .endpoint_builder()
                .handler(Ingress::for_engine(recovery_engine.clone()).unwrap())
                .start_with_registration()
                .await
                .unwrap();
            let publisher_a = EventPublisher::for_endpoint_id_with_transport(
                &drt,
                &kv_endpoint,
                KV_EVENT_TOPIC,
                EventTransportKind::Zmq,
            )
            .await
            .unwrap();
            let source_a = source_for(
                &kv_endpoint,
                worker,
                publisher_a.publisher_id(),
                Some(recovery_endpoint.instance().clone()),
            );
            let source_a_instance = discovery
                .register(DiscoverySpec::EventSource {
                    scope: EventScope::Endpoint {
                        endpoint: kv_endpoint.clone(),
                    },
                    topic: KV_EVENT_TOPIC.to_string(),
                    publisher_id: source_a.publisher_id,
                    metadata: serde_json::to_value(&source_a).unwrap(),
                })
                .await
                .unwrap();

            let (kv_indexer, indexer) = indexer();
            let cancel = CancellationToken::new();
            let model_manager = ModelManager::new();
            let membership_watch = model_manager
                .get_or_create_kv_source_membership_watch(&serving)
                .await
                .unwrap();
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

            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publisher_a
                        .publish(&vec![store_for(worker, 1)])
                        .await
                        .unwrap();
                    kv_indexer.flush().await;
                    if contains_block(&kv_indexer.dump_events().await.unwrap(), 1) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("source A did not become active");
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publisher_a
                        .publish(&vec![store_for(worker, 3)])
                        .await
                        .unwrap();
                    kv_indexer.flush().await;
                    if recovery_engine.calls.load(Ordering::SeqCst) >= 2 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("source A gap recovery did not start");

            let publisher_b = EventPublisher::for_endpoint_id_with_transport(
                &drt,
                &kv_endpoint,
                KV_EVENT_TOPIC,
                EventTransportKind::Zmq,
            )
            .await
            .unwrap();
            let source_b = source_for(&kv_endpoint, worker, publisher_b.publisher_id(), None);
            let source_b_instance = discovery
                .register(DiscoverySpec::EventSource {
                    scope: EventScope::Endpoint {
                        endpoint: kv_endpoint.clone(),
                    },
                    topic: KV_EVENT_TOPIC.to_string(),
                    publisher_id: source_b.publisher_id,
                    metadata: serde_json::to_value(&source_b).unwrap(),
                })
                .await
                .unwrap();
            let ambiguity = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    kv_indexer.flush().await;
                    if kv_indexer.dump_events().await.unwrap().is_empty() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await;
            if ambiguity.is_err() {
                panic!(
                    "overlapping source advertisements did not fail closed: {:?}",
                    kv_indexer.dump_events().await.unwrap()
                );
            }

            discovery.unregister(source_a_instance).await.unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publisher_b
                        .publish(&vec![store_for(worker, 10)])
                        .await
                        .unwrap();
                    kv_indexer.flush().await;
                    if contains_block(&kv_indexer.dump_events().await.unwrap(), 10) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("source B did not activate after exact source A removal");
            recovery_engine.delayed_release.notify_waiters();
            publisher_a
                .publish(&vec![store_for(worker, 11)])
                .await
                .unwrap();
            for _ in 0..100 {
                tokio::task::yield_now().await;
            }
            kv_indexer.flush().await;
            let final_events = kv_indexer.dump_events().await.unwrap();
            assert!(contains_block(&final_events, 10));
            assert!(!contains_block(&final_events, 2));
            assert!(!contains_block(&final_events, 11));
            assert!(configs.borrow().contains_key(&worker.worker_id));

            cancel.cancel();
            discovery.unregister(source_b_instance).await.unwrap();
            discovery.unregister(model_instance).await.unwrap();
            discovery.unregister(serving_instance).await.unwrap();
            recovery_endpoint.shutdown().await.unwrap();
        })
        .await
        .expect("direct ZMQ KV source lifecycle test timed out");
    }
}
