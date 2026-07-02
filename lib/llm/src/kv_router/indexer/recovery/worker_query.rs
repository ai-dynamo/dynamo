// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_runtime::component::{Component, Instance};
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryQuery, EndpointInstanceId};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{Mutex, Semaphore};

use super::worker_query_directory::{DiscoveredQueryEndpoint, WorkerQueryEndpointDirectory};
#[cfg(test)]
use super::worker_query_endpoint::WorkerKvQueryEngine;
use super::worker_query_health::spawn_kv_event_source_health_monitor;
use super::worker_query_state::{LiveEventAction, PendingDrainAction, RecoveryKey, WorkerState};
use super::worker_query_transport::{RuntimeWorkerQueryTransport, WorkerQueryTransport};
use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::Indexer;
use dynamo_kv_router::{
    indexer::WorkerKvQueryResponse,
    protocols::{DpRank, KvCacheEventData, RouterEvent, WorkerId},
};

#[cfg(test)]
use super::worker_query_state::RankState;
#[cfg(test)]
use crate::kv_router::{
    worker_kv_indexer_query_endpoint, worker_kv_indexer_query_endpoint_for_worker,
};
#[cfg(test)]
use async_trait::async_trait;
#[cfg(test)]
use dynamo_kv_router::indexer::{LocalKvIndexer, WorkerKvQueryRequest};
#[cfg(test)]
use dynamo_kv_router::recovery::CursorState;
#[cfg(test)]
use dynamo_runtime::pipeline::{AsyncEngine, SingleIn};

// Recovery retry configuration
const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;
const RECOVERY_CONCURRENCY_LIMIT: usize = 16;

/// Router-side client for querying worker local KV indexers.
///
/// Discovers query endpoints via `ComponentEndpoints` discovery, filtering for
/// the `worker_kv_indexer_query_dp{N}` name pattern. Coordinates restore and
/// gap recovery at the worker level while still querying each `(worker_id,
/// dp_rank)` endpoint independently. Recovery sends strict direct requests to
/// the discovered endpoint instance; it does not use serving-router load
/// balancing, busy handling, or fallback worker selection.
///
/// Also handles worker lifecycle (add/remove) by tracking known endpoints and
/// sending removal events to the router indexer when all dp_ranks for a worker
/// disappear.
pub struct WorkerQueryClient {
    component: Component,
    transport: Arc<dyn WorkerQueryTransport>,
    /// Indexer for applying recovered events and worker removals.
    indexer: Indexer,
    worker_states: DashMap<WorkerId, Arc<Mutex<WorkerState>>>,
    query_endpoints: Arc<WorkerQueryEndpointDirectory>,
    recovery_semaphore: Arc<Semaphore>,
    /// Per-rank cancellation for in-flight recovery tasks; cancelled on rank
    /// removal so retry backoff stops polling workers that no longer exist.
    recovery_cancels: DashMap<RecoveryKey, tokio_util::sync::CancellationToken>,
}

impl WorkerQueryClient {
    fn new(
        component: Component,
        indexer: Indexer,
        transport: Arc<dyn WorkerQueryTransport>,
    ) -> Arc<Self> {
        Arc::new(Self {
            component,
            transport,
            indexer,
            worker_states: DashMap::new(),
            query_endpoints: Arc::new(WorkerQueryEndpointDirectory::default()),
            recovery_semaphore: Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)),
            recovery_cancels: DashMap::new(),
        })
    }

    /// Create a new WorkerQueryClient and spawn its background discovery loop.
    ///
    /// The background loop watches `ComponentEndpoints` discovery for query endpoints,
    /// recovers each `(worker_id, dp_rank)` as it appears, and sends worker removal
    /// events when all dp_ranks for a worker disappear.
    pub async fn spawn(
        component: Component,
        indexer: Indexer,
        workers_with_configs: RuntimeConfigWatch,
    ) -> Result<Arc<Self>> {
        let transport = Arc::new(RuntimeWorkerQueryTransport::new(&component).await?);
        let client = Self::new(component.clone(), indexer, transport);

        spawn_kv_event_source_health_monitor(
            component.clone(),
            workers_with_configs,
            client.query_endpoints.clone(),
        );

        let client_bg = client.clone();
        let cancel_token = component.drt().primary_token();
        tokio::spawn(async move {
            if let Err(e) = client_bg.run_discovery_loop(cancel_token).await {
                tracing::error!("WorkerQueryClient discovery loop failed: {e}");
            }
        });

        Ok(client)
    }

    /// Background loop: watches ComponentEndpoints and schedules worker-coordinated recovery.
    async fn run_discovery_loop(
        self: Arc<Self>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let discovery = self.component.drt().discovery();
        let mut stream = discovery
            .list_and_watch(
                DiscoveryQuery::ComponentEndpoints {
                    namespace: self.component.namespace().name(),
                    component: self.component.name().to_string(),
                },
                Some(cancel_token.clone()),
            )
            .await?;

        while let Some(result) = stream.next().await {
            if cancel_token.is_cancelled() {
                break;
            }

            let event = match result {
                Ok(event) => event,
                Err(e) => {
                    tracing::warn!("Discovery event error in WorkerQueryClient: {e}");
                    continue;
                }
            };

            match event {
                DiscoveryEvent::Added(instance) => {
                    let Some(endpoint) = WorkerQueryEndpointDirectory::parse_added(instance) else {
                        continue;
                    };
                    self.handle_discovered_query_endpoint(endpoint).await;
                }
                DiscoveryEvent::Removed(id) => {
                    let Some((worker_id, dp_rank, endpoint_id)) =
                        WorkerQueryEndpointDirectory::parse_removed(id)
                    else {
                        continue;
                    };
                    self.handle_removed_query_endpoint(worker_id, dp_rank, endpoint_id)
                        .await;
                }
            }
        }

        Ok(())
    }

    fn get_or_create_worker_state(&self, worker_id: WorkerId) -> Arc<Mutex<WorkerState>> {
        self.worker_states
            .entry(worker_id)
            .or_insert_with(|| Arc::new(Mutex::new(WorkerState::default())))
            .clone()
    }

    fn query_target_for(&self, worker_id: WorkerId, dp_rank: DpRank) -> Option<Instance> {
        if let Some(target) = self.query_endpoints.target_for(worker_id, dp_rank) {
            return Some(target);
        }

        #[cfg(test)]
        {
            Some(Instance {
                namespace: self.component.namespace().name().to_string(),
                component: self.component.name().to_string(),
                endpoint: worker_kv_indexer_query_endpoint(dp_rank),
                instance_id: worker_id,
                transport: dynamo_runtime::component::TransportType::Nats(String::new()),
                device_type: None,
            })
        }

        #[cfg(not(test))]
        None
    }

    #[cfg(test)]
    async fn handle_discovered_worker(self: &Arc<Self>, worker_id: WorkerId, dp_rank: DpRank) {
        let endpoint = DiscoveredQueryEndpoint {
            worker_id,
            dp_rank,
            target: Instance {
                namespace: self.component.namespace().name().to_string(),
                component: self.component.name().to_string(),
                endpoint: worker_kv_indexer_query_endpoint(dp_rank),
                instance_id: worker_id,
                transport: dynamo_runtime::component::TransportType::Nats(String::new()),
                device_type: None,
            },
        };
        self.handle_discovered_query_endpoint(endpoint).await;
    }

    async fn handle_discovered_query_endpoint(self: &Arc<Self>, endpoint: DiscoveredQueryEndpoint) {
        let worker_id = endpoint.worker_id;
        let dp_rank = endpoint.dp_rank;
        let endpoint_id = endpoint.target.endpoint_instance_id();
        self.transport.clear_instance_tombstone(&endpoint_id).await;

        let replaced = match self.query_endpoints.insert(&endpoint) {
            Some(previous) if previous != endpoint.target => Some(previous),
            _ => None,
        };
        if let Some(previous) = replaced.as_ref() {
            tracing::warn!(
                "WorkerQueryClient: query endpoint for worker {worker_id} dp_rank {dp_rank} \
                 changed from {:?} to {:?}; resetting rank state",
                previous,
                endpoint.target
            );
            self.transport
                .cancel_instance_streams(&previous.endpoint_instance_id())
                .await;
        }

        let worker_state = self.get_or_create_worker_state(worker_id);
        let spawn = {
            let mut worker_state = worker_state.lock().await;
            let action = worker_state.handle_discovered_rank(dp_rank, replaced.is_some());
            if action.reset_rank {
                self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            if action.restore_epoch.is_some() {
                tracing::info!(
                    "WorkerQueryClient: discovered worker {worker_id} dp_rank {dp_rank}, scheduling restore"
                );
            }
            action.restore_epoch
        };

        if let Some(epoch) = spawn {
            self.spawn_recovery_task((worker_id, dp_rank), epoch, None, None);
        }
    }

    #[cfg(test)]
    async fn handle_removed_worker_dp(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.query_endpoints.remove_dp(worker_id, dp_rank);
        self.remove_worker_dp_state(worker_id, dp_rank).await;
    }

    async fn handle_removed_query_endpoint(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        endpoint_id: EndpointInstanceId,
    ) {
        if !self
            .query_endpoints
            .remove_if_matches(worker_id, dp_rank, &endpoint_id)
        {
            return;
        }

        self.transport.cancel_instance_streams(&endpoint_id).await;
        self.remove_worker_dp_state(worker_id, dp_rank).await;
    }

    async fn remove_worker_dp_state(&self, worker_id: WorkerId, dp_rank: DpRank) {
        let Some(worker_state) = self
            .worker_states
            .get(&worker_id)
            .map(|entry| entry.clone())
        else {
            return;
        };

        let should_remove_worker = {
            let mut worker_state = worker_state.lock().await;
            if !worker_state.remove_rank(dp_rank) {
                return;
            }
            worker_state.is_empty()
        };

        // Stop any in-flight recovery retry/backoff loop for this rank.  This
        // runs AFTER the rank teardown above: a racing spawn either passed its
        // liveness check before `remove_rank` (so its token is already
        // registered and cancelled here), or it observes the rank as gone and
        // exits on its own.
        if let Some((_, cancel)) = self.recovery_cancels.remove(&(worker_id, dp_rank)) {
            cancel.cancel();
        }

        if should_remove_worker {
            tracing::warn!("WorkerQueryClient: all dp_ranks gone for worker {worker_id}, removing");
            self.worker_states.remove(&worker_id);
            self.indexer.remove_worker(worker_id).await;
        }
    }

    async fn apply_worker_clear_locked(&self, worker_state: &mut WorkerState, event: RouterEvent) {
        let worker_id = event.worker_id;
        let clear_dp_rank = event.event.dp_rank;
        let clear_event_id = event.event.event_id;

        worker_state.apply_worker_clear_barrier(clear_dp_rank, clear_event_id);

        tracing::info!(
            "Applying clear barrier for worker {worker_id}; invalidating recovery across {} dp_ranks",
            worker_state.ranks.len()
        );
        self.indexer.apply_event(event).await;
    }

    async fn apply_tree_dump_replace_locked(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) {
        self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
        for event in events {
            self.indexer.apply_event(event).await;
        }
    }

    pub(crate) async fn handle_live_event(self: &Arc<Self>, event: RouterEvent) {
        let worker_id = event.worker_id;
        let dp_rank = event.event.dp_rank;
        let key = (worker_id, dp_rank);

        let action = {
            let worker_state = self.get_or_create_worker_state(worker_id);
            let mut worker_state = worker_state.lock().await;
            match worker_state.observe_live_event(event) {
                LiveEventAction::ApplyClear(event) => {
                    tracing::info!(
                        "Applying clear barrier for worker {worker_id}; invalidating recovery across {} dp_ranks",
                        worker_state.ranks.len()
                    );
                    self.indexer.apply_event(event).await;
                    return;
                }
                action => action,
            }
        };

        match action {
            LiveEventAction::Ignore => {}
            LiveEventAction::ApplyDirect(event) => {
                self.indexer.apply_event(event).await;
            }
            LiveEventAction::ApplyClear(_) => unreachable!("clear is applied under worker lock"),
            LiveEventAction::SpawnFullRestore { epoch } => {
                self.spawn_recovery_task(key, epoch, None, None);
            }
            LiveEventAction::SpawnIncremental {
                epoch,
                start_event_id,
            } => {
                self.spawn_recovery_task(key, epoch, Some(start_event_id), None);
            }
        }
    }

    fn spawn_recovery_task(
        self: &Arc<Self>,
        key: RecoveryKey,
        epoch: u64,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) {
        let client = self.clone();

        tokio::spawn(async move {
            // Re-check liveness under the worker-state lock, then register the
            // cancellation token only if the rank is still present. Removal holds
            // this same lock to drop the rank before it drains `recovery_cancels`,
            // so a token registered here is guaranteed to be observed — and
            // cancelled — by that drain. A rank that is already gone registers
            // nothing, so a spawn that loses the removal race cannot leak an entry
            // that no later teardown would reclaim.
            let cancel = {
                let live = match client.worker_states.get(&key.0).map(|e| e.clone()) {
                    Some(worker_state) => {
                        let worker_state = worker_state.lock().await;
                        // TODO(#10580 follow-up): pre-existing ABA case — if removing
                        // the final rank deletes `WorkerState` and rediscovery
                        // recreates it at epoch 0, a stale follow-up still carrying
                        // epoch 0 can pass this check against the freshly discovered
                        // endpoint. Out of scope for the #10580 fix (the base branch
                        // already has a resettable epoch + dynamic target resolution);
                        // a follow-up should fold the endpoint/rank incarnation into
                        // the recovery identity, or use a generation that survives
                        // removal.
                        (worker_state.epoch == epoch && worker_state.ranks.contains_key(&key.1))
                            .then(|| client.recovery_cancels.entry(key).or_default().clone())
                    }
                    None => None,
                };
                let Some(cancel) = live else {
                    tracing::debug!(
                        "Skipping recovery for worker {} dp_rank {}: rank removed or epoch changed",
                        key.0,
                        key.1
                    );
                    return;
                };
                cancel
            };

            let recovery = async {
                // Add jitter only for full-restore (start_event_id is None)
                // to permute semaphore acquisition order and reduce thundering herd risk on initial discovery.
                // This distributes load when multiple routers start simultaneously.
                if start_event_id.is_none() {
                    let jitter_us = rand::rng().random_range(0..3000u64);
                    tokio::time::sleep(Duration::from_micros(jitter_us)).await;
                }

                let _permit = client
                    .recovery_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .ok()?;

                Some(
                    client
                        .fetch_recovery_response(key.0, key.1, start_event_id, end_event_id)
                        .await,
                )
            };

            let result = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::debug!("Recovery cancelled for worker {} dp_rank {}", key.0, key.1);
                    return;
                }
                result = recovery => result,
            };

            if let Some(result) = result {
                client.finish_recovery_task(key, epoch, result).await;
            }
        });
    }

    async fn finish_recovery_task(
        self: Arc<Self>,
        key: RecoveryKey,
        epoch: u64,
        result: Result<WorkerKvQueryResponse>,
    ) {
        let Some(worker_state) = self.worker_states.get(&key.0).map(|entry| entry.clone()) else {
            return;
        };
        let mut worker_state = worker_state.lock().await;
        if worker_state.epoch != epoch {
            tracing::debug!(
                "Discarding stale recovery result for worker {} dp_rank {} due to epoch change",
                key.0,
                key.1
            );
            return;
        }

        let Some(mut new_cursor) = worker_state.rank_cursor(key.1) else {
            return;
        };

        let mut successful_response = false;

        match result {
            Ok(WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            }) => {
                tracing::debug!(
                    "Got {count} buffered events from worker {} dp_rank {}",
                    key.0,
                    key.1,
                    count = events.len()
                );
                for event in events {
                    let event_id = event.event.event_id;
                    if matches!(&event.event.data, KvCacheEventData::Cleared) {
                        self.apply_worker_clear_locked(&mut worker_state, event)
                            .await;
                        new_cursor = new_cursor.apply_barrier(event_id);
                        continue;
                    }
                    self.indexer.apply_event(event).await;
                    new_cursor = new_cursor.advance_to(event_id);
                }
                new_cursor = new_cursor.advance_to(last_event_id);
                successful_response = true;
            }
            Ok(WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            }) => {
                let represented_blocks = events
                    .iter()
                    .map(|event| match &event.event.data {
                        KvCacheEventData::Stored(store) => store.blocks.len(),
                        _ => 0,
                    })
                    .sum::<usize>();
                tracing::info!(
                    worker_id = key.0,
                    dp_rank = key.1,
                    event_count = events.len(),
                    represented_block_count = represented_blocks,
                    last_event_id,
                    "Got tree dump (range too old or unspecified)"
                );
                self.apply_tree_dump_replace_locked(key.0, key.1, events)
                    .await;
                new_cursor = new_cursor.advance_to(last_event_id);
                successful_response = true;
            }
            Ok(WorkerKvQueryResponse::TooNew {
                newest_available, ..
            }) => {
                tracing::warn!(
                    "Requested recovery is newer than available (newest: {newest_available}) for worker {} dp_rank {}",
                    key.0,
                    key.1
                );
            }
            Ok(WorkerKvQueryResponse::InvalidRange { start_id, end_id }) => {
                tracing::error!(
                    "Invalid range for worker {} dp_rank {}: end_id ({end_id}) < start_id ({start_id})",
                    key.0,
                    key.1
                );
            }
            Ok(WorkerKvQueryResponse::Error(message)) => {
                tracing::error!(
                    "Worker {} dp_rank {} query error: {}",
                    key.0,
                    key.1,
                    message
                );
            }
            Err(error) => {
                tracing::warn!(
                    "Failed recovery from worker {} dp_rank {}: {}",
                    key.0,
                    key.1,
                    error
                );
            }
        }

        let mut follow_up_start = None;
        if successful_response {
            worker_state.begin_successful_recovery_drain(key.1, new_cursor);
            loop {
                match worker_state.next_pending_drain_action(key.1) {
                    PendingDrainAction::Apply(event) => {
                        self.indexer.apply_event(event).await;
                    }
                    PendingDrainAction::RecoverFrom(start_event_id) => {
                        follow_up_start = Some(start_event_id);
                        break;
                    }
                    PendingDrainAction::Complete => break,
                }
            }
        } else {
            worker_state.finish_failed_recovery(key.1);
        }
        let follow_up_epoch = worker_state.epoch;
        drop(worker_state);

        if let Some(start_event_id) = follow_up_start {
            self.spawn_recovery_task(key, follow_up_epoch, Some(start_event_id), None);
        }
    }

    async fn resolve_query_target_for_recovery(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        attempt: u32,
    ) -> Option<Instance> {
        if let Some(target) = self.query_target_for(worker_id, dp_rank) {
            return Some(target);
        }

        if attempt < RECOVERY_MAX_RETRIES - 1 {
            let backoff_ms = RECOVERY_INITIAL_BACKOFF_MS * 2_u64.pow(attempt);
            tracing::warn!(
                "Worker {worker_id} dp_rank {dp_rank} query owner missing on attempt {attempt}, \
                 retrying after {backoff_ms}ms"
            );
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        }

        None
    }

    /// Query a worker's local KV indexer with exponential backoff retry.
    async fn fetch_recovery_response(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        tracing::debug!(
            "Attempting recovery from worker {worker_id} dp_rank {dp_rank}, \
             start_event_id: {start_event_id:?}, end_event_id: {end_event_id:?}"
        );

        let mut last_error = None;

        for attempt in 0..RECOVERY_MAX_RETRIES {
            let Some(target) = self
                .resolve_query_target_for_recovery(worker_id, dp_rank, attempt)
                .await
            else {
                last_error = Some(anyhow::anyhow!(
                    "No query owner discovered for worker {worker_id} dp_rank {dp_rank}"
                ));
                continue;
            };

            match self
                .transport
                .query_worker(worker_id, dp_rank, target, start_event_id, end_event_id)
                .await
            {
                Ok(resp) => {
                    if attempt > 0 {
                        tracing::info!(
                            "Worker {worker_id} dp_rank {dp_rank} query succeeded after retry {attempt}"
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < RECOVERY_MAX_RETRIES - 1 {
                        let backoff_ms = RECOVERY_INITIAL_BACKOFF_MS * 2_u64.pow(attempt);
                        tracing::warn!(
                            "Worker {worker_id} dp_rank {dp_rank} query failed on attempt {attempt}, \
                             retrying after {backoff_ms}ms"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("No response after {RECOVERY_MAX_RETRIES} retries")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::{Indexer, indexer::LowerTierIndexers};
    use dynamo_kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::{Instance, TransportType},
        discovery::{DiscoveryInstance, DiscoveryInstanceId, EndpointInstanceId},
        distributed::DistributedConfig,
    };
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;

    #[derive(Clone)]
    struct MockQueryAction {
        started: Option<Arc<Notify>>,
        release: Option<Arc<Notify>>,
        response: Result<WorkerKvQueryResponse, String>,
    }

    #[derive(Default)]
    struct MockWorkerQueryTransport {
        actions: DashMap<RecoveryKey, Arc<StdMutex<VecDeque<MockQueryAction>>>>,
        #[allow(clippy::type_complexity)]
        calls: Arc<StdMutex<Vec<(RecoveryKey, Option<u64>, Option<u64>)>>>,
        targets: Arc<StdMutex<Vec<(RecoveryKey, Instance)>>>,
        cancelled_instances: Arc<StdMutex<Vec<EndpointInstanceId>>>,
        cleared_tombstones: Arc<StdMutex<Vec<EndpointInstanceId>>>,
    }

    impl MockWorkerQueryTransport {
        fn push_action(&self, key: RecoveryKey, action: MockQueryAction) {
            let queue = self
                .actions
                .entry(key)
                .or_insert_with(|| Arc::new(StdMutex::new(VecDeque::new())))
                .clone();
            queue.lock().unwrap().push_back(action);
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn calls(&self) -> Vec<(RecoveryKey, Option<u64>, Option<u64>)> {
            self.calls.lock().unwrap().clone()
        }

        fn targets(&self) -> Vec<(RecoveryKey, Instance)> {
            self.targets.lock().unwrap().clone()
        }

        fn cancelled_instances(&self) -> Vec<EndpointInstanceId> {
            self.cancelled_instances.lock().unwrap().clone()
        }

        fn cleared_tombstones(&self) -> Vec<EndpointInstanceId> {
            self.cleared_tombstones.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl WorkerQueryTransport for MockWorkerQueryTransport {
        async fn query_worker(
            &self,
            worker_id: WorkerId,
            dp_rank: DpRank,
            target: Instance,
            start_event_id: Option<u64>,
            end_event_id: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            let key = (worker_id, dp_rank);
            self.calls
                .lock()
                .unwrap()
                .push((key, start_event_id, end_event_id));
            self.targets.lock().unwrap().push((key, target));

            let queue = self
                .actions
                .get(&key)
                .unwrap_or_else(|| {
                    panic!("Missing action queue for worker {worker_id} dp_rank {dp_rank}")
                })
                .clone();
            let action = queue.lock().unwrap().pop_front().unwrap_or_else(|| {
                panic!("Missing action for worker {worker_id} dp_rank {dp_rank}")
            });

            if let Some(started) = action.started {
                started.notify_waiters();
            }
            if let Some(release) = action.release {
                release.notified().await;
            }

            match action.response {
                Ok(response) => Ok(response),
                Err(message) => Err(anyhow::anyhow!(message)),
            }
        }

        async fn cancel_instance_streams(&self, endpoint_id: &EndpointInstanceId) -> usize {
            self.cancelled_instances
                .lock()
                .unwrap()
                .push(endpoint_id.clone());
            0
        }

        async fn clear_instance_tombstone(&self, endpoint_id: &EndpointInstanceId) {
            self.cleared_tombstones
                .lock()
                .unwrap()
                .push(endpoint_id.clone());
        }
    }

    async fn make_test_component(name: &str) -> Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt.namespace(format!("test-ns-{name}")).unwrap();
        namespace
            .component(format!("test-component-{name}"))
            .unwrap()
    }

    fn make_test_indexer() -> (KvIndexer, Indexer) {
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let kv_indexer = KvIndexer::new(token, 4, metrics);
        (
            kv_indexer.clone(),
            Indexer::KvIndexer {
                primary: kv_indexer,
                lower_tier: LowerTierIndexers::new(1, 4),
                approx: None,
                primary_records_routing_decisions: false,
            },
        )
    }

    async fn make_test_client(
        name: &str,
    ) -> (
        Arc<WorkerQueryClient>,
        Arc<MockWorkerQueryTransport>,
        KvIndexer,
    ) {
        let component = make_test_component(name).await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());
        (client, transport, kv_indexer)
    }

    fn make_instance(endpoint: String, instance_id: WorkerId) -> Instance {
        Instance {
            namespace: "test-ns".to_string(),
            component: "test-component".to_string(),
            endpoint,
            instance_id,
            transport: TransportType::Nats("nats://127.0.0.1:4222".to_string()),
            device_type: None,
        }
    }

    fn make_endpoint_instance(endpoint: String, instance_id: WorkerId) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(make_instance(endpoint, instance_id))
    }

    fn make_endpoint_instance_id(endpoint: String, instance_id: WorkerId) -> DiscoveryInstanceId {
        DiscoveryInstanceId::Endpoint(EndpointInstanceId {
            namespace: "test-ns".to_string(),
            component: "test-component".to_string(),
            endpoint,
            instance_id,
        })
    }

    fn make_store_event(worker_id: WorkerId, dp_rank: DpRank, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
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
                dp_rank,
            },
        )
    }

    fn make_clear_event(worker_id: WorkerId, dp_rank: DpRank, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank,
            },
        )
    }

    fn stored_block_hashes(events: &[RouterEvent]) -> Vec<u64> {
        let mut hashes = events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => {
                    data.blocks.first().map(|block| block.block_hash.0)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        hashes.sort_unstable();
        hashes
    }

    fn stored_block_hashes_for(
        events: &[RouterEvent],
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Vec<u64> {
        let mut hashes = events
            .iter()
            .filter(|event| event.worker_id == worker_id && event.event.dp_rank == dp_rank)
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => {
                    data.blocks.first().map(|block| block.block_hash.0)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        hashes.sort_unstable();
        hashes
    }

    async fn wait_for<F>(mut check: F)
    where
        F: FnMut() -> bool,
    {
        for _ in 0..100 {
            if check() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition not met before timeout");
    }

    fn rank_state_matches<F>(client: &Arc<WorkerQueryClient>, key: RecoveryKey, check: F) -> bool
    where
        F: FnOnce(&RankState) -> bool,
    {
        client
            .worker_states
            .get(&key.0)
            .map(|worker_state| match worker_state.try_lock() {
                Ok(worker_state) => worker_state.ranks.get(&key.1).is_some_and(check),
                Err(_) => false,
            })
            .unwrap_or(false)
    }

    #[test]
    fn test_parse_legacy_query_endpoint_uses_route_instance_as_worker() {
        let endpoint_name = worker_kv_indexer_query_endpoint(4);
        let instance = make_endpoint_instance(endpoint_name.clone(), 11);

        let parsed = WorkerQueryEndpointDirectory::parse_added(instance)
            .expect("legacy query endpoint should parse");

        assert_eq!(
            parsed,
            DiscoveredQueryEndpoint {
                worker_id: 11,
                dp_rank: 4,
                target: make_instance(endpoint_name, 11),
            }
        );
    }

    #[test]
    fn test_parse_worker_scoped_query_endpoint_keeps_logical_worker_id() {
        let endpoint_name = worker_kv_indexer_query_endpoint_for_worker(100, 4);
        let instance = make_endpoint_instance(endpoint_name.clone(), 11);
        let instance_id = make_endpoint_instance_id(endpoint_name.clone(), 11);

        let parsed = WorkerQueryEndpointDirectory::parse_added(instance)
            .expect("worker-scoped query endpoint should parse");
        let parsed_id = WorkerQueryEndpointDirectory::parse_removed(instance_id)
            .expect("worker-scoped query endpoint id should parse");

        let expected = DiscoveredQueryEndpoint {
            worker_id: 100,
            dp_rank: 4,
            target: make_instance(endpoint_name, 11),
        };
        assert_eq!(parsed, expected.clone());
        assert_eq!(parsed_id.0, expected.worker_id);
        assert_eq!(parsed_id.1, expected.dp_rank);
        assert_eq!(parsed_id.2, expected.target.endpoint_instance_id());
    }

    #[tokio::test]
    async fn test_recovery_routes_to_discovered_instance_for_logical_worker() {
        let (client, transport, kv_indexer) = make_test_client("logical-route").await;
        let endpoint_name = worker_kv_indexer_query_endpoint_for_worker(100, 4);
        let endpoint = DiscoveredQueryEndpoint {
            worker_id: 100,
            dp_rank: 4,
            target: make_instance(endpoint_name.clone(), 11),
        };

        transport.push_action(
            (100, 4),
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 0,
                }),
            },
        );

        client.handle_discovered_query_endpoint(endpoint).await;

        wait_for(|| transport.call_count() == 1).await;
        assert_eq!(
            transport.targets(),
            vec![((100, 4), make_instance(endpoint_name, 11))]
        );
        kv_indexer.flush().await;
    }

    #[tokio::test]
    async fn test_query_endpoint_replacement_resets_rank_and_restores_new_target() {
        let (client, transport, kv_indexer) = make_test_client("replace-route").await;
        let key = (100, 4);
        let first_endpoint_name = worker_kv_indexer_query_endpoint_for_worker(key.0, key.1);
        let second_endpoint_name = first_endpoint_name.clone();
        let first_target = make_instance(first_endpoint_name, 11);
        let second_target = make_instance(second_endpoint_name, 12);
        let first_id = first_target.endpoint_instance_id();
        let second_id = second_target.endpoint_instance_id();

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![make_store_event(key.0, key.1, 90)],
                    last_event_id: 90,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![make_store_event(key.0, key.1, 1)],
                    last_event_id: 1,
                }),
            },
        );

        client
            .handle_discovered_query_endpoint(DiscoveredQueryEndpoint {
                worker_id: key.0,
                dp_rank: key.1,
                target: first_target.clone(),
            })
            .await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(90) && !state.recovery_inflight
            })
        })
        .await;

        client
            .handle_discovered_query_endpoint(DiscoveredQueryEndpoint {
                worker_id: key.0,
                dp_rank: key.1,
                target: second_target.clone(),
            })
            .await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(1) && !state.recovery_inflight
            })
        })
        .await;

        assert_eq!(transport.cancelled_instances(), vec![first_id.clone()]);
        assert_eq!(transport.cleared_tombstones(), vec![first_id, second_id]);
        assert_eq!(
            transport.targets(),
            vec![(key, first_target), (key, second_target)]
        );

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes_for(&events, key.0, key.1), vec![1]);
    }

    #[tokio::test]
    async fn test_removed_query_endpoint_cancels_inflight_streams() {
        let (client, transport, kv_indexer) = make_test_client("remove-cancels").await;
        let key = (100, 4);
        let endpoint_name = worker_kv_indexer_query_endpoint_for_worker(key.0, key.1);
        let target = make_instance(endpoint_name, 11);
        let endpoint_id = target.endpoint_instance_id();

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 0,
                }),
            },
        );

        client
            .handle_discovered_query_endpoint(DiscoveredQueryEndpoint {
                worker_id: key.0,
                dp_rank: key.1,
                target,
            })
            .await;
        wait_for(|| transport.call_count() == 1).await;

        client
            .handle_removed_query_endpoint(key.0, key.1, endpoint_id.clone())
            .await;

        assert_eq!(transport.cancelled_instances(), vec![endpoint_id]);
        wait_for(|| !rank_state_matches(&client, key, |_| true)).await;
        kv_indexer.flush().await;
    }

    #[tokio::test]
    async fn test_removed_query_endpoint_stops_recovery_retries() {
        let (client, transport, kv_indexer) = make_test_client("remove-stops-retries").await;
        let key = (100, 4);
        let endpoint_name = worker_kv_indexer_query_endpoint_for_worker(key.0, key.1);
        let target = make_instance(endpoint_name, 11);
        let endpoint_id = target.endpoint_instance_id();

        // Queue a single failing attempt; the failure schedules retries with
        // exponential backoff, and any retry would call the transport again.
        let started = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: None,
                response: Err("transient failure".to_string()),
            },
        );

        client
            .handle_discovered_query_endpoint(DiscoveredQueryEndpoint {
                worker_id: key.0,
                dp_rank: key.1,
                target,
            })
            .await;
        started.notified().await;

        // Removing the rank mid-backoff must cancel the in-flight recovery.
        client
            .handle_removed_query_endpoint(key.0, key.1, endpoint_id)
            .await;
        wait_for(|| client.recovery_cancels.is_empty()).await;

        // A non-cancelled task would retry after RECOVERY_INITIAL_BACKOFF_MS.
        tokio::time::sleep(Duration::from_millis(3 * RECOVERY_INITIAL_BACKOFF_MS)).await;
        assert_eq!(transport.call_count(), 1);
        kv_indexer.flush().await;
    }

    #[tokio::test]
    async fn test_spawn_recovery_for_removed_rank_does_not_query() {
        let (client, transport, kv_indexer) = make_test_client("spawn-after-remove").await;

        // A follow-up spawn can race rank removal (the spawn decision happens
        // outside the worker-state lock). With no rank state, the task's
        // liveness check must drop it before it ever queries the transport — and
        // without registering a cancellation token that no later teardown would
        // ever reclaim.
        client.spawn_recovery_task((1, 0), 0, Some(5), None);

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(transport.call_count(), 0);
        assert!(
            client.recovery_cancels.is_empty(),
            "a spawn that lost the removal race must not leak a recovery_cancels entry"
        );
        kv_indexer.flush().await;
    }

    #[tokio::test]
    async fn test_worker_kv_query_engine_returns_buffered_events() {
        let worker_id = 7u64;
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token, 4, metrics, 32));

        let event = RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        );
        local_indexer
            .apply_event_with_buffer(event)
            .await
            .expect("apply_event_with_buffer should succeed");

        let engine = WorkerKvQueryEngine {
            worker_id,
            dp_rank: 0,
            local_indexer,
            processing_semaphore: Semaphore::new(1),
        };

        let request = WorkerKvQueryRequest {
            worker_id,
            dp_rank: 0,
            start_event_id: Some(1),
            end_event_id: Some(1),
        };

        let mut stream = engine
            .generate(SingleIn::new(request))
            .await
            .expect("generate should succeed");

        let response = stream
            .next()
            .await
            .expect("response stream should yield one item");

        match response {
            WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            } => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].event.event_id, 1);
                assert_eq!(last_event_id, 1);
            }
            other => panic!("Unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_discovery_restore_does_not_block_other_workers() {
        let (client, transport, kv_indexer) = make_test_client("discovery-concurrency").await;

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            (1, 0),
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 0,
                }),
            },
        );
        transport.push_action(
            (2, 0),
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 0,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        first_started.notified().await;
        client.handle_discovered_worker(2, 0).await;

        wait_for(|| transport.call_count() == 2).await;
        first_release.notify_waiters();
        kv_indexer.flush().await;
    }

    #[tokio::test]
    async fn test_gap_recovery_follows_high_water_mark() {
        let (client, transport, kv_indexer) = make_test_client("high-water").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (11..=15).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 15,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        first_started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 16)).await;
        client.handle_live_event(make_store_event(1, 0, 17)).await;
        client.handle_live_event(make_store_event(1, 0, 18)).await;
        first_release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(18) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.calls(), vec![(key, Some(11), None)]);

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![11, 12, 13, 14, 15, 16, 17, 18]
        );
    }

    #[tokio::test]
    async fn test_tree_dump_replays_pending_live_tail_after_replace() {
        let (client, transport, kv_indexer) = make_test_client("tree-dump-pending-tail").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: (11..=15).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 15,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 16)).await;
        client.handle_live_event(make_store_event(1, 0, 17)).await;
        release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(17) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.calls(), vec![(key, Some(11), None)]);

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![11, 12, 13, 14, 15, 16, 17]
        );
    }

    #[tokio::test]
    async fn test_pending_gap_retains_tail_across_follow_up_recovery() {
        let (client, transport, kv_indexer) = make_test_client("pending-gap-retains-tail").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(15);
        }

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (16..=20).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 20,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 23)],
                    last_event_id: 23,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 20)).await;
        first_started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 21)).await;
        client.handle_live_event(make_store_event(1, 0, 22)).await;
        client.handle_live_event(make_store_event(1, 0, 24)).await;
        client.handle_live_event(make_store_event(1, 0, 25)).await;
        first_release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(25) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(
            transport.calls(),
            vec![(key, Some(16), None), (key, Some(23), None)]
        );

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        );
    }

    #[tokio::test]
    async fn test_pending_duplicate_is_stale_dropped_during_drain() {
        let (client, transport, kv_indexer) = make_test_client("pending-duplicate").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(15);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (16..=20).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 20,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 20)).await;
        started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 21)).await;
        client.handle_live_event(make_store_event(1, 0, 21)).await;
        client.handle_live_event(make_store_event(1, 0, 22)).await;
        release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(22) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.calls(), vec![(key, Some(16), None)]);

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![16, 17, 18, 19, 20, 21, 22]
        );
    }

    #[tokio::test]
    async fn test_missing_buffered_head_schedules_follow_up_and_replays_tail() {
        let (client, transport, kv_indexer) = make_test_client("missing-buffered-head").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (11..=15).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 15,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 16)],
                    last_event_id: 16,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        first_started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 17)).await;
        client.handle_live_event(make_store_event(1, 0, 18)).await;
        first_release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(18) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(
            transport.calls(),
            vec![(key, Some(11), None), (key, Some(16), None)]
        );

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![11, 12, 13, 14, 15, 16, 17, 18]
        );
    }

    #[tokio::test]
    async fn test_pending_starts_after_next_event_recovers_gap_then_replays_tail() {
        let (client, transport, kv_indexer) = make_test_client("pending-starts-after-next").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(15);
        }

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (16..=20).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 20,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 21)],
                    last_event_id: 21,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 20)).await;
        first_started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 22)).await;
        client.handle_live_event(make_store_event(1, 0, 23)).await;
        first_release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(23) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(
            transport.calls(),
            vec![(key, Some(16), None), (key, Some(21), None)]
        );

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![16, 17, 18, 19, 20, 21, 22, 23]
        );
    }

    #[tokio::test]
    async fn test_failed_recovery_clears_pending_without_applying_buffered_events() {
        let (client, transport, kv_indexer) =
            make_test_client("failed-recovery-clears-pending").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Error("query failed".to_string())),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 16)).await;
        release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(10)
                    && !state.recovery_inflight
                    && state.pending_live_events.is_empty()
            })
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(events.is_empty());
        assert_eq!(transport.calls(), vec![(key, Some(11), None)]);
    }

    #[tokio::test]
    async fn test_initial_restore_updates_cursor_for_live_and_gap_paths() {
        let (client, transport, kv_indexer) = make_test_client("initial-restore-cursor").await;
        let key = (1, 0);

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 10,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 12), make_store_event(1, 0, 13)],
                    last_event_id: 13,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(10) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.call_count(), 1);

        client.handle_live_event(make_store_event(1, 0, 11)).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(11) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.call_count(), 1);

        client.handle_live_event(make_store_event(1, 0, 13)).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(13) && !state.recovery_inflight
            })
        })
        .await;
        assert_eq!(transport.call_count(), 2);

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes(&events), vec![11, 12, 13]);
    }

    #[tokio::test]
    async fn test_initial_restore_tree_dump_with_safe_tail_advances_cursor() {
        let (client, transport, kv_indexer) = make_test_client("initial-restore-safe-tail").await;
        let key = (1, 0);

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![make_store_event(1, 0, 0), make_store_event(1, 0, 11)],
                    last_event_id: 11,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(11) && !state.recovery_inflight
            })
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes(&events), vec![0, 11]);
        assert_eq!(transport.call_count(), 1);
    }

    #[tokio::test]
    async fn test_tree_dump_replaces_stale_state_for_recovered_rank() {
        let (client, transport, kv_indexer) = make_test_client("tree-dump-replaces-rank").await;
        let key = (1, 0);

        kv_indexer.apply_event(make_store_event(1, 0, 90)).await;
        kv_indexer.apply_event(make_store_event(1, 0, 91)).await;
        kv_indexer.flush().await;

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![make_store_event(1, 0, 11)],
                    last_event_id: 11,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(11) && !state.recovery_inflight
            })
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes_for(&events, 1, 0), vec![11]);
    }

    #[tokio::test]
    async fn test_tree_dump_recovery_does_not_clear_other_dp_ranks() {
        let (client, transport, kv_indexer) = make_test_client("tree-dump-preserves-sibling").await;
        let key = (1, 0);

        kv_indexer.apply_event(make_store_event(1, 0, 90)).await;
        kv_indexer.apply_event(make_store_event(1, 1, 77)).await;
        kv_indexer.flush().await;

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![make_store_event(1, 0, 11)],
                    last_event_id: 11,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(11) && !state.recovery_inflight
            })
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes_for(&events, 1, 0), vec![11]);
        assert_eq!(stored_block_hashes_for(&events, 1, 1), vec![77]);
    }

    #[tokio::test]
    async fn test_empty_tree_dump_clears_only_recovered_rank() {
        let (client, transport, kv_indexer) = make_test_client("tree-dump-empty-clears-rank").await;
        let key = (1, 0);

        kv_indexer.apply_event(make_store_event(1, 0, 90)).await;
        kv_indexer.apply_event(make_store_event(1, 1, 77)).await;
        kv_indexer.flush().await;

        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::TreeDump {
                    events: vec![],
                    last_event_id: 11,
                }),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(11) && !state.recovery_inflight
            })
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(stored_block_hashes_for(&events, 1, 0).is_empty());
        assert_eq!(stored_block_hashes_for(&events, 1, 1), vec![77]);
    }

    #[tokio::test]
    async fn test_live_event_for_other_worker_is_not_blocked_by_inflight_recovery() {
        let (client, transport, kv_indexer) = make_test_client("live-concurrency").await;

        let delayed_key = (1, 0);
        {
            let worker_state = client.get_or_create_worker_state(delayed_key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(delayed_key.1).or_default().cursor = CursorState::Live(10);
        }
        let other_key = (2, 0);
        {
            let worker_state = client.get_or_create_worker_state(other_key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(other_key.1).or_default().cursor = CursorState::Live(20);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            delayed_key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![
                        make_store_event(1, 0, 11),
                        make_store_event(1, 0, 12),
                        make_store_event(1, 0, 13),
                    ],
                    last_event_id: 13,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 13)).await;
        started.notified().await;
        client.handle_live_event(make_store_event(2, 0, 21)).await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(events.iter().any(|event| {
            event.worker_id == 2
                && event.event.dp_rank == 0
                && matches!(
                    &event.event.data,
                    KvCacheEventData::Stored(data)
                        if data.blocks.first().map(|block| block.block_hash.0) == Some(21)
                )
        }));

        release.notify_waiters();
    }

    #[tokio::test]
    async fn test_worker_removal_discards_late_recovery_result() {
        let (client, transport, kv_indexer) = make_test_client("remove-race").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 11), make_store_event(1, 0, 12)],
                    last_event_id: 12,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 12)).await;
        started.notified().await;
        client.handle_removed_worker_dp(1, 0).await;
        release.notify_waiters();

        wait_for(|| !rank_state_matches(&client, key, |_| true)).await;
        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_live_cleared_invalidates_inflight_recovery_without_restore() {
        let (client, transport, kv_indexer) = make_test_client("live-cleared-no-restore").await;
        let key0 = (1, 0);
        let key1 = (1, 1);

        {
            let worker_state = client.get_or_create_worker_state(1);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(0).or_default().cursor = CursorState::Live(10);
            worker_state.ranks.entry(1).or_default().cursor = CursorState::Live(20);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key0,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![
                        make_store_event(1, 0, 11),
                        make_store_event(1, 0, 12),
                        make_store_event(1, 0, 13),
                    ],
                    last_event_id: 13,
                }),
            },
        );
        client.handle_live_event(make_store_event(1, 0, 13)).await;
        started.notified().await;
        client.handle_live_event(make_clear_event(1, 0, 14)).await;

        wait_for(|| transport.call_count() == 1).await;
        release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key0, |state| {
                state.last_applied_id() == Some(14) && !state.recovery_inflight
            }) && rank_state_matches(&client, key1, |state| {
                state.last_applied_id() == Some(20) && !state.recovery_inflight
            })
        })
        .await;
        assert!(rank_state_matches(&client, key1, |state| {
            matches!(state.cursor, CursorState::InvalidatedByBarrier(Some(20)))
        }));

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        client.handle_live_event(make_store_event(1, 1, 30)).await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(transport.call_count(), 1);
        assert_eq!(stored_block_hashes(&events), vec![15, 30]);
    }

    #[tokio::test]
    async fn test_recovered_cleared_resumes_live_without_restore() {
        let (client, transport, kv_indexer) =
            make_test_client("recovered-cleared-no-restore").await;
        let key0 = (1, 0);
        let key1 = (1, 1);

        {
            let worker_state = client.get_or_create_worker_state(1);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(0).or_default().cursor = CursorState::Live(10);
            worker_state.ranks.entry(1).or_default().cursor = CursorState::Live(20);
        }

        transport.push_action(
            key0,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![
                        make_store_event(1, 0, 11),
                        make_clear_event(1, 0, 12),
                        make_store_event(1, 0, 13),
                    ],
                    last_event_id: 13,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 13)).await;

        wait_for(|| {
            rank_state_matches(&client, key0, |state| {
                state.last_applied_id() == Some(13) && !state.recovery_inflight
            }) && rank_state_matches(&client, key1, |state| {
                state.last_applied_id() == Some(20) && !state.recovery_inflight
            })
        })
        .await;
        assert!(rank_state_matches(&client, key1, |state| {
            matches!(state.cursor, CursorState::InvalidatedByBarrier(Some(20)))
        }));

        assert_eq!(transport.call_count(), 1);

        client.handle_live_event(make_store_event(1, 0, 14)).await;
        client.handle_live_event(make_store_event(1, 1, 30)).await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes(&events), vec![13, 14, 30]);
    }

    #[tokio::test]
    async fn test_recovered_cleared_follows_coalesced_live_tail() {
        let (client, transport, kv_indexer) = make_test_client("recovered-cleared-live-tail").await;
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            let mut worker_state = worker_state.lock().await;
            worker_state.ranks.entry(key.1).or_default().cursor = CursorState::Live(10);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![
                        make_store_event(1, 0, 11),
                        make_clear_event(1, 0, 12),
                        make_store_event(1, 0, 13),
                    ],
                    last_event_id: 13,
                }),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: vec![make_store_event(1, 0, 14), make_store_event(1, 0, 15)],
                    last_event_id: 15,
                }),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 13)).await;
        started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 15)).await;
        release.notify_waiters();

        wait_for(|| {
            rank_state_matches(&client, key, |state| {
                state.last_applied_id() == Some(15)
                    && !state.recovery_inflight
                    && state.pending_live_events.is_empty()
            })
        })
        .await;
        assert_eq!(
            transport.calls(),
            vec![(key, Some(11), None), (key, Some(14), None)]
        );

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes(&events), vec![13, 14, 15]);
    }
}
