// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_runtime::component::{Component, Instance};
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, EndpointInstanceId,
};
use dynamo_runtime::pipeline::{
    AddressedPushRouter, AddressedRequest, AsyncEngine, AsyncEngineContextProvider, ManyOut,
    ResponseStream, SingleIn, network::Ingress,
};
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_runtime::stream;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{Mutex, Semaphore};

use super::Indexer;
use crate::kv_router::{
    worker_kv_indexer_query_endpoint, worker_kv_indexer_query_endpoint_for_worker,
};
use dynamo_kv_router::{
    indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse},
    protocols::{DpRank, KvCacheEventData, RouterEvent, WorkerId},
    recovery::{CursorObservation, CursorState},
};

// Recovery retry configuration
const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;
const RECOVERY_CONCURRENCY_LIMIT: usize = 16;

/// Prefix for worker KV indexer query endpoint names.
const QUERY_ENDPOINT_PREFIX: &str = "worker_kv_indexer_query_dp";

type RecoveryKey = (WorkerId, DpRank);

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoveredQueryEndpoint {
    worker_id: WorkerId,
    dp_rank: DpRank,
    target: Instance,
}

#[derive(Debug, Default)]
struct WorkerQueryEndpointDirectory {
    targets: DashMap<RecoveryKey, Instance>,
}

impl WorkerQueryEndpointDirectory {
    fn parse_endpoint_name(
        endpoint_name: &str,
        route_worker_id: WorkerId,
    ) -> Option<(WorkerId, DpRank)> {
        let suffix = endpoint_name.strip_prefix(QUERY_ENDPOINT_PREFIX)?;
        let (dp_rank, worker_id) = match suffix.split_once("_worker") {
            Some((dp_rank, worker_id)) => (dp_rank.parse().ok()?, worker_id.parse().ok()?),
            None => (suffix.parse().ok()?, route_worker_id),
        };
        Some((worker_id, dp_rank))
    }

    fn parse_added(instance: DiscoveryInstance) -> Option<DiscoveredQueryEndpoint> {
        let DiscoveryInstance::Endpoint(inst) = instance else {
            return None;
        };
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&inst.endpoint, inst.instance_id)?;
        Some(DiscoveredQueryEndpoint {
            worker_id,
            dp_rank,
            target: inst,
        })
    }

    fn parse_removed(id: DiscoveryInstanceId) -> Option<(WorkerId, DpRank, EndpointInstanceId)> {
        let DiscoveryInstanceId::Endpoint(eid) = id else {
            return None;
        };
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&eid.endpoint, eid.instance_id)?;
        Some((worker_id, dp_rank, eid))
    }

    fn insert(&self, endpoint: &DiscoveredQueryEndpoint) -> Option<Instance> {
        self.targets.insert(
            (endpoint.worker_id, endpoint.dp_rank),
            endpoint.target.clone(),
        )
    }

    fn target_for(&self, worker_id: WorkerId, dp_rank: DpRank) -> Option<Instance> {
        self.targets
            .get(&(worker_id, dp_rank))
            .map(|target| target.value().clone())
    }

    #[cfg(test)]
    fn remove_dp(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.targets.remove(&(worker_id, dp_rank));
    }

    fn remove_if_matches(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        endpoint_id: &EndpointInstanceId,
    ) -> bool {
        let key = (worker_id, dp_rank);
        let should_remove = self
            .targets
            .get(&key)
            .is_some_and(|target| target.value().endpoint_instance_id() == *endpoint_id);
        if should_remove {
            self.targets.remove(&key);
        }
        should_remove
    }
}

#[derive(Debug, Default)]
struct RankState {
    cursor: CursorState,
    max_seen_live_id: Option<u64>,
    recovery_inflight: bool,
}

impl RankState {
    fn last_applied_id(&self) -> Option<u64> {
        self.cursor.last_applied_id()
    }

    fn observe_live_id(&mut self, event_id: u64) {
        self.max_seen_live_id = Some(self.max_seen_live_id.unwrap_or(0).max(event_id));
    }

    fn clear_max_seen_if_caught_up(&mut self, last_applied_id: u64) {
        if self
            .max_seen_live_id
            .is_some_and(|max_seen| max_seen <= last_applied_id)
        {
            self.max_seen_live_id = None;
        }
    }
}

#[derive(Debug, Default)]
struct WorkerState {
    epoch: u64,
    ranks: HashMap<DpRank, RankState>,
}

#[async_trait]
trait WorkerQueryTransport: Send + Sync {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse>;

    async fn cancel_instance_streams(&self, _endpoint_id: &EndpointInstanceId) -> usize {
        0
    }

    async fn clear_instance_tombstone(&self, _endpoint_id: &EndpointInstanceId) {}
}

struct RuntimeWorkerQueryTransport {
    addressed: Arc<AddressedPushRouter>,
}

impl RuntimeWorkerQueryTransport {
    async fn new(component: &Component) -> Result<Self> {
        Ok(Self {
            addressed: AddressedPushRouter::from_runtime_provider(component).await?,
        })
    }
}

#[async_trait]
impl WorkerQueryTransport for RuntimeWorkerQueryTransport {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let request = WorkerKvQueryRequest {
            worker_id,
            dp_rank,
            start_event_id,
            end_event_id,
        };
        let instance = target;
        let instance_id = instance.instance_id;
        let endpoint_name = instance.endpoint.clone();
        let addressed_request =
            SingleIn::new(request).map(|req| AddressedRequest::for_instance(req, instance));
        let mut stream: ManyOut<WorkerKvQueryResponse> = self
            .addressed
            .generate(addressed_request)
            .await
            .with_context(|| {
                format!(
                    "Failed to send worker KV query to worker {worker_id} dp_rank {dp_rank} \
                     via endpoint {endpoint_name} instance {instance_id}"
                )
            })?;

        let response = stream
            .next()
            .await
            .context("Worker KV query returned an empty response stream")?;

        if let Some(err) = response.err() {
            return Err(err).context("Worker KV query response error");
        }

        Ok(response)
    }

    async fn cancel_instance_streams(&self, endpoint_id: &EndpointInstanceId) -> usize {
        self.addressed.cancel_instance_streams(endpoint_id).await
    }

    async fn clear_instance_tombstone(&self, endpoint_id: &EndpointInstanceId) {
        self.addressed.clear_instance_tombstone(endpoint_id).await;
    }
}

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
    query_endpoints: WorkerQueryEndpointDirectory,
    recovery_semaphore: Arc<Semaphore>,
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
            query_endpoints: WorkerQueryEndpointDirectory::default(),
            recovery_semaphore: Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)),
        })
    }

    /// Create a new WorkerQueryClient and spawn its background discovery loop.
    ///
    /// The background loop watches `ComponentEndpoints` discovery for query endpoints,
    /// recovers each `(worker_id, dp_rank)` as it appears, and sends worker removal
    /// events when all dp_ranks for a worker disappear.
    pub async fn spawn(component: Component, indexer: Indexer) -> Result<Arc<Self>> {
        let transport = Arc::new(RuntimeWorkerQueryTransport::new(&component).await?);
        let client = Self::new(component.clone(), indexer, transport);

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
            if replaced.is_some() {
                worker_state.epoch += 1;
                worker_state.ranks.insert(dp_rank, RankState::default());
                self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            let rank_state = worker_state.ranks.entry(dp_rank).or_default();
            if matches!(rank_state.cursor, CursorState::Initial) && !rank_state.recovery_inflight {
                tracing::info!(
                    "WorkerQueryClient: discovered worker {worker_id} dp_rank {dp_rank}, scheduling restore"
                );
                rank_state.recovery_inflight = true;
                Some(worker_state.epoch)
            } else {
                None
            }
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
            if worker_state.ranks.remove(&dp_rank).is_none() {
                return;
            }
            worker_state.ranks.is_empty()
        };

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

        worker_state.epoch += 1;
        for rank_state in worker_state.ranks.values_mut() {
            let post_clear_max_seen = rank_state
                .max_seen_live_id
                .filter(|max_seen| *max_seen > clear_event_id);
            rank_state.cursor = rank_state.cursor.invalidate_by_barrier();
            rank_state.max_seen_live_id = post_clear_max_seen;
            rank_state.recovery_inflight = false;
        }

        let rank_state = worker_state.ranks.entry(clear_dp_rank).or_default();
        rank_state.cursor = rank_state.cursor.apply_barrier(clear_event_id);

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
        let event_id = event.event.event_id;
        let key = (worker_id, dp_rank);

        enum Action {
            ApplyDirect,
            SpawnFullRestore { epoch: u64 },
            SpawnIncremental { epoch: u64, start_event_id: u64 },
        }

        let action = {
            let worker_state = self.get_or_create_worker_state(worker_id);
            let mut worker_state = worker_state.lock().await;
            let rank_state = worker_state.ranks.entry(dp_rank).or_default();

            // `Cleared` is worker-wide in the indexer, so it bypasses the per-rank gap logic and
            // instead installs a worker barrier that invalidates any inflight recovery.
            if matches!(&event.event.data, KvCacheEventData::Cleared) {
                if rank_state
                    .last_applied_id()
                    .is_none_or(|last_applied_id| event_id > last_applied_id)
                {
                    self.apply_worker_clear_locked(&mut worker_state, event)
                        .await;
                }
                // Already applied the event, so no further action needed.
                return;
            }

            match rank_state.cursor.observe(event_id) {
                CursorObservation::Stale { .. } => return,
                observation if rank_state.recovery_inflight => {
                    match observation {
                        CursorObservation::Initial { .. }
                        | CursorObservation::Contiguous { .. }
                        | CursorObservation::Gap { .. }
                        | CursorObservation::FreshAfterBarrier { .. } => {
                            rank_state.observe_live_id(event_id);
                        }
                        CursorObservation::Stale { .. } => {}
                    }
                    return;
                }
                CursorObservation::Initial { .. } => {
                    rank_state.observe_live_id(event_id);
                    rank_state.recovery_inflight = true;
                    Action::SpawnFullRestore {
                        epoch: worker_state.epoch,
                    }
                }
                CursorObservation::Gap { expected, .. } => {
                    rank_state.observe_live_id(event_id);
                    rank_state.recovery_inflight = true;
                    Action::SpawnIncremental {
                        epoch: worker_state.epoch,
                        start_event_id: expected,
                    }
                }
                CursorObservation::Contiguous { got }
                | CursorObservation::FreshAfterBarrier { got, .. } => {
                    rank_state.cursor = rank_state.cursor.advance_to(got);
                    rank_state.clear_max_seen_if_caught_up(got);
                    Action::ApplyDirect
                }
            }
        };

        match action {
            Action::ApplyDirect => {
                self.indexer.apply_event(event).await;
            }
            Action::SpawnFullRestore { epoch } => {
                self.spawn_recovery_task(key, epoch, None, None);
            }
            Action::SpawnIncremental {
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
            // Add jitter only for full-restore (start_event_id is None)
            // to permute semaphore acquisition order and reduce thundering herd risk on initial discovery.
            // This distributes load when multiple routers start simultaneously.
            if start_event_id.is_none() {
                let jitter_us = rand::rng().random_range(0..3000u64);
                tokio::time::sleep(Duration::from_micros(jitter_us)).await;
            }

            let Ok(_permit) = client.recovery_semaphore.clone().acquire_owned().await else {
                return;
            };

            let result = client
                .fetch_recovery_response(key.0, key.1, start_event_id, end_event_id)
                .await;
            client.finish_recovery_task(key, epoch, result).await;
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

        let Some(rank_state) = worker_state.ranks.get(&key.1) else {
            return;
        };

        let mut new_cursor = rank_state.cursor;
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
                tracing::info!(
                    "Got tree dump from worker {} dp_rank {} (range too old or unspecified), count: {}, last_event_id: {}",
                    key.0,
                    key.1,
                    events.len(),
                    last_event_id
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
        {
            let rank_state = worker_state
                .ranks
                .get_mut(&key.1)
                .expect("rank state should exist while finishing recovery");
            rank_state.recovery_inflight = false;
            rank_state.cursor = new_cursor;

            let last_applied_id = rank_state.last_applied_id().unwrap_or(0);
            rank_state.clear_max_seen_if_caught_up(last_applied_id);

            if successful_response
                && rank_state
                    .max_seen_live_id
                    .is_some_and(|max_seen| max_seen > last_applied_id)
            {
                rank_state.recovery_inflight = true;
                follow_up_start = Some(last_applied_id.saturating_add(1));
            }
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

/// Worker-side endpoint registration for Router -> LocalKvIndexer query service
pub(crate) async fn start_worker_kv_query_endpoint(
    component: Component,
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
) {
    let engine = Arc::new(WorkerKvQueryEngine {
        worker_id,
        dp_rank,
        local_indexer,
        processing_semaphore: Semaphore::new(1),
    });

    let ingress = match Ingress::for_engine(engine) {
        Ok(ingress) => ingress,
        Err(e) => {
            tracing::error!(
                "Failed to build WorkerKvQuery endpoint handler for worker {worker_id} dp_rank {dp_rank}: {e}"
            );
            return;
        }
    };

    let route_worker_id = component.drt().connection_id();
    let endpoint_name = if route_worker_id == worker_id {
        worker_kv_indexer_query_endpoint(dp_rank)
    } else {
        worker_kv_indexer_query_endpoint_for_worker(worker_id, dp_rank)
    };
    tracing::info!(
        "WorkerKvQuery endpoint starting for worker {worker_id} dp_rank {dp_rank} \
         routed by instance {route_worker_id} on endpoint '{endpoint_name}'"
    );

    if let Err(e) = component
        .endpoint(&endpoint_name)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start()
        .await
    {
        tracing::error!(
            "WorkerKvQuery endpoint failed for worker {worker_id} dp_rank {dp_rank}: {e}"
        );
    }
}

struct WorkerKvQueryEngine {
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
    /// Semaphore limiting concurrent recovery request processing to 1.
    /// Prevents multiple routers from overwhelming the worker with heavy tree dump operations.
    processing_semaphore: Semaphore,
}

#[async_trait]
impl AsyncEngine<SingleIn<WorkerKvQueryRequest>, ManyOut<WorkerKvQueryResponse>, anyhow::Error>
    for WorkerKvQueryEngine
{
    async fn generate(
        &self,
        request: SingleIn<WorkerKvQueryRequest>,
    ) -> anyhow::Result<ManyOut<WorkerKvQueryResponse>> {
        let (request, ctx) = request.into_parts();

        tracing::debug!(
            "Received query request for worker {}: {:?}",
            self.worker_id,
            request
        );

        if request.worker_id != self.worker_id {
            let error_message = format!(
                "WorkerKvQueryEngine::generate worker_id mismatch: request.worker_id={} this.worker_id={}",
                request.worker_id, self.worker_id
            );
            let response = WorkerKvQueryResponse::Error(error_message);
            return Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![response])),
                ctx.context(),
            ));
        }

        if request.dp_rank != self.dp_rank {
            let error_message = format!(
                "WorkerKvQueryEngine::generate dp_rank mismatch: request.dp_rank={} this.dp_rank={}",
                request.dp_rank, self.dp_rank
            );
            let response = WorkerKvQueryResponse::Error(error_message);
            return Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![response])),
                ctx.context(),
            ));
        }

        // Check if this request can likely be served from buffer (fast path).
        // If not, acquire semaphore for tree dump (heavy operation).
        let likely_buffer_read = self
            .local_indexer
            .likely_served_from_buffer(request.start_event_id);

        let _maybe_permit = if !likely_buffer_read {
            // Acquire semaphore permit before processing tree dump.
            // This prevents multiple heavy tree dump operations from running concurrently
            let engine_ctx = ctx.context();
            let permit = tokio::select! {
                result = self.processing_semaphore.acquire() => {
                    result.map_err(|_| anyhow::anyhow!("Worker KV query semaphore closed"))?
                }
                _ = futures::future::select(engine_ctx.stopped(), engine_ctx.killed()) => {
                    tracing::warn!("Worker<>Router KV query request cancelled while waiting for semaphore");
                    return Ok(ResponseStream::new(
                        // this response will be dropped on the router side since the request was cancelled,
                        // but we return it here to satisfy the function signature and provide some context in logs if it does get processed for some reason.
                        Box::pin(stream::iter(vec![WorkerKvQueryResponse::Error(
                            "Request cancelled by client".to_string(),
                        )])),
                        ctx.context(),
                    ));
                }
            };
            Some(permit)
        } else {
            // Fast buffer read - no semaphore needed
            None
        };

        // Start slow-query logging only once the request is actively executing the slow path.
        // Queued requests waiting on the semaphore should remain silent.
        let _slow_query_guard = if !likely_buffer_read {
            Some(SlowQueryGuard::spawn(self.worker_id))
        } else {
            None
        };

        let response = self
            .local_indexer
            .get_events_in_id_range(request.start_event_id, request.end_event_id)
            .await;

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

/// RAII guard that aborts a slow query logger task on drop.
struct SlowQueryGuard(tokio::task::JoinHandle<()>);

impl SlowQueryGuard {
    fn spawn(worker_id: u64) -> Self {
        Self(tokio::spawn(async move {
            let mut elapsed_secs = 0u64;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                elapsed_secs += 5;
                tracing::warn!(
                    worker_id,
                    elapsed_secs,
                    "Worker KV query still running - possible slow tree dump",
                );
            }
        }))
    }
}

impl Drop for SlowQueryGuard {
    fn drop(&mut self) {
        self.0.abort();
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
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events {
                    events: (16..=18).map(|id| make_store_event(1, 0, id)).collect(),
                    last_event_id: 18,
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
                state.last_applied_id() == Some(15) && !state.recovery_inflight
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
