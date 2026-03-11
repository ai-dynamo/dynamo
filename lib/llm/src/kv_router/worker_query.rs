// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, PushRouter, ResponseStream, RouterMode,
    SingleIn, network::Ingress,
};
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_runtime::stream;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use tokio::sync::{Mutex, Semaphore};

use crate::kv_router::Indexer;
use crate::kv_router::indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::protocols::{DpRank, RouterEvent, WorkerId};
use crate::kv_router::worker_kv_indexer_query_endpoint;

// Recovery retry configuration
const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;
const RECOVERY_CONCURRENCY_LIMIT: usize = 16;

/// Prefix for worker KV indexer query endpoint names.
const QUERY_ENDPOINT_PREFIX: &str = "worker_kv_indexer_query_dp";

type RecoveryKey = (WorkerId, DpRank);

#[derive(Debug, Default)]
struct RecoveryState {
    last_applied_id: Option<u64>,
    max_seen_live_id: Option<u64>,
    recovery_inflight: bool,
}

#[derive(Debug, Default)]
struct WorkerState {
    epoch: u64,
    known_dp_ranks: HashSet<DpRank>,
}

#[async_trait]
trait WorkerQueryTransport: Send + Sync {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse>;
}

struct RuntimeWorkerQueryTransport {
    component: Component,
    routers: DashMap<DpRank, Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>>,
}

impl RuntimeWorkerQueryTransport {
    fn new(component: Component) -> Self {
        Self {
            component,
            routers: DashMap::new(),
        }
    }

    async fn get_router_for_dp_rank(
        &self,
        dp_rank: DpRank,
    ) -> Result<Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>> {
        if let Some(router) = self.routers.get(&dp_rank) {
            return Ok(router.clone());
        }

        let endpoint_name = worker_kv_indexer_query_endpoint(dp_rank);
        let endpoint = self.component.endpoint(&endpoint_name);
        let client = endpoint.client().await?;
        let router = Arc::new(
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?,
        );

        Ok(self.routers.entry(dp_rank).or_insert(router).clone())
    }
}

#[async_trait]
impl WorkerQueryTransport for RuntimeWorkerQueryTransport {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let router = self.get_router_for_dp_rank(dp_rank).await?;

        let request = WorkerKvQueryRequest {
            worker_id,
            start_event_id,
            end_event_id,
        };
        let mut stream = router
            .direct(SingleIn::new(request), worker_id)
            .await
            .with_context(|| {
                format!("Failed to send worker KV query to worker {worker_id} dp_rank {dp_rank}")
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
}

/// Router-side client for querying worker local KV indexers.
///
/// Discovers query endpoints via `ComponentEndpoints` discovery, filtering for
/// the `worker_kv_indexer_query_dp{N}` name pattern. Recovers each
/// `(worker_id, dp_rank)` individually as it appears in discovery.
///
/// Also handles worker lifecycle (add/remove) by tracking known endpoints and
/// sending removal events to the router indexer when all dp_ranks for a worker
/// disappear.
pub struct WorkerQueryClient {
    component: Component,
    transport: Arc<dyn WorkerQueryTransport>,
    /// Indexer for applying recovered events and worker removals.
    indexer: Indexer,
    recovery_states: DashMap<RecoveryKey, Arc<Mutex<RecoveryState>>>,
    worker_states: DashMap<WorkerId, Arc<Mutex<WorkerState>>>,
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
            recovery_states: DashMap::new(),
            worker_states: DashMap::new(),
            recovery_semaphore: Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)),
        })
    }

    /// Create a new WorkerQueryClient and spawn its background discovery loop.
    ///
    /// The background loop watches `ComponentEndpoints` discovery for query endpoints,
    /// recovers each `(worker_id, dp_rank)` as it appears, and sends worker removal
    /// events when all dp_ranks for a worker disappear.
    pub async fn spawn(component: Component, indexer: Indexer) -> Result<Arc<Self>> {
        let transport = Arc::new(RuntimeWorkerQueryTransport::new(component.clone()));
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

    /// Background loop: watches ComponentEndpoints, recovers per (worker_id, dp_rank).
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
                    let Some((worker_id, dp_rank)) = Self::parse_query_endpoint(&instance) else {
                        continue;
                    };
                    self.handle_discovered_worker(worker_id, dp_rank).await;
                }
                DiscoveryEvent::Removed(id) => {
                    let Some((worker_id, dp_rank)) = Self::parse_instance_id(&id) else {
                        continue;
                    };
                    self.handle_removed_worker_dp(worker_id, dp_rank).await;
                }
            }
        }

        Ok(())
    }

    /// Parse a query endpoint from a discovery instance.
    /// Returns `(worker_id, dp_rank)` if the instance is a query endpoint, else None.
    fn parse_query_endpoint(instance: &DiscoveryInstance) -> Option<(WorkerId, DpRank)> {
        let DiscoveryInstance::Endpoint(inst) = instance else {
            return None;
        };
        let dp_rank = inst.endpoint.strip_prefix(QUERY_ENDPOINT_PREFIX)?;
        let dp_rank: DpRank = dp_rank.parse().ok()?;
        Some((inst.instance_id, dp_rank))
    }

    /// Parse a query endpoint from a discovery instance ID (for removals).
    fn parse_instance_id(
        id: &dynamo_runtime::discovery::DiscoveryInstanceId,
    ) -> Option<(WorkerId, DpRank)> {
        let dynamo_runtime::discovery::DiscoveryInstanceId::Endpoint(eid) = id else {
            return None;
        };
        let dp_rank = eid.endpoint.strip_prefix(QUERY_ENDPOINT_PREFIX)?;
        let dp_rank: DpRank = dp_rank.parse().ok()?;
        Some((eid.instance_id, dp_rank))
    }

    fn get_or_create_recovery_state(&self, key: RecoveryKey) -> Arc<Mutex<RecoveryState>> {
        self.recovery_states
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(RecoveryState::default())))
            .clone()
    }

    fn get_or_create_worker_state(&self, worker_id: WorkerId) -> Arc<Mutex<WorkerState>> {
        self.worker_states
            .entry(worker_id)
            .or_insert_with(|| Arc::new(Mutex::new(WorkerState::default())))
            .clone()
    }

    async fn current_worker_epoch(&self, worker_id: WorkerId) -> u64 {
        let worker_state = self.get_or_create_worker_state(worker_id);
        worker_state.lock().await.epoch
    }

    pub(crate) async fn handle_discovered_worker(
        self: &Arc<Self>,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) {
        let worker_state = self.get_or_create_worker_state(worker_id);
        let is_new = {
            let mut worker_state = worker_state.lock().await;
            worker_state.known_dp_ranks.insert(dp_rank)
        };

        if !is_new {
            return;
        }

        tracing::info!(
            "WorkerQueryClient: discovered worker {worker_id} dp_rank {dp_rank}, scheduling restore"
        );

        let key = (worker_id, dp_rank);
        let recovery_state = self.get_or_create_recovery_state(key);
        let should_spawn = {
            let mut recovery_state = recovery_state.lock().await;
            if recovery_state.last_applied_id.is_none() && !recovery_state.recovery_inflight {
                recovery_state.recovery_inflight = true;
                true
            } else {
                false
            }
        };

        if should_spawn {
            let epoch = self.current_worker_epoch(worker_id).await;
            self.spawn_recovery_task(key, epoch, None, None);
        }
    }

    pub(crate) async fn handle_removed_worker_dp(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.recovery_states.remove(&(worker_id, dp_rank));

        let Some(worker_state) = self
            .worker_states
            .get(&worker_id)
            .map(|entry| entry.clone())
        else {
            return;
        };

        let should_remove_worker = {
            let mut worker_state = worker_state.lock().await;
            if !worker_state.known_dp_ranks.remove(&dp_rank) {
                return;
            }
            if worker_state.known_dp_ranks.is_empty() {
                worker_state.epoch += 1;
                true
            } else {
                false
            }
        };

        if should_remove_worker {
            tracing::warn!("WorkerQueryClient: all dp_ranks gone for worker {worker_id}, removing");
            self.indexer.remove_worker(worker_id).await;
        }
    }

    pub(crate) async fn handle_live_event(self: &Arc<Self>, event: RouterEvent) {
        let worker_id = event.worker_id;
        let dp_rank = event.event.dp_rank;
        let event_id = event.event.event_id;
        let key = (worker_id, dp_rank);
        let recovery_state = self.get_or_create_recovery_state(key);

        enum Action {
            ApplyDirect,
            SpawnFullRestore,
            SpawnIncremental { start_event_id: u64 },
            Drop,
        }

        let action = {
            let mut recovery_state = recovery_state.lock().await;

            match recovery_state.last_applied_id {
                None => {
                    recovery_state.max_seen_live_id = Some(
                        recovery_state
                            .max_seen_live_id
                            .map_or(event_id, |max_seen| max_seen.max(event_id)),
                    );
                    if !recovery_state.recovery_inflight {
                        recovery_state.recovery_inflight = true;
                        Action::SpawnFullRestore
                    } else {
                        Action::Drop
                    }
                }
                Some(last_applied_id) => {
                    if event_id <= last_applied_id {
                        Action::Drop
                    } else if recovery_state.recovery_inflight {
                        recovery_state.max_seen_live_id = Some(
                            recovery_state
                                .max_seen_live_id
                                .map_or(event_id, |max_seen| max_seen.max(event_id)),
                        );
                        Action::Drop
                    } else if event_id > last_applied_id.saturating_add(1) {
                        recovery_state.max_seen_live_id = Some(
                            recovery_state
                                .max_seen_live_id
                                .map_or(event_id, |max_seen| max_seen.max(event_id)),
                        );
                        recovery_state.recovery_inflight = true;
                        Action::SpawnIncremental {
                            start_event_id: last_applied_id.saturating_add(1),
                        }
                    } else {
                        recovery_state.last_applied_id = Some(event_id);
                        if recovery_state
                            .max_seen_live_id
                            .is_some_and(|max_seen| max_seen <= event_id)
                        {
                            recovery_state.max_seen_live_id = None;
                        }
                        Action::ApplyDirect
                    }
                }
            }
        };

        match action {
            Action::ApplyDirect => {
                self.indexer.apply_event(event).await;
            }
            Action::SpawnFullRestore => {
                let epoch = self.current_worker_epoch(worker_id).await;
                self.spawn_recovery_task(key, epoch, None, None);
            }
            Action::SpawnIncremental { start_event_id } => {
                let epoch = self.current_worker_epoch(worker_id).await;
                self.spawn_recovery_task(key, epoch, Some(start_event_id), None);
            }
            Action::Drop => {}
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
        let worker_state = self.get_or_create_worker_state(key.0);
        let worker_state = worker_state.lock().await;
        if worker_state.epoch != epoch {
            tracing::debug!(
                "Discarding stale recovery result for worker {} dp_rank {} due to epoch change",
                key.0,
                key.1
            );
            return;
        }

        let Some(recovery_state) = self.recovery_states.get(&key).map(|entry| entry.clone()) else {
            return;
        };

        let mut new_last_applied_id = {
            let recovery_state = recovery_state.lock().await;
            recovery_state.last_applied_id
        };
        let mut successful_response = false;

        match result {
            Ok(WorkerKvQueryResponse::Events(events)) => {
                tracing::debug!(
                    "Got {count} buffered events from worker {} dp_rank {}",
                    key.0,
                    key.1,
                    count = events.len()
                );
                for event in &events {
                    self.indexer.apply_event(event.clone()).await;
                }
                if let Some(last_event) = events.last() {
                    new_last_applied_id = Some(last_event.event.event_id);
                }
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
                for event in &events {
                    self.indexer.apply_event(event.clone()).await;
                }
                new_last_applied_id = Some(last_event_id);
                successful_response = true;
            }
            Ok(WorkerKvQueryResponse::TooNew {
                requested_start,
                requested_end,
                newest_available,
            }) => {
                tracing::warn!(
                    "Requested range [{requested_start:?}, {requested_end:?}] is newer than available (newest: {newest_available}) for worker {} dp_rank {}",
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
            let mut recovery_state = recovery_state.lock().await;
            recovery_state.recovery_inflight = false;
            recovery_state.last_applied_id = new_last_applied_id;

            let last_applied_id = recovery_state.last_applied_id.unwrap_or(0);
            if recovery_state
                .max_seen_live_id
                .is_some_and(|max_seen| max_seen <= last_applied_id)
            {
                recovery_state.max_seen_live_id = None;
            }

            if successful_response
                && recovery_state
                    .max_seen_live_id
                    .is_some_and(|max_seen| max_seen > last_applied_id)
            {
                recovery_state.recovery_inflight = true;
                follow_up_start = Some(last_applied_id.saturating_add(1));
            }
        }
        drop(worker_state);

        if let Some(start_event_id) = follow_up_start {
            self.spawn_recovery_task(key, epoch, Some(start_event_id), None);
        }
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
            match self
                .transport
                .query_worker(worker_id, dp_rank, start_event_id, end_event_id)
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

// ============================================================================
// Worker-side endpoint registration (unchanged)
// ============================================================================

/// Worker-side endpoint registration for Router -> LocalKvIndexer query service
pub(crate) async fn start_worker_kv_query_endpoint(
    component: Component,
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
) {
    let engine = Arc::new(WorkerKvQueryEngine {
        worker_id,
        local_indexer,
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

    let endpoint_name = worker_kv_indexer_query_endpoint(dp_rank);
    tracing::info!(
        "WorkerKvQuery endpoint starting for worker {worker_id} dp_rank {dp_rank} on endpoint '{endpoint_name}'"
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
    local_indexer: Arc<LocalKvIndexer>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::Indexer;
    use crate::kv_router::RouterEvent;
    use crate::kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
    use crate::kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash,
    };
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
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
        calls: Arc<StdMutex<Vec<(RecoveryKey, Option<u64>, Option<u64>)>>>,
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
    }

    #[async_trait]
    impl WorkerQueryTransport for MockWorkerQueryTransport {
        async fn query_worker(
            &self,
            worker_id: WorkerId,
            dp_rank: DpRank,
            start_event_id: Option<u64>,
            end_event_id: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            let key = (worker_id, dp_rank);
            self.calls
                .lock()
                .unwrap()
                .push((key, start_event_id, end_event_id));

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
        (kv_indexer.clone(), Indexer::KvIndexer(kv_indexer))
    }

    fn make_store_event(worker_id: WorkerId, dp_rank: DpRank, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
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
            local_indexer,
        };

        let request = WorkerKvQueryRequest {
            worker_id,
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
            WorkerKvQueryResponse::Events(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].event.event_id, 1);
            }
            other => panic!("Unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_discovery_restore_does_not_block_other_workers() {
        let component = make_test_component("discovery-concurrency").await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());

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
        let component = make_test_component("high-water").await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            worker_state.lock().await.known_dp_ranks.insert(key.1);
            let recovery_state = client.get_or_create_recovery_state(key);
            recovery_state.lock().await.last_applied_id = Some(10);
        }

        let first_started = Arc::new(Notify::new());
        let first_release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(first_started.clone()),
                release: Some(first_release.clone()),
                response: Ok(WorkerKvQueryResponse::Events(
                    (11..=15).map(|id| make_store_event(1, 0, id)).collect(),
                )),
            },
        );
        transport.push_action(
            key,
            MockQueryAction {
                started: None,
                release: None,
                response: Ok(WorkerKvQueryResponse::Events(
                    (16..=18).map(|id| make_store_event(1, 0, id)).collect(),
                )),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 15)).await;
        first_started.notified().await;
        client.handle_live_event(make_store_event(1, 0, 16)).await;
        client.handle_live_event(make_store_event(1, 0, 17)).await;
        client.handle_live_event(make_store_event(1, 0, 18)).await;
        first_release.notify_waiters();

        wait_for(|| {
            client
                .recovery_states
                .get(&key)
                .map(|state| {
                    matches!(
                        state.try_lock(),
                        Ok(ref guard)
                            if guard.last_applied_id == Some(18) && !guard.recovery_inflight
                    )
                })
                .unwrap_or(false)
        })
        .await;

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(
            stored_block_hashes(&events),
            vec![11, 12, 13, 14, 15, 16, 17, 18]
        );
    }

    #[tokio::test]
    async fn test_initial_restore_updates_cursor_for_live_and_gap_paths() {
        let component = make_test_component("initial-restore-cursor").await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());
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
                response: Ok(WorkerKvQueryResponse::Events(vec![
                    make_store_event(1, 0, 12),
                    make_store_event(1, 0, 13),
                ])),
            },
        );

        client.handle_discovered_worker(1, 0).await;
        wait_for(|| {
            client
                .recovery_states
                .get(&key)
                .map(|state| {
                    matches!(
                        state.try_lock(),
                        Ok(ref guard)
                            if guard.last_applied_id == Some(10) && !guard.recovery_inflight
                    )
                })
                .unwrap_or(false)
        })
        .await;
        assert_eq!(transport.call_count(), 1);

        client.handle_live_event(make_store_event(1, 0, 11)).await;
        wait_for(|| {
            client
                .recovery_states
                .get(&key)
                .map(|state| {
                    matches!(
                        state.try_lock(),
                        Ok(ref guard)
                            if guard.last_applied_id == Some(11) && !guard.recovery_inflight
                    )
                })
                .unwrap_or(false)
        })
        .await;
        assert_eq!(transport.call_count(), 1);

        client.handle_live_event(make_store_event(1, 0, 13)).await;
        wait_for(|| {
            client
                .recovery_states
                .get(&key)
                .map(|state| {
                    matches!(
                        state.try_lock(),
                        Ok(ref guard)
                            if guard.last_applied_id == Some(13) && !guard.recovery_inflight
                    )
                })
                .unwrap_or(false)
        })
        .await;
        assert_eq!(transport.call_count(), 2);

        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert_eq!(stored_block_hashes(&events), vec![11, 12, 13]);
    }

    #[tokio::test]
    async fn test_live_event_for_other_worker_is_not_blocked_by_inflight_recovery() {
        let component = make_test_component("live-concurrency").await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());

        let delayed_key = (1, 0);
        {
            let worker_state = client.get_or_create_worker_state(delayed_key.0);
            worker_state
                .lock()
                .await
                .known_dp_ranks
                .insert(delayed_key.1);
            let recovery_state = client.get_or_create_recovery_state(delayed_key);
            recovery_state.lock().await.last_applied_id = Some(10);
        }
        let other_key = (2, 0);
        {
            let worker_state = client.get_or_create_worker_state(other_key.0);
            worker_state.lock().await.known_dp_ranks.insert(other_key.1);
            let recovery_state = client.get_or_create_recovery_state(other_key);
            recovery_state.lock().await.last_applied_id = Some(20);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            delayed_key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events(vec![
                    make_store_event(1, 0, 11),
                    make_store_event(1, 0, 12),
                    make_store_event(1, 0, 13),
                ])),
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
        let component = make_test_component("remove-race").await;
        let (kv_indexer, indexer) = make_test_indexer();
        let transport = Arc::new(MockWorkerQueryTransport::default());
        let client = WorkerQueryClient::new(component, indexer, transport.clone());
        let key = (1, 0);

        {
            let worker_state = client.get_or_create_worker_state(key.0);
            worker_state.lock().await.known_dp_ranks.insert(key.1);
            let recovery_state = client.get_or_create_recovery_state(key);
            recovery_state.lock().await.last_applied_id = Some(10);
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        transport.push_action(
            key,
            MockQueryAction {
                started: Some(started.clone()),
                release: Some(release.clone()),
                response: Ok(WorkerKvQueryResponse::Events(vec![
                    make_store_event(1, 0, 11),
                    make_store_event(1, 0, 12),
                ])),
            },
        );

        client.handle_live_event(make_store_event(1, 0, 12)).await;
        started.notified().await;
        client.handle_removed_worker_dp(1, 0).await;
        release.notify_waiters();

        wait_for(|| !client.recovery_states.contains_key(&key)).await;
        kv_indexer.flush().await;
        let events = kv_indexer.dump_events().await.unwrap();
        assert!(events.is_empty());
    }
}
