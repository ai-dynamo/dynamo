// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use dashmap::DashMap;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, PushRouter, ResponseStream, RouterMode,
    SingleIn, async_trait, network::Ingress,
};
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_runtime::stream;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;

use crate::kv_router::Indexer;
use crate::kv_router::indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::protocols::{DpRank, WorkerId};
use crate::kv_router::worker_kv_indexer_query_endpoint;

// Recovery retry configuration
const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;

/// Prefix for worker KV indexer query endpoint names.
const QUERY_ENDPOINT_PREFIX: &str = "worker_kv_indexer_query_dp";

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
    /// Routers keyed by dp_rank — each dp_rank has its own endpoint. Created lazily.
    routers: Arc<DashMap<DpRank, Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>>>,
    /// Indexer for applying recovered events and worker removals.
    indexer: Indexer,
}

impl WorkerQueryClient {
    /// Create a new WorkerQueryClient and spawn its background discovery loop.
    ///
    /// The background loop watches `ComponentEndpoints` discovery for query endpoints,
    /// recovers each `(worker_id, dp_rank)` as it appears, and sends worker removal
    /// events when all dp_ranks for a worker disappear.
    pub async fn spawn(component: Component, indexer: Indexer) -> Result<Arc<Self>> {
        let client = Arc::new(Self {
            component: component.clone(),
            routers: Arc::new(DashMap::new()),
            indexer,
        });

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
        &self,
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

        // Track known (worker_id, dp_rank) pairs to detect removals
        let mut known: HashMap<WorkerId, HashSet<DpRank>> = HashMap::new();

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

                    if known.entry(worker_id).or_default().insert(dp_rank) {
                        tracing::info!(
                            "WorkerQueryClient: discovered worker {worker_id} dp_rank {dp_rank}, recovering"
                        );
                        match self
                            .recover_from_worker(worker_id, dp_rank, None, None)
                            .await
                        {
                            Ok(count) => {
                                if count > 0 {
                                    tracing::info!(
                                        "Recovered {count} events from worker {worker_id} dp_rank {dp_rank}"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to recover from worker {worker_id} dp_rank {dp_rank}: {e}"
                                );
                            }
                        }
                    }
                }
                DiscoveryEvent::Removed(id) => {
                    let Some((worker_id, dp_rank)) = Self::parse_instance_id(&id) else {
                        continue;
                    };

                    if let Some(dp_ranks) = known.get_mut(&worker_id) {
                        dp_ranks.remove(&dp_rank);
                        if dp_ranks.is_empty() {
                            known.remove(&worker_id);
                            tracing::warn!(
                                "WorkerQueryClient: all dp_ranks gone for worker {worker_id}, removing"
                            );
                            self.indexer.remove_worker(worker_id).await;
                        }
                    }
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

    /// Get or create a router for the specified dp_rank's endpoint.
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

        Ok(self
            .routers
            .entry(dp_rank)
            .or_insert(router)
            .value()
            .clone())
    }

    /// Query a specific worker's local KV indexer for a specific dp_rank.
    pub async fn query_worker(
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

    /// Query a worker's local KV indexer with exponential backoff retry.
    async fn query_worker_with_retry(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let mut last_error = None;

        for attempt in 0..RECOVERY_MAX_RETRIES {
            match self
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

    /// Recover missed KV events from a specific worker's dp_rank with retry logic.
    ///
    /// Called both by the internal discovery loop (initial recovery) and by the
    /// event plane task in subscriber.rs (gap recovery).
    pub async fn recover_from_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<usize> {
        tracing::debug!(
            "Attempting recovery from worker {worker_id} dp_rank {dp_rank}, \
             start_event_id: {start_event_id:?}, end_event_id: {end_event_id:?}"
        );

        let response = self
            .query_worker_with_retry(worker_id, dp_rank, start_event_id, end_event_id)
            .await?;

        let events = match response {
            WorkerKvQueryResponse::Events(events) => {
                tracing::debug!(
                    "Got {count} buffered events from worker {worker_id} dp_rank {dp_rank}",
                    count = events.len()
                );
                events
            }
            WorkerKvQueryResponse::TreeDump(events) => {
                tracing::info!(
                    "Got tree dump from worker {worker_id} dp_rank {dp_rank} \
                     (range too old or unspecified), count: {count}",
                    count = events.len()
                );
                events
            }
            WorkerKvQueryResponse::TooNew {
                requested_start,
                requested_end,
                newest_available,
            } => {
                tracing::warn!(
                    "Requested range [{requested_start:?}, {requested_end:?}] is newer than \
                     available (newest: {newest_available}) for worker {worker_id} dp_rank {dp_rank}"
                );
                return Ok(0);
            }
            WorkerKvQueryResponse::InvalidRange { start_id, end_id } => {
                anyhow::bail!(
                    "Invalid range for worker {worker_id} dp_rank {dp_rank}: \
                     end_id ({end_id}) < start_id ({start_id})"
                );
            }
            WorkerKvQueryResponse::Error(msg) => {
                anyhow::bail!("Worker {worker_id} dp_rank {dp_rank} query error: {msg}");
            }
        };

        let count = events.len();
        if count == 0 {
            tracing::debug!("No events to recover from worker {worker_id} dp_rank {dp_rank}");
            return Ok(0);
        }

        tracing::info!("Recovered {count} events from worker {worker_id} dp_rank {dp_rank}");

        for event in events {
            self.indexer.apply_event(event).await;
        }

        Ok(count)
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
    use crate::kv_router::RouterEvent;
    use crate::kv_router::indexer::KvIndexerMetrics;
    use crate::kv_router::protocols::{KvCacheEvent, KvCacheEventData};
    use tokio_util::sync::CancellationToken;

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
}
