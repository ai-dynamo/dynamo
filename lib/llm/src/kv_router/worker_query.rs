// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use dashmap::DashMap;
use dynamo_runtime::component::Component;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, PushRouter, ResponseStream, RouterMode,
    SingleIn, async_trait, network::Ingress,
};
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_runtime::stream;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use crate::discovery::RuntimeConfigsSubscriber;
use crate::kv_router::indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::protocols::{DpRank, RouterEvent, WorkerId};
use crate::kv_router::worker_kv_indexer_query_endpoint;

// Recovery retry configuration
const RECOVERY_MAX_RETRIES: u32 = 8;
const RECOVERY_INITIAL_BACKOFF_MS: u64 = 200;

/// Router-side client for querying worker local KV indexers
///
/// Performs request/reply communication with workers via request plane endpoint routing.
/// (Only queries workers that have `enable_local_indexer=true` in their MDC user_data)
/// The client is spawned by KvRouter; it uses a subscriber from RuntimeConfigs.
///
/// Each dp_rank has its own LocalKvIndexer and query endpoint, so we maintain separate
/// routers per dp_rank to ensure queries go to the correct endpoint.
pub struct WorkerQueryClient {
    component: Component,
    /// Subscriber for runtime configs (includes shared configs DashMap)
    subscriber: RuntimeConfigsSubscriber,
    /// Routers keyed by dp_rank - each dp_rank has its own endpoint
    routers: DashMap<DpRank, Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>>,
}

impl WorkerQueryClient {
    /// Create a new WorkerQueryClient with a subscriber to runtime configs
    pub fn new(component: Component, subscriber: RuntimeConfigsSubscriber) -> Self {
        Self {
            component,
            subscriber,
            routers: DashMap::new(),
        }
    }

    /// Wait until at least one worker has a known runtime config (Some).
    /// Returns the list of worker IDs that have configs.
    pub async fn wait_for_ready(&mut self) -> Vec<WorkerId> {
        self.subscriber.wait_for_some().await
    }

    /// Check if a worker has local indexer enabled
    pub fn has_local_indexer(&self, worker_id: WorkerId) -> bool {
        self.subscriber
            .configs
            .get(&worker_id)
            .and_then(|entry| entry.value().as_ref().map(|c| c.enable_local_indexer))
            .unwrap_or(false)
    }

    /// Get the data_parallel_size for a worker (defaults to 1 if not found)
    pub fn get_data_parallel_size(&self, worker_id: WorkerId) -> u32 {
        self.subscriber
            .configs
            .get(&worker_id)
            .and_then(|entry| entry.value().as_ref().map(|c| c.data_parallel_size))
            .unwrap_or(1)
    }

    /// Get or create a router for the specified dp_rank's endpoint
    async fn get_router_for_dp_rank(
        &self,
        dp_rank: DpRank,
    ) -> Result<Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>> {
        // Fast path: check if router already exists
        if let Some(router) = self.routers.get(&dp_rank) {
            return Ok(router.clone());
        }

        // Slow path: create new router
        let endpoint_name = worker_kv_indexer_query_endpoint(dp_rank);
        let endpoint = self.component.endpoint(&endpoint_name);
        let client = endpoint.client().await?;
        let router = Arc::new(PushRouter::from_client(client, RouterMode::RoundRobin).await?);

        // Insert and return (if another thread inserted first, use theirs)
        Ok(self
            .routers
            .entry(dp_rank)
            .or_insert(router)
            .value()
            .clone())
    }

    /// Query a specific worker's local KV indexer for a specific dp_rank and return its buffered events.
    /// Returns an error if the worker does not have enable_local_indexer=true.
    pub async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        // Check if worker has local indexer enabled
        if !self.has_local_indexer(worker_id) {
            anyhow::bail!(
                "Worker {worker_id} does not have local indexer enabled (enable_local_indexer=false or not set in MDC user_data)"
            );
        }

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
                format!("Failed to send worker KV query request to worker {worker_id} dp_rank {dp_rank} via endpoint")
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

    /// Recover events from all dp_ranks of a single worker.
    ///
    /// # Returns
    /// Total number of events recovered across all dp_ranks
    pub async fn recover_all_dp_ranks(
        &self,
        worker_id: WorkerId,
        event_tx: &mpsc::Sender<RouterEvent>,
    ) -> usize {
        let dp_size = self.get_data_parallel_size(worker_id);
        let mut total_recovered = 0;

        for dp_rank in 0..dp_size {
            match self
                .recover_from_worker(worker_id, dp_rank, None, None, event_tx)
                .await
            {
                Ok(count) => {
                    total_recovered += count;
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

        total_recovered
    }

    /// Recover missed KV events from a specific worker's dp_rank with retry logic.
    ///
    /// # Returns
    /// Number of events recovered, or error if recovery failed after all retries
    pub async fn recover_from_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
        event_tx: &mpsc::Sender<RouterEvent>,
    ) -> Result<usize> {
        if !self.has_local_indexer(worker_id) {
            tracing::debug!(
                "Worker {worker_id} does not have local indexer enabled, skipping recovery"
            );
            return Ok(0);
        }

        tracing::debug!(
            "Attempting recovery from worker {worker_id} dp_rank {dp_rank}, \
             start_event_id: {start_event_id:?}, end_event_id: {end_event_id:?}"
        );

        // Query worker with retry logic for transient failures
        let mut response = None;
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
                    response = Some(resp);
                    break;
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

        let response = match response {
            Some(r) => r,
            None => return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No response"))),
        };

        // Handle response variants
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

        // Send recovered events to the indexer
        let count = events.len();
        if count == 0 {
            tracing::debug!("No events to recover from worker {worker_id} dp_rank {dp_rank}");
            return Ok(0);
        }

        tracing::info!("Recovered {count} events from worker {worker_id} dp_rank {dp_rank}");

        for event in events {
            if let Err(e) = event_tx.send(event).await {
                tracing::error!(
                    "Failed to send recovered event to indexer for worker {worker_id} dp_rank {dp_rank}: {e}"
                );
                anyhow::bail!("Failed to send recovered event: {e}");
            }
        }

        Ok(count)
    }
}

// Worker-side endpoint registration for Router -> LocalKvIndexer query service
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

        // This is a sanity check to ensure the request is for the correct worker.
        // In production, this should never happen since the router should only
        // send requests to the worker it is associated with.
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
    use tokio_stream::StreamExt;
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
