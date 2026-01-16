// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_runtime::component::Component;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, PushRouter, ResponseStream, RouterMode,
    SingleIn, async_trait, network::Ingress,
};
use dynamo_runtime::protocols::maybe_error::MaybeError;
use tokio::sync::{OnceCell, watch};
use tokio_stream::StreamExt;

use crate::kv_router::WORKER_KV_INDEXER_QUERY_ENDPOINT;
use crate::kv_router::indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::stream;

/// Router-side client for querying worker local KV indexers
///
/// Performs request/reply communication with workers via request plane endpoint routing.
/// (Only queries workers that have `enable_local_indexer=true` in their MDC user_data)
/// The client is spawned by KvRouter; it watches same discovery stream as the router.
pub struct WorkerQueryClient {
    component: Component,
    /// Watch receiver for enable_local_indexer state per worker
    model_runtime_config_rx: watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>,
    router: OnceCell<Arc<PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>>>,
}

impl WorkerQueryClient {
    /// Create a new WorkerQueryClient with a watch receiver for local indexer states
    pub fn new(
        component: Component,
        model_runtime_config_rx: watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>,
    ) -> Self {
        Self {
            component,
            model_runtime_config_rx,
            router: OnceCell::new(),
        }
    }

    /// Check if a worker has local indexer enabled
    pub fn has_local_indexer(&self, worker_id: WorkerId) -> bool {
        self.model_runtime_config_rx
            .borrow()
            .get(&worker_id)
            .map(|config| config.enable_local_indexer)
            .unwrap_or(false)
    }

    /// Query a specific worker's local KV indexer and return its buffered events.
    /// Returns an error if the worker does not have enable_local_indexer=true.
    pub async fn query_worker(
        &self,
        worker_id: WorkerId,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        // Check if worker has local indexer enabled
        if !self.has_local_indexer(worker_id) {
            anyhow::bail!(
                "Worker {worker_id} does not have local indexer enabled (enable_local_indexer=false or not set in MDC user_data)"
            );
        }

        let router = self
            .router
            .get_or_try_init(|| async {
                let endpoint = self.component.endpoint(WORKER_KV_INDEXER_QUERY_ENDPOINT);
                let client = endpoint.client().await?;
                let router = PushRouter::from_client(client, RouterMode::RoundRobin).await?;
                Ok::<_, anyhow::Error>(Arc::new(router))
            })
            .await?;

        let request = WorkerKvQueryRequest {
            worker_id,
            start_event_id,
            end_event_id,
        };
        let mut stream = router
            .direct(SingleIn::new(request), worker_id)
            .await
            .with_context(|| {
                format!("Failed to send worker KV query request to worker {worker_id} via endpoint")
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

// Worker-side endpoint registration for Router -> LocalKvIndexer query service
pub(crate) async fn start_worker_kv_query_endpoint(
    component: Component,
    worker_id: u64,
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
                "Failed to build WorkerKvQuery endpoint handler for worker {worker_id}: {e}"
            );
            return;
        }
    };

    tracing::info!(
        "WorkerKvQuery endpoint starting for worker {worker_id} on endpoint '{}'",
        WORKER_KV_INDEXER_QUERY_ENDPOINT
    );

    if let Err(e) = component
        .endpoint(WORKER_KV_INDEXER_QUERY_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start()
        .await
    {
        tracing::error!("WorkerKvQuery endpoint failed for worker {worker_id}: {e}");
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
