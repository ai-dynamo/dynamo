// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::{
    indexer::{LocalKvIndexer, WorkerKvQueryRequest, WorkerKvQueryResponse},
    protocols::DpRank,
};
use dynamo_runtime::{
    component::Component,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
    },
    stream,
    traits::DistributedRuntimeProvider,
};
use tokio::sync::Semaphore;

use crate::kv_router::{
    worker_kv_indexer_query_endpoint, worker_kv_indexer_query_endpoint_for_worker,
};

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

pub(super) struct WorkerKvQueryEngine {
    pub(super) worker_id: u64,
    pub(super) dp_rank: DpRank,
    pub(super) local_indexer: Arc<LocalKvIndexer>,
    /// Semaphore limiting concurrent recovery request processing to 1.
    /// Prevents multiple routers from overwhelming the worker with heavy tree dump operations.
    pub(super) processing_semaphore: Semaphore,
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
