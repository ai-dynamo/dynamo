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
    protocols::maybe_error::MaybeError,
    stream,
    traits::DistributedRuntimeProvider,
};
use tokio::sync::Semaphore;

use crate::kv_router::{
    worker_kv_indexer_query_endpoint, worker_kv_indexer_query_endpoint_for_worker,
};

/// Velo endpoint name for worker KV gap-recovery queries.
///
/// Workers register a unary handler under this name; the router's
/// [`super::worker_query_transport::VeloWorkerQueryTransport`] sends to it.
pub(super) const VELO_WORKER_QUERY_HANDLER: &str = "kv_router.worker.recovery_query";

/// Worker-side endpoint registration for the Router -> LocalKvIndexer query service.
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
                "Failed to build WorkerKvQuery endpoint handler \
                 for worker {worker_id} dp_rank {dp_rank}: {e}"
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

// Velo handler registration

/// Per-messenger dispatch table for Velo worker KV gap-recovery queries.
///
/// A single Velo handler (named [`VELO_WORKER_QUERY_HANDLER`]) is registered per
/// `Messenger`.  Internally it dispatches to the appropriate
/// [`WorkerKvQueryEngine`] based on the `(worker_id, dp_rank)` in the request,
/// so workers with multiple dp-ranks share one handler without naming conflicts.
///
/// ## Usage
///
/// ```rust,ignore
/// let dispatch = VeloQueryDispatch::new();
/// dispatch.add_engine(worker_id, 0, local_indexer_rank_0);
/// dispatch.add_engine(worker_id, 1, local_indexer_rank_1);
/// dispatch.register_on(&messenger)?;
/// ```
///
/// For single-rank workers the [`register_velo_query_handler`] convenience
/// wrapper does all of the above in one call.
#[cfg(feature = "velo-recovery")]
type EngineMap = dashmap::DashMap<super::worker_query_state::RecoveryKey, Arc<WorkerKvQueryEngine>>;

#[cfg(feature = "velo-recovery")]
pub(crate) struct VeloQueryDispatch {
    engines: Arc<EngineMap>,
}

#[cfg(feature = "velo-recovery")]
impl VeloQueryDispatch {
    pub(crate) fn new() -> Self {
        Self {
            engines: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Add a `(worker_id, dp_rank)` engine to the dispatch table.
    ///
    /// Overwrites any previously registered engine for the same key.
    pub(crate) fn add_engine(
        &self,
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
        self.engines.insert((worker_id, dp_rank), engine);
    }

    /// Register the Velo unary handler on `messenger`.
    ///
    /// Must be called exactly once per messenger after all engines have been
    /// added via [`add_engine`].  Returns an error if the handler name is
    /// already registered on this messenger.
    pub(crate) fn register_on(self, messenger: &Arc<velo::Messenger>) -> anyhow::Result<()> {
        use bytes::Bytes;
        use velo::Handler;

        let engines = self.engines.clone();

        let handler = Handler::unary_handler_async(VELO_WORKER_QUERY_HANDLER, move |ctx| {
            let engines = engines.clone();
            async move {
                let request: WorkerKvQueryRequest =
                    serde_json::from_slice(&ctx.payload).map_err(|e| {
                        anyhow::anyhow!(
                            "failed to deserialize WorkerKvQueryRequest \
                                 from {}: {e}",
                            VELO_WORKER_QUERY_HANDLER
                        )
                    })?;

                tracing::debug!(
                    worker_id = request.worker_id,
                    dp_rank = request.dp_rank,
                    start_event_id = ?request.start_event_id,
                    "Velo worker KV recovery query received"
                );

                let engine = match engines.get(&(request.worker_id, request.dp_rank)) {
                    Some(e) => e.clone(),
                    None => {
                        let msg = format!(
                            "no engine registered for worker_id={} dp_rank={}",
                            request.worker_id, request.dp_rank
                        );
                        tracing::warn!("{msg}");
                        let resp = WorkerKvQueryResponse::Error(msg);
                        return Ok(Some(Bytes::from(serde_json::to_vec(&resp)?)));
                    }
                };

                let response = engine.handle(request).await;
                Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
            }
        })
        .build();

        messenger.register_handler(handler)?;
        tracing::info!(
            handler = VELO_WORKER_QUERY_HANDLER,
            n_engines = self.engines.len(),
            "Registered Velo worker KV recovery query handler"
        );
        Ok(())
    }
}

/// Convenience wrapper for single-rank workers.
///
/// Builds a [`VeloQueryDispatch`] with one engine and registers it on
/// `messenger`.  Multi-rank workers must use [`VeloQueryDispatch`] directly to
/// avoid registering the handler twice on the same messenger.
///
/// Call this after the Velo `Messenger` is built and before publishing the
/// worker's peer info for routers to discover.
#[cfg(feature = "velo-recovery")]
pub(crate) fn register_velo_query_handler(
    messenger: &Arc<velo::Messenger>,
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
) -> anyhow::Result<()> {
    let dispatch = VeloQueryDispatch::new();
    dispatch.add_engine(worker_id, dp_rank, local_indexer);
    dispatch.register_on(messenger)
}

// Velo peer-info endpoint

/// Wire request for the worker Velo peer-info endpoint (empty — no parameters needed).
#[cfg(feature = "velo-recovery")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(super) struct WorkerVeloPeerRequest;

/// Wire response for the worker Velo peer-info endpoint.
#[cfg(feature = "velo-recovery")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(super) enum WorkerVeloPeerResponse {
    /// Serialized peer connection info for the worker's Velo messenger.
    Success {
        /// `serde_json`-encoded `velo::PeerInfo`.
        peer_info_bytes: Vec<u8>,
        /// `serde_json`-encoded `velo::InstanceId` for
        /// [`VeloWorkerQueryTransport::register_peer`].
        instance_id_bytes: Vec<u8>,
    },
    Error(String),
}

#[cfg(feature = "velo-recovery")]
impl MaybeError for WorkerVeloPeerResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        WorkerVeloPeerResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        match self {
            Self::Error(msg) => Some(dynamo_runtime::error::DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

/// Worker-side engine that serves the local Velo messenger's `PeerInfo` on demand.
#[cfg(feature = "velo-recovery")]
struct WorkerVeloPeerEngine {
    messenger: Arc<velo::Messenger>,
}

#[cfg(feature = "velo-recovery")]
#[async_trait]
impl AsyncEngine<SingleIn<WorkerVeloPeerRequest>, ManyOut<WorkerVeloPeerResponse>, anyhow::Error>
    for WorkerVeloPeerEngine
{
    async fn generate(
        &self,
        request: SingleIn<WorkerVeloPeerRequest>,
    ) -> anyhow::Result<ManyOut<WorkerVeloPeerResponse>> {
        let (_req, ctx) = request.into_parts();

        let peer_info_bytes = serde_json::to_vec(&self.messenger.peer_info())
            .map_err(|e| anyhow::anyhow!("failed to serialize velo::PeerInfo: {e}"))?;
        let instance_id_bytes = serde_json::to_vec(&self.messenger.instance_id())
            .map_err(|e| anyhow::anyhow!("failed to serialize velo::InstanceId: {e}"))?;

        let response = WorkerVeloPeerResponse::Success {
            peer_info_bytes,
            instance_id_bytes,
        };
        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

/// Worker-side endpoint that publishes the local Velo `PeerInfo` for routers to discover.
///
/// Registers a Dynamo endpoint named `worker_kv_velo_peer_dp{dp_rank}` (or
/// `worker_kv_velo_peer_dp{dp_rank}_worker{worker_id}` when the route instance id
/// differs from the logical worker id, matching the convention used by
/// [`start_worker_kv_query_endpoint`]).  When the router's Velo-aware discovery
/// loop sees this endpoint it issues a bounded-retry query to obtain the `PeerInfo`
/// and register the worker as a Velo peer.
///
/// Under `velo-recovery` this endpoint is the sole recovery lifecycle anchor for
/// `(worker_id, dp_rank)`.  The legacy [`start_worker_kv_query_endpoint`] is not
/// started when the feature is active.
#[cfg(feature = "velo-recovery")]
pub(crate) async fn start_worker_kv_velo_peer_endpoint(
    component: Component,
    worker_id: u64,
    dp_rank: DpRank,
    messenger: Arc<velo::Messenger>,
) {
    use crate::kv_router::{worker_kv_velo_peer_endpoint, worker_kv_velo_peer_endpoint_for_worker};
    use dynamo_runtime::traits::DistributedRuntimeProvider;

    let route_worker_id = component.drt().connection_id();
    let endpoint_name = if route_worker_id == worker_id {
        worker_kv_velo_peer_endpoint(dp_rank)
    } else {
        worker_kv_velo_peer_endpoint_for_worker(worker_id, dp_rank)
    };

    let engine = Arc::new(WorkerVeloPeerEngine { messenger });
    let ingress = match Ingress::for_engine(engine) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!(
                worker_id,
                dp_rank,
                "Failed to build WorkerVeloPeer endpoint handler: {e}"
            );
            return;
        }
    };

    tracing::info!(
        worker_id,
        dp_rank,
        endpoint = %endpoint_name,
        "WorkerVeloPeer endpoint starting"
    );

    if let Err(e) = component
        .endpoint(&endpoint_name)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start()
        .await
    {
        tracing::error!(worker_id, dp_rank, "WorkerVeloPeer endpoint failed: {e}");
    }
}

pub(super) struct WorkerKvQueryEngine {
    pub(super) worker_id: u64,
    pub(super) dp_rank: DpRank,
    pub(super) local_indexer: Arc<LocalKvIndexer>,
    /// Limits concurrent tree-dump operations to 1 per worker.
    /// Prevents multiple routers from issuing heavy dump operations simultaneously.
    pub(super) processing_semaphore: Semaphore,
}

impl WorkerKvQueryEngine {
    /// Validate that the request targets this engine's worker and rank.
    ///
    /// Returns `Some(error_response)` on mismatch, `None` if valid.
    fn validate_ids(&self, req: &WorkerKvQueryRequest) -> Option<WorkerKvQueryResponse> {
        if req.worker_id != self.worker_id {
            return Some(WorkerKvQueryResponse::Error(format!(
                "WorkerKvQueryEngine worker_id mismatch: \
                 request={} this={}",
                req.worker_id, self.worker_id
            )));
        }
        if req.dp_rank != self.dp_rank {
            return Some(WorkerKvQueryResponse::Error(format!(
                "WorkerKvQueryEngine dp_rank mismatch: \
                 request={} this={}",
                req.dp_rank, self.dp_rank
            )));
        }
        None
    }

    /// Execute the indexer call after validation and semaphore acquisition.
    ///
    /// Callers are responsible for holding the semaphore permit for tree-dump
    /// requests before calling this method.
    async fn do_query(&self, req: WorkerKvQueryRequest) -> WorkerKvQueryResponse {
        self.local_indexer
            .get_events_in_id_range(req.start_event_id, req.end_event_id)
            .await
    }

    /// Handle a recovery query: validate, acquire semaphore, execute.
    ///
    /// Used by the Velo unary handler.  Does not implement client-side
    /// cancellation; callers that need it (the Dynamo runtime endpoint) use
    /// `validate_ids` + a `select!`-guarded semaphore acquire + `do_query`
    /// directly instead.
    pub(super) async fn handle(&self, req: WorkerKvQueryRequest) -> WorkerKvQueryResponse {
        if let Some(err) = self.validate_ids(&req) {
            return err;
        }

        let likely_buffer_read = self
            .local_indexer
            .likely_served_from_buffer(req.start_event_id);

        let _permit = if !likely_buffer_read {
            match self.processing_semaphore.acquire().await {
                Ok(permit) => Some(permit),
                Err(_) => {
                    return WorkerKvQueryResponse::Error(
                        "Worker KV query semaphore closed".to_string(),
                    );
                }
            }
        } else {
            None
        };

        let _slow_query_guard = if !likely_buffer_read {
            Some(SlowQueryGuard::spawn(self.worker_id))
        } else {
            None
        };

        self.do_query(req).await
    }
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
            worker_id = self.worker_id,
            start_event_id = ?request.start_event_id,
            "Received Dynamo runtime worker KV query request"
        );

        if let Some(err) = self.validate_ids(&request) {
            return Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![err])),
                ctx.context(),
            ));
        }

        let likely_buffer_read = self
            .local_indexer
            .likely_served_from_buffer(request.start_event_id);

        // Acquire the semaphore for tree-dump requests, with client-cancellation
        // support: if the Dynamo runtime cancels the request while we are queued,
        // return an error response rather than blocking indefinitely.
        let _maybe_permit = if !likely_buffer_read {
            let engine_ctx = ctx.context();
            let permit = tokio::select! {
                result = self.processing_semaphore.acquire() => {
                    result.map_err(|_| anyhow::anyhow!("Worker KV query semaphore closed"))?
                }
                _ = futures::future::select(engine_ctx.stopped(), engine_ctx.killed()) => {
                    tracing::warn!(
                        worker_id = self.worker_id,
                        "Worker KV query request cancelled while waiting for semaphore"
                    );
                    return Ok(ResponseStream::new(
                        Box::pin(stream::iter(vec![WorkerKvQueryResponse::Error(
                            "Request cancelled by client".to_string(),
                        )])),
                        ctx.context(),
                    ));
                }
            };
            Some(permit)
        } else {
            None
        };

        // Start slow-query logging only once the request is actively running.
        // Requests queued behind the semaphore should remain silent.
        let _slow_query_guard = if !likely_buffer_read {
            Some(SlowQueryGuard::spawn(self.worker_id))
        } else {
            None
        };

        let response = self.do_query(request).await;
        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

/// RAII guard that aborts a slow-query logger task on drop.
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
