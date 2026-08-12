// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use dynamo_kv_router::{
    indexer::{
        KvStateAgentStatus, KvStateRecoveryReceipt, LocalKvIndexer, WorkerKvQueryKind,
        WorkerKvQueryRequest, WorkerKvQueryResponse,
    },
    protocols::DpRank,
};
use dynamo_runtime::{
    component::{Component, StartedEndpoint},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
    },
    stream,
    traits::DistributedRuntimeProvider,
};
use tokio::sync::Semaphore;

/// Worker-side endpoint registration for Router -> LocalKvIndexer query service
pub(crate) async fn start_worker_kv_query_endpoint(
    component: Component,
    publisher_id: u64,
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
) -> anyhow::Result<StartedEndpoint> {
    start_worker_kv_query_endpoint_with_status(
        component,
        publisher_id,
        worker_id,
        dp_rank,
        local_indexer,
        None,
    )
    .await
}

pub(crate) async fn start_worker_kv_query_endpoint_with_status(
    component: Component,
    publisher_id: u64,
    worker_id: u64,
    dp_rank: DpRank,
    local_indexer: Arc<LocalKvIndexer>,
    status: Option<Arc<ArcSwap<KvStateAgentStatus>>>,
) -> anyhow::Result<StartedEndpoint> {
    let engine = Arc::new(WorkerKvQueryEngine {
        worker_id,
        dp_rank,
        local_indexer,
        status,
        processing_semaphore: Semaphore::new(1),
    });

    let ingress = Ingress::for_engine(engine)?;

    let route_worker_id = component.drt().connection_id();
    let endpoint_name = format!("worker_kv_query_source_{publisher_id:x}");
    tracing::info!(
        "WorkerKvQuery endpoint starting for worker {worker_id} dp_rank {dp_rank} \
         routed by instance {route_worker_id} on endpoint '{endpoint_name}'"
    );

    component
        .endpoint(&endpoint_name)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start_with_registration()
        .await
}

pub(super) struct WorkerKvQueryEngine {
    pub(super) worker_id: u64,
    pub(super) dp_rank: DpRank,
    pub(super) local_indexer: Arc<LocalKvIndexer>,
    pub(super) status: Option<Arc<ArcSwap<KvStateAgentStatus>>>,
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

        if let WorkerKvQueryKind::Status {
            expected,
            expected_attachment_generation,
        } = &request.kind
        {
            let Some(status) = self.status.as_ref() else {
                return Ok(single_response(
                    WorkerKvQueryResponse::Error(
                        "KV state-agent status is not supported by this endpoint".to_string(),
                    ),
                    ctx.context(),
                ));
            };
            let status = status.load();
            if status.identity != *expected {
                return Ok(single_response(
                    WorkerKvQueryResponse::Error(
                        "KV state-agent status identity mismatch".to_string(),
                    ),
                    ctx.context(),
                ));
            }
            if let Some(expected_generation) = expected_attachment_generation
                && status
                    .attachment
                    .as_ref()
                    .map(|attachment| attachment.generation)
                    != Some(*expected_generation)
            {
                return Ok(single_response(
                    WorkerKvQueryResponse::Error(
                        "KV state-agent attachment generation mismatch".to_string(),
                    ),
                    ctx.context(),
                ));
            }

            // Status intentionally bypasses the full-dump semaphore. The
            // snapshot is immutable and cheap to clone, so dump backpressure
            // cannot be mistaken for agent death.
            return Ok(single_response(
                WorkerKvQueryResponse::Status((**status).clone()),
                ctx.context(),
            ));
        }

        let state_agent_recovery = match &request.kind {
            WorkerKvQueryKind::StateAgentRecovery {
                expected,
                expected_attachment_generation,
            } => {
                let Some(status) = self.status.as_ref().map(|status| status.load()) else {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "state-agent recovery is not supported by this endpoint".to_string(),
                        ),
                        ctx.context(),
                    ));
                };
                if status.identity != *expected {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "KV state-agent recovery identity mismatch".to_string(),
                        ),
                        ctx.context(),
                    ));
                }
                if !status.cache_owner_ready {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "KV state-agent CacheOwner recovery is not ready".to_string(),
                        ),
                        ctx.context(),
                    ));
                }
                Some((expected.clone(), *expected_attachment_generation))
            }
            WorkerKvQueryKind::Recovery | WorkerKvQueryKind::Status { .. } => None,
        };

        let recovery_attachment = if let Some((expected, expected_generation)) =
            state_agent_recovery.as_ref()
        {
            if let Some(expected_generation) = *expected_generation {
                let status = self.status.as_ref().expect("validated above").load();
                let Some(attachment) = status.attachment.as_ref().filter(|attachment| {
                    attachment.ready && attachment.generation == expected_generation
                }) else {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "KV state-agent attachment is not ready for Worker recovery"
                                .to_string(),
                        ),
                        ctx.context(),
                    ));
                };
                if request.worker_id != attachment.worker.worker_id
                    || request.dp_rank != attachment.worker.dp_rank
                {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "KV state-agent recovery request targets a stale attachment"
                                .to_string(),
                        ),
                        ctx.context(),
                    ));
                }
                Some((
                    expected.cache_owner_id,
                    attachment.worker,
                    expected_generation,
                ))
            } else {
                if request.dp_rank != self.dp_rank {
                    return Ok(single_response(
                        WorkerKvQueryResponse::Error(
                            "KV state-agent recovery request targets a different stable slot"
                                .to_string(),
                        ),
                        ctx.context(),
                    ));
                }
                None
            }
        } else if let Some(status) = self.status.as_ref() {
            let status = status.load();
            let Some(attachment) = status
                .attachment
                .as_ref()
                .filter(|attachment| attachment.ready)
            else {
                return Ok(single_response(
                    WorkerKvQueryResponse::Error(
                        "KV state-agent attachment is not ready for Worker recovery".to_string(),
                    ),
                    ctx.context(),
                ));
            };
            if request.worker_id != attachment.worker.worker_id
                || request.dp_rank != attachment.worker.dp_rank
            {
                return Ok(single_response(
                    WorkerKvQueryResponse::Error(
                        "KV state-agent recovery request targets a stale attachment".to_string(),
                    ),
                    ctx.context(),
                ));
            }
            Some((
                status.identity.cache_owner_id,
                attachment.worker,
                attachment.generation,
            ))
        } else {
            None
        };

        if self.status.is_none() && request.worker_id != self.worker_id {
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

        if self.status.is_none() && request.dp_rank != self.dp_rank {
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
        let mut response = negotiate_tree_dump_failure(response, request.supports_tree_dump_failed);
        if let Some((expected, expected_generation)) = &state_agent_recovery {
            response = filter_state_agent_events(
                response,
                expected.cache_owner_id,
                expected_generation.is_some(),
                false,
            );
        } else if let Some((state_source, _, _)) = recovery_attachment {
            response = filter_state_agent_events(response, state_source, true, true);
        }
        if let Some(status) = self.status.as_ref() {
            let status = status.load();
            let state_agent_changed =
                if let Some((expected, expected_generation)) = &state_agent_recovery {
                    &status.identity != expected
                        || !status.cache_owner_ready
                        || expected_generation.is_some_and(|generation| {
                            status
                                .attachment
                                .as_ref()
                                .map(|attachment| (attachment.generation, attachment.ready))
                                != Some((generation, true))
                        })
                } else if let Some((_, _, expected_generation)) = recovery_attachment {
                    status
                        .attachment
                        .as_ref()
                        .map(|attachment| (attachment.generation, attachment.ready))
                        != Some((expected_generation, true))
                } else {
                    false
                };
            if state_agent_changed {
                response = WorkerKvQueryResponse::Error(
                    "KV state-agent attachment or source changed during recovery".to_string(),
                );
            }
        }
        if let Some((expected, expected_attachment_generation)) = state_agent_recovery
            && let Some(recovered_through_cursor) = recovery_response_cursor(&response)
        {
            response = WorkerKvQueryResponse::StateAgentRecovery {
                response: Box::new(response),
                receipt: KvStateRecoveryReceipt {
                    identity: expected,
                    attachment_generation: expected_attachment_generation,
                    recovered_through_cursor,
                },
            };
        }

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

fn recovery_response_cursor(response: &WorkerKvQueryResponse) -> Option<u64> {
    match response {
        WorkerKvQueryResponse::Events { last_event_id, .. }
        | WorkerKvQueryResponse::TreeDump { last_event_id, .. } => Some(*last_event_id),
        _ => None,
    }
}

fn filter_state_agent_events(
    response: WorkerKvQueryResponse,
    state_source: dynamo_kv_router::identity::CacheOwnerId,
    include_worker: bool,
    worker_only: bool,
) -> WorkerKvQueryResponse {
    let retain = |event: &dynamo_kv_router::protocols::RouterEvent| {
        event
            .resolved_residency_domain()
            .is_ok_and(|domain| match domain {
                dynamo_kv_router::protocols::ResidencyDomain::Worker => include_worker,
                dynamo_kv_router::protocols::ResidencyDomain::CacheOwner => {
                    !worker_only && event.state_source == Some(state_source)
                }
            })
    };
    match response {
        WorkerKvQueryResponse::Events {
            mut events,
            last_event_id,
        } => {
            if !include_worker {
                return WorkerKvQueryResponse::Error(
                    "CacheOwner-only incremental recovery is unsupported; request a full state-agent snapshot"
                        .to_string(),
                );
            }
            events.retain(retain);
            WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            }
        }
        WorkerKvQueryResponse::TreeDump {
            mut events,
            last_event_id,
            reset_scope: _,
        } => {
            events.retain(retain);
            let reset_scope = if worker_only {
                // A missing recovery capability is the legacy Worker-only
                // contract. Its consumer has no CacheOwner state and accepts
                // only an all-rank replacement, so preserve that encoding.
                dynamo_kv_router::protocols::ResetScope::All
            } else if !include_worker {
                dynamo_kv_router::protocols::ResetScope::Domain(
                    dynamo_kv_router::protocols::ResidencyDomain::CacheOwner,
                )
            } else {
                dynamo_kv_router::protocols::ResetScope::All
            };
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                reset_scope,
            }
        }
        response => response,
    }
}

fn single_response(
    response: WorkerKvQueryResponse,
    context: Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>,
) -> ManyOut<WorkerKvQueryResponse> {
    ResponseStream::new(Box::pin(stream::iter(vec![response])), context)
}

fn negotiate_tree_dump_failure(
    response: WorkerKvQueryResponse,
    supports_tree_dump_failed: bool,
) -> WorkerKvQueryResponse {
    match response {
        WorkerKvQueryResponse::TreeDumpFailed { message, .. } if !supports_tree_dump_failed => {
            // Error predates TreeDumpFailed and remains non-authoritative for legacy readers.
            // Never translate snapshot failure into an empty all-domain replacement.
            WorkerKvQueryResponse::Error(format!("worker tree dump failed: {message}"))
        }
        response => response,
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
    use dynamo_kv_router::identity::{
        CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheEventData, ResidencyDomain, RouterEvent, StorageTier,
    };

    use super::*;

    fn test_owner() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    #[test]
    fn explicit_dump_failure_is_capability_negotiated() {
        let failure = || WorkerKvQueryResponse::TreeDumpFailed {
            last_event_id: 17,
            message: "offline".to_string(),
        };

        assert!(matches!(
            negotiate_tree_dump_failure(failure(), true),
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id: 17,
                ..
            }
        ));
        assert!(matches!(
            negotiate_tree_dump_failure(failure(), false),
            WorkerKvQueryResponse::Error(message)
                if message == "worker tree dump failed: offline"
        ));
    }

    #[test]
    fn state_agent_recovery_views_keep_primary_worker_dumps_generation_free() {
        let event = |event_id, domain| {
            RouterEvent::with_residency_domain(
                17,
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Cleared,
                    dp_rank: 3,
                },
                StorageTier::HostPinned,
                domain,
            )
        };
        let response = WorkerKvQueryResponse::TreeDump {
            events: vec![
                event(0, ResidencyDomain::Worker),
                event(1, ResidencyDomain::CacheOwner).with_state_source(test_owner()),
            ],
            last_event_id: 3,
            reset_scope: dynamo_kv_router::protocols::ResetScope::All,
        };

        let WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id,
            reset_scope,
        } = filter_state_agent_events(response, test_owner(), true, false)
        else {
            panic!("expected full state-agent dump")
        };
        assert_eq!(last_event_id, 3);
        assert_eq!(events.len(), 2);
        assert_eq!(reset_scope, dynamo_kv_router::protocols::ResetScope::All);

        let detached = filter_state_agent_events(
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                reset_scope,
            },
            test_owner(),
            false,
            false,
        );
        assert!(matches!(
            detached,
            WorkerKvQueryResponse::TreeDump {
                events,
                reset_scope: dynamo_kv_router::protocols::ResetScope::Domain(
                    ResidencyDomain::CacheOwner
                ),
                ..
            } if events.len() == 1
        ));

        let incremental = filter_state_agent_events(
            WorkerKvQueryResponse::Events {
                events: vec![event(4, ResidencyDomain::CacheOwner).with_state_source(test_owner())],
                last_event_id: 4,
            },
            test_owner(),
            false,
            false,
        );
        assert!(matches!(incremental, WorkerKvQueryResponse::Error(_)));
    }
}
