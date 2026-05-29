// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dynamo_kv_router::{
    indexer::{WorkerKvQueryRequest, WorkerKvQueryResponse},
    protocols::{DpRank, WorkerId},
};
use dynamo_runtime::{
    component::{Component, Instance},
    discovery::EndpointInstanceId,
    pipeline::{AddressedPushRouter, AddressedRequest, AsyncEngine, ManyOut, SingleIn},
    protocols::maybe_error::MaybeError,
};
use futures::StreamExt;

#[async_trait]
pub(super) trait WorkerQueryTransport: Send + Sync {
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

pub(super) struct RuntimeWorkerQueryTransport {
    addressed: Arc<AddressedPushRouter>,
}

impl RuntimeWorkerQueryTransport {
    pub(super) async fn new(component: &Component) -> Result<Self> {
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

// Velo transport

/// Velo unary transport for worker KV gap-recovery queries.
///
/// Each `(worker_id, dp_rank)` pair must be registered via [`register_peer`]
/// before queries can be sent.  Registration is driven by the caller after it
/// resolves a worker's Velo `InstanceId` through peer discovery.
///
/// The `target: Instance` parameter of [`WorkerQueryTransport::query_worker`]
/// is ignored — Velo uses the `InstanceId` from the internal peers map.
/// The existing [`RuntimeWorkerQueryTransport`] remains the default; this
/// transport is only active when the `velo-recovery` feature is compiled in.
#[cfg(feature = "velo-recovery")]
pub(super) struct VeloWorkerQueryTransport {
    messenger: Arc<velo::Messenger>,
    peers: dashmap::DashMap<super::worker_query_state::RecoveryKey, velo::InstanceId>,
}

#[cfg(feature = "velo-recovery")]
impl VeloWorkerQueryTransport {
    pub(super) fn new(messenger: Arc<velo::Messenger>) -> Self {
        Self {
            messenger,
            peers: dashmap::DashMap::new(),
        }
    }

    /// Register a worker's Velo `InstanceId` so it can receive recovery queries.
    ///
    /// Must be called before the first `query_worker` call for this
    /// `(worker_id, dp_rank)` pair.  Overwrites any previously registered peer.
    pub(super) fn register_peer(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        instance_id: velo::InstanceId,
    ) {
        self.peers.insert((worker_id, dp_rank), instance_id);
        tracing::info!(
            worker_id,
            dp_rank,
            ?instance_id,
            "Registered Velo peer for worker KV recovery queries"
        );
    }

    /// Unregister a worker's Velo peer (called on worker removal from discovery).
    pub(super) fn unregister_peer(&self, worker_id: WorkerId, dp_rank: DpRank) {
        if self.peers.remove(&(worker_id, dp_rank)).is_some() {
            tracing::debug!(
                worker_id,
                dp_rank,
                "Unregistered Velo peer for worker KV recovery queries"
            );
        }
    }
}

#[cfg(feature = "velo-recovery")]
#[async_trait]
impl WorkerQueryTransport for VeloWorkerQueryTransport {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        _target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        use super::worker_query_endpoint::VELO_WORKER_QUERY_HANDLER;
        use bytes::Bytes;

        let instance_id = self
            .peers
            .get(&(worker_id, dp_rank))
            .map(|r| *r)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no Velo peer registered for worker {worker_id} dp_rank {dp_rank}; \
                     call register_peer after resolving the worker's InstanceId"
                )
            })?;

        let request = WorkerKvQueryRequest {
            worker_id,
            dp_rank,
            start_event_id,
            end_event_id,
        };
        let payload = Bytes::from(
            serde_json::to_vec(&request).context("failed to serialize WorkerKvQueryRequest")?,
        );

        let response_bytes = self
            .messenger
            .unary(VELO_WORKER_QUERY_HANDLER)?
            .raw_payload(payload)
            .instance(instance_id)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Velo worker KV recovery query failed \
                     for worker {worker_id} dp_rank {dp_rank}"
                )
            })?;

        let response = serde_json::from_slice::<WorkerKvQueryResponse>(&response_bytes)
            .context("failed to deserialize WorkerKvQueryResponse from Velo")?;

        if let Some(err) = response.err() {
            return Err(err).context("Velo worker KV query response error");
        }

        Ok(response)
    }
}

// Tests

#[cfg(all(test, feature = "velo-recovery"))]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer};
    use tokio_util::sync::CancellationToken;
    use velo::backend::tcp::TcpTransportBuilder;

    use super::super::worker_query_endpoint::register_velo_query_handler;
    use super::*;

    // Helpers

    async fn make_messenger() -> Arc<velo::Messenger> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let transport: Arc<dyn velo::backend::Transport> = Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        );
        velo::Messenger::builder()
            .add_transport(transport)
            .build()
            .await
            .unwrap()
    }

    fn make_local_indexer() -> Arc<LocalKvIndexer> {
        Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            16,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            128,
        ))
    }

    /// Create a bidirectional messenger pair with peer registration.
    async fn make_pair() -> (Arc<velo::Messenger>, Arc<velo::Messenger>) {
        let a = make_messenger().await;
        let b = make_messenger().await;
        a.register_peer(b.peer_info()).unwrap();
        b.register_peer(a.peer_info()).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        (a, b)
    }

    /// Dummy Dynamo Instance — not used by VeloWorkerQueryTransport but
    /// required by the WorkerQueryTransport trait signature.
    fn dummy_instance() -> Instance {
        Instance {
            namespace: "test".to_string(),
            component: "test".to_string(),
            endpoint: "test".to_string(),
            instance_id: 0,
            transport: dynamo_runtime::component::TransportType::Nats(String::new()),
            device_type: None,
        }
    }

    // Tests

    /// Basic end-to-end: router messenger sends a query, worker messenger
    /// handles it via the Velo unary handler, router receives the response.
    #[tokio::test]
    async fn velo_transport_end_to_end_empty_response() {
        let (router_messenger, worker_messenger) = make_pair().await;
        let local_indexer = make_local_indexer();

        register_velo_query_handler(&worker_messenger, 1, 0, local_indexer).unwrap();

        let transport = VeloWorkerQueryTransport::new(router_messenger.clone());
        transport.register_peer(1, 0, worker_messenger.instance_id());

        let response = transport
            .query_worker(1, 0, dummy_instance(), None, None)
            .await
            .unwrap();

        // Empty indexer returns an empty tree dump or buffer response — not an Error.
        assert!(
            !matches!(response, WorkerKvQueryResponse::Error(_)),
            "expected a non-error response from empty indexer, got {response:?}"
        );
    }

    /// Validation: querying a (worker_id, dp_rank) with no registered engine returns
    /// `Err` — not `Ok(WorkerKvQueryResponse::Error(...))`.
    ///
    /// The transport now maps application-level `Error` responses to `Err` so
    /// that the retry loop in `fetch_recovery_response` can act on them.
    ///
    /// The handler dispatch table holds an engine for (1, 0).  The transport has a
    /// peer entry for (99, 0) pointing at the same worker messenger, so the
    /// transport layer succeeds.  The handler returns `WorkerKvQueryResponse::Error`
    /// for the unregistered key, and the transport converts that to `Err`.
    #[tokio::test]
    async fn velo_transport_unregistered_worker_id_returns_err() {
        let (router_messenger, worker_messenger) = make_pair().await;
        let local_indexer = make_local_indexer();

        // Handler dispatch table has an engine only for (worker_id=1, dp_rank=0).
        register_velo_query_handler(&worker_messenger, 1, 0, local_indexer).unwrap();

        let transport = VeloWorkerQueryTransport::new(router_messenger.clone());
        // Route (99, 0) to the same worker messenger so the request reaches the handler.
        transport.register_peer(99, 0, worker_messenger.instance_id());

        let result = transport
            .query_worker(99, 0, dummy_instance(), None, None)
            .await;

        assert!(result.is_err(), "expected Err for unregistered worker_id");
        let chain = format!("{:#}", result.unwrap_err());
        assert!(
            chain.contains("no engine registered"),
            "expected 'no engine registered' in error chain, got: {chain}"
        );
    }

    /// Validation: querying a (worker_id, dp_rank) with no registered engine returns
    /// `Err` for an unregistered dp_rank.
    ///
    /// Same rationale as `velo_transport_unregistered_worker_id_returns_err`.
    #[tokio::test]
    async fn velo_transport_unregistered_dp_rank_returns_err() {
        let (router_messenger, worker_messenger) = make_pair().await;
        let local_indexer = make_local_indexer();

        // Handler dispatch table has an engine only for (worker_id=1, dp_rank=0).
        register_velo_query_handler(&worker_messenger, 1, 0, local_indexer).unwrap();

        let transport = VeloWorkerQueryTransport::new(router_messenger.clone());
        // Route (1, 3) to the same worker messenger so the request reaches the handler.
        transport.register_peer(1, 3, worker_messenger.instance_id());

        let result = transport
            .query_worker(1, 3, dummy_instance(), None, None)
            .await;

        assert!(result.is_err(), "expected Err for unregistered dp_rank");
        let chain = format!("{:#}", result.unwrap_err());
        assert!(
            chain.contains("no engine registered"),
            "expected 'no engine registered' in error chain, got: {chain}"
        );
    }

    /// query_worker returns an Err (not a WorkerKvQueryResponse::Error) when
    /// no Velo peer has been registered for the given (worker_id, dp_rank).
    #[tokio::test]
    async fn velo_transport_no_peer_returns_transport_error() {
        let router_messenger = make_messenger().await;
        let transport = VeloWorkerQueryTransport::new(router_messenger);

        let result = transport
            .query_worker(42, 0, dummy_instance(), None, None)
            .await;

        assert!(result.is_err(), "expected Err when no peer is registered");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("no Velo peer registered"),
            "unexpected error message: {msg}"
        );
    }

    /// `register_peer` overwrites a stale entry; subsequent queries use the new `InstanceId`.
    ///
    /// First registration uses the router's own `InstanceId` (which has no handler), so a
    /// query would fail if the stale entry were used.  After overwriting with the worker's
    /// real `InstanceId`, the query must succeed.
    #[tokio::test]
    async fn velo_transport_register_peer_overwrites_stale_entry() {
        let (router_messenger, worker_messenger) = make_pair().await;
        let local_indexer = make_local_indexer();

        register_velo_query_handler(&worker_messenger, 1, 0, local_indexer).unwrap();

        let transport = VeloWorkerQueryTransport::new(router_messenger.clone());

        // Stale registration: router's own InstanceId, which has no query handler.
        transport.register_peer(1, 0, router_messenger.instance_id());

        // Overwrite with the real worker InstanceId.
        transport.register_peer(1, 0, worker_messenger.instance_id());

        let response = transport
            .query_worker(1, 0, dummy_instance(), None, None)
            .await
            .unwrap();

        assert!(
            !matches!(response, WorkerKvQueryResponse::Error(_)),
            "query should succeed after re-registration, got {response:?}"
        );
    }
}
