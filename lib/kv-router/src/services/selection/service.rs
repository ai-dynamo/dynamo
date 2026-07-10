// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::config::KvRouterConfig;
use crate::protocols::WorkerId;
use crate::scheduling::PotentialLoad;
use crate::services::common::replica_sync::{
    PeerManager, ReplicaPeerError, ReplicaSyncRuntime, setup_replica_sync,
};

use super::cache::SelectionCacheConfig;
use super::core::{SelectionCore, SelectionServiceConfig};
use super::error::SelectionError;
use super::types::{
    ModelLoadResponse, OverlapScoresRequest, OverlapScoresResponse, PotentialLoadsRequest,
    ReadyResponse, ReservationRequest, ReservationResponse, SelectAndReserveRequest, SelectRequest,
    SelectResponse, WorkerCatalogRecord, WorkerPatchRequest, WorkerRequest,
};

pub struct SelectionServiceBuilder {
    kv_router_config: KvRouterConfig,
    indexer_threads: usize,
    indexer_peers: Vec<String>,
    replica_sync_port: Option<u16>,
    replica_sync_peers: Vec<String>,
    selection_cache: SelectionCacheConfig,
}

impl SelectionServiceBuilder {
    pub fn new(kv_router_config: KvRouterConfig) -> Self {
        Self {
            kv_router_config,
            indexer_threads: 4,
            indexer_peers: Vec::new(),
            replica_sync_port: None,
            replica_sync_peers: Vec::new(),
            selection_cache: SelectionCacheConfig::default(),
        }
    }

    pub fn indexer_threads(mut self, indexer_threads: usize) -> Self {
        self.indexer_threads = indexer_threads;
        self
    }

    pub fn indexer_peers(mut self, indexer_peers: Vec<String>) -> Self {
        self.indexer_peers = indexer_peers;
        self
    }

    pub fn replica_sync(mut self, port: u16, peers: Vec<String>) -> Self {
        self.replica_sync_port = Some(port);
        self.replica_sync_peers = peers;
        self
    }

    pub fn selection_cache(mut self, config: SelectionCacheConfig) -> Self {
        self.selection_cache = config;
        self
    }

    pub async fn build(self) -> anyhow::Result<SelectionService> {
        let cancel_token = CancellationToken::new();
        let mut startup_guard = StartupGuard::new(cancel_token.clone());
        let replica_runtime = setup_replica_sync(
            self.replica_sync_port,
            &self.replica_sync_peers,
            cancel_token.child_token(),
        )?;
        let replica_config = replica_runtime.as_ref().map(ReplicaSyncRuntime::config);
        let core = Arc::new(SelectionCore::new_managed(
            self.kv_router_config,
            self.indexer_threads,
            cancel_token.clone(),
            replica_config,
            self.selection_cache,
        ));

        if !self.indexer_peers.is_empty() {
            match core.recover_indexer_from_peers(&self.indexer_peers).await {
                Ok(true) => tracing::info!("Selection indexer recovery completed"),
                Ok(false) => {
                    tracing::warn!(
                        "No reachable selection indexer peers; starting with empty state"
                    )
                }
                Err(error) => {
                    tracing::warn!(%error, "Selection indexer recovery failed; starting with empty state")
                }
            }
        }
        core.signal_indexer_ready();

        let peer_manager = if replica_runtime.is_some() {
            let weak_core = Arc::downgrade(&core);
            Some(PeerManager::start(
                self.replica_sync_peers,
                cancel_token.child_token(),
                move |event| {
                    if let Some(core) = weak_core.upgrade() {
                        core.dispatch_replica_event(event);
                    }
                },
            )?)
        } else {
            None
        };

        startup_guard.disarm();
        Ok(SelectionService {
            core,
            peer_manager,
            replica_runtime,
            cancel_token,
        })
    }
}

impl SelectionServiceConfig {
    pub fn service_builder(&self) -> SelectionServiceBuilder {
        let mut builder = SelectionServiceBuilder::new(self.kv_router_config.clone())
            .indexer_threads(self.threads)
            .indexer_peers(self.indexer_peers.clone())
            .selection_cache(self.selection_cache.clone());
        if let Some(port) = self.replica_sync_port {
            builder = builder.replica_sync(port, self.replica_sync_peers.clone());
        }
        builder
    }
}

struct StartupGuard {
    cancel_token: CancellationToken,
    armed: bool,
}

impl StartupGuard {
    fn new(cancel_token: CancellationToken) -> Self {
        Self {
            cancel_token,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StartupGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancel_token.cancel();
        }
    }
}

pub struct SelectionService {
    core: Arc<SelectionCore>,
    peer_manager: Option<PeerManager>,
    replica_runtime: Option<ReplicaSyncRuntime>,
    cancel_token: CancellationToken,
}

impl SelectionService {
    #[cfg(test)]
    pub(super) fn new_local_for_test(
        kv_router_config: KvRouterConfig,
        indexer_threads: usize,
    ) -> Self {
        let cancel_token = CancellationToken::new();
        Self {
            core: Arc::new(SelectionCore::new_local(
                kv_router_config,
                indexer_threads,
                cancel_token.clone(),
                SelectionCacheConfig::default(),
            )),
            peer_manager: None,
            replica_runtime: None,
            cancel_token,
        }
    }

    pub async fn upsert_worker(
        &self,
        req: WorkerRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.core.upsert_worker(req).await
    }

    pub async fn patch_worker(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.core.patch_worker(worker_id, patch).await
    }

    pub async fn delete_worker(
        &self,
        worker_id: WorkerId,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.core.delete_worker(worker_id).await
    }

    pub fn list_workers(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        self.core.list_workers(model_name, routing_group)
    }

    pub async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        self.core.select(req).await
    }

    pub async fn select_with_policy_class(
        &self,
        req: SelectRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        self.core.select_with_policy_class(req, policy_class).await
    }

    pub async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError> {
        self.core.select_and_reserve(req).await
    }

    pub async fn select_and_reserve_with_policy_class(
        &self,
        req: SelectAndReserveRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        self.core
            .select_and_reserve_with_policy_class(req, policy_class)
            .await
    }

    pub async fn create_reservation(
        &self,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        self.core.create_reservation(req).await
    }

    pub async fn prefill_complete(&self, reservation_id: &str) -> Result<(), SelectionError> {
        self.core.prefill_complete(reservation_id).await
    }

    pub fn add_output_block(
        &self,
        reservation_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SelectionError> {
        self.core.add_output_block(reservation_id, decay_fraction)
    }

    pub async fn free_reservation(&self, reservation_id: &str) -> Result<(), SelectionError> {
        self.core.free_reservation(reservation_id).await
    }

    pub fn ready(&self) -> ReadyResponse {
        self.core.ready()
    }

    pub fn loads(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<ModelLoadResponse> {
        self.core.loads(model_name, routing_group)
    }

    pub async fn potential_loads(
        &self,
        req: PotentialLoadsRequest,
    ) -> Result<Vec<PotentialLoad>, SelectionError> {
        self.core.potential_loads(req).await
    }

    pub async fn overlap_scores(
        &self,
        req: OverlapScoresRequest,
    ) -> Result<OverlapScoresResponse, SelectionError> {
        self.core.overlap_scores(req).await
    }

    pub async fn register_replica_peer(&self, endpoint: String) -> Result<bool, ReplicaPeerError> {
        let peer_manager = self
            .peer_manager
            .as_ref()
            .ok_or(ReplicaPeerError::Disabled)?;
        peer_manager.register_peer(endpoint).await
    }

    pub async fn deregister_replica_peer(
        &self,
        endpoint: String,
    ) -> Result<bool, ReplicaPeerError> {
        let peer_manager = self
            .peer_manager
            .as_ref()
            .ok_or(ReplicaPeerError::Disabled)?;
        peer_manager.deregister_peer(endpoint).await
    }

    pub fn list_replica_peers(&self) -> Vec<String> {
        self.peer_manager
            .as_ref()
            .map(PeerManager::list_peers)
            .unwrap_or_default()
    }

    pub async fn indexer_snapshot(&self) -> serde_json::Value {
        self.core.dump_indexer_events().await
    }

    pub async fn recover_indexer_from_peers(&self, peers: &[String]) -> anyhow::Result<bool> {
        self.core.recover_indexer_from_peers(peers).await
    }

    pub async fn cancelled(&self) {
        self.cancel_token.cancelled().await;
    }

    pub async fn shutdown(&self) {
        self.cancel_token.cancel();
        self.core.shutdown();
        if let (Some(peer_manager), Some(replica_runtime)) =
            (&self.peer_manager, &self.replica_runtime)
        {
            tokio::join!(peer_manager.shutdown(), replica_runtime.shutdown());
        } else if let Some(peer_manager) = &self.peer_manager {
            peer_manager.shutdown().await;
        } else if let Some(replica_runtime) = &self.replica_runtime {
            replica_runtime.shutdown().await;
        }
    }
}

impl Drop for SelectionService {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        self.core.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener as StdTcpListener;
    use std::time::Duration;

    use axum::Json;
    use axum::Router;
    use axum::extract::State;
    use axum::routing::get;
    use tokio::sync::Notify;

    use super::*;

    fn test_config() -> KvRouterConfig {
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        }
    }

    fn reserve_tcp_port() -> u16 {
        StdTcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    async fn build_on_port(port: u16) -> SelectionService {
        tokio::time::timeout(Duration::from_secs(5), async move {
            loop {
                match SelectionServiceBuilder::new(test_config())
                    .indexer_threads(1)
                    .replica_sync(port, Vec::new())
                    .build()
                    .await
                {
                    Ok(service) => return service,
                    Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
                }
            }
        })
        .await
        .expect("replica port was not released")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_and_shutdown_release_replica_resources() {
        let port = reserve_tcp_port();
        let failed = SelectionServiceBuilder::new(test_config())
            .indexer_threads(1)
            .replica_sync(port, vec!["invalid".to_string()])
            .build()
            .await;
        assert!(failed.is_err());

        let service = build_on_port(port).await;
        let weak_core = Arc::downgrade(&service.core);
        service.shutdown().await;
        let replacement = build_on_port(port).await;
        drop(service);
        assert!(weak_core.upgrade().is_none());
        replacement.shutdown().await;

        let drop_port = reserve_tcp_port();
        let dropped = build_on_port(drop_port).await;
        let dropped_core = Arc::downgrade(&dropped.core);
        drop(dropped);
        assert!(dropped_core.upgrade().is_none());
        let replacement = build_on_port(drop_port).await;
        replacement.shutdown().await;
    }

    #[derive(Clone)]
    struct RecoveryGate {
        requested: Arc<Notify>,
        release: Arc<Notify>,
    }

    async fn gated_dump(State(gate): State<RecoveryGate>) -> Json<serde_json::Value> {
        gate.requested.notify_one();
        gate.release.notified().await;
        Json(serde_json::json!({}))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_waits_for_indexer_recovery() {
        let gate = RecoveryGate {
            requested: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let peer_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/dump", get(gated_dump))
            .with_state(gate.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await });

        let build = tokio::spawn(
            SelectionServiceBuilder::new(test_config())
                .indexer_threads(1)
                .indexer_peers(vec![peer_url])
                .build(),
        );
        tokio::time::timeout(Duration::from_secs(3), gate.requested.notified())
            .await
            .expect("recovery peer was not queried");
        assert!(!build.is_finished());

        gate.release.notify_one();
        let service = build.await.unwrap().unwrap();
        service.shutdown().await;
        server.abort();
    }
}
