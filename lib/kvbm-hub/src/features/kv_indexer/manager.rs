// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the KV indexer feature.
//!
//! Owns a [`PositionalIndex`], binds the ZMQ ingest socket during
//! [`FeatureManager::attach`], and exports its own HTTP surface under
//! `/v1/features/kv-index` (the server nests it via
//! [`FeatureManager::route_prefix`]).

use std::sync::{Arc, OnceLock};

use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use futures::future::BoxFuture;
use tokio::task::JoinHandle;
use velo_ext::{InstanceId, PeerInfo};

use super::index::PositionalIndex;
use super::ingest::run_ingest_loop;
use super::protocol::{
    self, ByPositionResponse, KvIndexerConfigResponse, QueryRequest, QueryResponse,
};
use super::zmq::{bind_sub_socket, bound_endpoint, port_of};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey};

/// Default host advertised in `GET /config`'s `zmq_endpoint` when none is
/// configured. Single-host / loopback deployments work out of the box;
/// multi-host deployments must set an explicit advertise host.
const DEFAULT_ADVERTISE_HOST: &str = "127.0.0.1";

/// Hub-side KV block index feature manager.
pub struct KvIndexerManager {
    index: Arc<PositionalIndex>,
    /// ZMQ bind spec (e.g. `tcp://0.0.0.0:0`).
    zmq_bind: String,
    /// Host advertised to publishers in `GET /config`.
    advertise_host: String,
    /// Resolved advertised endpoint (`tcp://host:port`), set during `attach`.
    endpoint: OnceLock<String>,
    /// Ingest task handle (set once spawned during `attach`).
    ingest_task: OnceLock<JoinHandle<()>>,
}

impl std::fmt::Debug for KvIndexerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvIndexerManager")
            .field("max_seq_len", &self.index.max_seq_len())
            .field("block_size", &self.index.block_size())
            .field("num_positions", &self.index.num_positions())
            .field("endpoint", &self.endpoint.get())
            .finish()
    }
}

impl KvIndexerManager {
    /// Builds a manager sized for `max_seq_len`/`block_size`, binding ingest to
    /// `zmq_bind` (defaults to `tcp://0.0.0.0:0`) and advertising
    /// `advertise_host` (defaults to `127.0.0.1`).
    pub fn new(
        max_seq_len: usize,
        block_size: usize,
        zmq_bind: Option<String>,
        advertise_host: Option<String>,
    ) -> anyhow::Result<Self> {
        let index = Arc::new(PositionalIndex::new(max_seq_len, block_size)?);
        Ok(Self {
            index,
            zmq_bind: zmq_bind.unwrap_or_else(|| "tcp://0.0.0.0:0".to_string()),
            advertise_host: advertise_host.unwrap_or_else(|| DEFAULT_ADVERTISE_HOST.to_string()),
            endpoint: OnceLock::new(),
            ingest_task: OnceLock::new(),
        })
    }

    /// Shared index handle (for tests / introspection).
    pub fn index(&self) -> &Arc<PositionalIndex> {
        &self.index
    }

    /// Resolved advertised ZMQ endpoint, once `attach` has bound it.
    pub fn endpoint(&self) -> Option<&String> {
        self.endpoint.get()
    }

    fn config_response(&self) -> KvIndexerConfigResponse {
        KvIndexerConfigResponse {
            max_seq_len: self.index.max_seq_len(),
            block_size: self.index.block_size(),
            num_positions: self.index.num_positions(),
            zmq_endpoint: self.endpoint.get().cloned().unwrap_or_default(),
        }
    }
}

impl FeatureManager for KvIndexerManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::KvIndexer
    }

    fn route_prefix(&self) -> Option<&'static str> {
        Some(protocol::ROUTE_PREFIX)
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let sub = bind_sub_socket(&self.zmq_bind)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("kv-index bind: {e}")))?;
            let bound = bound_endpoint(&sub)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("kv-index endpoint: {e}")))?;
            let port = port_of(&bound)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("kv-index port: {e}")))?;
            let advertised = format!("tcp://{}:{}", self.advertise_host, port);
            tracing::info!(
                bound = %bound,
                advertised = %advertised,
                max_seq_len = self.index.max_seq_len(),
                block_size = self.index.block_size(),
                "kv-index ingest bound"
            );
            let _ = self.endpoint.set(advertised);

            let task = tokio::spawn(run_ingest_loop(sub, Arc::clone(&self.index), ctx.cancel));
            let _ = self.ingest_task.set(task);
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        // No client-side `Feature::KvIndexer` payload exists — workers opt in
        // by publishing to the ZMQ endpoint, not via the registration list.
        // This is only reachable if the dispatcher misroutes a key.
        Box::pin(async move {
            Err(FeatureError::KeyMismatch {
                manager: FeatureKey::KvIndexer,
                payload: feature.key(),
            })
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        // Bridge the registry's velo InstanceId to the u128 the events wire
        // format carries (publishers stamp `velo_id.as_u128()`).
        self.index.remove_instance(instance_id.as_u128());
    }

    fn on_register_any<'a>(
        &'a self,
        _instance_id: InstanceId,
        _peer: &'a PeerInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    fn control_router(self: Arc<Self>) -> Router {
        routes(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        routes(self)
    }
}

fn routes(manager: Arc<KvIndexerManager>) -> Router {
    Router::new()
        .route(protocol::paths::CONFIG, get(get_config))
        .route(protocol::paths::BY_POSITION, get(get_by_position))
        .route(protocol::paths::QUERY, post(post_query))
        .with_state(manager)
}

async fn get_config(State(mgr): State<Arc<KvIndexerManager>>) -> Json<KvIndexerConfigResponse> {
    Json(mgr.config_response())
}

async fn get_by_position(
    State(mgr): State<Arc<KvIndexerManager>>,
    Path(pos): Path<usize>,
) -> Json<ByPositionResponse> {
    Json(mgr.index.by_position(pos))
}

async fn post_query(
    State(mgr): State<Arc<KvIndexerManager>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    Json(QueryResponse {
        hit: mgr.index.query(&req.hashes),
    })
}
