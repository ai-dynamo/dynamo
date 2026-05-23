// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the KV indexer feature.
//!
//! Owns a [`PositionalIndex`], binds the ZMQ ingest socket during
//! [`FeatureManager::attach`], and exports its own HTTP surface under
//! `/v1/features/indexer` (the server nests it via
//! [`FeatureManager::route_prefix`]).

use std::collections::HashSet;
use std::sync::{Arc, OnceLock, RwLock};

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
    self, ByPositionResponse, IndexerConfigResponse, InstancesResponse, QueryRequest, QueryResponse,
};
use super::zmq::{bind_sub_socket, bound_endpoint, port_of};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey};

/// Default host advertised in `GET /config`'s `zmq_endpoint` when none is
/// configured. Single-host / loopback deployments work out of the box;
/// multi-host deployments must set an explicit advertise host.
const DEFAULT_ADVERTISE_HOST: &str = "127.0.0.1";

/// Hub-side KV block index feature manager.
pub struct IndexerManager {
    index: Arc<PositionalIndex>,
    /// ZMQ bind spec (e.g. `tcp://0.0.0.0:0`).
    zmq_bind: String,
    /// Host advertised to publishers in `GET /config`.
    advertise_host: String,
    /// Resolved advertised endpoint (`tcp://host:port`), set during `attach`.
    endpoint: OnceLock<String>,
    /// Ingest task handle (set once spawned during `attach`).
    ingest_task: OnceLock<JoinHandle<()>>,
    /// Instances that declared `Feature::Indexer` at registration. Tracked
    /// separately from the index contents: an instance can register
    /// (participate) before — or without ever — emitting KV events, so this is
    /// the *registered* set, not the *emitting* set. Maintained by
    /// `on_register`/`on_unregister`; `GET /instances` sorts the output for a
    /// stable response (`InstanceId` is not `Ord`).
    instances: RwLock<HashSet<InstanceId>>,
}

impl std::fmt::Debug for IndexerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerManager")
            .field("max_seq_len", &self.index.max_seq_len())
            .field("block_size", &self.index.block_size())
            .field("num_positions", &self.index.num_positions())
            .field("endpoint", &self.endpoint.get())
            .finish()
    }
}

impl IndexerManager {
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
            instances: RwLock::new(HashSet::new()),
        })
    }

    /// Snapshot of the registered instance set as decimal `u128` strings
    /// (matching the holder ids in [`IndexEntry::instances`]), sorted for a
    /// stable response.
    fn instances_response(&self) -> InstancesResponse {
        let mut instances: Vec<String> = self
            .instances
            .read()
            .map(|s| s.iter().map(|id| id.as_u128().to_string()).collect())
            .unwrap_or_default();
        instances.sort();
        InstancesResponse { instances }
    }

    /// Shared index handle (for tests / introspection).
    pub fn index(&self) -> &Arc<PositionalIndex> {
        &self.index
    }

    /// Resolved advertised ZMQ endpoint, once `attach` has bound it.
    pub fn endpoint(&self) -> Option<&String> {
        self.endpoint.get()
    }

    fn config_response(&self) -> IndexerConfigResponse {
        IndexerConfigResponse {
            max_seq_len: self.index.max_seq_len(),
            block_size: self.index.block_size(),
            num_positions: self.index.num_positions(),
            zmq_endpoint: self.endpoint.get().cloned().unwrap_or_default(),
        }
    }
}

impl FeatureManager for IndexerManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::Indexer
    }

    fn config_requirements(&self) -> crate::features::FeatureConfigRequirements {
        // The publisher's page size must match the index's block size or events
        // hash/bucket wrong. `max_seq_len` is NOT a must-match: a larger value
        // simply grows the index (see `on_register`).
        crate::features::FeatureConfigRequirements {
            block_size: true,
            block_layout: false,
        }
    }

    fn requires_runtime_summary(&self) -> bool {
        // KV-index is new (introduced with the runtime summary): mandate it so
        // a publisher cannot register without its block_size being checked
        // against the hub's index block size.
        true
    }

    fn authoritative_block_size(&self) -> Option<usize> {
        // The index block size is the source of truth publishers must match.
        // Reconciled into `primary` at startup so validation never depends on
        // the operator having also set `primary` explicitly.
        Some(self.index.block_size())
    }

    fn descriptor(&self, _primary: &crate::protocol::PrimaryConfig) -> serde_json::Value {
        // Advertise the ZMQ ingest endpoint + sizing so the connector can wire
        // its publisher straight from the aggregate config (no separate probe).
        serde_json::to_value(self.config_response()).unwrap_or(serde_json::Value::Null)
    }

    fn route_prefix(&self) -> Option<&'static str> {
        Some(protocol::ROUTE_PREFIX)
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let sub = bind_sub_socket(&self.zmq_bind)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer bind: {e}")))?;
            let bound = bound_endpoint(&sub)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer endpoint: {e}")))?;
            let port = port_of(&bound)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer port: {e}")))?;
            let advertised = format!("tcp://{}:{}", self.advertise_host, port);
            tracing::info!(
                bound = %bound,
                advertised = %advertised,
                max_seq_len = self.index.max_seq_len(),
                block_size = self.index.block_size(),
                "indexer ingest bound"
            );
            let _ = self.endpoint.set(advertised);

            let task = tokio::spawn(run_ingest_loop(sub, Arc::clone(&self.index), ctx.cancel));
            let _ = self.ingest_task.set(task);

            // Expose the velo-plane block lookup (`QUERY_HANDLER`) when the hub
            // runs with a transport. Discovery-only hubs skip it — clients fall
            // back to the HTTP `POST /query` surface.
            if let Some(velo) = ctx.velo.as_ref() {
                velo.messenger()
                    .register_handler(super::handlers::create_query_handler(Arc::clone(
                        &self.index,
                    )))
                    .map_err(|e| {
                        FeatureError::Other(anyhow::anyhow!("indexer query handler: {e}"))
                    })?;
            }
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        // The client declares `Feature::Indexer` so the hub can reclaim its
        // index entries on unregister (`on_unregister` → `remove_instance`).
        // The index itself is populated out-of-band via the ZMQ ingest socket,
        // so there is nothing to do here beyond accepting the (empty) payload
        // and rejecting a misrouted key. Block-size / max-seq-len consistency
        // is validated centrally via `RuntimeConfigSummary`.
        Box::pin(async move {
            match feature {
                Feature::Indexer(cfg) => {
                    // Grow the index to fit this registrant's max_seq_len (never
                    // shrinks). Block-size consistency is validated centrally.
                    if let Some(max_seq_len) = cfg.max_seq_len {
                        self.index.grow_to_max_seq_len(max_seq_len);
                    }
                    // Track the registered (participating) instance so
                    // `GET /instances` can report it even before it emits any
                    // KV events.
                    if let Ok(mut set) = self.instances.write() {
                        set.insert(instance_id);
                    }
                    tracing::debug!(
                        instance = %instance_id,
                        max_seq_len = ?cfg.max_seq_len,
                        num_positions = self.index.num_positions(),
                        "indexer participation registered"
                    );
                    Ok(())
                }
                _ => Err(FeatureError::KeyMismatch {
                    manager: FeatureKey::Indexer,
                    payload: feature.key(),
                }),
            }
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        // Bridge the registry's velo InstanceId to the u128 the events wire
        // format carries (publishers stamp `velo_id.as_u128()`).
        self.index.remove_instance(instance_id.as_u128());
        if let Ok(mut set) = self.instances.write() {
            set.remove(&instance_id);
        }
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

fn routes(manager: Arc<IndexerManager>) -> Router {
    Router::new()
        .route(protocol::paths::CONFIG, get(get_config))
        .route(protocol::paths::INSTANCES, get(get_instances))
        .route(protocol::paths::BY_POSITION, get(get_by_position))
        .route(protocol::paths::QUERY, post(post_query))
        .with_state(manager)
}

async fn get_config(State(mgr): State<Arc<IndexerManager>>) -> Json<IndexerConfigResponse> {
    Json(mgr.config_response())
}

async fn get_instances(State(mgr): State<Arc<IndexerManager>>) -> Json<InstancesResponse> {
    Json(mgr.instances_response())
}

async fn get_by_position(
    State(mgr): State<Arc<IndexerManager>>,
    Path(pos): Path<usize>,
) -> Json<ByPositionResponse> {
    Json(mgr.index.by_position(pos))
}

async fn post_query(
    State(mgr): State<Arc<IndexerManager>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    Json(QueryResponse {
        hit: mgr.index.query(&req.hashes),
    })
}
