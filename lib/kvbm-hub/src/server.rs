// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum-based HTTP server for the KVBM hub.
//!
//! Runs two listeners:
//!
//! - **Discovery port** (`1337` default) — serves only the PeerDiscovery
//!   HTTP surface. This is the port a velo client's [`HubClient`](crate::HubClient)
//!   hits for peer lookups.
//! - **Control port** (`8337` default) — serves the full control plane
//!   (registration, heartbeat, health, + discovery for convenience).
//!
//! The server is also intended to be a `velo::Velo` participant — register
//! hub-side handlers on it (future work) so it can push messages (e.g.
//! heartbeats) to connected clients via velo active messaging.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use parking_lot::RwLock;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use velo_common::{InstanceId, PeerInfo, WorkerId};

use crate::protocol::{
    self, ErrorBody, ErrorCode, HeartbeatResponse, PeerLookupResponse, RegisterRequest,
    RegisterResponse,
};

/// Server-side in-memory peer registry.
#[derive(Debug, Default)]
struct Registry {
    by_instance: HashMap<InstanceId, PeerInfo>,
    by_worker: HashMap<WorkerId, InstanceId>,
}

/// Shared hub server state (cheap to clone, all state is inside `Arc`s).
#[derive(Clone, Debug)]
pub struct HubServerState {
    registry: Arc<RwLock<Registry>>,
}

impl Default for HubServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl HubServerState {
    /// Create fresh, empty hub state.
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(Registry::default())),
        }
    }

    /// Snapshot the currently registered peers.
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.registry.read().by_instance.values().cloned().collect()
    }
}

/// Builder for [`HubServer`].
#[derive(Debug, Clone)]
pub struct HubServerBuilder {
    bind_addr: IpAddr,
    discovery_port: u16,
    control_port: u16,
    state: Option<HubServerState>,
}

impl Default for HubServerBuilder {
    fn default() -> Self {
        Self {
            bind_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            discovery_port: protocol::DEFAULT_DISCOVERY_PORT,
            control_port: protocol::DEFAULT_CONTROL_PORT,
            state: None,
        }
    }
}

impl HubServerBuilder {
    /// New builder with default bind `0.0.0.0` and default ports.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind address (default `0.0.0.0`).
    pub fn bind_addr(mut self, addr: IpAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Discovery HTTP port (default `1337`).
    pub fn discovery_port(mut self, port: u16) -> Self {
        self.discovery_port = port;
        self
    }

    /// Control-plane HTTP port (default `8337`).
    pub fn control_port(mut self, port: u16) -> Self {
        self.control_port = port;
        self
    }

    /// Provide a pre-constructed shared state. Optional.
    pub fn state(mut self, state: HubServerState) -> Self {
        self.state = Some(state);
        self
    }

    /// Bind both listeners and spawn them. Returns a running [`HubServer`].
    pub async fn serve(self) -> Result<HubServer> {
        let state = self.state.unwrap_or_default();

        let discovery_addr = SocketAddr::new(self.bind_addr, self.discovery_port);
        let control_addr = SocketAddr::new(self.bind_addr, self.control_port);

        let discovery_listener = TcpListener::bind(discovery_addr)
            .await
            .with_context(|| format!("binding discovery port {discovery_addr}"))?;
        let control_listener = TcpListener::bind(control_addr)
            .await
            .with_context(|| format!("binding control port {control_addr}"))?;

        let discovery_local = discovery_listener
            .local_addr()
            .context("discovery local_addr")?;
        let control_local = control_listener
            .local_addr()
            .context("control local_addr")?;

        let cancel = CancellationToken::new();

        let discovery_router = discovery_router(state.clone());
        let control_router = control_router(state.clone());

        let discovery_task = spawn_server(discovery_listener, discovery_router, cancel.clone());
        let control_task = spawn_server(control_listener, control_router, cancel.clone());

        Ok(HubServer {
            state,
            discovery_addr: discovery_local,
            control_addr: control_local,
            cancel,
            discovery_task: Some(discovery_task),
            control_task: Some(control_task),
        })
    }
}

/// A running hub server.
///
/// Drop or call [`shutdown`](Self::shutdown) to cancel both listeners and
/// wait for them to terminate.
pub struct HubServer {
    state: HubServerState,
    discovery_addr: SocketAddr,
    control_addr: SocketAddr,
    cancel: CancellationToken,
    discovery_task: Option<JoinHandle<()>>,
    control_task: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for HubServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServer")
            .field("discovery_addr", &self.discovery_addr)
            .field("control_addr", &self.control_addr)
            .finish()
    }
}

impl HubServer {
    /// Builder entry point.
    pub fn builder() -> HubServerBuilder {
        HubServerBuilder::new()
    }

    /// Resolved discovery socket address (useful when binding port `0`).
    pub fn discovery_addr(&self) -> SocketAddr {
        self.discovery_addr
    }

    /// Resolved control socket address.
    pub fn control_addr(&self) -> SocketAddr {
        self.control_addr
    }

    /// Shared state handle (for attaching velo hub-side handlers, etc.).
    pub fn state(&self) -> &HubServerState {
        &self.state
    }

    /// Trigger shutdown and await both listeners.
    pub async fn shutdown(mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(t) = self.discovery_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.control_task.take() {
            let _ = t.await;
        }
        Ok(())
    }
}

impl Drop for HubServer {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

fn spawn_server(
    listener: TcpListener,
    router: Router,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
            cancel.cancelled().await;
        });
        if let Err(e) = serve.await {
            tracing::error!(error = %e, "kvbm-hub listener exited with error");
        }
    })
}

fn discovery_router(state: HubServerState) -> Router {
    Router::new()
        .route(
            protocol::paths::PEERS_BY_INSTANCE,
            get(get_peer_by_instance),
        )
        .route(protocol::paths::PEERS_BY_WORKER, get(get_peer_by_worker))
        .route(protocol::paths::HEALTH, get(health))
        .with_state(state)
}

fn control_router(state: HubServerState) -> Router {
    Router::new()
        .route(protocol::paths::INSTANCES, post(register_instance))
        .route(protocol::paths::INSTANCE_BY_ID, delete(unregister_instance))
        .route(protocol::paths::INSTANCE_HEARTBEAT, post(heartbeat))
        // Discovery endpoints are mirrored here for convenience.
        .route(
            protocol::paths::PEERS_BY_INSTANCE,
            get(get_peer_by_instance),
        )
        .route(protocol::paths::PEERS_BY_WORKER, get(get_peer_by_worker))
        .route(protocol::paths::HEALTH, get(health))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn register_instance(
    State(state): State<HubServerState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, HubError> {
    let peer = req.peer_info;
    let instance_id = peer.instance_id();
    let worker_id = peer.worker_id();

    let mut reg = state.registry.write();
    if let Some(existing) = reg.by_worker.get(&worker_id)
        && *existing != instance_id
    {
        return Err(HubError::conflict(format!(
            "worker_id {worker_id} already held by instance {existing}"
        )));
    }
    reg.by_worker.insert(worker_id, instance_id);
    reg.by_instance.insert(instance_id, peer);
    Ok(Json(RegisterResponse { instance_id }))
}

async fn unregister_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<StatusCode, HubError> {
    let mut reg = state.registry.write();
    let removed = reg.by_instance.remove(&instance_id);
    if let Some(p) = &removed {
        reg.by_worker.remove(&p.worker_id());
    }
    if removed.is_some() {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(HubError::not_found(format!(
            "instance {instance_id} not registered"
        )))
    }
}

async fn heartbeat(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<HeartbeatResponse>, HubError> {
    let registered = state.registry.read().by_instance.contains_key(&instance_id);
    Ok(Json(HeartbeatResponse {
        acknowledged: registered,
    }))
}

async fn get_peer_by_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    let reg = state.registry.read();
    reg.by_instance
        .get(&instance_id)
        .cloned()
        .map(|peer_info| Json(PeerLookupResponse { peer_info }))
        .ok_or_else(|| HubError::not_found(format!("instance {instance_id} not found")))
}

async fn get_peer_by_worker(
    State(state): State<HubServerState>,
    Path(worker_id): Path<u64>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    let wid = WorkerId::from_u64(worker_id);
    let reg = state.registry.read();
    let instance_id = reg
        .by_worker
        .get(&wid)
        .copied()
        .ok_or_else(|| HubError::not_found(format!("worker {worker_id} not found")))?;
    let peer_info = reg
        .by_instance
        .get(&instance_id)
        .cloned()
        .ok_or_else(|| HubError::internal("registry inconsistency".to_string()))?;
    Ok(Json(PeerLookupResponse { peer_info }))
}

// ---------------------------------------------------------------------------
// Error plumbing
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HubError {
    status: StatusCode,
    body: ErrorBody,
}

impl HubError {
    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorBody {
                code: ErrorCode::NotFound,
                message,
            },
        }
    }

    fn conflict(message: String) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            body: ErrorBody {
                code: ErrorCode::Conflict,
                message,
            },
        }
    }

    fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorBody {
                code: ErrorCode::Internal,
                message,
            },
        }
    }
}

impl IntoResponse for HubError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}
