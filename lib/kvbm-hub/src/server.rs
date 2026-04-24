// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum-based HTTP server for the KVBM hub.
//!
//! Runs two listeners:
//!
//! - **Discovery port** (`1337` default) — serves only the `PeerDiscovery`
//!   HTTP surface. This is the port a velo client's [`HubClient`](crate::HubClient)
//!   hits for peer lookups.
//! - **Control port** (`8337` default) — serves the full control plane
//!   (registration, heartbeat, health) plus mirrored discovery for
//!   convenience.
//!
//! When one or more transports are supplied via
//! [`HubServerBuilder::add_transport`], the hub also participates in velo: it
//! builds an internal `velo::Velo`, self-registers in the registry, and can
//! push active messages (heartbeats, probes) to registered clients.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use velo_common::{InstanceId, PeerInfo, WorkerId};

use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use crate::protocol::{
    self, ErrorBody, ErrorCode, FeatureKey, HeartbeatResponse, ListInstancesResponse,
    PeerLookupResponse, ProbeResponse, RegisterRequest, RegisterResponse,
};
use crate::registry::{InMemoryRegistry, PeerRegistry, RegistryError};

/// Default liveness TTL used by the in-memory registry.
pub const DEFAULT_REGISTRATION_TTL: Duration = Duration::from_secs(30);

/// Default reaper tick interval used by the in-memory registry.
pub const DEFAULT_PRUNE_INTERVAL: Duration = Duration::from_secs(10);

/// Shared hub server state (cheap to clone, all state is inside `Arc`s).
#[derive(Clone)]
pub struct HubServerState {
    registry: Arc<dyn PeerRegistry>,
    velo: Option<Arc<velo::Velo>>,
    managers: Arc<HashMap<FeatureKey, Arc<dyn FeatureManager>>>,
}

impl std::fmt::Debug for HubServerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServerState")
            .field("peers_count", &self.registry.list().len())
            .field("velo_attached", &self.velo.is_some())
            .field("feature_managers", &self.managers.len())
            .finish()
    }
}

impl Default for HubServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl HubServerState {
    /// Create fresh, empty hub state backed by a default
    /// [`InMemoryRegistry`] and no velo participant.
    pub fn new() -> Self {
        let mem: Arc<InMemoryRegistry> = Arc::new(InMemoryRegistry::builder().build());
        Self {
            registry: mem,
            velo: None,
            managers: Arc::new(HashMap::new()),
        }
    }

    /// Snapshot the currently registered peers.
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.registry.list()
    }

    /// Access the underlying registry (useful for tests / advanced usage).
    pub fn registry(&self) -> &Arc<dyn PeerRegistry> {
        &self.registry
    }

    /// Access the hub's Velo instance, if one was attached.
    pub fn velo(&self) -> Option<&Arc<velo::Velo>> {
        self.velo.as_ref()
    }

    fn fan_out_unregister(&self, id: InstanceId) {
        for mgr in self.managers.values() {
            mgr.on_unregister(id);
        }
    }
}

/// Builder for [`HubServer`].
#[derive(Clone)]
pub struct HubServerBuilder {
    bind_addr: IpAddr,
    discovery_port: u16,
    control_port: u16,
    registry: Option<Arc<dyn PeerRegistry>>,
    transports: Vec<Arc<dyn velo::Transport>>,
    registration_ttl: Duration,
    prune_interval: Duration,
    feature_managers: Vec<Arc<dyn FeatureManager>>,
}

impl std::fmt::Debug for HubServerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServerBuilder")
            .field("bind_addr", &self.bind_addr)
            .field("discovery_port", &self.discovery_port)
            .field("control_port", &self.control_port)
            .field("transports", &self.transports.len())
            .field("registry_injected", &self.registry.is_some())
            .field("registration_ttl", &self.registration_ttl)
            .field("prune_interval", &self.prune_interval)
            .field("feature_managers", &self.feature_managers.len())
            .finish()
    }
}

impl Default for HubServerBuilder {
    fn default() -> Self {
        Self {
            bind_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            discovery_port: protocol::DEFAULT_DISCOVERY_PORT,
            control_port: protocol::DEFAULT_CONTROL_PORT,
            registry: None,
            transports: Vec::new(),
            registration_ttl: DEFAULT_REGISTRATION_TTL,
            prune_interval: DEFAULT_PRUNE_INTERVAL,
            feature_managers: Vec::new(),
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

    /// Inject a custom `PeerRegistry` backend (etcd, consul, ...). If unset,
    /// the hub uses a default in-memory registry and spawns its own reaper.
    ///
    /// Custom backends are expected to manage their own liveness (e.g. etcd
    /// leases); the hub will not protect any self-entry on them.
    pub fn registry(mut self, r: Arc<dyn PeerRegistry>) -> Self {
        self.registry = Some(r);
        self
    }

    /// Attach a velo transport. When at least one transport is supplied, the
    /// hub builds an internal `velo::Velo` and participates in active
    /// messaging with registered clients.
    pub fn add_transport(mut self, transport: Arc<dyn velo::Transport>) -> Self {
        self.transports.push(transport);
        self
    }

    /// Override the liveness TTL used by the default in-memory registry.
    /// Ignored when a custom registry is injected via [`registry`](Self::registry).
    pub fn registration_ttl(mut self, d: Duration) -> Self {
        self.registration_ttl = d;
        self
    }

    /// Override the reaper tick interval used by the default in-memory
    /// registry. Ignored when a custom registry is injected.
    pub fn prune_interval(mut self, d: Duration) -> Self {
        self.prune_interval = d;
        self
    }

    /// Attach a [`FeatureManager`] to the hub. Each manager contributes axum
    /// routes to both listeners and receives register/unregister dispatch
    /// for its [`FeatureKey`]. Duplicate keys cause [`serve`](Self::serve) to
    /// fail.
    pub fn add_feature_manager(mut self, mgr: Arc<dyn FeatureManager>) -> Self {
        self.feature_managers.push(mgr);
        self
    }

    /// Bind both listeners and spawn them. Returns a running [`HubServer`].
    pub async fn serve(self) -> Result<HubServer> {
        // Duplicate-key guard — enforced once at startup.
        let mut managers: HashMap<FeatureKey, Arc<dyn FeatureManager>> = HashMap::new();
        for mgr in &self.feature_managers {
            let key = mgr.key();
            if managers.insert(key, Arc::clone(mgr)).is_some() {
                return Err(anyhow::anyhow!(
                    "duplicate FeatureManager registered for key {key:?}"
                ));
            }
        }

        // Two-phase registry construction: keep a concrete handle to
        // `InMemoryRegistry` (if we built one) until after we've called
        // `.protect()` with the hub's own id and installed the eviction
        // callback. After that we only need the trait object.
        let (registry, mem_concrete): (Arc<dyn PeerRegistry>, Option<Arc<InMemoryRegistry>>) =
            match self.registry {
                Some(r) => (r, None),
                None => {
                    let mem: Arc<InMemoryRegistry> = Arc::new(
                        InMemoryRegistry::builder()
                            .ttl(self.registration_ttl)
                            .prune_interval(self.prune_interval)
                            .build(),
                    );
                    let dyn_reg: Arc<dyn PeerRegistry> = mem.clone();
                    (dyn_reg, Some(mem))
                }
            };

        // Build the hub's own Velo if any transports were supplied.
        let velo = if !self.transports.is_empty() {
            let discovery: Arc<dyn velo::discovery::PeerDiscovery> = registry.clone();
            let mut vb = velo::Velo::builder().discovery(discovery);
            for t in self.transports {
                vb = vb.add_transport(t);
            }
            let v = vb.build().await.context("building hub velo")?;
            // Self-register so clients can discover the hub via
            // `GET /v1/peers/instance/{hub_id}`.
            registry
                .register(v.peer_info())
                .await
                .map_err(|e| anyhow::anyhow!("hub self-register: {e}"))?;
            if let Some(mem) = &mem_concrete {
                mem.protect(v.instance_id());
            }
            Some(v)
        } else {
            None
        };

        // Attach every manager now that the registry and (optional) Velo
        // are ready.
        let ctx = HubContext {
            velo: velo.clone(),
            registry: registry.clone(),
        };
        for (key, mgr) in &managers {
            mgr.attach(ctx.clone())
                .await
                .map_err(|e| anyhow::anyhow!("FeatureManager({key:?}) attach: {e}"))?;
        }

        let managers = Arc::new(managers);

        // Wire eviction fan-out when we own the concrete in-memory registry.
        // Custom backends manage their own eviction semantics.
        if let Some(mem) = &mem_concrete {
            let managers_for_cb = Arc::clone(&managers);
            mem.set_eviction_callback(Arc::new(move |id: InstanceId| {
                for mgr in managers_for_cb.values() {
                    mgr.on_unregister(id);
                }
            }));
        }

        let state = HubServerState {
            registry: registry.clone(),
            velo,
            managers: Arc::clone(&managers),
        };

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

        let reaper_task = registry.clone().spawn_liveness_task(cancel.child_token());

        let mut discovery_router = discovery_router(state.clone());
        let mut control_router = control_router(state.clone());
        for mgr in managers.values() {
            discovery_router = discovery_router.merge(Arc::clone(mgr).public_router());
            control_router = control_router.merge(Arc::clone(mgr).control_router());
        }

        let discovery_task = spawn_server(discovery_listener, discovery_router, cancel.clone());
        let control_task = spawn_server(control_listener, control_router, cancel.clone());

        Ok(HubServer {
            state,
            discovery_addr: discovery_local,
            control_addr: control_local,
            cancel,
            discovery_task: Some(discovery_task),
            control_task: Some(control_task),
            reaper_task,
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
    reaper_task: Option<JoinHandle<()>>,
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

    /// Shared state handle.
    pub fn state(&self) -> &HubServerState {
        &self.state
    }

    /// Trigger shutdown and await both listeners plus the reaper.
    pub async fn shutdown(mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(t) = self.discovery_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.control_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.reaper_task.take() {
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
        .route(
            protocol::paths::INSTANCES,
            get(list_instances).post(register_instance),
        )
        .route(protocol::paths::INSTANCE_BY_ID, delete(unregister_instance))
        .route(protocol::paths::INSTANCE_HEARTBEAT, post(heartbeat))
        .route(protocol::paths::INSTANCE_PROBE, post(probe_instance))
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

async fn list_instances(State(state): State<HubServerState>) -> Json<ListInstancesResponse> {
    Json(ListInstancesResponse {
        instances: state.peers(),
    })
}

async fn register_instance(
    State(state): State<HubServerState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, HubError> {
    let peer = req.peer_info;
    let instance_id = peer.instance_id();

    state
        .registry
        .register(peer.clone())
        .await
        .map_err(HubError::from_registry)?;

    // Feature dispatch. All-or-nothing: if any feature rejects the payload
    // (unknown manager, invalid config, role conflict, ...) we roll back the
    // base registration so the client sees a single consistent outcome.
    for feature in &req.features {
        let dispatch: Result<(), HubError> = match state.managers.get(&feature.key()) {
            None => Err(HubError::bad_request(format!(
                "no feature manager registered for {:?}",
                feature.key()
            ))),
            Some(mgr) => mgr
                .on_register(instance_id, feature)
                .await
                .map_err(HubError::from_feature),
        };
        if let Err(err) = dispatch {
            // Roll back: remove base entry and every manager notification we
            // already emitted (for features dispatched earlier in this loop).
            let _ = state.registry.unregister(instance_id).await;
            state.fan_out_unregister(instance_id);
            return Err(err);
        }
    }

    // Proactively inform the hub's Velo about this peer so outbound sends
    // don't need to round-trip through discovery. Matches pre-trait behavior.
    if let Some(velo) = state.velo.as_ref() {
        velo.register_peer(peer)
            .map_err(|e| HubError::internal(format!("velo register_peer: {e}")))?;
    }

    Ok(Json(RegisterResponse {
        instance_id,
        hub_instance_id: state.velo.as_ref().map(|v| v.instance_id()),
    }))
}

async fn probe_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<ProbeResponse>, HubError> {
    if !state.registry.contains(instance_id) {
        return Err(HubError::not_found(format!(
            "instance {instance_id} not registered"
        )));
    }

    let velo = state
        .velo
        .as_ref()
        .ok_or_else(|| HubError::internal("hub velo not configured".to_string()))?;

    let ack: HeartbeatAck = velo
        .typed_unary(HEARTBEAT_HANDLER)
        .map_err(|e| HubError::bad_gateway(format!("probe setup: {e}")))?
        .payload(&HeartbeatRequest { seq: 0 })
        .map_err(|e| HubError::bad_gateway(format!("probe payload: {e}")))?
        .instance(instance_id)
        .send()
        .await
        .map_err(|e| HubError::bad_gateway(format!("probe failed: {e}")))?;

    Ok(Json(ProbeResponse {
        seq: ack.seq,
        ok: ack.ok,
    }))
}

async fn unregister_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<StatusCode, HubError> {
    state
        .registry
        .unregister(instance_id)
        .await
        .map_err(HubError::from_registry)?;
    // In-memory registry's eviction callback already notifies managers; the
    // explicit fan-out here covers custom registry backends that don't wire
    // a callback. The call is cheap and manager `on_unregister` is required
    // to be idempotent.
    state.fan_out_unregister(instance_id);
    Ok(StatusCode::NO_CONTENT)
}

async fn heartbeat(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<HeartbeatResponse>, HubError> {
    match state.registry.touch(instance_id).await {
        Ok(()) => Ok(Json(HeartbeatResponse { acknowledged: true })),
        Err(RegistryError::NotFound(_)) => Ok(Json(HeartbeatResponse {
            acknowledged: false,
        })),
        Err(e) => Err(HubError::from_registry(e)),
    }
}

async fn get_peer_by_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    state
        .registry
        .discover_by_instance_id(instance_id)
        .await
        .map(|peer_info| Json(PeerLookupResponse { peer_info }))
        .map_err(|_| HubError::not_found(format!("instance {instance_id} not found")))
}

async fn get_peer_by_worker(
    State(state): State<HubServerState>,
    Path(worker_id): Path<u64>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    let wid = WorkerId::from_u64(worker_id);
    state
        .registry
        .discover_by_worker_id(wid)
        .await
        .map(|peer_info| Json(PeerLookupResponse { peer_info }))
        .map_err(|_| HubError::not_found(format!("worker {worker_id} not found")))
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
    fn from_registry(e: RegistryError) -> Self {
        match e {
            RegistryError::Conflict { .. } => Self::conflict(e.to_string()),
            RegistryError::NotFound(_) => Self::not_found(e.to_string()),
            RegistryError::Backend(err) => Self::internal(format!("registry backend: {err}")),
        }
    }

    fn from_feature(e: FeatureError) -> Self {
        match e {
            FeatureError::InvalidConfig(m) => Self::bad_request(m),
            FeatureError::KeyMismatch { .. } => Self::internal(e.to_string()),
            FeatureError::Other(err) => Self::internal(err.to_string()),
        }
    }

    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorBody {
                code: ErrorCode::NotFound,
                message,
            },
        }
    }

    fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorBody {
                code: ErrorCode::BadRequest,
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

    fn bad_gateway(message: String) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_server_state_starts_empty() {
        assert!(HubServerState::new().peers().is_empty());
    }

    #[test]
    fn hub_server_state_default_starts_empty() {
        assert!(HubServerState::default().peers().is_empty());
    }

    #[tokio::test]
    async fn builder_binds_os_assigned_ports() {
        let server = HubServerBuilder::new()
            .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
            .discovery_port(0)
            .control_port(0)
            .serve()
            .await
            .unwrap();
        assert_ne!(server.discovery_addr().port(), 0);
        assert_ne!(server.control_addr().port(), 0);
        assert_ne!(server.discovery_addr().port(), server.control_addr().port());
    }

    #[tokio::test]
    async fn server_entry_point_builder() {
        let server = HubServer::builder()
            .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
            .discovery_port(0)
            .control_port(0)
            .serve()
            .await
            .unwrap();
        assert_eq!(server.state().peers().len(), 0);
        server.shutdown().await.unwrap();
    }
}
