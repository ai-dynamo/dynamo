// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: (DEP-635) this file should be renamed to system_http_server.rs
//  it is being used not just for status, health, but others like loras management.

#[cfg(test)]
mod probe_tests;

use crate::config::HealthStatus;
use crate::config::environment_names::logging as env_logging;
use crate::config::environment_names::runtime::canary as env_canary;
use crate::config::environment_names::runtime::system as env_system;
use crate::metrics::MetricsHierarchy;
use crate::traits::DistributedRuntimeProvider;
use crate::utils::ip_resolver::{
    DefaultIpResolver, IpResolutionError, IpResolver, resolve_advertise_ip_for_bind,
};
use axum::{
    Router,
    body::Bytes,
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{any, delete, get, post},
    serve::Listener,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

/// Selects probe semantics for the runtime-owned HTTP server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SystemProbePolicy {
    /// Preserve the configured worker health handler for both probes.
    #[default]
    Worker,
    /// Static liveness and readiness based on runtime dependencies, not the engine.
    RuntimeOnly,
}

/// System status server information containing socket address and handle
#[derive(Debug)]
pub struct SystemStatusServerInfo {
    pub socket_addr: SocketAddr,
    pub handle: Option<Arc<JoinHandle<()>>>,
}

impl SystemStatusServerInfo {
    pub fn new(socket_addr: SocketAddr, handle: Option<JoinHandle<()>>) -> Self {
        Self {
            socket_addr,
            handle: handle.map(Arc::new),
        }
    }

    pub fn address(&self) -> String {
        self.socket_addr.to_string()
    }

    pub fn hostname(&self) -> String {
        self.socket_addr.ip().to_string()
    }

    pub fn port(&self) -> u16 {
        self.socket_addr.port()
    }

    /// Return an address that is served by the bound socket.
    ///
    /// A wildcard bind is replaced with a usable address from the same address
    /// family. If interface enumeration fails, the same-family loopback is
    /// used so IPv4 and IPv6 are never combined.
    pub fn advertised_socket_addr(&self) -> SocketAddr {
        match advertised_socket_addr(self.socket_addr, &DefaultIpResolver) {
            Ok(address) => {
                if self.socket_addr.ip().is_unspecified() && address.ip().is_loopback() {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        tracing::warn!(
                            bind = %self.socket_addr,
                            advertised = %address,
                            "System status server found no non-loopback address in the bound family, so remote callers cannot reach the advertised address; on IPv6-only hosts, set DYN_SYSTEM_HOST=::"
                        );
                    });
                }
                address
            }
            Err(error) => {
                let fallback = match self.socket_addr.ip() {
                    IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
                    IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
                };
                tracing::warn!(
                    %error,
                    %fallback,
                    "Failed to resolve the system status advertisement address; using loopback"
                );
                SocketAddr::new(fallback, self.socket_addr.port())
            }
        }
    }
}

fn advertised_socket_addr<R: IpResolver>(
    bound: SocketAddr,
    resolver: &R,
) -> Result<SocketAddr, IpResolutionError> {
    let advertise_ip = resolve_advertise_ip_for_bind(bound.ip(), resolver)?;
    Ok(SocketAddr::new(advertise_ip, bound.port()))
}

impl Clone for SystemStatusServerInfo {
    fn clone(&self) -> Self {
        Self {
            socket_addr: self.socket_addr,
            handle: self.handle.clone(),
        }
    }
}

/// System status server state containing the distributed runtime reference
pub struct SystemStatusState {
    // global drt registry is for printing out the entire Prometheus format output
    root_drt: Arc<crate::DistributedRuntime>,
    // Last /metrics render, behind the gate that serializes renders. Overlapping
    // scrapes wait here on the runtime and share one blocking render.
    last_scrape: Arc<tokio::sync::Mutex<LastScrape>>,
    // Discovery metadata (only for Kubernetes backend)
    discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
}

/// The most recent `/metrics` render and when it started collecting.
#[derive(Default)]
struct LastScrape {
    started_at: Option<Instant>,
    text: bytes::Bytes,
}

impl SystemStatusState {
    /// Create new system status server state with the provided distributed runtime
    pub fn new(
        drt: Arc<crate::DistributedRuntime>,
        discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            root_drt: drt,
            last_scrape: Arc::new(tokio::sync::Mutex::new(LastScrape::default())),
            discovery_metadata,
        })
    }

    /// Get a reference to the distributed runtime
    pub fn drt(&self) -> &crate::DistributedRuntime {
        &self.root_drt
    }

    /// Get a reference to the discovery metadata if available
    pub fn discovery_metadata(
        &self,
    ) -> Option<&Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>> {
        self.discovery_metadata.as_ref()
    }
}

/// Request body for POST /v1/loras
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadLoraRequest {
    pub lora_name: String,
    pub source: LoraSource,
}

/// Source information for loading a LoRA
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraSource {
    pub uri: String,
}

/// Response body for LoRA operations
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
}

const SYSTEM_STATUS_REBIND_BACKOFF: Duration = Duration::from_secs(1);

/// Start the complete system HTTP server after runtime initialization.
pub async fn spawn_system_status_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
    discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
    policy: SystemProbePolicy,
) -> anyhow::Result<(SocketAddr, JoinHandle<()>)> {
    // Create system status server state with the provided distributed runtime
    let server_state = Arc::new(SystemStatusState::new(drt, discovery_metadata)?);
    let app = build_system_status_router(server_state, policy);

    let initial_bind_address = format!("{}:{}", host, port);
    tracing::info!("[spawn_system_status_server] binding to: {initial_bind_address}");
    let (listener, actual_address) = bind_system_status_listener(&initial_bind_address)
        .await
        .map_err(|error| {
            tracing::error!("Failed to bind to address {initial_bind_address}: {error}");
            error
        })?;
    tracing::info!("[spawn_system_status_server] system status server bound to: {actual_address}");

    // Reuse the concrete address so an ephemeral port remains stable across rebinds.
    let listener =
        RebindingTcpListener::new(listener, actual_address, SYSTEM_STATUS_REBIND_BACKOFF);
    let observer = cancel_token.child_token();

    // Spawn the server in the background and return the handle
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
        {
            tracing::error!("System status server error: {e}");
        }
    });

    Ok((actual_address, handle))
}

fn build_system_status_router(
    server_state: Arc<SystemStatusState>,
    policy: SystemProbePolicy,
) -> Router {
    let system_health = server_state.drt().system_health();
    let (health_path, live_path) = {
        let health = system_health.lock();
        (
            health.health_path().to_string(),
            health.live_path().to_string(),
        )
    };
    let mut app = match policy {
        SystemProbePolicy::Worker => Router::new()
            .route(
                &health_path,
                get({
                    let state = Arc::clone(&server_state);
                    move || health_handler(state, Probe::Readiness)
                }),
            )
            .route(
                &live_path,
                get({
                    let state = Arc::clone(&server_state);
                    move || health_handler(state, Probe::Liveness)
                }),
            ),
        SystemProbePolicy::RuntimeOnly => Router::new()
            .route(&live_path, get(|| async { StatusCode::OK }))
            .route(
                &health_path,
                get({
                    let state = Arc::clone(&server_state);
                    move || runtime_health_handler(state)
                }),
            ),
    };
    // Check if LoRA feature is enabled
    let lora_enabled =
        crate::config::env_is_truthy(crate::config::environment_names::llm::DYN_LORA_ENABLED);

    app = app
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        )
        .route(
            "/metadata",
            get({
                let state = Arc::clone(&server_state);
                move || metadata_handler(state)
            }),
        )
        .route(
            "/engine/{*path}",
            any({
                let state = Arc::clone(&server_state);
                move |path, method, body| engine_route_handler(state, path, method, body)
            }),
        )
        .route(
            HOLD_PATH,
            get({
                let state = Arc::clone(&server_state);
                move || hold_status_handler(state)
            }),
        )
        .route(
            HOLD_RELEASE_PATH,
            post({
                let state = Arc::clone(&server_state);
                move |body| hold_release_handler(state, body)
            }),
        );

    // Add LoRA routes only if DYN_LORA_ENABLED is set to true
    if lora_enabled {
        app = app
            .route(
                "/v1/loras",
                get({
                    let state = Arc::clone(&server_state);
                    move || list_loras_handler(State(state))
                })
                .post({
                    let state = Arc::clone(&server_state);
                    move |body| load_lora_handler(State(state), body)
                }),
            )
            .route(
                "/v1/loras/{*lora_name}",
                delete({
                    let state = Arc::clone(&server_state);
                    move |path| unload_lora_handler(State(state), path)
                }),
            );
    }

    // Self-hosted MDC files. Always mounted; empty registry → 404.
    // The endpoint triple disambiguates multi-LocalModel-per-DRT; the
    // suffix segment (LoRA slug or `_base`) scopes per-registration so
    // detaching one doesn't wipe another's entries.
    app.route(
        "/v1/metadata/{namespace}/{component}/{endpoint}/{model_slug}/{model_suffix}/{*filename}",
        get({
            let state = Arc::clone(&server_state);
            move |path| metadata_file_handler(State(state), path)
        }),
    )
    .fallback(|uri: axum::http::Uri| async move {
        tracing::debug!(%uri, "system status server has no route for this request");
        (StatusCode::NOT_FOUND, "Route not found").into_response()
    })
    .layer(TraceLayer::new_for_http().make_span_with(crate::logging::make_system_request_span))
}

async fn bind_system_status_listener(
    address: &str,
) -> anyhow::Result<(TcpListener, std::net::SocketAddr)> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| anyhow::anyhow!("Failed to bind to address: {error}"))?;

    let actual_address = listener.local_addr()?;

    Ok((listener, actual_address))
}

/// Axum's [`Listener::accept`] cannot return an error, so recovery must happen here.
struct RebindingTcpListener {
    /// Concrete address returned by the initial bind, including an assigned ephemeral port.
    address: std::net::SocketAddr,
    listener: Option<TcpListener>,
    rebind_backoff: Duration,
}

impl RebindingTcpListener {
    fn new(listener: TcpListener, address: std::net::SocketAddr, rebind_backoff: Duration) -> Self {
        Self {
            address,
            listener: Some(listener),
            rebind_backoff,
        }
    }
}

impl Listener for RebindingTcpListener {
    type Io = tokio::net::TcpStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let mut next_rebind = tokio::time::Instant::now();
        loop {
            let Some(listener) = self.listener.as_ref() else {
                tokio::time::sleep_until(next_rebind).await;
                next_rebind = tokio::time::Instant::now() + self.rebind_backoff;
                tracing::debug!(address = %self.address, "System status server rebinding");
                match TcpListener::bind(self.address).await {
                    Ok(listener) => {
                        tracing::info!(address = %self.address, "System status server rebound");
                        self.listener = Some(listener);
                    }
                    Err(error) => {
                        tracing::error!(
                            address = %self.address,
                            %error,
                            retry_after = ?self.rebind_backoff,
                            "System status server failed to rebind"
                        );
                    }
                }
                continue;
            };

            match listener.accept().await {
                Ok(accepted) => return accepted,
                // Connection errors do not invalidate the listener.
                Err(error) if is_connection_error(&error) => {
                    tracing::trace!(
                        address = %self.address,
                        %error,
                        "System status connection failed before accept"
                    );
                }
                // Resource exhaustion does not invalidate the listener.
                Err(error) if is_resource_exhaustion_error(&error) => {
                    tracing::error!(
                        address = %self.address,
                        %error,
                        retry_after = ?self.rebind_backoff,
                        "System status listener lacks resources; retrying the same socket"
                    );
                    tokio::time::sleep(self.rebind_backoff).await;
                }
                Err(error) => {
                    tracing::error!(
                        address = %self.address,
                        %error,
                        "System status listener stopped accepting; rebinding"
                    );
                    // Drop before rebinding or the address remains in use.
                    self.listener = None;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.address)
    }
}

fn is_connection_error(error: &io::Error) -> bool {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return true;
    }

    // Linux accept(2) also reports pending network errors for the queued connection.
    #[cfg(target_os = "linux")]
    if let Some(code) = error.raw_os_error() {
        return matches!(
            code,
            libc::ENETDOWN
                | libc::EPROTO
                | libc::ENOPROTOOPT
                | libc::EHOSTDOWN
                | libc::ENONET
                | libc::EHOSTUNREACH
                | libc::EOPNOTSUPP
                | libc::ENETUNREACH
        );
    }

    false
}

/// EMFILE, ENFILE, and ENOBUFS require raw errno checks because Rust has no distinct kinds.
fn is_resource_exhaustion_error(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::OutOfMemory {
        return true;
    }

    #[cfg(unix)]
    {
        if let Some(code) = error.raw_os_error() {
            return matches!(code, libc::ENFILE | libc::EMFILE | libc::ENOBUFS);
        }
    }

    false
}

/// Which probe a health route answers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Probe {
    /// Not ready while the worker is held: it takes no ordinary traffic yet, so nothing that
    /// waits on its readiness, such as a rollout, should move on.
    Readiness,
    /// Unaffected by the hold. A held worker is alive and must not be restarted for it.
    Liveness,
}

/// Health handler with optional active health checking
#[tracing::instrument(skip_all, level = "trace")]
async fn health_handler(state: Arc<SystemStatusState>, probe: Probe) -> impl IntoResponse {
    // Get basic health status
    let system_health = state.drt().system_health();
    let system_health_lock = system_health.lock();
    let (healthy, endpoints) = system_health_lock.get_health_status();
    let uptime = Some(system_health_lock.uptime());
    drop(system_health_lock);

    let held = probe == Probe::Readiness && state.drt().is_routing_held();
    let healthy = healthy && !held;
    let healthy_string = if healthy { "ready" } else { "notready" };
    let status_code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let mut response = json!({
        "status": healthy_string,
        "uptime": uptime,
        "endpoints": endpoints,
    });
    if held {
        response["held"] = json!(true);
    }

    tracing::trace!("Response {}", response.to_string());

    (status_code, response.to_string())
}

/// Sidecar readiness is independent of engine health and model registration.
async fn runtime_health_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    let held = state.drt().is_routing_held();
    let ready = !held
        && matches!(
            tokio::time::timeout(Duration::from_secs(1), state.drt().check_dependencies()).await,
            Ok(Ok(()))
        );
    let (code, status) = if ready {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "notready")
    };
    let mut response = json!({ "status": status });
    if held {
        response["held"] = json!(true);
    }
    (code, Json(response))
}

/// Reports whether this worker is held out of automatic routing, and for which instance.
const HOLD_PATH: &str = "/v1/hold";
/// Releases this worker's hold. See [`hold_release_handler`].
const HOLD_RELEASE_PATH: &str = "/v1/hold/release";

/// A worker instance ID in a request body. IDs can exceed the integers a JSON number carries
/// exactly in some clients, so a decimal string is accepted too.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum InstanceIdField {
    Number(u64),
    Text(String),
}

#[derive(serde::Deserialize)]
struct HoldReleaseRequest {
    instance_id: InstanceIdField,
}

fn hold_status(drt: &crate::DistributedRuntime) -> serde_json::Value {
    json!({
        "instance_id": drt.connection_id(),
        "held": drt.is_routing_held(),
    })
}

/// `GET /v1/hold`: the hold of the worker instance that runs in this process. A controller
/// reads the instance ID here to tell this worker from an older one at the same address.
async fn hold_status_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    Json(hold_status(state.drt()))
}

/// `POST /v1/hold/release` with `{"instance_id": <id>}`: release the hold of that instance.
///
/// - `200`: the worker is released. `released` is false when it already was, so a retry is safe.
/// - `400`: the body names no instance.
/// - `409`: the body names another instance. This worker keeps its hold.
/// - `500`: the discovery records could not all be updated. The worker is still held, and
///   repeating the release completes it.
async fn hold_release_handler(state: Arc<SystemStatusState>, body: Bytes) -> Response {
    use crate::routing_hold::{HoldRelease, HoldReleaseError};

    let drt = state.drt();
    let rejected = |code: StatusCode, error: String| {
        let mut response = hold_status(drt);
        response["error"] = json!(error);
        (code, Json(response)).into_response()
    };

    let instance_id = match serde_json::from_slice::<HoldReleaseRequest>(&body) {
        Ok(HoldReleaseRequest {
            instance_id: InstanceIdField::Number(instance_id),
        }) => Some(instance_id),
        Ok(HoldReleaseRequest {
            instance_id: InstanceIdField::Text(instance_id),
        }) => instance_id.parse::<u64>().ok(),
        Err(_) => None,
    };
    let Some(instance_id) = instance_id else {
        return rejected(
            StatusCode::BAD_REQUEST,
            "the request body must be a JSON object with the worker's instance_id".to_string(),
        );
    };

    // The release runs in its own task. A caller that hangs up must not stop it between
    // writing the discovery records and clearing the hold: that would leave a worker that
    // reports held while routers already send it ordinary traffic.
    let release = {
        let drt = drt.clone();
        tokio::spawn(async move { drt.release_routing_hold(instance_id).await })
    };
    let outcome = match release.await {
        Ok(outcome) => outcome,
        Err(error) => {
            tracing::error!(%error, "the worker routing hold release did not finish");
            return rejected(
                StatusCode::INTERNAL_SERVER_ERROR,
                "the release did not finish".to_string(),
            );
        }
    };
    match outcome {
        Ok(outcome) => {
            let mut response = hold_status(drt);
            response["released"] = json!(outcome == HoldRelease::Released);
            Json(response).into_response()
        }
        Err(error @ HoldReleaseError::StaleInstance { .. }) => {
            rejected(StatusCode::CONFLICT, error.to_string())
        }
        Err(error @ HoldReleaseError::Discovery(_)) => {
            tracing::error!(%error, "failed to release the worker routing hold");
            rejected(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
        }
    }
}

/// Metrics handler with DistributedRuntime uptime
#[tracing::instrument(skip_all, level = "trace")]
async fn metrics_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Get all metrics from the DistributedRuntime.
    // The uptime gauge is updated automatically via a PrometheusUpdateCallback
    // registered in DistributedRuntime::new(), so it is always fresh before scrape.
    //
    // NOTE: We use a multi-registry model (e.g. one registry per endpoint) and merge at scrape time,
    // so /metrics traverses registered child registries and produces a single combined output.
    let registry = state.drt().get_metrics_registry().clone();
    match render_scrape_at(state.last_scrape.clone(), registry, Instant::now()).await {
        Ok(text) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; charset=utf-8",
            )],
            text,
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Failed to get metrics from registry: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get metrics").into_response()
        }
    }
}

/// Render `/metrics` for a scrape that arrived at `requested_at`. A render
/// that started collecting after the scrape arrived is reused; otherwise the
/// scrape renders on the blocking pool. The owned guard moves into the
/// closure, so a cancelled request cannot release the gate before its render
/// finishes and let the next scrape queue inside the pool.
async fn render_scrape_at(
    last_scrape: Arc<tokio::sync::Mutex<LastScrape>>,
    registry: crate::metrics::MetricsRegistry,
    requested_at: Instant,
) -> anyhow::Result<bytes::Bytes> {
    let mut last = last_scrape.lock_owned().await;
    if last.started_at.is_some_and(|at| at >= requested_at) {
        return Ok(last.text.clone());
    }
    tokio::task::spawn_blocking(move || {
        let started_at = Instant::now();
        let text = bytes::Bytes::from(registry.prometheus_expfmt_combined()?);
        *last = LastScrape {
            started_at: Some(started_at),
            text: text.clone(),
        };
        Ok(text)
    })
    .await
    .map_err(|e| anyhow::anyhow!("metrics render task failed: {e}"))?
}

/// Metadata handler
#[tracing::instrument(skip_all, level = "trace")]
async fn metadata_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Check if discovery metadata is available
    let metadata = match state.discovery_metadata() {
        Some(metadata) => metadata,
        None => {
            tracing::debug!("Metadata endpoint called but no discovery metadata available");
            return (
                StatusCode::NOT_FOUND,
                "Discovery metadata not available".to_string(),
            )
                .into_response();
        }
    };

    // Read the metadata
    let metadata_guard = metadata.read().await;

    // Serialize to JSON
    match serde_json::to_string(&*metadata_guard) {
        Ok(json) => {
            tracing::trace!("Returning metadata: {} bytes", json.len());
            (StatusCode::OK, json).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to serialize metadata: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to serialize metadata".to_string(),
            )
                .into_response()
        }
    }
}

/// Handler for POST /v1/loras - Load a LoRA adapter
#[tracing::instrument(skip_all, level = "debug")]
async fn load_lora_handler(
    State(state): State<Arc<SystemStatusState>>,
    Json(request): Json<LoadLoraRequest>,
) -> impl IntoResponse {
    tracing::info!("Loading LoRA: {}", request.lora_name);

    // Call the load_lora endpoint for each available backend
    match call_lora_endpoint(
        state.drt(),
        "load_lora",
        json!({
            "lora_name": request.lora_name,
            "source": {
                "uri": request.source.uri
            },
        }),
    )
    .await
    {
        Ok(response) => {
            if response.status == "error" {
                tracing::error!(
                    "Failed to load LoRA {}: {}",
                    request.lora_name,
                    response.message.as_deref().unwrap_or("Unknown error")
                );
                (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
            } else {
                tracing::info!("LoRA loaded successfully: {}", request.lora_name);
                (StatusCode::OK, Json(response))
            }
        }
        Err(e) => {
            tracing::error!("Failed to load LoRA {}: {}", request.lora_name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: Some(request.lora_name),
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// Handler for DELETE /v1/loras/*lora_name - Unload a LoRA adapter
#[tracing::instrument(skip_all, level = "debug")]
async fn unload_lora_handler(
    State(state): State<Arc<SystemStatusState>>,
    Path(lora_name): Path<String>,
) -> impl IntoResponse {
    // Strip the leading slash from the wildcard capture
    let lora_name = lora_name
        .strip_prefix('/')
        .unwrap_or(&lora_name)
        .to_string();
    tracing::info!("Unloading LoRA: {lora_name}");

    // Call the unload_lora endpoint for each available backend
    match call_lora_endpoint(
        state.drt(),
        "unload_lora",
        json!({
            "lora_name": lora_name.clone(),
        }),
    )
    .await
    {
        Ok(response) => {
            if response.status == "error" {
                tracing::error!(
                    "Failed to unload LoRA {}: {}",
                    lora_name,
                    response.message.as_deref().unwrap_or("Unknown error")
                );
                (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
            } else {
                tracing::info!("LoRA unloaded successfully: {lora_name}");
                (StatusCode::OK, Json(response))
            }
        }
        Err(e) => {
            tracing::error!("Failed to unload LoRA {}: {}", lora_name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: Some(lora_name),
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// Handler for GET /v1/loras - List all LoRA adapters
#[tracing::instrument(skip_all, level = "debug")]
async fn list_loras_handler(State(state): State<Arc<SystemStatusState>>) -> impl IntoResponse {
    tracing::info!("Listing all LoRAs");

    // Call the list_loras endpoint for each available backend
    match call_lora_endpoint(state.drt(), "list_loras", json!({})).await {
        Ok(response) => {
            tracing::info!("Successfully retrieved LoRA list");
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to list LoRAs: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: None,
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// `GET /v1/metadata/{namespace}/{component}/{endpoint}/{slug}/{suffix}/{filename}`
/// — 404 on miss, 500 on read error, raw bytes on hit. Consumer blake3-verifies.
async fn metadata_file_handler(
    State(state): State<Arc<SystemStatusState>>,
    Path((namespace, component, endpoint, model_slug, model_suffix, filename)): Path<(
        String,
        String,
        String,
        String,
        String,
        String,
    )>,
) -> impl IntoResponse {
    let path = match state.drt().metadata_artifacts().get(
        &namespace,
        &component,
        &endpoint,
        &model_slug,
        &model_suffix,
        &filename,
    ) {
        Some(p) => p,
        None => {
            tracing::debug!(
                namespace,
                component,
                endpoint,
                model_slug,
                model_suffix,
                filename,
                "metadata artifact not registered for self-host"
            );
            return (StatusCode::NOT_FOUND, "Not found").into_response();
        }
    };

    match tokio::fs::read(&path).await {
        Ok(bytes) => (StatusCode::OK, bytes).into_response(),
        Err(err) => {
            tracing::error!(
                namespace,
                component,
                endpoint,
                model_slug,
                model_suffix,
                filename,
                path = %path.display(),
                %err,
                "failed to read self-hosted metadata file"
            );
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read file").into_response()
        }
    }
}

/// Helper function to call a LoRA management endpoint for the local worker.
///
/// Resolution order (both are in-process, never network discovery):
/// 1. The legacy local endpoint registry, populated by non-unified workers via
///    `.register_local_engine()`.
/// 2. The generic engine-route registry (`/engine/*`). Unified-backend workers
///    advertise LoRA lifecycle ops (`load_lora`/`unload_lora`/`list_loras`) as
///    engine *updates*, registered under `update/<name>`; this fallback maps the
///    bare LoRA name onto that key so the legacy `/v1/loras` surface forwards to
///    them (the `/v1/loras` compatibility shim).
///
/// Because legacy workers populate the local registry, they never reach the
/// fallback — their `/v1/loras` behavior is unchanged. If neither registry
/// holds the name, returns an explicit "LoRA management not available" error
/// rather than an opaque "endpoint not found".
async fn call_lora_endpoint(
    drt: &crate::DistributedRuntime,
    endpoint_name: &str,
    request_body: serde_json::Value,
) -> anyhow::Result<LoraResponse> {
    use crate::engine::AsyncEngine;

    tracing::debug!("Calling LoRA endpoint: '{endpoint_name}'");

    // 1. Legacy local registry (in-process call only).
    if let Some(engine) = drt.local_endpoint_registry().get(endpoint_name) {
        tracing::debug!(
            "Found endpoint '{}' in local registry, calling directly",
            endpoint_name
        );

        let request = crate::pipeline::SingleIn::new(request_body);
        let mut stream = engine.generate(request).await?;

        if let Some(response) = stream.next().await {
            let response_data = response.data.unwrap_or_default();
            let lora_response = serde_json::from_value::<LoraResponse>(response_data.clone())
                .unwrap_or_else(|_| parse_lora_response(&response_data));
            return Ok(lora_response);
        }

        anyhow::bail!("No response received from endpoint '{}'", endpoint_name)
    }

    // 2. Unified-backend engine-update registry fallback. The unified Worker
    //    registers LoRA ops as engine updates under `update/<name>`, so map the
    //    bare LoRA endpoint name onto that namespaced key.
    let update_key = format!("update/{endpoint_name}");
    if let Some(callback) = drt.engine_routes().get(&update_key) {
        tracing::debug!(
            "Found '{}' in engine routes registry, invoking update callback",
            update_key
        );
        let response_data = callback(request_body).await?;
        let lora_response = serde_json::from_value::<LoraResponse>(response_data.clone())
            .unwrap_or_else(|_| parse_lora_response(&response_data));
        return Ok(lora_response);
    }

    anyhow::bail!(
        "LoRA management not available: no '{}' handler is registered \
         (neither a local LoRA endpoint nor an engine update). This worker \
         either has LoRA disabled or its backend does not support LoRA \
         management.",
        endpoint_name
    )
}

/// Helper to parse response data into LoraResponse
fn parse_lora_response(response_data: &serde_json::Value) -> LoraResponse {
    LoraResponse {
        status: response_data
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("success")
            .to_string(),
        message: response_data
            .get("message")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string()),
        lora_name: response_data
            .get("lora_name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        lora_id: response_data.get("lora_id").and_then(|id| id.as_u64()),
        loras: response_data.get("loras").cloned(),
        count: response_data
            .get("count")
            .and_then(|c| c.as_u64())
            .map(|c| c as usize),
    }
}

/// Engine route handler for /engine/* routes
///
/// This handler looks up registered callbacks in the engine routes registry
/// and invokes them with the request body, returning the response as JSON.
#[tracing::instrument(skip_all, level = "trace", fields(path = %path))]
async fn engine_route_handler(
    state: Arc<SystemStatusState>,
    Path(path): Path<String>,
    method: axum::http::Method,
    body: Bytes,
) -> Response {
    crate::engine_routes::dispatch(state.drt().engine_routes(), path, method, body).await
}

// Regular tests: cargo test system_status_server --lib
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ip_resolver::test_support::StubResolver;
    use tokio::time::Duration;

    /// Engine callback whose renders block until the test releases them, and
    /// which reports each render start, so the tests never sleep-and-hope.
    fn held_registry() -> (
        crate::metrics::MetricsRegistry,
        tokio::sync::mpsc::UnboundedReceiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let registry = crate::metrics::MetricsRegistry::new();
        let (started_tx, started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = std::sync::Mutex::new(release_rx);
        registry.add_expfmt_callback(Arc::new(move || {
            let _ = started_tx.send(());
            release_rx.lock().unwrap().recv().expect("release");
            Ok("engine_metric 1\n".to_string())
        }));
        (registry, started_rx, release_tx)
    }

    /// A scrape that arrives while a render is in flight waits on the runtime,
    /// not in the blocking pool, and then renders once for every later arrival
    /// to share; cancelling the active request does not release the gate.
    #[test]
    fn scrapes_queue_on_the_gate_and_share_the_next_render() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            let (registry, mut started, release) = held_registry();
            let last = Arc::new(tokio::sync::Mutex::new(LastScrape::default()));
            let scrape = |last: &Arc<tokio::sync::Mutex<LastScrape>>| {
                let (last, registry) = (last.clone(), registry.clone());
                tokio::spawn(async move { render_scrape_at(last, registry, Instant::now()).await })
            };

            // First scrape starts a render, then its request is cancelled.
            let first = scrape(&last);
            started.recv().await.expect("first render started");
            first.abort();
            let _ = first.await;

            // Two scrapes arrive during that render. They must wait on the gate,
            // leaving the pool's other thread free for unrelated work.
            let second = scrape(&last);
            let third = scrape(&last);
            tokio::time::timeout(Duration::from_secs(5), tokio::task::spawn_blocking(|| ()))
                .await
                .expect("a waiting scrape occupied a blocking thread")
                .expect("probe");

            // Releasing the cancelled render lets the second scrape render; the
            // third arrived before that render started and shares it.
            release.send(()).unwrap();
            started.recv().await.expect("second render started");
            release.send(()).unwrap();
            let second = second.await.expect("task").expect("render");
            let third = third.await.expect("task").expect("render");
            assert_eq!(second, third);
            assert!(
                std::str::from_utf8(&second)
                    .unwrap()
                    .contains("engine_metric 1")
            );
            assert!(
                started.try_recv().is_err(),
                "the third scrape must share the second render, not start its own"
            );
        });
    }

    #[test]
    fn advertised_address_stays_in_the_bound_family() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![
            ("lo", "127.0.0.1".parse().unwrap()),
            ("eth0", "2001:db8::20".parse().unwrap()),
        ];

        let advertised = advertised_socket_addr("0.0.0.0:8080".parse().unwrap(), &resolver)
            .expect("wildcard advertisement should resolve");

        assert_eq!(advertised, "127.0.0.1:8080".parse().unwrap());
    }

    // This is a basic test to verify the HTTP server is working before testing other more complicated tests
    async fn status_and_body(response: impl IntoResponse) -> (StatusCode, serde_json::Value) {
        let response = response.into_response();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    /// A worker that starts held is discovered as held and not ready, and stays alive. Only a
    /// release for its own instance clears the hold, in place, and a repeat is a no-op.
    #[tokio::test]
    async fn held_worker_reports_its_hold_and_releases_it_for_its_own_instance() {
        let runtime = crate::Runtime::from_current().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::new(
                runtime.clone(),
                crate::distributed::DistributedConfig::process_local(),
            )
            .await
            .unwrap(),
        );
        let state = Arc::new(SystemStatusState::new(drt.clone(), None).unwrap());
        let (ready_code, _) =
            status_and_body(health_handler(state.clone(), Probe::Readiness).await).await;
        let (live_code, _) =
            status_and_body(health_handler(state.clone(), Probe::Liveness).await).await;

        drt.hold_routing_for_test();
        let endpoint = drt
            .namespace("held-worker".to_string())
            .unwrap()
            .component("worker".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        endpoint.register_endpoint_instance().await.unwrap();
        let client = endpoint.client().await.unwrap();
        let registered = client.wait_for_instances().await.unwrap();
        assert!(registered[0].held);
        let instance_id = drt.connection_id();
        // A router sees the worker and keeps it out of automatic routing.
        let mut dispatchable = client.instance_dispatch_watcher();
        tokio::time::timeout(
            Duration::from_secs(1),
            dispatchable.wait_for(|instances| instances.is_held(instance_id)),
        )
        .await
        .expect("the held worker was not discovered")
        .unwrap();
        let mut routable = client.instance_avail_watcher();
        assert!(routable.borrow_and_update().is_empty());

        // Status names the instance. Readiness fails for the hold; liveness does not move.
        let (code, body) = status_and_body(hold_status_handler(state.clone()).await).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body, json!({"instance_id": instance_id, "held": true}));
        let (code, body) =
            status_and_body(health_handler(state.clone(), Probe::Readiness).await).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "notready");
        assert_eq!(body["held"], true);
        let (code, body) =
            status_and_body(health_handler(state.clone(), Probe::Liveness).await).await;
        assert_eq!(code, live_code);
        assert!(body.get("held").is_none());
        let (code, body) = status_and_body(runtime_health_handler(state.clone()).await).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["held"], true);

        let release = |body: serde_json::Value| {
            let state = state.clone();
            async move {
                status_and_body(
                    hold_release_handler(state, Bytes::from(serde_json::to_vec(&body).unwrap()))
                        .await,
                )
                .await
            }
        };

        // A body that names no instance, and a release meant for another instance, leave
        // the worker held.
        let (code, body) = release(json!({})).await;
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["held"], true);
        let (code, body) = release(json!({"instance_id": instance_id.wrapping_add(1)})).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert_eq!(body["instance_id"], instance_id);
        assert_eq!(body["held"], true);
        assert!(drt.is_routing_held());

        // The release, with the instance ID as a decimal string.
        let (code, body) = release(json!({"instance_id": instance_id.to_string()})).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(
            body,
            json!({"instance_id": instance_id, "held": false, "released": true})
        );
        // The discovery record changed in place: same instance, hold cleared.
        let mut instances = client.instance_source.as_ref().clone();
        let released = tokio::time::timeout(
            Duration::from_secs(1),
            instances.wait_for(|instances| instances.iter().all(|instance| !instance.held)),
        )
        .await
        .expect("the endpoint record kept its hold")
        .unwrap()
        .clone();
        assert_eq!(
            released,
            vec![crate::component::Instance {
                held: false,
                ..registered[0].clone()
            }]
        );
        // The router that watched the record now routes to the worker.
        tokio::time::timeout(
            Duration::from_secs(1),
            routable.wait_for(|instance_ids| instance_ids == &[instance_id]),
        )
        .await
        .expect("the released worker did not become routable")
        .unwrap();
        assert!(!client.is_instance_held(instance_id));

        // A retry is safe, and the worker is ready again.
        let (code, body) = release(json!({"instance_id": instance_id})).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["released"], false);
        let (code, body) =
            status_and_body(health_handler(state.clone(), Probe::Readiness).await).await;
        assert_eq!(code, ready_code);
        assert!(body.get("held").is_none());
        // An endpoint that registers after the release is not held.
        let late = drt
            .namespace("held-worker".to_string())
            .unwrap()
            .component("worker".to_string())
            .unwrap()
            .endpoint("late".to_string());
        late.register_endpoint_instance().await.unwrap();
        let late_client = late.client().await.unwrap();
        assert!(!late_client.wait_for_instances().await.unwrap()[0].held);

        runtime.shutdown();
    }

    /// A controller that gives up closes its connection, which drops the handler. The release
    /// it asked for must still finish, or the records and the worker would disagree.
    #[tokio::test]
    async fn a_release_finishes_after_its_caller_goes_away() {
        let runtime = crate::Runtime::from_current().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::new(
                runtime.clone(),
                crate::distributed::DistributedConfig::process_local(),
            )
            .await
            .unwrap(),
        );
        let state = Arc::new(SystemStatusState::new(drt.clone(), None).unwrap());
        drt.hold_routing_for_test();
        let endpoint = drt
            .namespace("held-worker-caller-gone".to_string())
            .unwrap()
            .component("worker".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        endpoint.register_endpoint_instance().await.unwrap();
        let client = endpoint.client().await.unwrap();
        assert!(client.wait_for_instances().await.unwrap()[0].held);

        // An endpoint that is registering makes the release wait, so the caller leaves first.
        let registration = drt.routing_hold().registration().await;
        let body = serde_json::to_vec(&json!({"instance_id": drt.connection_id()})).unwrap();
        let mut call = Box::pin(hold_release_handler(state.clone(), Bytes::from(body)));
        assert!(futures::poll!(&mut call).is_pending());
        drop(call);
        drop(registration);

        let mut instances = client.instance_source.as_ref().clone();
        tokio::time::timeout(
            Duration::from_secs(5),
            instances.wait_for(|instances| instances.iter().all(|instance| !instance.held)),
        )
        .await
        .expect("the release stopped when its caller went away")
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while drt.is_routing_held() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the worker still reports held after its records were released");

        runtime.shutdown();
    }

    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // Test basic HTTP server lifecycle without DistributedRuntime
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
        });

        // server starts immediately, no need to wait

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_rebinding_listener_serves_again_after_listener_shutdown() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();

        let shutdown_result = socket2::SockRef::from(&listener).shutdown(std::net::Shutdown::Both);

        let mut rebinding = RebindingTcpListener::new(listener, address, Duration::from_millis(10));

        if let Err(error) = shutdown_result {
            eprintln!("skipping: this platform refused shutdown on a listening socket: {error}");
            return;
        }

        let accepted = tokio::spawn(async move { rebinding.accept().await });

        let mut accepted = accepted;
        let mut clients = Vec::new();
        let peer = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let Ok(stream) = tokio::net::TcpStream::connect(address).await else {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                };
                let client_addr = stream.local_addr().unwrap();
                clients.push((stream, client_addr));
                tokio::select! {
                    result = &mut accepted => {
                        let (_io, peer) = result.expect("accept task should not panic");
                        return peer;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
            }
        })
        .await
        .expect("listener should rebind and accept connections again");

        assert!(
            clients.iter().any(|(_, client_addr)| *client_addr == peer),
            "accepted connection should be one we opened"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_rebinding_listener_keeps_trying_after_a_rebind_fails() {
        let holder = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let contested_address = holder.local_addr().unwrap();

        let broken = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let shutdown_result = socket2::SockRef::from(&broken).shutdown(std::net::Shutdown::Both);

        let mut rebinding =
            RebindingTcpListener::new(broken, contested_address, Duration::from_millis(10));

        if let Err(error) = shutdown_result {
            eprintln!("skipping: this platform refused shutdown on a listening socket: {error}");
            return;
        }

        let accepted = tokio::spawn(async move { rebinding.accept().await });

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(holder);

        let client = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(stream) = tokio::net::TcpStream::connect(contested_address).await {
                    return stream;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("listener should rebind once the address is free again");

        let (_io, peer) = tokio::time::timeout(Duration::from_secs(5), accepted)
            .await
            .expect("accept should return once a later rebind succeeds")
            .expect("accept task should not panic");

        assert_eq!(
            peer,
            client.local_addr().unwrap(),
            "accepted connection should be the one we opened"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_resource_exhaustion_does_not_look_like_a_dead_listener() {
        for code in [libc::ENFILE, libc::ENOBUFS] {
            let error = io::Error::from_raw_os_error(code);
            assert!(
                is_resource_exhaustion_error(&error),
                "errno {code} should be treated as a process resource shortage, got {:?}",
                error.kind()
            );
            assert!(
                !is_connection_error(&error),
                "errno {code} is not a dead connection"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_pending_network_errors_leave_the_listener_usable() {
        for code in [
            libc::ENETDOWN,
            libc::EPROTO,
            libc::ENOPROTOOPT,
            libc::EHOSTDOWN,
            libc::ENONET,
            libc::EHOSTUNREACH,
            libc::EOPNOTSUPP,
            libc::ENETUNREACH,
        ] {
            let error = io::Error::from_raw_os_error(code);
            assert!(
                is_connection_error(&error),
                "errno {code} should retry accept"
            );
        }
    }

    #[tokio::test]
    async fn test_rebinding_listener_shutdown_interrupts_failed_rebinds() {
        let holder = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listener = RebindingTcpListener {
            address: holder.local_addr().unwrap(),
            listener: None,
            rebind_backoff: Duration::from_secs(60),
        };
        let cancel_token = CancellationToken::new();
        let observer = cancel_token.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new())
                .with_graceful_shutdown(observer.cancelled_owned())
                .await
                .unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !server.is_finished(),
            "server should keep retrying the held address"
        );
        cancel_token.cancel();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("shutdown should interrupt the 60-second rebind backoff")
            .expect("server task should not panic");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_rebinding_listener_preserves_backlog_during_resource_exhaustion() {
        const CHILD_ENV: &str = "DYNAMO_TEST_LISTENER_EMFILE_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            // Descriptor limits are process-wide; isolate this test from sibling tests.
            let output = tokio::time::timeout(
                Duration::from_secs(15),
                tokio::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "system_status_server::tests::test_rebinding_listener_preserves_backlog_during_resource_exhaustion",
                        "--nocapture",
                    ])
                    .env(CHILD_ENV, "1")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .expect("descriptor-exhaustion child should finish promptly")
            .expect("test child should start");
            assert!(
                output.status.success(),
                "descriptor-exhaustion child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = tokio::net::TcpStream::connect(address).await.unwrap();
        let peer = client.local_addr().unwrap();

        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: limit is writable and correctly aligned. Only this child is affected.
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(96);
        // SAFETY: limit is initialized; the soft limit is lowered without changing the hard limit.
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);

        let mut descriptors = Vec::new();
        loop {
            match std::fs::File::open("/dev/null") {
                Ok(file) => descriptors.push(file),
                Err(error) => {
                    assert_eq!(error.raw_os_error(), Some(libc::EMFILE));
                    break;
                }
            }
        }
        let error = listener.accept().await.unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EMFILE));

        let backoff = Duration::from_millis(200);
        let mut rebinding = RebindingTcpListener::new(listener, address, backoff);
        let started = tokio::time::Instant::now();
        let accepted = rebinding.accept();
        tokio::pin!(accepted);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), accepted.as_mut())
                .await
                .is_err(),
            "accept should wait during descriptor exhaustion"
        );
        drop(descriptors);

        // A fresh bind rules out lingering descriptor pressure as the cause of AddrInUse.
        let _control = TcpListener::bind("127.0.0.1:0").await.unwrap();
        assert_eq!(
            TcpListener::bind(address).await.unwrap_err().kind(),
            io::ErrorKind::AddrInUse,
            "the original listening address should remain held"
        );
        let (_stream, accepted_peer) = tokio::time::timeout(Duration::from_secs(2), accepted)
            .await
            .expect("the queued connection should survive the shortage");
        assert!(
            started.elapsed() >= backoff,
            "accept should finish its resource-shortage backoff before retrying"
        );
        assert_eq!(accepted_peer, peer);
    }
}

// Integration tests: cargo test system_status_server --lib --features integration
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::config::environment_names::logging as env_logging;
    use crate::config::environment_names::runtime::canary as env_canary;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::metrics::MetricsHierarchy;
    use anyhow::Result;
    use rstest::rstest;
    use std::sync::Arc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_uptime_from_system_health() {
        // Test that uptime is available from SystemHealth
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Get uptime from SystemHealth
            let uptime = drt.system_health().lock().uptime();
            // Uptime should exist (even if close to zero)
            assert!(uptime.as_nanos() > 0 || uptime.is_zero());

            // Sleep briefly and check uptime increases
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let uptime_after = drt.system_health().lock().uptime();
            assert!(uptime_after > uptime);
        })
        .await;
    }

    #[tokio::test]
    async fn test_runtime_metrics_initialization_and_namespace() {
        // Test that metrics have correct namespace
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;
            // SystemStatusState is already created in distributed.rs
            // so we don't need to create it again here

            // The uptime_seconds metric should already be registered and available
            let response = drt.metrics().prometheus_expfmt().unwrap();
            println!("Full metrics response:\n{}", response);

            // Check that uptime_seconds metric is present with correct namespace
            assert!(
                response.contains("# HELP dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds help text"
            );
            assert!(
                response.contains("# TYPE dynamo_component_uptime_seconds gauge"),
                "Should contain uptime_seconds type"
            );
            assert!(
                response.contains("dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds metric with correct namespace"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_uptime_gauge_updates() {
        // Test that the uptime gauge is properly updated and increases over time
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Get initial uptime
            let initial_uptime = drt.system_health().lock().uptime();

            // Update the gauge with initial value
            drt.system_health().lock().update_uptime_gauge();

            // Sleep for 100ms
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            // Get uptime after sleep
            let uptime_after_sleep = drt.system_health().lock().uptime();

            // Update the gauge again
            drt.system_health().lock().update_uptime_gauge();

            // Verify uptime increased by at least 100ms
            let elapsed = uptime_after_sleep - initial_uptime;
            assert!(
                elapsed >= std::time::Duration::from_millis(100),
                "Uptime should have increased by at least 100ms after sleep, but only increased by {:?}",
                elapsed
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_http_requests_fail_when_system_disabled() {
        // Test that system status server is not running when disabled
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Verify that system status server info is None when disabled
            let system_info = drt.system_status_server_info();
            assert!(
                system_info.is_none(),
                "System status server should not be running when disabled"
            );

            println!("✓ System status server correctly disabled when not enabled");
        })
        .await;
    }

    /// This test verifies the health and liveness endpoints of the system status server.
    /// It checks that the endpoints respond with the correct HTTP status codes and bodies
    /// based on the initial health status and any custom endpoint paths provided via environment variables.
    /// The test is parameterized using multiple #[case] attributes to cover various scenarios,
    /// including different initial health states ("ready" and "notready"), default and custom endpoint paths,
    /// and expected response codes and bodies.
    #[rstest]
    #[case("ready", 200, "ready", None, None, 3)]
    #[case("notready", 503, "notready", None, None, 3)]
    #[case("ready", 200, "ready", Some("/custom/health"), Some("/custom/live"), 5)]
    #[case(
        "notready",
        503,
        "notready",
        Some("/custom/health"),
        Some("/custom/live"),
        5
    )]
    #[tokio::test]
    #[cfg(feature = "integration")]
    async fn test_health_endpoints(
        #[case] starting_health_status: &'static str,
        #[case] expected_status: u16,
        #[case] expected_body: &'static str,
        #[case] custom_health_path: Option<&'static str>,
        #[case] custom_live_path: Option<&'static str>,
        #[case] expected_num_tests: usize,
    ) {
        use std::sync::Arc;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests

        // Closure call is needed here to satisfy async_with_vars

        crate::logging::init();

        #[allow(clippy::redundant_closure_call)]
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (
                    env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
                    Some(starting_health_status),
                ),
                (env_system::DYN_SYSTEM_HEALTH_PATH, custom_health_path),
                (env_system::DYN_SYSTEM_LIVE_PATH, custom_live_path),
            ],
            (async || {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;

                let client = reqwest::Client::new();

                // Prepare test cases
                let mut test_cases = vec![];
                match custom_health_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/health", expected_status, expected_body));
                    }
                    Some(chp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/health", 404, "Route not found"));
                        test_cases.push((chp, expected_status, expected_body));
                    }
                }
                match custom_live_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/live", expected_status, expected_body));
                    }
                    Some(clp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/live", 404, "Route not found"));
                        test_cases.push((clp, expected_status, expected_body));
                    }
                }
                test_cases.push(("/someRandomPathNotFoundHere", 404, "Route not found"));
                assert_eq!(test_cases.len(), expected_num_tests);

                for (path, expect_status, expect_body) in test_cases {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    assert_eq!(
                        status, expect_status,
                        "Response: status={}, body={:?}",
                        status, body
                    );
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
            })(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_health_endpoint_tracing() -> Result<()> {
        use std::sync::Arc;

        // Closure call is needed here to satisfy async_with_vars

        #[allow(clippy::redundant_closure_call)]
        let _ = temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("ready")),
                (env_logging::DYN_LOGGING_JSONL, Some("1")),
                (env_logging::DYN_LOG, Some("trace")),
            ],
            (async || {
                // TODO Add proper testing for
                // trace id and parent id

                crate::logging::init();

                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for path in [("/health"), ("/live"), ("/someRandomPathNotFoundHere")] {
                    let traceparent_value =
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
                    let tracestate_value = "vendor1=opaqueValue1,vendor2=opaqueValue2";
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::HeaderName::from_static("traceparent"),
                        reqwest::header::HeaderValue::from_str(traceparent_value)?,
                    );
                    headers.insert(
                        reqwest::header::HeaderName::from_static("tracestate"),
                        reqwest::header::HeaderValue::from_str(tracestate_value)?,
                    );
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).headers(headers).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    tracing::info!(body = body, status = status.to_string());
                }

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    #[tokio::test]
    async fn test_health_endpoint_with_changing_health_status() {
        // Test health endpoint starts in not ready status, then becomes ready
        // when endpoints are created (DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=generate)
        const ENDPOINT_NAME: &str = "generate";
        const ENDPOINT_HEALTH_CONFIG: &str = "[\"generate\"]";
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("notready")),
                (env_system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS, Some(ENDPOINT_HEALTH_CONFIG)),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Check if system status server was started
                let system_info_opt = drt.system_status_server_info();

                // Ensure system status server was spawned by DRT
                assert!(
                    system_info_opt.is_some(),
                    "System status server was not spawned by DRT. Expected DRT to spawn server when DYN_SYSTEM_PORT is set to a positive value, but system_status_server_info() returned None. Environment: DYN_SYSTEM_PORT={:?}",
                    std::env::var(env_system::DYN_SYSTEM_PORT)
                );

                // Get the system status server info from DRT - this should never fail now due to above check
                let system_info = system_info_opt.unwrap();
                let addr = system_info.socket_addr;

                // Initially check health - should be not ready
                let client = reqwest::Client::new();
                let health_url = format!("http://{}/health", addr);

                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();

                // Health should be not ready (503) initially
                assert_eq!(status, 503, "Health should be 503 (not ready) initially, got: {}", status);
                assert!(body.contains("\"status\":\"notready\""), "Health should contain status notready");

                // Now create a namespace, component, and endpoint to make the system healthy
                let namespace = drt.namespace("ns1234").unwrap();
                let component = namespace.component("comp1234").unwrap();

                // Create a simple test handler
                use crate::pipeline::{async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, SingleIn};
                use crate::protocols::annotated::Annotated;

                struct TestHandler;

                #[async_trait]
                impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> for TestHandler {
                    async fn generate(&self, input: SingleIn<String>) -> anyhow::Result<ManyOut<Annotated<String>>> {
                        let (data, ctx) = input.into_parts();
                        let response = Annotated::from_data(format!("You responded: {}", data));
                        Ok(crate::pipeline::ResponseStream::new(
                            Box::pin(crate::stream::iter(vec![response])),
                            ctx.context()
                        ))
                    }
                }

                // Create the ingress and start the endpoint service
                let ingress = Ingress::for_engine(std::sync::Arc::new(TestHandler)).unwrap();

                // Start the service and endpoint with a health check payload
                // This will automatically register the endpoint for health monitoring
                tokio::spawn(async move {
                    let _ = component.endpoint(ENDPOINT_NAME)
                        .endpoint_builder()
                        .handler(ingress)
                        .health_check_payload(serde_json::json!({
                            "test": "health_check"
                        }))
                        .start()
                        .await;
                });

                // Hit health endpoint 200 times to verify consistency
                let mut success_count = 0;
                let mut failures = Vec::new();

                for i in 1..=200 {
                    let response = client.get(&health_url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();

                    if status == 200 && body.contains("\"status\":\"ready\"") {
                        success_count += 1;
                    } else {
                        failures.push((i, status.as_u16(), body.clone()));
                        if failures.len() <= 5 {  // Only log first 5 failures
                            tracing::warn!("Request {}: status={}, body={}", i, status, body);
                        }
                    }
                }

                tracing::info!("Health endpoint test results: {success_count}/200 requests succeeded");
                if !failures.is_empty() {
                    tracing::warn!("Failed requests: {}", failures.len());
                }

                // Expect at least 150 out of 200 requests to be successful
                assert!(success_count >= 150, "Expected at least 150 out of 200 requests to succeed, but only {} succeeded", success_count);
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_spawn_system_status_server_endpoints() {
        // use reqwest for HTTP requests
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("ready")),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for (path, expect_200, expect_body) in [
                    ("/health", true, "ready"),
                    ("/live", true, "ready"),
                    ("/someRandomPathNotFoundHere", false, "Route not found"),
                ] {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    if expect_200 {
                        assert_eq!(status, 200, "Response: status={}, body={:?}", status, body);
                    } else {
                        assert_eq!(status, 404, "Response: status={}, body={:?}", status, body);
                    }
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
                // DRT handles server cleanup automatically
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_payloadless_endpoint_is_healthy_with_canary_enabled() {
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (
                    env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
                    Some("notready"),
                ),
                ("DYN_HEALTH_CHECK_ENABLED", Some("true")),
            ],
            async {
                let runtime = crate::Runtime::from_current().unwrap();
                let drt = Arc::new(
                    crate::DistributedRuntime::new(
                        runtime,
                        crate::distributed::DistributedConfig::process_local(),
                    )
                    .await
                    .unwrap(),
                );
                let addr = drt
                    .system_status_server_info()
                    .expect("System status server should be started")
                    .socket_addr;

                use crate::pipeline::{
                    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, SingleIn, async_trait,
                    network::Ingress,
                };
                use crate::protocols::annotated::Annotated;

                struct PayloadlessHandler;

                #[async_trait]
                impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error>
                    for PayloadlessHandler
                {
                    async fn generate(
                        &self,
                        input: SingleIn<String>,
                    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
                        let (data, ctx) = input.into_parts();
                        Ok(crate::pipeline::ResponseStream::new(
                            Box::pin(crate::stream::iter(vec![Annotated::from_data(data)])),
                            ctx.context(),
                        ))
                    }
                }

                let namespace = drt.namespace("test").unwrap();
                let component = namespace.component("backend").unwrap();
                let ingress = Ingress::for_engine(Arc::new(PayloadlessHandler)).unwrap();
                tokio::spawn(async move {
                    // Unified decode workers deliberately follow this path: the
                    // endpoint is registered without a canary payload.
                    let _ = component
                        .endpoint("generate")
                        .endpoint_builder()
                        .handler(ingress)
                        .start()
                        .await;
                });

                let client = reqwest::Client::new();
                for path in ["/health", "/live"] {
                    let url = format!("http://{addr}{path}");
                    let mut last_response = None;
                    for _ in 0..50 {
                        let response = client.get(&url).send().await.unwrap();
                        let status = response.status();
                        let body = response.text().await.unwrap();
                        if status == 200 && body.contains("\"status\":\"ready\"") {
                            last_response = Some((status, body));
                            break;
                        }
                        last_response = Some((status, body));
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                    let (status, body) = last_response.unwrap();
                    assert_eq!(status, 200, "{path} response: {body}");
                    assert!(
                        body.contains("\"status\":\"ready\""),
                        "{path} response: {body}"
                    );
                }
            },
        )
        .await;
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_health_check_with_payload_and_timeout() {
        // Test the complete health check flow with the new canary-based system:
        crate::logging::init();

        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (
                    env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
                    Some("notready"),
                ),
                (
                    env_system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS,
                    Some("[\"test.endpoint\"]"),
                ),
                // Enable health check with short intervals for testing
                ("DYN_HEALTH_CHECK_ENABLED", Some("true")),
                (env_canary::DYN_CANARY_WAIT_TIME, Some("1")), // Send canary after 1 second of inactivity
                ("DYN_HEALTH_CHECK_REQUEST_TIMEOUT", Some("1")), // Immediately timeout to mimic unresponsiveness
                ("RUST_LOG", Some("info")),                      // Enable logging for test
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started");
                let addr = system_info.socket_addr;

                let client = reqwest::Client::new();
                let health_url = format!("http://{}/health", addr);

                // Register an endpoint with health check payload
                let endpoint = "test.endpoint";
                let health_check_payload = serde_json::json!({
                    "prompt": "health check test",
                    "_health_check": true
                });

                // Register the endpoint and its health check payload
                {
                    let system_health = drt.system_health();
                    let system_health_lock = system_health.lock();
                    system_health_lock.register_health_check_target(
                        endpoint,
                        crate::component::Instance {
                            component: "test_component".to_string(),
                            endpoint: "health".to_string(),
                            namespace: "test_namespace".to_string(),
                            instance_id: 1,
                            transport: crate::component::TransportType::Nats(endpoint.to_string()),
                            device_type: None,
                            request_plane_codec: None,
                            held: false,
                        },
                        health_check_payload.clone(),
                    );
                }

                // Check initial health - should be ready (default state)
                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();
                assert_eq!(status, 503, "Should be unhealthy initially (default state)");
                assert!(
                    body.contains("\"status\":\"notready\""),
                    "Should show notready status initially"
                );

                // Set endpoint to healthy state
                drt.system_health()
                    .lock()
                    .set_endpoint_health_status(endpoint, HealthStatus::Ready);

                // Check health again - should now be healthy
                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();

                assert_eq!(status, 200, "Should be healthy due to recent response");
                assert!(
                    body.contains("\"status\":\"ready\""),
                    "Should show ready status after response"
                );

                // Verify the endpoint status in SystemHealth directly
                let endpoint_status = drt
                    .system_health()
                    .lock()
                    .get_endpoint_health_status(endpoint);
                assert_eq!(
                    endpoint_status,
                    Some(HealthStatus::Ready),
                    "SystemHealth should show endpoint as Ready after response"
                );
            },
        )
        .await;
    }

    /// `/v1/loras` compat shim: with the legacy local registry empty, a LoRA
    /// update registered in `engine_routes()` under `update/<name>` resolves via
    /// the fallback and its JSON response is parsed into a `LoraResponse`. This
    /// is the path unified-backend workers take for `/v1/loras`.
    #[tokio::test]
    async fn test_call_lora_endpoint_resolves_via_engine_routes() {
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            let callback: crate::engine_routes::EngineRouteCallback = Arc::new(|_body| {
                Box::pin(async move {
                    Ok(serde_json::json!({
                        "status": "success",
                        "lora_name": "adapterA",
                        "lora_id": 42,
                    }))
                })
            });
            // Unified Worker registers LoRA ops under the `update/` namespace.
            drt.engine_routes().register("update/load_lora", callback);

            // Local registry is empty, so resolution must fall through to
            // engine_routes.
            assert!(drt.local_endpoint_registry().get("load_lora").is_none());

            let response = call_lora_endpoint(
                &drt,
                "load_lora",
                serde_json::json!({"lora_name": "adapterA"}),
            )
            .await
            .expect("engine_routes fallback should resolve the control");

            assert_eq!(response.status, "success");
            assert_eq!(response.lora_name.as_deref(), Some("adapterA"));
            assert_eq!(response.lora_id, Some(42));
        })
        .await;
    }

    /// When neither the local registry nor `engine_routes()` holds the name,
    /// the caller gets an explicit "LoRA management not available" error
    /// rather than an opaque "endpoint not found".
    #[tokio::test]
    async fn test_call_lora_endpoint_missing_returns_clean_error() {
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            let err = call_lora_endpoint(&drt, "load_lora", serde_json::json!({}))
                .await
                .expect_err("missing handler must error");

            assert!(
                err.to_string().contains("LoRA management not available"),
                "expected explicit unavailable message, got: {err}"
            );
        })
        .await;
    }

    /// An update callback that returns `{"status":"error",...}` (rather than
    /// raising) surfaces as a `LoraResponse{status:"error"}`. The `/v1/loras`
    /// load/unload handlers map this to HTTP 500, preserving legacy semantics;
    /// the direct `/engine/*` route would instead return HTTP 200 + this JSON.
    #[tokio::test]
    async fn test_call_lora_endpoint_propagates_error_status() {
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            let callback: crate::engine_routes::EngineRouteCallback = Arc::new(|_body| {
                Box::pin(async move {
                    Ok(serde_json::json!({
                        "status": "error",
                        "message": "adapter not found",
                    }))
                })
            });
            // Unified Worker registers LoRA ops under the `update/` namespace.
            drt.engine_routes().register("update/unload_lora", callback);

            let response = call_lora_endpoint(&drt, "unload_lora", serde_json::json!({}))
                .await
                .expect("a non-raising callback returns Ok even on logical error");

            assert_eq!(response.status, "error");
            assert_eq!(response.message.as_deref(), Some("adapter not found"));
        })
        .await;
    }
}
