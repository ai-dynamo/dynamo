// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for the KVBM hub.
//!
//! [`HubClient`] is both a `velo::discovery::PeerDiscovery` backend and the
//! registration handle for a local velo instance. It talks HTTP to a
//! [`HubServer`](crate::HubServer) on two ports:
//!
//! - **Discovery port** (default `1337`) — peer lookups.
//! - **Control port** (default `8337`) — registration, heartbeat.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow};
use futures::future::BoxFuture;
use reqwest::StatusCode;
use url::Url;
use velo::discovery::{PeerDiscovery, PeerRegistrationGuard};
use velo_common::{InstanceId, PeerInfo, WorkerId};

use crate::handlers;
use crate::protocol::{
    self, DEFAULT_CONTROL_PORT, DEFAULT_DISCOVERY_PORT, ErrorBody, Feature, PeerLookupResponse,
    RegisterRequest, RegisterResponse,
};

/// HTTP client for a [`HubServer`](crate::HubServer).
///
/// Construct via [`HubClientBuilder`] / [`crate::create_client_builder`].
///
/// The same `Arc<HubClient>` serves three roles:
/// 1. `Arc<dyn velo::discovery::PeerDiscovery>` — pass to `VeloBuilder::discovery`.
/// 2. Registration — [`register_instance`](Self::register_instance) stores a
///    guard in an inner `OnceLock`; dropping the client (or calling
///    [`unregister`](Self::unregister)) issues an HTTP `DELETE`.
/// 3. Control plane — [`register_handlers`](Self::register_handlers) installs
///    velo handlers (heartbeat, ...) on the caller's [`velo::Velo`] instance.
pub struct HubClient {
    config: HubClientConfig,
    http: reqwest::Client,
    guard: OnceLock<HubRegistrationGuard>,
    /// Last hub-heartbeat sequence observed via the velo handler. `0` when
    /// no heartbeat has been received.
    pub(crate) last_heartbeat_seq: AtomicU64,
    /// Last hub-heartbeat arrival time (Unix ms). `0` when no heartbeat has
    /// been received.
    pub(crate) last_heartbeat_at_ms: AtomicU64,
}

impl std::fmt::Debug for HubClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubClient")
            .field("config", &self.config)
            .field("registered", &self.guard.get().is_some())
            .finish()
    }
}

/// Resolved connection info for a [`HubClient`].
#[derive(Debug, Clone)]
pub struct HubClientConfig {
    /// Base URL of the peer-discovery HTTP endpoint (port 1337 by default).
    pub discovery_url: Url,
    /// Base URL of the control-plane HTTP endpoint (port 8337 by default).
    pub control_url: Url,
}

/// Builder for [`HubClient`].
///
/// Set `host` and (optionally) the discovery/control ports. `build()` returns
/// an `Arc<HubClient>` that is not yet registered — call
/// [`HubClient::register_instance`] after constructing `Velo`.
#[derive(Debug, Clone)]
pub struct HubClientBuilder {
    host: Option<String>,
    scheme: String,
    discovery_port: u16,
    control_port: u16,
    http_client: Option<reqwest::Client>,
}

impl Default for HubClientBuilder {
    fn default() -> Self {
        Self {
            host: None,
            scheme: "http".to_string(),
            discovery_port: DEFAULT_DISCOVERY_PORT,
            control_port: DEFAULT_CONTROL_PORT,
            http_client: None,
        }
    }
}

impl HubClientBuilder {
    /// Create a new builder with default ports (`1337` discovery, `8337` control).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hub host (name or IP). Required.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Override the URL scheme (defaults to `http`).
    pub fn scheme(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Set the peer-discovery port (default `1337`).
    pub fn port(mut self, port: u16) -> Self {
        self.discovery_port = port;
        self
    }

    /// Explicitly set the discovery port. Equivalent to [`port`](Self::port).
    pub fn discovery_port(mut self, port: u16) -> Self {
        self.discovery_port = port;
        self
    }

    /// Set the control-plane port (default `8337`).
    pub fn control_port(mut self, port: u16) -> Self {
        self.control_port = port;
        self
    }

    /// Provide a pre-configured `reqwest::Client` (e.g. with custom timeouts
    /// or TLS config). Optional — a default client is built otherwise.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Build the `Arc<HubClient>`.
    pub fn build(self) -> Result<Arc<HubClient>> {
        let host = self
            .host
            .ok_or_else(|| anyhow!("HubClientBuilder: host is required"))?;
        let discovery_url = Url::parse(&format!(
            "{}://{}:{}",
            self.scheme, host, self.discovery_port
        ))
        .context("building discovery URL")?;
        let control_url = Url::parse(&format!("{}://{}:{}", self.scheme, host, self.control_port))
            .context("building control URL")?;

        let http = match self.http_client {
            Some(c) => c,
            None => reqwest::Client::builder()
                .build()
                .context("building default reqwest client")?,
        };

        Ok(Arc::new(HubClient {
            config: HubClientConfig {
                discovery_url,
                control_url,
            },
            http,
            guard: OnceLock::new(),
            last_heartbeat_seq: AtomicU64::new(0),
            last_heartbeat_at_ms: AtomicU64::new(0),
        }))
    }
}

impl HubClient {
    /// Connection info.
    pub fn config(&self) -> &HubClientConfig {
        &self.config
    }

    /// Register the velo control-plane handlers supplied by the hub client on
    /// the given `Velo` instance. Must be called **before**
    /// [`register_instance`](Self::register_instance) so the hub can reach the
    /// instance via velo active messaging immediately after registration.
    ///
    /// Currently registers:
    /// - [`handlers::HEARTBEAT_HANDLER`] — velo-level liveness probe.
    pub fn register_handlers(self: &Arc<Self>, velo: &velo::Velo) -> Result<()> {
        self.register_handlers_messenger(velo.messenger())
    }

    /// Register the velo control-plane handlers directly on a [`velo::Messenger`].
    ///
    /// Equivalent to [`register_handlers`](Self::register_handlers); prefer
    /// this when the caller holds only an [`Arc<velo::Messenger>`] (e.g. from
    /// `kvbm_engine::runtime::KvbmRuntime::messenger`) rather than a full
    /// [`velo::Velo`].
    pub fn register_handlers_messenger(
        self: &Arc<Self>,
        messenger: &Arc<velo::Messenger>,
    ) -> Result<()> {
        messenger.register_handler(handlers::create_heartbeat_handler(Arc::clone(self)))?;
        Ok(())
    }

    /// Register an instance with the hub.
    ///
    /// Stores the returned RAII guard in an inner `OnceLock`. Subsequent calls
    /// return an error. Dropping the `HubClient` issues an HTTP `DELETE`
    /// against the control-plane port.
    ///
    /// Returns the hub's own velo `InstanceId` when the hub runs with a velo
    /// participant, or `None` when the hub is discovery-only. Callers can use
    /// the returned id to look up the hub's `PeerInfo` via
    /// [`discover_by_instance_id`](velo::discovery::PeerDiscovery::discover_by_instance_id)
    /// and wire it into their own Velo for bidirectional messaging.
    pub async fn register_instance(&self, peer_info: PeerInfo) -> Result<Option<InstanceId>> {
        self.register_instance_inner(peer_info, Vec::new()).await
    }

    /// Register an instance with the hub, declaring feature participation.
    ///
    /// Semantically identical to [`register_instance`](Self::register_instance)
    /// except each entry in `features` is dispatched to its hub-side
    /// [`FeatureManager`](crate::features::FeatureManager) after base
    /// registration. If any feature manager rejects the payload, the hub
    /// rolls back base registration and returns an error — this method
    /// surfaces that as an `Err` and does **not** store a registration guard.
    pub async fn register_instance_with_features(
        &self,
        peer_info: PeerInfo,
        features: Vec<Feature>,
    ) -> Result<Option<InstanceId>> {
        self.register_instance_inner(peer_info, features).await
    }

    async fn register_instance_inner(
        &self,
        peer_info: PeerInfo,
        features: Vec<Feature>,
    ) -> Result<Option<InstanceId>> {
        if self.guard.get().is_some() {
            anyhow::bail!("HubClient: instance already registered");
        }
        let instance_id = peer_info.instance_id();
        let url = self.control_url(protocol::paths::INSTANCES)?;
        let body = RegisterRequest {
            peer_info,
            features,
        };
        let resp = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .context("POST /v1/instances")?;
        let parsed: RegisterResponse = parse_json(resp).await?;

        let guard = HubRegistrationGuard::new(
            self.http.clone(),
            self.config.control_url.clone(),
            instance_id,
        );
        self.guard
            .set(guard)
            .map_err(|_| anyhow!("HubClient: instance already registered (race)"))?;
        Ok(parsed.hub_instance_id)
    }

    /// Explicitly unregister the current instance (if any).
    ///
    /// Prefer this over relying on `Drop` when you can await — `Drop` only
    /// fires `DELETE` best-effort from the current tokio runtime.
    pub async fn unregister(&self) -> Result<()> {
        let Some(guard) = self.guard.get() else {
            return Ok(());
        };
        guard.unregister_http().await
    }

    /// Return `true` if an instance is currently registered.
    pub fn is_registered(&self) -> bool {
        self.guard.get().is_some()
    }

    /// Send a control-plane heartbeat for the registered instance over HTTP.
    ///
    /// Returns an error if no instance has been registered on this client.
    pub async fn send_heartbeat(&self) -> Result<()> {
        let instance_id = self
            .guard
            .get()
            .map(|g| g.instance_id)
            .ok_or_else(|| anyhow!("HubClient: no registered instance to heartbeat"))?;
        let url = self.control_url(&protocol::instance_heartbeat(instance_id))?;
        let resp = self
            .http
            .post(url)
            .send()
            .await
            .context("POST /v1/instances/{id}/heartbeat")?;
        let _: protocol::HeartbeatResponse = parse_json(resp).await?;
        Ok(())
    }

    fn discovery_url(&self, path: &str) -> Result<Url> {
        self.config
            .discovery_url
            .join(path)
            .with_context(|| format!("joining discovery path {path}"))
    }

    fn control_url(&self, path: &str) -> Result<Url> {
        self.config
            .control_url
            .join(path)
            .with_context(|| format!("joining control path {path}"))
    }

    async fn lookup(&self, path: String) -> Result<PeerInfo> {
        let url = self.discovery_url(&path)?;
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .with_context(|| format!("GET {path}"))?;
        let body: PeerLookupResponse = parse_json(resp).await?;
        Ok(body.peer_info)
    }
}

impl PeerDiscovery for HubClient {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(self.lookup(protocol::peers_by_worker(worker_id)))
    }

    fn discover_by_instance_id(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(self.lookup(protocol::peers_by_instance(instance_id)))
    }
}

/// RAII guard for a single registered instance.
///
/// Issues an HTTP `DELETE` on drop (best-effort, from whatever tokio runtime
/// is available) or via [`PeerRegistrationGuard::unregister`].
pub struct HubRegistrationGuard {
    http: reqwest::Client,
    control_url: Url,
    instance_id: InstanceId,
    unregistered: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for HubRegistrationGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubRegistrationGuard")
            .field("control_url", &self.control_url.as_str())
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl HubRegistrationGuard {
    fn new(http: reqwest::Client, control_url: Url, instance_id: InstanceId) -> Self {
        Self {
            http,
            control_url,
            instance_id,
            unregistered: std::sync::atomic::AtomicBool::new(false),
        }
    }

    async fn unregister_http(&self) -> Result<()> {
        if self
            .unregistered
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return Ok(());
        }
        let url = self
            .control_url
            .join(&protocol::instance_by_id(self.instance_id))
            .context("joining instance delete path")?;
        let resp = self
            .http
            .delete(url)
            .send()
            .await
            .context("DELETE /v1/instances/{id}")?;
        if !resp.status().is_success() && resp.status() != StatusCode::NOT_FOUND {
            let body: ErrorBody = parse_json(resp).await?;
            return Err(anyhow!("hub returned error: {body}"));
        }
        Ok(())
    }
}

impl PeerRegistrationGuard for HubRegistrationGuard {
    fn unregister(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(self.unregister_http())
    }
}

impl Drop for HubRegistrationGuard {
    fn drop(&mut self) {
        if self.unregistered.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let http = self.http.clone();
        let url = match self
            .control_url
            .join(&protocol::instance_by_id(self.instance_id))
        {
            Ok(u) => u,
            Err(_) => return,
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = http.delete(url).send().await;
            });
        }
    }
}

async fn parse_json<T: serde::de::DeserializeOwned>(resp: reqwest::Response) -> Result<T> {
    let status = resp.status();
    if status.is_success() {
        return resp
            .json::<T>()
            .await
            .context("decoding hub success response");
    }
    let url = resp.url().clone();
    let body_bytes = resp.bytes().await.context("reading hub error body")?;
    if let Ok(err) = serde_json::from_slice::<ErrorBody>(&body_bytes) {
        return Err(anyhow!("hub {url} returned {status}: {err}"));
    }
    let snippet = String::from_utf8_lossy(&body_bytes);
    Err(anyhow!(
        "hub {url} returned {status}: {} bytes of body: {snippet}",
        body_bytes.len()
    ))
}
