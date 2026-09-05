// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network Manager - Single Source of Truth for Network Configuration
//!
//! This module consolidates ALL network-related configuration and creation logic.
//! It is the ONLY place in the codebase that:
//! - Reads environment variables for network configuration
//! - Knows about transport-specific types
//! - Performs mode selection based on RequestPlaneMode
//! - Creates servers and clients
//!
//! The rest of the codebase works exclusively with trait objects and never
//! directly accesses transport implementations or configuration.

use super::egress::unified_client::RequestPlaneClient;
use super::ingress::shared_tcp_endpoint::SharedTcpServer;
use super::ingress::unified_server::RequestPlaneServer;
use crate::distributed::RequestPlaneMode;
use anyhow::Result;
use async_once_cell::OnceCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use tokio_util::sync::CancellationToken;

/// TCP server state shared by all managers running on the same Tokio runtime.
///
/// When multiple workers run in the same process, they must share a single TCP server
/// to ensure all endpoints are registered on the same server. Without this, each worker
/// would create its own server on a different port, but all would publish the same port
/// to discovery, causing "No handler found" errors.
///
/// The server's background tasks are spawned onto the current Tokio runtime, so this
/// state cannot be process-global: a test (or embedding application) may tear down one
/// runtime and create another in the same process. The second runtime must not inherit
/// the first runtime's now-dead server.
struct RuntimeTcpScope {
    server: tokio::sync::OnceCell<Arc<SharedTcpServer>>,
    actual_port: OnceLock<u16>,
    // This token deliberately belongs to the Tokio-runtime scope instead of any
    // individual NetworkManager. Multiple DistributedRuntimes can share one Tokio
    // runtime, and cancelling one manager's endpoint token must not stop the server
    // used by its siblings. It also lets Phase 1 endpoint shutdown drain queued work
    // before the runtime itself tears down this scope.
    cancellation_token: CancellationToken,
    runtime_alive: AtomicBool,
}

impl RuntimeTcpScope {
    fn new() -> Arc<Self> {
        let scope = Arc::new(Self {
            server: tokio::sync::OnceCell::const_new(),
            actual_port: OnceLock::new(),
            cancellation_token: CancellationToken::new(),
            runtime_alive: AtomicBool::new(true),
        });

        // Tokio runtime IDs may be reused after shutdown. Mark this scope dead when
        // the runtime drops so a retained NetworkManager cannot make a later runtime
        // with the same ID inherit this server.
        let guard = RuntimeShutdownGuard(Arc::downgrade(&scope));
        let cancellation_token = scope.cancellation_token.clone();
        tokio::spawn(async move {
            let _guard = guard;
            cancellation_token.cancelled().await;
        });

        scope
    }

    fn is_live(&self) -> bool {
        self.runtime_alive.load(Ordering::Acquire) && !self.cancellation_token.is_cancelled()
    }
}

impl Drop for RuntimeTcpScope {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

struct RuntimeShutdownGuard(Weak<RuntimeTcpScope>);

impl Drop for RuntimeShutdownGuard {
    fn drop(&mut self) {
        if let Some(scope) = self.0.upgrade() {
            scope.runtime_alive.store(false, Ordering::Release);
            scope.cancellation_token.cancel();
        }
    }
}

/// Weak entries avoid keeping a TCP scope alive after its managers are dropped.
/// Runtime IDs are only unique among live runtimes, so dead scopes are replaced.
static RUNTIME_TCP_SCOPES: LazyLock<Mutex<HashMap<tokio::runtime::Id, Weak<RuntimeTcpScope>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn current_runtime_tcp_scope() -> Arc<RuntimeTcpScope> {
    let runtime_id = tokio::runtime::Handle::current().id();
    let mut scopes = RUNTIME_TCP_SCOPES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if let Some(scope) = scopes.get(&runtime_id).and_then(Weak::upgrade)
        && scope.is_live()
    {
        return scope;
    }

    // A long-lived process may create many short-lived runtimes. Do not retain
    // one dead Weak entry for every runtime it has ever created.
    scopes.retain(|_, weak| weak.upgrade().is_some_and(|scope| scope.is_live()));

    let scope = RuntimeTcpScope::new();
    scopes.insert(runtime_id, Arc::downgrade(&scope));
    scope
}

/// Network configuration loaded from environment variables
#[derive(Clone)]
struct NetworkConfig {
    // TCP server configuration
    tcp_host: String,
    /// TCP port to bind to. If None, the OS will assign a free port.
    ///
    /// Each live Tokio runtime owns a distinct listener, so concurrent runtimes in
    /// one process must use OS-assigned or otherwise distinct ports.
    tcp_port: Option<u16>,

    // TCP client configuration
    tcp_client_config: super::egress::tcp_client::TcpRequestConfig,

    // NATS configuration (provided externally, not from env)
    nats_client: Option<async_nats::Client>,
}

impl NetworkConfig {
    /// Load configuration from environment variables
    ///
    /// This is the ONLY place where network-related environment variables are read.
    fn from_env(nats_client: Option<async_nats::Client>) -> Self {
        Self {
            // TCP server configuration
            // If DYN_TCP_RPC_PORT is set, use that port; otherwise None means OS will assign a free port
            tcp_host: crate::utils::tcp_rpc_host_from_env(),
            tcp_port: std::env::var("DYN_TCP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse().ok()),

            // TCP client configuration (reads DYN_TCP_* env vars)
            tcp_client_config: super::egress::tcp_client::TcpRequestConfig::from_env(),

            // NATS (external)
            nats_client,
        }
    }
}

/// Network Manager - Central coordinator for all network resources
///
/// # Responsibilities
///
/// 1. **Configuration Management**: Reads and manages all network-related environment variables
/// 2. **Server Creation**: Creates and starts request plane servers based on mode
/// 3. **Client Creation**: Creates request plane clients on demand
/// 4. **Abstraction**: Hides all transport-specific details from the rest of the codebase
///
/// # Design Principles
///
/// - **Single Source of Truth**: All network config and creation logic lives here
/// - **Lazy Initialization**: Servers are created only when first accessed
/// - **Transport Agnostic Interface**: Exposes only trait objects to callers
/// - **No Leaky Abstractions**: Transport types never escape this module
///
/// # Example
///
/// ```ignore
/// // Create manager (typically done once in DistributedRuntime)
/// let manager = NetworkManager::new(cancel_token, nats_client, component_registry, request_plane_mode);
///
/// // Get server (lazy init, cached)
/// let server = manager.server().await?;
/// server.register_endpoint(...).await?;
///
/// // Create client (not cached, lightweight)
/// let client = manager.create_client()?;
/// client.send_request(...).await?;
/// ```
pub struct NetworkManager {
    mode: RequestPlaneMode,
    config: NetworkConfig,
    server: Arc<OnceCell<Arc<dyn RequestPlaneServer>>>,
    tcp_scope: OnceLock<Arc<RuntimeTcpScope>>,
    cancellation_token: CancellationToken,
    component_registry: crate::component::Registry,
}

impl NetworkManager {
    /// Create a new network manager
    ///
    /// This is the single constructor for NetworkManager. All configuration
    /// is loaded from environment variables internally.
    ///
    /// # Arguments
    ///
    /// * `cancellation_token` - Token for graceful shutdown of servers
    /// * `nats_client` - Optional NATS client (required only for NATS mode)
    /// * `component_registry` - Component registry to get NATS service groups from
    ///
    /// # Returns
    ///
    /// Returns an Arc-wrapped NetworkManager ready to create servers and clients.
    pub fn new(
        cancellation_token: CancellationToken,
        nats_client: Option<async_nats::Client>,
        component_registry: crate::component::Registry,
        mode: RequestPlaneMode,
    ) -> Self {
        let config = NetworkConfig::from_env(nats_client);

        match mode {
            RequestPlaneMode::Tcp => {
                let port_display = config
                    .tcp_port
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "OS-assigned".to_string());
                tracing::info!(
                    %mode,
                    host = %config.tcp_host,
                    port = %port_display,
                    "Initializing NetworkManager with TCP request plane"
                );
            }
            RequestPlaneMode::Nats => {
                tracing::info!(
                    %mode,
                    "Initializing NetworkManager with NATS request plane"
                );
            }
        }

        Self {
            mode,
            config,
            server: Arc::new(OnceCell::new()),
            tcp_scope: OnceLock::new(),
            cancellation_token,
            component_registry,
        }
    }

    /// Get or create the request plane server
    ///
    /// The server is created lazily on first access and cached for subsequent calls.
    /// The server is automatically started in the background.
    ///
    /// # Returns
    ///
    /// Returns a trait object that abstracts over TCP/NATS implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Server creation fails (e.g., port already in use)
    /// - NATS mode is selected but NATS client is not available
    /// - Configuration is invalid (e.g., malformed bind address)
    pub async fn server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        let server = self
            .server
            .get_or_try_init(async { self.create_server().await })
            .await?;

        Ok(server.clone())
    }

    /// Create a new request plane client
    ///
    /// Clients are lightweight and not cached. Each call creates a new client instance.
    ///
    /// # Returns
    ///
    /// Returns a trait object that abstracts over TCP/NATS implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Client creation fails (e.g., invalid configuration)
    /// - NATS mode is selected but NATS client is not available
    pub fn create_client(&self) -> Result<Arc<dyn RequestPlaneClient>> {
        match self.mode {
            RequestPlaneMode::Tcp => self.create_tcp_client(),
            RequestPlaneMode::Nats => self.create_nats_client(),
        }
    }

    /// Get the current request plane mode
    ///
    /// This is provided primarily for logging and debugging purposes.
    /// Application logic should not branch on mode - use trait objects instead.
    pub fn mode(&self) -> RequestPlaneMode {
        self.mode
    }

    /// Return the actual bound TCP port after the request-plane server has started.
    pub fn actual_tcp_rpc_port(&self) -> Result<u16> {
        self.tcp_scope
            .get()
            .and_then(|scope| scope.actual_port.get())
            .copied()
            .ok_or_else(|| {
                tracing::error!(
                    "TCP RPC port not set - request_plane_server() must be called before actual_tcp_rpc_port()"
                );
                anyhow::anyhow!("TCP RPC port not initialized. This is not expected.")
            })
    }

    // ============================================================================
    // PRIVATE: Server Creation
    // ============================================================================

    async fn create_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        match self.mode {
            RequestPlaneMode::Tcp => self.create_tcp_server().await,
            RequestPlaneMode::Nats => self.create_nats_server().await,
        }
    }

    async fn create_tcp_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        // Share one TCP server among all managers on this Tokio runtime. Its background
        // tasks cannot safely outlive or be reused by a different runtime.
        let tcp_scope = self
            .tcp_scope
            .get_or_init(current_runtime_tcp_scope)
            .clone();
        let server = tcp_scope
            .server
            .get_or_try_init(|| async {
                // Use configured port if specified, otherwise use port 0 (OS assigns free port)
                let port = self.config.tcp_port.unwrap_or(0);
                let bind_addr = format!("{}:{}", self.config.tcp_host, port)
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Invalid TCP bind address: {}", e))?;

                tracing::info!(
                    bind_addr = %bind_addr,
                    port_source = if self.config.tcp_port.is_some() { "DYN_TCP_RPC_PORT" } else { "OS-assigned" },
                    "Creating TCP request plane server"
                );

                let server =
                    SharedTcpServer::new(bind_addr, tcp_scope.cancellation_token.clone())?;

                // Bind and start server, getting the actual bound address
                let actual_addr = server.clone().bind_and_start().await?;

                // Store the actual port in the same runtime scope as the server.
                if let Err(existing) = tcp_scope.actual_port.set(actual_addr.port()) {
                    tracing::warn!(
                        existing_port = existing,
                        new_port = actual_addr.port(),
                        "TCP RPC port already set for this runtime, ignoring new value"
                    );
                }

                tracing::info!(
                    actual_addr = %actual_addr,
                    actual_port = actual_addr.port(),
                    "TCP request plane server started"
                );

                Ok::<_, anyhow::Error>(server)
            })
            .await?;

        Ok(server.clone() as Arc<dyn RequestPlaneServer>)
    }

    async fn create_nats_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        use super::ingress::nats_server::NatsMultiplexedServer;

        let nats_client = self
            .config
            .nats_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NATS client required for NATS mode"))?;

        tracing::info!("Creating NATS request plane server");

        Ok(NatsMultiplexedServer::new(
            nats_client.clone(),
            self.component_registry.clone(),
            self.cancellation_token.clone(),
        ) as Arc<dyn RequestPlaneServer>)
    }

    // ============================================================================
    // PRIVATE: Client Creation
    // ============================================================================

    fn create_tcp_client(&self) -> Result<Arc<dyn RequestPlaneClient>> {
        use super::egress::tcp_client::TcpRequestClient;

        tracing::debug!("Creating TCP request plane client with config from NetworkManager");
        Ok(Arc::new(TcpRequestClient::with_config(
            self.config.tcp_client_config.clone(),
        )?))
    }

    fn create_nats_client(&self) -> Result<Arc<dyn RequestPlaneClient>> {
        use super::egress::nats_client::NatsRequestClient;

        let nats_client = self
            .config
            .nats_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NATS client required for NATS mode"))?;

        tracing::debug!("Creating NATS request plane client");
        Ok(Arc::new(NatsRequestClient::new(nats_client.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use std::time::Duration;

    fn manager_for(mode: RequestPlaneMode) -> NetworkManager {
        manager_for_with_token(mode, CancellationToken::new())
    }

    fn manager_for_with_token(
        mode: RequestPlaneMode,
        cancellation_token: CancellationToken,
    ) -> NetworkManager {
        NetworkManager::new(
            cancellation_token,
            None,
            crate::component::Registry::new(),
            mode,
        )
    }

    #[test]
    fn tcp_mode_creates_tcp_client_without_nats_client() {
        let tcp = manager_for(RequestPlaneMode::Tcp).create_client().unwrap();
        assert_eq!(tcp.transport_name(), "tcp");
    }

    #[test]
    fn nats_mode_requires_nats_client() {
        match manager_for(RequestPlaneMode::Nats).create_client() {
            Ok(client) => panic!(
                "expected NATS mode without NATS client to fail, got {} client",
                client.transport_name()
            ),
            Err(err) => assert!(err.to_string().contains("NATS client required")),
        }
    }

    async fn start_and_probe_tcp_server() -> (SocketAddr, tokio::runtime::Id, NetworkManager) {
        let manager = manager_for(RequestPlaneMode::Tcp);
        let server = manager.server().await.unwrap();
        let address = server
            .address()
            .strip_prefix("tcp://")
            .unwrap()
            .parse::<SocketAddr>()
            .unwrap();

        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::net::TcpStream::connect(address),
        )
        .await
        .expect("TCP server did not accept a connection before the deadline")
        .expect("TCP server address was not reachable");

        (address, tokio::runtime::Handle::current().id(), manager)
    }

    #[test]
    fn tcp_server_is_not_reused_after_its_tokio_runtime_drops() {
        temp_env::with_var_unset("DYN_TCP_RPC_PORT", || {
            let first_runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (_, first_runtime_id, first_manager) =
                first_runtime.block_on(start_and_probe_tcp_server());
            let first_scope = first_manager.tcp_scope.get().unwrap().clone();
            drop(first_runtime);

            assert!(
                !first_scope.is_live(),
                "dropping the Tokio runtime must mark its retained TCP scope dead"
            );

            let second_runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (_, second_runtime_id, second_manager) =
                second_runtime.block_on(start_and_probe_tcp_server());
            let second_scope = second_manager.tcp_scope.get().unwrap();

            assert!(second_scope.is_live());
            assert!(
                !Arc::ptr_eq(&first_scope, second_scope),
                "a new Tokio runtime must not inherit a dead TCP scope"
            );

            let scopes = RUNTIME_TCP_SCOPES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if first_runtime_id != second_runtime_id {
                assert!(
                    !scopes.contains_key(&first_runtime_id),
                    "dead entries from prior runtimes must be pruned"
                );
            }
            assert!(scopes.contains_key(&second_runtime_id));
        });
    }

    #[tokio::test]
    async fn tcp_managers_share_server_within_one_tokio_runtime() {
        temp_env::async_with_vars([("DYN_TCP_RPC_PORT", None::<&str>)], async {
            let first_manager = manager_for(RequestPlaneMode::Tcp);
            let second_manager = manager_for(RequestPlaneMode::Tcp);

            let first_server = first_manager.server().await.unwrap();
            let second_server = second_manager.server().await.unwrap();

            assert_eq!(first_server.address(), second_server.address());
            assert_eq!(
                first_manager.actual_tcp_rpc_port().unwrap(),
                second_manager.actual_tcp_rpc_port().unwrap()
            );
        })
        .await;
    }

    #[tokio::test]
    async fn manager_shutdown_does_not_stop_server_shared_on_tokio_runtime() {
        temp_env::async_with_vars([("DYN_TCP_RPC_PORT", None::<&str>)], async {
            let first_shutdown = CancellationToken::new();
            let second_shutdown = CancellationToken::new();
            let first_manager =
                manager_for_with_token(RequestPlaneMode::Tcp, first_shutdown.clone());
            let second_manager =
                manager_for_with_token(RequestPlaneMode::Tcp, second_shutdown.clone());
            let first_server = first_manager.server().await.unwrap();
            let second_server = second_manager.server().await.unwrap();
            let address = first_server
                .address()
                .strip_prefix("tcp://")
                .unwrap()
                .parse::<SocketAddr>()
                .unwrap();

            assert_eq!(first_server.address(), second_server.address());

            first_shutdown.cancel();
            tokio::task::yield_now().await;

            assert!(first_shutdown.is_cancelled());
            assert!(
                !second_shutdown.is_cancelled(),
                "manager shutdown tokens must remain independent"
            );
            assert!(
                !second_manager
                    .tcp_scope
                    .get()
                    .unwrap()
                    .cancellation_token
                    .is_cancelled(),
                "one manager must not cancel the Tokio-runtime-shared TCP scope"
            );
            tokio::time::timeout(
                Duration::from_secs(1),
                tokio::net::TcpStream::connect(address),
            )
            .await
            .expect("shared TCP server did not accept a connection before the deadline")
            .expect("shared TCP server stopped when only one manager shut down");
        })
        .await;
    }
}
