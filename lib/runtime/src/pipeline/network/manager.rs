// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network Manager - Single Source of Truth for Network Configuration
//!
//! This module consolidates ALL network-related configuration and creation logic.
//! It is the ONLY place in the codebase that:
//! - Reads environment variables for network configuration
//! - Knows about transport-specific types (SharedHttpServer, TcpRequestClient, etc.)
//! - Performs mode selection based on RequestPlaneMode
//! - Creates servers and clients
//!
//! The rest of the codebase works exclusively with trait objects and never
//! directly accesses transport implementations or configuration.

use super::egress::unified_client::RequestPlaneClient;
use super::ingress::unified_server::RequestPlaneServer;
use crate::distributed::RequestPlaneMode;
use anyhow::Result;
use async_once_cell::OnceCell;
use std::sync::Arc;
use std::sync::OnceLock;
use tokio_util::sync::CancellationToken;

/// Global storage for the actual TCP RPC port after binding.
/// Uses OnceLock since the port is set once when the server binds and never changes.
static ACTUAL_TCP_RPC_PORT: OnceLock<u16> = OnceLock::new();

/// Get the actual TCP RPC port that the server is listening on.
pub fn get_actual_tcp_rpc_port() -> anyhow::Result<u16> {
    ACTUAL_TCP_RPC_PORT.get().copied().ok_or_else(|| {
        tracing::error!(
            "TCP RPC port not set - request_plane_server() must be called before get_actual_tcp_rpc_port()"
        );
        anyhow::anyhow!(
            "TCP RPC port not initialized. This is not expected."
        )
    })
}

/// Set the actual TCP RPC port (called internally after server binds).
fn set_actual_tcp_rpc_port(port: u16) {
    if let Err(existing) = ACTUAL_TCP_RPC_PORT.set(port) {
        tracing::warn!(
            existing_port = existing,
            new_port = port,
            "TCP RPC port already set, ignoring new value"
        );
    }
}

/// Network configuration loaded from environment variables
#[derive(Clone)]
struct NetworkConfig {
    // HTTP server configuration
    http_host: String,
    http_port: u16,
    http_rpc_root: String,

    // TCP server configuration
    tcp_host: String,
    /// TCP port to bind to. If None, the OS will assign a free port.
    tcp_port: Option<u16>,

    // HTTP client configuration
    http_client_config: super::egress::http_router::Http2Config,

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
            // HTTP server configuration
            http_host: std::env::var("DYN_HTTP_RPC_HOST")
                .unwrap_or_else(|_| crate::utils::get_http_rpc_host_from_env()),
            http_port: std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8888),
            http_rpc_root: std::env::var("DYN_HTTP_RPC_ROOT_PATH")
                .unwrap_or_else(|_| "/v1/rpc".to_string()),

            // TCP server configuration
            // If DYN_TCP_RPC_PORT is set, use that port; otherwise None means OS will assign a free port
            tcp_host: std::env::var("DYN_TCP_RPC_HOST")
                .unwrap_or_else(|_| crate::utils::get_tcp_rpc_host_from_env()),
            tcp_port: std::env::var("DYN_TCP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse().ok()),

            // HTTP client configuration (reads DYN_HTTP2_* env vars)
            http_client_config: super::egress::http_router::Http2Config::from_env(),

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
            RequestPlaneMode::Http => {
                tracing::info!(
                    %mode,
                    host = %config.http_host,
                    port = config.http_port,
                    rpc_root = %config.http_rpc_root,
                    "Initializing NetworkManager with HTTP request plane"
                );
            }
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
    /// Returns a trait object that abstracts over HTTP/TCP/NATS implementations.
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
    /// Returns a trait object that abstracts over HTTP/TCP/NATS implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Client creation fails (e.g., invalid configuration)
    /// - NATS mode is selected but NATS client is not available
    pub fn create_client(&self) -> Result<Arc<dyn RequestPlaneClient>> {
        match self.mode {
            RequestPlaneMode::Http => self.create_http_client(),
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

    // ============================================================================
    // PRIVATE: Server Creation
    // ============================================================================

    async fn create_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        match self.mode {
            RequestPlaneMode::Http => self.create_http_server().await,
            RequestPlaneMode::Tcp => self.create_tcp_server().await,
            RequestPlaneMode::Nats => self.create_nats_server().await,
        }
    }

    async fn create_http_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        use super::ingress::http_endpoint::SharedHttpServer;

        let bind_addr = format!("{}:{}", self.config.http_host, self.config.http_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid HTTP bind address: {}", e))?;

        tracing::info!(
            bind_addr = %bind_addr,
            rpc_root = %self.config.http_rpc_root,
            "Creating HTTP request plane server"
        );

        let server = SharedHttpServer::new(bind_addr, self.cancellation_token.clone());

        // Start server in background
        let server_clone = server.clone();
        tokio::spawn(async move {
            if let Err(e) = server_clone.start().await {
                tracing::error!("HTTP request plane server error: {}", e);
            }
        });

        Ok(server as Arc<dyn RequestPlaneServer>)
    }

    async fn create_tcp_server(&self) -> Result<Arc<dyn RequestPlaneServer>> {
        use super::ingress::shared_tcp_endpoint::SharedTcpServer;

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

        let server = SharedTcpServer::new(bind_addr, self.cancellation_token.clone());

        // Bind and start server, getting the actual bound address
        let actual_addr = server.clone().bind_and_start().await?;

        // Store the actual bound port globally so build_transport_type() can access it
        set_actual_tcp_rpc_port(actual_addr.port());

        tracing::info!(
            actual_addr = %actual_addr,
            actual_port = actual_addr.port(),
            "TCP request plane server started"
        );

        Ok(server as Arc<dyn RequestPlaneServer>)
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

    fn create_http_client(&self) -> Result<Arc<dyn RequestPlaneClient>> {
        use super::egress::http_router::HttpRequestClient;

        tracing::debug!("Creating HTTP request plane client with config from NetworkManager");
        Ok(Arc::new(HttpRequestClient::with_config(
            self.config.http_client_config.clone(),
        )?))
    }

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
