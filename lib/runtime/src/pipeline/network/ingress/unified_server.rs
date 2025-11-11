// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Request Plane Server Interface
//!
//! This module defines a transport-agnostic interface for receiving requests
//! in the request plane. All transport implementations (TCP, HTTP, NATS)
//! implement this trait to provide a consistent interface for the ingress endpoint.

use super::*;
use anyhow::Result;
use async_trait::async_trait;
use std::net::SocketAddr;
use std::sync::Arc;

/// Instance information for server registration
#[derive(Clone)]
pub struct InstanceInfo {
    pub instance_id: u64,
    pub namespace: String,
    pub component_name: String,
    pub endpoint_name: String,
    pub system_health: Arc<parking_lot::Mutex<crate::SystemHealth>>,
}

/// Unified interface for request plane servers
///
/// This trait abstracts over different transport mechanisms (TCP, HTTP, NATS)
/// providing a consistent interface for receiving and handling requests.
///
/// # Design Principles
///
/// 1. **Transport Agnostic**: Implementations can be swapped without changing handler logic
/// 2. **Async by Default**: All operations are async to support high concurrency
/// 3. **Graceful Shutdown**: Servers should support graceful shutdown via cancellation tokens
/// 4. **Multiplexing Support**: HTTP-like transports can handle multiple endpoints on one port
/// 5. **Handler Pattern**: All servers use the `PushWorkHandler` trait for processing requests
///
/// # Example
///
/// ```ignore
/// use dynamo_runtime::pipeline::network::ingress::RequestPlaneServer;
///
/// async fn start_server(server: Arc<dyn RequestPlaneServer>) -> Result<()> {
///     let bind_addr = "0.0.0.0:8080".parse()?;
///     let handler = Arc::new(MyRequestHandler::new());
///     let instance_info = InstanceInfo { /* ... */ };
///
///     server.start(bind_addr, handler, instance_info).await?;
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    /// Start the server on the specified address
    ///
    /// This method starts the server and blocks until it's shut down via the cancellation token.
    /// For single-endpoint transports (TCP,), this binds to the address and handles
    /// all requests with the provided handler.
    ///
    /// For multiplexed transports (HTTP), this starts the shared server that can handle
    /// multiple endpoints.
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Socket address to bind to (e.g., `0.0.0.0:8080`)
    /// * `handler` - Request handler implementing `PushWorkHandler`
    /// * `instance_info` - Instance metadata for logging and health checks
    ///
    /// # Returns
    ///
    /// Returns when the server shuts down gracefully. Returns an error if:
    /// - Failed to bind to the address
    /// - Critical error during operation
    ///
    /// # Cancellation
    ///
    /// The server should monitor the cancellation token passed during construction
    /// and shut down gracefully when cancelled.
    async fn start(
        &self,
        bind_addr: SocketAddr,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: InstanceInfo,
    ) -> Result<()>;

    /// Register an endpoint handler (for multiplexed transports)
    ///
    /// For transports that support handling multiple endpoints on a single port
    /// (like HTTP), this registers a handler for a specific subject/path.
    ///
    /// For single-endpoint transports (TCP,), this method may:
    /// - Return an error (not supported)
    /// - Be a no-op (already registered via `start`)
    ///
    /// # Arguments
    ///
    /// * `subject` - Endpoint identifier (e.g., HTTP path, NATS subject)
    /// * `handler` - Request handler for this endpoint
    /// * `instance_info` - Instance metadata
    ///
    /// # Example
    ///
    /// ```ignore
    /// // HTTP transport - register multiple endpoints on same port
    /// server.register_endpoint(
    ///     "namespace.default.component.worker.endpoint.generate.instance.0".to_string(),
    ///     handler1,
    ///     instance_info1,
    /// ).await?;
    ///
    /// server.register_endpoint(
    ///     "namespace.default.component.worker.endpoint.healthcheck.instance.0".to_string(),
    ///     handler2,
    ///     instance_info2,
    /// ).await?;
    /// ```
    async fn register_endpoint(
        &self,
        subject: String,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: InstanceInfo,
    ) -> Result<()>;

    /// Unregister an endpoint handler
    ///
    /// Removes a previously registered endpoint handler. Only applicable for
    /// multiplexed transports.
    ///
    /// # Arguments
    ///
    /// * `subject` - Endpoint identifier to unregister
    /// * `endpoint_name` - Endpoint name for health status updates
    async fn unregister_endpoint(&self, subject: &str, endpoint_name: &str) -> Result<()>;

    /// Stop the server gracefully
    ///
    /// This method triggers graceful shutdown:
    /// 1. Stop accepting new connections
    /// 2. Wait for in-flight requests to complete (up to a timeout)
    /// 3. Close all connections
    /// 4. Release resources
    ///
    /// Note: Most implementations use a cancellation token for shutdown,
    /// so this method may be a no-op that just returns Ok(()).
    async fn stop(&self) -> Result<()>;

    /// Get the transport name
    ///
    /// Returns a static string identifier for the transport type.
    ///
    /// # Examples
    ///
    /// - `"tcp"` - Raw TCP transport
    /// - `"http"` or `"http2"` - HTTP/2 transport
    /// - `"nats"` - NATS messaging
    /// - `"zmq"` - ZeroMQ
    /// - `"uds"` - Unix Domain Sockets
    fn transport_name(&self) -> &'static str;

    /// Get server statistics (optional)
    ///
    /// Returns runtime statistics about the server for monitoring and debugging.
    /// Default implementation returns empty statistics.
    fn stats(&self) -> ServerStats {
        ServerStats::default()
    }

    /// Check if server supports endpoint multiplexing
    ///
    /// Returns `true` if the transport supports handling multiple endpoints
    /// on a single port (like HTTP). Returns `false` for single-endpoint
    /// transports (like TCP,).
    fn supports_multiplexing(&self) -> bool {
        false
    }
}

/// Server runtime statistics
///
/// Used for monitoring and debugging transport server performance.
#[derive(Debug, Clone, Default)]
pub struct ServerStats {
    /// Total number of requests received
    pub requests_received: u64,

    /// Total number of requests handled successfully
    pub requests_handled: u64,

    /// Total number of errors
    pub errors: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Total bytes sent (acknowledgments)
    pub bytes_sent: u64,

    /// Number of active connections
    pub active_connections: usize,

    /// Number of in-flight requests
    pub inflight_requests: usize,

    /// Average request processing time in microseconds (0 if not available)
    pub avg_processing_time_us: u64,
}

impl ServerStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if statistics are available (non-zero)
    pub fn is_available(&self) -> bool {
        self.requests_received > 0 || self.active_connections > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_stats_default() {
        let stats = ServerStats::default();
        assert_eq!(stats.requests_received, 0);
        assert_eq!(stats.requests_handled, 0);
        assert!(!stats.is_available());
    }

    #[test]
    fn test_server_stats_is_available() {
        let mut stats = ServerStats::default();
        assert!(!stats.is_available());

        stats.requests_received = 1;
        assert!(stats.is_available());

        let mut stats2 = ServerStats::default();
        stats2.active_connections = 1;
        assert!(stats2.is_available());
    }
}
