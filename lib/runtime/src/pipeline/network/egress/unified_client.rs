// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Request Plane Client Interface
//!
//! This module defines a transport-agnostic interface for sending requests
//! in the request plane. All transport implementations (TCP, HTTP, NATS)
//! implement this trait to provide a consistent interface for the egress router.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::HashMap;

/// Type alias for request headers
pub type Headers = HashMap<String, String>;

/// Unified interface for request plane clients
///
/// This trait abstracts over different transport mechanisms (TCP, HTTP, NATS)
/// providing a consistent interface for sending requests and receiving acknowledgments.
///
/// # Design Principles
///
/// 1. **Transport Agnostic**: Implementations can be swapped without changing router logic
/// 2. **Async by Default**: All operations are async to support high concurrency
/// 3. **Headers Support**: All transports must support custom headers for tracing, etc.
/// 4. **Health Checks**: Implementations should provide connection health information
/// 5. **Error Handling**: All errors are wrapped in anyhow::Result for flexibility
///
/// # Example
///
/// ```ignore
/// use dynamo_runtime::pipeline::network::egress::RequestPlaneClient;
///
/// async fn send_request(client: &dyn RequestPlaneClient) -> Result<()> {
///     let mut headers = HashMap::new();
///     headers.insert("x-request-id".to_string(), "123".to_string());
///
///     let response = client.send_request(
///         "service-endpoint".to_string(),
///         Bytes::from("payload"),
///         headers,
///     ).await?;
///
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    /// Send a request to a specific address and wait for acknowledgment
    ///
    /// # Arguments
    ///
    /// * `address` - Transport-specific address:
    ///   - HTTP: `http://host:port/path`
    ///   - TCP: `host:port` or `tcp://host:port`
    ///   - NATS: `subject.name`
    /// * `payload` - Request payload (encoded as bytes)
    /// * `headers` - Custom headers for tracing, authentication, etc.
    ///
    /// # Returns
    ///
    /// Returns an acknowledgment response. Note that for streaming responses,
    /// the actual response data comes over the TCP response plane, not through
    /// this acknowledgment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Connection to the endpoint fails
    /// - Request times out
    /// - Transport-specific errors occur (e.g., NATS server unavailable)
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes>;

    /// Get the transport name
    ///
    /// Returns a static string identifier for the transport type.
    /// Used for logging and debugging.
    ///
    /// # Examples
    ///
    /// - `"tcp"` - Raw TCP transport
    /// - `"http"` or `"http2"` - HTTP/2 transport
    /// - `"nats"` - NATS messaging
    fn transport_name(&self) -> &'static str;

    /// Check connection health
    ///
    /// Returns `true` if the client is healthy and ready to send requests.
    /// This is a lightweight check that doesn't perform actual network I/O.
    ///
    /// Implementations should return `false` if:
    /// - Connection pool is exhausted
    /// - Underlying transport is disconnected
    /// - Client has been explicitly closed
    fn is_healthy(&self) -> bool;

    /// Get client statistics (optional)
    ///
    /// Returns runtime statistics about the client for monitoring and debugging.
    /// Default implementation returns empty statistics.
    fn stats(&self) -> ClientStats {
        ClientStats::default()
    }

    /// Close the client gracefully (optional)
    ///
    /// Implementations should:
    /// - Close all active connections
    /// - Wait for in-flight requests to complete (or timeout)
    /// - Release all resources
    ///
    /// Default implementation does nothing.
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Client runtime statistics
///
/// Used for monitoring and debugging transport client performance.
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    /// Total number of requests sent
    pub requests_sent: u64,

    /// Total number of successful responses
    pub responses_received: u64,

    /// Total number of errors
    pub errors: u64,

    /// Total bytes sent
    pub bytes_sent: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Number of active connections (for connection-pooled transports)
    pub active_connections: usize,

    /// Number of idle connections in pool
    pub idle_connections: usize,

    /// Average request latency in microseconds (0 if not available)
    pub avg_latency_us: u64,
}

impl ClientStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if statistics are available (non-zero)
    pub fn is_available(&self) -> bool {
        self.requests_sent > 0 || self.active_connections > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_stats_default() {
        let stats = ClientStats::default();
        assert_eq!(stats.requests_sent, 0);
        assert_eq!(stats.responses_received, 0);
        assert!(!stats.is_available());
    }

    #[test]
    fn test_client_stats_is_available() {
        let mut stats = ClientStats::default();
        assert!(!stats.is_available());

        stats.requests_sent = 1;
        assert!(stats.is_available());

        let stats2 = ClientStats {
            active_connections: 1,
            ..Default::default()
        };
        assert!(stats2.is_available());
    }
}
