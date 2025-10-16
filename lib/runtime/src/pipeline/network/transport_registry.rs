// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport Registry and Abstraction Layer
//!
//! This module provides a unified interface for registering and managing different
//! transport mechanisms (HTTP, TCP, NATS, etc.) for both request and response planes.
//!
//! See TRANSPORT_REGISTRY_DESIGN.md for full architectural details.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    request_plane::{Headers, RequestPlaneClient, RequestPlaneServer},
    AsyncEngineContext, ControlMessage, StreamOptions,
};

// ============================================================================
// Transport Identification and Capabilities
// ============================================================================

/// Identifies a transport mechanism
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportId {
    Nats,
    Http,
    Tcp,
    Grpc,
    // Future: Quic, WebSocket, etc.
}

/// Identifies request vs response plane
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaneType {
    Request,
    Response,
}

/// Transport capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportCapabilities {
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports persistent connections
    pub persistent_connections: bool,
    /// Supports bidirectional communication
    pub bidirectional: bool,
    /// Maximum message size (None = unlimited)
    pub max_message_size: Option<usize>,
}

impl TransportCapabilities {
    /// Check if this transport satisfies the required capabilities
    pub fn satisfies(&self, required: &TransportCapabilities) -> bool {
        (!required.streaming || self.streaming)
            && (!required.persistent_connections || self.persistent_connections)
            && (!required.bidirectional || self.bidirectional)
            && match (self.max_message_size, required.max_message_size) {
                (Some(actual), Some(required)) => actual >= required,
                (None, _) => true,
                (Some(_), None) => true,
            }
    }
}

/// Transport registration information
pub struct TransportRegistration {
    pub transport_id: TransportId,
    pub plane_type: PlaneType,
    pub capabilities: TransportCapabilities,
    pub priority: u8, // Lower = higher priority
}

// ============================================================================
// Transport Addressing
// ============================================================================

/// Unified address type for different transports
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TransportAddress {
    Http {
        url: String,
    },
    Tcp {
        host: String,
        port: u16,
    },
    Nats {
        subject: String,
    },
    Grpc {
        endpoint: String,
    },
}

impl TransportAddress {
    pub fn transport_id(&self) -> TransportId {
        match self {
            Self::Http { .. } => TransportId::Http,
            Self::Tcp { .. } => TransportId::Tcp,
            Self::Nats { .. } => TransportId::Nats,
            Self::Grpc { .. } => TransportId::Grpc,
        }
    }
}

// ============================================================================
// Response Plane Abstraction
// ============================================================================

/// Connection information for response streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseConnectionInfo {
    /// Transport-specific address information
    pub address: TransportAddress,
    /// Optional context for multiplexing
    pub context_id: String,
    /// Optional subject/topic for routing
    pub subject: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Response plane client - sends responses back to requester
#[async_trait]
pub trait ResponsePlaneClient: Send + Sync {
    /// Create or reuse connection to send response stream
    async fn create_response_stream(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ResponseConnectionInfo,
    ) -> Result<Box<dyn ResponseStreamWriter>>;

    /// Get transport capabilities
    fn capabilities(&self) -> &TransportCapabilities;

    /// Release/return connection to pool
    async fn release_connection(&self, connection_info: &ResponseConnectionInfo) -> Result<()>;

    /// Health check
    async fn is_healthy(&self) -> bool;
}

/// Response plane server - receives response streams
#[async_trait]
pub trait ResponsePlaneServer: Send + Sync {
    /// Register a new response stream endpoint
    async fn register_stream(
        &self,
        options: StreamOptions,
    ) -> Result<PendingResponseStream>;

    /// Start the response plane server
    async fn start(&mut self, bind_address: String) -> Result<()>;

    /// Stop the response plane server gracefully
    async fn stop(&mut self) -> Result<()>;

    /// Get the public address where this server is accessible
    fn public_address(&self) -> String;

    /// Get transport capabilities
    fn capabilities(&self) -> &TransportCapabilities;
}

/// Unified response stream writer interface
#[async_trait]
pub trait ResponseStreamWriter: Send + Sync {
    /// Send data on the response stream
    async fn send(&mut self, data: Bytes) -> Result<()>;

    /// Send control message
    async fn send_control(&mut self, control: ControlMessage) -> Result<()>;

    /// Flush any buffered data
    async fn flush(&mut self) -> Result<()>;

    /// Close the stream gracefully
    async fn close(&mut self) -> Result<()>;

    /// Get the transport ID
    fn transport_id(&self) -> TransportId;
}

/// Pending response stream from registration
pub struct PendingResponseStream {
    pub connection_info: ResponseConnectionInfo,
    pub stream_receiver: tokio::sync::oneshot::Receiver<Result<Box<dyn ResponseStreamReader>>>,
}

/// Response stream reader (router-side)
#[async_trait]
pub trait ResponseStreamReader: Send {
    /// Receive next chunk of data
    async fn recv(&mut self) -> Option<Result<Bytes>>;

    /// Get the transport ID
    fn transport_id(&self) -> TransportId;
}

// ============================================================================
// Transport Registry
// ============================================================================

/// Central registry for all transport implementations
pub struct TransportRegistry {
    request_clients: RwLock<HashMap<TransportId, Arc<dyn RequestPlaneClient>>>,
    request_servers: RwLock<HashMap<TransportId, Arc<dyn RequestPlaneServer>>>,
    response_clients: RwLock<HashMap<TransportId, Arc<dyn ResponsePlaneClient>>>,
    response_servers: RwLock<HashMap<TransportId, Arc<dyn ResponsePlaneServer>>>,

    // Capabilities for each transport
    request_capabilities: RwLock<HashMap<TransportId, TransportCapabilities>>,
    response_capabilities: RwLock<HashMap<TransportId, TransportCapabilities>>,

    // Priority ordering for transport selection
    request_priorities: RwLock<Vec<(TransportId, u8)>>,
    response_priorities: RwLock<Vec<(TransportId, u8)>>,
}

impl TransportRegistry {
    pub fn new() -> Self {
        Self {
            request_clients: RwLock::new(HashMap::new()),
            request_servers: RwLock::new(HashMap::new()),
            response_clients: RwLock::new(HashMap::new()),
            response_servers: RwLock::new(HashMap::new()),
            request_capabilities: RwLock::new(HashMap::new()),
            response_capabilities: RwLock::new(HashMap::new()),
            request_priorities: RwLock::new(Vec::new()),
            response_priorities: RwLock::new(Vec::new()),
        }
    }

    /// Register request plane transport
    pub async fn register_request_transport(
        &self,
        registration: TransportRegistration,
        client: Arc<dyn RequestPlaneClient>,
        server: Arc<dyn RequestPlaneServer>,
    ) -> Result<()> {
        let id = registration.transport_id.clone();

        self.request_clients.write().await.insert(id.clone(), client);
        self.request_servers.write().await.insert(id.clone(), server);
        self.request_capabilities.write().await.insert(id.clone(), registration.capabilities);

        // Update priority list
        let mut priorities = self.request_priorities.write().await;
        priorities.push((id.clone(), registration.priority));
        priorities.sort_by_key(|(_, p)| *p);

        tracing::info!(
            transport = ?id,
            priority = registration.priority,
            "Registered request plane transport"
        );

        Ok(())
    }

    /// Register response plane transport
    pub async fn register_response_transport(
        &self,
        registration: TransportRegistration,
        client: Arc<dyn ResponsePlaneClient>,
        server: Arc<dyn ResponsePlaneServer>,
    ) -> Result<()> {
        let id = registration.transport_id.clone();

        self.response_clients
            .write()
            .await
            .insert(id.clone(), client);
        self.response_servers
            .write()
            .await
            .insert(id.clone(), server);
        self.response_capabilities.write().await.insert(id.clone(), registration.capabilities);

        // Update priority list
        let mut priorities = self.response_priorities.write().await;
        priorities.push((id.clone(), registration.priority));
        priorities.sort_by_key(|(_, p)| *p);

        tracing::info!(
            transport = ?id,
            priority = registration.priority,
            "Registered response plane transport"
        );

        Ok(())
    }

    /// Select best transport for given requirements
    pub async fn select_transport(
        &self,
        plane: PlaneType,
        required_capabilities: &TransportCapabilities,
    ) -> Result<TransportId> {
        match plane {
            PlaneType::Request => {
                let priorities = self.request_priorities.read().await;
                let capabilities = self.request_capabilities.read().await;

                // Find first transport that satisfies capabilities
                for (transport_id, _) in priorities.iter() {
                    if let Some(caps) = capabilities.get(transport_id) {
                        if caps.satisfies(required_capabilities) {
                            return Ok(transport_id.clone());
                        }
                    }
                }
            }
            PlaneType::Response => {
                let priorities = self.response_priorities.read().await;
                let capabilities = self.response_capabilities.read().await;

                // Find first transport that satisfies capabilities
                for (transport_id, _) in priorities.iter() {
                    if let Some(caps) = capabilities.get(transport_id) {
                        if caps.satisfies(required_capabilities) {
                            return Ok(transport_id.clone());
                        }
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "No transport available for required capabilities"
        ))
    }

    /// Get request client implementation
    pub async fn get_request_client(
        &self,
        id: &TransportId,
    ) -> Option<Arc<dyn RequestPlaneClient>> {
        self.request_clients.read().await.get(id).cloned()
    }

    /// Get response client implementation
    pub async fn get_response_client(
        &self,
        id: &TransportId,
    ) -> Option<Arc<dyn ResponsePlaneClient>> {
        self.response_clients.read().await.get(id).cloned()
    }

    /// Get request server implementation
    pub async fn get_request_server(
        &self,
        id: &TransportId,
    ) -> Option<Arc<dyn RequestPlaneServer>> {
        self.request_servers.read().await.get(id).cloned()
    }

    /// Get response server implementation
    pub async fn get_response_server(
        &self,
        id: &TransportId,
    ) -> Option<Arc<dyn ResponsePlaneServer>> {
        self.response_servers.read().await.get(id).cloned()
    }

    /// List all registered transports
    pub async fn list_transports(&self, plane: PlaneType) -> Vec<(TransportId, u8)> {
        match plane {
            PlaneType::Request => self.request_priorities.read().await.clone(),
            PlaneType::Response => self.response_priorities.read().await.clone(),
        }
    }
}

impl Default for TransportRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Configuration Enums
// ============================================================================

/// Response plane mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponsePlaneMode {
    /// TCP response streaming (current default)
    Tcp,
    /// HTTP response streaming (Priority 1)
    Http,
}

impl ResponsePlaneMode {
    pub fn from_env() -> Self {
        std::env::var("DYN_RESPONSE_PLANE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "http" | "http2" => Some(Self::Http),
                "tcp" => Some(Self::Tcp),
                _ => None,
            })
            .unwrap_or(Self::Tcp)
    }

    pub fn is_tcp(&self) -> bool {
        matches!(self, Self::Tcp)
    }

    pub fn is_http(&self) -> bool {
        matches!(self, Self::Http)
    }
}

impl From<ResponsePlaneMode> for TransportId {
    fn from(mode: ResponsePlaneMode) -> Self {
        match mode {
            ResponsePlaneMode::Tcp => TransportId::Tcp,
            ResponsePlaneMode::Http => TransportId::Http,
        }
    }
}

impl Default for ResponsePlaneMode {
    fn default() -> Self {
        Self::Tcp
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get HTTP response server host from environment
pub fn get_http_response_host_from_env() -> String {
    std::env::var("DYN_HTTP_RESPONSE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string())
}

/// Get HTTP response server port from environment
pub fn get_http_response_port_from_env() -> u16 {
    std::env::var("DYN_HTTP_RESPONSE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8082)
}

/// Get HTTP response root path from environment
pub fn get_http_response_root_path_from_env() -> String {
    std::env::var("DYN_HTTP_RESPONSE_ROOT_PATH")
        .unwrap_or_else(|_| "/v1/rpc/response".to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_capabilities_satisfies() {
        let actual = TransportCapabilities {
            streaming: true,
            persistent_connections: true,
            bidirectional: false,
            max_message_size: Some(1024 * 1024),
        };

        let required_ok = TransportCapabilities {
            streaming: true,
            persistent_connections: false,
            bidirectional: false,
            max_message_size: Some(512 * 1024),
        };

        assert!(actual.satisfies(&required_ok));

        let required_fail = TransportCapabilities {
            streaming: true,
            persistent_connections: true,
            bidirectional: true, // actual doesn't support this
            max_message_size: Some(512 * 1024),
        };

        assert!(!actual.satisfies(&required_fail));
    }

    #[test]
    fn test_transport_address_id() {
        let http = TransportAddress::Http {
            url: "http://localhost:8080".to_string(),
        };
        assert_eq!(http.transport_id(), TransportId::Http);

        let tcp = TransportAddress::Tcp {
            host: "localhost".to_string(),
            port: 9090,
        };
        assert_eq!(tcp.transport_id(), TransportId::Tcp);
    }

    #[test]
    fn test_response_plane_mode_from_env() {
        // Test default
        std::env::remove_var("DYN_RESPONSE_PLANE");
        assert_eq!(ResponsePlaneMode::from_env(), ResponsePlaneMode::Tcp);

        // Test http
        std::env::set_var("DYN_RESPONSE_PLANE", "http");
        assert_eq!(ResponsePlaneMode::from_env(), ResponsePlaneMode::Http);

        // Test tcp
        std::env::set_var("DYN_RESPONSE_PLANE", "tcp");
        assert_eq!(ResponsePlaneMode::from_env(), ResponsePlaneMode::Tcp);
    }

    #[tokio::test]
    async fn test_transport_registry() {
        let registry = TransportRegistry::new();

        // Initially empty
        assert_eq!(registry.list_transports(PlaneType::Request).await.len(), 0);
        assert_eq!(
            registry.list_transports(PlaneType::Response).await.len(),
            0
        );

        // Priority ordering is maintained
        let priorities = registry.list_transports(PlaneType::Request).await;
        for i in 1..priorities.len() {
            assert!(priorities[i - 1].1 <= priorities[i].1);
        }
    }
}

