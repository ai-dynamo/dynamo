// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport with bidirectional streaming
//!
//! This implementation provides a gRPC transport where:
//! - Messages are pre-framed using TCP frame format (11-byte header + data)
//! - Each peer connection uses a bidirectional gRPC stream
//! - Streams are used unidirectionally (client sends, server receives)
//! - Simple retry logic with exponential backoff

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::transport::{HealthCheckError, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

use super::client::ConnectionHandle;
use super::server::GrpcServer;

/// gRPC transport with bidirectional streaming
///
/// This transport uses gRPC streams for message exchange without protobuf files.
/// Messages are pre-framed using the TCP frame format for compatibility.
pub struct GrpcTransport {
    // Identity
    key: TransportKey,
    bind_addr: SocketAddr,
    local_address: OnceLock<WorkerAddress>,

    // Peer registry (instance_id → gRPC URL)
    peers: Arc<DashMap<dynamo_identity::InstanceId, String>>,

    // Active connections (instance_id → connection handle)
    connections: Arc<DashMap<dynamo_identity::InstanceId, ConnectionHandle>>,

    // Runtime handle for spawning tasks
    runtime: OnceLock<tokio::runtime::Handle>,

    // Channels for routing received messages
    channels: OnceLock<TransportAdapter>,

    // Server handle (set during start)
    server_handle: Arc<parking_lot::Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>>,

    // Shutdown coordination
    cancel_token: CancellationToken,
}

impl GrpcTransport {
    /// Create a new gRPC transport
    pub fn new(bind_addr: SocketAddr, key: TransportKey) -> Self {
        Self {
            key,
            bind_addr,
            local_address: OnceLock::new(),
            peers: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            runtime: OnceLock::new(),
            channels: OnceLock::new(),
            server_handle: Arc::new(parking_lot::Mutex::new(None)),
            cancel_token: CancellationToken::new(),
        }
    }
}

impl Transport for GrpcTransport {
    fn key(&self) -> TransportKey {
        self.key.clone()
    }

    fn address(&self) -> WorkerAddress {
        self.local_address
            .get()
            .cloned()
            .unwrap_or_else(|| WorkerAddress::builder().build().unwrap())
    }

    fn register(&self, peer_info: PeerInfo) -> Result<(), TransportError> {
        // Get endpoint from peer's address
        let endpoint = peer_info
            .worker_address()
            .get_entry(&self.key)
            .map_err(|_| TransportError::NoEndpoint)?
            .ok_or(TransportError::NoEndpoint)?;

        // Parse gRPC URL (expected format: "http://host:port" or "grpc://host:port")
        let url = String::from_utf8(endpoint.to_vec()).map_err(|e| {
            error!("Failed to parse gRPC endpoint as UTF-8: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Validate URL format
        if !url.starts_with("http://")
            && !url.starts_with("https://")
            && !url.starts_with("grpc://")
        {
            error!("Invalid gRPC URL format: {}", url);
            return Err(TransportError::InvalidEndpoint);
        }

        // Normalize grpc:// to http:// for tonic
        let normalized_url = if url.starts_with("grpc://") {
            url.replace("grpc://", "http://")
        } else {
            url
        };

        // Store peer URL (without trailing slash)
        let base_url = normalized_url.trim_end_matches('/').to_string();
        self.peers.insert(peer_info.instance_id(), base_url.clone());

        debug!(
            "Registered peer {} at {}",
            peer_info.instance_id(),
            base_url
        );

        Ok(())
    }

    #[inline]
    fn send_message(
        &self,
        instance_id: dynamo_identity::InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: std::sync::Arc<dyn TransportErrorHandler>,
    ) {
        // Convert to Bytes
        let header = Bytes::from(header);
        let payload = Bytes::from(payload);

        // Look up peer URL
        let url = match self.peers.get(&instance_id) {
            Some(url) => url.clone(),
            None => {
                on_error.on_error(
                    header,
                    payload,
                    format!("peer not registered: {}", instance_id),
                );
                return;
            }
        };

        // Try to get existing connection first (fast path)
        if let Some(handle) = self.connections.get(&instance_id)
            && handle
                .try_send(message_type, header.clone(), payload.clone())
                .is_ok()
        {
            return; // Success on fast path
        }
        // Connection might be dead, fall through to slow path

        // Slow path: establish new connection
        let connections = self.connections.clone();
        let cancel_token = self.cancel_token.clone();

        if let Some(rt) = self.runtime.get() {
            rt.spawn(async move {
                match super::client::establish_connection(url, instance_id, cancel_token).await {
                    Ok(handle) => {
                        connections.insert(instance_id, handle.clone());
                        if let Err(e) = handle
                            .send(message_type, header.clone(), payload.clone())
                            .await
                        {
                            error!("Failed to send on new connection: {}", e);
                            on_error.on_error(header, payload, format!("send failed: {}", e));
                        }
                    }
                    Err(e) => {
                        error!("Failed to establish connection: {}", e);
                        on_error.on_error(header, payload, format!("connection failed: {}", e));
                    }
                }
            });
        } else {
            // Fallback if runtime not set (shouldn't happen)
            on_error.on_error(header, payload, "runtime not initialized".to_string());
        }
    }

    fn start(
        &self,
        _instance_id: dynamo_identity::InstanceId,
        channels: TransportAdapter,
        rt: tokio::runtime::Handle,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
        let bind_addr = self.bind_addr;
        let key = self.key.clone();
        let cancel_token = self.cancel_token.clone();
        let server_handle_arc = self.server_handle.clone();

        Box::pin(async move {
            info!("Starting gRPC transport on {}", bind_addr);

            // Bind listener to get actual address (important for port 0)
            let listener = tokio::net::TcpListener::bind(bind_addr).await?;
            let actual_addr = listener.local_addr()?;

            info!("gRPC server bound to {}", actual_addr);

            // Store runtime handle and channels
            self.runtime.set(rt.clone()).ok();
            self.channels.set(channels.clone()).ok();

            // Build WorkerAddress with actual bound address
            // Replace 0.0.0.0 with 127.0.0.1 for local communication
            let advertise_addr = if actual_addr.ip().is_unspecified() {
                std::net::SocketAddr::new(
                    std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
                    actual_addr.port(),
                )
            } else {
                actual_addr
            };
            let local_url = format!("grpc://{}", advertise_addr);
            let mut addr_builder = WorkerAddress::builder();
            addr_builder.add_entry(key, local_url.as_bytes().to_vec())?;
            let local_address = addr_builder.build()?;

            // Store local address
            self.local_address.set(local_address).ok();

            // Start gRPC server using provided runtime, passing the listener
            let server_handle = rt.spawn(async move {
                let server = GrpcServer::with_listener(listener, channels, cancel_token);
                server.run().await
            });

            *server_handle_arc.lock() = Some(server_handle);

            info!("gRPC transport started on {}", actual_addr);

            Ok(())
        })
    }

    fn shutdown(&self) {
        info!("Shutting down gRPC transport");

        // Cancel the server
        self.cancel_token.cancel();

        // Wait for server to stop
        if let Some(handle) = self.server_handle.lock().take() {
            // Give server a moment to stop accepting requests
            std::thread::sleep(std::time::Duration::from_millis(100));
            handle.abort();
        }

        // Close all connections
        self.connections.clear();

        // Clear peers
        self.peers.clear();

        info!("gRPC transport shut down");
    }

    fn check_health(
        &self,
        instance_id: dynamo_identity::InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    > {
        Box::pin(async move {
            let url = self
                .peers
                .get(&instance_id)
                .ok_or(HealthCheckError::PeerNotRegistered)?
                .clone();

            // Simple health check: try to establish connection
            match tokio::time::timeout(
                timeout,
                super::client::establish_connection(url, instance_id, self.cancel_token.clone()),
            )
            .await
            {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(_)) => Err(HealthCheckError::ConnectionFailed),
                Err(_) => Err(HealthCheckError::Timeout),
            }
        })
    }
}

/// Builder for GrpcTransport
pub struct GrpcTransportBuilder {
    bind_addr: Option<SocketAddr>,
    key: Option<TransportKey>,
}

impl GrpcTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            bind_addr: None,
            key: None,
        }
    }

    /// Set the bind address
    ///
    /// If not set, defaults to 0.0.0.0:0 (OS-provided port)
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }

    /// Set the transport key
    pub fn key(mut self, key: TransportKey) -> Self {
        self.key = Some(key);
        self
    }

    /// Build the GrpcTransport
    pub fn build(self) -> Result<GrpcTransport> {
        // Default to OS-provided port
        let bind_addr = self
            .bind_addr
            .unwrap_or_else(|| "0.0.0.0:0".parse().unwrap());

        let key = self.key.unwrap_or_else(|| TransportKey::from("grpc"));

        Ok(GrpcTransport::new(bind_addr, key))
    }
}

impl Default for GrpcTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        // Should work without bind_addr (defaults to OS-provided port)
        let result = GrpcTransportBuilder::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_bind_addr() {
        let bind_addr = "127.0.0.1:50051".parse().unwrap();
        let result = GrpcTransportBuilder::new().bind_addr(bind_addr).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_key() {
        let result = GrpcTransportBuilder::new()
            .key(TransportKey::from("my-grpc"))
            .build();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().key().as_str(), "my-grpc");
    }
}
