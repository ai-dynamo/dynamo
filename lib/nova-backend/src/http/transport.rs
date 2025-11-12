// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP transport with fire-and-forget semantics
//!
//! This implementation provides a simple HTTP transport where:
//! - Header bytes are base64-encoded in X-Transport-Header HTTP header
//! - Payload bytes are sent as raw bytes in HTTP body
//! - Three separate routes handle Message, Response, and Event types
//! - All sends are fire-and-forget (202 Accepted response)

use anyhow::Result;
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::transport::{HealthCheckError, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

use super::server::HttpServer;

/// HTTP transport with fire-and-forget messaging
///
/// This transport uses HTTP POST requests for one-way messaging.
/// Header bytes are base64-encoded in HTTP headers, payload in body.
pub struct HttpTransport {
    // Identity
    key: TransportKey,
    bind_addr: SocketAddr,
    local_address: OnceLock<WorkerAddress>,

    // Peer registry (instance_id → base URL)
    peers: Arc<DashMap<dynamo_identity::InstanceId, String>>,

    // HTTP client for sending requests
    client: reqwest::Client,

    // Runtime handle for spawning tasks
    runtime: OnceLock<tokio::runtime::Handle>,

    // Server handle (set during start)
    server_handle: Arc<parking_lot::Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>>,

    // Shutdown coordination
    cancel_token: CancellationToken,
}

impl HttpTransport {
    /// Create a new HTTP transport
    pub fn new(bind_addr: SocketAddr, key: TransportKey) -> Self {
        // Create HTTP client with reasonable timeouts and rustls
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            key,
            bind_addr,
            local_address: OnceLock::new(),
            peers: Arc::new(DashMap::new()),
            client,
            runtime: OnceLock::new(),
            server_handle: Arc::new(parking_lot::Mutex::new(None)),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Send HTTP POST request to peer
    async fn send_http_request(
        client: reqwest::Client,
        url: String,
        header: Bytes,
        payload: Bytes,
        on_error: Arc<dyn TransportErrorHandler>,
    ) {
        // Base64 encode header for HTTP header
        let header_b64 = base64::engine::general_purpose::STANDARD.encode(&header);

        // Send HTTP POST
        let result = client
            .post(&url)
            .header("X-Transport-Header", header_b64)
            .header("Content-Type", "application/octet-stream")
            .body(payload.to_vec())
            .send()
            .await;

        match result {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    on_error.on_error(
                        header,
                        payload,
                        format!("HTTP request failed with status: {}", status),
                    );
                }
            }
            Err(e) => {
                on_error.on_error(header, payload, format!("HTTP request failed: {}", e));
            }
        }
    }
}

impl Transport for HttpTransport {
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

        // Parse HTTP URL (expected format: "http://host:port" or "https://host:port")
        let url = String::from_utf8(endpoint.to_vec()).map_err(|e| {
            error!("Failed to parse HTTP endpoint as UTF-8: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Validate URL format
        if !url.starts_with("http://") && !url.starts_with("https://") {
            error!("Invalid HTTP URL format: {}", url);
            return Err(TransportError::InvalidEndpoint);
        }

        // Store peer URL (without trailing slash)
        let base_url = url.trim_end_matches('/').to_string();
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
        let base_url = match self.peers.get(&instance_id) {
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

        // Determine route based on message type
        let path = match message_type {
            MessageType::Message => "/message",
            MessageType::Response => "/response",
            MessageType::Ack | MessageType::Event => "/event",
        };

        let url = format!("{}{}", base_url, path);

        // Spawn async task to send HTTP request using stored runtime handle
        let client = self.client.clone();
        if let Some(rt) = self.runtime.get() {
            rt.spawn(async move {
                Self::send_http_request(client, url, header, payload, on_error).await;
            });
        } else {
            // Fallback if runtime not set (shouldn't happen)
            tokio::spawn(async move {
                Self::send_http_request(client, url, header, payload, on_error).await;
            });
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
            info!("Starting HTTP transport on {}", bind_addr);

            // Use std TcpListener to bind synchronously
            let std_listener = std::net::TcpListener::bind(bind_addr)?;

            // Get actual bound address (important for port 0)
            let actual_addr = std_listener.local_addr()?;

            // Set non-blocking for tokio
            std_listener.set_nonblocking(true)?;

            info!("HTTP server bound to {}", actual_addr);

            // Build WorkerAddress with actual address
            let local_url = format!("http://{}", actual_addr);
            let mut addr_builder = WorkerAddress::builder();
            addr_builder.add_entry(key, local_url.as_bytes().to_vec())?;
            let local_address = addr_builder.build()?;

            // Store runtime handle and local address
            self.runtime.set(rt.clone()).ok();
            self.local_address.set(local_address).ok();

            // Convert to tokio listener and start server using provided runtime
            let server_handle = rt.spawn(async move {
                // Convert std listener to tokio listener
                let tokio_listener = tokio::net::TcpListener::from_std(std_listener)
                    .map_err(|e| anyhow::anyhow!("Failed to convert listener: {}", e))?;

                let server = HttpServer::new(tokio_listener, channels, cancel_token);

                if let Err(e) = server.run().await {
                    error!("HTTP server error: {}", e);
                }
                Ok(())
            });

            *server_handle_arc.lock() = Some(server_handle);

            info!("HTTP transport started");

            Ok(())
        })
    }

    fn shutdown(&self) {
        info!("Shutting down HTTP transport");

        // Cancel the server
        self.cancel_token.cancel();

        // Wait for server to stop
        if let Some(handle) = self.server_handle.lock().take() {
            // Give server a moment to stop accepting requests
            std::thread::sleep(std::time::Duration::from_millis(100));
            handle.abort();
        }

        // Clear peers
        self.peers.clear();

        info!("HTTP transport shut down");
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

            let health_url = format!("{}/health", url);

            match tokio::time::timeout(timeout, self.client.head(&health_url).send()).await {
                Ok(Ok(resp)) if resp.status().is_success() => Ok(()),
                Ok(Ok(_)) | Ok(Err(_)) => Err(HealthCheckError::ConnectionFailed),
                Err(_) => Err(HealthCheckError::Timeout),
            }
        })
    }
}

/// Builder for HttpTransport
pub struct HttpTransportBuilder {
    bind_addr: Option<SocketAddr>,
    key: Option<TransportKey>,
}

impl HttpTransportBuilder {
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

    /// Build the HttpTransport
    pub fn build(self) -> Result<HttpTransport> {
        // Default to OS-provided port
        let bind_addr = self
            .bind_addr
            .unwrap_or_else(|| "0.0.0.0:0".parse().unwrap());

        let key = self.key.unwrap_or_else(|| TransportKey::from("http"));

        Ok(HttpTransport::new(bind_addr, key))
    }
}

impl Default for HttpTransportBuilder {
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
        let result = HttpTransportBuilder::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_bind_addr() {
        let bind_addr = "127.0.0.1:8080".parse().unwrap();
        let result = HttpTransportBuilder::new().bind_addr(bind_addr).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_key() {
        let result = HttpTransportBuilder::new()
            .key(TransportKey::from("my-http"))
            .build();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().key().as_str(), "my-http");
    }
}
