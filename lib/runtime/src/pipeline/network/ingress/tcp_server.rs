// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Request Plane Server
//!
//! Implements a TCP server for receiving requests over raw TCP connections.

use super::unified_server::{InstanceInfo, RequestPlaneServer, ServerStats};
use super::*;
use anyhow::Result;
use async_trait::async_trait;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// TCP request plane server
pub struct TcpRequestServer {
    cancellation_token: CancellationToken,
    stats: Arc<TcpServerStats>,
}

struct TcpServerStats {
    requests_received: AtomicU64,
    requests_handled: AtomicU64,
    errors: AtomicU64,
    bytes_received: AtomicU64,
    bytes_sent: AtomicU64,
    inflight: AtomicU64,
}

impl TcpRequestServer {
    /// Create a new TCP request server
    pub fn new(cancellation_token: CancellationToken) -> Self {
        Self {
            cancellation_token,
            stats: Arc::new(TcpServerStats {
                requests_received: AtomicU64::new(0),
                requests_handled: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                bytes_received: AtomicU64::new(0),
                bytes_sent: AtomicU64::new(0),
                inflight: AtomicU64::new(0),
            }),
        }
    }

    /// Handle a single TCP connection
    async fn handle_connection(
        stream: TcpStream,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: Arc<InstanceInfo>,
        stats: Arc<TcpServerStats>,
        notify: Arc<Notify>,
    ) -> Result<()> {
        let mut stream = stream;

        loop {
            // Read request length
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Connection closed
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }

            let len = u32::from_be_bytes(len_buf) as usize;

            // Sanity check
            if len > 16 * 1024 * 1024 {
                anyhow::bail!("Request too large: {} bytes", len);
            }

            // Read request payload
            let mut payload = vec![0u8; len];
            stream.read_exact(&mut payload).await?;

            stats.requests_received.fetch_add(1, Ordering::Relaxed);
            stats
                .bytes_received
                .fetch_add(len as u64, Ordering::Relaxed);
            stats.inflight.fetch_add(1, Ordering::SeqCst);

            // Send acknowledgment immediately (empty response for now)
            let ack = b"";
            let ack_len = ack.len() as u32;
            stream.write_all(&ack_len.to_be_bytes()).await?;
            stream.write_all(ack).await?;
            stream.flush().await?;

            stats
                .bytes_sent
                .fetch_add(4 + ack.len() as u64, Ordering::Relaxed);

            // Process request asynchronously
            let handler = handler.clone();
            let stats = stats.clone();
            let notify = notify.clone();
            let instance_id = instance_info.instance_id;
            let namespace = instance_info.namespace.clone();
            let component_name = instance_info.component_name.clone();
            let endpoint_name = instance_info.endpoint_name.clone();

            tokio::spawn(async move {
                tracing::trace!(instance_id, "handling TCP request");

                let result = handler
                    .handle_payload(payload.into())
                    .instrument(tracing::info_span!(
                        "handle_payload",
                        component = component_name.as_str(),
                        endpoint = endpoint_name.as_str(),
                        namespace = namespace.as_str(),
                        instance_id = instance_id,
                    ))
                    .await;

                match result {
                    Ok(_) => {
                        tracing::trace!(instance_id, "TCP request handled successfully");
                        stats.requests_handled.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to handle TCP request: {}", e);
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }

                stats.inflight.fetch_sub(1, Ordering::SeqCst);
                notify.notify_one();
            });
        }

        Ok(())
    }
}

#[async_trait]
impl RequestPlaneServer for TcpRequestServer {
    async fn start(
        &self,
        bind_addr: SocketAddr,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: InstanceInfo,
    ) -> Result<()> {
        let instance_info = Arc::new(instance_info);
        let notify = Arc::new(Notify::new());

        // Set initial health status
        instance_info
            .system_health
            .lock()
            .set_endpoint_health_status(
                &instance_info.endpoint_name,
                crate::config::HealthStatus::Ready,
            );

        tracing::info!("Starting TCP request server on {}", bind_addr);

        let listener = TcpListener::bind(bind_addr).await?;
        let stats = self.stats.clone();
        let cancellation_token = self.cancellation_token.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            tracing::trace!("Accepted TCP connection from {}", peer_addr);

                            let handler = handler.clone();
                            let instance_info = instance_info.clone();
                            let stats = stats.clone();
                            let notify = notify.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(
                                    stream,
                                    handler,
                                    instance_info,
                                    stats,
                                    notify,
                                ).await {
                                    tracing::debug!("TCP connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept TCP connection: {}", e);
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    tracing::info!("TCP request server received cancellation signal");
                    break;
                }
            }
        }

        // Mark as not ready
        instance_info
            .system_health
            .lock()
            .set_endpoint_health_status(
                &instance_info.endpoint_name,
                crate::config::HealthStatus::NotReady,
            );

        // Wait for inflight requests
        tracing::info!(
            "Waiting for {} inflight requests to complete",
            self.stats.inflight.load(Ordering::SeqCst)
        );

        while self.stats.inflight.load(Ordering::SeqCst) > 0 {
            notify.notified().await;
        }

        tracing::info!("All inflight requests completed");

        Ok(())
    }

    async fn register_endpoint(
        &self,
        _subject: String,
        _handler: Arc<dyn PushWorkHandler>,
        _instance_info: InstanceInfo,
    ) -> Result<()> {
        // TCP server doesn't support multiplexing
        // Each server instance handles one endpoint
        anyhow::bail!(
            "TCP transport does not support endpoint multiplexing. Use HTTP transport for multiple endpoints on one port."
        )
    }

    async fn unregister_endpoint(&self, _subject: &str, _endpoint_name: &str) -> Result<()> {
        // Not supported for TCP
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.cancellation_token.cancel();
        Ok(())
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }

    fn stats(&self) -> ServerStats {
        ServerStats {
            requests_received: self.stats.requests_received.load(Ordering::Relaxed),
            requests_handled: self.stats.requests_handled.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(Ordering::Relaxed),
            active_connections: 0,
            inflight_requests: self.stats.inflight.load(Ordering::Relaxed) as usize,
            avg_processing_time_us: 0,
        }
    }

    fn supports_multiplexing(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_server_creation() {
        let token = CancellationToken::new();
        let server = TcpRequestServer::new(token);
        assert_eq!(server.transport_name(), "tcp");
        assert!(!server.supports_multiplexing());
    }
}
