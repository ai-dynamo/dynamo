// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-performance TCP transport with single-threaded optimizations
//!
//! This implementation uses Rc+RefCell+LocalSet for maximum performance on a single CPU core.
//! All operations run on the same thread as the TCP listener for optimal cache locality.

use anyhow::{Context, Result};
use bytes::Bytes;
use dashmap::DashMap;
use futures::SinkExt;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::Framed;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::transport::{HealthCheckError, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

use super::framing::TcpFrameCodec;
use super::listener::TcpListener;

/// High-performance TCP transport with lock-free concurrent access
///
/// This transport uses `DashMap` for lock-free concurrent access to connection state.
/// Tasks are spawned using `tokio::spawn` for compatibility with the `Transport` trait.
/// For single-threaded performance, run the entire transport in a `LocalSet` context.
pub struct TcpTransport {
    // Identity (immutable, no wrapper needed)
    key: TransportKey,
    bind_addr: SocketAddr,
    local_address: WorkerAddress,

    // Shared mutable state with DashMap (lock-free)
    peers: Arc<DashMap<dynamo_identity::InstanceId, SocketAddr>>,
    connections: Arc<DashMap<dynamo_identity::InstanceId, ConnectionHandle>>,

    // Runtime handle for spawning tasks
    runtime: OnceLock<tokio::runtime::Handle>,

    // Shutdown coordination
    cancel_token: CancellationToken,
}

/// Handle to a connection's writer task
#[derive(Clone)]
struct ConnectionHandle {
    tx: flume::Sender<SendTask>,
}

/// Task sent to writer task containing pre-encoded frame
struct SendTask {
    framed_bytes: Bytes,
    on_error: Arc<dyn TransportErrorHandler>,
    header: Bytes,
    payload: Bytes,
}

impl TcpTransport {
    /// Create a new TCP transport
    pub fn new(bind_addr: SocketAddr, key: TransportKey, local_address: WorkerAddress) -> Self {
        Self {
            key,
            bind_addr,
            local_address,
            peers: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            runtime: OnceLock::new(),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Optional: Pre-establish connection after registration
    ///
    /// This can be called after `register()` to eagerly establish the TCP connection
    /// instead of waiting for the first `send_message()` call.
    pub fn ensure_connected(&self, instance_id: dynamo_identity::InstanceId) -> Result<()> {
        self.get_or_create_connection(instance_id)?;
        Ok(())
    }

    /// Get or create a connection to a peer (lazy initialization)
    fn get_or_create_connection(
        &self,
        instance_id: dynamo_identity::InstanceId,
    ) -> Result<ConnectionHandle> {
        // Fast path: check if connection exists
        if let Some(handle) = self.connections.get(&instance_id) {
            return Ok(handle.clone());
        }

        // Slow path: create new connection
        let addr = *self
            .peers
            .get(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("peer not registered: {}", instance_id))?
            .value();

        // Create channel (bounded for backpressure)
        let (tx, rx) = flume::bounded(256);

        let handle = ConnectionHandle { tx };

        // Insert into connection map
        self.connections.insert(instance_id, handle.clone());

        // Spawn writer task using stored runtime handle
        let cancel = self.cancel_token.clone();
        if let Some(rt) = self.runtime.get() {
            rt.spawn(connection_writer_task(addr, rx, cancel));
        } else {
            // Fallback to tokio::spawn if runtime not yet set (shouldn't happen)
            tokio::spawn(connection_writer_task(addr, rx, cancel));
        }

        debug!("Created new connection to {} ({})", instance_id, addr);

        Ok(handle)
    }

    /// Slow path for send_message when fast path fails
    #[cold]
    fn send_slow_path(
        &self,
        instance_id: dynamo_identity::InstanceId,
        framed: Bytes,
        header: Bytes,
        payload: Bytes,
        on_error: Arc<dyn TransportErrorHandler>,
    ) {
        // Get or create connection
        let handle = match self.get_or_create_connection(instance_id) {
            Ok(h) => h,
            Err(e) => {
                error!("Failed to create connection: {}", e);
                on_error.on_error(header, payload, e.to_string());
                return;
            }
        };

        let task = SendTask {
            framed_bytes: framed,
            on_error,
            header: header.clone(),
            payload: payload.clone(),
        };

        // Spawn task to send (avoid blocking current task) using stored runtime handle
        if let Some(rt) = self.runtime.get() {
            rt.spawn(async move {
                match tokio::time::timeout(Duration::from_millis(100), handle.tx.send_async(task))
                    .await
                {
                    Ok(Ok(())) => { /* Success */ }
                    Ok(Err(_)) => {
                        // Channel closed - writer task exited
                        error!("Connection channel closed for {}", instance_id);
                    }
                    Err(_) => {
                        // Timeout - backpressure
                        warn!("Send timeout for {}", instance_id);
                    }
                }
            });
        } else {
            // Fallback to tokio::spawn if runtime not yet set (shouldn't happen)
            tokio::spawn(async move {
                match tokio::time::timeout(Duration::from_millis(100), handle.tx.send_async(task))
                    .await
                {
                    Ok(Ok(())) => { /* Success */ }
                    Ok(Err(_)) => {
                        // Channel closed - writer task exited
                        error!("Connection channel closed for {}", instance_id);
                    }
                    Err(_) => {
                        // Timeout - backpressure
                        warn!("Send timeout for {}", instance_id);
                    }
                }
            });
        }
    }
}

impl Transport for TcpTransport {
    fn key(&self) -> TransportKey {
        self.key.clone()
    }

    fn address(&self) -> WorkerAddress {
        self.local_address.clone()
    }

    fn register(&self, peer_info: PeerInfo) -> Result<(), TransportError> {
        // Get endpoint from peer's address
        let endpoint = peer_info
            .worker_address()
            .get_entry(&self.key)
            .map_err(|_| TransportError::NoEndpoint)?
            .ok_or(TransportError::NoEndpoint)?;

        // Parse TCP endpoint (expected format: "tcp://host:port" or "host:port")
        let addr = parse_tcp_endpoint(&endpoint).map_err(|e| {
            error!("Failed to parse TCP endpoint: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Store peer address
        self.peers.insert(peer_info.instance_id(), addr);

        debug!("Registered peer {} at {}", peer_info.instance_id(), addr);

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
        // Convert to Bytes (one allocation each)
        let header = Bytes::from(header);
        let payload = Bytes::from(payload);

        // Pre-encode frame (off critical path)
        let framed =
            match TcpFrameCodec::encode_frame(message_type, header.clone(), payload.clone()) {
                Ok(f) => f,
                Err(e) => {
                    // Early return on encoding error
                    on_error.on_error(header, payload, e.to_string());
                    return;
                }
            };

        // Fast path: connection exists
        let tx = self.connections.get(&instance_id).map(|h| h.tx.clone());

        if let Some(tx) = tx {
            let task = SendTask {
                framed_bytes: framed.clone(),
                on_error: on_error.clone(),
                header: header.clone(),
                payload: payload.clone(),
            };

            // Try non-blocking send
            if tx.try_send(task).is_ok() {
                return; // Fast path success
            }
            // Fall through to slow path if channel full
        }

        // Slow path: establish connection or handle full channel
        self.send_slow_path(instance_id, framed, header, payload, on_error);
    }

    fn start(
        &self,
        _instance_id: dynamo_identity::InstanceId,
        channels: TransportAdapter,
        rt: tokio::runtime::Handle,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
        // Store runtime handle for use in send_message
        self.runtime.set(rt.clone()).ok();

        let bind_addr = self.bind_addr;
        let cancel_token = self.cancel_token.clone();

        Box::pin(async move {
            // Create error handler that routes to the transport error handler
            struct DefaultErrorHandler;
            impl TransportErrorHandler for DefaultErrorHandler {
                fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
                    warn!("Transport error: {}", error);
                }
            }

            // Start TCP listener
            let listener = TcpListener::builder()
                .bind_addr(bind_addr)
                .adapter(channels)
                .error_handler(std::sync::Arc::new(DefaultErrorHandler))
                .cancel_token(cancel_token)
                .build()?;

            rt.spawn(async move {
                if let Err(e) = listener.serve().await {
                    error!("TCP listener error: {}", e);
                }
            });

            info!("TCP transport started on {}", bind_addr);

            Ok(())
        })
    }

    fn shutdown(&self) {
        info!("Shutting down TCP transport");
        self.cancel_token.cancel();

        // Clear connections
        self.connections.clear();
    }

    fn check_health(
        &self,
        instance_id: dynamo_identity::InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    > {
        Box::pin(async move {
            // Check if we have an existing connection
            let connection_exists = self.connections.contains_key(&instance_id);

            if let Some(handle) = self.connections.get(&instance_id) {
                // Check if the channel is still connected (socket is still live)
                // If the writer task has exited (socket closed), the channel will be disconnected
                if !handle.tx.is_disconnected() {
                    return Ok(()); // Connection is alive and healthy
                }
                // Channel is disconnected, connection is dead - fall through to connect check
            }

            // No existing connection or connection is dead - verify peer is reachable
            let addr = *self
                .peers
                .get(&instance_id)
                .ok_or(HealthCheckError::PeerNotRegistered)?
                .value();

            // Try to connect (and immediately drop) to verify peer is reachable
            match tokio::time::timeout(timeout, TcpStream::connect(addr)).await {
                Ok(Ok(_stream)) => {
                    // Connection successful, drop immediately
                    // If we never had a connection before, report NeverConnected
                    // If we had one before that failed, report Ok (peer is reachable now)
                    if connection_exists {
                        Ok(())
                    } else {
                        Err(HealthCheckError::NeverConnected)
                    }
                }
                Ok(Err(_)) => Err(HealthCheckError::ConnectionFailed),
                Err(_) => Err(HealthCheckError::Timeout),
            }
        })
    }
}

/// Connection writer task
///
/// This task runs on the LocalSet and handles writing framed bytes to the TCP stream.
/// It receives pre-encoded frames via a flume channel and writes them to the socket.
async fn connection_writer_task(
    addr: SocketAddr,
    rx: flume::Receiver<SendTask>,
    cancel_token: CancellationToken,
) -> Result<()> {
    debug!("Connecting to {}", addr);

    // Connect to remote peer
    let stream = TcpStream::connect(addr).await.context("connect failed")?;

    // Configure socket for low latency
    if let Err(e) = stream.set_nodelay(true) {
        warn!("Failed to set TCP_NODELAY: {}", e);
    }

    let sock = socket2::SockRef::from(&stream);
    if let Err(e) = sock.set_tcp_keepalive(
        &socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10)),
    ) {
        warn!("Failed to set keepalive: {}", e);
    }

    if let Err(e) = sock.set_send_buffer_size(1_048_576) {
        warn!("Failed to set send buffer size: {}", e);
    }

    // Create framed writer
    let mut framed = Framed::new(stream, TcpFrameCodec::new());

    debug!("Connected to {}", addr);

    // Main send loop
    loop {
        tokio::select! {
            Ok(task) = rx.recv_async() => {
                // Send pre-framed bytes
                match framed.send(task.framed_bytes).await {
                    Ok(()) => {
                        // Success - continue
                    }
                    Err(e) => {
                        // Network error - invoke callback and exit
                        error!("TCP write error to {}: {}", addr, e);
                        task.on_error.on_error(
                            task.header,
                            task.payload,
                            format!("TCP write failed: {}", e)
                        );
                        break;
                    }
                }
            }
            _ = cancel_token.cancelled() => {
                debug!("Writer task for {} cancelled, draining queue", addr);
                // Graceful shutdown: drain channel
                while let Ok(task) = rx.try_recv() {
                    if framed.send(task.framed_bytes).await.is_err() {
                        break;
                    }
                }
                break;
            }
        }
    }

    // Flush and close
    let _ = framed.flush().await;
    debug!("Connection to {} closed", addr);

    Ok(())
}

/// Parse a TCP endpoint string into a SocketAddr
///
/// Accepts formats:
/// - "tcp://host:port"
/// - "host:port"
fn parse_tcp_endpoint(endpoint: &[u8]) -> Result<SocketAddr> {
    let endpoint_str = std::str::from_utf8(endpoint).context("endpoint is not valid UTF-8")?;

    // Strip "tcp://" prefix if present
    let addr_str = endpoint_str.strip_prefix("tcp://").unwrap_or(endpoint_str);

    // Parse as socket address
    let mut addrs = addr_str
        .to_socket_addrs()
        .context("failed to parse socket address")?;

    addrs
        .next()
        .ok_or_else(|| anyhow::anyhow!("no addresses resolved"))
}

/// Builder for TcpTransport
pub struct TcpTransportBuilder {
    bind_addr: Option<SocketAddr>,
    key: Option<TransportKey>,
    channel_capacity: usize,
}

impl TcpTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            bind_addr: None,
            key: None,
            channel_capacity: 256,
        }
    }

    /// Set the bind address
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }

    /// Set the transport key
    pub fn key(mut self, key: TransportKey) -> Self {
        self.key = Some(key);
        self
    }

    /// Set the channel capacity for backpressure (default: 256)
    pub fn channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Build the TcpTransport
    pub fn build(self) -> Result<TcpTransport> {
        let bind_addr = self
            .bind_addr
            .ok_or_else(|| anyhow::anyhow!("bind_addr is required"))?;
        let key = self.key.unwrap_or_else(|| TransportKey::from("tcp"));

        // Build local address (just the bind address for now)
        let local_endpoint = format!("tcp://{}", bind_addr);
        let mut addr_builder = WorkerAddress::builder();
        addr_builder.add_entry(key.clone(), local_endpoint.as_bytes().to_vec())?;
        let local_address = addr_builder.build()?;

        Ok(TcpTransport::new(bind_addr, key, local_address))
    }
}

impl Default for TcpTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tcp_endpoint() {
        // With tcp:// prefix
        let addr = parse_tcp_endpoint(b"tcp://127.0.0.1:5555").unwrap();
        assert_eq!(addr.port(), 5555);

        // Without prefix
        let addr = parse_tcp_endpoint(b"127.0.0.1:6666").unwrap();
        assert_eq!(addr.port(), 6666);

        // Invalid
        assert!(parse_tcp_endpoint(b"invalid").is_err());
    }

    #[test]
    fn test_builder_requires_bind_addr() {
        let result = TcpTransportBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_bind_addr() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let result = TcpTransportBuilder::new().bind_addr(addr).build();
        assert!(result.is_ok());
    }
}
