// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! UDS transport implementation
//!
//! Structural mirror of the TCP transport (`tcp/transport.rs`), replacing
//! `TcpStream`/`TcpListener` with `UnixStream`/`UnixListener`.
//! Reuses `TcpFrameCodec` for framing since it operates on any `AsyncRead + AsyncWrite`.

use anyhow::{Context, Result};
use bytes::Bytes;
use dashmap::DashMap;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::transport::{HealthCheckError, ShutdownState, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

use super::listener::UdsListener;
use crate::tcp::TcpFrameCodec;

/// UDS transport with lock-free concurrent access
///
/// Mirrors `TcpTransport` but uses Unix domain sockets.
pub struct UdsTransport {
    key: TransportKey,
    socket_path: PathBuf,
    local_address: WorkerAddress,

    // Shared mutable state with DashMap (lock-free)
    peers: Arc<DashMap<crate::InstanceId, PathBuf>>,
    connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>>,

    // Runtime handle for spawning tasks
    runtime: OnceLock<tokio::runtime::Handle>,

    // Shutdown coordination
    cancel_token: CancellationToken,
    shutdown_state: OnceLock<ShutdownState>,

    // Send channel capacity for backpressure
    channel_capacity: usize,
}

/// Handle to a connection's writer task
#[derive(Clone)]
struct ConnectionHandle {
    tx: flume::Sender<SendTask>,
}

/// Task sent to writer task containing pre-encoded frame
struct SendTask {
    msg_type: MessageType,
    header: Bytes,
    payload: Bytes,
    on_error: Arc<dyn TransportErrorHandler>,
}

impl SendTask {
    fn on_error(self, error: impl Into<String>) {
        self.on_error
            .on_error(self.header, self.payload, error.into());
    }
}

impl UdsTransport {
    /// Create a new UDS transport
    pub fn new(
        socket_path: PathBuf,
        key: TransportKey,
        local_address: WorkerAddress,
        channel_capacity: usize,
    ) -> Self {
        Self {
            key,
            socket_path,
            local_address,
            peers: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            runtime: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            shutdown_state: OnceLock::new(),
            channel_capacity,
        }
    }

    /// Get the socket path this transport is bound to
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Optional: Pre-establish connection after registration
    pub fn ensure_connected(&self, instance_id: crate::InstanceId) -> Result<()> {
        self.get_or_create_connection(instance_id)?;
        Ok(())
    }

    /// Get or create a connection to a peer (lazy initialization)
    fn get_or_create_connection(&self, instance_id: crate::InstanceId) -> Result<ConnectionHandle> {
        // Fast path: connection already exists and is alive
        if let Some(handle) = self.connections.get(&instance_id) {
            if !handle.tx.is_disconnected() {
                return Ok(handle.clone());
            }
            // Stale — drop guard before mutating the map
            drop(handle);
            self.connections
                .remove_if(&instance_id, |_, h| h.tx.is_disconnected());
        }

        let rt = self.runtime.get().ok_or(TransportError::NotStarted)?;

        // Atomic check-and-insert via entry API
        let handle = match self.connections.entry(instance_id) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                if !entry.get().tx.is_disconnected() {
                    entry.get().clone()
                } else {
                    // Stale entry — replace in-place with a fresh connection
                    let handle = self.create_connection(instance_id, rt)?;
                    entry.insert(handle.clone());
                    handle
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let handle = self.create_connection(instance_id, rt)?;
                entry.insert(handle.clone());
                handle
            }
        };

        Ok(handle)
    }

    /// Create a new connection handle and spawn the writer task.
    fn create_connection(
        &self,
        instance_id: crate::InstanceId,
        rt: &tokio::runtime::Handle,
    ) -> Result<ConnectionHandle> {
        let path = self
            .peers
            .get(&instance_id)
            .ok_or(TransportError::PeerNotRegistered(instance_id))?
            .value()
            .clone();

        let (tx, rx) = flume::bounded(self.channel_capacity);
        let handle = ConnectionHandle { tx };

        let cancel = self.cancel_token.clone();
        let conns = Arc::clone(&self.connections);

        debug!("Created new UDS connection to {} ({:?})", instance_id, path);
        rt.spawn(connection_writer_task(path, instance_id, rx, conns, cancel));
        Ok(handle)
    }
}

impl Transport for UdsTransport {
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

        // Parse UDS endpoint (expected format: "uds:///path/to/socket" or "/path/to/socket")
        let path = parse_uds_endpoint(&endpoint).map_err(|e| {
            error!("Failed to parse UDS endpoint: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Store peer path
        self.peers.insert(peer_info.instance_id(), path.clone());

        debug!("Registered peer {} at {:?}", peer_info.instance_id(), path);

        Ok(())
    }

    #[inline]
    fn send_message(
        &self,
        instance_id: crate::InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
    ) {
        let header = Bytes::from(header);
        let payload = Bytes::from(payload);

        let send_msg = SendTask {
            msg_type: message_type,
            header,
            payload,
            on_error,
        };

        // Fast path: try to send on existing connection
        let send_msg = match self.connections.get(&instance_id) {
            Some(handle) => match handle.tx.try_send(send_msg) {
                Ok(()) => return,
                Err(flume::TrySendError::Full(send_msg)) => send_msg,
                Err(flume::TrySendError::Disconnected(send_msg)) => {
                    // Drop the guard before mutating the map
                    drop(handle);
                    self.connections
                        .remove_if(&instance_id, |_, h| h.tx.is_disconnected());
                    // Fall through to slow path to create a fresh connection
                    send_msg
                }
            },
            None => send_msg,
        };

        // Slow path: create new connection
        let rt = match self.runtime.get() {
            Some(rt) => rt,
            None => {
                send_msg.on_error("Transport not started");
                return;
            }
        };

        let handle = match self.get_or_create_connection(instance_id) {
            Ok(h) => h,
            Err(e) => {
                send_msg.on_error(format!("Failed to create connection: {}", e));
                return;
            }
        };

        rt.spawn(async move {
            if let Err(flume::SendError(send_msg)) = handle.tx.send_async(send_msg).await {
                send_msg.on_error("Connection closed");
            }
        });
    }

    fn start(
        &self,
        _instance_id: crate::InstanceId,
        channels: TransportAdapter,
        rt: tokio::runtime::Handle,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
        // Store runtime handle for use in send_message
        self.runtime.set(rt.clone()).ok();

        // Capture shutdown state from the adapter
        self.shutdown_state
            .set(channels.shutdown_state.clone())
            .ok();

        let socket_path = self.socket_path.clone();
        let shutdown_state = channels.shutdown_state.clone();

        Box::pin(async move {
            struct DefaultErrorHandler;
            impl TransportErrorHandler for DefaultErrorHandler {
                fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
                    warn!("UDS transport error: {}", error);
                }
            }

            // Remove a stale socket file only when it is safe to do so.
            if socket_path.exists() {
                let is_socket = std::fs::metadata(&socket_path)
                    .map(|m| m.file_type().is_socket())
                    .unwrap_or(false);
                if !is_socket {
                    anyhow::bail!(
                        "path {:?} exists and is not a Unix domain socket",
                        socket_path
                    );
                }
                // Probe liveness: a successful connect means a live listener owns it.
                match tokio::time::timeout(
                    Duration::from_millis(100),
                    UnixStream::connect(&socket_path),
                )
                .await
                {
                    Ok(Ok(_)) => {
                        anyhow::bail!(
                            "a live UDS listener is already running at {:?}",
                            socket_path
                        );
                    }
                    _ => {
                        // Stale (connection refused / timeout) — safe to unlink.
                        std::fs::remove_file(&socket_path).ok();
                    }
                }
            }

            // Build and bind before spawning so that start() only returns Ok
            // after the OS-level bind succeeds.
            let uds_listener = UdsListener::builder()
                .socket_path(socket_path.clone())
                .adapter(channels)
                .error_handler(Arc::new(DefaultErrorHandler))
                .shutdown_state(shutdown_state)
                .build()?;

            let bound_listener = uds_listener.bind()?;

            rt.spawn(async move {
                if let Err(e) = bound_listener.serve().await {
                    error!("UDS listener error: {}", e);
                }
            });

            info!("UDS transport started on {:?}", socket_path);

            Ok(())
        })
    }

    fn begin_drain(&self) {
        if let Some(state) = self.shutdown_state.get() {
            state.begin_drain();
        }
    }

    fn shutdown(&self) {
        info!("Shutting down UDS transport");

        // Cancel the teardown token (Phase 3) to stop the listener and connection handlers
        if let Some(state) = self.shutdown_state.get() {
            state.teardown_token().cancel();
        }
        self.cancel_token.cancel();

        // Clear connections
        self.connections.clear();
    }

    fn check_health(
        &self,
        instance_id: crate::InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    > {
        Box::pin(async move {
            let connection_exists = self.connections.contains_key(&instance_id);

            if let Some(handle) = self.connections.get(&instance_id) {
                if !handle.tx.is_disconnected() {
                    return Ok(());
                }
                // Channel is disconnected — drop guard and remove stale entry
                drop(handle);
                self.connections
                    .remove_if(&instance_id, |_, h| h.tx.is_disconnected());
            }

            // No existing connection or connection is dead - verify peer is reachable
            let path = self
                .peers
                .get(&instance_id)
                .ok_or(HealthCheckError::PeerNotRegistered)?
                .value()
                .clone();

            // Try to connect (and immediately drop) to verify peer is reachable
            match tokio::time::timeout(timeout, UnixStream::connect(&path)).await {
                Ok(Ok(_stream)) => {
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

/// Connection writer task for UDS
///
/// Mirrors the TCP connection_writer_task. Cleanup (draining queued messages
/// and removing the stale map entry) always runs, even if the initial connect fails.
async fn connection_writer_task(
    path: PathBuf,
    instance_id: crate::InstanceId,
    rx: flume::Receiver<SendTask>,
    connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>>,
    cancel_token: CancellationToken,
) -> Result<()> {
    let result = connection_writer_inner(&path, instance_id, &rx, &cancel_token).await;

    // Always drain queued messages and notify their error handlers.
    while let Ok(msg) = rx.try_recv() {
        msg.on_error("Connection closed");
    }

    // Drop the receiver so our sender half becomes disconnected, then remove
    // the stale entry. The predicate ensures we only remove our own entry —
    // a replacement connection's tx will still be connected.
    drop(rx);
    connections.remove_if(&instance_id, |_, h| h.tx.is_disconnected());

    debug!("UDS connection to {} ({:?}) closed", instance_id, path);

    result
}

/// Inner loop: connect and send frames until the channel closes or a write error occurs.
async fn connection_writer_inner(
    path: &Path,
    instance_id: crate::InstanceId,
    rx: &flume::Receiver<SendTask>,
    cancel_token: &CancellationToken,
) -> Result<()> {
    debug!("Connecting to UDS {:?}", path);

    let mut stream = tokio::select! {
        _ = cancel_token.cancelled() => return Ok(()),
        res = UnixStream::connect(path) => res.context("UDS connect failed")?,
    };

    debug!("Connected to UDS {:?}", path);

    // Main send loop
    loop {
        let msg = tokio::select! {
            _ = cancel_token.cancelled() => break,
            res = rx.recv_async() => match res {
                Ok(msg) => msg,
                Err(_) => break,
            },
        };
        if let Err(e) =
            TcpFrameCodec::encode_frame(&mut stream, msg.msg_type, &msg.header, &msg.payload).await
        {
            error!("Write error to {} ({:?}): {}", instance_id, path, e);
            msg.on_error(format!("Failed to write to UDS stream: {}", e));
            break;
        }
    }

    Ok(())
}

/// Parse a UDS endpoint string into a PathBuf
///
/// Accepts formats:
/// - "uds:///path/to/socket"
/// - "/path/to/socket"
fn parse_uds_endpoint(endpoint: &[u8]) -> Result<PathBuf> {
    let endpoint_str = std::str::from_utf8(endpoint).context("endpoint is not valid UTF-8")?;

    // Strip "uds://" prefix if present
    let path_str = endpoint_str.strip_prefix("uds://").unwrap_or(endpoint_str);

    if path_str.is_empty() {
        anyhow::bail!("empty UDS socket path");
    }

    Ok(PathBuf::from(path_str))
}

/// Builder for UdsTransport
pub struct UdsTransportBuilder {
    socket_path: Option<PathBuf>,
    key: Option<TransportKey>,
    channel_capacity: usize,
}

impl UdsTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            socket_path: None,
            key: None,
            channel_capacity: 256,
        }
    }

    /// Set the socket path
    pub fn socket_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.socket_path = Some(path.into());
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

    /// Build the UdsTransport
    pub fn build(self) -> Result<UdsTransport> {
        let socket_path = self
            .socket_path
            .ok_or_else(|| anyhow::anyhow!("socket_path is required"))?;
        let key = self.key.unwrap_or_else(|| TransportKey::from("uds"));

        let local_endpoint = format!("uds://{}", socket_path.display());
        let mut addr_builder = crate::address::WorkerAddressBuilder::new();
        addr_builder.add_entry(key.clone(), local_endpoint.as_bytes().to_vec())?;
        let local_address = addr_builder.build()?;

        Ok(UdsTransport::new(
            socket_path,
            key,
            local_address,
            self.channel_capacity,
        ))
    }
}

impl Default for UdsTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::WorkerAddressBuilder;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use velo_common::PeerInfo;

    /// Error handler that discards errors (for tests that don't need to track them).
    struct NullErrorHandler;
    impl TransportErrorHandler for NullErrorHandler {
        fn on_error(&self, _: Bytes, _: Bytes, _: String) {}
    }

    /// Error handler that counts errors (for tests that verify error routing).
    struct TrackingErrorHandler {
        count: AtomicUsize,
    }

    impl TrackingErrorHandler {
        fn new() -> Self {
            Self {
                count: AtomicUsize::new(0),
            }
        }

        fn error_count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
    }

    impl TransportErrorHandler for TrackingErrorHandler {
        fn on_error(&self, _: Bytes, _: Bytes, _: String) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Build a `PeerInfo` whose UDS endpoint points at `path`.
    fn make_uds_peer(path: &Path) -> PeerInfo {
        let instance_id = crate::InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("uds", format!("uds://{}", path.display()).into_bytes())
            .unwrap();
        PeerInfo::new(instance_id, builder.build().unwrap())
    }

    /// Build a `UdsTransport` with its runtime set, bound to a temp socket path.
    /// Returns `(transport, socket_path)`.
    fn make_transport() -> (UdsTransport, PathBuf) {
        let dir = std::env::temp_dir().join(format!("uds-test-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("test.sock");
        let transport = UdsTransportBuilder::new()
            .socket_path(&socket_path)
            .build()
            .unwrap();
        transport
            .runtime
            .set(tokio::runtime::Handle::current())
            .ok();
        (transport, socket_path)
    }

    /// Insert a stale `ConnectionHandle` into the transport's connections map.
    fn insert_stale_handle(transport: &UdsTransport, instance_id: crate::InstanceId) {
        let (tx, _rx) = flume::bounded::<SendTask>(1);
        // Drop _rx immediately so tx.is_disconnected() == true
        transport
            .connections
            .insert(instance_id, ConnectionHandle { tx });
    }

    #[test]
    fn test_parse_uds_endpoint() {
        // With uds:// prefix
        let path = parse_uds_endpoint(b"uds:///tmp/test.sock").unwrap();
        assert_eq!(path, PathBuf::from("/tmp/test.sock"));

        // Without prefix
        let path = parse_uds_endpoint(b"/var/run/anvil.sock").unwrap();
        assert_eq!(path, PathBuf::from("/var/run/anvil.sock"));

        // Empty path
        assert!(parse_uds_endpoint(b"").is_err());
    }

    #[test]
    fn test_builder_requires_socket_path() {
        let result = UdsTransportBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_socket_path() {
        let result = UdsTransportBuilder::new()
            .socket_path("/tmp/test.sock")
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_custom_key() {
        let transport = UdsTransportBuilder::new()
            .socket_path("/tmp/test.sock")
            .key(TransportKey::from("custom-uds"))
            .build()
            .unwrap();
        assert_eq!(transport.key(), TransportKey::from("custom-uds"));
    }

    #[test]
    fn test_transport_socket_path() {
        let transport = UdsTransportBuilder::new()
            .socket_path("/tmp/test.sock")
            .build()
            .unwrap();
        assert_eq!(transport.socket_path(), Path::new("/tmp/test.sock"));
    }

    #[tokio::test]
    async fn test_get_or_create_connection_replaces_stale_handle() {
        let (transport, _socket_path) = make_transport();

        // Start a UDS listener that the transport can connect to
        let dir = std::env::temp_dir().join(format!("uds-peer-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let peer_socket = dir.join("peer.sock");
        let peer_listener = tokio::net::UnixListener::bind(&peer_socket).unwrap();

        let peer = make_uds_peer(&peer_socket);
        let iid = peer.instance_id();
        transport.register(peer).unwrap();

        // Insert a stale handle
        insert_stale_handle(&transport, iid);
        assert!(
            transport
                .connections
                .get(&iid)
                .unwrap()
                .tx
                .is_disconnected()
        );

        // get_or_create_connection should replace the stale handle with a live one
        let handle = transport.get_or_create_connection(iid).unwrap();
        assert!(!handle.tx.is_disconnected());

        // The map entry should also be live
        let entry = transport.connections.get(&iid).unwrap();
        assert!(!entry.tx.is_disconnected());

        // Cleanup
        drop(peer_listener);
        std::fs::remove_file(&peer_socket).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_check_health_removes_stale_entry() {
        let (transport, _socket_path) = make_transport();

        // Start a UDS listener so the peer is "reachable"
        let dir = std::env::temp_dir().join(format!("uds-peer-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let peer_socket = dir.join("peer.sock");
        let _peer_listener = tokio::net::UnixListener::bind(&peer_socket).unwrap();

        let peer = make_uds_peer(&peer_socket);
        let iid = peer.instance_id();
        transport.register(peer).unwrap();

        // Insert stale handle — simulates a dead writer task
        insert_stale_handle(&transport, iid);
        assert!(transport.connections.contains_key(&iid));

        // check_health should remove the stale entry and verify the peer is reachable
        let result = transport.check_health(iid, Duration::from_secs(2)).await;

        // Stale entry should be gone
        assert!(!transport.connections.contains_key(&iid));

        // Since there WAS a previous connection entry, check_health returns Ok
        assert!(result.is_ok());

        // Cleanup
        std::fs::remove_file(&peer_socket).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_writer_task_cleans_up_on_write_error() {
        // Bind a UDS listener, accept once, then drop everything to cause a write error
        let dir = std::env::temp_dir().join(format!("uds-test-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("writer-test.sock");
        let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();

        let iid = crate::InstanceId::new_v4();
        let (tx, rx) = flume::bounded::<SendTask>(8);

        let connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>> =
            Arc::new(DashMap::new());
        connections.insert(iid, ConnectionHandle { tx: tx.clone() });

        let conns = Arc::clone(&connections);
        let cancel = CancellationToken::new();

        // Spawn the writer task
        let writer = tokio::spawn(connection_writer_task(
            socket_path.clone(),
            iid,
            rx,
            conns,
            cancel,
        ));

        // Accept the connection, then immediately drop it + the listener
        let (stream, _) = listener.accept().await.unwrap();
        drop(stream);
        drop(listener);

        // Send a message — the writer should hit a broken-pipe error
        tx.send(SendTask {
            msg_type: MessageType::Message,
            header: Bytes::from_static(b"hdr"),
            payload: Bytes::from_static(b"pay"),
            on_error: Arc::new(NullErrorHandler),
        })
        .unwrap();

        // Wait for writer task to finish
        let _ = writer.await;

        // The writer should have removed the stale entry from the map
        assert!(
            !connections.contains_key(&iid),
            "writer task should clean up its DashMap entry on write error"
        );

        // Cleanup
        std::fs::remove_file(&socket_path).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_send_message_does_not_fail_on_stale_handle() {
        let (transport, _socket_path) = make_transport();

        // Start a UDS listener that accepts connections (simulates a healthy peer)
        let dir = std::env::temp_dir().join(format!("uds-peer-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let peer_socket = dir.join("peer.sock");
        let peer_listener = tokio::net::UnixListener::bind(&peer_socket).unwrap();

        let peer = make_uds_peer(&peer_socket);
        let iid = peer.instance_id();
        transport.register(peer).unwrap();

        // Insert a stale handle
        insert_stale_handle(&transport, iid);

        // send_message should detect the stale handle and create a new one
        let error_handler = Arc::new(TrackingErrorHandler::new());
        transport.send_message(
            iid,
            b"test-header".to_vec(),
            b"test-payload".to_vec(),
            MessageType::Message,
            error_handler.clone(),
        );

        // Accept the connection that the new writer task will establish
        let (mut stream, _) = peer_listener.accept().await.unwrap();

        // Read the framed message from the stream to confirm delivery
        use tokio::io::AsyncReadExt;
        let mut buf = [0u8; 256];
        let n = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf))
            .await
            .expect("timed out waiting for data")
            .expect("read error");
        assert!(n > 0, "expected data from the writer task");

        // No errors should have been reported
        assert_eq!(
            error_handler.error_count(),
            0,
            "send_message should retry on stale handle, not fail"
        );

        // The connections map should now contain a live handle
        let entry = transport.connections.get(&iid).unwrap();
        assert!(
            !entry.tx.is_disconnected(),
            "stale handle should have been replaced with a live one"
        );

        // Cleanup
        std::fs::remove_file(&peer_socket).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_double_bind_returns_err() {
        use crate::transport::make_channels;

        let dir = std::env::temp_dir().join(format!("uds-test-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("double-bind.sock");

        let transport1 = UdsTransportBuilder::new()
            .socket_path(&socket_path)
            .build()
            .unwrap();

        let instance_id = crate::InstanceId::new_v4();
        let (adapter1, _streams1) = make_channels();
        let rt = tokio::runtime::Handle::current();

        // First bind must succeed.
        transport1
            .start(instance_id, adapter1, rt.clone())
            .await
            .unwrap();

        // Second transport on the same path must fail.
        let transport2 = UdsTransportBuilder::new()
            .socket_path(&socket_path)
            .build()
            .unwrap();
        let (adapter2, _streams2) = make_channels();
        let result = transport2.start(instance_id, adapter2, rt).await;
        assert!(
            result.is_err(),
            "start() should return Err when a live listener already owns the socket"
        );

        // Cleanup
        transport1.shutdown();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_begin_drain_activates_draining_flag() {
        use crate::transport::make_channels;

        let dir = std::env::temp_dir().join(format!("uds-test-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("drain-test.sock");

        let transport = UdsTransportBuilder::new()
            .socket_path(&socket_path)
            .build()
            .unwrap();

        let instance_id = crate::InstanceId::new_v4();
        let (adapter, _streams) = make_channels();
        let rt = tokio::runtime::Handle::current();

        transport.start(instance_id, adapter, rt).await.unwrap();

        assert!(
            !transport.shutdown_state.get().unwrap().is_draining(),
            "should not be draining before begin_drain()"
        );

        transport.begin_drain();

        assert!(
            transport.shutdown_state.get().unwrap().is_draining(),
            "should be draining after begin_drain()"
        );

        // Cleanup
        transport.shutdown();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_writer_task_drains_on_connect_failure() {
        // Use a socket path where nothing is listening so connect will fail.
        let dir = std::env::temp_dir().join(format!("uds-test-{}", crate::InstanceId::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let dead_socket = dir.join("dead.sock");

        let iid = crate::InstanceId::new_v4();
        let (tx, rx) = flume::bounded::<SendTask>(8);

        let connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>> =
            Arc::new(DashMap::new());
        connections.insert(iid, ConnectionHandle { tx: tx.clone() });

        // Queue a message before the writer task starts
        let error_handler = Arc::new(TrackingErrorHandler::new());
        tx.send(SendTask {
            msg_type: MessageType::Message,
            header: Bytes::from_static(b"hdr"),
            payload: Bytes::from_static(b"pay"),
            on_error: error_handler.clone(),
        })
        .unwrap();

        let conns = Arc::clone(&connections);
        let cancel = CancellationToken::new();

        let writer = tokio::spawn(connection_writer_task(dead_socket, iid, rx, conns, cancel));
        let _ = writer.await;

        assert_eq!(
            error_handler.error_count(),
            1,
            "queued message should have its on_error called when connect fails"
        );

        assert!(
            !connections.contains_key(&iid),
            "writer task should clean up its DashMap entry on connect failure"
        );

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }
}
