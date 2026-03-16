// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-performance TCP transport with single-threaded optimizations
//!
//! This implementation uses Rc+RefCell+LocalSet for maximum performance on a single CPU core.
//! All operations run on the same thread as the TCP listener for optimal cache locality.

use anyhow::{Context, Result};
use bytes::Bytes;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::transport::{HealthCheckError, ShutdownState, TransportError, TransportErrorHandler};
use crate::utils::interfaces::{
    InterfaceEndpoint, InterfaceFilter, parse_endpoints, resolve_advertise_endpoints,
    select_best_endpoint,
};
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
    peers: Arc<DashMap<crate::InstanceId, SocketAddr>>,
    connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>>,

    // Runtime handle for spawning tasks
    runtime: OnceLock<tokio::runtime::Handle>,

    // Shutdown coordination
    cancel_token: CancellationToken,
    shutdown_state: OnceLock<ShutdownState>,

    // Send channel capacity for backpressure
    channel_capacity: usize,

    // Connect timeout for outbound connections
    connect_timeout: Duration,

    // Optional pre-bound listener (used for tests to avoid port races)
    listener: Mutex<Option<std::net::TcpListener>>,

    // Cached local interfaces for endpoint selection
    local_interfaces: OnceLock<Vec<InterfaceEndpoint>>,

    // NUMA hint for topology-aware NIC selection
    numa_hint: Option<u32>,
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

impl TcpTransport {
    /// Create a new TCP transport bound to `bind_addr` with the given transport key.
    ///
    /// An optional pre-bound `listener` can be provided (useful for tests binding
    /// to port 0). `channel_capacity` controls backpressure on per-connection
    /// writer channels (default 256).
    pub fn new(
        bind_addr: SocketAddr,
        key: TransportKey,
        local_address: WorkerAddress,
        channel_capacity: usize,
        connect_timeout: Duration,
        listener: Option<std::net::TcpListener>,
        numa_hint: Option<u32>,
    ) -> Self {
        Self {
            key,
            bind_addr,
            local_address,
            peers: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            runtime: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            shutdown_state: OnceLock::new(),
            channel_capacity,
            connect_timeout,
            listener: Mutex::new(listener),
            local_interfaces: OnceLock::new(),
            numa_hint,
        }
    }

    /// Optional: Pre-establish connection after registration
    ///
    /// This can be called after `register()` to eagerly establish the TCP connection
    /// instead of waiting for the first `send_message()` call.
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
        let addr = *self
            .peers
            .get(&instance_id)
            .ok_or(TransportError::PeerNotRegistered(instance_id))?
            .value();

        let (tx, rx) = flume::bounded(self.channel_capacity);
        let handle = ConnectionHandle { tx };

        let cancel = self.cancel_token.clone();
        let conns = Arc::clone(&self.connections);
        let connect_timeout = self.connect_timeout;
        rt.spawn(connection_writer_task(
            addr,
            instance_id,
            rx,
            conns,
            cancel,
            connect_timeout,
        ));

        debug!("Created new connection to {} ({})", instance_id, addr);
        Ok(handle)
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

        // Parse endpoints (supports both new multi-endpoint and legacy formats)
        let remote_endpoints = parse_endpoints(&endpoint).map_err(|e| {
            error!("Failed to parse TCP endpoint: {}", e);
            TransportError::InvalidEndpoint
        })?;

        // Lazy-init local interfaces for endpoint selection
        let local = self.local_interfaces.get_or_init(|| {
            resolve_advertise_endpoints(self.bind_addr, &InterfaceFilter::All).unwrap_or_default()
        });

        // Select best endpoint based on NUMA + subnet affinity
        let addr = select_best_endpoint(&remote_endpoints, local, self.numa_hint)
            .ok_or(TransportError::InvalidEndpoint)?;

        // Store peer address
        self.peers.insert(peer_info.instance_id(), addr);

        debug!("Registered peer {} at {}", peer_info.instance_id(), addr);

        Ok(())
    }

    #[inline]
    fn send_message(
        &self,
        instance_id: crate::InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: std::sync::Arc<dyn TransportErrorHandler>,
    ) {
        // Convert to Bytes (one allocation each)
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

        let bind_addr = self.bind_addr;
        let shutdown_state = channels.shutdown_state.clone();
        // Take ownership of the listener (if present) - we can only start once
        let listener = self
            .listener
            .lock()
            .expect("Listener mutex poisoned")
            .take();

        Box::pin(async move {
            // Create error handler that routes to the transport error handler
            struct DefaultErrorHandler;
            impl TransportErrorHandler for DefaultErrorHandler {
                fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
                    warn!("Transport error: {}", error);
                }
            }

            // Start TCP listener
            let tcp_listener = TcpListener::builder()
                .bind_addr(bind_addr)
                .adapter(channels)
                .error_handler(std::sync::Arc::new(DefaultErrorHandler))
                .shutdown_state(shutdown_state)
                .listener(listener)
                .build()?;

            rt.spawn(async move {
                if let Err(e) = tcp_listener.serve().await {
                    error!("TCP listener error: {}", e);
                }
            });

            info!("TCP transport started on {}", bind_addr);

            Ok(())
        })
    }

    fn begin_drain(&self) {
        // Per-frame gate in the listener handles drain — no-op here.
    }

    fn shutdown(&self) {
        info!("Shutting down TCP transport");

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
            // Check if we have an existing connection
            let connection_exists = self.connections.contains_key(&instance_id);

            if let Some(handle) = self.connections.get(&instance_id) {
                // Check if the channel is still connected (socket is still live)
                // If the writer task has exited (socket closed), the channel will be disconnected
                if !handle.tx.is_disconnected() {
                    return Ok(()); // Connection is alive and healthy
                }
                // Channel is disconnected — drop guard and remove stale entry
                drop(handle);
                self.connections
                    .remove_if(&instance_id, |_, h| h.tx.is_disconnected());
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
///
/// Cleanup (draining queued messages and removing the stale map entry) always runs,
/// even if the initial TCP connect fails.
async fn connection_writer_task(
    addr: SocketAddr,
    instance_id: crate::InstanceId,
    rx: flume::Receiver<SendTask>,
    connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>>,
    cancel_token: CancellationToken,
    connect_timeout: Duration,
) -> Result<()> {
    let result =
        connection_writer_inner(addr, instance_id, &rx, &cancel_token, connect_timeout).await;

    // Always drain queued messages and notify their error handlers.
    //
    // TODO: There is a tiny race between the drain finishing and `drop(rx)`:
    // a sender on another thread could `try_send` successfully in that window,
    // and the message would be silently dropped when rx is destroyed. Closing
    // this fully would require swapping the map entry with a "poisoned" handle
    // (a disconnected tx) before draining, so fast-path senders see a failure
    // instead. Not worth the complexity today — at most one message is affected,
    // and async senders already get `SendError` once rx is dropped.
    while let Ok(msg) = rx.try_recv() {
        msg.on_error("Connection closed");
    }

    // Drop the receiver so our sender half becomes disconnected, then remove
    // the stale entry. The predicate ensures we only remove our own entry —
    // a replacement connection's tx will still be connected.
    drop(rx);
    connections.remove_if(&instance_id, |_, h| h.tx.is_disconnected());

    debug!("Connection to {} ({}) closed", instance_id, addr);

    result
}

/// Inner loop: connect, configure the socket, and send frames until the channel
/// closes or a write error occurs.
async fn connection_writer_inner(
    addr: SocketAddr,
    instance_id: crate::InstanceId,
    rx: &flume::Receiver<SendTask>,
    cancel_token: &CancellationToken,
    connect_timeout: Duration,
) -> Result<()> {
    debug!("Connecting to {}", addr);

    let mut stream = tokio::select! {
        _ = cancel_token.cancelled() => return Ok(()),
        res = tokio::time::timeout(connect_timeout, TcpStream::connect(addr)) => {
            res.context("connect timeout")?.context("connect failed")?
        },
    };

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

    if let Err(e) = sock.set_send_buffer_size(2_097_152) {
        warn!("Failed to set send buffer size: {}", e);
    }

    if let Err(e) = sock.set_recv_buffer_size(2_097_152) {
        warn!("Failed to set recv buffer size: {}", e);
    }

    debug!("Connected to {}", addr);

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
            error!("Write error to {} ({}): {}", instance_id, addr, e);
            msg.on_error(format!("Failed to write to stream: {}", e));
            break;
        }
    }

    Ok(())
}

/// Parse a TCP endpoint string into a SocketAddr (legacy format, used in tests).
#[cfg(test)]
fn parse_tcp_endpoint(endpoint: &[u8]) -> Result<SocketAddr> {
    use std::net::ToSocketAddrs;

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
    connect_timeout: Duration,
    listener: Option<std::net::TcpListener>,
    interface_filter: InterfaceFilter,
    numa_hint: Option<u32>,
}

impl TcpTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            bind_addr: None,
            key: None,
            channel_capacity: 256,
            connect_timeout: Duration::from_secs(5),
            listener: None,
            interface_filter: InterfaceFilter::default(),
            numa_hint: None,
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

    /// Set the connect timeout for outbound connections (default: 5s)
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the interface selection filter for multi-NIC environments.
    pub fn interface_filter(mut self, filter: InterfaceFilter) -> Self {
        self.interface_filter = filter;
        self
    }

    /// Set the NUMA node hint for topology-aware NIC selection.
    ///
    /// Callers typically resolve this via `dynamo_memory::numa::get_device_numa_node(gpu_id)`.
    pub fn numa_hint(mut self, node: u32) -> Self {
        self.numa_hint = Some(node);
        self
    }

    /// Use a pre-bound TcpListener instead of binding to a specific address
    ///
    /// This is useful for tests where you want to bind to port 0 and get an OS-assigned
    /// port without creating a race condition between binding and starting the transport.
    ///
    /// Note: This is mutually exclusive with `bind_addr()`. Using both will result in an error.
    pub fn from_listener(mut self, listener: std::net::TcpListener) -> Result<Self> {
        // Validate mutual exclusivity: can't use both bind_addr() and from_listener()
        if self.bind_addr.is_some() {
            anyhow::bail!(
                "Cannot use both bind_addr() and from_listener() - they are mutually exclusive"
            );
        }

        let addr = listener
            .local_addr()
            .context("Failed to get local address from listener")?;
        self.bind_addr = Some(addr);
        self.listener = Some(listener);
        Ok(self)
    }

    /// Build the TcpTransport
    pub fn build(self) -> Result<TcpTransport> {
        let key = self.key.unwrap_or_else(|| TransportKey::from("tcp"));

        // If we have a listener, use its address; otherwise pre-bind to resolve port 0.
        let (bind_addr, listener) = if let Some(listener) = self.listener {
            let addr = listener.local_addr()?;
            (addr, Some(listener))
        } else {
            let requested = self
                .bind_addr
                .unwrap_or_else(|| "0.0.0.0:0".parse().unwrap());
            let std_listener = std::net::TcpListener::bind(requested)
                .context("Failed to pre-bind TCP listener")?;
            let actual = std_listener.local_addr()?;
            (actual, Some(std_listener))
        };

        // Resolve advertise endpoints (multi-interface discovery)
        let endpoints = resolve_advertise_endpoints(bind_addr, &self.interface_filter)?;

        // Warn if NUMA hint conflicts with interface filter
        if let (Some(numa), InterfaceFilter::ByName(name)) =
            (self.numa_hint, &self.interface_filter)
        {
            for ep in &endpoints {
                if let Some(ep_numa) = ep.numa_node
                    && ep_numa != numa as i32
                {
                    warn!(
                        "NIC {} is on NUMA node {} but GPU NUMA hint is {}",
                        name, ep_numa, numa
                    );
                }
            }
        }

        let encoded =
            rmp_serde::to_vec(&endpoints).context("Failed to encode interface endpoints")?;
        let mut addr_builder = crate::address::WorkerAddressBuilder::new();
        addr_builder.add_entry(key.clone(), encoded)?;
        let local_address = addr_builder.build()?;

        Ok(TcpTransport::new(
            bind_addr,
            key,
            local_address,
            self.channel_capacity,
            self.connect_timeout,
            listener,
            self.numa_hint,
        ))
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

    /// Build a `PeerInfo` whose TCP endpoint points at `addr` using legacy format.
    fn make_tcp_peer(addr: SocketAddr) -> PeerInfo {
        let instance_id = crate::InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        builder
            .add_entry("tcp", format!("tcp://{}", addr).into_bytes())
            .unwrap();
        PeerInfo::new(instance_id, builder.build().unwrap())
    }

    /// Build a `PeerInfo` whose TCP endpoint uses the new multi-endpoint format.
    fn make_tcp_peer_multi(endpoints: Vec<InterfaceEndpoint>) -> PeerInfo {
        let instance_id = crate::InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        let encoded = rmp_serde::to_vec(&endpoints).unwrap();
        builder.add_entry("tcp", encoded).unwrap();
        PeerInfo::new(instance_id, builder.build().unwrap())
    }

    /// Build a `TcpTransport` with its runtime set, bound to a real listener.
    /// Returns `(transport, listener_addr)`.
    fn make_transport() -> (TcpTransport, SocketAddr) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let transport = TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap();
        // Set the runtime handle so `get_or_create_connection` can spawn tasks.
        transport
            .runtime
            .set(tokio::runtime::Handle::current())
            .ok();
        (transport, addr)
    }

    /// Insert a stale `ConnectionHandle` into the transport's connections map.
    /// A "stale" handle is one whose receiver has been dropped.
    fn insert_stale_handle(transport: &TcpTransport, instance_id: crate::InstanceId) {
        let (tx, _rx) = flume::bounded::<SendTask>(1);
        // Drop _rx immediately so tx.is_disconnected() == true
        transport
            .connections
            .insert(instance_id, ConnectionHandle { tx });
    }

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
    fn test_builder_default_prebinds() {
        // Builder without explicit bind_addr should pre-bind to 0.0.0.0:0
        let result = TcpTransportBuilder::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_bind_addr() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let result = TcpTransportBuilder::new().bind_addr(addr).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_listener() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let result = TcpTransportBuilder::new().from_listener(listener);
        assert!(result.is_ok());
        let result = result.unwrap().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_bind_addr_and_listener_mutually_exclusive() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let result = TcpTransportBuilder::new()
            .bind_addr(addr)
            .from_listener(listener);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("mutually exclusive"));
    }

    #[test]
    fn test_builder_multi_endpoint_format() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let transport = TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap();

        // The address should contain msgpack-encoded endpoints
        let wa = transport.address();
        let raw = wa.get_entry("tcp").unwrap().unwrap();
        let endpoints: Vec<InterfaceEndpoint> = rmp_serde::from_slice(&raw).unwrap();
        assert!(!endpoints.is_empty());
        // All endpoints should have the correct port
        for ep in &endpoints {
            assert_eq!(ep.port, addr.port());
        }
    }

    #[tokio::test]
    async fn test_register_legacy_format() {
        let (transport, _our_addr) = make_transport();
        let peer_addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let peer = make_tcp_peer(peer_addr);
        let iid = peer.instance_id();
        // Legacy "tcp://host:port" format should still work
        transport.register(peer).unwrap();
        assert!(transport.peers.contains_key(&iid));
    }

    #[tokio::test]
    async fn test_register_multi_endpoint_format() {
        let (transport, _our_addr) = make_transport();
        let endpoints = vec![InterfaceEndpoint {
            name: "eth0".to_string(),
            ip: "127.0.0.1".to_string(),
            port: 9999,
            prefix_len: 8,
            numa_node: None,
        }];
        let peer = make_tcp_peer_multi(endpoints);
        let iid = peer.instance_id();
        transport.register(peer).unwrap();
        assert!(transport.peers.contains_key(&iid));
    }

    #[tokio::test]
    async fn test_get_or_create_connection_replaces_stale_handle() {
        let (transport, _our_addr) = make_transport();

        // Start a listener that the transport can connect to
        let peer_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer_addr = peer_listener.local_addr().unwrap();

        let peer = make_tcp_peer(peer_addr);
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
    }

    #[tokio::test]
    async fn test_check_health_removes_stale_entry() {
        let (transport, _our_addr) = make_transport();

        // Start a listener so the peer is "reachable"
        let peer_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let peer_addr = peer_listener.local_addr().unwrap();

        let peer = make_tcp_peer(peer_addr);
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
        // (the peer is reachable via our test listener)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_writer_task_cleans_up_on_write_error() {
        // Bind a listener, accept once, then drop everything to cause a write error
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let iid = crate::InstanceId::new_v4();
        let (tx, rx) = flume::bounded::<SendTask>(8);

        let connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>> =
            Arc::new(DashMap::new());
        connections.insert(iid, ConnectionHandle { tx: tx.clone() });

        let conns = Arc::clone(&connections);
        let cancel = CancellationToken::new();

        // Spawn the writer task
        let writer = tokio::spawn(connection_writer_task(
            addr,
            iid,
            rx,
            conns,
            cancel,
            Duration::from_secs(5),
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
    }

    #[tokio::test]
    async fn test_send_message_does_not_fail_on_stale_handle() {
        let (transport, _our_addr) = make_transport();

        // Start a listener that accepts connections (simulates a healthy peer)
        let peer_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let peer_addr = peer_listener.local_addr().unwrap();

        let peer = make_tcp_peer(peer_addr);
        let iid = peer.instance_id();
        transport.register(peer).unwrap();

        // Insert a stale handle
        insert_stale_handle(&transport, iid);

        // send_message should detect the stale handle and create a new one,
        // NOT immediately call on_error
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
        // Give the async writer a moment to flush the frame
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
    }

    #[tokio::test]
    async fn test_writer_task_drains_on_connect_failure() {
        // Use an address where nothing is listening so connect will fail.
        // Binding then immediately dropping gives us a port that is guaranteed closed.
        let tmp = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = tmp.local_addr().unwrap();
        drop(tmp);

        let iid = crate::InstanceId::new_v4();
        let (tx, rx) = flume::bounded::<SendTask>(8);

        let connections: Arc<DashMap<crate::InstanceId, ConnectionHandle>> =
            Arc::new(DashMap::new());
        connections.insert(iid, ConnectionHandle { tx: tx.clone() });

        // Queue a message *before* the writer task even starts — this simulates
        // the race between create_connection returning and connect completing.
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

        let writer = tokio::spawn(connection_writer_task(
            addr,
            iid,
            rx,
            conns,
            cancel,
            Duration::from_secs(5),
        ));
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
    }
}
