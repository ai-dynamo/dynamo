// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common test utilities for transport integration tests
//!
//! This module provides a transport-agnostic test infrastructure that can be reused
//! across different transport implementations (TCP, RDMA, UDP, UDS, etc.).

#![allow(dead_code)]

#[cfg(feature = "grpc")]
use velo_transports::grpc::{GrpcTransport, GrpcTransportBuilder};
// #[cfg(feature = "http")]
// use velo_transports::http::{HttpTransport, HttpTransportBuilder};
// #[cfg(feature = "nats")]
// use velo_transports::nats::{NatsTransport, NatsTransportBuilder};

use bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::timeout;
use velo_transports::{
    DataStreams, InstanceId, MessageType, PeerInfo, Transport, TransportErrorHandler,
    tcp::{TcpTransport, TcpTransportBuilder},
};

#[cfg(unix)]
use velo_transports::uds::{UdsTransport, UdsTransportBuilder};

use std::sync::Once;
use tracing_subscriber::FmtSubscriber;

#[allow(dead_code)]
static INIT: Once = Once::new();

#[allow(dead_code)]
pub fn init_tracing() {
    INIT.call_once(|| {
        let _ = FmtSubscriber::builder()
            .with_env_filter("trace") // or "info"
            .try_init();
    });
}

pub mod scenarios;
pub mod shutdown_scenarios;

/// Test error handler that tracks errors for verification
#[derive(Clone)]
pub struct TestErrorHandler {
    errors: Arc<Mutex<Vec<(Bytes, Bytes, String)>>>,
}

impl TestErrorHandler {
    pub fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_errors(&self) -> Vec<(Bytes, Bytes, String)> {
        self.errors.lock().unwrap().clone()
    }

    pub fn error_count(&self) -> usize {
        self.errors.lock().unwrap().len()
    }

    pub fn clear(&self) {
        self.errors.lock().unwrap().clear();
    }
}

impl TransportErrorHandler for TestErrorHandler {
    fn on_error(&self, header: Bytes, payload: Bytes, error: String) {
        self.errors.lock().unwrap().push((header, payload, error));
    }
}

/// Handle to a transport instance with its streams for testing
///
/// This is a generic test handle that works with any transport implementation.
/// Use `TestTransportHandle::with_factory()` to create instances with custom transports,
/// or use convenience methods like `TestTransportHandle::new()` for TCP transport.
pub struct TestTransportHandle<T: Transport> {
    pub transport: T,
    pub streams: DataStreams,
    pub instance_id: InstanceId,
    pub error_handler: Arc<TestErrorHandler>,
    runtime: tokio::runtime::Handle,
}

impl<T: Transport> TestTransportHandle<T> {
    /// Create a new test transport using a factory function
    ///
    /// This is the generic constructor that works with any transport implementation.
    /// The factory function should create and return a transport instance.
    ///
    /// # Example
    /// ```ignore
    /// let handle = TestTransportHandle::with_factory(|| {
    ///     MyTransportBuilder::new().build()
    /// }).await?;
    /// ```
    pub async fn with_factory<F>(factory: F) -> anyhow::Result<Self>
    where
        F: FnOnce() -> anyhow::Result<T>,
    {
        let transport = factory()?;
        let instance_id = InstanceId::new_v4();
        let error_handler = Arc::new(TestErrorHandler::new());

        // Create channels for this transport
        let (adapter, streams) = velo_transports::make_channels();

        // Get runtime handle
        let runtime = tokio::runtime::Handle::current();

        // Start the transport
        transport
            .start(instance_id, adapter, runtime.clone())
            .await?;

        // Give the listener a moment to bind and start accepting connections
        tokio::time::sleep(Duration::from_millis(50)).await;

        Ok(Self {
            transport,
            streams,
            instance_id,
            error_handler,
            runtime,
        })
    }

    /// Register another transport as a peer
    pub fn register_peer<U: Transport>(
        &self,
        other: &TestTransportHandle<U>,
    ) -> anyhow::Result<()> {
        let peer_info = PeerInfo::new(other.instance_id, other.transport.address());
        self.transport
            .register(peer_info)
            .map_err(|e| anyhow::anyhow!("Failed to register peer: {:?}", e))?;
        Ok(())
    }

    /// Send a message to a peer
    pub fn send(
        &self,
        target: InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        msg_type: MessageType,
    ) {
        self.transport.send_message(
            target,
            header,
            payload,
            msg_type,
            self.error_handler.clone(),
        );
    }

    /// Receive a message with timeout
    pub async fn recv_message(&self, timeout_duration: Duration) -> anyhow::Result<(Bytes, Bytes)> {
        timeout(timeout_duration, self.streams.message_stream.recv_async())
            .await
            .map_err(|_| anyhow::anyhow!("Timeout waiting for message"))?
            .map_err(|e| anyhow::anyhow!("Channel error: {}", e))
    }

    /// Receive a response with timeout
    pub async fn recv_response(
        &self,
        timeout_duration: Duration,
    ) -> anyhow::Result<(Bytes, Bytes)> {
        timeout(timeout_duration, self.streams.response_stream.recv_async())
            .await
            .map_err(|_| anyhow::anyhow!("Timeout waiting for response"))?
            .map_err(|e| anyhow::anyhow!("Channel error: {}", e))
    }

    /// Receive an event with timeout
    pub async fn recv_event(&self, timeout_duration: Duration) -> anyhow::Result<(Bytes, Bytes)> {
        timeout(timeout_duration, self.streams.event_stream.recv_async())
            .await
            .map_err(|_| anyhow::anyhow!("Timeout waiting for event"))?
            .map_err(|e| anyhow::anyhow!("Channel error: {}", e))
    }

    /// Collect multiple messages with timeout
    pub async fn collect_messages(
        &self,
        count: usize,
        timeout_duration: Duration,
    ) -> anyhow::Result<Vec<(Bytes, Bytes)>> {
        let mut messages = Vec::new();
        for _ in 0..count {
            messages.push(self.recv_message(timeout_duration).await?);
        }
        Ok(messages)
    }

    /// Collect multiple messages with timeout, sorted by header for order-independent comparison
    ///
    /// This is useful for testing transports that don't guarantee delivery order (e.g., HTTP).
    /// Messages are sorted by header bytes to enable deterministic comparison regardless of
    /// delivery order.
    pub async fn collect_messages_unordered(
        &self,
        count: usize,
        timeout_duration: Duration,
    ) -> anyhow::Result<Vec<(Bytes, Bytes)>> {
        let mut messages = self.collect_messages(count, timeout_duration).await?;
        messages.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(messages)
    }

    /// Collect multiple responses with timeout
    pub async fn collect_responses(
        &self,
        count: usize,
        timeout_duration: Duration,
    ) -> anyhow::Result<Vec<(Bytes, Bytes)>> {
        let mut responses = Vec::new();
        for _ in 0..count {
            responses.push(self.recv_response(timeout_duration).await?);
        }
        Ok(responses)
    }

    /// Shutdown the transport
    pub fn shutdown(self) {
        self.transport.shutdown();
    }
}

// TCP-specific convenience constructors
impl TestTransportHandle<TcpTransport> {
    /// Create a new TCP transport on a random available port
    ///
    /// This is a convenience method for creating TCP transports.
    /// For other transport types, use `with_factory()`.
    pub async fn new_tcp() -> anyhow::Result<Self> {
        Self::with_factory(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            TcpTransportBuilder::new().from_listener(listener)?.build()
        })
        .await
    }

    /// Alias for `new_tcp()` to maintain backward compatibility
    pub async fn new() -> anyhow::Result<Self> {
        Self::new_tcp().await
    }
}

// UDS-specific convenience constructors
#[cfg(unix)]
impl TestTransportHandle<UdsTransport> {
    /// Create a new UDS transport using a temp directory socket path
    pub async fn new_uds() -> anyhow::Result<Self> {
        Self::with_factory(|| {
            let dir = std::env::temp_dir().join(format!(
                "velo-uds-test-{}",
                velo_transports::InstanceId::new_v4()
            ));
            std::fs::create_dir_all(&dir)?;
            let socket_path = dir.join("transport.sock");
            UdsTransportBuilder::new().socket_path(&socket_path).build()
        })
        .await
    }
}

// // UCX-specific convenience constructors
// #[cfg(feature = "ucx")]
// impl TestTransportHandle<UcxTransport> {
//     /// Create a new UCX transport
//     ///
//     /// This is a convenience method for creating UCX transports.
//     /// For other transport types, use `with_factory()`.
//     pub async fn new_ucx() -> anyhow::Result<Self> {
//         Self::with_factory(|| UcxTransportBuilder::new().build()).await
//     }
// }

// // HTTP-specific convenience constructors
// #[cfg(feature = "http")]
// impl TestTransportHandle<HttpTransport> {
//     /// Create a new HTTP transport with OS-provided port
//     ///
//     /// This is a convenience method for creating HTTP transports.
//     /// For other transport types, use `with_factory()`.
//     pub async fn new_http() -> anyhow::Result<Self> {
//         Self::with_factory(|| {
//             // Use default builder which binds to 0.0.0.0:0 (OS-provided port)
//             HttpTransportBuilder::new().build()
//         })
//         .await
//     }
// }

// // NATS-specific convenience constructor
// #[cfg(feature = "nats")]
// impl TestTransportHandle<NatsTransport> {
//     /// Create a new NATS transport
//     ///
//     /// This is a convenience method for creating NATS transports.
//     /// For other transport types, use `with_factory()`.
//     ///
//     /// Note: NATS transport requires special handling because it needs the instance_id
//     /// at construction time to set up subject subscriptions. We can't use the generic
//     /// with_factory() because it creates the instance_id AFTER calling the factory.
//     pub async fn new_nats() -> anyhow::Result<Self> {
//         // Create instance_id
//         let instance_id = InstanceId::new_v4();
//         let error_handler = Arc::new(TestErrorHandler::new());

//         // Build transport
//         let transport = NatsTransportBuilder::new()
//             .nats_url("nats://127.0.0.1:4222")
//             .build()?;

//         // Create channels for this transport
//         let (adapter, streams) = velo_transports::make_channels();

//         // Get runtime handle
//         let runtime = tokio::runtime::Handle::current();

//         // Start the transport
//         transport
//             .start(instance_id, adapter, runtime.clone())
//             .await?;

//         // Give NATS a moment to establish subscriptions
//         tokio::time::sleep(Duration::from_millis(50)).await;

//         Ok(Self {
//             transport,
//             streams,
//             instance_id,
//             error_handler,
//             runtime,
//         })
//     }
// }

// gRPC-specific convenience constructors
#[cfg(feature = "grpc")]
impl TestTransportHandle<GrpcTransport> {
    /// Create a new gRPC transport with OS-provided port
    pub async fn new_grpc() -> anyhow::Result<Self> {
        Self::with_factory(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            GrpcTransportBuilder::new().from_listener(listener)?.build()
        })
        .await
    }
}

/// Multi-transport test cluster
///
/// A generic cluster that works with any transport implementation.
/// All transports in the cluster are registered with each other in a full mesh topology.
pub struct TestCluster<T: Transport> {
    transports: Vec<TestTransportHandle<T>>,
}

impl<T: Transport> TestCluster<T> {
    /// Create a new test cluster using a factory function
    ///
    /// This is the generic constructor that works with any transport implementation.
    /// The factory function will be called `size` times to create each transport.
    ///
    /// # Example
    /// ```ignore
    /// let cluster = TestCluster::with_factory(3, || {
    ///     MyTransportBuilder::new().build()
    /// }).await?;
    /// ```
    pub async fn with_factory<F>(size: usize, factory: F) -> anyhow::Result<Self>
    where
        F: Fn() -> anyhow::Result<T>,
    {
        let mut transports = Vec::new();

        for _ in 0..size {
            transports.push(TestTransportHandle::with_factory(&factory).await?);
        }

        // Register all peers with each other (full mesh)
        for i in 0..transports.len() {
            for j in 0..transports.len() {
                if i != j {
                    transports[i].register_peer(&transports[j])?;
                }
            }
        }

        Ok(Self { transports })
    }

    /// Get a transport by index
    pub fn get(&self, index: usize) -> &TestTransportHandle<T> {
        &self.transports[index]
    }

    /// Get all transports
    pub fn all(&self) -> &[TestTransportHandle<T>] {
        &self.transports
    }

    /// Shutdown all transports
    pub fn shutdown(self) {
        for transport in self.transports {
            transport.shutdown();
        }
    }
}

// TCP-specific convenience constructor
impl TestCluster<TcpTransport> {
    /// Create a new TCP test cluster with the specified number of transports
    ///
    /// This is a convenience method for creating TCP clusters.
    /// For other transport types, use `with_factory()`.
    pub async fn new(size: usize) -> anyhow::Result<Self> {
        Self::with_factory(size, || {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            TcpTransportBuilder::new().from_listener(listener)?.build()
        })
        .await
    }
}

// UDS-specific convenience constructor
#[cfg(unix)]
impl TestCluster<UdsTransport> {
    /// Create a new UDS test cluster with the specified number of transports
    pub async fn new_uds(size: usize) -> anyhow::Result<Self> {
        Self::with_factory(size, || {
            let dir = std::env::temp_dir().join(format!(
                "velo-uds-test-{}",
                velo_transports::InstanceId::new_v4()
            ));
            std::fs::create_dir_all(&dir)?;
            let socket_path = dir.join("transport.sock");
            UdsTransportBuilder::new().socket_path(&socket_path).build()
        })
        .await
    }
}

// // HTTP-specific convenience constructor
// #[cfg(feature = "http")]
// impl TestCluster<HttpTransport> {
//     /// Create a new HTTP test cluster with the specified number of transports
//     ///
//     /// This is a convenience method for creating HTTP clusters.
//     /// For other transport types, use `with_factory()`.
//     pub async fn new_http(size: usize) -> anyhow::Result<Self> {
//         Self::with_factory(size, || {
//             // Use default builder which binds to OS-provided ports
//             HttpTransportBuilder::new().build()
//         })
//         .await
//     }
// }

// // NATS-specific convenience constructor
// #[cfg(feature = "nats")]
// impl TestCluster<NatsTransport> {
//     /// Create a new NATS test cluster with the specified number of transports
//     ///
//     /// This is a convenience method for creating NATS clusters.
//     /// For other transport types, use `with_factory()`.
//     ///
//     /// Note: NATS transport requires special handling because it needs the instance_id
//     /// at construction time. We can't use the generic with_factory() which creates
//     /// instance_id after calling the factory function.
//     pub async fn new_nats(size: usize) -> anyhow::Result<Self> {
//         let mut transports = Vec::new();

//         for _ in 0..size {
//             transports.push(TestTransportHandle::new_nats().await?);
//         }

//         // Register all peers with each other (full mesh)
//         for i in 0..transports.len() {
//             for j in 0..transports.len() {
//                 if i != j {
//                     transports[i].register_peer(&transports[j])?;
//                 }
//             }
//         }

//         Ok(Self { transports })
//     }
// }

// gRPC-specific convenience constructor
#[cfg(feature = "grpc")]
impl TestCluster<GrpcTransport> {
    /// Create a new gRPC test cluster with the specified number of transports
    pub async fn new_grpc(size: usize) -> anyhow::Result<Self> {
        Self::with_factory(size, || {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            GrpcTransportBuilder::new().from_listener(listener)?.build()
        })
        .await
    }
}

// Helper utilities

/// Get a random available port
pub fn get_random_port() -> u16 {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Create test data with the specified size
pub fn test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Create a test message with predictable content
pub fn test_message(id: u32) -> (Vec<u8>, Vec<u8>) {
    let header = format!("header-{}", id).into_bytes();
    let payload = format!("payload-{}", id).into_bytes();
    (header, payload)
}

/// Assert that a received message matches expected values
pub fn assert_message_eq(
    received: (Bytes, Bytes),
    expected_header: &[u8],
    expected_payload: &[u8],
) {
    assert_eq!(received.0.as_ref(), expected_header, "Header mismatch");
    assert_eq!(received.1.as_ref(), expected_payload, "Payload mismatch");
}

// ---------------------------------------------------------------------------
// ShutdownTestClient trait + implementations
// ---------------------------------------------------------------------------

/// Trait abstracting over transport-specific shutdown test operations.
///
/// This allows shutdown tests to be written generically and instantiated
/// for TCP, UDS, etc. via the `transport_shutdown_tests!` macro.
pub trait ShutdownTestClient {
    type Transport: Transport;
    type Stream: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send;

    /// Create a new transport handle for testing.
    fn new_handle()
    -> impl std::future::Future<Output = anyhow::Result<TestTransportHandle<Self::Transport>>> + Send;

    /// Connect a raw client to the transport and send one frame. Returns the stream.
    fn connect_and_send_frame(
        handle: &TestTransportHandle<Self::Transport>,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> impl std::future::Future<Output = Self::Stream> + Send;

    /// Read one frame from the raw stream.
    fn read_one_frame(
        stream: &mut Self::Stream,
    ) -> impl std::future::Future<Output = (MessageType, Bytes, Bytes)> + Send;
}

/// TCP shutdown test client
pub struct TcpShutdownClient;

impl ShutdownTestClient for TcpShutdownClient {
    type Transport = TcpTransport;
    type Stream = tokio::net::TcpStream;

    async fn new_handle() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_tcp().await
    }

    async fn connect_and_send_frame(
        handle: &TestTransportHandle<Self::Transport>,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> Self::Stream {
        use velo_transports::tcp::TcpFrameCodec;

        let addr = {
            let wa = handle.transport.address();
            let key = handle.transport.key();
            let endpoint = wa.get_entry(&key).unwrap().unwrap();
            let s = std::str::from_utf8(&endpoint).unwrap();
            let s = s.strip_prefix("tcp://").unwrap_or(s);
            s.parse::<std::net::SocketAddr>().unwrap()
        };
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        TcpFrameCodec::encode_frame(&mut stream, msg_type, header, payload)
            .await
            .unwrap();
        stream
    }

    async fn read_one_frame(stream: &mut Self::Stream) -> (MessageType, Bytes, Bytes) {
        use futures::StreamExt;
        use tokio_util::codec::Framed;
        use velo_transports::tcp::TcpFrameCodec;

        let mut framed = Framed::new(stream, TcpFrameCodec::new());
        framed.next().await.unwrap().unwrap()
    }
}

/// UDS shutdown test client
#[cfg(unix)]
pub struct UdsShutdownClient;

#[cfg(unix)]
impl ShutdownTestClient for UdsShutdownClient {
    type Transport = UdsTransport;
    type Stream = tokio::net::UnixStream;

    async fn new_handle() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_uds().await
    }

    async fn connect_and_send_frame(
        handle: &TestTransportHandle<Self::Transport>,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> Self::Stream {
        use velo_transports::tcp::TcpFrameCodec;

        let socket_path = {
            let wa = handle.transport.address();
            let key = handle.transport.key();
            let endpoint = wa.get_entry(&key).unwrap().unwrap();
            let s = std::str::from_utf8(&endpoint).unwrap();
            let s = s.strip_prefix("uds://").unwrap_or(s);
            std::path::PathBuf::from(s)
        };
        let mut stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        TcpFrameCodec::encode_frame(&mut stream, msg_type, header, payload)
            .await
            .unwrap();
        stream
    }

    async fn read_one_frame(stream: &mut Self::Stream) -> (MessageType, Bytes, Bytes) {
        use futures::StreamExt;
        use tokio_util::codec::Framed;
        use velo_transports::tcp::TcpFrameCodec;

        let mut framed = Framed::new(stream, TcpFrameCodec::new());
        framed.next().await.unwrap().unwrap()
    }
}

/// gRPC shutdown test client
///
/// gRPC shutdown tests connect via raw TCP to the tonic server's bind address
/// and send framed data directly. The tonic server's connection handler
/// uses the same frame codec, so raw TCP connections work for testing.
#[cfg(feature = "grpc")]
pub struct GrpcShutdownClient;

#[cfg(feature = "grpc")]
impl ShutdownTestClient for GrpcShutdownClient {
    type Transport = GrpcTransport;
    type Stream = tokio::net::TcpStream;

    async fn new_handle() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_grpc().await
    }

    async fn connect_and_send_frame(
        handle: &TestTransportHandle<Self::Transport>,
        msg_type: MessageType,
        header: &[u8],
        payload: &[u8],
    ) -> Self::Stream {
        use velo_transports::tcp::TcpFrameCodec;

        let addr = {
            let wa = handle.transport.address();
            let key = handle.transport.key();
            let endpoint = wa.get_entry(&key).unwrap().unwrap();
            let s = std::str::from_utf8(&endpoint).unwrap();
            let s = s.strip_prefix("grpc://").unwrap_or(s);
            s.parse::<std::net::SocketAddr>().unwrap()
        };
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        TcpFrameCodec::encode_frame(&mut stream, msg_type, header, payload)
            .await
            .unwrap();
        stream
    }

    async fn read_one_frame(stream: &mut Self::Stream) -> (MessageType, Bytes, Bytes) {
        use futures::StreamExt;
        use tokio_util::codec::Framed;
        use velo_transports::tcp::TcpFrameCodec;

        let mut framed = Framed::new(stream, TcpFrameCodec::new());
        framed.next().await.unwrap().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Test generation macros
// ---------------------------------------------------------------------------

/// Macro to generate integration tests for a transport factory.
///
/// This eliminates the boilerplate of writing individual `#[tokio::test]` functions
/// for each scenario when the only difference is the factory type parameter.
#[allow(unused_macros)]
macro_rules! transport_integration_tests {
    ($factory:ty) => {
        paste::paste! {
            #[tokio::test]
            async fn test_single_message_round_trip() {
                scenarios::single_message_round_trip::<$factory>().await;
            }
            #[tokio::test]
            async fn test_bidirectional_messaging() {
                scenarios::bidirectional_messaging::<$factory>().await;
            }
            #[tokio::test]
            async fn test_multiple_messages_same_connection() {
                scenarios::multiple_messages_same_connection::<$factory>().await;
            }
            #[tokio::test]
            async fn test_response_message_type() {
                scenarios::response_message_type::<$factory>().await;
            }
            #[tokio::test]
            async fn test_event_message_type() {
                scenarios::event_message_type::<$factory>().await;
            }
            #[tokio::test]
            async fn test_ack_message_type() {
                scenarios::ack_message_type::<$factory>().await;
            }
            #[tokio::test]
            async fn test_mixed_message_types() {
                scenarios::mixed_message_types::<$factory>().await;
            }
            #[tokio::test]
            async fn test_large_payload() {
                scenarios::large_payload::<$factory>().await;
            }
            #[tokio::test]
            async fn test_empty_header_and_payload() {
                scenarios::empty_header_and_payload::<$factory>().await;
            }
            #[tokio::test]
            async fn test_cluster_mesh_communication() {
                scenarios::cluster_mesh_communication::<$factory>().await;
            }
            #[tokio::test]
            async fn test_concurrent_senders() {
                scenarios::concurrent_senders::<$factory>().await;
            }
            #[tokio::test]
            async fn test_send_to_unregistered_peer() {
                scenarios::send_to_unregistered_peer::<$factory>().await;
            }
            #[tokio::test]
            async fn test_connection_reuse() {
                scenarios::connection_reuse::<$factory>().await;
            }
            #[tokio::test]
            async fn test_graceful_shutdown() {
                scenarios::graceful_shutdown::<$factory>().await;
            }
            #[tokio::test]
            async fn test_high_throughput() {
                scenarios::high_throughput::<$factory>().await;
            }
            #[tokio::test]
            async fn test_zero_copy_efficiency() {
                scenarios::zero_copy_efficiency::<$factory>().await;
            }
            #[tokio::test]
            async fn test_drain_rejects_messages() {
                scenarios::drain_rejects_messages::<$factory>().await;
            }
            #[tokio::test]
            async fn test_drain_accepts_responses() {
                scenarios::drain_accepts_responses::<$factory>().await;
            }
            #[tokio::test]
            async fn test_drain_accepts_events() {
                scenarios::drain_accepts_events::<$factory>().await;
            }
            #[tokio::test]
            async fn test_health_during_drain() {
                scenarios::health_during_drain::<$factory>().await;
            }
        }
    };
}

/// Macro to generate shutdown tests for a transport.
///
/// Generates tests with names like `test_{prefix}_drain_rejects_messages`.
#[allow(unused_macros)]
macro_rules! transport_shutdown_tests {
    ($prefix:ident, $client:ty) => {
        paste::paste! {
            #[tokio::test]
            async fn [<test_ $prefix _drain_rejects_messages>]() {
                shutdown_scenarios::drain_rejects_messages::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _drain_accepts_responses>]() {
                shutdown_scenarios::drain_accepts_responses::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _drain_accepts_events>]() {
                shutdown_scenarios::drain_accepts_events::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _new_connection_during_drain>]() {
                shutdown_scenarios::new_connection_during_drain::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _graceful_shutdown_lifecycle>]() {
                shutdown_scenarios::graceful_shutdown_lifecycle::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _shutdown_timeout_forces_teardown>]() {
                shutdown_scenarios::shutdown_timeout_forces_teardown::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _outbound_sends_during_drain>]() {
                shutdown_scenarios::outbound_sends_during_drain::<$client>().await;
            }
            #[tokio::test]
            async fn [<test_ $prefix _connection_writer_exits_on_teardown>]() {
                shutdown_scenarios::connection_writer_exits_on_teardown::<$client>().await;
            }
        }
    };
}

// Transport factory abstraction for parameterized tests

/// Transport factory trait for creating transports in parameterized tests
pub trait TransportFactory {
    type Transport: Transport;

    async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>>;
    async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>>;
}

/// TCP transport factory
pub struct TcpFactory;

impl TransportFactory for TcpFactory {
    type Transport = TcpTransport;

    async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_tcp().await
    }

    async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
        TestCluster::new(size).await
    }
}

/// UDS transport factory
#[cfg(unix)]
pub struct UdsFactory;

#[cfg(unix)]
impl TransportFactory for UdsFactory {
    type Transport = UdsTransport;

    async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_uds().await
    }

    async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
        TestCluster::new_uds(size).await
    }
}

// /// UCX transport factory
// #[cfg(feature = "ucx")]
// pub struct UcxFactory;

// #[cfg(feature = "ucx")]
// impl TransportFactory for UcxFactory {
//     type Transport = UcxTransport;

//     async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
//         TestTransportHandle::new_ucx().await
//     }

//     async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
//         TestCluster::new_ucx(size).await
//     }
// }

// /// HTTP transport factory
// #[cfg(feature = "http")]
// pub struct HttpFactory;

// #[cfg(feature = "http")]
// impl TransportFactory for HttpFactory {
//     type Transport = HttpTransport;

//     async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
//         TestTransportHandle::new_http().await
//     }

//     async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
//         TestCluster::new_http(size).await
//     }
// }

// /// NATS transport factory
// #[cfg(feature = "nats")]
// pub struct NatsFactory;

// #[cfg(feature = "nats")]
// impl TransportFactory for NatsFactory {
//     type Transport = NatsTransport;

//     async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
//         TestTransportHandle::new_nats().await
//     }

//     async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
//         TestCluster::new_nats(size).await
//     }
// }

/// gRPC transport factory
#[cfg(feature = "grpc")]
pub struct GrpcFactory;

#[cfg(feature = "grpc")]
impl TransportFactory for GrpcFactory {
    type Transport = GrpcTransport;

    async fn create() -> anyhow::Result<TestTransportHandle<Self::Transport>> {
        TestTransportHandle::new_grpc().await
    }

    async fn create_cluster(size: usize) -> anyhow::Result<TestCluster<Self::Transport>> {
        TestCluster::new_grpc(size).await
    }
}
