// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP-based [`FrameTransport`] implementation for dedicated per-stream connections.
//!
//! [`TcpFrameTransport`] gives each stream its own TCP connection with a
//! token-based handshake for validation. This bypasses the AM (Active Message)
//! fire-and-forget overhead of [`crate::velo_transport::VeloFrameTransport`],
//! providing dedicated bandwidth and flow control per stream.
//!
//! # Endpoint Format
//!
//! ```text
//! tcp://{bind_addr}:{port}/{uuid_token}
//! ```
//!
//! # Connection Lifecycle
//!
//! 1. **bind()**: Opens a `TcpListener` on an ephemeral port, generates a UUID
//!    token, spawns an accept-once task, and returns the endpoint + receiver.
//! 2. **connect()**: Parses the endpoint, connects to the listener, sends the
//!    16-byte token for validation, and spawns a pump task that bridges the
//!    flume channel to the TCP stream using `TcpFrameCodec`.
//! 3. **Accept-once**: The listener validates the token, accepts exactly one
//!    connection, then closes. Invalid tokens are rejected; the listener
//!    continues waiting (up to a 30-second timeout).
//!
//! # Socket Configuration
//!
//! All sockets are configured with TCP_NODELAY, keepalive (60s time / 10s
//! interval), and 1MB send/recv buffers, matching the `TcpTransport` pattern
//! from `velo-transports`.

use std::net::IpAddr;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::Framed;
use uuid::Uuid;
use velo_transports::tcp::TcpFrameCodec;
use velo_transports::MessageType;

use crate::transport::FrameTransport;

/// Timeout for the accept-once listener to receive a valid connection.
const ACCEPT_TIMEOUT: Duration = Duration::from_secs(30);

/// TCP-based [`FrameTransport`] providing dedicated per-stream connections.
///
/// Each `bind()` call creates an independent TCP listener on an ephemeral port
/// with a unique UUID token. The corresponding `connect()` establishes the TCP
/// connection, validates via the token handshake, and returns a `flume::Sender`
/// for frame delivery.
pub struct TcpFrameTransport {
    bind_addr: IpAddr,
}

impl TcpFrameTransport {
    /// Create a new `TcpFrameTransport` with the given bind address.
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - IP address to bind TCP listeners on. Use `0.0.0.0` for
    ///   all interfaces or a specific address for targeted binding.
    pub fn new(bind_addr: IpAddr) -> Self {
        Self { bind_addr }
    }
}

impl Default for TcpFrameTransport {
    fn default() -> Self {
        Self {
            bind_addr: IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
        }
    }
}

/// Configure TCP socket options matching TcpTransport patterns.
///
/// Sets TCP_NODELAY, keepalive (60s time / 10s interval), and 1MB send/recv
/// buffers on both accept-side and connect-side sockets. Errors are logged
/// as warnings (non-fatal) since socket options are best-effort optimizations.
fn configure_socket(stream: &TcpStream) {
    if let Err(e) = stream.set_nodelay(true) {
        tracing::warn!("Failed to set TCP_NODELAY: {}", e);
    }
    let sock = socket2::SockRef::from(stream);
    if let Err(e) = sock.set_tcp_keepalive(
        &socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10)),
    ) {
        tracing::warn!("Failed to set TCP keepalive: {}", e);
    }
    if let Err(e) = sock.set_send_buffer_size(1_048_576) {
        tracing::warn!("Failed to set send buffer size: {}", e);
    }
    if let Err(e) = sock.set_recv_buffer_size(1_048_576) {
        tracing::warn!("Failed to set recv buffer size: {}", e);
    }
}

/// Parse a `tcp://` endpoint into `(SocketAddr, Uuid)`.
///
/// Expected format: `tcp://{host}:{port}/{uuid_token}`
///
/// # Errors
///
/// Returns `Err` on malformed URIs (missing prefix, invalid address, invalid UUID).
///
/// # Examples
///
/// ```
/// # use velo_streaming::tcp_transport::parse_tcp_endpoint;
/// let (addr, token) = parse_tcp_endpoint("tcp://127.0.0.1:8080/550e8400-e29b-41d4-a716-446655440000").unwrap();
/// assert_eq!(addr.port(), 8080);
/// ```
pub fn parse_tcp_endpoint(endpoint: &str) -> Result<(std::net::SocketAddr, Uuid)> {
    let stripped = endpoint
        .strip_prefix("tcp://")
        .ok_or_else(|| anyhow::anyhow!("missing tcp:// prefix: {}", endpoint))?;
    let slash_pos = stripped
        .rfind('/')
        .ok_or_else(|| anyhow::anyhow!("missing token in endpoint: {}", endpoint))?;
    let addr_str = &stripped[..slash_pos];
    let token_str = &stripped[slash_pos + 1..];
    let addr: std::net::SocketAddr = addr_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address '{}': {}", addr_str, e))?;
    let token: Uuid = token_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid token '{}': {}", token_str, e))?;
    Ok((addr, token))
}

/// Check if the given raw frame bytes represent a terminal sentinel.
///
/// Terminal sentinels are Dropped, Detached, Finalized, and TransportError.
/// We compare against cached sentinel bytes for the common cases and fall
/// back to deserialization for TransportError (which carries a string payload).
fn is_terminal_sentinel(bytes: &[u8]) -> bool {
    use crate::sender::{cached_detached, cached_dropped, cached_finalized};

    if bytes == cached_dropped().as_slice()
        || bytes == cached_detached().as_slice()
        || bytes == cached_finalized().as_slice()
    {
        return true;
    }

    // TransportError carries a dynamic string, so we must try to deserialize
    if let Ok(frame) = rmp_serde::from_slice::<crate::frame::StreamFrame<()>>(bytes) {
        matches!(frame, crate::frame::StreamFrame::TransportError(_))
    } else {
        false
    }
}

impl FrameTransport for TcpFrameTransport {
    fn bind(&self, _anchor_id: u64, _session_id: u64) -> BoxFuture<'_, Result<(String, flume::Receiver<Vec<u8>>)>> {
        let bind_addr = self.bind_addr;
        Box::pin(async move {
            // Bind on ephemeral port (OS-assigned)
            let listener = TcpListener::bind((bind_addr, 0u16)).await?;
            let port = listener.local_addr()?.port();
            let token = Uuid::new_v4();

            // Frame delivery channel (same capacity as VeloFrameTransport)
            let (frame_tx, frame_rx) = flume::bounded::<Vec<u8>>(256);

            // Construct endpoint before spawning so it's ready immediately
            let endpoint = format!("tcp://{}:{}/{}", bind_addr, port, token);

            // Spawn accept-once task
            let expected_token = token;
            tokio::spawn(async move {
                let accept_result = tokio::time::timeout(ACCEPT_TIMEOUT, async {
                    loop {
                        match listener.accept().await {
                            Ok((mut stream, _addr)) => {
                                // Read 16-byte token
                                let mut token_buf = [0u8; 16];
                                if let Err(e) =
                                    tokio::io::AsyncReadExt::read_exact(&mut stream, &mut token_buf)
                                        .await
                                {
                                    tracing::warn!(
                                        "TCP streaming handshake read failed: {}",
                                        e
                                    );
                                    // Connection dropped before sending token; continue waiting
                                    continue;
                                }

                                let received_token = Uuid::from_bytes(token_buf);
                                if received_token != expected_token {
                                    tracing::warn!(
                                        "TCP streaming handshake: invalid token (expected {}, got {})",
                                        expected_token,
                                        received_token
                                    );
                                    // Reject and continue waiting for a valid connection
                                    drop(stream);
                                    continue;
                                }

                                // Valid connection -- configure socket and start reading
                                return Some(stream);
                            }
                            Err(e) => {
                                tracing::error!("TCP streaming accept error: {}", e);
                                return None;
                            }
                        }
                    }
                })
                .await;

                let stream = match accept_result {
                    Ok(Some(stream)) => stream,
                    Ok(None) => {
                        // Accept error -- drop frame_tx to close channel
                        tracing::warn!("TCP streaming: accept failed, closing channel");
                        return;
                    }
                    Err(_timeout) => {
                        // Timeout -- drop frame_tx to close channel
                        tracing::warn!("TCP streaming: accept timed out after {:?}", ACCEPT_TIMEOUT);
                        return;
                    }
                };

                // Configure socket options
                configure_socket(&stream);

                // Read frames via TcpFrameCodec and forward to frame_tx
                let mut framed = Framed::new(stream, TcpFrameCodec::new());
                let mut last_was_terminal = false;

                while let Some(result) = framed.next().await {
                    match result {
                        Ok((_msg_type, _header, payload)) => {
                            let payload_vec = payload.to_vec();
                            last_was_terminal = is_terminal_sentinel(&payload_vec);
                            if frame_tx.send_async(payload_vec).await.is_err() {
                                break; // consumer dropped
                            }
                        }
                        Err(e) => {
                            tracing::warn!("TCP streaming read error: {}", e);
                            break;
                        }
                    }
                }

                // If no terminal sentinel was the last frame, inject Dropped
                if !last_was_terminal {
                    let _ = frame_tx.send_async(crate::sender::cached_dropped().clone()).await;
                }
            });

            Ok((endpoint, frame_rx))
        })
    }

    fn connect(
        &self,
        endpoint: &str,
        _anchor_id: u64,
        _session_id: u64,
    ) -> BoxFuture<'_, Result<flume::Sender<Vec<u8>>>> {
        let endpoint = endpoint.to_string();
        Box::pin(async move {
            let (addr, token) = parse_tcp_endpoint(&endpoint)?;

            // Connect to the listener
            let stream = TcpStream::connect(addr).await?;

            // Configure socket options
            configure_socket(&stream);

            // Send 16-byte token for handshake validation
            let (_read_half, mut write_half) = tokio::io::split(stream);
            write_half.write_all(token.as_bytes()).await?;

            // Create channel for frame pump
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);

            // Spawn pump task: reads from channel, writes to TCP via TcpFrameCodec
            tokio::spawn(async move {
                while let Ok(frame_bytes) = rx.recv_async().await {
                    if let Err(e) = TcpFrameCodec::encode_frame(
                        &mut write_half,
                        MessageType::Message,
                        &[], // empty header for streaming
                        &frame_bytes,
                    )
                    .await
                    {
                        tracing::error!("TCP streaming write error: {}", e);
                        break;
                    }
                }

                // Sender-initiated close: flush and shutdown write half
                // This puts TIME_WAIT on the sender side (correct behavior)
                if let Err(e) = write_half.flush().await {
                    tracing::debug!("TCP streaming flush on close: {}", e);
                }
                if let Err(e) = write_half.shutdown().await {
                    tracing::debug!("TCP streaming shutdown on close: {}", e);
                }
            });

            Ok(tx)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bind_returns_tcp_endpoint() {
        let transport = TcpFrameTransport::default();
        let (endpoint, _rx) = transport.bind(1, 0).await.unwrap();

        // Verify endpoint format: tcp://addr:port/uuid
        assert!(endpoint.starts_with("tcp://"), "endpoint should start with tcp://: {}", endpoint);

        let (addr, token) = parse_tcp_endpoint(&endpoint).unwrap();
        assert!(addr.port() > 0, "port should be non-zero");
        assert_ne!(token, Uuid::nil(), "token should not be nil UUID");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_connect_handshake() {
        let transport = TcpFrameTransport::default();
        let (endpoint, rx) = transport.bind(1, 0).await.unwrap();

        // Connect and send some data
        let sender = transport.connect(&endpoint, 1, 1).await.unwrap();

        let frame_bytes = rmp_serde::to_vec(&crate::frame::StreamFrame::<String>::Item(
            "hello world".to_string(),
        ))
        .unwrap();

        sender.send_async(frame_bytes.clone()).await.unwrap();

        // Receive and verify
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .expect("timeout waiting for frame")
            .expect("channel closed unexpectedly");

        assert_eq!(received, frame_bytes, "received frame should match sent frame");

        // Drop sender to trigger cleanup
        drop(sender);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_invalid_token_rejected() {
        let transport = TcpFrameTransport::default();
        let (endpoint, rx) = transport.bind(1, 0).await.unwrap();

        // Parse endpoint to get addr but use a different token
        let (addr, _valid_token) = parse_tcp_endpoint(&endpoint).unwrap();

        // Connect manually with wrong token
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let wrong_token = Uuid::new_v4();
        stream.write_all(wrong_token.as_bytes()).await.unwrap();

        // The connection should be rejected. The listener should keep waiting
        // and eventually time out. We can't connect with the valid token after
        // the wrong one since the listener is still waiting. Instead, let's
        // verify that the wrong token doesn't produce frames.
        //
        // Give a short window to ensure no frames arrive from the wrong connection.
        let result = tokio::time::timeout(Duration::from_millis(500), rx.recv_async()).await;

        // Should timeout (no frames from invalid token connection)
        assert!(result.is_err(), "should not receive frames from invalid token connection");

        // Clean up -- write some data on the rejected stream and verify it's not forwarded
        drop(stream);
    }

    #[test]
    fn test_parse_tcp_endpoint_valid() {
        let endpoint = "tcp://127.0.0.1:8080/550e8400-e29b-41d4-a716-446655440000";
        let (addr, token) = parse_tcp_endpoint(endpoint).unwrap();
        assert_eq!(addr.port(), 8080);
        assert_eq!(
            token.to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn test_parse_tcp_endpoint_invalid_no_prefix() {
        assert!(parse_tcp_endpoint("http://127.0.0.1:8080/token").is_err());
    }

    #[test]
    fn test_parse_tcp_endpoint_invalid_no_token() {
        assert!(parse_tcp_endpoint("tcp://127.0.0.1:8080").is_err());
    }

    #[test]
    fn test_parse_tcp_endpoint_invalid_bad_uuid() {
        assert!(parse_tcp_endpoint("tcp://127.0.0.1:8080/not-a-uuid").is_err());
    }

    #[test]
    fn test_parse_tcp_endpoint_invalid_bad_addr() {
        assert!(parse_tcp_endpoint("tcp://not-an-addr/550e8400-e29b-41d4-a716-446655440000").is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_connect_round_trip_multiple_frames() {
        let transport = TcpFrameTransport::default();
        let (endpoint, rx) = transport.bind(42, 0).await.unwrap();

        let sender = transport.connect(&endpoint, 42, 1).await.unwrap();

        // Send multiple frames
        let mut expected_frames = Vec::new();
        for i in 0..10 {
            let frame = rmp_serde::to_vec(&crate::frame::StreamFrame::<u32>::Item(i)).unwrap();
            expected_frames.push(frame.clone());
            sender.send_async(frame).await.unwrap();
        }

        // Receive and verify order
        for (i, expected) in expected_frames.iter().enumerate() {
            let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
                .await
                .unwrap_or_else(|_| panic!("timeout waiting for frame {}", i))
                .unwrap_or_else(|_| panic!("channel closed at frame {}", i));
            assert_eq!(&received, expected, "frame {} mismatch", i);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dropped_sentinel_injected_on_abrupt_close() {
        let transport = TcpFrameTransport::default();
        let (endpoint, rx) = transport.bind(1, 0).await.unwrap();

        // Connect and send one frame, then drop sender without finalize
        let sender = transport.connect(&endpoint, 1, 1).await.unwrap();
        let frame = rmp_serde::to_vec(&crate::frame::StreamFrame::<String>::Item(
            "data".to_string(),
        ))
        .unwrap();
        sender.send_async(frame.clone()).await.unwrap();

        // Drop sender to trigger abrupt close (no terminal sentinel sent by user)
        drop(sender);

        // Should receive the data frame first
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(received, frame);

        // Then the injected Dropped sentinel
        let sentinel = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .expect("timeout waiting for Dropped sentinel")
            .expect("channel closed before Dropped sentinel");

        assert_eq!(
            sentinel.as_slice(),
            crate::sender::cached_dropped().as_slice(),
            "should receive Dropped sentinel after abrupt close"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_no_extra_dropped_after_finalized() {
        let transport = TcpFrameTransport::default();
        let (endpoint, rx) = transport.bind(1, 0).await.unwrap();

        let sender = transport.connect(&endpoint, 1, 1).await.unwrap();

        // Send a Finalized sentinel explicitly
        let finalized =
            rmp_serde::to_vec(&crate::frame::StreamFrame::<()>::Finalized).unwrap();
        sender.send_async(finalized.clone()).await.unwrap();

        // Drop sender
        drop(sender);

        // Should receive Finalized
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(received, finalized);

        // Should NOT receive an extra Dropped sentinel -- channel should close
        let result = tokio::time::timeout(Duration::from_secs(2), rx.recv_async()).await;
        match result {
            Ok(Ok(extra)) => {
                // If we get something, it should NOT be a Dropped sentinel
                assert_ne!(
                    extra.as_slice(),
                    crate::sender::cached_dropped().as_slice(),
                    "should not inject Dropped after Finalized"
                );
            }
            Ok(Err(_)) => {} // channel closed -- expected
            Err(_) => {}     // timeout -- also fine
        }
    }
}
