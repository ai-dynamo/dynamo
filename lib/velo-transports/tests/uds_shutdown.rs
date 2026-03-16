// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for UDS graceful shutdown
//!
//! These tests verify the 3-phase shutdown behavior:
//! 1. Gate: new Message frames are rejected with ShuttingDown
//! 2. Drain: in-flight work completes, responses/events still flow
//! 3. Teardown: listener and writer tasks exit

#![cfg(unix)]

mod common;

use bytes::Bytes;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use velo_transports::tcp::TcpFrameCodec;
use velo_transports::uds::UdsTransport;
use velo_transports::{MessageType, Transport};

use common::TestTransportHandle;

/// Get the socket path from a UDS transport by parsing its WorkerAddress.
fn get_socket_path(handle: &TestTransportHandle<UdsTransport>) -> std::path::PathBuf {
    let addr = handle.transport.address();
    let key = handle.transport.key();
    let endpoint = addr.get_entry(&key).unwrap().unwrap();
    let s = std::str::from_utf8(&endpoint).unwrap();
    let s = s.strip_prefix("uds://").unwrap_or(s);
    std::path::PathBuf::from(s)
}

/// Helper: connect a raw UDS client to the transport's socket and send a frame.
async fn connect_and_send_frame(
    socket_path: &std::path::Path,
    msg_type: MessageType,
    header: &[u8],
    payload: &[u8],
) -> tokio::net::UnixStream {
    let mut stream = tokio::net::UnixStream::connect(socket_path).await.unwrap();
    TcpFrameCodec::encode_frame(&mut stream, msg_type, header, payload)
        .await
        .unwrap();
    stream
}

/// Helper: read one frame from a raw UDS stream.
async fn read_one_frame(stream: &mut tokio::net::UnixStream) -> (MessageType, Bytes, Bytes) {
    use futures::StreamExt;
    use tokio_util::codec::Framed;

    let mut framed = Framed::new(stream, TcpFrameCodec::new());
    framed.next().await.unwrap().unwrap()
}

// --- Test: Drain rejects Message frames ---
#[tokio::test]
async fn test_uds_drain_rejects_messages() {
    let handle = TestTransportHandle::new_uds().await.unwrap();
    let socket_path = get_socket_path(&handle);

    // Begin drain
    handle.streams.shutdown_state.begin_drain();

    // Give listener time to be ready
    sleep(Duration::from_millis(50)).await;

    // Connect and send a Message frame
    let mut stream = connect_and_send_frame(
        &socket_path,
        MessageType::Message,
        b"req-header",
        b"req-payload",
    )
    .await;

    // Should get ShuttingDown back
    let (msg_type, header, payload) = read_one_frame(&mut stream).await;
    assert_eq!(msg_type, MessageType::ShuttingDown);
    assert_eq!(&header[..], b"req-header"); // Original header echoed back
    assert_eq!(payload.len(), 0); // Empty payload

    handle.streams.shutdown_state.teardown_token().cancel();
}

// --- Test: Drain accepts Response frames ---
#[tokio::test]
async fn test_uds_drain_accepts_responses() {
    let handle = TestTransportHandle::new_uds().await.unwrap();
    let socket_path = get_socket_path(&handle);

    // Begin drain
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    // Connect and send a Response frame
    connect_and_send_frame(
        &socket_path,
        MessageType::Response,
        b"resp-header",
        b"resp-payload",
    )
    .await;

    // Should arrive on the response stream
    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    assert_eq!(&header[..], b"resp-header");
    assert_eq!(&payload[..], b"resp-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

// --- Test: Drain accepts Event frames ---
#[tokio::test]
async fn test_uds_drain_accepts_events() {
    let handle = TestTransportHandle::new_uds().await.unwrap();
    let socket_path = get_socket_path(&handle);

    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    connect_and_send_frame(
        &socket_path,
        MessageType::Event,
        b"evt-header",
        b"evt-payload",
    )
    .await;

    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.event_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    assert_eq!(&header[..], b"evt-header");
    assert_eq!(&payload[..], b"evt-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

// --- Test: New connection during drain still accepts responses ---
#[tokio::test]
async fn test_uds_new_connection_during_drain() {
    let handle = TestTransportHandle::new_uds().await.unwrap();
    let socket_path = get_socket_path(&handle);

    // Begin drain BEFORE connecting
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    // Establish a NEW connection after drain starts
    connect_and_send_frame(
        &socket_path,
        MessageType::Response,
        b"new-resp",
        b"new-payload",
    )
    .await;

    // Should arrive on the response stream
    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    assert_eq!(&header[..], b"new-resp");
    assert_eq!(&payload[..], b"new-payload");

    handle.streams.shutdown_state.teardown_token().cancel();
}

// --- Test: Full graceful shutdown lifecycle ---
#[tokio::test]
async fn test_uds_graceful_shutdown_lifecycle() {
    let handle = TestTransportHandle::new_uds().await.unwrap();
    let socket_path = get_socket_path(&handle);

    // Verify normal operation: send a message, receive it
    connect_and_send_frame(
        &socket_path,
        MessageType::Message,
        b"normal-msg",
        b"normal-pay",
    )
    .await;
    let (header, _payload) = timeout(
        Duration::from_secs(2),
        handle.streams.message_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"normal-msg");

    // Acquire an InFlightGuard (simulate in-progress request)
    let guard = handle.streams.shutdown_state.acquire();
    assert_eq!(handle.streams.shutdown_state.in_flight_count(), 1);

    // Begin drain (Phase 1)
    handle.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    // Verify new messages are rejected
    let mut stream =
        connect_and_send_frame(&socket_path, MessageType::Message, b"reject-me", b"").await;
    let (msg_type, _, _) = read_one_frame(&mut stream).await;
    assert_eq!(msg_type, MessageType::ShuttingDown);

    // Verify responses still flow
    connect_and_send_frame(&socket_path, MessageType::Response, b"still-ok", b"data").await;
    let (header, _) = timeout(
        Duration::from_secs(2),
        handle.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");
    assert_eq!(&header[..], b"still-ok");

    // Spawn graceful_shutdown in background (will block on drain since guard is held)
    let shutdown_state = handle.streams.shutdown_state.clone();
    let shutdown_handle = tokio::spawn(async move {
        // Phase 2: wait for drain
        shutdown_state.wait_for_drain().await;
        // Phase 3: teardown
        shutdown_state.teardown_token().cancel();
    });

    // Verify shutdown hasn't completed yet (guard still held)
    sleep(Duration::from_millis(100)).await;
    assert!(!shutdown_handle.is_finished());

    // Drop guard -> drain completes -> teardown fires
    drop(guard);

    timeout(Duration::from_secs(2), shutdown_handle)
        .await
        .expect("shutdown should complete")
        .unwrap();

    assert!(
        handle
            .streams
            .shutdown_state
            .teardown_token()
            .is_cancelled()
    );
}

// --- Test: Shutdown timeout forces teardown ---
#[tokio::test]
async fn test_uds_shutdown_timeout_forces_teardown() {
    let handle = TestTransportHandle::new_uds().await.unwrap();

    // Acquire guard and hold it
    let _guard = handle.streams.shutdown_state.acquire();

    let shutdown_state = handle.streams.shutdown_state.clone();
    let shutdown_handle = tokio::spawn(async move {
        shutdown_state.begin_drain();

        // Phase 2: wait with short timeout
        let _ =
            tokio::time::timeout(Duration::from_millis(100), shutdown_state.wait_for_drain()).await;

        // Phase 3: teardown (forced, guard still held)
        shutdown_state.teardown_token().cancel();
    });

    timeout(Duration::from_secs(2), shutdown_handle)
        .await
        .expect("shutdown should complete via timeout")
        .unwrap();

    // Teardown should have fired even though guard is held
    assert!(
        handle
            .streams
            .shutdown_state
            .teardown_token()
            .is_cancelled()
    );
    // Guard is still held (not a problem — teardown was forced)
    assert_eq!(handle.streams.shutdown_state.in_flight_count(), 1);
}

// --- Test: Outbound sends during drain ---
#[tokio::test]
async fn test_uds_outbound_sends_during_drain() {
    // Create two transports and register them as peers
    let handle_a = TestTransportHandle::new_uds().await.unwrap();
    let handle_b = TestTransportHandle::new_uds().await.unwrap();

    handle_a.register_peer(&handle_b).unwrap();
    handle_b.register_peer(&handle_a).unwrap();

    // Begin drain on transport A
    handle_a.streams.shutdown_state.begin_drain();
    sleep(Duration::from_millis(50)).await;

    // Send a Response from A to B (outbound sends should work during drain)
    handle_a.send(
        handle_b.instance_id,
        b"response-hdr".to_vec(),
        b"response-pay".to_vec(),
        MessageType::Response,
    );

    // B should receive the response
    let (header, payload) = timeout(
        Duration::from_secs(2),
        handle_b.streams.response_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    assert_eq!(&header[..], b"response-hdr");
    assert_eq!(&payload[..], b"response-pay");

    handle_a.streams.shutdown_state.teardown_token().cancel();
    handle_b.streams.shutdown_state.teardown_token().cancel();
}

// --- Test: Connection writer exits on teardown ---
#[tokio::test]
async fn test_uds_connection_writer_exits_on_teardown() {
    let handle_a = TestTransportHandle::new_uds().await.unwrap();
    let handle_b = TestTransportHandle::new_uds().await.unwrap();

    handle_a.register_peer(&handle_b).unwrap();

    // Send a message to establish the connection writer task
    handle_a.send(
        handle_b.instance_id,
        b"setup".to_vec(),
        b"data".to_vec(),
        MessageType::Message,
    );

    // Wait for it to arrive
    timeout(
        Duration::from_secs(2),
        handle_b.streams.message_stream.recv_async(),
    )
    .await
    .expect("timeout")
    .expect("recv");

    // Shutdown transport A
    handle_a.transport.shutdown();

    // Give writer tasks time to exit
    sleep(Duration::from_millis(200)).await;

    // Sending after shutdown: the cancel_token is already cancelled, so any new
    // writer task returns immediately without connecting. The error handler must
    // be invoked and the message must not arrive on handle_b.
    handle_a.error_handler.clear();
    handle_a.send(
        handle_b.instance_id,
        b"should-fail".to_vec(),
        b"data".to_vec(),
        MessageType::Message,
    );

    // Give time for the async error path to complete
    sleep(Duration::from_millis(100)).await;

    // Error handler must have been invoked for the failed send
    assert!(
        handle_a.error_handler.error_count() >= 1,
        "error handler should be invoked for post-shutdown send"
    );

    // The message must not have been delivered to handle_b
    let not_delivered = timeout(
        Duration::from_millis(100),
        handle_b.streams.message_stream.recv_async(),
    )
    .await;
    assert!(
        not_delivered.is_err(),
        "post-shutdown message must not arrive at handle_b"
    );

    handle_b.streams.shutdown_state.teardown_token().cancel();
}
