// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for VeloFrameTransport (XPRT-04).
//!
//! These tests use real TCP Messenger instances with loopback transport
//! to validate end-to-end frame delivery through the active message system.

use std::sync::Arc;
use std::time::Duration;

use velo_messenger::Messenger;
use velo_streaming::velo_transport::{VeloFrameTransport, parse_velo_uri};
use velo_streaming::FrameTransport;
use velo_transports::tcp::TcpTransportBuilder;

/// Create a TcpTransport bound to an OS-assigned port (no TOCTOU race).
fn new_transport() -> Arc<velo_transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

// ---------------------------------------------------------------------------
// Test 1: Two-Messenger round-trip
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_two_messenger_round_trip() {
    // Create two TCP transports and two Messengers (server, client).
    let server_transport = new_transport();
    let client_transport = new_transport();

    let server_messenger = Messenger::new(vec![server_transport], None)
        .await
        .expect("create server messenger");
    let client_messenger = Messenger::new(vec![client_transport], None)
        .await
        .expect("create client messenger");

    // Register peers (bidirectional).
    let server_peer = server_messenger.peer_info();
    let client_peer = client_messenger.peer_info();
    client_messenger
        .register_peer(server_peer.clone())
        .expect("register server peer on client");
    server_messenger
        .register_peer(client_peer.clone())
        .expect("register client peer on server");

    // Wait for TCP connections to establish.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let server_worker_id = server_messenger.instance_id().worker_id();
    let client_worker_id = client_messenger.instance_id().worker_id();

    // Build VeloFrameTransport on server (receiver) side.
    let server_vft = VeloFrameTransport::new(server_messenger.clone(), server_worker_id)
        .expect("create server VeloFrameTransport");

    // Bind anchor 42 on the server side.
    let (endpoint, rx) = server_vft.bind(42).await.expect("bind anchor 42");
    assert!(
        endpoint.starts_with("velo://"),
        "endpoint must start with velo://"
    );

    // Build VeloFrameTransport on client (sender) side.
    let client_vft = VeloFrameTransport::new(client_messenger.clone(), client_worker_id)
        .expect("create client VeloFrameTransport");

    // Connect from client to server's endpoint.
    let tx = client_vft
        .connect(&endpoint, 42, 1)
        .await
        .expect("connect to server");

    // Send 10 test frames.
    for i in 0..10 {
        let frame = format!("frame-{}", i).into_bytes();
        tx.send_async(frame).await.expect("send frame");
        // Small yield to let the pump task process sequentially
        tokio::task::yield_now().await;
    }

    // Receive 10 frames and verify content matches.
    let mut received_frames = Vec::new();
    for _ in 0..10 {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .expect("timeout waiting for frame")
            .expect("recv frame");
        received_frames.push(String::from_utf8(received).expect("valid utf8"));
    }

    // Verify all 10 frames arrived with correct content.
    let mut expected: Vec<String> = (0..10).map(|i| format!("frame-{}", i)).collect();
    let mut actual = received_frames.clone();
    expected.sort();
    actual.sort();
    assert_eq!(
        actual, expected,
        "all 10 frames must be received with correct content"
    );

    // Drop sender to close the channel.
    drop(tx);
}

// ---------------------------------------------------------------------------
// Test 2: URI parsing
// ---------------------------------------------------------------------------

#[test]
fn test_uri_parsing_valid() {
    let (wid, aid) = parse_velo_uri("velo://123/stream/456").unwrap();
    assert_eq!(wid, 123);
    assert_eq!(aid, 456);
}

#[test]
fn test_uri_parsing_missing_prefix() {
    assert!(parse_velo_uri("http://wrong").is_err());
}

#[test]
fn test_uri_parsing_non_numeric_worker() {
    assert!(parse_velo_uri("velo://abc/stream/456").is_err());
}

#[test]
fn test_uri_parsing_wrong_path_segment() {
    assert!(parse_velo_uri("velo://123/wrong/456").is_err());
}

#[test]
fn test_uri_parsing_non_numeric_anchor() {
    assert!(parse_velo_uri("velo://123/stream/xyz").is_err());
}

// ---------------------------------------------------------------------------
// Test 3: Sender drop closes pump
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_sender_drop_closes_pump() {
    // Two-Messenger setup.
    let server_transport = new_transport();
    let client_transport = new_transport();

    let server_messenger = Messenger::new(vec![server_transport], None)
        .await
        .expect("create server messenger");
    let client_messenger = Messenger::new(vec![client_transport], None)
        .await
        .expect("create client messenger");

    let server_peer = server_messenger.peer_info();
    let client_peer = client_messenger.peer_info();
    client_messenger
        .register_peer(server_peer.clone())
        .expect("register server peer on client");
    server_messenger
        .register_peer(client_peer.clone())
        .expect("register client peer on server");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let server_worker_id = server_messenger.instance_id().worker_id();
    let client_worker_id = client_messenger.instance_id().worker_id();

    let server_vft = VeloFrameTransport::new(server_messenger.clone(), server_worker_id)
        .expect("create server VeloFrameTransport");
    let (endpoint, rx) = server_vft.bind(100).await.expect("bind");

    let client_vft = VeloFrameTransport::new(client_messenger.clone(), client_worker_id)
        .expect("create client VeloFrameTransport");
    let tx = client_vft
        .connect(&endpoint, 100, 1)
        .await
        .expect("connect");

    // Drop the sender immediately.
    drop(tx);

    // The pump task should exit. After a short delay, the recv should fail
    // because no more data will arrive. We give some time for cleanup.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify no frames are received (channel should be quiet).
    // The transport-side receiver won't be closed by sender drop, but no frames arrive.
    let result = tokio::time::timeout(Duration::from_millis(200), rx.recv_async()).await;
    assert!(
        result.is_err(),
        "should timeout (no data sent before sender was dropped)"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Bind/unbind cleanup
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_bind_unbind_cleanup() {
    // Single Messenger is sufficient (no AM send needed, just dispatch map management).
    let transport = new_transport();
    let messenger = Messenger::new(vec![transport], None)
        .await
        .expect("create messenger");

    let worker_id = messenger.instance_id().worker_id();
    let vft = VeloFrameTransport::new(messenger.clone(), worker_id)
        .expect("create VeloFrameTransport");

    // bind(42) -- dispatch map has entry for 42.
    let (endpoint1, _rx1) = vft.bind(42).await.expect("bind 42");
    assert!(endpoint1.contains("/stream/42"));

    // unbind(42) -- dispatch map entry removed.
    vft.unbind(42);

    // Verify via another bind(42) succeeding (fresh channel).
    let (endpoint2, _rx2) = vft.bind(42).await.expect("re-bind 42");
    assert_eq!(endpoint1, endpoint2, "same URI after re-bind");
}
