// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for TcpFrameTransport.
//!
//! Validates the full TCP streaming lifecycle: round-trip data flow,
//! connection drop handling, sender-initiated close, and transport
//! registry integration via VeloBuilder.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use velo_common::WorkerId;
use velo_messenger::Messenger;
use velo_streaming::{
    AnchorManagerBuilder, FrameTransport, StreamAnchorHandle, StreamFrame, TcpFrameTransport,
};
use velo_transports::tcp::TcpTransportBuilder;

// ---------------------------------------------------------------------------
// Sentinel helpers (cached_* are pub(crate), so we serialize directly)
// ---------------------------------------------------------------------------

fn dropped_bytes() -> Vec<u8> {
    rmp_serde::to_vec(&StreamFrame::<()>::Dropped).unwrap()
}

fn finalized_bytes() -> Vec<u8> {
    rmp_serde::to_vec(&StreamFrame::<()>::Finalized).unwrap()
}

// ---------------------------------------------------------------------------
// Test 1: TCP round-trip (TCP-03)
// ---------------------------------------------------------------------------

/// Validates that frames sent through a TcpFrameTransport bind/connect pair
/// are received in order and that the Dropped sentinel is injected after
/// sender close.
#[tokio::test(flavor = "multi_thread")]
async fn test_local_round_trip() {
    let transport = TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into());

    let (endpoint, rx) = transport.bind(1).await.unwrap();
    let tx = transport.connect(&endpoint, 1, 1).await.unwrap();

    // Send 10 string items
    let mut expected = Vec::new();
    for i in 0..10 {
        let frame = rmp_serde::to_vec(&StreamFrame::<String>::Item(format!("item-{}", i))).unwrap();
        expected.push(frame.clone());
        tx.send_async(frame).await.unwrap();
    }

    // Send Finalized sentinel
    let fin = finalized_bytes();
    tx.send_async(fin.clone()).await.unwrap();

    // Drop sender (triggers TCP close sequence)
    drop(tx);

    // Receive all 10 items
    for (i, exp) in expected.iter().enumerate() {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap_or_else(|_| panic!("timeout waiting for item {}", i))
            .unwrap_or_else(|_| panic!("channel closed at item {}", i));
        assert_eq!(&received, exp, "item {} mismatch", i);
    }

    // Receive Finalized sentinel
    let received_fin = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for Finalized")
        .expect("channel closed before Finalized");
    assert_eq!(received_fin, fin, "expected Finalized sentinel");

    // No extra Dropped sentinel should follow Finalized
    let tail = tokio::time::timeout(Duration::from_secs(2), rx.recv_async()).await;
    match tail {
        Ok(Ok(extra)) => {
            assert_ne!(
                extra.as_slice(),
                dropped_bytes().as_slice(),
                "should not inject Dropped after Finalized"
            );
        }
        Ok(Err(_)) => {} // channel closed -- expected
        Err(_) => {}     // timeout -- also fine
    }
}

// ---------------------------------------------------------------------------
// Test 2: Connection drop injects Dropped sentinel (TCP-05)
// ---------------------------------------------------------------------------

/// Validates that when the sender drops without sending a terminal sentinel,
/// the bind-side read loop detects TCP EOF and injects a Dropped sentinel.
#[tokio::test(flavor = "multi_thread")]
async fn test_connection_drop() {
    let transport = TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into());

    let (endpoint, rx) = transport.bind(1).await.unwrap();
    let tx = transport.connect(&endpoint, 1, 1).await.unwrap();

    // Send a few items
    for i in 0..3 {
        let frame =
            rmp_serde::to_vec(&StreamFrame::<String>::Item(format!("data-{}", i))).unwrap();
        tx.send_async(frame).await.unwrap();
    }

    // Drop sender without finalize -- simulates crash/disconnect
    drop(tx);

    // Receive the 3 data items
    for i in 0..3 {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap_or_else(|_| panic!("timeout waiting for data item {}", i))
            .unwrap_or_else(|_| panic!("channel closed at data item {}", i));
        let frame: StreamFrame<String> = rmp_serde::from_slice(&received).unwrap();
        assert!(
            matches!(frame, StreamFrame::Item(ref s) if s == &format!("data-{}", i)),
            "expected Item(data-{}) but got {:?}",
            i,
            frame
        );
    }

    // Should receive injected Dropped sentinel
    let sentinel = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for Dropped sentinel")
        .expect("channel closed before Dropped sentinel");

    assert_eq!(
        sentinel.as_slice(),
        dropped_bytes().as_slice(),
        "expected Dropped sentinel after abrupt close"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Sender-initiated close (TCP-07)
// ---------------------------------------------------------------------------

/// Validates proper TCP close sequence when sender initiates close.
/// Sender drops tx (initiating FIN), receiver detects and cleans up.
#[tokio::test(flavor = "multi_thread")]
async fn test_sender_close_first() {
    let transport = TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into());

    let (endpoint, rx) = transport.bind(1).await.unwrap();
    let tx = transport.connect(&endpoint, 1, 1).await.unwrap();

    // Send items then explicitly drop tx (sender-initiated close)
    for i in 0..5 {
        let frame =
            rmp_serde::to_vec(&StreamFrame::<u32>::Item(i)).unwrap();
        tx.send_async(frame).await.unwrap();
    }

    // Explicitly drop sender to initiate TCP FIN
    drop(tx);

    // Receive all 5 items
    for i in 0..5u32 {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap_or_else(|_| panic!("timeout waiting for item {}", i))
            .unwrap_or_else(|_| panic!("channel closed at item {}", i));
        let frame: StreamFrame<u32> = rmp_serde::from_slice(&received).unwrap();
        assert!(
            matches!(frame, StreamFrame::Item(v) if v == i),
            "expected Item({}) but got {:?}",
            i,
            frame
        );
    }

    // Should receive Dropped sentinel (no terminal sentinel was sent by user)
    let sentinel = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for Dropped sentinel")
        .expect("channel closed before Dropped sentinel");

    assert_eq!(
        sentinel.as_slice(),
        dropped_bytes().as_slice(),
        "expected Dropped sentinel after sender-initiated close without terminal"
    );

    // Channel should close after sentinel
    let result = tokio::time::timeout(Duration::from_secs(2), rx.recv_async()).await;
    match result {
        Ok(Err(_)) => {} // channel closed -- expected
        Err(_) => {}     // timeout -- also fine
        Ok(Ok(extra)) => panic!("unexpected extra frame after Dropped: {:?}", extra),
    }
}

// ---------------------------------------------------------------------------
// Test 4: Two independent transports (TCP-08)
// ---------------------------------------------------------------------------

/// Validates that two separate TcpFrameTransport instances can communicate
/// by having one bind and the other connect. This exercises the cross-transport
/// path that would be used in remote-attach scenarios.
#[tokio::test(flavor = "multi_thread")]
async fn test_remote_attach() {
    // Transport A (receiver side): binds a listener
    let transport_a = TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into());
    // Transport B (sender side): connects to transport A's endpoint
    let transport_b = TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into());

    let (endpoint, rx) = transport_a.bind(42).await.unwrap();

    // B connects to A's endpoint
    let tx = transport_b.connect(&endpoint, 42, 1).await.unwrap();

    // Send items from B
    let mut expected = Vec::new();
    for i in 0..20 {
        let frame =
            rmp_serde::to_vec(&StreamFrame::<String>::Item(format!("remote-{}", i))).unwrap();
        expected.push(frame.clone());
        tx.send_async(frame).await.unwrap();
    }

    // Send Finalized from B
    let fin = finalized_bytes();
    tx.send_async(fin.clone()).await.unwrap();
    drop(tx);

    // Receive all 20 items on A's side
    for (i, exp) in expected.iter().enumerate() {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap_or_else(|_| panic!("timeout on remote item {}", i))
            .unwrap_or_else(|_| panic!("channel closed at remote item {}", i));
        assert_eq!(&received, exp, "remote item {} mismatch", i);
    }

    // Receive Finalized
    let received_fin = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for remote Finalized")
        .expect("channel closed before remote Finalized");
    assert_eq!(received_fin, fin, "expected Finalized from remote sender");
}

// ---------------------------------------------------------------------------
// Test 5: AnchorManager with TCP transport registry
// ---------------------------------------------------------------------------

/// Validates that an AnchorManager constructed with TcpFrameTransport as
/// default and a populated transport_registry correctly resolves the TCP
/// transport for bind/connect operations.
#[tokio::test(flavor = "multi_thread")]
async fn test_anchor_manager_tcp_registry() {
    let transport = Arc::new(TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into()));
    let mut registry = HashMap::new();
    registry.insert(
        "tcp".to_string(),
        transport.clone() as Arc<dyn FrameTransport>,
    );

    let manager = Arc::new(
        velo_streaming::AnchorManagerBuilder::default()
            .worker_id(velo_common::WorkerId::from_u64(1))
            .transport(transport as Arc<dyn FrameTransport>)
            .transport_registry(Arc::new(registry))
            .build()
            .unwrap(),
    );

    // The transport_registry should contain "tcp"
    assert!(
        !manager.transport_registry.is_empty(),
        "transport_registry should not be empty"
    );
    assert!(
        manager.transport_registry.contains_key("tcp"),
        "transport_registry should contain 'tcp' scheme"
    );

    // Create an anchor and verify it works with the TCP transport
    let anchor = manager.create_anchor::<String>();
    let _handle = anchor.handle();
    // Anchor creation should succeed regardless of transport type
}

// ---------------------------------------------------------------------------
// Test 6: VeloBuilder with TCP transport (TCP-09 / GRPC-08)
// ---------------------------------------------------------------------------

/// Validates that VeloBuilder.stream_config(StreamConfig::Tcp(None)) creates a
/// TcpFrameTransport and populates the transport_registry with both "tcp" and
/// "velo" schemes. This is the canonical backward-compat test for GRPC-08:
/// StreamConfig::Tcp(None) must produce identical AnchorManager setup as the
/// old stream_bind_addr(0.0.0.0) call.
#[tokio::test(flavor = "multi_thread")]
async fn test_velo_builder_tcp_transport() {
    use velo::StreamConfig;

    let transport = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        Arc::new(
            velo_transports::tcp::TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        )
    };

    let velo = velo::Velo::builder()
        .add_transport(transport)
        .stream_config(StreamConfig::Tcp(None))
        .expect("stream_config should succeed on first call")
        .build()
        .await
        .unwrap();

    // Verify transport_registry contains both schemes
    let registry = &velo.anchor_manager().transport_registry;
    assert!(
        registry.contains_key("tcp"),
        "transport_registry should contain 'tcp' scheme"
    );
    assert!(
        registry.contains_key("velo"),
        "transport_registry should contain 'velo' scheme"
    );
    assert_eq!(registry.len(), 2, "transport_registry should have exactly 2 entries");

    // Create an anchor to verify the setup works end-to-end
    let _anchor = velo.create_anchor::<String>();
}

// ---------------------------------------------------------------------------
// Helpers for two-worker AM dispatch test
// ---------------------------------------------------------------------------

/// Create a TcpTransport bound to an OS-assigned port (for AM transport, not streaming).
fn new_am_tcp_transport() -> Arc<velo_transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

/// Set up two Messenger instances connected to each other over TCP loopback.
async fn make_two_messengers() -> (Arc<Messenger>, Arc<Messenger>) {
    let t1 = new_am_tcp_transport();
    let t2 = new_am_tcp_transport();

    let m1 = Messenger::new(vec![t1], None)
        .await
        .expect("create messenger 1");
    let m2 = Messenger::new(vec![t2], None)
        .await
        .expect("create messenger 2");

    let p1 = m1.peer_info();
    let p2 = m2.peer_info();

    // Register bidirectionally so each can reach the other.
    m2.register_peer(p1).expect("register m1 on m2");
    m1.register_peer(p2).expect("register m2 on m1");

    // Wait for TCP connections to establish.
    tokio::time::sleep(Duration::from_millis(200)).await;

    (m1, m2)
}

// ---------------------------------------------------------------------------
// Test 7: Full AM-dispatch TCP integration test (TCP-08 gap closure)
// ---------------------------------------------------------------------------

/// TCP-08 gap closure: Two-worker remote attach via full AM control-plane + TCP data transport.
///
/// This test validates the COMPLETE path that was missing from test_remote_attach:
///   1. Worker A creates AnchorManager with TcpFrameTransport + "tcp" in transport_registry
///   2. Worker A calls register_handlers to wire _anchor_attach/_anchor_finalize handlers
///   3. Worker A creates an anchor, handle transferred as u128 to Worker B
///   4. Worker B creates AnchorManager with TcpFrameTransport + "tcp" in transport_registry
///   5. Worker B calls register_handlers (sets messenger_lock for attach_remote)
///   6. Worker B calls attach_stream_anchor (remote path triggers):
///      - sends _anchor_attach AM to Worker A
///      - Worker A's handler calls manager.transport.bind() (TcpFrameTransport) -> tcp:// endpoint
///      - Worker A's handler spawns reader_pump, returns tcp:// endpoint in AnchorAttachResponse
///      - Worker B's attach_remote receives tcp:// endpoint
///      - Worker B calls resolve_transport("tcp") -> TcpFrameTransport
///      - Worker B calls TcpFrameTransport.connect(tcp://...) -> flume::Sender
///   7. Worker B sends items + finalize via StreamSender
///   8. Worker A's StreamAnchor yields all items then Finalized
#[tokio::test(flavor = "multi_thread")]
async fn test_remote_attach_am_dispatch() {
    let (messenger_a, messenger_b) = make_two_messengers().await;
    let worker_id_a = messenger_a.instance_id().worker_id();
    let worker_id_b = messenger_b.instance_id().worker_id();

    // --- Worker A setup ---
    // TcpFrameTransport is the DEFAULT transport (used by _anchor_attach handler's bind())
    let tcp_a = Arc::new(TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into()));
    let mut registry_a = HashMap::new();
    registry_a.insert("tcp".to_string(), tcp_a.clone() as Arc<dyn FrameTransport>);

    let am_a: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_a)
            .transport(tcp_a as Arc<dyn FrameTransport>)
            .transport_registry(Arc::new(registry_a))
            .build()
            .expect("AM worker A"),
    );

    // Register all 5 control-plane handlers on messenger_a
    am_a.register_handlers(Arc::clone(&messenger_a))
        .expect("register_handlers on worker A");

    // Worker A: create anchor
    let mut anchor_stream = am_a.create_anchor::<u32>();
    let handle = anchor_stream.handle();

    // Simulate cross-worker handle transfer (u128 round-trip)
    let handle_raw: u128 = handle.as_u128();
    let hi = (handle_raw >> 64) as u64;
    let lo = handle_raw as u64;
    let handle_transferred = StreamAnchorHandle::pack(WorkerId::from_u64(hi), lo);

    // Verify transfer preserves worker_id
    let (recovered_worker, _) = handle_transferred.unpack();
    assert_eq!(recovered_worker, worker_id_a);

    // --- Worker B setup ---
    let tcp_b = Arc::new(TcpFrameTransport::new(std::net::Ipv4Addr::LOCALHOST.into()));
    let mut registry_b = HashMap::new();
    registry_b.insert("tcp".to_string(), tcp_b.clone() as Arc<dyn FrameTransport>);

    let am_b: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_b)
            .transport(tcp_b as Arc<dyn FrameTransport>)
            .transport_registry(Arc::new(registry_b))
            .build()
            .expect("AM worker B"),
    );

    // Register handlers on Worker B (sets messenger_lock so attach_remote can send _anchor_attach AM)
    am_b.register_handlers(Arc::clone(&messenger_b))
        .expect("register_handlers on worker B");

    // --- Worker B: remote attach ---
    // handle_transferred.worker_id == worker_id_a != worker_id_b -> triggers attach_remote
    // attach_remote sends _anchor_attach AM to Worker A -> Worker A binds TcpFrameTransport
    // -> returns tcp:// endpoint -> Worker B resolves "tcp" from registry -> connects
    let sender = am_b
        .attach_stream_anchor::<u32>(handle_transferred)
        .await
        .expect("remote attach via TCP must succeed");

    // --- Worker B: send data ---
    for i in 0u32..10 {
        sender.send(i).await.expect("send item");
    }

    // TCP transport delivers items in order (dedicated connection, not AM dispatch),
    // so no sleep barrier needed before finalize (unlike VeloFrameTransport which uses
    // concurrent AM sends). However, add a small yield to let the pump task flush.
    tokio::time::sleep(Duration::from_millis(50)).await;

    sender.finalize().expect("finalize");

    // --- Worker A: collect items ---
    let mut items = Vec::new();
    let collect = async {
        while let Some(frame) = anchor_stream.next().await {
            match frame.expect("no stream error") {
                StreamFrame::Item(v) => items.push(v),
                StreamFrame::Finalized => break,
                other => panic!("unexpected frame: {:?}", other),
            }
        }
        items
    };

    let items = tokio::time::timeout(Duration::from_secs(10), collect)
        .await
        .expect("timed out waiting for Worker A to receive all items");

    // TCP transport preserves ordering (single connection, no concurrent dispatch)
    assert_eq!(
        items,
        (0u32..10).collect::<Vec<_>>(),
        "Worker A must receive all 10 items from Worker B in order via TCP"
    );
}
