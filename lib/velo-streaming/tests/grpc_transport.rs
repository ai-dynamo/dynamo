// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for GrpcFrameTransport.
//!
//! Validates GRPC-01..04, GRPC-09: endpoint format, round-trip data flow,
//! Dropped sentinel injection on abrupt close, exclusive-attach enforcement,
//! and full two-worker AM dispatch integration.
//!
//! This file requires `--features grpc` to compile (enforced via Cargo.toml [[test]]).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use velo_common::WorkerId;
use velo_messenger::Messenger;
use velo_streaming::grpc_transport::parse_grpc_endpoint;
use velo_streaming::{
    AnchorManagerBuilder, FrameTransport, GrpcFrameTransport, StreamAnchorHandle, StreamFrame,
};
use velo_transports::tcp::TcpTransportBuilder;

// ---------------------------------------------------------------------------
// Sentinel helpers
// ---------------------------------------------------------------------------

fn dropped_bytes() -> Vec<u8> {
    rmp_serde::to_vec(&StreamFrame::<()>::Dropped).unwrap()
}

#[allow(dead_code)]
fn finalized_bytes() -> Vec<u8> {
    rmp_serde::to_vec(&StreamFrame::<()>::Finalized).unwrap()
}

// ---------------------------------------------------------------------------
// Test GRPC-01: bind returns grpc:// endpoint
// ---------------------------------------------------------------------------

/// Validates that `bind(anchor_id)` returns an endpoint starting with "grpc://"
/// and that `parse_grpc_endpoint` can parse it correctly.
#[tokio::test(flavor = "multi_thread")]
async fn test_bind_returns_grpc_endpoint() {
    let transport = GrpcFrameTransport::new("0.0.0.0:0".parse().unwrap())
        .await
        .unwrap();

    let (endpoint, _rx) = transport.bind(1).await.unwrap();

    assert!(
        endpoint.starts_with("grpc://"),
        "endpoint should start with grpc://: {}",
        endpoint
    );

    let (addr, anchor_id) = parse_grpc_endpoint(&endpoint).unwrap();
    assert!(addr.port() > 0, "port should be non-zero, got {}", addr.port());
    assert_eq!(anchor_id, 1, "anchor_id should match bound value");
}

// ---------------------------------------------------------------------------
// Test GRPC-02: connect round-trip
// ---------------------------------------------------------------------------

/// Validates that frames sent through connect() arrive in order at the bind() receiver.
#[tokio::test(flavor = "multi_thread")]
async fn test_connect_round_trip() {
    let transport = GrpcFrameTransport::new("0.0.0.0:0".parse().unwrap())
        .await
        .unwrap();

    let (endpoint, rx) = transport.bind(2).await.unwrap();
    let tx = transport.connect(&endpoint, 2, 1).await.unwrap();

    // Send 5 frames
    let mut expected = Vec::new();
    for i in 0u32..5 {
        let frame = rmp_serde::to_vec(&StreamFrame::<u32>::Item(i)).unwrap();
        expected.push(frame.clone());
        tx.send_async(frame).await.unwrap();
    }

    // Receive all 5 frames in order
    for (i, exp) in expected.iter().enumerate() {
        let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
            .await
            .unwrap_or_else(|_| panic!("timeout waiting for frame {}", i))
            .unwrap_or_else(|_| panic!("channel closed at frame {}", i));
        assert_eq!(&received, exp, "frame {} mismatch", i);
    }
}

// ---------------------------------------------------------------------------
// Test GRPC-03: Dropped sentinel injected on abrupt close
// ---------------------------------------------------------------------------

/// Validates that when the sender drops without a terminal sentinel,
/// the server pump injects a Dropped sentinel into the receiver channel.
#[tokio::test(flavor = "multi_thread")]
async fn test_dropped_on_abrupt_close() {
    let transport = GrpcFrameTransport::new("0.0.0.0:0".parse().unwrap())
        .await
        .unwrap();

    let (endpoint, rx) = transport.bind(3).await.unwrap();
    let tx = transport.connect(&endpoint, 3, 1).await.unwrap();

    // Send one data frame
    let frame = rmp_serde::to_vec(&StreamFrame::<String>::Item("data".to_string())).unwrap();
    tx.send_async(frame.clone()).await.unwrap();

    // Drop tx without sending a terminal sentinel (simulates abrupt close)
    drop(tx);

    // Should receive the data frame first
    let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for data frame")
        .expect("channel closed before data frame");
    assert_eq!(received, frame, "data frame should arrive first");

    // Then the injected Dropped sentinel
    let sentinel = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for Dropped sentinel")
        .expect("channel closed before Dropped sentinel");
    assert_eq!(
        sentinel.as_slice(),
        dropped_bytes().as_slice(),
        "Dropped sentinel should be injected after abrupt close"
    );
}

// ---------------------------------------------------------------------------
// Test GRPC-04: Exclusive attach enforcement
// ---------------------------------------------------------------------------

/// Validates that a second `connect()` to an anchor_id that already has an active
/// stream returns an Err (gRPC ALREADY_EXISTS propagated as anyhow::Error).
#[tokio::test(flavor = "multi_thread")]
async fn test_exclusive_attach_enforcement() {
    let transport = GrpcFrameTransport::new("0.0.0.0:0".parse().unwrap())
        .await
        .unwrap();

    let (endpoint, rx) = transport.bind(4).await.unwrap();

    // First connect should succeed
    let tx1 = transport.connect(&endpoint, 4, 1).await.unwrap();

    // Second connect to the same anchor_id must return Err
    let result2 = transport.connect(&endpoint, 4, 2).await;
    assert!(
        result2.is_err(),
        "second connect to active anchor_id must return Err, got Ok"
    );

    // First sender should still be usable: send one frame and receive it
    let frame = rmp_serde::to_vec(&StreamFrame::<u32>::Item(99)).unwrap();
    tx1.send_async(frame.clone()).await.unwrap();

    let received = tokio::time::timeout(Duration::from_secs(5), rx.recv_async())
        .await
        .expect("timeout waiting for frame after exclusive-attach check")
        .expect("channel closed unexpectedly");
    assert_eq!(received, frame, "first sender should still deliver frames");

    // Keep tx1 alive through test
    let _ = tx1;
}

// ---------------------------------------------------------------------------
// Test GRPC-09: Full two-worker remote attach via gRPC AM dispatch
// ---------------------------------------------------------------------------

/// GRPC-09: Two-worker remote attach via full AM control-plane + gRPC data transport.
///
/// Path:
///   1. Worker A creates AnchorManager with GrpcFrameTransport + "grpc" in registry
///   2. Worker A calls register_handlers to wire _anchor_attach/_anchor_finalize
///   3. Worker A creates anchor
///   4. Worker B creates AnchorManager with GrpcFrameTransport + "grpc" in registry
///   5. Worker B calls register_handlers
///   6. Worker B calls attach_stream_anchor (remote path):
///      - sends _anchor_attach AM to Worker A
///      - Worker A's handler calls transport.bind() -> grpc:// endpoint
///      - Worker B receives grpc:// endpoint, resolves "grpc" transport, calls connect()
///   7. Worker B sends items + finalize
///   8. Worker A's StreamAnchor yields all items then Finalized
#[tokio::test(flavor = "multi_thread")]
async fn test_remote_attach_am_dispatch() {
    // Helper: create TcpTransport bound to OS-assigned port (for AM messaging)
    fn new_am_transport() -> Arc<velo_transports::tcp::TcpTransport> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        )
    }

    async fn make_two_messengers() -> (Arc<Messenger>, Arc<Messenger>) {
        let t1 = new_am_transport();
        let t2 = new_am_transport();
        let m1 = Messenger::new(vec![t1], None).await.expect("m1");
        let m2 = Messenger::new(vec![t2], None).await.expect("m2");
        let p1 = m1.peer_info();
        let p2 = m2.peer_info();
        m2.register_peer(p1).expect("register m1 on m2");
        m1.register_peer(p2).expect("register m2 on m1");
        tokio::time::sleep(Duration::from_millis(200)).await;
        (m1, m2)
    }

    let (messenger_a, messenger_b) = make_two_messengers().await;
    let worker_id_a = messenger_a.instance_id().worker_id();
    let worker_id_b = messenger_b.instance_id().worker_id();

    // Worker A: GrpcFrameTransport (server — binds on Worker A's machine)
    let grpc_a = Arc::new(
        GrpcFrameTransport::new("127.0.0.1:0".parse().unwrap())
            .await
            .expect("GrpcFrameTransport worker A"),
    );
    let mut registry_a = HashMap::new();
    registry_a.insert("grpc".to_string(), grpc_a.clone() as Arc<dyn FrameTransport>);

    let am_a: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_a)
            .transport(grpc_a as Arc<dyn FrameTransport>)
            .transport_registry(Arc::new(registry_a))
            .build()
            .expect("AM worker A"),
    );
    am_a.register_handlers(Arc::clone(&messenger_a))
        .expect("register_handlers A");

    let mut anchor_stream = am_a.create_anchor::<u32>();
    let handle = anchor_stream.handle();

    // Simulate cross-worker handle transfer (u128 round-trip)
    let handle_raw: u128 = handle.as_u128();
    let hi = (handle_raw >> 64) as u64;
    let lo = handle_raw as u64;
    let handle_transferred = StreamAnchorHandle::pack(WorkerId::from_u64(hi), lo);
    let (recovered_worker, _) = handle_transferred.unpack();
    assert_eq!(recovered_worker, worker_id_a);

    // Worker B: GrpcFrameTransport (client — connects to Worker A's gRPC server)
    let grpc_b = Arc::new(
        GrpcFrameTransport::new("127.0.0.1:0".parse().unwrap())
            .await
            .expect("GrpcFrameTransport worker B"),
    );
    let mut registry_b = HashMap::new();
    registry_b.insert("grpc".to_string(), grpc_b.clone() as Arc<dyn FrameTransport>);

    let am_b: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_b)
            .transport(grpc_b as Arc<dyn FrameTransport>)
            .transport_registry(Arc::new(registry_b))
            .build()
            .expect("AM worker B"),
    );
    am_b.register_handlers(Arc::clone(&messenger_b))
        .expect("register_handlers B");

    // Worker B: remote attach (triggers _anchor_attach AM to Worker A -> gRPC bind -> connect)
    let sender = am_b
        .attach_stream_anchor::<u32>(handle_transferred)
        .await
        .expect("remote attach via gRPC must succeed");

    // Worker B: send 10 items
    for i in 0u32..10 {
        sender.send(i).await.expect("send item");
    }

    // gRPC uses async pump tasks -- add small barrier before finalize
    // (same pattern as VeloFrameTransport concurrent dispatch tests)
    tokio::time::sleep(Duration::from_millis(50)).await;

    sender.finalize().expect("finalize");

    // Worker A: collect items
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
        .expect("timed out waiting for Worker A to receive items from Worker B via gRPC");

    // gRPC stream preserves ordering (single bidirectional stream per anchor)
    assert_eq!(
        items,
        (0u32..10).collect::<Vec<_>>(),
        "Worker A must receive all 10 items from Worker B in order via gRPC"
    );
}
