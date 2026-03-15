// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for GrpcFrameTransport.
//!
//! Validates GRPC-01..04: endpoint format, round-trip data flow,
//! Dropped sentinel injection on abrupt close, and exclusive-attach enforcement.
//!
//! This file requires `--features grpc` to compile (enforced via Cargo.toml [[test]]).

use std::time::Duration;

use velo_streaming::grpc_transport::parse_grpc_endpoint;
use velo_streaming::{GrpcFrameTransport, StreamFrame};

// ---------------------------------------------------------------------------
// Sentinel helpers
// ---------------------------------------------------------------------------

fn dropped_bytes() -> Vec<u8> {
    rmp_serde::to_vec(&StreamFrame::<()>::Dropped).unwrap()
}

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
