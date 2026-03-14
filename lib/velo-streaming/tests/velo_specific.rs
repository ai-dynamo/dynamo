// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-specific integration tests that cannot live in the shared macro suite.
//! TEST-03: Exclusive reject — requires multi_thread runtime and concurrent attach.
//! TEST-07: Transport error propagation — tests closed transport receiver path.

mod common;

use std::sync::Arc;
use std::time::Duration;
use futures::StreamExt;
use velo_streaming::{AnchorManager, AttachError, StreamError, StreamFrame};
use velo_common::WorkerId;
use common::MockFrameTransport;

fn make_manager() -> Arc<AnchorManager> {
    Arc::new(AnchorManager::new(
        WorkerId::from_u64(1),
        Arc::new(MockFrameTransport::new()),
    ))
}

// TEST-03: Exclusive reject
// Requires multi_thread flavor to expose true concurrency in DashMap::entry() path.
#[tokio::test(flavor = "multi_thread")]
async fn test_03_exclusive_reject() {
    let mgr = make_manager();
    let (handle, _stream) = mgr.create_anchor::<u32>();

    // First attach succeeds
    let _sender = mgr
        .attach_stream_anchor::<u32>(handle, "mock://1", 1)
        .await
        .expect("first attach must succeed");

    // Second attach while first is active must return AlreadyAttached
    let result = mgr.attach_stream_anchor::<u32>(handle, "mock://1", 2).await;
    assert!(
        matches!(result, Err(AttachError::AlreadyAttached { .. })),
        "concurrent attach must return AlreadyAttached, got {:?}", result
    );
}

// TEST-07: Transport error propagation
// Strategy: Attach a sender so the stream is active, then drop the sender without
// explicit detach/finalize. impl Drop sends StreamFrame::Dropped synchronously,
// which AnchorStream maps to Err(StreamError::SenderDropped) — stream terminates.
//
// This validates the transport-close path: when the sender side disappears
// (network drop simulated by sender drop), the consumer-side AnchorStream
// observes the termination sentinel and stops yielding items.
#[tokio::test(flavor = "multi_thread")]
async fn test_07_transport_error_propagation() {
    let mgr = make_manager();

    let (handle, mut stream) = mgr.create_anchor::<u32>();
    // Attach a sender to start the stream.
    let sender = mgr
        .attach_stream_anchor::<u32>(handle, "mock://1", 1)
        .await
        .expect("attach must succeed");

    // Send one item to confirm normal delivery.
    sender.send(42u32).await.expect("send must succeed");

    // Drop the sender WITHOUT finalize/detach — simulates transport closure.
    // impl Drop sends StreamFrame::Dropped synchronously before returning.
    drop(sender);

    // Stream must deliver the item, then the Dropped sentinel, then None.
    let frame1 = tokio::time::timeout(Duration::from_secs(5), stream.next()).await
        .expect("stream must resolve within 5s");
    assert!(
        matches!(frame1, Some(Ok(StreamFrame::Item(42u32)))),
        "first frame must be Item(42), got {:?}", frame1
    );

    // Stream yields Err(SenderDropped) for the Dropped sentinel.
    let frame2 = tokio::time::timeout(Duration::from_secs(5), stream.next()).await
        .expect("stream must resolve within 5s");
    assert!(
        matches!(frame2, Some(Err(StreamError::SenderDropped))),
        "second frame must be Err(SenderDropped), got {:?}", frame2
    );

    // Stream must yield None after the terminal sentinel.
    assert!(
        stream.next().await.is_none(),
        "stream must be exhausted after SenderDropped"
    );
}
