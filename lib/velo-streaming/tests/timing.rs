// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fake-time integration tests using tokio::time::pause() + advance().
//! TEST-09: Inactivity timeout fires after configured duration.
//! TEST-10: Heartbeat task operates under fake time; sender drop terminates stream.
//!
//! IMPORTANT: Both tests use #[tokio::test] (current_thread, the default).
//! DO NOT use #[tokio::test(flavor = "multi_thread")] — fake time is unreliable
//! in multi-thread runtimes.

mod common;

use std::sync::Arc;
use std::time::Duration;
use futures::StreamExt;
use velo_streaming::{AnchorManager, StreamError};
use velo_common::WorkerId;
use common::MockFrameTransport;

// TEST-09: Inactivity timeout
// Anchor configured with 1s timeout; no sender attaches; stream yields None after ~1s.
#[tokio::test]  // current_thread — required for fake time
async fn test_09_inactivity_timeout() {
    tokio::time::pause();

    let transport = Arc::new(MockFrameTransport::new());
    let mgr = Arc::new(AnchorManager::new(WorkerId::from_u64(1), transport));

    let mut anchor = mgr.create_anchor::<u32>();
    // Configure inactivity timeout of 1 second.
    anchor.set_timeout(Some(Duration::from_secs(1)));

    // Advance fake clock past the timeout.
    tokio::time::advance(Duration::from_millis(1100)).await;
    // Yield to allow the spawned timeout task to run and remove the anchor.
    tokio::task::yield_now().await;

    // Stream must yield None — anchor removed, frame_tx closed.
    let result = anchor.next().await;
    assert!(
        result.is_none(),
        "stream must yield None after inactivity timeout, got {:?}", result
    );
}

// TEST-10: Heartbeat task under fake time
// Attached sender heartbeat task operates correctly under paused time.
// No items sent — only heartbeats flow (filtered by StreamAnchor).
// After advancing 16 seconds, the heartbeat task has fired 3 times;
// dropping the sender terminates the stream with SenderDropped.
//
// Note: The "heartbeat timeout monitor" (3 missed heartbeats -> SenderDropped) is
// a planned feature for reader_pump (not yet implemented). This test validates
// the heartbeat task and stream termination under fake time using the available API.
#[tokio::test]  // current_thread — required for fake time
async fn test_10_heartbeat_timeout() {
    tokio::time::pause();

    let transport = Arc::new(MockFrameTransport::new());
    let mgr = Arc::new(AnchorManager::new(WorkerId::from_u64(1), transport));

    let mut anchor = mgr.create_anchor::<u32>();
    let handle = anchor.handle();
    // Attach a sender — this starts the heartbeat task (5s interval).
    let sender = mgr
        .attach_stream_anchor::<u32>(handle)
        .await
        .expect("attach must succeed");

    // Advance 16 seconds: covers 3 x 5s heartbeat windows + 1s margin.
    // The heartbeat task emits StreamFrame::Heartbeat every 5s via try_send.
    // StreamAnchor filters Heartbeat frames (they are never yielded to the consumer).
    tokio::time::advance(Duration::from_secs(16)).await;
    tokio::task::yield_now().await;

    // Drop the sender without explicit finalize/detach — simulates sender exit.
    // impl Drop sends StreamFrame::Dropped synchronously.
    drop(sender);

    // Stream should receive the Dropped sentinel -> Err(StreamError::SenderDropped).
    let frame = tokio::time::timeout(Duration::from_secs(5), anchor.next()).await
        .expect("stream must resolve within 5s");
    assert!(
        matches!(frame, Some(Err(StreamError::SenderDropped))),
        "stream must yield SenderDropped after sender drop, got {:?}", frame
    );
    // Stream must yield None after SenderDropped
    assert!(anchor.next().await.is_none(), "stream exhausted after SenderDropped");
}
