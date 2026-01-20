// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the NovaStreaming API.
//!
//! These tests verify the NovaStreaming-centric API including:
//! - Stream creation and handle passing
//! - Attach via Nova RPC
//! - Stream cancellation
//! - Frame delivery

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_nova::Nova;
use dynamo_nova_backend::tcp::TcpTransportBuilder;
use dynamo_nova_streaming::{
    NovaStreaming, STREAM_ATTACH_HANDLER, STREAM_CANCEL_HANDLER, STREAM_DATA_HANDLER, StreamFrame,
    StreamHandle,
};

/// Helper to create a Nova instance with TCP transport on a random port.
async fn create_nova() -> Result<Arc<Nova>> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let tcp_transport = TcpTransportBuilder::new()
        .from_listener(listener)?
        .build()?;

    Nova::builder()
        .add_transport(Arc::new(tcp_transport))
        .build()
        .await
}

/// Helper to create two connected Nova instances.
async fn create_connected_pair() -> Result<(Arc<Nova>, Arc<Nova>)> {
    let nova1 = create_nova().await?;
    let nova2 = create_nova().await?;

    // Register peers with each other
    nova1.register_peer(nova2.peer_info())?;
    nova2.register_peer(nova1.peer_info())?;

    // Give time for connections to establish
    tokio::time::sleep(Duration::from_millis(50)).await;

    Ok((nova1, nova2))
}

// =============================================================================
// Basic NovaStreaming Tests
// =============================================================================

#[tokio::test]
async fn test_nova_streaming_builder() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova.clone())
        .buffer_capacity(128)
        .build()
        .expect("build streaming");

    // Verify we can access the underlying Nova
    assert_eq!(streaming.nova().instance_id(), nova.instance_id());
}

#[tokio::test]
async fn test_create_stream_returns_handle() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova.clone())
        .build()
        .expect("build streaming");

    let (receiver, handle): (_, StreamHandle) = streaming.create_stream::<String>();

    // Handle should be an AnchorId that matches the receiver
    assert_eq!(handle, receiver.anchor_id());

    // Handle's owner worker should match Nova's worker
    assert_eq!(handle.owner_worker(), nova.instance_id().worker_id());
}

#[tokio::test]
async fn test_create_stream_with_config() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    let (receiver, _handle) = streaming.create_stream_with::<String, _>(|b| {
        b.attach_timeout(Duration::from_secs(10))
            .message_timeout(Duration::from_secs(30))
            .capacity(64)
    });

    // Verify the config was applied
    assert_eq!(receiver.message_timeout(), Some(Duration::from_secs(30)));
}

#[tokio::test]
async fn test_stream_handle_is_serializable() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    let (_receiver, handle) = streaming.create_stream::<String>();

    // Handle should serialize/deserialize cleanly
    let json = serde_json::to_string(&handle).expect("serialize");
    let deserialized: StreamHandle = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(handle, deserialized);

    // StreamHandle is just AnchorId which wraps a u128 (16 bytes)
    // Verify the raw value is accessible
    assert_eq!(handle.as_u128(), deserialized.as_u128());
}

#[tokio::test]
async fn test_arm_handle() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    let (_receiver, handle) = streaming.create_stream::<String>();

    // Arm the handle for sending
    let armed = streaming.arm::<String>(handle);

    // Armed handle should have the same anchor ID
    assert_eq!(armed.anchor_id(), handle);
}

// =============================================================================
// Handler Registration Tests
// =============================================================================

#[tokio::test]
async fn test_handler_constants_exported() {
    // Verify the handler name constants are exported and have expected values
    // The actual handler functionality is tested in the attach/cancel tests
    assert_eq!(STREAM_ATTACH_HANDLER, "stream_attach");
    assert_eq!(STREAM_CANCEL_HANDLER, "stream_cancel");
    assert_eq!(STREAM_DATA_HANDLER, "stream_data");
}

// =============================================================================
// Cancellation Tests
// =============================================================================

#[tokio::test]
async fn test_cancel_stream_unattached() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    let (receiver, handle) = streaming.create_stream::<String>();

    // Cancel without any sender attached
    streaming.cancel_stream(handle, "test cancellation").await;

    // Receiver should get Cancelled frame
    let frame = receiver.try_recv();
    assert!(matches!(frame, Some(StreamFrame::Cancelled(reason)) if reason == "test cancellation"));
}

#[tokio::test]
async fn test_cancel_stream_prevents_attach() {
    let (nova1, nova2) = create_connected_pair().await.expect("create pair");

    let streaming1 = NovaStreaming::builder(nova1.clone())
        .build()
        .expect("build streaming1");

    let streaming2 = NovaStreaming::builder(nova2.clone())
        .build()
        .expect("build streaming2");

    // Create stream on instance 1
    let (receiver, handle) = streaming1.create_stream::<String>();

    // Cancel it before any sender attaches
    streaming1.cancel_stream(handle, "cancelled early").await;

    // Verify receiver gets Cancelled frame
    let frame = receiver.try_recv();
    assert!(matches!(frame, Some(StreamFrame::Cancelled(_))));

    // Wait for handler to be available on remote
    nova2
        .wait_for_handler(nova1.instance_id(), STREAM_ATTACH_HANDLER)
        .await
        .expect("wait for handler");

    // Try to attach from instance 2 - should fail since cancelled
    let armed = streaming2.arm::<String>(handle);
    let result = armed.attach().await;

    // Attach should fail (receiver cancelled/removed)
    assert!(result.is_err());
}

// =============================================================================
// Attach Tests
// =============================================================================

#[tokio::test]
async fn test_attach_returns_sender() {
    let (nova1, nova2) = create_connected_pair().await.expect("create pair");

    let streaming1 = NovaStreaming::builder(nova1.clone())
        .build()
        .expect("build streaming1");

    let streaming2 = NovaStreaming::builder(nova2.clone())
        .build()
        .expect("build streaming2");

    // Create stream on instance 1 with specific timeouts
    let (_receiver, handle) =
        streaming1.create_stream_with::<String, _>(|b| b.message_timeout(Duration::from_secs(60)));

    // Wait for handler to be available
    nova2
        .wait_for_handler(nova1.instance_id(), STREAM_ATTACH_HANDLER)
        .await
        .expect("wait for handler");

    // Attach from instance 2 - should return a StreamSender
    let armed = streaming2.arm::<String>(handle);
    let sender = armed.attach().await.expect("attach");

    // Sender should have the correct anchor ID
    assert_eq!(sender.anchor_id(), handle);

    // Target instance should be instance 1
    assert_eq!(sender.target_instance(), nova1.instance_id());

    // Heartbeat should be half of message timeout (30 seconds)
    assert_eq!(sender.heartbeat_interval(), Some(Duration::from_secs(30)));
}

// =============================================================================
// End-to-End Send/Receive Tests
// =============================================================================

#[tokio::test]
async fn test_full_send_receive() {
    let (nova1, nova2) = create_connected_pair().await.expect("create pair");

    let streaming1 = NovaStreaming::builder(nova1.clone())
        .build()
        .expect("build streaming1");

    let streaming2 = NovaStreaming::builder(nova2.clone())
        .build()
        .expect("build streaming2");

    // Create stream on instance 1
    let (receiver, handle) = streaming1.create_stream::<String>();

    // Wait for handler to be available on remote
    nova2
        .wait_for_handler(nova1.instance_id(), STREAM_ATTACH_HANDLER)
        .await
        .expect("wait for handler");

    // Attach from instance 2
    let sender = streaming2
        .arm::<String>(handle)
        .attach()
        .await
        .expect("attach");

    // Send data
    sender.send("hello".to_string()).await.expect("send hello");
    sender.send("world".to_string()).await.expect("send world");
    sender.finalize().await.expect("finalize");

    // Give a moment for frames to be delivered
    // tokio::time::sleep(Duration::from_millis(100)).await;

    // Receive and verify - first we should get Attached frame
    let frame1 = receiver.recv().await.expect("recv attached");
    assert!(
        matches!(frame1, StreamFrame::Attached { .. }),
        "expected Attached, got {:?}",
        frame1
    );

    // Then data frames
    let frame2 = receiver.recv().await.expect("recv hello");
    assert!(
        matches!(&frame2, StreamFrame::Data(s) if s == "hello"),
        "expected Data(hello), got {:?}",
        frame2
    );

    let frame3 = receiver.recv().await.expect("recv world");
    assert!(
        matches!(&frame3, StreamFrame::Data(s) if s == "world"),
        "expected Data(world), got {:?}",
        frame3
    );

    let frame4 = receiver.recv().await.expect("recv finalized");
    assert!(
        matches!(frame4, StreamFrame::Finalized),
        "expected Finalized, got {:?}",
        frame4
    );
}

// =============================================================================
// Multiple Streams Tests
// =============================================================================

#[tokio::test]
async fn test_multiple_streams_same_instance() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    // Create multiple streams
    let (recv1, handle1) = streaming.create_stream::<String>();
    let (recv2, handle2) = streaming.create_stream::<i32>();
    let (recv3, handle3) = streaming.create_stream::<Vec<u8>>();

    // Handles should be different
    assert_ne!(handle1, handle2);
    assert_ne!(handle2, handle3);

    // But all should have the same owner worker
    assert_eq!(handle1.owner_worker(), handle2.owner_worker());
    assert_eq!(handle2.owner_worker(), handle3.owner_worker());

    // Anchor IDs should match receivers
    assert_eq!(handle1, recv1.anchor_id());
    assert_eq!(handle2, recv2.anchor_id());
    assert_eq!(handle3, recv3.anchor_id());
}

#[tokio::test]
async fn test_cancel_one_stream_doesnt_affect_others() {
    let nova = create_nova().await.expect("create nova");
    let streaming = NovaStreaming::builder(nova)
        .build()
        .expect("build streaming");

    // Create multiple streams
    let (recv1, handle1) = streaming.create_stream::<String>();
    let (recv2, handle2) = streaming.create_stream::<String>();

    // Cancel one stream
    streaming.cancel_stream(handle1, "cancelled").await;

    // First receiver should get Cancelled
    let frame1 = recv1.try_recv();
    assert!(matches!(frame1, Some(StreamFrame::Cancelled(_))));

    // Second receiver should still be available (no frame yet)
    let frame2 = recv2.try_recv();
    assert!(frame2.is_none());

    // Can still cancel the second one separately
    streaming.cancel_stream(handle2, "also cancelled").await;
    let frame2 = recv2.try_recv();
    assert!(matches!(frame2, Some(StreamFrame::Cancelled(_))));
}
