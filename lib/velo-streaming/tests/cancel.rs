// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for upstream cancellation (Phase 11).
//!
//! TEST-CANCEL-01: Drop StreamAnchor with attached sender — token fires, send() fails.
//! TEST-CANCEL-02: StreamController::cancel() with attached sender — same effect.
//! TEST-CANCEL-03: Remote cross-worker cancel via _stream_cancel AM over TCP.
//! TEST-CANCEL-04: Drop StreamAnchor with no sender — registry cleared, attach fails.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use velo_common::WorkerId;
use velo_streaming::{AnchorManager, AnchorManagerBuilder, AttachError};

// ---------------------------------------------------------------------------
// LocalMockTransport — in-memory transport for local (non-TCP) cancel tests
// ---------------------------------------------------------------------------

struct LocalMockTransport;

impl velo_streaming::FrameTransport for LocalMockTransport {
    fn bind(&self, anchor_id: u64, _session_id: u64) -> BoxFuture<'_, anyhow::Result<(String, flume::Receiver<Vec<u8>>)>> {
        Box::pin(async move {
            let (_tx, rx) = flume::bounded::<Vec<u8>>(256);
            Ok((format!("mock://{}", anchor_id), rx))
        })
    }

    fn connect(
        &self,
        _endpoint: &str,
        _anchor_id: u64,
        _session_id: u64,
    ) -> BoxFuture<'_, anyhow::Result<flume::Sender<Vec<u8>>>> {
        Box::pin(async move {
            let (tx, _rx) = flume::bounded::<Vec<u8>>(1);
            Ok(tx)
        })
    }
}

fn make_local_manager() -> Arc<AnchorManager> {
    let worker_id = WorkerId::from_u64(1);
    Arc::new(AnchorManager::new(worker_id, Arc::new(LocalMockTransport)))
}

// ---------------------------------------------------------------------------
// TEST-CANCEL-01: Drop StreamAnchor with attached sender — token fires, send() fails
// ---------------------------------------------------------------------------

/// Drop StreamAnchor (with an attached sender) triggers:
///   - impl Drop -> controller.cancel() -> SenderRegistry lookup -> cancel_token fires
///   - subsequent send() returns Err(SendError::ChannelClosed)
#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_01_drop_with_sender() {
    let mgr = make_local_manager();
    let anchor = mgr.create_anchor::<u32>();
    let handle = anchor.handle();

    // Attach sender
    let sender = mgr
        .attach_stream_anchor::<u32>(handle)
        .await
        .expect("attach must succeed");

    // Get cancellation token before any drop
    let cancel_token = sender.cancellation_token();
    assert!(!cancel_token.is_cancelled(), "token must not be cancelled initially");

    // Drop the StreamAnchor — triggers impl Drop -> controller.cancel()
    // which removes anchor from registry, fires token via SenderRegistry
    drop(anchor);

    // Give the async cancellation a moment to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Token must have fired
    assert!(
        cancel_token.is_cancelled(),
        "cancel_token must fire after StreamAnchor is dropped"
    );

    // send() must return Err(ChannelClosed)
    let result = sender.send(42u32).await;
    assert!(
        matches!(result, Err(velo_streaming::SendError::ChannelClosed)),
        "send() must return ChannelClosed after anchor dropped, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// TEST-CANCEL-02: StreamController::cancel() with attached sender
// ---------------------------------------------------------------------------

/// StreamController::cancel() triggers the same effect as dropping StreamAnchor.
#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_02_controller_cancel() {
    let mgr = make_local_manager();
    let anchor = mgr.create_anchor::<u32>();
    let handle = anchor.handle();

    // Obtain the controller BEFORE moving the stream
    let ctrl = anchor.controller();

    // Attach sender
    let sender = mgr
        .attach_stream_anchor::<u32>(handle)
        .await
        .expect("attach must succeed");

    // Get cancel token from sender
    let cancel_token = sender.cancellation_token();
    assert!(!cancel_token.is_cancelled(), "token must not be cancelled initially");

    // Explicitly cancel via the controller (anchor still alive, so Drop hasn't fired yet)
    ctrl.cancel();

    // Drop anchor after — Drop's cancel() will no-op due to AtomicBool gate
    drop(anchor);

    // Wait for async propagation
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Token must have fired
    assert!(
        cancel_token.is_cancelled(),
        "cancel_token must fire after StreamController::cancel()"
    );

    // send() must return Err
    let result = sender.send(42u32).await;
    assert!(
        matches!(result, Err(velo_streaming::SendError::ChannelClosed)),
        "send() must return ChannelClosed after controller cancel, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// TEST-CANCEL-04: Drop StreamAnchor with no sender — registry cleared, attach fails
// ---------------------------------------------------------------------------

/// Dropping StreamAnchor without attaching a sender:
///   - removes anchor from registry (via impl Drop -> controller.cancel())
///   - subsequent attach returns AnchorNotFound
#[tokio::test]
async fn test_cancel_04_drop_no_sender() {
    let mgr = make_local_manager();
    let anchor = mgr.create_anchor::<u32>();
    let handle = anchor.handle();

    // Drop without attaching any sender
    drop(anchor);

    // Give any async operations a moment to flush
    tokio::task::yield_now().await;

    // Subsequent attach must return AnchorNotFound — proves registry was cleared
    let result = mgr.attach_stream_anchor::<u32>(handle).await;
    assert!(
        matches!(result, Err(AttachError::AnchorNotFound { .. })),
        "attach after no-sender drop must return AnchorNotFound, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// TCP helpers + TEST-CANCEL-03 (remote two-worker cancel)
// ---------------------------------------------------------------------------

use velo_messenger::Messenger;
use velo_streaming::{StreamCancelRequest, create_stream_cancel_handler};
use velo_transports::tcp::TcpTransportBuilder;

// serde_json is needed for AM payload serialization (typed_unary_async handlers use JSON)

/// Create a TcpTransport bound to an OS-assigned port.
fn new_tcp_transport() -> Arc<velo_transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

/// Set up two Messenger instances connected over TCP loopback.
async fn make_two_messengers() -> (Arc<Messenger>, Arc<Messenger>) {
    let t1 = new_tcp_transport();
    let t2 = new_tcp_transport();

    let m1 = Messenger::new(vec![t1], None).await.expect("m1");
    let m2 = Messenger::new(vec![t2], None).await.expect("m2");

    m2.register_peer(m1.peer_info()).expect("register m1 on m2");
    m1.register_peer(m2.peer_info()).expect("register m2 on m1");

    tokio::time::sleep(Duration::from_millis(200)).await;
    (m1, m2)
}

/// TEST-CANCEL-03: Remote cross-worker cancel via _stream_cancel AM over TCP.
///
/// Worker A (consumer side) sends a `_stream_cancel` active message to Worker B
/// (sender side). Worker B's handler fires the CancellationToken for the active
/// StreamSender identified by `sender_stream_id`.
#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_03_remote_cancel() {
    let (messenger_a, messenger_b) = make_two_messengers().await;
    let worker_id_b = messenger_b.instance_id().worker_id();

    // Worker B: create AnchorManager with sender_registry + register _stream_cancel handler
    let mock_transport_b = Arc::new(LocalMockTransport);
    let am_b = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_b)
            .transport(mock_transport_b)
            .build()
            .expect("am_b"),
    );
    let _ = messenger_b.register_streaming_handler(create_stream_cancel_handler(
        Arc::clone(&am_b.sender_registry),
    ));

    // Worker B: create a local anchor + attach local sender (to populate SenderRegistry)
    let anchor_b = am_b.create_anchor::<u32>();
    let handle_b = anchor_b.handle();
    let sender_b = am_b
        .attach_stream_anchor::<u32>(handle_b)
        .await
        .expect("attach sender B");
    let cancel_token = sender_b.cancellation_token();

    // sender_stream_id is 1 (first attach on am_b)
    let sender_stream_id = 1u64;
    assert!(
        am_b.sender_registry.senders.contains_key(&sender_stream_id),
        "SenderEntry must be in registry before cancel"
    );

    // Worker A: send _stream_cancel AM to worker B with sender_stream_id
    // NOTE: typed_unary_async handlers deserialize with serde_json, not rmp_serde.
    let payload = serde_json::to_vec(&StreamCancelRequest { sender_stream_id })
        .expect("serialize StreamCancelRequest");
    messenger_a
        .am_send_streaming("_stream_cancel")
        .expect("am_send_streaming builder")
        .raw_payload(bytes::Bytes::from(payload))
        .worker(worker_id_b)
        .send()
        .await
        .expect("send _stream_cancel AM");

    // Wait for handler to process
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Assert: cancel_token fired on worker B
    assert!(
        cancel_token.is_cancelled(),
        "sender cancel_token must fire after _stream_cancel AM received"
    );
}
