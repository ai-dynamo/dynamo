// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the full control-plane AM dispatch path (TEST-02).
//!
//! Tests two-worker simulation using real Messenger + TCP loopback with the complete
//! AnchorManager control-plane wired up end-to-end:
//! - Worker A creates an AnchorManager, calls `register_handlers`, and creates an anchor.
//! - The handle is transferred to Worker B as a raw u128 (simulating cross-worker serialization).
//! - Worker B calls `attach_stream_anchor` with the transferred handle (worker_id mismatch triggers
//!   the remote path): sends `_anchor_attach` AM to Worker A, Worker A's handler binds the transport
//!   and spawns reader_pump, returns the stream endpoint, Worker B connects and receives StreamSender.
//! - Worker B sends N items + finalize via StreamSender.
//! - Worker A's StreamAnchor yields all N items then None.
//!
//! This validates the full stack: register_handlers -> remote attach_stream_anchor ->
//! _anchor_attach AM dispatch -> reader_pump -> StreamAnchor receives remote frames.

mod common;

use std::sync::Arc;
use std::time::Duration;

use velo_messenger::Messenger;
use velo_streaming::{AnchorManagerBuilder, FrameTransport, StreamAnchorHandle, StreamFrame};
use velo_streaming::velo_transport::VeloFrameTransport;
use velo_transports::tcp::TcpTransportBuilder;
use velo_common::WorkerId;

/// Create a TcpTransport bound to an OS-assigned port (no TOCTOU race).
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

/// Set up two Messenger instances connected to each other over TCP loopback.
async fn make_two_messengers() -> (Arc<Messenger>, Arc<Messenger>) {
    let t1 = new_tcp_transport();
    let t2 = new_tcp_transport();

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
// TEST-02: Two-worker full AM dispatch integration test
// ---------------------------------------------------------------------------
//
// Validates the full control-plane path:
//   1. Worker A creates AnchorManager, calls register_handlers, creates an anchor.
//   2. The handle is transferred as u128 to Worker B (simulates cross-worker serialization).
//   3. Worker B calls attach_stream_anchor (remote path fires):
//      - sends _anchor_attach AM to Worker A
//      - Worker A's handler binds transport + spawns reader_pump, returns endpoint
//      - Worker B connects transport and receives StreamSender
//   4. Worker B sends 5 items + finalize via StreamSender.
//   5. Worker A's StreamAnchor yields all 5 items then None.

#[tokio::test(flavor = "multi_thread")]
async fn test_02_remote_attach() {
    let (messenger_a, messenger_b) = make_two_messengers().await;
    let worker_id_a = messenger_a.instance_id().worker_id();
    let worker_id_b = messenger_b.instance_id().worker_id();

    // Worker A: create VeloFrameTransport and AnchorManager
    let vft_a = Arc::new(
        VeloFrameTransport::new(Arc::clone(&messenger_a), worker_id_a)
            .expect("VFT worker A"),
    );
    let am_a: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_a)
            .transport(Arc::clone(&vft_a) as Arc<dyn FrameTransport>)
            .build()
            .expect("AM worker A"),
    );

    // Worker A: register all five control-plane handlers on messenger_a
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

    // Verify transfer
    let (recovered_worker, _) = handle_transferred.unpack();
    assert_eq!(recovered_worker, worker_id_a);

    // Worker B: create VeloFrameTransport and AnchorManager
    let vft_b = Arc::new(
        VeloFrameTransport::new(Arc::clone(&messenger_b), worker_id_b)
            .expect("VFT worker B"),
    );
    let am_b: Arc<velo_streaming::AnchorManager> = Arc::new(
        AnchorManagerBuilder::default()
            .worker_id(worker_id_b)
            .transport(Arc::clone(&vft_b) as Arc<dyn FrameTransport>)
            .build()
            .expect("AM worker B"),
    );

    // Worker B: register handlers (sets messenger_lock so attach_remote can send _anchor_attach AM)
    am_b.register_handlers(Arc::clone(&messenger_b))
        .expect("register_handlers on worker B");

    // Worker B: attach via remote path (handle_transferred.worker_id == worker_id_a != worker_id_b)
    // This triggers attach_remote: sends _anchor_attach AM to Worker A,
    // Worker A's handler binds transport + spawns reader_pump, returns endpoint,
    // Worker B connects transport and receives StreamSender.
    let sender = am_b
        .attach_stream_anchor::<u32>(handle_transferred)
        .await
        .expect("remote attach must succeed");

    // Worker B: send 5 items
    for i in 0u32..5 {
        sender.send(i).await.expect("send item");
    }

    // Wait for all item AMs to be delivered to Worker A's _stream_data handler before
    // sending Finalized. AM dispatch is concurrent (spawned handlers) so Finalized could
    // arrive before later items without this barrier. 50ms >> loopback RTT.
    tokio::time::sleep(Duration::from_millis(50)).await;

    sender.finalize().expect("finalize");

    // Worker A: collect items from StreamAnchor until None
    use futures::StreamExt;
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

    // Content check (order may vary under concurrent AM delivery — Phase 09-02 decision)
    let mut items_sorted = items.clone();
    items_sorted.sort_unstable();
    assert_eq!(
        items_sorted,
        vec![0u32, 1, 2, 3, 4],
        "Worker A must receive all 5 items from Worker B"
    );
}

// ---------------------------------------------------------------------------
// VeloFrameTransport macro suite: NOT included
// ---------------------------------------------------------------------------
//
// The run_transport_tests! macro hardcodes "mock://1" as the endpoint string in
// all attach_stream_anchor calls. VeloFrameTransport::connect parses this as a
// velo:// URI and returns TransportError("invalid velo URI: missing velo:// prefix: mock://1").
//
// Changing the macro endpoint to a valid velo:// URI would require a worker_id and
// anchor_id that match a bound anchor — making the shared macro incompatible with
// MockFrameTransport (which ignores the URI entirely).
//
// Result: velo_suite is DEFERRED. Shared scenario coverage is provided by mock_suite
// (in mock_transport.rs), which runs all 8 macro-generated tests against MockFrameTransport.
// TEST-02 (above) validates VeloFrameTransport end-to-end data delivery independently.
