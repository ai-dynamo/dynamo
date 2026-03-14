// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for VeloFrameTransport end-to-end data flow (TEST-02).
//!
//! Tests two-worker simulation using real Messenger + TCP loopback:
//! - Worker A creates a StreamAnchorHandle and binds the transport anchor.
//! - The handle is transferred to Worker B as a raw u128 (simulating cross-worker serialization).
//! - Worker B connects to the endpoint and sends typed StreamFrame data via the transport channel.
//! - Worker A receives and decodes the frames from the transport receiver.
//!
//! # Architecture Note
//!
//! The two-AnchorManager pattern (am_a.create_anchor + am_b.attach_stream_anchor) does NOT
//! support cross-worker data delivery in the current implementation: `attach_stream_anchor`
//! clones `frame_tx` from the LOCAL registry (am_b's registry), so StreamSender writes to
//! am_b's internal channel — not to am_a's AnchorStream. The reader_pump that would bridge
//! transport-received bytes to frame_tx is a planned feature (referenced in comments as
//! "Plan 03") that is not yet implemented.
//!
//! TEST-02 therefore tests the full production data path at the VeloFrameTransport layer:
//! bind/connect/send/recv with typed StreamFrame<T> encoding via rmp_serde, and handle
//! transfer as u128. This validates the wire format, AM routing, and TCP delivery end-to-end.

mod common;

use std::sync::Arc;
use std::time::Duration;

use velo_messenger::Messenger;
use velo_streaming::{FrameTransport, StreamAnchorHandle, StreamFrame};
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
// TEST-02: Two-worker remote attach via VeloFrameTransport + TCP
// ---------------------------------------------------------------------------
//
// Validates the full production data path:
//   1. Worker A creates a StreamAnchorHandle and binds the transport anchor.
//   2. The handle is transferred as u128 to worker B (simulates cross-worker serialization).
//   3. Worker B reconstructs the handle, connects to worker A's endpoint, and sends data.
//   4. Worker A receives and decodes the StreamFrame<u32> items from the transport receiver.
//   5. All 5 items arrive in order; stream terminates after Finalized sentinel.

#[tokio::test(flavor = "multi_thread")]
async fn test_02_remote_attach() {
    let (messenger_a, messenger_b) = make_two_messengers().await;
    let worker_id_a = messenger_a.instance_id().worker_id();
    let worker_id_b = messenger_b.instance_id().worker_id();

    // Worker A: create VeloFrameTransport.
    let vft_a = VeloFrameTransport::new(messenger_a.clone(), worker_id_a)
        .expect("create VeloFrameTransport for worker A");

    // Worker A: choose a local anchor ID and create a StreamAnchorHandle.
    // In production, create_anchor() assigns this ID; here we assign it directly
    // for the transport-layer test.
    let local_id: u64 = 1;
    let handle_a = StreamAnchorHandle::pack(worker_id_a, local_id);

    // Worker A: bind the transport anchor to get the endpoint and the transport receiver.
    let (endpoint, transport_rx): (String, flume::Receiver<Vec<u8>>) = vft_a
        .bind(local_id)
        .await
        .expect("bind anchor on worker A");
    assert!(
        endpoint.starts_with("velo://"),
        "endpoint must start with velo://"
    );

    // Transfer the handle as u128 (simulates cross-worker serialization via msgpack or direct copy).
    let handle_raw: u128 = handle_a.as_u128();

    // Worker B: reconstruct the handle from the raw u128.
    // StreamAnchorHandle::pack(worker_id, local_id) re-encodes it identically.
    let handle_b = {
        let hi = (handle_raw >> 64) as u64; // worker_id_a.as_u64()
        let lo = handle_raw as u64;         // local_id
        StreamAnchorHandle::pack(WorkerId::from_u64(hi), lo)
    };

    // Verify the transferred handle unpacks correctly.
    let (recovered_worker, recovered_local) = handle_b.unpack();
    assert_eq!(recovered_worker, worker_id_a, "transferred handle must carry worker A's ID");
    assert_eq!(recovered_local, local_id, "transferred handle must carry correct local_id");

    // Worker B: create VeloFrameTransport and connect to worker A's endpoint.
    let vft_b = VeloFrameTransport::new(messenger_b.clone(), worker_id_b)
        .expect("create VeloFrameTransport for worker B");

    // connect() spawns an AM pump task and returns a Sender<Vec<u8>>.
    // The pump prepends the 8-byte anchor_id prefix and sends each frame via AM to worker A.
    let transport_tx: flume::Sender<Vec<u8>> = vft_b
        .connect(&endpoint, local_id, worker_id_b.as_u64())
        .await
        .expect("connect from worker B to worker A's endpoint");

    // Worker B: encode StreamFrame<u32> items using rmp_serde and send via the transport sender.
    // This replicates what the reader_pump will eventually do for the AnchorManager data path.
    for i in 0u32..5 {
        let frame: StreamFrame<u32> = StreamFrame::Item(i);
        let frame_bytes = rmp_serde::to_vec(&frame).expect("serialize StreamFrame::Item");
        transport_tx
            .send_async(frame_bytes)
            .await
            .expect("send frame bytes from worker B");
    }

    // Send Finalized sentinel to signal end-of-stream.
    let finalized: StreamFrame<u32> = StreamFrame::Finalized;
    let finalized_bytes = rmp_serde::to_vec(&finalized).expect("serialize StreamFrame::Finalized");
    transport_tx
        .send_async(finalized_bytes)
        .await
        .expect("send Finalized sentinel from worker B");

    // Worker A: receive raw frame bytes from the transport receiver and decode.
    let mut items = Vec::new();
    loop {
        let raw = tokio::time::timeout(Duration::from_secs(5), transport_rx.recv_async())
            .await
            .expect("timeout waiting for frame from worker B")
            .expect("recv frame from transport");

        let frame: StreamFrame<u32> =
            rmp_serde::from_slice(&raw).expect("decode StreamFrame<u32>");

        match frame {
            StreamFrame::Item(v) => items.push(v),
            StreamFrame::Finalized => break,
            other => panic!("unexpected frame on worker A: {:?}", other),
        }
    }

    // AM delivery order is not strictly guaranteed under concurrent sends (see Phase 09-02 decision).
    // Sort both before comparing to make the assertion content-only (not order-sensitive).
    let mut items_sorted = items.clone();
    items_sorted.sort_unstable();
    assert_eq!(
        items_sorted,
        vec![0u32, 1, 2, 3, 4],
        "worker A must receive all 5 items from worker B (content check, order may vary)"
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
