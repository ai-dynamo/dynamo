// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test: in-process consolidator spawn via `InstanceLeader::with_consolidator`.
//!
//! Test flow:
//! 1. Allocate an egress port.
//! 2. Build a minimal `InstanceLeader` + `EventsManager`.
//! 3. Call `with_consolidator` — consolidator binds its ZMQ PUB socket.
//! 4. Connect a test ZMQ SUB socket (AFTER the PUB binds, to avoid slow-joiner).
//! 5. Inject a KVBM store via `ConsolidatorHandle` and assert a `BlockStored`
//!    batch arrives on the SUB socket within 3 s.
//! 6. Verify idempotency — second `with_consolidator` returns Err.
//! 7. Drop the leader and assert no hang beyond 5 s.

#![cfg(feature = "testing")]

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use kvbm_engine::G2;
use kvbm_engine::leader::{ConsolidatorParams, EventSource, InstanceLeader};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_logical::events::EventsManager;
use serde::Deserialize;
use tokio::sync::mpsc;

const BLOCK_SIZE: usize = 16;
const BLOCK_COUNT: usize = 32;

// ─── Wire mirror types ────────────────────────────────────────────────────────

/// Deserializable mirror of `kvbm_consolidator::wire::router_out::EventBatch`.
/// Encoded as a 3-tuple `(timestamp, events, dp_rank)` on the msgpack wire.
#[derive(Debug, Deserialize)]
struct EventBatch(f64, Vec<EventMirror>, Option<i32>);

/// Deserializable mirror of `kvbm_consolidator::wire::router_out::Event`.
/// Uses `#[serde(tag = "type")]` to match the publisher's internally-tagged format.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // fields only matter for Debug; matched via `matches!`
enum EventMirror {
    BlockStored {
        block_hashes: Vec<u64>,
        #[serde(default)]
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        #[serde(default)]
        lora_name: Option<String>,
        #[serde(default)]
        medium: Option<String>,
    },
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(default)]
        medium: Option<String>,
    },
    AllBlocksCleared {},
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        velo::transports::tcp::TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

/// Build a minimal `InstanceLeader` with an `EventsManager`-wired registry.
///
/// The same `BlockRegistry` (with `EventsManager` attached) is used both for
/// the leader's registry field and the G2 `BlockManager` — block registrations
/// flow to the `EventsManager` and onward to the consolidator.
async fn make_leader(events_manager: Arc<EventsManager>) -> Arc<InstanceLeader> {
    let velo = new_velo().await;

    let registry = TestRegistryBuilder::new()
        .events_manager(events_manager)
        .build();

    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );

    Arc::new(
        InstanceLeader::builder()
            .messenger(velo.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    )
}

/// Connect a ZMQ SUB socket and drain it in a background task into an mpsc channel.
///
/// Must be called AFTER the consolidator has bound its PUB socket so the slow-joiner
/// race is on the subscription-propagation side only (handled by the retry loop in the
/// test body).
async fn spawn_egress_sub(endpoint: &str) -> mpsc::Receiver<Vec<Vec<u8>>> {
    let ctx = tmq::Context::new();
    let mut socket = tmq::subscribe(&ctx)
        .set_linger(0)
        .connect(endpoint)
        .expect("sub connect")
        .subscribe(b"")
        .expect("subscribe all");

    let (tx, rx) = mpsc::channel::<Vec<Vec<u8>>>(256);

    tokio::spawn(async move {
        while let Some(Ok(multipart)) = socket.next().await {
            let frames: Vec<Vec<u8>> = multipart.into_iter().map(|f| f.to_vec()).collect();
            if tx.send(frames).await.is_err() {
                break;
            }
        }
    });

    rx
}

// ─── Integration test ─────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn consolidator_spawns_and_receives_block_stored() {
    // 1. Allocate the egress port up front.
    let egress_port = portpicker::pick_unused_port().expect("no free port");
    let egress_endpoint = format!("tcp://127.0.0.1:{egress_port}");

    // 2. Build leader + EventsManager.
    let events_manager = Arc::new(EventsManager::builder().build());
    let leader = make_leader(events_manager.clone()).await;

    // 3. Spawn the consolidator.
    //    vllm_zmq_endpoint points to a dead port — no vLLM process is listening.
    //    The ZMQ subscriber must retry without panicking (acceptance criterion 5).
    //    `with_consolidator` binds the egress ZMQ PUB socket as part of `build().await`.
    let dead_vllm_endpoint = "tcp://127.0.0.1:65530".to_string();
    leader
        .with_consolidator(ConsolidatorParams {
            vllm_zmq_endpoint: Some(dead_vllm_endpoint),
            egress_endpoint: egress_endpoint.clone(),
            engine_source: EventSource::Vllm,
            events_manager: events_manager.clone(),
        })
        .await
        .expect("with_consolidator");

    // 4. Connect the test SUB socket AFTER the consolidator has bound its PUB.
    //    Connecting after the bind avoids the fast-path ZMQ slow-joiner race.
    let mut sub_rx = spawn_egress_sub(&egress_endpoint).await;

    // 6. Idempotency: a second call must return Err immediately.
    let second = leader
        .with_consolidator(ConsolidatorParams {
            vllm_zmq_endpoint: None,
            egress_endpoint: egress_endpoint.clone(),
            engine_source: EventSource::Vllm,
            events_manager: events_manager.clone(),
        })
        .await;
    assert!(second.is_err(), "second with_consolidator must return Err");

    // 5. Obtain a ConsolidatorHandle and inject a KVBM store with real token_ids /
    //    block_size so the tracker publishes (empty token_ids are suppressed by design).
    //
    //    ZMQ subscription propagation takes a moment after connect, so we retry in a
    //    loop until the batch arrives.  Tracker deduplication means only the first
    //    injection is visible on egress; subsequent calls with the same seq_hash are
    //    no-ops — they just give ZMQ more time.
    let handle = leader
        .consolidator_handle()
        .expect("consolidator_handle must be Some after with_consolidator");

    let token_ids: Vec<u32> = (0..BLOCK_SIZE as u32).collect();

    // Use a fresh seq_hash on every attempt so the tracker never deduplicates a
    // prior published entry — if ZMQ drops the first message (slow-joiner) the
    // next iteration still enqueues a new `BlockStored` event.
    let mut attempt: u64 = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let frames = loop {
        attempt += 1;
        let seq_hash = kvbm_common::SequenceHash::new(0xAB_CD + attempt, None, attempt);
        handle
            .handle_kvbm_store(seq_hash, token_ids.clone(), BLOCK_SIZE, None)
            .await;

        match tokio::time::timeout(Duration::from_millis(300), sub_rx.recv()).await {
            Ok(Some(f)) => break f,
            Ok(None) => panic!("sub_rx channel closed unexpectedly"),
            Err(_) => {}
        }

        assert!(
            tokio::time::Instant::now() < deadline,
            "should receive a multipart batch within 5 s"
        );
    };

    // Wire format: 3 frames — empty prefix, 8-byte BE sequence number, msgpack payload.
    assert_eq!(frames.len(), 3, "expected 3-frame multipart");
    assert!(frames[0].is_empty(), "frame 0 must be empty prefix");
    assert_eq!(frames[1].len(), 8, "frame 1 must be 8-byte sequence number");

    let batch: EventBatch = rmp_serde::from_slice(&frames[2]).expect("msgpack decode failed");
    let EventBatch(_ts, events, _rank) = batch;

    assert!(!events.is_empty(), "batch must contain at least one event");
    assert!(
        matches!(events[0], EventMirror::BlockStored { .. }),
        "first event must be BlockStored, got: {:?}",
        events[0]
    );

    // 7. Drop the leader — the consolidator guard's Drop spawns async shutdown
    //    detached; we assert the whole thing completes within 5 s.
    let drop_result = tokio::time::timeout(Duration::from_secs(5), async move {
        drop(sub_rx);
        drop(leader);
        // Allow the detached shutdown future to join the background tasks.
        tokio::time::sleep(Duration::from_millis(300)).await;
    })
    .await;

    assert!(
        drop_result.is_ok(),
        "leader drop + consolidator shutdown must complete within 5 s"
    );
}
