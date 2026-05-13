// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end replay tests (test 15 + 16).
//!
//! Loads deterministic msgpack fixture blobs, replays them through a live Consolidator
//! over ZMQ, collects egress batches, strips timestamps, and snapshot-asserts.

mod common;

use std::time::Duration;

use common::{TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, pick_port, sync_pulse};
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};
use serde::{Deserialize, Serialize};
use tokio::time::{interval, timeout};

// ---------------------------------------------------------------------------
// Snapshot-friendly egress types (Serialize + Deserialize).
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct SnapBatch(pub f64, pub Vec<SnapEvent>, pub Option<i32>);

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum SnapEvent {
    BlockStored {
        block_hashes: Vec<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    AllBlocksCleared {},
}

// ---------------------------------------------------------------------------
// Fixture format: Vec<Vec<u8>> — each blob is a pre-encoded ZMQ payload.
// ---------------------------------------------------------------------------

/// Build deterministic synthetic batches: root block STORE, child STORE, root REMOVE.
fn make_synthetic_payload_blobs() -> Vec<Vec<u8>> {
    let batches: Vec<TestBatch> = vec![
        // batch 0: root block STORE
        TestBatch(
            1_000.0,
            vec![RawKvEvent::BlockStored {
                block_hashes: vec![BlockHashValue::Unsigned(1)],
                parent_block_hash: None,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_name: None,
                medium: None,
                block_mm_infos: None,
                is_eagle: None,
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
            }],
            Some(0),
        ),
        // batch 1: child block STORE (parent=1)
        TestBatch(
            2_000.0,
            vec![RawKvEvent::BlockStored {
                block_hashes: vec![BlockHashValue::Unsigned(2)],
                parent_block_hash: Some(BlockHashValue::Unsigned(1)),
                token_ids: vec![50, 60, 70, 80],
                block_size: 4,
                lora_name: None,
                medium: None,
                block_mm_infos: None,
                is_eagle: None,
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
            }],
            Some(0),
        ),
        // batch 2: root block REMOVE
        TestBatch(
            3_000.0,
            vec![RawKvEvent::BlockRemoved {
                block_hashes: vec![BlockHashValue::Unsigned(1)],
                medium: None,
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
            }],
            Some(0),
        ),
    ];
    batches.iter().map(|b| b.encode()).collect()
}

/// Write fixture files from synthetic payload blobs.
#[allow(dead_code)]
fn regenerate_fixtures() -> anyhow::Result<()> {
    let blobs = make_synthetic_payload_blobs();
    let bytes = rmp_serde::to_vec(&blobs)?;
    std::fs::write("tests/fixtures/vllm_capture.msgpack", &bytes)?;
    std::fs::write("tests/fixtures/trtllm_capture.msgpack", &bytes)?;
    Ok(())
}

/// Ensure the fixture at `path` is non-empty; regenerate synthetically if missing or empty.
fn ensure_fixture(path: &str) {
    let needs_regen = std::fs::metadata(path)
        .map(|m| m.len() == 0)
        .unwrap_or(true);
    if needs_regen {
        regenerate_fixtures().expect("fixture regeneration failed");
    }
}

// ---------------------------------------------------------------------------
// Ignored manual regeneration test.
// ---------------------------------------------------------------------------

#[ignore]
#[test]
fn e2e_regenerate() {
    regenerate_fixtures().expect("regenerate_fixtures failed");
}

// ---------------------------------------------------------------------------
// Egress collection helpers
// ---------------------------------------------------------------------------

async fn collect_until_quiescent(sub: &mut ZmqSubHandle, quiesce_ms: u64) -> Vec<SnapBatch> {
    let quiesce = Duration::from_millis(quiesce_ms);
    let mut results = Vec::new();
    loop {
        match tokio::time::timeout(quiesce, sub.rx.recv()).await {
            Err(_elapsed) => break,
            Ok(None) => break,
            Ok(Some((_seq, batch))) => {
                let snap = convert_batch(batch);
                results.push(snap);
            }
        }
    }
    results
}

fn convert_batch(b: common::EventBatchMirror) -> SnapBatch {
    SnapBatch(0.0, b.1.into_iter().map(convert_event).collect(), b.2)
}

fn convert_event(e: common::EventMirror) -> SnapEvent {
    match e {
        common::EventMirror::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            medium,
        } => SnapEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            medium,
        },
        common::EventMirror::BlockRemoved {
            block_hashes,
            medium,
        } => SnapEvent::BlockRemoved {
            block_hashes,
            medium,
        },
        common::EventMirror::AllBlocksCleared {} => SnapEvent::AllBlocksCleared {},
    }
}

// ---------------------------------------------------------------------------
// Core replay logic
// ---------------------------------------------------------------------------

async fn run_replay(fixture_path: &str, engine_source: EventSource, snapshot_name: &str) {
    init_tracing();
    ensure_fixture(fixture_path);

    // Load fixture as Vec<Vec<u8>> (pre-encoded ZMQ payloads).
    let raw = std::fs::read(fixture_path).expect("read fixture");
    let payload_blobs: Vec<Vec<u8>> = rmp_serde::from_slice(&raw).expect("deserialize fixture");

    let egress_port = pick_port();
    let zmq_out = format!("tcp://127.0.0.1:{egress_port}");

    let pub_handle = ZmqPubHandle::spawn().await;

    let consolidator = ConsolidatorBuilder::new(&zmq_out, engine_source)
        .zmq_in(&pub_handle.endpoint)
        .poll_interval(Duration::from_millis(20))
        .build()
        .await
        .expect("build consolidator");

    let mut sub = ZmqSubHandle::spawn(&zmq_out).await.expect("spawn sub");

    assert!(
        sync_pulse(&pub_handle, &mut sub, Duration::from_secs(5)).await,
        "sync_pulse timed out"
    );

    // Replay blobs with 5ms pacing.
    let mut tick = interval(Duration::from_millis(5));
    for (seq, payload) in payload_blobs.iter().enumerate() {
        tick.tick().await;
        // Send as 3-frame multipart: [b"", seq_be, payload]
        let frames = vec![vec![], (seq as u64).to_be_bytes().to_vec(), payload.clone()];
        pub_handle.send_frames(frames).await.expect("send frame");
    }

    let collected = collect_until_quiescent(&mut sub, 500).await;

    consolidator.shutdown().await;

    insta::assert_yaml_snapshot!(snapshot_name, collected);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_full_vllm_replay() {
    timeout(
        Duration::from_secs(10),
        run_replay(
            "tests/fixtures/vllm_capture.msgpack",
            EventSource::Vllm,
            "vllm_replay",
        ),
    )
    .await
    .expect("test timed out");
}

#[tokio::test]
async fn e2e_full_trtllm_replay() {
    timeout(
        Duration::from_secs(10),
        run_replay(
            "tests/fixtures/trtllm_capture.msgpack",
            EventSource::Trtllm,
            "trtllm_replay",
        ),
    )
    .await
    .expect("test timed out");
}
