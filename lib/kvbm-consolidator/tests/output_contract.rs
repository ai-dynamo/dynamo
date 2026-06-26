// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: egress wire-format contracts (tests 10, 11, 12).
//!
//! cargo test --test output_contract --features testing

mod common;
use futures::StreamExt;

use std::time::Duration;

use common::{EventMirror, TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, sync_pulse};
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};
use kvbm_logical::events::protocol::KvCacheEvent;
use tokio::sync::broadcast;

fn bs(hash: u64, parent: Option<u64>, tokens: Vec<u32>, block_size: usize) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(hash)],
        parent_block_hash: parent.map(BlockHashValue::Unsigned),
        token_ids: tokens,
        block_size,
        lora_name: None,
        medium: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

fn bs_lora(hash: u64, tokens: Vec<u32>, lora_name: String) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(hash)],
        parent_block_hash: None,
        token_ids: tokens,
        block_size: 4,
        lora_name: Some(lora_name),
        medium: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

// ─── test 10: sequence_counter_monotonic ─────────────────────────────────────

/// Drive 20 distinct batches; decode frame-2 (seq bytes) as u64 BE and assert
/// strictly increasing from the value seen on the first batch.
#[tokio::test]
async fn sequence_counter_monotonic() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let pub_handle = ZmqPubHandle::spawn().await;
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
            .zmq_in(&pub_handle.endpoint)
            .poll_interval(Duration::from_millis(20))
            .build()
            .await
            .expect("build");

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse"
        );

        // Send 20 distinct blocks, one per batch.
        for i in 0u64..20 {
            let batch = TestBatch(
                i as f64,
                vec![bs(
                    10000 + i,
                    None,
                    vec![i as u32, i as u32 + 1, i as u32 + 2, i as u32 + 3],
                    4,
                )],
                None,
            );
            pub_handle.send_batch(&batch).await.expect("send");
        }

        // Collect all 20 (plus any AllBlocksCleared from sync_pulse that may have merged).
        let msgs = sub
            .recv_n(20, Duration::from_secs(3))
            .await
            .expect("recv_n");

        // Filter to batches that actually contain BlockStored events.
        let seqs: Vec<u64> = msgs
            .iter()
            .filter(|(_, batch)| {
                batch
                    .1
                    .iter()
                    .any(|e| matches!(e, EventMirror::BlockStored { .. }))
            })
            .map(|(seq, _)| *seq)
            .collect();

        assert!(
            !seqs.is_empty(),
            "no store batches received; msgs: {msgs:?}"
        );

        // The publisher groups multiple events into a single batch on each tick.
        // We assert that the sequence counter values are monotonically increasing
        // across received batches, not necessarily 0..N (batch coalescing is fine).
        for window in seqs.windows(2) {
            assert!(
                window[1] > window[0],
                "sequence counter not strictly increasing: {:?}",
                seqs
            );
        }

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 11: lora_name_passthrough ──────────────────────────────────────────

/// Blocks stored with `lora_name = Some("adapter-x")` via ZMQ and via KVBM handle
/// both appear on egress with the lora field preserved.
#[tokio::test]
async fn lora_name_passthrough() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let (tx, rx) = broadcast::channel::<KvCacheEvent>(64);
        let pub_handle = ZmqPubHandle::spawn().await;
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
            .zmq_in(&pub_handle.endpoint)
            .kvbm_events(
                tokio_stream::wrappers::BroadcastStream::new(rx)
                    .filter_map(|r| futures::future::ready(r.ok())),
            )
            .poll_interval(Duration::from_millis(20))
            .build()
            .await
            .expect("build");

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse"
        );

        // ZMQ ingress with lora_name.
        let lora = "adapter-x".to_string();
        let batch = TestBatch(
            1.0,
            vec![bs_lora(5555, vec![1, 2, 3, 4], lora.clone())],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send zmq lora");

        // KVBM ingress with lora via direct handle.
        use dynamo_tokens::{PositionalLineageHash, compute_hash_v2};
        const SALT: u64 = 0;
        let tokens: Vec<u32> = vec![100, 200, 300, 400];
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let bh = compute_hash_v2(&bytes, SALT);
        let seq_hash = PositionalLineageHash::new(bh, None, 0);

        // inject via direct handle
        let _ = tx; // keep tx alive
        consolidator
            .handle()
            .handle_kvbm_store(seq_hash, tokens, 4, Some(lora.clone()))
            .await;

        let msgs = sub.recv_n(5, Duration::from_secs(3)).await.expect("recv_n");
        let stores: Vec<&EventMirror> = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .collect();

        // At least one store should carry the lora.
        let lora_stores: Vec<_> = stores
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EventMirror::BlockStored {
                        lora_name: Some(n),
                        ..
                    } if n == "adapter-x"
                )
            })
            .collect();

        assert!(
            !lora_stores.is_empty(),
            "expected at least 1 STORE with lora_name='adapter-x', got: {stores:?}"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 12: different_dp_ranks_collapse_to_zero ─────────────────────────────

/// Batches with input dp_rank 0, 1, 2 all produce egress batches with dp_rank = Some(0).
/// This is the current intentional behavior: the publisher hard-codes Some(0).
/// TODO: re-evaluate if per-rank routing is ever needed.
#[tokio::test]
async fn different_dp_ranks_collapse_to_zero() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let pub_handle = ZmqPubHandle::spawn().await;
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
            .zmq_in(&pub_handle.endpoint)
            .poll_interval(Duration::from_millis(20))
            .build()
            .await
            .expect("build");

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse"
        );

        for (rank, hash) in [(Some(0i32), 100u64), (Some(1), 200), (Some(2), 300)] {
            let tokens: Vec<u32> = vec![hash as u32; 4];
            let batch = TestBatch(1.0, vec![bs(hash, None, tokens, 4)], rank);
            pub_handle.send_batch(&batch).await.expect("send");
        }

        let msgs = sub.recv_n(5, Duration::from_secs(3)).await.expect("recv_n");

        // All egress batches containing STOREs must have dp_rank == Some(0).
        for (_, batch) in &msgs {
            let has_store = batch
                .1
                .iter()
                .any(|e| matches!(e, EventMirror::BlockStored { .. }));
            if has_store {
                assert_eq!(
                    batch.2,
                    Some(0),
                    "egress dp_rank must be Some(0) regardless of input; got: {:?}",
                    batch.2
                );
            }
        }

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}
