// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: cross-source dedup, parent chain, and clear propagation (tests 4, 5, 6).
//!
//! cargo test --test dedup --features testing

mod common;
use futures::StreamExt;

use std::time::Duration;

use common::{EventMirror, TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, sync_pulse};
use dynamo_kv_hashing::Request;
use dynamo_tokens::PositionalLineageHash;
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};
use kvbm_logical::events::protocol::KvCacheEvent;
use tokio::sync::broadcast;

/// Canonical PLH chain for a token sequence via the kv-hashing crate — the same path
/// the consolidator uses internally to compute block identity for vLLM ingress.
fn canonical_chain(tokens: &[u32], block_size: usize) -> Vec<PositionalLineageHash> {
    Request::builder()
        .tokens(tokens.to_vec())
        .build()
        .expect("valid request")
        .into_blocks(block_size as u32)
        .expect("into_blocks")
        .into_iter()
        .map(|b| b.plh)
        .collect()
}

fn bs_event(
    block_hashes: Vec<u64>,
    parent: Option<u64>,
    tokens: Vec<u32>,
    block_size: usize,
    lora_name: Option<String>,
) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: block_hashes
            .into_iter()
            .map(BlockHashValue::Unsigned)
            .collect(),
        parent_block_hash: parent.map(BlockHashValue::Unsigned),
        token_ids: tokens,
        block_size,
        lora_name,
        medium: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

fn br_event(hashes: Vec<u64>) -> RawKvEvent {
    RawKvEvent::BlockRemoved {
        block_hashes: hashes.into_iter().map(BlockHashValue::Unsigned).collect(),
        medium: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

// ─── test 4: cross_source_dedup ──────────────────────────────────────────────

/// Same logical block via ZMQ (vLLM) AND broadcast (KVBM) → exactly 1 STORE.
/// Remove from vLLM only → no REMOVE. Remove from KVBM → 1 REMOVE.
#[tokio::test]
async fn cross_source_dedup() {
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

        // The shared tokens must hash to the same PositionalLineageHash for both sources.
        let tokens: Vec<u32> = vec![10u32, 20, 30, 40];
        let chain = canonical_chain(&tokens, 4);
        let seq_hash = chain[0];

        // Inject via ZMQ (vLLM path). The external-hash u64 used by vLLM is opaque to the
        // consolidator — it only uses it as a string key for parent lookups.
        let batch = TestBatch(
            1.0,
            vec![bs_event(vec![0xA11Cu64], None, tokens.clone(), 4, None)],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send zmq");

        // Inject same block via KVBM broadcast.
        tx.send(KvCacheEvent::Create(seq_hash))
            .expect("kvbm create");

        // Collect — allow time for both sources to be processed.
        let msgs = sub
            .recv_n(10, Duration::from_secs(2))
            .await
            .expect("recv_n");
        let all: Vec<&EventMirror> = msgs.iter().flat_map(|(_, b)| b.1.iter()).collect();
        let stores: Vec<_> = all
            .iter()
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .collect();
        let removes: Vec<_> = all
            .iter()
            .filter(|e| matches!(e, EventMirror::BlockRemoved { .. }))
            .collect();

        assert_eq!(
            stores.len(),
            1,
            "exactly 1 STORE expected from 2 sources, got: {all:?}"
        );
        assert_eq!(removes.len(), 0, "no REMOVE yet");

        // Remove from vLLM only (external hash is the seq u64 as string, used by
        // the subscriber internally — we must remove via the handle to avoid
        // coupling to internal string format).
        // Use consolidator handle to remove from KVBM side partially first.
        // Actually: the vLLM Remove path goes through ZMQ. The subscriber maps
        // block_hash strings to seq_hashes internally.  For this test we drive
        // vLLM Remove via ZMQ and KVBM Remove via handle, checking silence then final REMOVE.

        let handle = consolidator.handle();
        // Step 1: trigger vLLM Remove via ZMQ. Use the same opaque external hash we stored.
        let rmv = TestBatch(2.0, vec![br_event(vec![0xA11Cu64])], None);
        pub_handle.send_batch(&rmv).await.expect("send remove");

        let msgs2 = sub
            .recv_n(5, Duration::from_millis(500))
            .await
            .expect("recv_n 2");
        let removes2: Vec<_> = msgs2
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockRemoved { .. }))
            .collect();
        assert_eq!(
            removes2.len(),
            0,
            "vLLM-only remove must be silent (KVBM still holds)"
        );

        // Step 2: KVBM Remove → should now emit REMOVE.
        handle.handle_kvbm_remove(seq_hash).await;

        let msgs3 = sub
            .recv_n(1, Duration::from_secs(2))
            .await
            .expect("recv_n 3");
        let removes3: Vec<_> = msgs3
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockRemoved { .. }))
            .collect();
        assert_eq!(
            removes3.len(),
            1,
            "REMOVE expected after last source dropped"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 5: parent_chain_resolution ─────────────────────────────────────────

/// vLLM stores parent A, then child B (referencing A).
/// Egress STORE for B must have `parent_block_hash = Some(kvbm_consolidator::hash::router_block_hash(A))`.
#[tokio::test]
async fn parent_chain_resolution() {
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

        // Block A (root) and Block B (child of A). The vLLM external hashes are opaque
        // u64s — the consolidator recomputes the canonical PLH from tokens regardless.
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4];
        let tokens_b: Vec<u32> = vec![5, 6, 7, 8];
        let canonical = canonical_chain(&[tokens_a.as_slice(), tokens_b.as_slice()].concat(), 4);
        let hash_a = canonical[0];
        let ext_a: u64 = 0xAAAA_0001;
        let ext_b: u64 = 0xBBBB_0002;

        // Send both blocks in one batch (vLLM sends all hashes in sequence).
        let batch = TestBatch(
            1.0,
            vec![bs_event(
                vec![ext_a, ext_b],
                None,
                [tokens_a, tokens_b].concat(),
                4,
                None,
            )],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send");

        let msgs = sub.recv_n(5, Duration::from_secs(3)).await.expect("recv_n");
        let all: Vec<&EventMirror> = msgs.iter().flat_map(|(_, b)| b.1.iter()).collect();

        let stores: Vec<_> = all
            .iter()
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .collect();

        assert_eq!(stores.len(), 2, "expected 2 STOREs for A+B, got: {all:?}");

        // Find A and B by block_hash == expected fragment.
        let a_frag = kvbm_consolidator::hash::router_block_hash(hash_a);

        let store_a = stores.iter().find(|e| {
            matches!(
                e,
                EventMirror::BlockStored {
                    block_hashes,
                    ..
                } if block_hashes[0] == a_frag
            )
        });
        let store_b = stores.iter().find(|e| {
            matches!(
                e,
                EventMirror::BlockStored {
                    block_hashes,
                    ..
                } if block_hashes[0] != a_frag
            )
        });

        assert!(store_a.is_some(), "could not identify STORE for block A");
        assert!(store_b.is_some(), "could not identify STORE for block B");

        // B's parent should equal A's block_hash.
        if let Some(EventMirror::BlockStored {
            parent_block_hash, ..
        }) = store_b
        {
            assert_eq!(
                *parent_block_hash,
                Some(a_frag),
                "block B parent_block_hash must be A's fragment"
            );
        }

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 6: clear_all_propagates ────────────────────────────────────────────

/// Store 3 blocks, send AllBlocksCleared, verify egress ClearAll.
/// Then store the same tokens again — they must be treated as first-occurrence (STORE emitted).
#[tokio::test]
async fn clear_all_propagates() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(15), async {
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
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(10)).await,
            "sync_pulse"
        );

        // Store 3 blocks (all in one ZMQ message; publisher drains them together).
        let tokens: Vec<u32> = (1u32..=12).collect();
        let seqs: Vec<u64> = vec![1000, 2000, 3000];
        let batch = TestBatch(
            1.0,
            vec![bs_event(seqs.clone(), None, tokens, 4, None)],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send stores");

        // recv_n(1) since publisher emits all 3 stores in a single batch on next tick.
        let msgs = sub
            .recv_n(1, Duration::from_millis(800))
            .await
            .expect("recv stores");
        let stores_count: usize = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();
        assert_eq!(stores_count, 3, "expected 3 initial STOREs");

        // Send ClearAll.
        let clear = TestBatch(2.0, vec![RawKvEvent::AllBlocksCleared], None);
        pub_handle.send_batch(&clear).await.expect("send clear");

        let msgs2 = sub
            .recv_n(1, Duration::from_millis(800))
            .await
            .expect("recv clear");
        let has_clear = msgs2
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .any(|e| matches!(e, EventMirror::AllBlocksCleared {}));
        assert!(has_clear, "expected AllBlocksCleared on egress");

        // Store the same tokens again — tracker has been wiped, must produce 3 new STOREs.
        let tokens2: Vec<u32> = (1u32..=12).collect();
        let batch2 = TestBatch(3.0, vec![bs_event(seqs, None, tokens2, 4, None)], None);
        pub_handle
            .send_batch(&batch2)
            .await
            .expect("send re-stores");

        let msgs3 = sub
            .recv_n(1, Duration::from_millis(800))
            .await
            .expect("recv re-stores");
        let re_stores: usize = msgs3
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();
        assert_eq!(
            re_stores, 3,
            "blocks after ClearAll should be treated as first-occurrence"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}
