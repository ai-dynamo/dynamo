// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: KVBM broadcast bridge (tests 3, 14).
//!
//! cargo test --test kvbm_bridge --features testing

mod common;
use futures::StreamExt;

use std::time::Duration;

use common::{
    EventMirror, TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, sync_pulse, wait_for,
};
use dynamo_kv_hashing::Request;
use dynamo_tokens::PositionalLineageHash;
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};
use kvbm_logical::BlockRegistry;
use kvbm_logical::events::protocol::KvCacheEvent;
use tokio::sync::broadcast;

/// Drain any pending messages from the SUB queue without blocking.
fn drain(sub: &mut ZmqSubHandle) -> Vec<(u64, common::EventBatchMirror)> {
    let mut out = Vec::new();
    while let Ok(item) = sub.rx.try_recv() {
        out.push(item);
    }
    out
}

/// Canonical PLH chain — same path the consolidator uses for vLLM ingress.
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

fn bs_event(hashes: Vec<u64>, tokens: Vec<u32>, block_size: usize) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: hashes.into_iter().map(BlockHashValue::Unsigned).collect(),
        parent_block_hash: None,
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

fn br_event(hashes: Vec<u64>) -> RawKvEvent {
    RawKvEvent::BlockRemoved {
        block_hashes: hashes.into_iter().map(BlockHashValue::Unsigned).collect(),
        medium: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

// ─── test 3: kvbm_bridge_plumbing ─────────────────────────────────────────────

/// `KvCacheEvent::Create` alone carries no tokens / block_size and is therefore not
/// publishable (the router rejects stores with `block_size = 0`). The bridge registers
/// the PLH in the tracker but suppresses the Store. When a vLLM ZMQ event later
/// references the same logical block, that event's metadata publishes the Store.
/// Removing both sources then publishes a Remove.
#[tokio::test]
async fn kvbm_bridge_plumbing() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(8), async {
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

        // Canonical PLH for the shared block.
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let seq_hash = canonical_chain(&tokens, 4)[0];
        let expected_block_hash = kvbm_consolidator::hash::router_block_hash(seq_hash);

        // 1. KVBM bridge sees Create first → registered but NOT published.
        tx.send(KvCacheEvent::Create(seq_hash))
            .expect("send create");
        // Give the publisher a tick to confirm no Store leaks.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let drained = drain(&mut sub);
        assert!(
            drained
                .iter()
                .flat_map(|(_, b)| b.1.iter())
                .all(|e| !matches!(e, EventMirror::BlockStored { .. })),
            "KVBM-only Create must not publish a placeholder Store: {drained:?}"
        );

        // 2. vLLM ZMQ store for the same block → publishes the Store.
        let batch = TestBatch(1.0, vec![bs_event(vec![0xCAFE], tokens, 4)], None);
        pub_handle.send_batch(&batch).await.expect("send zmq");

        let msgs = sub
            .recv_n(1, Duration::from_secs(3))
            .await
            .expect("recv_n store");
        let store = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .find(|e| matches!(e, EventMirror::BlockStored { .. }));
        assert!(store.is_some(), "expected BlockStored after vLLM join");
        if let Some(EventMirror::BlockStored {
            block_hashes,
            block_size,
            ..
        }) = store
        {
            assert_eq!(
                block_hashes[0], expected_block_hash,
                "egress block_hash must equal router_block_hash(seq_hash)"
            );
            assert_eq!(*block_size, 4, "block_size must come from the real source");
        }

        // 3. vLLM Remove first (one source still holds) → no Remove on egress.
        pub_handle
            .send_batch(&TestBatch(2.0, vec![br_event(vec![0xCAFE])], None))
            .await
            .expect("send vllm remove");
        tokio::time::sleep(Duration::from_millis(80)).await;
        let drained = drain(&mut sub);
        assert!(
            drained
                .iter()
                .flat_map(|(_, b)| b.1.iter())
                .all(|e| !matches!(e, EventMirror::BlockRemoved { .. })),
            "vLLM-only Remove must be silent while KVBM still holds: {drained:?}"
        );

        // 4. KVBM Remove → last source, Remove publishes.
        tx.send(KvCacheEvent::Remove(seq_hash))
            .expect("send remove");
        let msgs = sub
            .recv_n(1, Duration::from_secs(3))
            .await
            .expect("recv_n remove");
        let remove = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .find(|e| matches!(e, EventMirror::BlockRemoved { .. }));
        assert!(remove.is_some(), "expected BlockRemoved after last source");

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 14: registry_presence_after_kvbm_store ─────────────────────────────

/// After a `KvCacheEvent::Create(seq_hash)` flows through the bridge, the wired
/// `BlockRegistry` reports the hash as registered.
#[tokio::test]
async fn registry_presence_after_kvbm_store() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let (tx, rx) = broadcast::channel::<KvCacheEvent>(64);
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);
        let registry = BlockRegistry::new();

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Kvbm)
            .kvbm_events(
                tokio_stream::wrappers::BroadcastStream::new(rx)
                    .filter_map(|r| futures::future::ready(r.ok())),
            )
            .registry(registry.clone())
            .poll_interval(Duration::from_millis(20))
            .build()
            .await
            .expect("build");

        let tokens: Vec<u32> = vec![10, 20, 30, 40];
        let seq_hash = canonical_chain(&tokens, 4)[0];

        tx.send(KvCacheEvent::Create(seq_hash))
            .expect("send create");

        // Wait until the registry records the hash.
        let found = wait_for(Duration::from_secs(4), || {
            registry.match_sequence_hash(seq_hash, false).is_some()
        })
        .await;
        assert!(
            found,
            "registry should know about seq_hash after KVBM Create"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}
