// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ZMQ ingress path (tests 1, 2, 8, 9).
//!
//! cargo test --test zmq_ingress --features testing

mod common;

use std::time::Duration;

use common::{EventMirror, TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, sync_pulse};
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};

// ─── helpers ─────────────────────────────────────────────────────────────────

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

// ─── test 1: zmq_ingress_roundtrip ───────────────────────────────────────────

/// A single vLLM batch with 3 chained blocks propagates to egress as 3 STOREs.
/// Parent chain: A→B→C, confirmed via `parent_block_hash` fields.
#[tokio::test]
async fn zmq_ingress_roundtrip() {
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
            .expect("build consolidator");

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("spawn sub");

        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse timed out"
        );

        // Now send a real batch: 3 blocks chained A→B→C.
        // Each block is 4 tokens; parent_block_hash chains externally.
        // The tracker sees these as positional hashes; we don't need the fragments
        // to match specific values — just verify the chain structure.
        let tokens_a: Vec<u32> = (1..=4).collect();
        let tokens_b: Vec<u32> = (5..=8).collect();
        let tokens_c: Vec<u32> = (9..=12).collect();

        // vLLM sends the entire sequence in one BlockStored event with all hashes.
        // The subscriber splits by block_size. We use sentinel hash values that
        // don't need to match the recomputed pos-lineage values (those are internal).
        let batch = TestBatch(
            1.0,
            vec![bs_event(
                vec![100, 200, 300],
                None,
                [tokens_a, tokens_b, tokens_c].concat(),
                4,
                None,
            )],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send_batch");

        // Collect events — the publisher drains every 20ms, so we wait up to 3s.
        let msgs = sub.recv_n(1, Duration::from_secs(3)).await.expect("recv_n");

        // We may get 1 or more batches; flatten all events.
        let all_events: Vec<&EventMirror> = msgs.iter().flat_map(|(_, b)| b.1.iter()).collect();
        let stores: Vec<&EventMirror> = all_events
            .iter()
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .copied()
            .collect();

        assert_eq!(
            stores.len(),
            3,
            "expected 3 BlockStored events, got: {all_events:?}"
        );

        // The first block should have no parent.
        let hashes_and_parents: Vec<(u64, Option<u64>)> = stores
            .iter()
            .map(|e| match e {
                EventMirror::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    ..
                } => (*block_hashes.first().unwrap(), *parent_block_hash),
                _ => unreachable!(),
            })
            .collect();

        // Root block: no parent.
        assert!(
            hashes_and_parents[0].1.is_none(),
            "first block should have no parent"
        );
        // Child blocks: parent == sibling's block_hash.
        assert_eq!(
            hashes_and_parents[1].1,
            Some(hashes_and_parents[0].0),
            "block B parent should be block A"
        );
        assert_eq!(
            hashes_and_parents[2].1,
            Some(hashes_and_parents[1].0),
            "block C parent should be block B"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 2: zmq_multipart_parsing ───────────────────────────────────────────

/// 2-frame and 3-frame payloads are accepted; 1-frame and 4-frame are dropped with WARN.
/// The loop must survive and process a valid batch after the bad ones.
#[tokio::test]
async fn zmq_multipart_parsing() {
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

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("spawn sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse timed out"
        );

        // 1-frame (bad): just a topic.
        pub_handle
            .send_frames(vec![vec![]])
            .await
            .expect("send 1-frame");

        // 4-frame (bad).
        let good_payload = TestBatch(0.0, vec![RawKvEvent::AllBlocksCleared], None).encode();
        pub_handle
            .send_frames(vec![vec![], vec![0u8; 8], good_payload.clone(), vec![0u8]])
            .await
            .expect("send 4-frame");

        // 2-frame (good): [topic, payload].
        let valid_2f = TestBatch(
            2.0,
            vec![bs_event(vec![111], None, vec![1, 2, 3, 4], 4, None)],
            None,
        );
        pub_handle
            .send_frames(vec![vec![], valid_2f.encode()])
            .await
            .expect("send 2-frame");

        // 3-frame (good): [topic, seq, payload].
        let valid_3f = TestBatch(
            3.0,
            vec![bs_event(vec![222], None, vec![5, 6, 7, 8], 4, None)],
            None,
        );
        pub_handle
            .send_batch(&valid_3f)
            .await
            .expect("send 3-frame");

        // We should receive exactly 2 STOREs from the 2-frame and 3-frame batches.
        let msgs = sub
            .recv_n(10, Duration::from_secs(3))
            .await
            .expect("recv_n");
        let stores: usize = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();
        assert!(
            stores >= 2,
            "expected ≥2 stores from good batches, got {stores}"
        );

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 8: malformed_msgpack_is_logged_not_fatal ───────────────────────────

/// Sending garbage bytes does not kill the loop; a valid batch after it is processed.
#[tokio::test]
async fn malformed_msgpack_is_logged_not_fatal() {
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

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("spawn sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse timed out"
        );

        // 2-frame with garbage payload.
        pub_handle
            .send_frames(vec![vec![], b"\x00garbage_not_msgpack".to_vec()])
            .await
            .expect("send garbage");

        // Now a valid batch — the loop must still be running.
        let valid = TestBatch(
            4.0,
            vec![bs_event(vec![999], None, vec![10, 20, 30, 40], 4, None)],
            None,
        );
        pub_handle.send_batch(&valid).await.expect("send valid");

        let msgs = sub.recv_n(1, Duration::from_secs(3)).await.expect("recv_n");
        let stores: usize = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();
        assert!(stores >= 1, "valid batch after garbage was not processed");

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 9: multiple_subscribers_receive_same_output ────────────────────────

/// Two SUB sockets both receive every egress batch from the same PUB.
#[tokio::test]
async fn multiple_subscribers_receive_same_output() {
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

        let mut sub1 = ZmqSubHandle::spawn(&egress_ep).await.expect("sub1");
        let mut sub2 = ZmqSubHandle::spawn(&egress_ep).await.expect("sub2");

        // Use sync_pulse on sub1 only (PUB fanout means sub2 also receives it).
        // Then drain both subs so no stale messages affect assertions.
        assert!(
            sync_pulse(&pub_handle, &mut sub1, Duration::from_secs(4)).await,
            "sub1 sync_pulse"
        );
        // Give sub2 a moment to receive the same pulse batch, then drain.
        let _ = sub2.recv_n(10, Duration::from_millis(300)).await;
        // Drain any residual messages from sub1.
        while sub1.rx.try_recv().is_ok() {}
        while sub2.rx.try_recv().is_ok() {}

        let batch = TestBatch(
            5.0,
            vec![bs_event(vec![777], None, vec![1, 2, 3, 4], 4, None)],
            None,
        );
        pub_handle.send_batch(&batch).await.expect("send");

        // Collect from both subs concurrently.
        let (msgs1, msgs2) = tokio::join!(
            sub1.recv_n(1, Duration::from_secs(3)),
            sub2.recv_n(1, Duration::from_secs(3)),
        );
        let msgs1 = msgs1.expect("sub1 recv");
        let msgs2 = msgs2.expect("sub2 recv");

        let stores1: usize = msgs1
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();
        let stores2: usize = msgs2
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .count();

        assert!(stores1 >= 1, "sub1 missed the store");
        assert!(stores2 >= 1, "sub2 missed the store");

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}
