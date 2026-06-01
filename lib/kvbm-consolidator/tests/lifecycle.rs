// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: backpressure/FIFO (test 7) and shutdown (test 13).
//!
//! cargo test --test lifecycle --features testing

mod common;

use std::time::Duration;

use common::{EventMirror, TestBatch, ZmqPubHandle, ZmqSubHandle, init_tracing, sync_pulse};
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_consolidator::{ConsolidatorBuilder, EventSource};

fn bs(hash: u64, tokens: Vec<u32>, block_size: usize) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(hash)],
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

// ─── test 7: backpressure_ingress ─────────────────────────────────────────────

/// Send 1000 distinct BlockStored events in 10 batches of 100.
/// All 1000 STOREs must arrive on egress; per-source FIFO confirmed via monotonic
/// token_id sequence embedded in events.
#[tokio::test]
async fn backpressure_ingress() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let pub_handle = ZmqPubHandle::spawn().await;
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
            .zmq_in(&pub_handle.endpoint)
            .poll_interval(Duration::from_millis(10))
            .build()
            .await
            .expect("build");

        let mut sub = ZmqSubHandle::spawn(&egress_ep).await.expect("sub");
        assert!(
            sync_pulse(&pub_handle, &mut sub, Duration::from_secs(4)).await,
            "sync_pulse"
        );

        // 10 batches × 100 events each.
        const TOTAL: u64 = 1000;
        const BATCH_SIZE: u64 = 100;
        for b in 0u64..(TOTAL / BATCH_SIZE) {
            let events: Vec<RawKvEvent> = (0u64..BATCH_SIZE)
                .map(|i| {
                    let global_id = b * BATCH_SIZE + i;
                    // Token encodes the monotonic sequence index so we can verify FIFO.
                    let tokens = vec![global_id as u32; 4];
                    bs(global_id, tokens, 4)
                })
                .collect();
            let batch = TestBatch(b as f64, events, None);
            pub_handle.send_batch(&batch).await.expect("send batch");
        }

        // Collect all 1000. Allow generous time — the publisher ticks at 10ms.
        let msgs = sub
            .recv_n(1000, Duration::from_secs(4))
            .await
            .expect("recv_n");

        let stores: Vec<&EventMirror> = msgs
            .iter()
            .flat_map(|(_, b)| b.1.iter())
            .filter(|e| matches!(e, EventMirror::BlockStored { .. }))
            .collect();

        assert_eq!(
            stores.len(),
            TOTAL as usize,
            "expected {TOTAL} STOREs, got {}",
            stores.len()
        );

        // Verify FIFO: token_ids[0] should appear in ascending order across all stores.
        // (Each block carries tokens [global_id; 4], so token_ids[0] is the batch index.)
        let token_sequence: Vec<i32> = stores
            .iter()
            .filter_map(|e| match e {
                EventMirror::BlockStored { token_ids, .. } => token_ids.first().copied(),
                _ => None,
            })
            .collect();

        for window in token_sequence.windows(2) {
            assert!(
                window[1] > window[0],
                "FIFO violated: token sequence not monotonically increasing near {:?}",
                window
            );
        }

        consolidator.shutdown().await;
    })
    .await
    .expect("timed out");
}

// ─── test 13: shutdown_cancellation ──────────────────────────────────────────

/// Spawn, push 100 events, call `consolidator.shutdown().await` within 1 second.
/// Must not hang; no lingering socket on the egress port.
#[tokio::test]
async fn shutdown_cancellation() {
    init_tracing();

    tokio::time::timeout(Duration::from_secs(5), async {
        let pub_handle = ZmqPubHandle::spawn().await;
        let egress_port = common::pick_port();
        let egress_ep = common::make_endpoint(egress_port);

        let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
            .zmq_in(&pub_handle.endpoint)
            .poll_interval(Duration::from_millis(50))
            .build()
            .await
            .expect("build");

        // Push 100 events before shutdown.
        for i in 0u64..100 {
            let tokens = vec![i as u32; 4];
            let batch = TestBatch(i as f64, vec![bs(i, tokens, 4)], None);
            let _ = pub_handle.send_batch(&batch).await;
        }

        // Shutdown must complete within 1 second.
        tokio::time::timeout(Duration::from_secs(1), consolidator.shutdown())
            .await
            .expect("shutdown timed out (> 1s)");
    })
    .await
    .expect("timed out");
}
