// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::kv_events::wire::decode_event_batch;
use serde_json::{Value, json};

use super::*;

/// What the ZMQ decoder returns for the engine's msgpack `[ts, events, rank]`.
fn decoded(batch: Value) -> Value {
    let payload = rmp_serde::to_vec(&batch).unwrap();
    serde_json::to_value(decode_event_batch(&payload).unwrap()).unwrap()
}

fn converted(batch: Option<KvBatch>) -> Value {
    let (sequence, batch) = batch.expect("a batch").expect("a convertible batch");
    assert_eq!(sequence, 7);
    serde_json::to_value(batch).unwrap()
}

#[test]
fn sglang_proto_events_convert_to_the_zmq_decoders_batch() {
    let tokens = |token_ids: &[u64]| {
        Some(pb::block_stored::TokenIds::Single(pb::TokenIdList {
            token_ids: token_ids.to_vec(),
        }))
    };
    let event = |event| pb::KvCacheEvent { event: Some(event) };
    let stored = |stored| event(pb::kv_cache_event::Event::BlockStored(stored));
    let removed = |removed| event(pb::kv_cache_event::Event::BlockRemoved(removed));
    let pairs: Vec<(u64, u64)> = (1..=8).map(|token| (token, token + 1)).collect();
    let wide = 1_u64 << 40;

    let msgpack = json!([2.5, [
        // An EAGLE page with a cache salt, a session and a LoRA.
        {
            "type": "BlockStored", "block_hashes": [-5, 6], "parent_block_hash": null,
            "token_ids": pairs, "block_size": 4, "lora_id": null, "medium": "GPU",
            "cache_salt": "tenant-b", "session_id": "session-b", "lora_name": "adapter-b",
        },
        // A multimodal pad that does not fit in 32 bits, on the host tier.
        {
            "type": "BlockStored", "block_hashes": [7], "parent_block_hash": 6,
            "token_ids": [9, wide, wide, 12], "block_size": 4, "lora_id": null,
            "medium": "CPU_PINNED",
        },
        {"type": "BlockRemoved", "block_hashes": [-5], "medium": "GPU"},
        {"type": "AllBlocksCleared"},
    ], 0]);
    let proto = pb::SubscribeKvEventsResponse {
        item: Some(pb::subscribe_kv_events_response::Item::Batch(
            pb::KvEventBatch {
                sequence_number: 7,
                ts: 2.5,
                attn_dp_rank: Some(0),
                events: vec![
                    stored(pb::BlockStored {
                        block_hashes: vec![-5, 6],
                        token_ids: Some(pb::block_stored::TokenIds::Pairs(pb::TokenIdPairList {
                            pairs: pairs
                                .iter()
                                .map(|&(first, second)| pb::TokenIdPair { first, second })
                                .collect(),
                        })),
                        block_size: 4,
                        medium: Some("GPU".to_string()),
                        cache_salt: Some("tenant-b".to_string()),
                        session_id: Some("session-b".to_string()),
                        lora_name: Some("adapter-b".to_string()),
                        ..Default::default()
                    }),
                    stored(pb::BlockStored {
                        block_hashes: vec![7],
                        parent_block_hash: Some(6),
                        token_ids: tokens(&[9, wide, wide, 12]),
                        block_size: 4,
                        medium: Some("CPU_PINNED".to_string()),
                        ..Default::default()
                    }),
                    removed(pb::BlockRemoved {
                        block_hashes: vec![-5],
                        medium: Some("GPU".to_string()),
                    }),
                    event(pb::kv_cache_event::Event::AllBlocksCleared(
                        pb::AllBlocksCleared {},
                    )),
                ],
            },
        )),
    };
    assert_eq!(converted(batch(proto)), decoded(msgpack));
}
