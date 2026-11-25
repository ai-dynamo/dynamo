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
fn vllm_proto_events_convert_to_the_zmq_decoders_batch() {
    let hash = |value| pb::BlockHash {
        value: Some(pb::block_hash::Value::UnsignedValue(value)),
    };
    let key = |value| pb::ExtraKeyValue { value: Some(value) };
    let text = |text: &str| key(pb::extra_key_value::Value::Text(text.to_string()));
    // A frontend-padded multimodal identifier and its offset in the block.
    let mm_identifier = format!("{:016x}{}", 0x0123_4567_89ab_cdef_u64, "0".repeat(48));
    let mm = key(pb::extra_key_value::Value::Tuple(pb::ExtraKeyTuple {
        items: vec![
            text(&mm_identifier),
            key(pb::extra_key_value::Value::Integer(1)),
        ],
    }));
    let event = |event| pb::KvCacheEvent { event: Some(event) };
    let stored = |stored| event(pb::kv_cache_event::Event::BlockStored(stored));
    let removed = |removed| event(pb::kv_cache_event::Event::BlockRemoved(removed));
    let response = |events| pb::SubscribeKvEventsResponse {
        item: Some(pb::subscribe_kv_events_response::Item::Batch(
            pb::KvEventBatch {
                sequence_number: 7,
                ts: 1.5,
                data_parallel_rank: Some(0),
                events,
            },
        )),
    };
    let image: u32 = 151_655;

    let msgpack = json!([1.5, [
        // LoRA, cache salt and an image, in the main attention group.
        {
            "type": "BlockStored", "block_hashes": [101, 102], "parent_block_hash": null,
            "token_ids": [1, image, image, 4, 5, 6, 7, 8], "block_size": 4, "lora_id": null,
            "medium": "GPU", "lora_name": "adapter-a",
            "extra_keys": [
                ["adapter-a", [mm_identifier, 1], "dynamo-cache-salt:tenant-a"],
                ["adapter-a"],
            ],
            "group_idx": 0, "kv_cache_spec_kind": "full_attention", "session_id": "session-a",
        },
        // A sliding-window group, which the normalizer filters.
        {
            "type": "BlockStored", "block_hashes": [201], "parent_block_hash": null,
            "token_ids": [1, 2, 3, 4], "block_size": 4, "lora_id": null, "medium": "GPU",
            "lora_name": null, "group_idx": 1, "kv_cache_spec_kind": "sliding_window",
            "kv_cache_spec_sliding_window": 128,
        },
        // The chain continues on the CPU tier, with a locality.
        {
            "type": "BlockStored", "block_hashes": [103], "parent_block_hash": 102,
            "token_ids": [9, 10, 11, 12], "block_size": 4, "lora_id": null, "medium": "CPU",
            "lora_name": null, "group_idx": 0, "locality": "REMOTE",
        },
        {"type": "BlockRemoved", "block_hashes": [102], "medium": "GPU", "group_idx": 0},
        {"type": "AllBlocksCleared"},
    ], 0]);
    let proto = response(vec![
        stored(pb::BlockStored {
            block_hashes: vec![hash(101), hash(102)],
            token_ids: vec![1, image, image, 4, 5, 6, 7, 8],
            block_size: 4,
            medium: Some("GPU".to_string()),
            lora_name: Some("adapter-a".to_string()),
            extra_keys: vec![
                pb::BlockExtraKeys {
                    keys: vec![text("adapter-a"), mm, text("dynamo-cache-salt:tenant-a")],
                },
                pb::BlockExtraKeys {
                    keys: vec![text("adapter-a")],
                },
            ],
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".to_string()),
            session_id: Some("session-a".to_string()),
            ..Default::default()
        }),
        stored(pb::BlockStored {
            block_hashes: vec![hash(201)],
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            medium: Some("GPU".to_string()),
            group_idx: Some(1),
            kv_cache_spec_kind: Some("sliding_window".to_string()),
            kv_cache_spec_sliding_window: Some(128),
            ..Default::default()
        }),
        stored(pb::BlockStored {
            block_hashes: vec![hash(103)],
            parent_block_hash: Some(hash(102)),
            token_ids: vec![9, 10, 11, 12],
            block_size: 4,
            medium: Some("CPU".to_string()),
            group_idx: Some(0),
            locality: Some("REMOTE".to_string()),
            ..Default::default()
        }),
        removed(pb::BlockRemoved {
            block_hashes: vec![hash(102)],
            medium: Some("GPU".to_string()),
            group_idx: Some(0),
            ..Default::default()
        }),
        event(pb::kv_cache_event::Event::AllBlocksCleared(
            pb::AllBlocksCleared {},
        )),
    ]);
    assert_eq!(converted(batch(proto)), decoded(msgpack));

    // A byte digest fails the batch in the ZMQ decoder, and here.
    let digest = serde_bytes::Bytes::new(&[7; 32]);
    let msgpack = rmp_serde::to_vec(&(1.5, [("BlockRemoved", [digest])], 0)).unwrap();
    assert!(decode_event_batch(&msgpack).is_err());
    let digest = pb::BlockHash {
        value: Some(pb::block_hash::Value::Digest(vec![7; 32])),
    };
    let proto = response(vec![removed(pb::BlockRemoved {
        block_hashes: vec![digest],
        ..Default::default()
    })]);
    assert!(batch(proto).expect("a batch").is_err());
}
