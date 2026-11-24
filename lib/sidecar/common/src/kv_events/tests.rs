// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_kv_events::zmq_wire::{RawKvEvent, decode_event_batch};
use dynamo_llm::kv_router::publisher::KvEventSourceConfig;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::Instant;

use super::*;

mod vllm {
    crate::kv_event_proto!(vllm_proto);
}

mod sglang {
    crate::kv_event_proto!(sglang_kv_types::kv::v1);
}

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
    use vllm_proto as pb;
    let hash = |value| pb::BlockHash {
        value: Some(pb::block_hash::Value::UnsignedValue(value)),
    };
    let tokens = |ids: &[u64]| {
        Some(pb::BlockTokenIds {
            ids: Some(pb::block_token_ids::Ids::Single(pb::TokenIdList {
                token_ids: ids.to_vec(),
            })),
        })
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
    let image = 151_655;

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
            token_ids: tokens(&[1, image, image, 4, 5, 6, 7, 8]),
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
            token_ids: tokens(&[1, 2, 3, 4]),
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
            token_ids: tokens(&[9, 10, 11, 12]),
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
    assert_eq!(converted(vllm::batch(proto)), decoded(msgpack));

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
    assert!(vllm::batch(proto).expect("a batch").is_err());
}

#[test]
fn sglang_proto_events_convert_to_the_zmq_decoders_batch() {
    use sglang_kv_types::kv::v1 as pb;
    let hash = |value| pb::BlockHash {
        value: Some(pb::block_hash::Value::SignedValue(value)),
    };
    let tokens = |ids: &[u64]| {
        Some(pb::BlockTokenIds {
            ids: Some(pb::block_token_ids::Ids::Single(pb::TokenIdList {
                token_ids: ids.to_vec(),
            })),
        })
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
                data_parallel_rank: Some(0),
                events: vec![
                    stored(pb::BlockStored {
                        block_hashes: vec![hash(-5), hash(6)],
                        token_ids: Some(pb::BlockTokenIds {
                            ids: Some(pb::block_token_ids::Ids::Pairs(pb::TokenIdPairList {
                                pairs: pairs
                                    .iter()
                                    .map(|&(first, second)| pb::TokenIdPair { first, second })
                                    .collect(),
                            })),
                        }),
                        block_size: 4,
                        medium: Some("GPU".to_string()),
                        cache_salt: Some("tenant-b".to_string()),
                        session_id: Some("session-b".to_string()),
                        lora_name: Some("adapter-b".to_string()),
                        ..Default::default()
                    }),
                    stored(pb::BlockStored {
                        block_hashes: vec![hash(7)],
                        parent_block_hash: Some(hash(6)),
                        token_ids: tokens(&[9, wide, wide, 12]),
                        block_size: 4,
                        medium: Some("CPU_PINNED".to_string()),
                        ..Default::default()
                    }),
                    removed(pb::BlockRemoved {
                        block_hashes: vec![hash(-5)],
                        medium: Some("GPU".to_string()),
                        ..Default::default()
                    }),
                    event(pb::kv_cache_event::Event::AllBlocksCleared(
                        pb::AllBlocksCleared {},
                    )),
                ],
            },
        )),
    };
    assert_eq!(converted(sglang::batch(proto)), decoded(msgpack));
}

#[tokio::test]
async fn subscription_failures_choose_zmq_or_no_feed() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let failing = |code| {
        let attempts = attempts.clone();
        (0..2).map(move |rank| {
            let attempts = attempts.clone();
            let subscribe: Subscribe = Box::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Box::pin(future::ready(Err(tonic::Status::new(code, "injected"))))
            });
            (rank, subscribe)
        })
    };
    let zmq = || {
        let (endpoint, topic) = Default::default();
        Ok(vec![(0, KvEventFeed::Zmq { endpoint, topic })])
    };

    // An engine without the KvEvents service: its ZMQ publishers.
    let feeds = kv_event_feeds(failing(tonic::Code::Unimplemented), zmq).await;
    assert!(matches!(feeds.unwrap()[..], [(0, KvEventFeed::Zmq { .. })]));

    // A stream it lists but cannot serve: one attempt, no ZMQ fallback, no feed.
    attempts.store(0, Ordering::SeqCst);
    let no_fallback = || unreachable!("KV events are off");
    let feeds = kv_event_feeds(failing(tonic::Code::FailedPrecondition), no_fallback).await;
    assert!(feeds.unwrap().is_empty());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn an_ended_stream_is_reopened_at_once_unless_it_delivered_nothing() {
    let drt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = drt.namespace("kv-resubscribe").unwrap();
    let endpoint = endpoint.component("worker").unwrap().endpoint("generate");
    let config = KvEventSourceConfig::Push {
        idempotent_stores: false,
    };
    let publisher =
        KvEventPublisher::new_with_local_indexer(endpoint, 4, Some(config), false, 0, None);
    let publisher = Arc::new(publisher.unwrap());

    let batch = |sequence| -> KvBatchStream {
        let batch = KvEventBatch {
            ts: 0.0,
            events: vec![RawKvEvent::AllBlocksCleared { ownership: None }],
            data_parallel_rank: Some(0),
        };
        stream::iter([Ok((sequence, batch))]).boxed()
    };
    // The reopened streams: one batch, then nothing, then one that stays open.
    let mut reopened = [batch(1), stream::empty().boxed(), stream::pending().boxed()].into_iter();
    let (opened, mut opened_at) = mpsc::unbounded_channel();
    let source = SidecarKvSource {
        dp_rank: 0,
        image_token_id: None,
        idempotent_stores: false,
        feed: KvEventFeed::Grpc {
            stream: batch(0),
            resubscribe: Box::new(move || {
                opened.send(Instant::now()).unwrap();
                Box::pin(future::ready(Ok(reopened.next().unwrap())))
            }),
        },
    };
    let cancel = CancellationToken::new();
    let KvEventSource::Push { on_ready, .. } = source.into_push(cancel.clone()) else {
        unreachable!("sidecar sources are push sources");
    };
    on_ready(publisher.clone()).unwrap();

    let mut opened = async || {
        let opened = tokio::time::timeout(Duration::from_secs(60), opened_at.recv());
        opened.await.expect("the stream was not reopened").unwrap()
    };
    let (first, second, third) = (opened().await, opened().await, opened().await);
    // Each stream that delivered a batch is reopened at once; the empty one after the delay.
    assert_eq!(second - first, Duration::ZERO);
    assert_eq!(third - second, RESUBSCRIBE_DELAY);
    // Both batches were translated: the initial stream's and the reopened stream's.
    assert_eq!(publisher.next_event_id(), 2);
    cancel.cancel();
    drt.shutdown();
}
