// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use rmp_serde::{from_slice, to_vec, to_vec_named};
use serde::Serialize;

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, ExternalSequenceBlockHash,
    KvCacheEventData, PlacementOwner, StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::filter::KvCacheSpecKind;
use super::types::Locality;
use super::*;

#[derive(Clone, Copy, Debug)]
enum TestEventKind {
    BlockStored,
    BlockRemoved,
}

#[test]
fn test_deserialize_bigram_block_stored_sequence() {
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11), BlockHashValue::Unsigned(12)],
        Option::<BlockHashValue>::None,
        vec![(10u32, 11u32), (11, 12), (12, 13), (13, 14)],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match event {
        RawKvEvent::BlockStored {
            token_ids,
            block_size,
            is_eagle,
            ..
        } => {
            assert_eq!(token_ids, vec![10, 11, 12, 13, 14]);
            assert_eq!(block_size, 2);
            assert_eq!(is_eagle, Some(true));
        }
        other => panic!("expected BlockStored, got {other:?}"),
    }
}

#[derive(Serialize)]
struct MapBlockStoredFixture {
    #[serde(rename = "type")]
    event_type: &'static str,
    block_hashes: Vec<BlockHashValue>,
    parent_block_hash: Option<BlockHashValue>,
    token_ids: Vec<u32>,
    block_size: usize,
    medium: Option<String>,
    lora_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_salt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_keys: Option<Vec<Option<Vec<String>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    locality: Option<&'static str>,
}

impl Default for MapBlockStoredFixture {
    fn default() -> Self {
        Self {
            event_type: "BlockStored",
            block_hashes: vec![BlockHashValue::Unsigned(11)],
            parent_block_hash: None,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: None,
            lora_name: None,
            cache_salt: None,
            extra_keys: None,
            locality: None,
        }
    }
}

#[derive(Serialize)]
struct MapBlockRemovedFixture {
    #[serde(rename = "type")]
    event_type: &'static str,
    block_hashes: Vec<BlockHashValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    locality: Option<&'static str>,
}

#[test]
fn test_deserialize_map_block_stored_cache_salt() {
    let encoded = to_vec_named(&MapBlockStoredFixture {
        cache_salt: Some("tenant-a".to_string()),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_deserialize_map_block_events_locality_symmetrically() {
    for (wire_value, expected) in [
        (Some("LOCAL"), Some(Locality::Local)),
        (Some("REMOTE"), Some(Locality::Remote)),
        (Some("FUTURE_VALUE"), Some(Locality::Unknown)),
        (None, None),
    ] {
        let encoded_events = [
            to_vec_named(&MapBlockStoredFixture {
                locality: wire_value,
                ..Default::default()
            })
            .unwrap(),
            to_vec_named(&MapBlockRemovedFixture {
                event_type: "BlockRemoved",
                block_hashes: vec![BlockHashValue::Unsigned(11)],
                locality: wire_value,
            })
            .unwrap(),
        ];

        for encoded in encoded_events {
            let event: RawKvEvent = from_slice(&encoded).unwrap();
            let locality = match event {
                RawKvEvent::BlockStored { locality, .. }
                | RawKvEvent::BlockRemoved { locality, .. } => locality,
                other => panic!("expected block event, got {other:?}"),
            };
            assert_eq!(locality, expected);
        }
    }
}

/// A `locality` key explicitly set to msgpack nil must decode like an
/// absent field: the map visitor reads it as `Option<Locality>`
/// (nil -> `None`) and `unwrap_or(None)` folds it into the absent case, so
/// FS/OBJ events carrying a nil field stay fail-closed downstream.
#[test]
fn test_deserialize_map_block_events_nil_locality_as_absent() {
    #[derive(Serialize)]
    struct NilLocalityBlockStored {
        #[serde(rename = "type")]
        event_type: &'static str,
        block_hashes: Vec<BlockHashValue>,
        parent_block_hash: Option<BlockHashValue>,
        token_ids: Vec<u32>,
        block_size: usize,
        // No `skip_serializing_if`: `None` encodes an explicit msgpack nil.
        locality: Option<&'static str>,
    }

    #[derive(Serialize)]
    struct NilLocalityBlockRemoved {
        #[serde(rename = "type")]
        event_type: &'static str,
        block_hashes: Vec<BlockHashValue>,
        locality: Option<&'static str>,
    }

    let encoded_events = [
        to_vec_named(&NilLocalityBlockStored {
            event_type: "BlockStored",
            block_hashes: vec![BlockHashValue::Unsigned(11)],
            parent_block_hash: None,
            token_ids: vec![10, 11],
            block_size: 2,
            locality: None,
        })
        .unwrap(),
        to_vec_named(&NilLocalityBlockRemoved {
            event_type: "BlockRemoved",
            block_hashes: vec![BlockHashValue::Unsigned(11)],
            locality: None,
        })
        .unwrap(),
    ];

    for encoded in encoded_events {
        let event: RawKvEvent = from_slice(&encoded).unwrap();
        let locality = match event {
            RawKvEvent::BlockStored { locality, .. }
            | RawKvEvent::BlockRemoved { locality, .. } => locality,
            other => panic!("expected block event, got {other:?}"),
        };
        assert_eq!(
            locality, None,
            "explicit nil must behave exactly like an absent field"
        );
    }
}

/// Locality matching is case-sensitive UPPERCASE: `"local"` / `"Remote"`
/// fall into the `#[serde(other)]` bucket and convert to a Shared placement
/// (fail closed), never to a local one.
#[test]
fn test_deserialize_mixed_case_locality_decodes_unknown_and_converts_shared() {
    let worker = WorkerWithDpRank::new(3, 0);

    for wire_value in ["local", "Remote"] {
        let encoded_events = [
            to_vec_named(&MapBlockStoredFixture {
                locality: Some(wire_value),
                ..Default::default()
            })
            .unwrap(),
            to_vec_named(&MapBlockRemovedFixture {
                event_type: "BlockRemoved",
                block_hashes: vec![BlockHashValue::Unsigned(11)],
                locality: Some(wire_value),
            })
            .unwrap(),
        ];

        for encoded in encoded_events {
            let event: RawKvEvent = from_slice(&encoded).unwrap();
            let locality = match &event {
                RawKvEvent::BlockStored { locality, .. }
                | RawKvEvent::BlockRemoved { locality, .. } => *locality,
                other => panic!("expected block event, got {other:?}"),
            };
            assert_eq!(
                locality,
                Some(Locality::Unknown),
                "`{wire_value}` must not match LOCAL/REMOTE"
            );

            let placement = convert_event(event, 1, 2, worker, &Arc::new(AtomicU32::new(0)), None)
                .expect("known medium should produce a placement");
            assert_eq!(placement.placement.owner, PlacementOwner::Shared);
        }
    }
}

#[test]
fn test_deserialize_extra_keys_cache_namespace_fallback() {
    let mm_hash = "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210";
    let encoded = to_vec_named(&MapBlockStoredFixture {
        lora_name: Some("adapter-a".to_string()),
        extra_keys: Some(vec![Some(vec![
            "adapter-a".to_string(),
            mm_hash.to_string(),
            "dynamo-cache-salt:tenant-a".to_string(),
        ])]),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_deserialize_hex_cache_namespace_is_not_multimodal() {
    let cache_namespace = "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210";
    let encoded = to_vec_named(&MapBlockStoredFixture {
        extra_keys: Some(vec![Some(vec![format!(
            "dynamo-cache-salt:{cache_namespace}"
        )])]),
        ..Default::default()
    })
    .unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    let RawKvEvent::BlockStored {
        cache_namespace: decoded_namespace,
        block_mm_infos,
        ..
    } = event
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(decoded_namespace.as_deref(), Some(cache_namespace));
    assert!(block_mm_infos.is_none());
}

fn block_stored_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn block_removed_sequence(
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<&'static str>,
) -> Vec<u8> {
    match (group_idx, kv_cache_spec_kind) {
        (Some(group_idx), Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (Some(group_idx), None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            group_idx,
        ))
        .unwrap(),
        (None, Some(kv_cache_spec_kind)) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            Option::<u32>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        (None, None) => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
        ))
        .unwrap(),
    }
}

fn sequence_with_group_idx(event_kind: TestEventKind, group_idx: Option<u32>) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, None),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, None),
    }
}

fn sequence_with_cache_spec_kind(
    event_kind: TestEventKind,
    group_idx: Option<u32>,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => block_stored_sequence(group_idx, Some(kv_cache_spec_kind)),
        TestEventKind::BlockRemoved => block_removed_sequence(group_idx, Some(kv_cache_spec_kind)),
    }
}

fn sequence_with_cache_spec_kind_without_group_idx_slot(
    event_kind: TestEventKind,
    kv_cache_spec_kind: &'static str,
) -> Vec<u8> {
    match event_kind {
        TestEventKind::BlockStored => to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Option::<String>::None,
            Option::<String>::None,
            Option::<u8>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
        TestEventKind::BlockRemoved => to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Option::<String>::None,
            kv_cache_spec_kind,
        ))
        .unwrap(),
    }
}

fn assert_parsed_event_kind(event: RawKvEvent, expected_kind: TestEventKind) {
    match (event, expected_kind) {
        (RawKvEvent::BlockStored { .. }, TestEventKind::BlockStored)
        | (RawKvEvent::BlockRemoved { .. }, TestEventKind::BlockRemoved) => {}
        (event, expected_kind) => {
            panic!("expected {expected_kind:?}, got {event:?}");
        }
    }
}

fn assert_event_metadata(
    event: &RawKvEvent,
    expected_group_idx: Option<u32>,
    expected_kind: Option<KvCacheSpecKind>,
    expected_sliding_window: Option<u32>,
) {
    let metadata = event.metadata();
    assert_eq!(metadata.group_idx, expected_group_idx);
    assert_eq!(metadata.kv_cache_spec_kind, expected_kind);
    assert_eq!(
        metadata.kv_cache_spec_sliding_window,
        expected_sliding_window
    );
}

#[test]
fn test_deserialize_sequence_accepts_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(0))).unwrap();

        assert_event_metadata(&event, Some(0), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_preserves_non_main_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, Some(1))).unwrap();

        assert_event_metadata(&event, Some(1), None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_missing_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, None)).unwrap();

        assert_event_metadata(&event, None, None, None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_legacy_sequences_have_unspecified_locality() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_group_idx(event_kind, None)).unwrap();

        let locality = match event {
            RawKvEvent::BlockStored { locality, .. }
            | RawKvEvent::BlockRemoved { locality, .. } => locality,
            other => panic!("expected block event, got {other:?}"),
        };
        assert_eq!(locality, None);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_with_nonzero_group_idx() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
            event_kind,
            Some(3),
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_sequence_accepts_main_attention_kind_without_group_idx_slot() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind_without_group_idx_slot(
            event_kind,
            "full_attention",
        ))
        .unwrap();

        assert_event_metadata(&event, None, Some(KvCacheSpecKind::FullAttention), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_deserialize_block_stored_sequence_preserves_block_mm_infos_and_metadata() {
    let block_mm_infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash: 99,
            offsets: vec![(0, 1)],
        }],
    })];
    let raw_event = (
        "BlockStored",
        vec![BlockHashValue::Unsigned(11)],
        Option::<BlockHashValue>::None,
        vec![10u32, 11],
        2usize,
        Option::<u64>::None,
        Option::<String>::None,
        Option::<String>::None,
        Option::<u8>::None,
        block_mm_infos.clone(),
        3u32,
        "full_attention",
    );
    let encoded = to_vec(&raw_event).unwrap();
    let event: RawKvEvent = from_slice(&encoded).unwrap();

    match &event {
        RawKvEvent::BlockStored {
            block_mm_infos: Some(parsed),
            ..
        } => assert_eq!(parsed, &block_mm_infos),
        other => panic!("expected BlockStored with block_mm_infos, got {other:?}"),
    }
    assert_event_metadata(&event, Some(3), Some(KvCacheSpecKind::FullAttention), None);

    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(event, worker).is_some());
    assert!(normalizer.preprocess(remove, worker).is_some());
}

#[test]
fn test_deserialize_sequence_preserves_non_main_attention_kind_with_group_idx_zero() {
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        let event: RawKvEvent =
            from_slice(&sequence_with_cache_spec_kind(event_kind, Some(0), "mamba")).unwrap();

        assert_event_metadata(&event, Some(0), Some(KvCacheSpecKind::Mamba), None);
        assert_parsed_event_kind(event, event_kind);
    }
}

#[test]
fn test_normalizer_ignores_non_main_group_idx_without_metadata() {
    let raw_event: RawKvEvent =
        from_slice(&block_removed_sequence(Some(1), None)).expect("valid raw event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_eq!(
        normalizer
            .preprocess_with_reason(raw_event, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::UnlearnedGroupIdx
    );
}

#[test]
fn test_normalizer_ignores_map_serialized_non_main_attention_kind() {
    #[derive(serde::Serialize)]
    struct MapBlockStoredEvent {
        #[serde(rename = "type")]
        event_type: &'static str,
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        block_size: usize,
        group_idx: Option<u32>,
        kv_cache_spec_kind: Option<&'static str>,
    }

    let event = MapBlockStoredEvent {
        event_type: "BlockStored",
        block_hashes: vec![11],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        group_idx: Some(1),
        kv_cache_spec_kind: Some("mamba"),
    };
    let encoded = rmp_serde::to_vec_named(&(0.0, vec![event], Some(0_i32)))
        .expect("serialize raw event batch");
    let mut batch = decode_event_batch(&encoded).expect("deserialize raw event batch");
    let decoded = batch.events.pop().expect("batch should contain event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert_event_metadata(&decoded, Some(1), Some(KvCacheSpecKind::Mamba), None);
    assert_eq!(
        normalizer
            .preprocess_with_reason(decoded, WorkerWithDpRank::new(3, 0))
            .unwrap_err(),
        ZmqEventFilterReason::NonMainAttentionKind
    );
}

#[test]
fn test_normalizer_metadata_is_dp_rank_scoped() {
    let store: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(3),
        "full_attention",
    ))
    .expect("valid store event");
    let same_rank_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid same-rank remove event");
    let different_rank_remove: RawKvEvent = from_slice(&block_removed_sequence(Some(3), None))
        .expect("valid different-rank remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);

    assert!(
        normalizer
            .preprocess(store, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(same_rank_remove, WorkerWithDpRank::new(7, 0))
            .is_some()
    );
    assert!(
        normalizer
            .preprocess(different_rank_remove, WorkerWithDpRank::new(7, 1))
            .is_none()
    );
}

#[test]
fn test_normalizer_does_not_learn_metadata_from_remove_events() {
    let metadata_remove: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockRemoved,
        Some(3),
        "full_attention",
    ))
    .expect("valid metadata remove event");
    let bare_remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(3), None)).expect("valid bare remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(7, 0);

    assert!(normalizer.preprocess(metadata_remove, worker).is_some());
    assert!(normalizer.preprocess(bare_remove, worker).is_none());
}

#[test]
fn test_normalizer_propagates_cache_namespace_from_parent() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let parent = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };
    let child = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(2)],
        parent_block_hash: Some(BlockHashValue::Unsigned(1)),
        token_ids: vec![12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };

    assert!(normalizer.preprocess(parent, worker).is_some());
    let child = normalizer.preprocess(child, worker).unwrap();

    let CacheNamespaceState::Namespaced(parent_namespace) =
        &normalizer.cache_namespaces[&(worker, 1)]
    else {
        panic!("expected namespaced parent");
    };
    let CacheNamespaceState::Namespaced(child_namespace) =
        &normalizer.cache_namespaces[&(worker, 2)]
    else {
        panic!("expected namespaced child");
    };
    assert!(Arc::ptr_eq(parent_namespace, child_namespace));

    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_normalizer_shares_cache_namespace_across_blocks() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1), BlockHashValue::Unsigned(2)],
        parent_block_hash: None,
        token_ids: vec![10, 11, 12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };

    assert!(normalizer.preprocess(event, worker).is_some());

    let CacheNamespaceState::Namespaced(first) = &normalizer.cache_namespaces[&(worker, 1)] else {
        panic!("expected first block namespace");
    };
    let CacheNamespaceState::Namespaced(second) = &normalizer.cache_namespaces[&(worker, 2)] else {
        panic!("expected second block namespace");
    };
    assert!(Arc::ptr_eq(first, second));
}

#[test]
fn test_normalizer_rejects_ambiguous_parent_cache_namespace() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let stored =
        |cache_namespace: Option<&str>, block_hashes, parent_block_hash| RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: None,
            lora_name: None,
            cache_namespace: cache_namespace.map(str::to_owned),
            block_mm_infos: None,
            is_eagle: Some(false),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
        };

    let parent_a = stored(Some("tenant-a"), vec![BlockHashValue::Unsigned(1)], None);
    let parent_b = stored(Some("tenant-b"), vec![BlockHashValue::Unsigned(1)], None);
    let child = stored(
        None,
        vec![BlockHashValue::Unsigned(2)],
        Some(BlockHashValue::Unsigned(1)),
    );

    assert!(normalizer.preprocess(parent_a, worker).is_some());
    assert!(normalizer.preprocess(parent_b, worker).is_some());
    assert_eq!(
        normalizer
            .preprocess_with_reason(child, worker)
            .expect_err("ambiguous parent must be rejected"),
        ZmqEventFilterReason::AmbiguousCacheNamespace
    );
}

#[test]
fn test_normalizer_treats_empty_namespace_as_absent() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let parent = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some("tenant-a".to_string()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };
    let child = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(2)],
        parent_block_hash: Some(BlockHashValue::Unsigned(1)),
        token_ids: vec![12, 13],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: Some(String::new()),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };

    assert!(normalizer.preprocess(parent, worker).is_some());
    let child = normalizer.preprocess(child, worker).unwrap();
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

#[test]
fn test_normalizer_ignores_non_main_attention_kind_with_group_idx_zero() {
    let raw_event: RawKvEvent = from_slice(&sequence_with_cache_spec_kind(
        TestEventKind::BlockStored,
        Some(0),
        "mamba",
    ))
    .expect("valid raw event");
    let remove: RawKvEvent =
        from_slice(&block_removed_sequence(Some(0), None)).expect("valid remove event");
    let mut normalizer = ZmqEventNormalizer::new(2);
    let worker = WorkerWithDpRank::new(3, 0);

    assert!(normalizer.preprocess(raw_event, worker).is_none());
    assert!(normalizer.preprocess(remove, worker).is_none());
}

#[test]
fn test_convert_event_bigram_emits_eagle_windows() {
    let raw_event = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(21), BlockHashValue::Unsigned(22)],
        parent_block_hash: None,
        token_ids: vec![10, 11, 12, 13, 14],
        block_size: 2,
        medium: None,
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: Some(true),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement_event = convert_event(
        raw_event,
        7,
        2,
        WorkerWithDpRank::new(3, 0),
        &warning_count,
        None,
    );

    match placement_event.unwrap().event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.blocks.len(), 2);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(21)
            );
            assert_eq!(
                store_data.blocks[1].block_hash,
                ExternalSequenceBlockHash(22)
            );

            let expected_first = compute_block_hash_for_seq(
                &[10, 11, 12],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );
            let expected_second = compute_block_hash_for_seq(
                &[12, 13, 14],
                2,
                BlockHashOptions {
                    is_eagle: Some(true),
                    ..Default::default()
                },
            );

            assert_eq!(store_data.blocks[0].tokens_hash, expected_first[0]);
            assert_eq!(store_data.blocks[1].tokens_hash, expected_second[0]);
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
}

struct CpuBlockStoredFixture<'a> {
    block_hashes: &'a [u64],
    token_ids: &'a [u32],
    block_size: usize,
    parent_block_hash: Option<u64>,
}

fn cpu_block_stored(fixture: CpuBlockStoredFixture<'_>) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: fixture
            .block_hashes
            .iter()
            .copied()
            .map(BlockHashValue::Unsigned)
            .collect(),
        parent_block_hash: fixture.parent_block_hash.map(BlockHashValue::Unsigned),
        token_ids: fixture.token_ids.to_vec(),
        block_size: fixture.block_size,
        medium: Some("CPU".to_string()),
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    }
}

#[test]
fn cpu_event_with_placeholder_payload_is_dropped_safely() {
    let raw = cpu_block_stored(CpuBlockStoredFixture {
        block_hashes: &[201, 202, 203],
        token_ids: &[],
        block_size: 0,
        parent_block_hash: None,
    });
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(
        raw,
        42,
        16,
        WorkerWithDpRank::new(7, 0),
        &warning_count,
        None,
    )
    .unwrap();

    assert_eq!(placement.placement.tier, StorageTier::HostPinned);
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            assert!(store_data.parent_hash.is_none());
            assert!(store_data.blocks.is_empty());
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
    assert!(warning_count.load(Ordering::Relaxed) >= 1);
}

#[test]
fn cpu_event_with_full_payload_is_indexable() {
    let raw = cpu_block_stored(CpuBlockStoredFixture {
        block_hashes: &[201, 202],
        token_ids: &[10, 11, 12, 13, 14, 15, 16, 17],
        block_size: 4,
        parent_block_hash: Some(200),
    });
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(
        raw,
        43,
        4,
        WorkerWithDpRank::new(7, 0),
        &warning_count,
        None,
    )
    .unwrap();

    assert_eq!(placement.placement.tier, StorageTier::HostPinned);
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => {
            assert_eq!(store_data.parent_hash, Some(ExternalSequenceBlockHash(200)));
            assert_eq!(store_data.blocks.len(), 2);
            assert_eq!(
                store_data.blocks[0].block_hash,
                ExternalSequenceBlockHash(201)
            );
            assert_eq!(
                store_data.blocks[1].block_hash,
                ExternalSequenceBlockHash(202)
            );
        }
        other => panic!("expected Stored event, got {other:?}"),
    }
    assert_eq!(warning_count.load(Ordering::Relaxed), 0);
}

fn raw_placement_event(
    event_kind: TestEventKind,
    medium: Option<&str>,
    locality: Option<Locality>,
) -> RawKvEvent {
    match event_kind {
        TestEventKind::BlockStored => RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(1)],
            parent_block_hash: None,
            token_ids: vec![10, 11],
            block_size: 2,
            medium: medium.map(str::to_owned),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: Some(false),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality,
        },
        TestEventKind::BlockRemoved => RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(1)],
            medium: medium.map(str::to_owned),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality,
        },
    }
}

#[test]
fn test_convert_event_resolves_locality_owner_and_storage_tier_symmetrically() {
    let worker = WorkerWithDpRank::new(3, 1);
    let cases = [
        (None, None, StorageTier::Device, false),
        (Some("CPU"), None, StorageTier::HostPinned, false),
        (Some("FS"), None, StorageTier::Disk, true),
        (Some("OBJ"), None, StorageTier::External, true),
        (Some("FS"), Some(Locality::Local), StorageTier::Disk, false),
        (
            Some("GPU"),
            Some(Locality::Remote),
            StorageTier::Device,
            true,
        ),
        (
            Some("GPU"),
            Some(Locality::Unknown),
            StorageTier::Device,
            true,
        ),
    ];

    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for (medium, locality, expected_tier, expected_shared) in cases {
            let placement = convert_event(
                raw_placement_event(event_kind, medium, locality),
                1,
                2,
                worker,
                &Arc::new(AtomicU32::new(0)),
                None,
            )
            .expect("known medium should produce a placement");

            assert_eq!(placement.placement.tier, expected_tier);
            let expected_owner = if expected_shared {
                PlacementOwner::Shared
            } else {
                PlacementOwner::LocalWorker(worker)
            };
            assert_eq!(placement.placement.owner, expected_owner);
            assert!(matches!(
                (event_kind, placement.event.data),
                (TestEventKind::BlockStored, KvCacheEventData::Stored(_))
                    | (TestEventKind::BlockRemoved, KvCacheEventData::Removed(_))
            ));
        }
    }
}

#[test]
fn test_convert_event_rejects_unknown_medium_instead_of_defaulting_to_device() {
    let worker = WorkerWithDpRank::new(3, 0);

    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for locality in [None, Some(Locality::Local)] {
            let placement = convert_event(
                raw_placement_event(event_kind, Some("FUTURE_TIER"), locality),
                1,
                2,
                worker,
                &Arc::new(AtomicU32::new(0)),
                None,
            );
            assert!(
                placement.is_none(),
                "unknown medium must be dropped even with locality {locality:?}"
            );
        }
    }
}

fn namespaced_block_stored(
    block_hash: u64,
    parent_block_hash: Option<u64>,
    cache_namespace: Option<&str>,
    locality: Locality,
) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(block_hash)],
        parent_block_hash: parent_block_hash.map(BlockHashValue::Unsigned),
        token_ids: vec![10, 11],
        block_size: 2,
        medium: Some("GPU".to_string()),
        lora_name: None,
        cache_namespace: cache_namespace.map(str::to_owned),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: Some(locality),
    }
}

#[test]
fn test_remote_events_do_not_pollute_local_cache_namespace_state() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);
    let remote_parent = namespaced_block_stored(1, None, Some("remote-tenant"), Locality::Remote);
    let local_parent = namespaced_block_stored(1, None, Some("local-tenant"), Locality::Local);
    let remote_remove = RawKvEvent::BlockRemoved {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        medium: Some("GPU".to_string()),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: Some(Locality::Remote),
    };
    let local_child = namespaced_block_stored(2, Some(1), None, Locality::Local);

    assert!(normalizer.preprocess(remote_parent, worker).is_some());
    assert!(normalizer.cache_namespaces.is_empty());
    assert!(normalizer.preprocess(local_parent, worker).is_some());
    assert!(normalizer.preprocess(remote_remove, worker).is_some());

    let local_child = normalizer
        .preprocess(local_child, worker)
        .expect("remote events must not make the local parent ambiguous");
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = local_child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("local-tenant"));
}

/// A LOCAL lower-tier placeholder (vLLM FS/OBJ events are hash-only:
/// token_ids=[], block_size=0, no extra_keys) must not mutate normalizer
/// state: it may not flip a Namespaced GPU hash to Ambiguous, and salted
/// chain continuations must keep inheriting the parent namespace.
#[test]
fn test_local_lower_tier_placeholders_do_not_pollute_cache_namespace_state() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);

    let salted_gpu = namespaced_block_stored(1, None, Some("tenant-a"), Locality::Local);
    assert!(normalizer.preprocess(salted_gpu, worker).is_some());

    let fs_placeholder = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(1)],
        parent_block_hash: None,
        token_ids: vec![],
        block_size: 0,
        medium: Some("FS".to_string()),
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: Some(0),
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: Some(Locality::Local),
    };
    assert!(normalizer.preprocess(fs_placeholder, worker).is_some());
    assert!(matches!(
        &normalizer.cache_namespaces[&(worker, 1)],
        CacheNamespaceState::Namespaced(ns) if ns.as_ref() == "tenant-a"
    ));

    // The salted chain continuation still inherits the parent namespace.
    let gpu_child = namespaced_block_stored(2, Some(1), None, Locality::Local);
    let child = normalizer
        .preprocess(gpu_child, worker)
        .expect("local FS placeholder must not make the parent ambiguous");
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}
