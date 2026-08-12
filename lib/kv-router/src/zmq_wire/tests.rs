// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use rmp_serde::{from_slice, to_vec, to_vec_named};
use serde::Serialize;

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, ExternalSequenceBlockHash,
    KvCacheEventData, PlacementEvent, PlacementOwner, StorageTier, WorkerWithDpRank,
    compute_block_hash_for_seq,
};

use super::filter::KvCacheSpecKind;
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

/// An empty-string `medium` is normalized to absent (like empty `cache_salt`),
/// so an unset medium stays on the device tier and is indexed rather than
/// failing closed as an unknown medium. Covers store and remove.
#[test]
fn test_deserialize_empty_medium_normalizes_to_absent_device() {
    let worker = WorkerWithDpRank::new(3, 0);

    let stored = to_vec_named(&MapBlockStoredFixture {
        medium: Some(String::new()),
        ..Default::default()
    })
    .unwrap();
    let stored: RawKvEvent = from_slice(&stored).unwrap();
    assert_eq!(
        stored.medium(),
        None,
        "empty medium must normalize to absent"
    );
    let placement =
        convert_placement(stored, worker).expect("empty-medium store must be indexed, not dropped");
    assert_eq!(placement.placement.tier, StorageTier::Device);

    let removed = to_vec_named(&MapBlockRemovedFixtureWithMedium {
        event_type: "BlockRemoved",
        block_hashes: vec![BlockHashValue::Unsigned(11)],
        medium: Some(String::new()),
    })
    .unwrap();
    let removed: RawKvEvent = from_slice(&removed).unwrap();
    assert_eq!(
        removed.medium(),
        None,
        "empty medium must normalize to absent"
    );
}

#[derive(Serialize)]
struct MapBlockRemovedFixtureWithMedium {
    #[serde(rename = "type")]
    event_type: &'static str,
    block_hashes: Vec<BlockHashValue>,
    medium: Option<String>,
}

/// The empty-medium normalization must also cover legacy positional (tuple)
/// events: an empty-string `medium` in the sequence slot decodes as absent and
/// stays on the device tier. Covers store and remove.
#[test]
fn test_deserialize_legacy_sequence_empty_medium_normalizes_to_absent_device() {
    let worker = WorkerWithDpRank::new(3, 0);

    let encoded_events = [
        to_vec(&(
            "BlockStored",
            vec![BlockHashValue::Unsigned(11)],
            Option::<BlockHashValue>::None,
            vec![10u32, 11],
            2usize,
            Option::<u64>::None,
            Some(String::new()),
        ))
        .unwrap(),
        to_vec(&(
            "BlockRemoved",
            vec![BlockHashValue::Unsigned(11)],
            Some(String::new()),
        ))
        .unwrap(),
    ];

    for encoded in encoded_events {
        let event: RawKvEvent = from_slice(&encoded).unwrap();
        assert_eq!(
            event.medium(),
            None,
            "empty sequence medium must normalize to absent"
        );
        let placement = convert_placement(event, worker)
            .expect("empty-medium legacy event must be indexed, not dropped");
        assert_eq!(placement.placement.tier, StorageTier::Device);
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

// ---- locality decode + storage-tier gating (this PR) ----

/// LOCAL/REMOTE decode to their variants, any other string folds into
/// `Unknown`, and an absent field stays `None`. Store and remove behave
/// symmetrically.
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

/// Legacy positional (tuple) events predate the locality field, so they must
/// decode with `locality == None` (byte-compat lock).
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

/// An explicit msgpack nil `locality` must behave exactly like an absent field
/// (`None`), so nil-carrying events stay fail-open-to-local downstream.
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

    let encoded = to_vec_named(&NilLocalityBlockStored {
        event_type: "BlockStored",
        block_hashes: vec![BlockHashValue::Unsigned(11)],
        parent_block_hash: None,
        token_ids: vec![10, 11],
        block_size: 2,
        locality: None,
    })
    .unwrap();

    let event: RawKvEvent = from_slice(&encoded).unwrap();
    let RawKvEvent::BlockStored { locality, .. } = event else {
        panic!("expected BlockStored");
    };
    assert_eq!(
        locality, None,
        "explicit nil must behave exactly like an absent field"
    );
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

fn convert_placement(event: RawKvEvent, worker: WorkerWithDpRank) -> Option<PlacementEvent> {
    convert_event(event, 1, 2, worker, &Arc::new(AtomicU32::new(0)), None)
}

/// Absent/LOCAL locality keeps events worker-local at their medium's tier
/// (STORAGE routes to Disk, this PR's core fix); REMOTE or any unknown locality
/// fails closed to a dropped event for both store and remove.
#[test]
fn test_convert_event_gates_on_locality_and_storage_tier() {
    let worker = WorkerWithDpRank::new(3, 1);

    let accepted = [
        (None, None, StorageTier::Device),
        (Some("GPU"), None, StorageTier::Device),
        (Some("CPU"), None, StorageTier::HostPinned),
        (Some("STORAGE"), None, StorageTier::Disk),
        (Some("STORAGE"), Some(Locality::Local), StorageTier::Disk),
        (Some("CPU"), Some(Locality::Local), StorageTier::HostPinned),
    ];
    let dropped = [
        (Some("STORAGE"), Some(Locality::Remote)),
        (Some("STORAGE"), Some(Locality::Unknown)),
        (Some("GPU"), Some(Locality::Remote)),
        (Some("GPU"), Some(Locality::Unknown)),
    ];

    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for (medium, locality, expected_tier) in accepted {
            let raw = raw_placement_event(event_kind, medium, locality);
            let placement = convert_placement(raw, worker)
                .expect("local/unspecified events must produce a placement");
            assert_eq!(placement.placement.tier, expected_tier, "medium={medium:?}");
            assert_eq!(
                placement.placement.owner,
                PlacementOwner::LocalWorker(worker),
                "medium={medium:?} locality={locality:?} must stay worker-local"
            );
        }
        for (medium, locality) in dropped {
            let raw = raw_placement_event(event_kind, medium, locality);
            assert!(
                convert_placement(raw, worker).is_none(),
                "medium={medium:?} locality={locality:?} must be dropped"
            );
        }
    }
}

/// Non-local events must be classified as filtered in `preprocess_with_reason`
/// (so the listener never burns an event id on them). The gate runs before the
/// lower-tier bypass, so it fires for every tier — Disk/External (which would
/// otherwise bypass) and Device (GPU) alike.
#[test]
fn test_preprocess_rejects_non_local_locality_for_every_tier() {
    let worker = WorkerWithDpRank::new(3, 0);

    let rejected = [
        (Some("STORAGE"), Locality::Remote),
        (Some("STORAGE"), Locality::Unknown),
        (Some("GPU"), Locality::Remote),
        (None, Locality::Remote),
    ];
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for (medium, locality) in rejected {
            let mut normalizer = ZmqEventNormalizer::new(2);
            let raw = raw_placement_event(event_kind, medium, Some(locality));
            assert_eq!(
                normalizer.preprocess_with_reason(raw, worker).unwrap_err(),
                ZmqEventFilterReason::NonLocalLocality,
                "medium={medium:?} locality={locality:?} must filter as non-local"
            );
        }
    }

    // LOCAL / absent locality still pass preprocess (STORAGE takes the bypass).
    for locality in [Some(Locality::Local), None] {
        let mut normalizer = ZmqEventNormalizer::new(2);
        let raw = raw_placement_event(TestEventKind::BlockStored, Some("STORAGE"), locality);
        assert!(
            normalizer.preprocess_with_reason(raw, worker).is_ok(),
            "STORAGE locality={locality:?} must pass preprocess"
        );
    }
}

/// Unrecognized media must be classified as filtered in `preprocess_with_reason`
/// (reason `UnknownMedium`), so the listener records an intentional filter rather
/// than accepting the event, burning a next_event_id, and dropping it only in
/// conversion — an id gap the event processor misreads as an engine drop. This is
/// the same trap the locality gate avoids. Recognized tiers are never filtered
/// here: Device/HostPinned stay on the normalizer path and Disk/External bypass.
#[test]
fn test_preprocess_rejects_unknown_medium() {
    let worker = WorkerWithDpRank::new(3, 0);

    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for medium in ["FS", "OBJ", "XYZ"] {
            let mut normalizer = ZmqEventNormalizer::new(2);
            let raw = raw_placement_event(event_kind, Some(medium), Some(Locality::Local));
            assert_eq!(
                normalizer.preprocess_with_reason(raw, worker).unwrap_err(),
                ZmqEventFilterReason::UnknownMedium,
                "medium={medium} must filter as unknown_medium"
            );
        }
    }

    // Recognized media are never rejected as unknown: GPU/CPU stay on the
    // normalizer path and STORAGE takes the lower-tier bypass.
    for medium in ["GPU", "CPU", "STORAGE"] {
        let mut normalizer = ZmqEventNormalizer::new(2);
        let raw = raw_placement_event(
            TestEventKind::BlockStored,
            Some(medium),
            Some(Locality::Local),
        );
        assert!(
            normalizer.preprocess_with_reason(raw, worker).is_ok(),
            "recognized medium={medium} must pass preprocess"
        );
    }
}

/// Mixed-case locality (`"local"`, `"Remote"`) is not UPPERCASE, so it folds to
/// `Unknown` and the event is dropped — it never decodes to a local placement.
#[test]
fn test_deserialize_mixed_case_locality_is_unknown_and_dropped() {
    let worker = WorkerWithDpRank::new(3, 0);
    for wire_value in ["local", "Remote", "GLOBAL"] {
        let encoded = to_vec_named(&MapBlockStoredFixture {
            medium: Some("STORAGE".to_string()),
            locality: Some(wire_value),
            ..Default::default()
        })
        .unwrap();
        let event: RawKvEvent = from_slice(&encoded).unwrap();
        let RawKvEvent::BlockStored { locality, .. } = &event else {
            panic!("expected BlockStored");
        };
        assert_eq!(
            *locality,
            Some(Locality::Unknown),
            "`{wire_value}` must fold to Unknown"
        );
        assert!(
            convert_placement(event, worker).is_none(),
            "`{wire_value}` locality must be dropped"
        );
    }
}

/// A placeholder STORAGE store (block_size=0, empty tokens) still resolves to
/// the Disk tier but yields an empty block list — a harmless index no-op — via
/// the existing block-size validation in `create_stored_blocks`.
#[test]
fn test_storage_placeholder_store_is_indexed_as_disk_noop() {
    let raw = RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(301)],
        parent_block_hash: None,
        token_ids: vec![],
        block_size: 0,
        medium: Some("STORAGE".to_string()),
        lora_name: None,
        cache_namespace: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    };
    let warning_count = Arc::new(AtomicU32::new(0));
    let placement = convert_event(
        raw,
        44,
        16,
        WorkerWithDpRank::new(7, 0),
        &warning_count,
        None,
    )
    .expect("placeholder STORAGE store still produces a placement");

    assert_eq!(placement.placement.tier, StorageTier::Disk);
    assert_eq!(
        placement.placement.owner,
        PlacementOwner::LocalWorker(WorkerWithDpRank::new(7, 0))
    );
    match placement.event.data {
        KvCacheEventData::Stored(store_data) => assert!(store_data.blocks.is_empty()),
        other => panic!("expected Stored event, got {other:?}"),
    }
    assert!(warning_count.load(Ordering::Relaxed) >= 1);
}

fn namespaced_block_stored(
    block_hash: u64,
    parent_block_hash: Option<u64>,
    cache_namespace: Option<&str>,
    medium: &str,
) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(block_hash)],
        parent_block_hash: parent_block_hash.map(BlockHashValue::Unsigned),
        token_ids: vec![10, 11],
        block_size: 2,
        medium: Some(medium.to_string()),
        lora_name: None,
        cache_namespace: cache_namespace.map(str::to_owned),
        block_mm_infos: None,
        is_eagle: Some(false),
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
    }
}

/// A lower-tier STORAGE event must not mutate normalizer state: it may not flip
/// a Namespaced GPU hash to Ambiguous, and a salted GPU chain continuation must
/// keep inheriting the parent namespace. This is the device-only preprocess
/// guard (§2.5) — the scenario called out in the original review.
#[test]
fn test_lower_tier_events_do_not_pollute_cache_namespace_state() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);

    let salted_gpu = namespaced_block_stored(1, None, Some("tenant-a"), "GPU");
    assert!(normalizer.preprocess(salted_gpu, worker).is_some());

    // A STORAGE event reusing the same hash must skip the normalizer entirely.
    let storage_same_hash = namespaced_block_stored(1, None, None, "STORAGE");
    assert!(normalizer.preprocess(storage_same_hash, worker).is_some());
    assert!(matches!(
        &normalizer.cache_namespaces[&(worker, 1)],
        CacheNamespaceState::Namespaced(ns) if ns.as_ref() == "tenant-a"
    ));

    // The salted chain continuation still inherits the parent namespace.
    let gpu_child = namespaced_block_stored(2, Some(1), None, "GPU");
    let child = normalizer
        .preprocess(gpu_child, worker)
        .expect("STORAGE event must not make the parent ambiguous");
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}

/// Unrecognized media (e.g. vLLM 0.26.0 `FS` / `OBJ`, or any future string) must
/// be dropped in conversion, never silently indexed on the device (G1) primary
/// tree — for both store and remove, and regardless of locality.
#[test]
fn test_convert_event_rejects_unknown_medium_instead_of_defaulting_to_device() {
    let worker = WorkerWithDpRank::new(3, 0);
    for event_kind in [TestEventKind::BlockStored, TestEventKind::BlockRemoved] {
        for medium in ["FS", "OBJ", "XYZ"] {
            for locality in [None, Some(Locality::Local)] {
                let raw = raw_placement_event(event_kind, Some(medium), locality);
                assert!(
                    convert_placement(raw, worker).is_none(),
                    "unrecognized medium {medium} must be dropped (locality {locality:?})"
                );
            }
        }
    }
}

/// Unrecognized media (vLLM 0.26.0 `FS` / `OBJ`) fail closed in preprocess, so
/// they never reach — let alone mutate — the salted-namespace state: a salted GPU
/// hash stays `Namespaced` and later salted GPU children keep inheriting it.
#[test]
fn test_unrecognized_media_do_not_pollute_cache_namespace_state() {
    let worker = WorkerWithDpRank::new(7, 0);
    let mut normalizer = ZmqEventNormalizer::new(2);

    let salted_gpu = namespaced_block_stored(1, None, Some("tenant-a"), "GPU");
    assert!(normalizer.preprocess(salted_gpu, worker).is_some());

    // An FS event reusing the same hash fails closed before any namespace work.
    let fs_same_hash = namespaced_block_stored(1, None, None, "FS");
    assert_eq!(
        normalizer
            .preprocess_with_reason(fs_same_hash, worker)
            .unwrap_err(),
        ZmqEventFilterReason::UnknownMedium
    );
    assert!(matches!(
        &normalizer.cache_namespaces[&(worker, 1)],
        CacheNamespaceState::Namespaced(ns) if ns.as_ref() == "tenant-a"
    ));

    // The salted chain continuation still inherits the parent namespace.
    let gpu_child = namespaced_block_stored(2, Some(1), None, "GPU");
    let child = normalizer
        .preprocess(gpu_child, worker)
        .expect("unrecognized-media event must not make the parent ambiguous");
    let RawKvEvent::BlockStored {
        cache_namespace, ..
    } = child
    else {
        panic!("expected BlockStored");
    };
    assert_eq!(cache_namespace.as_deref(), Some("tenant-a"));
}
