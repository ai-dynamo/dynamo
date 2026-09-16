// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_router::shared_cache::{hex_encode, mooncake_store_contract::test_contract};
use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, KvCacheStoredBlockData};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};

const SOURCE: WorkerWithDpRank = WorkerWithDpRank {
    worker_id: 7,
    dp_rank: 0,
};

fn contract() -> ValidatedContract {
    test_contract(&[(GroupKind::FullAttention, 16, None)], 16, 2)
}

fn cache(contract: ValidatedContract) -> MooncakeStoreCache {
    let cache = MooncakeStoreCache::new(|| false);
    cache.update_contract(Some(contract), HashSet::from([SOURCE]));
    cache
}

fn digest(hash: u64) -> [u8; 32] {
    let mut digest = [42; 32];
    digest[24..].copy_from_slice(&hash.to_be_bytes());
    digest
}

fn event(group: usize, rank: usize, hash: u64, medium: &str, stored: bool) -> MooncakeObjectEvent {
    MooncakeObjectEvent {
        event_type: if stored { "stored" } else { "removed" }.into(),
        object_key: Some(format!(
            "deployment@g{group}@r{rank}@{}",
            hex_encode(&digest(hash))
        )),
        tenant_id: "default".into(),
        group_id: Some("not-completion-evidence".into()),
        medium: Some(medium.into()),
    }
}

fn apply(cache: &MooncakeStoreCache, sequence: u64, events: Vec<MooncakeObjectEvent>) {
    let payload = rmp_serde::to_vec_named(&(0i64, events, 0u32)).unwrap();
    cache.apply_test_frames(&[vec![], sequence.to_be_bytes().to_vec(), payload]);
}

fn store_all(cache: &MooncakeStoreCache, sequence: u64, groups: usize, hashes: &[u64]) {
    apply(
        cache,
        sequence,
        (0..groups)
            .flat_map(|g| {
                hashes
                    .iter()
                    .flat_map(move |&h| (0..2).map(move |r| event(g, r, h, "cpu", true)))
            })
            .collect(),
    );
}

fn learned_data(parent: Option<u64>, hashes: &[u64]) -> KvCacheStoreData {
    KvCacheStoreData {
        parent_hash: parent.map(ExternalSequenceBlockHash),
        start_position: None,
        shared_cache_eligible: true,
        blocks: hashes
            .iter()
            .map(|&hash| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(hash),
                tokens_hash: LocalBlockHash(hash + 1000),
                mm_extra_info: None,
            })
            .collect(),
    }
}

fn learn(cache: &MooncakeStoreCache, parent: Option<u64>, hashes: &[u64]) {
    cache.observe_stored(SOURCE, &learned_data(parent, hashes));
}

fn hits(cache: &MooncakeStoreCache, hashes: &[u64], token_count: usize) -> usize {
    let tokens = vec![0; token_count];
    let block_hashes = hashes
        .iter()
        .map(|h| LocalBlockHash(h + 1000))
        .collect::<Vec<_>>();
    let block_size = cache
        .state
        .lock()
        .contract
        .as_ref()
        .map_or(16, |c| c.main_event_block_size());
    cache
        .lookup(SharedCacheQuery {
            tokens: &tokens,
            block_hashes: &block_hashes,
            block_size,
            cache_namespace: None,
            shared_cache_eligible: true,
        })
        .total_hits as usize
}

#[test]
fn tp2_requires_each_object_and_readable_medium() {
    let cache = cache(contract());
    learn(&cache, None, &[1, 2]);
    apply(
        &cache,
        1,
        vec![event(0, 0, 1, "cpu", true), event(0, 0, 1, "cpu", true)],
    );
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    apply(&cache, 2, vec![event(0, 1, 1, "cpu", true)]);
    assert_eq!(hits(&cache, &[1, 2], 33), 1);
    apply(
        &cache,
        3,
        vec![event(0, 0, 1, "disk", true), event(0, 0, 1, "cpu", false)],
    );
    assert_eq!(hits(&cache, &[1, 2], 33), 1);
    assert_eq!(cache.state.lock().memberships, 2);
    apply(
        &cache,
        4,
        vec![event(0, 0, 1, "disk", false), event(0, 0, 2, "cpu", true)],
    );
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
}

#[test]
fn exact_prefix_tenant_and_full_digest_filtering() {
    let cache = cache(contract());
    learn(&cache, None, &[1]);
    let mut other_tenant = event(0, 1, 1, "cpu", true);
    other_tenant.tenant_id = "tenant-a".into();
    let mut other_prefix = event(0, 1, 1, "cpu", true);
    other_prefix.object_key = Some(format!("other@{}", hex_encode(&digest(1))));
    apply(
        &cache,
        1,
        vec![event(0, 0, 1, "cpu", true), other_tenant, other_prefix],
    );
    assert_eq!(hits(&cache, &[1], 17), 0);
    apply(&cache, 2, vec![event(0, 1, 1, "cpu", true)]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    let mut unrelated = event(0, 0, 1, "cpu", false);
    unrelated.object_key = Some("other@not-a-hash".into());
    unrelated.medium = None;
    apply(&cache, 3, vec![unrelated]);
    assert_eq!(hits(&cache, &[1], 17), 1);
}

#[test]
fn cross_group_digest_ambiguity_survives_clear_and_removal() {
    let cache = cache(test_contract(
        &[
            (GroupKind::FullAttention, 16, None),
            (GroupKind::Mamba, 16, None),
        ],
        16,
        2,
    ));
    learn(&cache, None, &[1]);
    store_all(&cache, 1, 2, &[1]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    let mut collision = digest(1);
    collision[0] ^= 1;
    let mut collided = event(1, 0, 1, "cpu", true);
    collided.object_key = Some(format!("deployment@g1@r0@{}", hex_encode(&collision)));
    apply(&cache, 2, vec![collided]);
    assert_eq!(hits(&cache, &[1], 17), 0);
    apply(
        &cache,
        3,
        vec![MooncakeObjectEvent {
            event_type: "cleared".into(),
            object_key: None,
            tenant_id: "default".into(),
            group_id: None,
            medium: Some("nil".into()),
        }],
    );
    store_all(&cache, 4, 2, &[1]);
    assert_eq!(hits(&cache, &[1], 17), 0);
    assert!(cache.state.lock().digests[&1].ambiguous);
}

#[test]
fn keyless_clear_and_sequence_cursor_are_independent_of_residency() {
    let cache = cache(contract());
    learn(&cache, None, &[1]);
    store_all(&cache, 1, 1, &[1]);
    apply(
        &cache,
        2,
        vec![MooncakeObjectEvent {
            event_type: "AllBlocksCleared".into(),
            object_key: None,
            tenant_id: "default".into(),
            group_id: None,
            medium: None,
        }],
    );
    assert_eq!(hits(&cache, &[1], 17), 0);
    store_all(&cache, 2, 1, &[1]);
    assert_eq!(hits(&cache, &[1], 17), 0);
    store_all(&cache, 3, 1, &[1]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    apply(&cache, 3, vec![event(0, 0, 1, "cpu", false)]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    apply(&cache, 5, vec![event(0, 0, 1, "cpu", true)]);
    assert_eq!(hits(&cache, &[1], 17), 0);
    apply(&cache, 1, vec![event(0, 1, 1, "cpu", true)]);
    assert_eq!(hits(&cache, &[1], 17), 0);
    apply(&cache, 2, vec![event(0, 0, 1, "cpu", true)]);
    assert_eq!(hits(&cache, &[1], 17), 1);
}

#[test]
fn malformed_mutations_and_batches_clear_immediately() {
    let cache = cache(contract());
    learn(&cache, None, &[1]);
    let mutations = [
        json!({"event_type": "removed", "object_key": null, "medium": "cpu"}),
        json!({"event_type": "removed", "object_key": "deployment@g0@r0", "medium": "cpu"}),
        json!({"event_type": "removed", "object_key": "deployment@g0@r0@oops", "medium": "cpu"}),
        json!({"event_type": "removed", "object_key": format!("deployment@g0@r0@{}", hex_encode(&digest(1))), "medium": null}),
        json!({"event_type": "moved", "object_key": format!("deployment@g0@r0@{}", hex_encode(&digest(1))), "medium": "cpu"}),
    ];
    for (idx, mutation) in mutations.into_iter().enumerate() {
        let seq = (idx * 2 + 1) as u64;
        store_all(&cache, seq, 1, &[1]);
        assert_eq!(hits(&cache, &[1], 17), 1);
        let payload = rmp_serde::to_vec_named(&(0i64, vec![mutation], 0u32)).unwrap();
        cache.apply_test_frames(&[vec![], (seq + 1).to_be_bytes().to_vec(), payload]);
        assert_eq!(hits(&cache, &[1], 17), 0);
    }
    let too_many_events = rmp_serde::to_vec_named(&(
        0i64,
        (0..=MAX_BATCH_EVENTS)
            .map(|_| event(0, 0, 1, "cpu", true))
            .collect::<Vec<_>>(),
        0u32,
    ))
    .unwrap();
    assert!(too_many_events.len() < MAX_FRAME_BYTES);
    for (idx, frames) in [
        vec![vec![]],
        vec![vec![], vec![0; 7], vec![]],
        vec![vec![], 10u64.to_be_bytes().to_vec(), vec![0xc1]],
        vec![vec![0; MAX_FRAME_BYTES + 1]],
        vec![vec![], 10u64.to_be_bytes().to_vec(), too_many_events],
    ]
    .into_iter()
    .enumerate()
    {
        store_all(&cache, 11 + idx as u64, 1, &[1]);
        assert_eq!(hits(&cache, &[1], 17), 1);
        cache.apply_test_frames(&frames);
        assert_eq!(hits(&cache, &[1], 17), 0);
    }
    assert!(!cache.state.lock().edges.is_empty());
}

#[test]
fn learning_requires_admitted_eligible_root_and_preserves_conflicts() {
    let cache = cache(contract());
    store_all(&cache, 1, 1, &[1, 2]);
    cache.observe_stored(WorkerWithDpRank::new(8, 0), &learned_data(None, &[1, 2]));
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    let mut invalid = learned_data(None, &[1, 2]);
    invalid.shared_cache_eligible = false;
    cache.observe_stored(SOURCE, &invalid);
    invalid.shared_cache_eligible = true;
    invalid.start_position = Some(3);
    cache.observe_stored(SOURCE, &invalid);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    learn(&cache, Some(1), &[2]);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    learn(&cache, None, &[1]);
    assert_eq!(hits(&cache, &[1, 2], 33), 2);
    let mut conflicting = learned_data(None, &[1]);
    conflicting.blocks[0].block_hash = ExternalSequenceBlockHash(99);
    cache.observe_stored(SOURCE, &conflicting);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    assert_eq!(
        cache.state.lock().edges[&Edge {
            parent: None,
            local: LocalBlockHash(1001)
        }]
            .child,
        1
    );
    learn(&cache, None, &[1]);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
}

#[test]
fn pending_watches_and_generation_fence_observations_and_queries() {
    let pending = Arc::new(AtomicBool::new(false));
    let gate = pending.clone();
    let cache = MooncakeStoreCache::new(move || gate.load(Ordering::Acquire));
    cache.update_contract(Some(contract()), HashSet::from([SOURCE]));
    learn(&cache, None, &[1]);
    store_all(&cache, 1, 1, &[1]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    pending.store(true, Ordering::Release);
    assert_eq!(hits(&cache, &[1], 17), 0);
    learn(&cache, Some(1), &[2]);
    assert_eq!(cache.state.lock().edges.len(), 1);
    pending.store(false, Ordering::Release);
    cache.update_contract(
        Some(contract()),
        HashSet::from([SOURCE, WorkerWithDpRank::new(9, 0)]),
    );
    assert_eq!(hits(&cache, &[1], 17), 1);
    let previous_generation = cache.state.lock().generation;
    cache.update_contract(None, HashSet::new());
    cache.update_contract(Some(contract()), HashSet::from([SOURCE]));
    learn(&cache, None, &[1]);
    cache.apply_batch(
        previous_generation,
        2,
        vec![event(0, 0, 1, "cpu", true), event(0, 1, 1, "cpu", true)],
    );
    assert_eq!(hits(&cache, &[1], 17), 0);
    store_all(&cache, 1, 1, &[1]);
    cache.apply_batch(previous_generation, 3, vec![event(0, 0, 1, "cpu", false)]);
    cache.apply_frames(previous_generation, &[vec![]]);
    assert_eq!(hits(&cache, &[1], 17), 1);
    assert_eq!(cache.state.lock().sequence, Some(1));
}

#[test]
fn all_insertion_limits_preserve_safety_and_sequence() {
    for limits in [
        Limits {
            edges: 1,
            ..Limits::default()
        },
        Limits {
            digests: 1,
            ..Limits::default()
        },
    ] {
        let mut cache = cache(contract());
        cache.limits = limits;
        learn(&cache, None, &[1, 2]);
        store_all(&cache, 1, 1, &[1, 2]);
        assert!(cache.state.lock().disabled);
        assert_eq!(hits(&cache, &[1, 2], 33), 0);
    }
    let mut cache = cache(contract());
    cache.limits.memberships = 2;
    learn(&cache, None, &[1, 2]);
    store_all(&cache, 1, 1, &[1]);
    assert_eq!(hits(&cache, &[1, 2], 33), 1);
    apply(&cache, 2, vec![event(0, 0, 2, "cpu", true)]);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    store_all(&cache, 2, 1, &[1]);
    assert_eq!(hits(&cache, &[1, 2], 33), 0);
    assert_eq!(cache.state.lock().digests.len(), 2);
    store_all(&cache, 3, 1, &[1]);
    assert_eq!(hits(&cache, &[1, 2], 33), 1);
}

#[test]
fn later_sliding_window_and_mamba_candidates_recover_without_old_objects() {
    for kind in [GroupKind::SlidingWindow, GroupKind::Mamba] {
        let contract = test_contract(
            &[
                (GroupKind::FullAttention, 16, None),
                (kind, 16, (kind == GroupKind::SlidingWindow).then_some(32)),
            ],
            16,
            2,
        );
        let cache = cache(contract);
        learn(&cache, None, &[1, 2, 3, 4]);
        store_all(&cache, 1, 1, &[1, 2, 3, 4]);
        apply(
            &cache,
            2,
            [3, 4]
                .into_iter()
                .flat_map(|hash| (0..2).map(move |rank| event(1, rank, hash, "cpu", true)))
                .collect(),
        );
        assert_eq!(hits(&cache, &[1, 2, 3, 4], 65), 4);
        let exact_final = if kind == GroupKind::Mamba { 3 } else { 0 };
        assert_eq!(hits(&cache, &[1, 2, 3, 4], 64), exact_final);
        apply(&cache, 3, vec![event(1, 1, 4, "cpu", false)]);
        assert_eq!(hits(&cache, &[1, 2, 3, 4], 65), exact_final);
    }
}

#[test]
fn unequal_spans_only_claim_observable_aligned_endpoints() {
    for (b, group_kind, span, window) in [
        (16, GroupKind::FullAttention, 32, None),
        (16, GroupKind::SlidingWindow, 32, Some(64)),
        (32, GroupKind::SlidingWindow, 8, Some(8)),
        (32, GroupKind::Mamba, 8, None),
        (16, GroupKind::Mamba, 32, None),
    ] {
        let contract = test_contract(
            &[
                (GroupKind::FullAttention, b, None),
                (group_kind, span, window),
            ],
            b,
            2,
        );
        let cache = cache(contract);
        learn(&cache, None, &[1, 2, 3, 4]);
        store_all(&cache, 1, 2, &[1, 2, 3, 4]);
        assert_eq!(hits(&cache, &[1, 2, 3, 4], (b * 4 + 1) as usize), 4);
        let hits = hits(&cache, &[1, 2, 3, 4], (b * 4) as usize);
        assert!(hits < 4);
        assert_eq!((hits * b as usize) % span as usize, 0);
    }
}

#[test]
fn surfaced_stream_error_and_end_clear_before_reconnect() {
    let cache = cache(contract());
    learn(&cache, None, &[1]);
    let generation = cache.state.lock().generation;
    for (idx, failure) in [
        None,
        Some(Err(tmq::TmqError::Io(std::io::Error::other(
            "stream failed",
        )))),
    ]
    .into_iter()
    .enumerate()
    {
        let sequence = idx as u64 + 1;
        store_all(&cache, sequence, 1, &[1]);
        assert_eq!(hits(&cache, &[1], 17), 1);
        assert!(!cache.apply_message(generation, failure));
        assert_eq!(hits(&cache, &[1], 17), 0);
        let state = cache.state.lock();
        assert_eq!(state.sequence, Some(sequence));
        assert_eq!(state.edges.len(), 1);
        assert_eq!(state.digests.len(), 1);
    }
}

#[tokio::test]
async fn subscriber_cancellation_disables_the_retired_cache() {
    let cache = Arc::new(cache(contract()));
    learn(&cache, None, &[1]);
    store_all(&cache, 1, 1, &[1]);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let task = cache.spawn_subscriber("invalid://mooncake-store".into(), cancellation);
    tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(hits(&cache, &[1], 17), 0);
    assert!(cache.state.lock().disabled);
}

#[tokio::test]
async fn normalized_msgpack_learns_before_both_local_indexers() {
    use crate::kv_router::indexer::test_util::{
        flush_indexer, make_test_concurrent_indexer, make_test_indexer,
    };
    use dynamo_kv_router::{
        protocols::{BlockHashOptions, KvCacheEventData, compute_block_hash_for_seq},
        zmq_wire::{ZmqEventNormalizer, decode_event_batch},
    };

    for make_indexer in [make_test_indexer, make_test_concurrent_indexer] {
        for (namespace, lora) in [
            (None, None),
            (Some("tenant-a"), None),
            (Some("tenant-a"), Some("adapter-a")),
        ] {
            let cache = Arc::new(cache(contract()));
            let mut indexer = make_indexer(16);
            indexer.set_shared_cache(Some(cache.clone()));
            let tokens = (0..33).collect::<Vec<u32>>();
            let mut extra_keys = Vec::new();
            if let Some(lora) = lora {
                extra_keys.push(lora.to_string());
            }
            if let Some(namespace) = namespace {
                extra_keys.push(format!("dynamo-cache-salt:{namespace}"));
            }
            let wire = json!([0.0, [{"type":"BlockStored", "block_hashes":[1,2], "parent_block_hash":null,
                "token_ids":tokens[..32], "block_size":16, "medium":"GPU", "lora_name":lora,
                "extra_keys":[extra_keys, lora.map(|l| vec![l])]}], 0]);
            let mut normalizer = ZmqEventNormalizer::new(16);
            let batch = decode_event_batch(&rmp_serde::to_vec_named(&wire).unwrap()).unwrap();
            let mut raw = batch.events.into_iter();
            let event = normalizer
                .normalize(raw.next().unwrap(), 1, SOURCE)
                .unwrap()
                .into_router_event()
                .unwrap();
            assert!(
                matches!(&event.event.data, KvCacheEventData::Stored(data) if data.shared_cache_eligible)
            );
            indexer.try_apply_event(event).await.unwrap();
            flush_indexer(&indexer).await;
            store_all(&cache, 1, 1, &[1, 2]);
            let block_hashes = compute_block_hash_for_seq(
                &tokens,
                16,
                BlockHashOptions {
                    lora_name: lora,
                    cache_namespace: namespace,
                    ..Default::default()
                },
            );
            let query = SharedCacheQuery {
                tokens: &tokens,
                block_hashes: &block_hashes,
                block_size: 16,
                cache_namespace: namespace,
                shared_cache_eligible: true,
            };
            assert_eq!(cache.check_blocks(query).await.unwrap().total_hits, 2);
            assert_eq!(
                cache
                    .check_blocks(SharedCacheQuery {
                        shared_cache_eligible: false,
                        ..query
                    })
                    .await
                    .unwrap()
                    .total_hits,
                0
            );
            let wrong_hashes = compute_block_hash_for_seq(
                &tokens,
                16,
                BlockHashOptions {
                    cache_namespace: Some("other"),
                    ..Default::default()
                },
            );
            assert_eq!(
                cache
                    .check_blocks(SharedCacheQuery {
                        block_hashes: &wrong_hashes,
                        cache_namespace: Some("other"),
                        ..query
                    })
                    .await
                    .unwrap()
                    .total_hits,
                0
            );
            for (idx, kind) in ["BlockRemoved", "AllBlocksCleared"].into_iter().enumerate() {
                let wire = json!([0.0, [{"type":kind, "block_hashes":[1,2], "medium":"GPU"}], 0]);
                let batch = decode_event_batch(&rmp_serde::to_vec_named(&wire).unwrap()).unwrap();
                let event = normalizer
                    .normalize(
                        batch.events.into_iter().next().unwrap(),
                        2 + idx as u64,
                        SOURCE,
                    )
                    .unwrap()
                    .into_router_event()
                    .unwrap();
                indexer.try_apply_event(event).await.unwrap();
                flush_indexer(&indexer).await;
                assert_eq!(cache.check_blocks(query).await.unwrap().total_hits, 2);
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct OracleCase {
    name: String,
    main_event_block_size: u32,
    coordinator_alignment: u32,
    partial_hash_hits: bool,
    hash_block_size: u32,
    groups: Vec<OracleGroup>,
    token_ids: Vec<u32>,
    hashes: Vec<OracleHash>,
    present: Vec<(usize, String, String)>,
    request_token_count: usize,
    oracle_hit_tokens: u32,
    reusable_endpoints: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct OracleGroup {
    group_id: usize,
    kind: GroupKind,
    block_size: u32,
    window: Option<u32>,
    prefixes: Vec<String>,
}

#[derive(serde::Deserialize)]
struct OracleHash {
    end_token: u32,
    digest: String,
    low64: u64,
}

impl OracleCase {
    fn contract(&self) -> ValidatedContract {
        let key = super::super::mooncake_store_contract::RUNTIME_KEY;
        let mut runtime = test_runtime_config(self.main_event_block_size);
        let mut descriptor = runtime
            .get_engine_specific::<serde_json::Value>(key)
            .unwrap()
            .unwrap();
        descriptor["coordinator"]["lcm_block_size"] = json!(self.coordinator_alignment);
        descriptor["coordinator"]["partial_hash_hits"] = json!(self.partial_hash_hits);
        descriptor["gpu_to_store_group"] = json!((0..self.groups.len()).collect::<Vec<_>>());
        descriptor["groups"] =
            json!(self.groups.iter().map(|g| {
            let (spec, manager) = match g.kind {
                GroupKind::FullAttention => ("FullAttentionSpec", "FullAttentionManager"),
                GroupKind::SlidingWindow => ("SlidingWindowSpec", "SlidingWindowManager"),
                GroupKind::Mamba => ("MambaSpec", "MambaManager"),
            };
            json!({"group_id": g.group_id, "kind": g.kind, "spec": spec, "manager": manager,
                "block_size": g.block_size, "hash_block_size": self.hash_block_size,
                "sliding_window": g.window, "key_prefixes": g.prefixes,
                "mamba_cache_mode": (g.kind == GroupKind::Mamba).then_some("align")})
        }).collect::<Vec<_>>());
        runtime.set_engine_specific(key, descriptor).unwrap();
        ValidatedContract::from_runtime(&runtime).unwrap().unwrap()
    }
}

#[tokio::test]
async fn pinned_coordinator_oracle_through_normalized_local_indexers() {
    use crate::kv_router::indexer::test_util::{
        flush_indexer, make_test_concurrent_indexer, make_test_indexer,
    };
    use dynamo_kv_router::{
        protocols::{BlockHashOptions, compute_block_hash_for_seq},
        zmq_wire::{ZmqEventNormalizer, decode_event_batch},
    };

    #[derive(serde::Deserialize)]
    struct Fixture {
        vllm_revision: String,
        seed_policy: String,
        cases: Vec<OracleCase>,
    }
    let fixture: Fixture = serde_json::from_str(include_str!("oracle.json")).unwrap();
    assert_eq!(
        fixture.vllm_revision,
        "1085b64425a9e6f5ca52876ad32e55fda5665f4e"
    );
    assert_eq!(fixture.seed_policy, "pythonhashseed-0");
    let mut useful_families = HashSet::new();
    for case in &fixture.cases {
        let b = case.main_event_block_size;
        let tokens = &case.token_ids[..case.request_token_count];
        for hash in &case.hashes {
            let digest = decode_digest(&hash.digest).unwrap();
            assert_eq!(
                u64::from_be_bytes(digest[24..].try_into().unwrap()),
                hash.low64
            );
        }
        let main_hashes = case
            .hashes
            .iter()
            .filter(|h| h.end_token.is_multiple_of(b) && h.end_token as usize <= tokens.len())
            .map(|h| h.low64)
            .collect::<Vec<_>>();
        for make_indexer in [make_test_indexer, make_test_concurrent_indexer] {
            let cache = Arc::new(cache(case.contract()));
            let mut indexer = make_indexer(b);
            indexer.set_shared_cache(Some(cache.clone()));
            let wire = json!([0.0, [{"type":"BlockStored", "block_hashes": main_hashes,
                "parent_block_hash": null, "token_ids": tokens[..main_hashes.len() * b as usize],
                "block_size": b, "medium":"GPU"}], 0]);
            let batch = decode_event_batch(&rmp_serde::to_vec_named(&wire).unwrap()).unwrap();
            let mut normalizer = ZmqEventNormalizer::new(b);
            let event = normalizer
                .normalize(batch.events.into_iter().next().unwrap(), 1, SOURCE)
                .unwrap()
                .into_router_event()
                .unwrap();
            indexer.try_apply_event(event).await.unwrap();
            flush_indexer(&indexer).await;
            apply(
                &cache,
                1,
                case.present
                    .iter()
                    .map(|(group, prefix, digest)| {
                        assert!(case.groups[*group].prefixes.contains(prefix));
                        MooncakeObjectEvent {
                            event_type: "stored".into(),
                            object_key: Some(format!("{prefix}@{digest}")),
                            tenant_id: "default".into(),
                            group_id: None,
                            medium: Some("cpu".into()),
                        }
                    })
                    .collect(),
            );
            let local_hashes = compute_block_hash_for_seq(tokens, b, BlockHashOptions::default());
            let hits = cache
                .check_blocks(SharedCacheQuery {
                    block_hashes: &local_hashes,
                    tokens,
                    block_size: b,
                    cache_namespace: None,
                    shared_cache_eligible: true,
                })
                .await
                .unwrap();
            let hit_tokens = hits.total_hits * b;
            assert!(
                hit_tokens <= case.oracle_hit_tokens,
                "{}: overcount {hit_tokens}",
                case.name
            );
            assert!(
                hit_tokens == 0 || case.reusable_endpoints.contains(&hit_tokens),
                "{}: endpoint {hit_tokens} cannot restart",
                case.name
            );
            if matches!(
                case.name.as_str(),
                "fa16_swa16_window32_late" | "tp2_fa16_mamba16_late"
            ) {
                assert!(
                    hit_tokens > 0,
                    "{}: required hybrid family must have useful hits",
                    case.name
                );
                useful_families.insert(case.name.as_str());
            }
            eprintln!(
                "Mooncake Store oracle: {}: proven_tokens={hit_tokens}, oracle_tokens={}",
                case.name, case.oracle_hit_tokens
            );
        }
    }
    assert_eq!(useful_families.len(), 2);
}

#[test]
#[ignore = "lookup timing measurement, run with --ignored --nocapture"]
fn benchmark_long_multigroup_lookup() {
    use std::{hint::black_box, time::Instant};

    for blocks in [4096usize, 16_384] {
        let cache = cache(test_contract(
            &[
                (GroupKind::FullAttention, 16, None),
                (
                    GroupKind::SlidingWindow,
                    16,
                    Some((blocks as u32 / 2) * 16 + 1),
                ),
                (GroupKind::Mamba, 16, None),
            ],
            16,
            2,
        ));
        let hashes = (1..=blocks as u64).collect::<Vec<_>>();
        learn(&cache, None, &hashes);
        for (idx, chunk) in hashes.chunks(512).enumerate() {
            store_all(&cache, idx as u64 + 1, 3, chunk);
        }
        let tokens = vec![0; blocks * 16 + 1];
        let local = hashes
            .iter()
            .map(|hash| LocalBlockHash(hash + 1000))
            .collect::<Vec<_>>();
        let query = SharedCacheQuery {
            block_hashes: &local,
            tokens: &tokens,
            block_size: 16,
            cache_namespace: None,
            shared_cache_eligible: true,
        };
        assert_eq!(cache.lookup(query).total_hits as usize, blocks);
        let start = Instant::now();
        for _ in 0..32 {
            assert_eq!(
                black_box(cache.lookup(black_box(query))).total_hits as usize,
                blocks
            );
        }
        eprintln!(
            "Mooncake Store lookup: blocks={blocks}, tokens={}, groups=3, TP=2, runs=32, mean_us={:.1}",
            tokens.len(),
            start.elapsed().as_secs_f64() * 1e6 / 32.0
        );
    }
}

#[test]
fn entry_layout_and_map_allocations() {
    use std::mem::size_of;
    let mut edges = FxHashMap::<Edge, LearnedEdge>::default();
    let mut reverse = FxHashMap::<u64, Edge>::default();
    let mut digests = HashMap::<u64, DigestIdentity>::new();
    let mut objects = HashMap::<ObjectIdentity, u8>::new();
    for hash in 0..4096 {
        let edge = Edge {
            parent: Some(hash),
            local: LocalBlockHash(hash),
        };
        edges.insert(
            edge,
            LearnedEdge {
                child: hash,
                conflicted: false,
            },
        );
        reverse.insert(hash, edge);
        digests.insert(
            hash,
            DigestIdentity {
                digest: digest(hash),
                ambiguous: false,
            },
        );
        objects.insert(
            ObjectIdentity {
                digest: digest(hash),
                prefix: 0,
            },
            1,
        );
    }
    eprintln!(
        "Mooncake Store entries: edge={} reverse={} digest={} object={}; capacities={}/{}/{}/{} for 4096 entries",
        size_of::<(Edge, LearnedEdge)>(),
        size_of::<(u64, Edge)>(),
        size_of::<(u64, DigestIdentity)>(),
        size_of::<(ObjectIdentity, u8)>(),
        edges.capacity(),
        reverse.capacity(),
        digests.capacity(),
        objects.capacity()
    );
    assert!(size_of::<(Edge, LearnedEdge)>() <= 48);
    assert!(size_of::<(u64, DigestIdentity)>() <= 48);
    assert!(size_of::<(ObjectIdentity, u8)>() <= 40);
}
