// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use bytes::Bytes;
use dynamo_kv_router::global_view::overlap::KvOverlapScorer;
use dynamo_kv_router::global_view::state::{InMemoryPoolStateRepository, PoolLocation};
use dynamo_kv_router::global_view::{PoolKey, V1PoolIdDeriver};
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};
use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent,
};
use std::time::Duration;
use wire::v1::model_target;
use wire::{
    BaseModelTarget, CkfFormat, DigestIdentity, DynamoEndpointId, IdentitySource, IndexerDomainId,
    KvPoolId, KvQuerySemantics, ModelRegistration, ModelTarget, RelayIdentity,
};

fn descriptor() -> KvPoolDescriptor {
    KvPoolDescriptor {
        producer: Some(wire::ProducerIdentity {
            pool_id: Some(KvPoolId {
                identity_version: wire::POOL_IDENTITY_VERSION,
                indexer_domain: Some(IndexerDomainId {
                    cache_semantics: Some(DigestIdentity {
                        digest: Bytes::from_static(&[1; 16]),
                        source: IdentitySource::Explicit as i32,
                    }),
                    routing_scope: Some(DigestIdentity {
                        digest: Bytes::from_static(&[2; 16]),
                        source: IdentitySource::Explicit as i32,
                    }),
                }),
                dc_id: 7,
            }),
            producer_incarnation: 11,
            layout_generation: 13,
            ckf_format: Some(CkfFormat {
                format_version: 1,
                seed: 42,
                bucket_count: 1024,
                fingerprint_bits: 16,
                slots_per_bucket: 4,
            }),
        }),
        serving_endpoint: Some(DynamoEndpointId {
            namespace: "mocker".into(),
            component: "decode".into(),
            endpoint: "generate".into(),
        }),
        registrations: vec![ModelRegistration {
            canonical_model_id: "model".into(),
            target: Some(ModelTarget {
                target: Some(model_target::Target::Base(BaseModelTarget {
                    base_model: "model".into(),
                })),
            }),
            aliases: Vec::new(),
        }],
        query_semantics: Some(KvQuerySemantics {
            kv_block_size: 4,
            hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
        }),
        pool_roles: vec![WorkerRole::Aggregated as i32],
    }
}

fn assembler() -> Arc<PoolObservationAssembler> {
    Arc::new(PoolObservationAssembler::new(
        &PoolKey::new("site", "namespace", "dgd").unwrap(),
        PoolLocation {
            region: "region".into(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        &V1PoolIdDeriver,
        Arc::new(InMemoryPoolStateRepository::default()),
    ))
}

fn frame(descriptor: &KvPoolDescriptor, kind: FilterUpdateKind, sequence: u64) -> FilterUpdate {
    let format = FilterFormat::new(42, 1024).unwrap();
    let (base_sequence, payload) = match kind {
        FilterUpdateKind::SnapshotChunk => (
            sequence,
            Bytes::from(cbi1::encode_snapshot_chunk(
                format,
                7,
                sequence,
                0,
                1,
                &vec![0; 1024],
            )),
        ),
        FilterUpdateKind::Delta => (
            sequence - 1,
            Bytes::from(
                cbi1::encode_delta(
                    format,
                    7,
                    sequence - 1,
                    sequence,
                    &[cbi1::BucketImage {
                        bucket: 0,
                        value: 0,
                    }],
                )
                .unwrap(),
            ),
        ),
        _ => unreachable!(),
    };
    FilterUpdate {
        protocol_version: wire::RELAY_PROTOCOL_VERSION,
        relay: Some(RelayIdentity {
            drt_instance_id: 1,
            relay_incarnation: 2,
        }),
        producer: descriptor.producer.clone(),
        base_sequence,
        sequence,
        kind: kind as i32,
        payload,
        contract_marker: wire::RELAY_CONTRACT_MARKER,
        ..Default::default()
    }
}

#[test]
fn snapshot_enables_scoring_and_new_session_fences_old_one() {
    let descriptor = descriptor();
    let assembler = assembler();
    let pool_id = assembler.pool_id();
    let store = Arc::new(RelayCkfOverlapStore::new(Duration::from_secs(60)));
    let mut first = CkfStreamSession::new(
        pool_id.clone(),
        "model".into(),
        &descriptor,
        Arc::clone(&store),
        Arc::clone(&assembler),
    )
    .unwrap();
    let tokens = [1, 2, 3, 4];
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        None
    );
    first
        .process_update(&frame(&descriptor, FilterUpdateKind::SnapshotChunk, 1))
        .unwrap();
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        Some(0)
    );
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "other", &tokens),
        None
    );
    first
        .process_update(&frame(&descriptor, FilterUpdateKind::Delta, 2))
        .unwrap();

    let mut second = CkfStreamSession::new(
        pool_id.clone(),
        "model".into(),
        &descriptor,
        Arc::clone(&store),
        assembler,
    )
    .unwrap();
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        None
    );
    second
        .process_update(&frame(&descriptor, FilterUpdateKind::SnapshotChunk, 3))
        .unwrap();
    drop(first);
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        Some(0)
    );
    drop(second);
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        None
    );
}

#[test]
fn relay_snapshot_scores_a_real_cached_token_block() {
    let descriptor = descriptor();
    let assembler = assembler();
    let pool_id = assembler.pool_id();
    let store = Arc::new(RelayCkfOverlapStore::new(Duration::from_secs(60)));
    let mut session = CkfStreamSession::new(
        pool_id.clone(),
        "model".into(),
        &descriptor,
        Arc::clone(&store),
        assembler,
    )
    .unwrap();
    let tokens = [11, 22, 33, 44];
    let block_hash = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())[0];
    let mut config = CkfConfig::new(2048);
    config.seed = 42;
    let mut producer = DcCkfState::new(config).unwrap();
    assert_eq!(producer.format().bucket_count(), 1024);
    let outcome = producer.apply_event(RouterEvent::new(
        1,
        KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(99),
                    tokens_hash: block_hash,
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        },
    ));
    assert!(outcome.first_error().is_none());
    let (_, buckets) = producer.barrier_snapshot().unwrap();
    let mut update = frame(&descriptor, FilterUpdateKind::SnapshotChunk, 1);
    update.payload = Bytes::from(cbi1::encode_snapshot_chunk(
        FilterFormat::new(42, 1024).unwrap(),
        7,
        1,
        0,
        1,
        &buckets,
    ));
    session.process_update(&update).unwrap();
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &tokens),
        Some(4)
    );
}

#[test]
fn changed_producer_generation_fails_closed() {
    let descriptor = descriptor();
    let assembler = assembler();
    let pool_id = assembler.pool_id();
    let store = Arc::new(RelayCkfOverlapStore::new(Duration::from_secs(60)));
    let mut session = CkfStreamSession::new(
        pool_id.clone(),
        "model".into(),
        &descriptor,
        Arc::clone(&store),
        assembler,
    )
    .unwrap();
    let mut update = frame(&descriptor, FilterUpdateKind::SnapshotChunk, 1);
    update.producer.as_mut().unwrap().producer_incarnation += 1;
    assert!(session.process_update(&update).is_err());
    drop(session);
    assert_eq!(
        store.estimate_matched_prefix_tokens(&pool_id, "model", &[1, 2, 3, 4]),
        None
    );
}
