// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request cost of the shared selection core: one booked selection and its
//! release, swept over catalog size and router-hint capability.

use std::collections::HashMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, RoutingConstraints, StorageTier, WorkerWithDpRank,
    compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore, WorkerRequest,
};
use tokio_util::sync::CancellationToken;

const BLOCK_SIZE: u32 = 16;

fn prompt_tokens() -> Vec<u32> {
    (1..=256).collect()
}

/// `hints` makes worker 1 hold the whole prompt so every booking on another
/// worker computes a fetch hint from it.
fn core_with_workers(
    runtime: &tokio::runtime::Runtime,
    workers: u64,
    hints: bool,
) -> SelectionCore {
    let config = KvRouterConfig {
        use_kv_events: true,
        router_queue_threshold: None,
        ..Default::default()
    };
    let core = SelectionCore::try_new_local(
        config,
        1,
        CancellationToken::new(),
        SelectionCacheConfig::default(),
    )
    .expect("core");
    runtime.block_on(async {
        for worker_id in 1..=workers {
            let mut request = WorkerRequest {
                worker_id,
                endpoint: Some(format!("http://worker-{worker_id}:8000")),
                kv_events_endpoint: Some(format!("tcp://127.0.0.1:{}", 40_000 + worker_id)),
                block_size: Some(BLOCK_SIZE),
                max_num_batched_tokens: Some(8192),
                ..WorkerRequest::default()
            };
            if hints {
                request.router_hint_worker_type = Some("decode".to_string());
                request.router_hint_source_control_endpoints =
                    HashMap::from([(0, format!("tcp://worker-{worker_id}:9000"))]);
            }
            core.upsert_worker(request).await.expect("upsert");
        }
        if hints {
            seed_prefix_on_worker_one(&core).await;
        }
    });
    core
}

async fn seed_prefix_on_worker_one(core: &SelectionCore) {
    let key = RoutingPartitionId::new("default", "default");
    let local_hashes =
        compute_block_hash_for_seq(&prompt_tokens(), BLOCK_SIZE, BlockHashOptions::default());
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    let blocks = local_hashes
        .iter()
        .zip(sequence_hashes.iter())
        .map(|(&tokens_hash, &sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    let indexer = core.partition(&key).expect("partition").indexer().clone();
    indexer
        .apply_event_routed(RouterEvent::with_storage_tier(
            1,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .expect("seed index");
    if let Indexer::Single { primary, .. } = &indexer {
        primary.flush().await;
    }
    let mut request = reserve_request("seed".to_string());
    request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
    let response = core.select_and_reserve(request).await.expect("reserve");
    core.free_reservation("seed").await.expect("free");
    assert!(response.kv_hint.is_some(), "seeded prefix produced no hint");
}

fn reserve_request(selection_id: String) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: "default".to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(selection_id),
        prompt: PromptRequest {
            token_ids: Some(prompt_tokens()),
            ..PromptRequest::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
    }
}

fn bench_select_and_reserve(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let mut group = c.benchmark_group("selection_core/select_and_reserve_then_free");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    for &workers in &[8u64, 64, 512] {
        for hints in [false, true] {
            let core = core_with_workers(&runtime, workers, hints);
            let mut sequence = 0u64;
            group.bench_with_input(
                BenchmarkId::new(if hints { "hints" } else { "no_hints" }, workers),
                &core,
                |b, core| {
                    b.iter(|| {
                        sequence += 1;
                        let id = format!("bench-{sequence}");
                        // `no_hints` sweeps scoring across all workers; `hints`
                        // pins off the seeded worker so every booking computes a hint.
                        let mut request = reserve_request(id.clone());
                        if hints {
                            request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
                        }
                        runtime.block_on(async {
                            core.select_and_reserve(request).await.expect("reserve");
                            core.free_reservation(&id).await.expect("free");
                        });
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_select_and_reserve);
criterion_main!(benches);
