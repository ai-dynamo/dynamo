// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dynamo_kv_router::indexer::KvCacheGroupResidency;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupTag, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, ResidencyOwner, RouterEvent, WorkerWithDpRank,
};

fn recurrent_presence(c: &mut Criterion) {
    let table = KvCacheGroupResidency::new();
    c.bench_function("recurrent/dense_inactive", |b| {
        b.iter(|| black_box(table.is_active()))
    });
    let chain: Vec<_> = (1..=256).map(ExternalSequenceBlockHash).collect();
    for worker_id in 0..64 {
        let worker = WorkerWithDpRank::new(worker_id, 0);
        table.declare_groups(ResidencyOwner::worker(worker), &[0, 1, 2]);
        for group in 0..3 {
            // Named-wire construction keeps this benchmark usable against older tag schemas.
            let tag: KvCacheGroupTag = serde_json::from_value(serde_json::json!({
                "group_idx": group, "kind": "mamba", "expected_group_count": 3,
                "alignment_blocks": 1,
            }))
            .unwrap();
            table.apply(
                &RouterEvent::new(
                    worker_id,
                    KvCacheEvent {
                        event_id: 1,
                        dp_rank: 0,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: None,
                            blocks: chain[..192]
                                .iter()
                                .copied()
                                .map(|hash| KvCacheStoredBlockData {
                                    block_hash: hash,
                                    tokens_hash: LocalBlockHash(0),
                                    mm_extra_info: None,
                                })
                                .collect(),
                        }),
                    },
                )
                .with_kv_cache_group(tag),
            );
        }
    }
    c.bench_function("recurrent/64_workers_3_groups_hit", |b| {
        b.iter(|| {
            for worker_id in 0..64 {
                black_box(table.reusable_blocks(WorkerWithDpRank::new(worker_id, 0), &chain, 192));
            }
        })
    });
    c.bench_function("recurrent/64_workers_3_groups_scan_64", |b| {
        b.iter(|| {
            for worker_id in 0..64 {
                black_box(table.reusable_blocks(WorkerWithDpRank::new(worker_id, 0), &chain, 256));
            }
        })
    });
}

criterion_group!(benches, recurrent_presence);
criterion_main!(benches);
