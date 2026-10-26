// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{hint::black_box, sync::Arc, time::Duration};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use dynamo_kv_router::indexer::{
    KvIndexerDelegate, KvIndexerInterface, ThreadPoolIndexer,
    concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed,
};
use dynamo_kv_router::protocols::*;

struct Delegate;
impl KvIndexerDelegate for Delegate {
    fn on_create(&self, hash: ExternalSequenceBlockHash) {
        black_box(hash);
    }
    fn on_remove(&self, hash: ExternalSequenceBlockHash) {
        black_box(hash);
    }
}

fn bench(c: &mut Criterion) {
    const WORKERS: u64 = 128;
    const BLOCKS: usize = 64;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let events: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let local: Vec<_> = (0..BLOCKS)
                .map(|i| {
                    LocalBlockHash(if i < BLOCKS / 2 {
                        i as u64
                    } else {
                        WORKERS + worker * BLOCKS as u64 + i as u64
                    })
                })
                .collect();
            let sequence = compute_seq_hash_for_block(&local);
            RouterEvent::new(
                worker,
                KvCacheEvent {
                    event_id: 0,
                    dp_rank: 0,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: local
                            .into_iter()
                            .zip(sequence)
                            .map(|(tokens_hash, hash)| KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(hash),
                                tokens_hash,
                                mm_extra_info: None,
                            })
                            .collect(),
                    }),
                },
            )
        })
        .collect();
    let mut group = c.benchmark_group("indexer_delegate");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(WORKERS * BLOCKS as u64 * 2));
    for is_enabled in [false, true] {
        let backend = if is_enabled {
            ConcurrentRadixTreeCompressed::new_with_delegate(Arc::new(Delegate))
        } else {
            ConcurrentRadixTreeCompressed::new()
        };
        let indexer = ThreadPoolIndexer::new(backend, 8, 32);
        group.bench_function(if is_enabled { "enabled" } else { "disabled" }, |b| {
            b.iter(|| {
                runtime.block_on(async {
                    for event in &events {
                        indexer.apply_event(event.clone()).await;
                    }
                    indexer.flush().await;
                    for worker in 0..WORKERS {
                        indexer
                            .apply_event(RouterEvent::new(
                                worker,
                                KvCacheEvent {
                                    event_id: 1,
                                    dp_rank: 0,
                                    data: KvCacheEventData::Cleared,
                                },
                            ))
                            .await;
                    }
                    indexer.flush().await;
                })
            });
        });
        indexer.shutdown();
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
