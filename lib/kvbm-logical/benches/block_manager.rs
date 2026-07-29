// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for the `kvbm-logical` KV-cache hot path.
//!
//! Profiling flagged `BlockManager::match_blocks` (34.3% inclusive) and
//! `BlockStore::acquire_for_hash_locked` (27.7% inclusive). These groups
//! give a before/after baseline:
//!
//! - `match_blocks_active_hit`   — N-hash prefix fully resident in the
//!   active pool (immutables held). The inactive backend is never touched,
//!   so this group is LRU-only. The 27.7% path.
//! - `match_blocks_inactive_hit` — N-hash prefix in the inactive pool;
//!   every iteration resurrects it (and the returned `Vec` drops it
//!   straight back, synchronous RAII). Swept over every inactive backend
//!   because resurrection goes through the backend's `find_match`.
//! - `bulk_drop`                 — dropping a `Vec<ImmutableBlock>` of N;
//!   each drop's `release_primary` calls `inactive.insert`, so this is
//!   also swept over every backend. Baseline for the deferred batch-drop.
//!
//! Run: `cargo bench -p kvbm-logical --features testing`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use kvbm_logical::BlockManager;
use kvbm_logical::KvbmSequenceHashProvider;
use kvbm_logical::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::{BlockManagerConfigBuilder, LineageEviction};
use kvbm_logical::testing::{TestMeta, create_test_manager_with_backend};

const BLOCK_SIZE: u32 = 4; // `create_test_manager_with_backend` builds 4-token blocks.
const SIZES: [usize; 3] = [8, 32, 128];

/// Builder closure type for selecting an inactive backend.
type BackendConfig = fn(BlockManagerConfigBuilder<TestMeta>) -> BlockManagerConfigBuilder<TestMeta>;

/// All inactive backends, swept by the backend-sensitive groups. The
/// lineage backend appears twice — once per leaf-eviction policy.
const BACKENDS: &[(&str, BackendConfig)] = &[
    ("lru", |b| b.with_lru_backend()),
    ("multi_lru", |b| b.with_multi_lru_backend()),
    ("lineage_tick", |b| {
        b.with_lineage_backend_eviction(LineageEviction::Tick)
    }),
    ("lineage_fifo", |b| {
        b.with_lineage_backend_eviction(LineageEviction::Fifo)
    }),
];

/// Build a manager with `n` blocks registered as a single lineage chain,
/// returning the manager, the ordered prefix hashes, and the strong
/// `ImmutableBlock` handles (hold them to keep the blocks `Primary`; drop
/// them to push the blocks into the inactive pool).
fn build_chain(
    n: usize,
    backend: BackendConfig,
) -> (
    BlockManager<TestMeta>,
    Vec<SequenceHash>,
    Vec<ImmutableBlock<TestMeta>>,
) {
    let manager = create_test_manager_with_backend::<TestMeta>(n, backend);

    let tokens: Vec<u32> = (0..(n as u32 * BLOCK_SIZE)).collect();
    let token_blocks = {
        let seq = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(1337));
        seq.blocks().to_vec()
    };
    assert_eq!(token_blocks.len(), n, "expected {n} full blocks");

    let mut hashes = Vec::with_capacity(n);
    let mut complete = Vec::with_capacity(n);
    for tb in token_blocks.iter() {
        hashes.push(tb.kvbm_sequence_hash());
        let mutable = manager
            .allocate_blocks(1)
            .expect("allocate")
            .into_iter()
            .next()
            .unwrap();
        complete.push(mutable.complete(tb).expect("complete"));
    }
    let immutables = manager.register_blocks(complete);
    assert_eq!(immutables.len(), n);
    (manager, hashes, immutables)
}

fn bench_match_blocks_active_hit(c: &mut Criterion) {
    // Active-only: the inactive backend is never consulted — LRU only.
    let lru: BackendConfig = |b| b.with_lru_backend();
    let mut group = c.benchmark_group("match_blocks_active_hit");
    for &n in &SIZES {
        let (manager, hashes, _immutables) = build_chain(n, lru);
        // `_immutables` held for the duration → blocks stay `Primary`.
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let matched = manager.match_blocks(black_box(&hashes));
                debug_assert_eq!(matched.len(), n);
                matched
            });
        });
    }
    group.finish();
}

fn bench_match_blocks_inactive_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_blocks_inactive_hit");
    for &(backend_name, backend) in BACKENDS {
        for &n in &SIZES {
            let (manager, hashes, immutables) = build_chain(n, backend);
            // Drop the strong handles → all N blocks fall to the inactive pool.
            drop(immutables);
            group.bench_with_input(BenchmarkId::new(backend_name, n), &n, |b, _| {
                b.iter(|| {
                    // Each iteration resurrects the prefix; the returned
                    // `Vec` drops at end of iteration, pushing it back.
                    let matched = manager.match_blocks(black_box(&hashes));
                    debug_assert_eq!(matched.len(), n);
                    matched
                });
            });
        }
    }
    group.finish();
}

fn bench_bulk_drop(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_drop");
    for &(backend_name, backend) in BACKENDS {
        for &n in &SIZES {
            group.bench_with_input(BenchmarkId::new(backend_name, n), &n, |b, _| {
                b.iter_batched(
                    || build_chain(n, backend),
                    |(manager, _hashes, immutables)| {
                        drop(black_box(immutables));
                        // Returned, so the manager's own drop is outside
                        // the timing region — we only measure the drop.
                        manager
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_match_blocks_active_hit,
    bench_match_blocks_inactive_hit,
    bench_bulk_drop
);
criterion_main!(benches);
