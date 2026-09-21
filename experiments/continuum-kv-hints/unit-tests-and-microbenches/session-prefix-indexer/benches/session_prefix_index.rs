// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use dynamo_kv_router::{
    SessionPrefixIndexer,
    protocols::{ExternalSequenceBlockHash, WorkerWithDpRank},
};

const DEPTHS: [usize; 3] = [16, 64, 256];

fn hashes(depth: usize) -> Vec<ExternalSequenceBlockHash> {
    (1..=depth as u64).map(ExternalSequenceBlockHash).collect()
}

fn populated(
    depth: usize,
) -> (
    SessionPrefixIndexer,
    WorkerWithDpRank,
    Vec<ExternalSequenceBlockHash>,
) {
    let index = SessionPrefixIndexer::new();
    let worker = WorkerWithDpRank::new(1, 0);
    let blocks = hashes(depth);
    index
        .update_session_from_stored_blocks("session", worker, None, &blocks)
        .unwrap();
    (index, worker, blocks)
}

fn bench(c: &mut Criterion) {
    let mut lineage = c.benchmark_group("session_prefix_lineage");
    for depth in DEPTHS {
        let (index, worker, _) = populated(depth);
        lineage.throughput(Throughput::Elements(depth as u64));
        lineage.bench_with_input(BenchmarkId::new("resolve", depth), &depth, |b, _| {
            b.iter(|| {
                black_box(
                    index
                        .get_session_block_lineage("session", worker, None)
                        .unwrap(),
                )
            });
        });
    }
    lineage.finish();

    let mut updates = c.benchmark_group("session_prefix_updates");
    for depth in DEPTHS {
        updates.throughput(Throughput::Elements(depth as u64));
        updates.bench_with_input(
            BenchmarkId::new("store_chain", depth),
            &depth,
            |b, &depth| {
                b.iter_batched(
                    || (SessionPrefixIndexer::new(), hashes(depth)),
                    |(index, blocks)| {
                        black_box(
                            index
                                .update_session_from_stored_blocks(
                                    "session",
                                    WorkerWithDpRank::new(1, 0),
                                    None,
                                    &blocks,
                                )
                                .unwrap(),
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        let (index, worker, blocks) = populated(depth);
        let frontier = blocks[depth - 1];
        updates.throughput(Throughput::Elements(1));
        updates.bench_with_input(BenchmarkId::new("match_existing", depth), &depth, |b, _| {
            b.iter(|| {
                black_box(
                    index
                        .update_session_from_match("session", worker, frontier)
                        .unwrap(),
                )
            });
        });
    }
    updates.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
