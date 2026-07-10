// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_kv_router::{TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope};
use tempfile::NamedTempFile;

fn tracking_hash(c: &mut Criterion) {
    let mut key_file = NamedTempFile::new().unwrap();
    key_file.write_all(&[0x5a; 32]).unwrap();
    let public_context = TrackingHashContext::from_config(&KvRouterConfig::default()).unwrap();
    let keyed_context = TrackingHashContext::from_config(&KvRouterConfig {
        router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
        router_tracking_key_file: Some(key_file.path().to_path_buf()),
        router_tracking_key_id: Some("benchmark".to_string()),
        ..Default::default()
    })
    .unwrap();
    let block_size = 16;
    let scope = TrackingHashScope {
        model_name: "benchmark-model",
        routing_group: "default",
        block_size,
    };

    let mut group = c.benchmark_group("tracking_hash");
    for blocks in [1_u32, 32, 128] {
        let tokens = (0..blocks * block_size).collect::<Vec<_>>();
        let options = BlockHashOptions::default();
        let public_blocks = compute_block_hash_for_seq(&tokens, block_size, options);
        group.throughput(Throughput::Elements(u64::from(blocks)));
        group.bench_with_input(BenchmarkId::new("public", blocks), &blocks, |b, _| {
            b.iter(|| {
                black_box(public_context.compute_sequence_hashes(
                    scope,
                    black_box(&tokens),
                    options,
                    Some(&public_blocks),
                ))
            })
        });
        group.bench_with_input(BenchmarkId::new("keyed", blocks), &blocks, |b, _| {
            b.iter(|| {
                black_box(keyed_context.compute_sequence_hashes(
                    scope,
                    black_box(&tokens),
                    options,
                    Some(&public_blocks),
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, tracking_hash);
criterion_main!(benches);
