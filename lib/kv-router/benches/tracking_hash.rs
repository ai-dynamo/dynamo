// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_kv_router::{
    RoutingPartitionRef, TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
};
use tempfile::NamedTempFile;

const BENCHMARK_KEY: [u8; 32] = [0x5a; 32];
const BLAKE3_SCOPE_DOMAIN: &[u8] = b"dynamo.router.tracking-hash/benchmark-blake3-scope\0";
const BLAKE3_BLOCK_DOMAIN: &[u8] = b"dynamo.router.tracking-hash/benchmark-blake3-block\0";
const BLAKE3_CHAIN_DOMAIN: &[u8] = b"dynamo.router.tracking-hash/benchmark-blake3-chain\0";

fn benchmark_blake3_sequence_hashes(
    provider_key: &[u8; 32],
    scope: TrackingHashScope<'_>,
    tokens: &[u32],
) -> Vec<u64> {
    let mut scope_hasher = blake3::Hasher::new_keyed(provider_key);
    scope_hasher.update(BLAKE3_SCOPE_DOMAIN);
    for value in [
        "benchmark",
        scope.partition.model_name,
        scope.partition.routing_group,
    ] {
        scope_hasher.update(&(value.len() as u64).to_le_bytes());
        scope_hasher.update(value.as_bytes());
    }
    scope_hasher.update(&scope.block_size.to_le_bytes());
    let scope_key = *scope_hasher.finalize().as_bytes();

    let block_size = scope.block_size as usize;
    if block_size == 0 {
        return Vec::new();
    }
    let mut sequence_hashes: Vec<u64> = Vec::with_capacity(tokens.len() / block_size);
    for chunk in tokens.chunks_exact(block_size) {
        let mut block_hasher = blake3::Hasher::new_keyed(&scope_key);
        block_hasher.update(BLAKE3_BLOCK_DOMAIN);
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u32 is plain-old-data and its little-endian memory
            // representation matches the canonical token encoding.
            let chunk_bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(chunk),
                )
            };
            block_hasher.update(chunk_bytes);
        }
        #[cfg(not(target_endian = "little"))]
        for token in chunk {
            block_hasher.update(&token.to_le_bytes());
        }
        let block_hash =
            u64::from_le_bytes(block_hasher.finalize().as_bytes()[..8].try_into().unwrap());
        let sequence_hash = if let Some(parent) = sequence_hashes.last().copied() {
            let mut chain_hasher = blake3::Hasher::new_keyed(&scope_key);
            chain_hasher.update(BLAKE3_CHAIN_DOMAIN);
            chain_hasher.update(&parent.to_le_bytes());
            chain_hasher.update(&block_hash.to_le_bytes());
            u64::from_le_bytes(chain_hasher.finalize().as_bytes()[..8].try_into().unwrap())
        } else {
            block_hash
        };
        sequence_hashes.push(sequence_hash);
    }
    sequence_hashes
}

fn tracking_hash(c: &mut Criterion) {
    let mut key_file = NamedTempFile::new().unwrap();
    key_file.write_all(&BENCHMARK_KEY).unwrap();
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
        partition: RoutingPartitionRef::new("benchmark-model", "default"),
        block_size,
    };

    let mut group = c.benchmark_group("tracking_hash");
    for blocks in [1_u32, 32, 128, 62_500] {
        let tokens = (0..blocks * block_size).collect::<Vec<_>>();
        let options = BlockHashOptions::default();
        let public_blocks = compute_block_hash_for_seq(&tokens, block_size, options);
        group.throughput(Throughput::Elements(u64::from(blocks)));
        group.bench_with_input(BenchmarkId::new("public", blocks), &blocks, |b, _| {
            b.iter(|| {
                black_box(public_context.compute_sequence_hashes_for_tracking(
                    scope,
                    black_box(&tokens),
                    options,
                    true,
                    Some(&public_blocks),
                ))
            })
        });
        group.bench_with_input(BenchmarkId::new("keyed-xxh3", blocks), &blocks, |b, _| {
            b.iter(|| {
                black_box(keyed_context.compute_sequence_hashes_for_tracking(
                    scope,
                    black_box(&tokens),
                    options,
                    true,
                    Some(&public_blocks),
                ))
            })
        });
        group.bench_with_input(
            BenchmarkId::new("keyed-blake3-reference", blocks),
            &blocks,
            |b, _| {
                b.iter(|| {
                    black_box(benchmark_blake3_sequence_hashes(
                        &BENCHMARK_KEY,
                        scope,
                        black_box(&tokens),
                    ))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, tracking_hash);
criterion_main!(benches);
