// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sequential vs rayon-parallel block hashing for `TokenBlockSequence::split_tokens`.
//!
//! The parallel arm reproduces the previous implementation so the two can be
//! compared on the same machine:
//!
//! ```bash
//! cargo bench -p dynamo-llm --bench block_hash
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_llm::tokens::{Token, TokenBlockSequence, compute_hash_v2};
use rayon::prelude::*;

const BLOCK_SIZE: u32 = 32;
const SALT: u64 = 0x5eed;

fn tokens(n: usize) -> Vec<Token> {
    (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect()
}

fn parallel_block_hashes(tokens: &[Token], block_size: usize, salt: u64) -> Vec<u64> {
    tokens
        .par_chunks_exact(block_size)
        .map(|chunk| compute_hash_v2(bytemuck::cast_slice(chunk), salt))
        .collect()
}

fn bench_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_tokens");
    for &n in &[4_096usize, 131_072, 1_048_576] {
        let input = tokens(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("sequential", n), &input, |b, input| {
            b.iter(|| TokenBlockSequence::split_tokens(input, BLOCK_SIZE, SALT))
        });
        group.bench_with_input(
            BenchmarkId::new("rayon_block_hashes_only", n),
            &input,
            |b, input| b.iter(|| parallel_block_hashes(input, BLOCK_SIZE as usize, SALT)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_split);
criterion_main!(benches);
