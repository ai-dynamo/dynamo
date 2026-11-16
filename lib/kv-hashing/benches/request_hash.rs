// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocation-sensitive benchmark for [`dynamo_kv_hashing::Request`] block hashing.
//!
//! Cases:
//! - 128k tokens, block size 128, no multimodal runs (1000 text blocks)
//! - 8k tokens, block size 16, no multimodal runs
//! - 8k tokens, block size 16, one multimodal run in the middle

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_kv_hashing::{Request, RequestMmObjectInfo};

fn request(token_count: usize, mm_info: Vec<RequestMmObjectInfo>) -> Request {
    Request::builder()
        .tokens((0..token_count).map(|i| i as u32).collect::<Vec<_>>())
        .salt(Some("model-tag".to_string()))
        .mm_info(mm_info)
        .build()
        .expect("benchmark request should validate")
}

fn request_hash(c: &mut Criterion) {
    let text_128k = request(128 * 1024, Vec::new());
    let text_8k = request(8 * 1024, Vec::new());
    let mm_8k = request(
        8 * 1024,
        vec![RequestMmObjectInfo {
            mm_hash: 0x00A1_1CE5,
            offset: 4_000,
            length: 32,
        }],
    );

    let mut group = c.benchmark_group("request_into_blocks");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for (id, req, block_size) in [
        ("text_128k_block_128", &text_128k, 128u32),
        ("text_8k_block_16", &text_8k, 16u32),
        ("mm_mid_8k_block_16", &mm_8k, 16u32),
    ] {
        group.throughput(Throughput::Elements(req.tokens().len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(id), req, |b, req| {
            b.iter(|| {
                let blocks = req.into_blocks(black_box(block_size)).unwrap();
                black_box(blocks);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, request_hash);
criterion_main!(benches);
