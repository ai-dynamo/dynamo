// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmarks for TCP codec performance
//!
//! Run with: cargo bench --bench tcp_codec_perf

use bytes::{Bytes, BytesMut};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_runtime::pipeline::network::codec::{
    TcpRequestMessage, TcpResponseMessage, TwoPartCodec, TwoPartMessage,
};
use tokio_util::codec::Decoder;

/// Benchmark request encoding (hot path operation)
fn bench_request_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcp_request_encoding");

    // Test different payload sizes
    for size in [102_400, 1_024_000, 31_000_000].iter() {
        let payload = Bytes::from(vec![0u8; *size]);
        let endpoint = "api.endpoint.test".to_string();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let msg = TcpRequestMessage::new(endpoint.clone(), payload.clone());
                let encoded = msg.encode().unwrap();
                black_box(encoded);
            });
        });
    }

    group.finish();
}

/// Benchmark response encoding
fn bench_response_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcp_response_encoding");

    for size in [100, 1024, 10_240, 102_400].iter() {
        let data = Bytes::from(vec![0u8; *size]);

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let msg = TcpResponseMessage::new(data.clone());
                let encoded = msg.encode().unwrap();
                black_box(encoded);
            });
        });
    }

    group.finish();
}

/// Benchmark request decoding
fn bench_request_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcp_request_decoding");

    for size in [100, 1024, 10_240, 102_400].iter() {
        let payload = Bytes::from(vec![0u8; *size]);
        let msg = TcpRequestMessage::new("api.endpoint.test".to_string(), payload);
        let encoded = msg.encode().unwrap();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let decoded = TcpRequestMessage::decode(&encoded).unwrap();
                black_box(decoded);
            });
        });
    }

    group.finish();
}

fn bench_two_part_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_part_complete_decoding");
    let codec = TwoPartCodec::default();

    for size in [0, 100, 1024, 10_240, 102_400, 1_048_576] {
        let encoded = codec
            .encode_message(TwoPartMessage::from_parts(
                Bytes::from(vec![b'h'; 256]),
                Bytes::from(vec![0; size]),
            ))
            .unwrap();
        group.throughput(Throughput::Bytes(encoded.len() as u64));
        group.bench_function(BenchmarkId::new("owned_bytes", size), |b| {
            b.iter(|| {
                black_box(codec.decode_message(black_box(encoded.clone())).unwrap());
            });
        });
        group.bench_function(BenchmarkId::new("copy_then_decode", size), |b| {
            b.iter(|| {
                let data = black_box(encoded.clone());
                let mut buf = BytesMut::from(data.as_ref());
                black_box(codec.clone().decode(&mut buf).unwrap().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark response decoding
fn bench_response_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcp_response_decoding");

    for size in [100, 1024, 10_240, 102_400].iter() {
        let data = Bytes::from(vec![0u8; *size]);
        let msg = TcpResponseMessage::new(data);
        let encoded = msg.encode().unwrap();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let decoded = TcpResponseMessage::decode(&encoded).unwrap();
                black_box(decoded);
            });
        });
    }

    group.finish();
}

/// Benchmark full encode-decode cycle for requests
fn bench_request_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcp_request_roundtrip");

    for size in [100, 1024, 10_240].iter() {
        let payload = Bytes::from(vec![0u8; *size]);
        let endpoint = "api.endpoint.test".to_string();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let msg = TcpRequestMessage::new(endpoint.clone(), payload.clone());
                let encoded = msg.encode().unwrap();
                let decoded = TcpRequestMessage::decode(&encoded).unwrap();
                black_box(decoded);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_request_encoding,
    bench_response_encoding,
    bench_request_decoding,
    bench_two_part_decoding,
    bench_response_decoding,
    bench_request_roundtrip,
);
criterion_main!(benches);
