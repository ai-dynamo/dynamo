// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-frame cost of the ingress response encode.
//!
//! `Ingress::pump_response_stream` encodes one frame per item the engine yields,
//! so for a streaming response this runs once per generated token. Each frame is
//! encoded into its own buffer, which makes the buffer's starting capacity — not
//! just the allocation count — the thing that decides how much work a frame
//! costs.
//!
//! Three arms encode an identical frame and differ only in where the bytes land:
//!
//! - `encode` builds a `Vec` inside the codec. `serde_json::to_vec` pre-sizes
//!   that `Vec`; `rmp_serde::to_vec_named` starts it at zero capacity, so the
//!   two codecs are expected to diverge here and only here.
//! - `encode_into_zero_capacity` writes into a caller-owned `Vec` that also
//!   starts empty. It isolates the writer change from the sizing change, so a
//!   difference between this arm and the next is attributable to capacity alone.
//! - `encode_into_presized` is what the encoder does now: a caller-owned `Vec`
//!   started at [`RESPONSE_ENCODE_CAPACITY_HINT`].
//!
//! Every arm allocates and frees within the timed region, so the numbers include
//! the buffer's whole lifetime rather than just the serialization.
//!
//! The arms are asserted byte-identical before any of them is timed — a hot path
//! that quietly emitted different bytes would surface only as a decode failure at
//! the peer. That the real encoder agrees with the `encode_into_presized` arm is
//! pinned separately by `serde_ingress_encoder_matches_encode_byte_for_byte` in
//! `pipeline::network`.
//!
//! Run with: cargo bench --bench ingress_response_encode

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_runtime::pipeline::network::{
    NetworkStreamWrapper, RESPONSE_ENCODE_CAPACITY_HINT, RequestPlanePayloadCodec,
};
use dynamo_runtime::protocols::annotated::Annotated;
use serde::Serialize;

/// Stand-in for what a Rust-native engine streams during decode: some token ids
/// and their position. A typed struct rather than a `serde_json::Value`, because
/// the engines that reach `SerdeIngressPayloadAdapter` hand it typed responses —
/// routing a `Value` through here would time serde's dynamic walk instead of the
/// encode under test.
#[derive(Serialize)]
struct TokenFrame {
    token_ids: Vec<u32>,
    index: u64,
}

/// The exact envelope the response encoder builds, so the bench serializes the
/// same shape the wire carries: the `Annotated` wrapper plus the stream marker.
fn frame(tokens: usize) -> NetworkStreamWrapper<Annotated<TokenFrame>> {
    NetworkStreamWrapper {
        data: Some(Annotated::from_data(TokenFrame {
            // Spread across the msgpack integer widths rather than clustering in
            // the single-byte range, so a wider frame costs what a real one does.
            token_ids: (0..tokens as u32).map(|i| 1_000 + i * 37).collect(),
            index: 7,
        })),
        complete_final: false,
    }
}

fn bench_ingress_response_encode(c: &mut Criterion) {
    for codec in [
        RequestPlanePayloadCodec::Msgpack,
        RequestPlanePayloadCodec::Json,
    ] {
        let mut group = c.benchmark_group(format!("ingress_response_encode_{}", codec.name()));

        // 1 token is steady-state decode; the wider frames are what speculative
        // decode and MTP emit, and they show how the arms scale with frame size.
        for tokens in [1usize, 4, 16] {
            let frame = frame(tokens);

            let expected = codec.encode(&frame).expect("reference encode");
            let mut from_zero = Vec::new();
            codec
                .encode_into(&frame, &mut from_zero)
                .expect("zero-capacity encode");
            let mut presized = Vec::with_capacity(RESPONSE_ENCODE_CAPACITY_HINT);
            codec
                .encode_into(&frame, &mut presized)
                .expect("presized encode");
            assert_eq!(
                expected,
                from_zero,
                "codec={} tokens={tokens}",
                codec.name()
            );
            assert_eq!(expected, presized, "codec={} tokens={tokens}", codec.name());

            group.throughput(Throughput::Bytes(expected.len() as u64));

            group.bench_with_input(BenchmarkId::new("encode", tokens), &frame, |b, frame| {
                b.iter(|| black_box(codec.encode(frame).expect("encode")));
            });

            group.bench_with_input(
                BenchmarkId::new("encode_into_zero_capacity", tokens),
                &frame,
                |b, frame| {
                    b.iter(|| {
                        let mut bytes = Vec::new();
                        codec.encode_into(frame, &mut bytes).expect("encode_into");
                        black_box(bytes)
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("encode_into_presized", tokens),
                &frame,
                |b, frame| {
                    b.iter(|| {
                        let mut bytes = Vec::with_capacity(RESPONSE_ENCODE_CAPACITY_HINT);
                        codec.encode_into(frame, &mut bytes).expect("encode_into");
                        black_box(bytes)
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench_ingress_response_encode);
criterion_main!(benches);
