// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use dynamo_llm::backend::Decoder;
use dynamo_llm::protocols::common::StopConditions;
use dynamo_llm::tokenizers::DecodeStream;
use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;
use dynamo_llm::tokenizers::traits::{Encoder, Tokenizer};
use dynamo_llm::types::TokenIdType;

const TEST_TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

/// Input Sequence Length for tokenizer
const TARGET_ISL: usize = 8_000;

// A string of length exactly 128 bytes.
const INPUT_STR: &str = "The cat sat by the window, watching raindrops race down the glass. Far thunder rumbled. She purred softly, feeling safe at home.";

/// `cargo bench -- encode` to run it
pub fn encode(c: &mut Criterion) {
    let test_str: &str = &INPUT_STR.repeat(TARGET_ISL / INPUT_STR.len());

    let encoder = HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap();
    let mut group = c.benchmark_group("encode-group");
    group.throughput(Throughput::Bytes(test_str.len() as u64));
    group.bench_function("tokenizer_encode", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(test_str)).unwrap();
        })
    });
    group.finish();
}

pub fn decode(c: &mut Criterion) {
    const TEST_TOKS: [TokenIdType; 34] = [
        450, 6635, 3290, 491, 278, 3474, 29892, 21217, 1153, 513, 307, 567, 8175, 1623, 278, 12917,
        29889, 8413, 266, 5062, 364, 25443, 29889, 2296, 3708, 1127, 4964, 368, 29892, 11223, 9109,
        472, 3271, 29889,
    ];

    let mut group = c.benchmark_group("decode-group");
    group.throughput(Throughput::Elements(TEST_TOKS.len() as u64));
    group.bench_function("tokenizer_decoder", |b| {
        b.iter_with_setup(
            || {
                let tokenizer: Arc<dyn Tokenizer> =
                    Arc::new(HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap());
                let ds = DecodeStream::new(tokenizer, &[], false);
                Decoder::new(ds, StopConditions::default(), false)
            },
            |mut decoder| {
                for tok in black_box(TEST_TOKS) {
                    let _ = decoder.step(tok).unwrap();
                }
            },
        )
    });
    group.finish();
}

pub fn decode_big(c: &mut Criterion) {
    const NUM_TOKENS: usize = 2048;

    const BIG_TEST_TOKS: [TokenIdType; NUM_TOKENS] = [450; NUM_TOKENS];
    let mut group = c.benchmark_group("decode-big-group");
    group.throughput(Throughput::Elements(NUM_TOKENS as u64));
    group.bench_function("tokenizer_decoder_big", |b| {
        b.iter_with_setup(
            || {
                let tokenizer: Arc<dyn Tokenizer> =
                    Arc::new(HuggingFaceTokenizer::from_file(TEST_TOKENIZER).unwrap());
                let ds = DecodeStream::new(tokenizer, &[], false);
                Decoder::new(ds, StopConditions::default(), false)
            },
            |mut decoder| {
                for tok in black_box(&BIG_TEST_TOKS) {
                    let _ = decoder.step(*tok).unwrap();
                }
            },
        )
    });
    group.finish();
}

criterion_group!(benches, encode, decode, decode_big);
criterion_main!(benches);
