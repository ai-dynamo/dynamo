// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use dynamo_llm::backend::Decoder;
use dynamo_llm::protocols::common::StopConditions;
use dynamo_llm::tokenizers::DecodeStream;
use dynamo_llm::tokenizers::FastTokenizer;
use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;
use dynamo_llm::tokenizers::tiktoken::TikTokenTokenizer;
use dynamo_llm::tokenizers::traits::{Encoder, Tokenizer};
use dynamo_llm::types::TokenIdType;
use std::path::Path;

const TEST_TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

const TEST_TIKTOKEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/sample-models/mock-tiktoken/tiktoken.model"
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
                Decoder::new(ds, StopConditions::default(), false, None)
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
                Decoder::new(ds, StopConditions::default(), false, None)
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

pub fn tiktoken_encode(c: &mut Criterion) {
    let test_str: &str = &INPUT_STR.repeat(TARGET_ISL / INPUT_STR.len());

    let encoder = TikTokenTokenizer::from_file_auto(TEST_TIKTOKEN).unwrap();
    let mut group = c.benchmark_group("tiktoken-encode-group");
    group.throughput(Throughput::Bytes(test_str.len() as u64));
    group.bench_function("tiktoken_encode", |b| {
        b.iter(|| {
            let _ = encoder.encode(black_box(test_str)).unwrap();
        })
    });
    group.finish();
}

pub fn tiktoken_decode(c: &mut Criterion) {
    // Encode a test string to get realistic token IDs for this tokenizer
    let encoder = TikTokenTokenizer::from_file_auto(TEST_TIKTOKEN).unwrap();
    let encoding = encoder.encode(INPUT_STR).unwrap();
    let test_toks: Vec<TokenIdType> = encoding.token_ids().to_vec();

    let mut group = c.benchmark_group("tiktoken-decode-group");
    group.throughput(Throughput::Elements(test_toks.len() as u64));
    group.bench_function("tiktoken_decoder", |b| {
        let toks = test_toks.clone();
        b.iter_with_setup(
            || {
                let tokenizer: Arc<dyn Tokenizer> =
                    Arc::new(TikTokenTokenizer::from_file_auto(TEST_TIKTOKEN).unwrap());
                let ds = DecodeStream::new(tokenizer, &[], false);
                Decoder::new(ds, StopConditions::default(), false, None)
            },
            |mut decoder| {
                for tok in black_box(&toks) {
                    let _ = decoder.step(*tok).unwrap();
                }
            },
        )
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Tokenizer backend benchmarks
//
// By default these use the in-tree TinyLlama tokenizer. Override with a
// production-size tokenizer for more realistic numbers:
//   TOKENIZER_PATH=/path/to/tokenizer.json cargo bench -- fastokens
//   TOKENIZER_PATH=Qwen/Qwen3-0.6B        cargo bench -- fastokens
// ---------------------------------------------------------------------------

/// Default HuggingFace model to download when TOKENIZER_PATH is not set.
const DEFAULT_HF_MODEL: &str = "Qwen/Qwen3-0.6B";

/// Resolve a tokenizer.json path from TOKENIZER_PATH env var or download from HF Hub.
fn resolve_tokenizer_path() -> String {
    let input = std::env::var("TOKENIZER_PATH").ok();

    if let Some(ref p) = input
        && Path::new(p).is_file()
    {
        return p.clone();
    }

    let model_name = input.as_deref().unwrap_or(DEFAULT_HF_MODEL);
    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to create HuggingFace API client");

    let repo = api.model(model_name.to_string());
    repo.get("tokenizer.json")
        .expect("Failed to download tokenizer.json from HuggingFace Hub")
        .display()
        .to_string()
}

const FASTOKENS_BATCH_SIZE: usize = 64;

pub fn fastokens_encode(c: &mut Criterion) {
    let tokenizer_path = resolve_tokenizer_path();
    let test_str: &str = &INPUT_STR.repeat(TARGET_ISL / INPUT_STR.len());

    let hf_encoder = HuggingFaceTokenizer::from_file(&tokenizer_path).unwrap();
    let fast_encoder = FastTokenizer::from_file(&tokenizer_path).unwrap();

    // Verify parity before benchmarking
    let hf_ids = hf_encoder.encode(INPUT_STR).unwrap();
    let fast_ids = fast_encoder.encode(INPUT_STR).unwrap();
    assert_eq!(
        hf_ids.token_ids(),
        fast_ids.token_ids(),
        "fastokens and HuggingFace must produce identical token IDs"
    );

    let mut group = c.benchmark_group("fastokens-encode");
    group.throughput(Throughput::Bytes(test_str.len() as u64));

    group.bench_function("hf_encode", |b| {
        b.iter(|| {
            let _ = hf_encoder.encode(black_box(test_str)).unwrap();
        })
    });

    group.bench_function("fastokens_encode", |b| {
        b.iter(|| {
            let _ = fast_encoder.encode(black_box(test_str)).unwrap();
        })
    });

    group.finish();
}

pub fn fastokens_batch_encode(c: &mut Criterion) {
    let tokenizer_path = resolve_tokenizer_path();
    let batch: Vec<&str> = (0..FASTOKENS_BATCH_SIZE).map(|_| INPUT_STR).collect();
    let total_bytes: u64 = batch.iter().map(|s| s.len() as u64).sum();

    let hf_encoder = HuggingFaceTokenizer::from_file(&tokenizer_path).unwrap();
    let fast_encoder = FastTokenizer::from_file(&tokenizer_path).unwrap();

    // Verify batch parity before benchmarking
    let hf_batch = hf_encoder.encode_batch(&batch).unwrap();
    let fast_batch = fast_encoder.encode_batch(&batch).unwrap();
    assert_eq!(
        hf_batch.len(),
        fast_batch.len(),
        "batch result count mismatch: hf={} vs ft={}",
        hf_batch.len(),
        fast_batch.len()
    );
    for (i, (hf_enc, ft_enc)) in hf_batch.iter().zip(fast_batch.iter()).enumerate() {
        assert_eq!(
            hf_enc.token_ids(),
            ft_enc.token_ids(),
            "batch item {i}: fastokens and HuggingFace must produce identical token IDs"
        );
    }

    let mut group = c.benchmark_group("fastokens-batch-encode");
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("hf_batch_encode", |b| {
        b.iter(|| {
            let _ = hf_encoder.encode_batch(black_box(&batch)).unwrap();
        })
    });

    group.bench_function("fastokens_batch_encode", |b| {
        b.iter(|| {
            let _ = fast_encoder.encode_batch(black_box(&batch)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    encode,
    decode,
    decode_big,
    tiktoken_encode,
    tiktoken_decode,
    fastokens_encode,
    fastokens_batch_encode
);
criterion_main!(benches);
