// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
//
// CHAT_TURNS corpus adapted from sgl-project/llm-tokenizer v1.3.2
// (`tests/tokenizer_cache_correctness_test.rs`). The ChatML markers `<|im_start|>` /
// `<|im_end|>` were rewritten to TinyLlama's `<s>` / `</s>` so the test runs offline
// against the tokenizer fixture bundled at
// `lib/llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json`. Both marker pairs
// are atomic in their respective BPE vocabularies (`special: true, normalized: false`),
// so the same L1 correctness invariant is exercised:
//
//   tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)

//! Integration test: cached vs uncached encoding must produce identical token IDs across
//! a representative corpus of multi-turn chat prompts.

use std::sync::Arc;

use dynamo_tokenizers::{
    CachedTokenizer, HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer},
};

const TINYLLAMA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

/// 29 multi-turn ChatML strings re-keyed onto TinyLlama's `<s>`/`</s>` markers.
const CHAT_TURNS: [&str; 29] = [
    // Basic conversation patterns
    "<s>system\nYou are a helpful AI assistant.</s>",
    "<s>system\nYou are a helpful AI assistant.</s><s>user\nWhat is the capital of France?</s>",
    "<s>system\nYou are a helpful AI assistant.</s><s>user\nWhat is the capital of France?</s><s>assistant\nThe capital of France is Paris.</s>",
    // Different system prompts (testing different prefix patterns)
    "<s>system\nYou are a coding tutor specializing in Rust programming.</s><s>user\nExplain ownership.</s>",
    "<s>system\nYou are a math teacher.</s><s>user\nSolve: 2x + 5 = 13</s>",
    // Long conversation with multiple turns (testing longer prefixes)
    "<s>system\nYou are a helpful AI assistant.</s><s>user\nTell me about deep learning.</s><s>assistant\nDeep learning is a subset of machine learning that uses neural networks with multiple layers.</s><s>user\nWhat are the main architectures?</s>",
    // Code snippets (testing different character patterns)
    "<s>system\nYou are a code reviewer.</s><s>user\nReview this code:\nfn main() {\n    println!(\"Hello, world!\");\n}\n</s>",
    "<s>system\nYou are a code reviewer.</s><s>user\nExplain this Rust code:\nimpl<T> Drop for Box<T> {\n    fn drop(&mut self) { /* ... */ }\n}\n</s>",
    // Mathematical content
    "<s>system\nYou are a math tutor.</s><s>user\nProve that sqrt(2) is irrational using proof by contradiction.</s>",
    "<s>system\nYou are a math tutor.</s><s>user\nCalculate: integral of (x^2 + 3x + 2) dx from 0 to 5</s>",
    // Multilingual content
    "<s>system\nYou are a multilingual assistant.</s><s>user\nTranslate to French: The quick brown fox jumps over the lazy dog.</s>",
    "<s>system\nYou are a multilingual assistant.</s><s>user\n你好，请帮我翻译这句话：I love programming in Rust.</s>",
    "<s>system\nYou are a multilingual assistant.</s><s>user\nこんにちは！Rustについて教えてください。</s>",
    // Special characters and emojis
    "<s>system\nYou are a friendly chatbot.</s><s>user\nWhat do you think about emojis? 😀🎉🚀💻</s>",
    "<s>system\nYou are a data analyst.</s><s>user\nAnalyze this: {\"name\": \"test\", \"value\": 42, \"nested\": {\"key\": \"value\"}}</s>",
    // Very long message (testing large token counts)
    "<s>system\nYou are a literature expert.</s><s>user\nAnalyze the themes in this passage: In the vast expanse of the digital realm, where bits and bytes dance in harmonious symphony, there exists a paradigm that transcends mere computation. This paradigm, known as machine learning, represents humanity's quest to imbue silicon with the spark of cognition. Deep neural networks, inspired by the intricate architecture of biological brains, layer upon layer of artificial neurons, each connection a synapse firing in the dark recesses of mathematical space. Through gradient descent, these networks learn patterns invisible to human perception, extracting meaning from chaos, signal from noise. The transformer architecture revolutionized this field, introducing attention mechanisms that allowed models to focus on relevant information, much like how humans selectively attend to important details in their environment.</s>",
    // Edge case: Multiple special tokens in sequence
    "<s>system\nYou are helpful.</s><s>user\nHi</s><s>assistant\nHello!</s><s>user\nHow are you?</s>",
    // Edge case: Empty-ish messages
    "<s>system\n</s><s>user\nTest</s>",
    "<s>system\nBrief.</s><s>user\nOK</s>",
    // Technical documentation style
    "<s>system\nYou are a technical writer.</s><s>user\nDocument the following API:\n\n```rust\npub struct CachedTokenizer {\n    inner: Arc<dyn Tokenizer>,\n    l1: L1Cache,\n}\n\nimpl Encoder for CachedTokenizer {\n    fn encode(&self, input: &str) -> Result<Encoding>;\n}\n```\n</s>",
    // Conversation with code review
    "<s>system\nYou are a senior Rust developer.</s><s>user\nReview for correctness:\n\nlet specials: Vec<&str> = self.special_tokens.iter().map(String::as_str).collect();</s>",
    // Markdown formatted content
    "<s>system\nYou are a documentation assistant.</s><s>user\nFormat this as markdown:\n\n# Cache Architecture\n\n## L1 Cache\n- Prefix match\n- Special token boundaries\n- 50MB memory\n</s>",
    // Complex nested structures
    "<s>system\nYou are a JSON expert.</s><s>user\nValidate this JSON:\n{\n  \"tokenizer_cache\": {\n    \"enabled\": true,\n    \"max_memory\": 52428800,\n    \"stats\": {\n      \"hits\": [1, 2, 3],\n      \"misses\": {\"count\": 5}\n    }\n  }\n}\n</s>",
    // SQL queries
    "<s>system\nYou are a database expert.</s><s>user\nOptimize this query:\nSELECT u.name, COUNT(p.id) as post_count\nFROM users u\nLEFT JOIN posts p ON u.id = p.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id, u.name\nHAVING COUNT(p.id) > 5\nORDER BY post_count DESC;</s>",
    // Regex patterns
    "<s>system\nYou are a regex expert.</s><s>user\nExplain this regex: ^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\\.)+[a-zA-Z]{2,}$</s>",
    // Command line examples
    "<s>system\nYou are a DevOps engineer.</s><s>user\nExplain this command:\ncargo bench --bench tokenizer_benchmark -- --color=never | tee results.txt</s>",
    // Unicode edge cases
    "<s>system\nYou are helpful.</s><s>user\nTest: café, naïve, Zürich, 北京, 東京, मुंबई, Москва</s>",
    // Mixed content complexity
    "<s>system\nYou are a software architect.</s><s>user\nDesign a caching system that:\n1. Handles 10K+ QPS\n2. Maintains 99.9% uptime\n3. Supports L1 (prefix) caching\n4. Uses Blake3 for hashing\n5. Implements LRU eviction\n6. Thread-safe with lock-free reads\n</s>",
    // Very long technical discussion
    "<s>system\nYou are a compiler expert.</s><s>user\nExplain why BPE tokenizers are not prefix-stable:\n\nThe core issue is that BPE applies merges based on local context. When you tokenize 'prefix' alone, it might apply merge rules differently than when tokenizing 'prefix + suffix' as a whole. For example:\n\ntokenize('hello world') might produce [hello, _world]\ntokenize('hello') + tokenize(' world') might produce [hel, lo, _wo, rld]\n\nThis is because the merge rules see different contexts. The space before 'world' in the first case is part of the token boundary, but in the second case, ' world' is tokenized in isolation.\n\nSpecial tokens solve this because they are:\n1. Atomic (never split or merged)\n2. Protected from normalization\n3. Marked with special: true flag\n4. Have normalized: false property\n\nThis guarantees: tokenize(prefix + special + suffix) = tokenize(prefix + special) + tokenize(suffix)\n\nOur L1 cache exploits this by:\n1. Finding all special token boundaries\n2. Re-tokenizing prefixes at those boundaries\n3. Caching the exact token IDs\n4. On cache hit, appending suffix tokens\n</s>",
];

fn build_cached() -> (Arc<dyn Tokenizer>, CachedTokenizer) {
    let base = Arc::new(
        HuggingFaceTokenizer::from_file(TINYLLAMA_PATH).expect("load TinyLlama tokenizer"),
    );
    let specials = vec!["<s>".to_string(), "</s>".to_string()];
    let cached = CachedTokenizer::new(base.clone(), specials, 50 * 1024 * 1024);
    (base, cached)
}

#[test]
fn cached_vs_uncached_first_pass() {
    // First-pass through each turn: cold cache, then identical request — both produce
    // the same token IDs as the uncached tokenizer (covers both miss-path and hit-path).
    let (base, cached) = build_cached();

    let mut hits_at_start = cached.cache_stats().hits;
    let mut hits_grew_at_least_once = false;

    for (i, turn) in CHAT_TURNS.iter().enumerate() {
        let plain = base
            .encode(turn)
            .expect("uncached encode")
            .token_ids()
            .to_vec();

        // First call — miss path; cache populates at every special-token boundary.
        let cached_first = cached
            .encode(turn)
            .expect("cached encode (miss)")
            .token_ids()
            .to_vec();
        assert_eq!(
            cached_first, plain,
            "turn {i}: miss-path cached encode != uncached encode"
        );

        // Second call — hit path; merged prefix + suffix must equal plain encode.
        let cached_second = cached
            .encode(turn)
            .expect("cached encode (hit)")
            .token_ids()
            .to_vec();
        assert_eq!(
            cached_second, plain,
            "turn {i}: hit-path cached encode != uncached encode"
        );

        let now = cached.cache_stats().hits;
        if now > hits_at_start {
            hits_grew_at_least_once = true;
        }
        hits_at_start = now;
    }

    assert!(
        hits_grew_at_least_once,
        "expected at least one turn to produce an L1 hit"
    );
}

#[test]
fn cross_turn_shared_prefix_hits() {
    // Turns 0/1/2 share progressively longer prefixes. Encoding them in order should
    // produce L1 hits on turns 1 and 2 (their prefixes were populated on turn 0).
    let (_base, cached) = build_cached();

    let before = cached.cache_stats();
    for &turn in &CHAT_TURNS[..3] {
        let _ = cached.encode(turn).unwrap();
    }
    let after = cached.cache_stats();

    assert!(
        after.hits > before.hits,
        "shared-prefix turns should produce L1 hits (before={}, after={})",
        before.hits,
        after.hits
    );
}

#[test]
fn cache_disabled_by_empty_specials_is_transparent() {
    // No specials registered -> L1 always misses, every encode goes through the inner
    // tokenizer. Output must still equal uncached encode.
    let base = Arc::new(
        HuggingFaceTokenizer::from_file(TINYLLAMA_PATH).expect("load TinyLlama tokenizer"),
    );
    let cached = CachedTokenizer::new(base.clone(), Vec::new(), 4096);
    for (i, turn) in CHAT_TURNS.iter().enumerate() {
        let plain = base.encode(turn).unwrap().token_ids().to_vec();
        let through = cached.encode(turn).unwrap().token_ids().to_vec();
        assert_eq!(plain, through, "turn {i}: transparent encode mismatch");
    }
    assert_eq!(
        cached.cache_stats().hits,
        0,
        "with no specials, L1 must produce zero hits"
    );
}
