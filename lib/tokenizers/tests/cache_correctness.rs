// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
//
// CHAT_TURNS corpus adapted from sgl-project/llm-tokenizer v1.3.2
// (`tests/tokenizer_cache_correctness_test.rs`). The ChatML markers `<|im_start|>` /
// `<|im_end|>` were rewritten to TinyLlama's `<s>` / `</s>` so the test runs offline
// against in-tree tokenizer fixtures. The same corpus is also rendered into Llama-3.1
// format and tested against the bundled mock-llama-3.1 fixture, exercising a second
// special-token family with a different added-token layout (4 boundary tokens instead
// of 2, distinct surface strings `<|begin_of_text|>` / `<|start_header_id|>` /
// `<|end_header_id|>` / `<|eot_id|>`).
//
// All boundary tokens used in both formats are atomic in their respective BPE
// vocabularies (`special: true, normalized: false`), so the same L1 correctness
// invariant is exercised for each:
//
//   tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)

//! Integration test: cached vs uncached encoding must produce identical token IDs across
//! a representative corpus of multi-turn chat prompts, on multiple tokenizer families.

use std::sync::Arc;

use dynamo_tokenizers::{
    CachedTokenizer, HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer},
};

const TINYLLAMA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

/// In-tree mock Llama-3.1 fixture. Has the full set of Llama-3.1 special tokens
/// (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`,
/// reserved_special_token_*, `<|end_of_text|>`) but an empty BPE vocab. Regular text
/// encodes to `[]`; special tokens encode to their atomic IDs. The L1 invariant
/// `tokenize(prefix) ++ tokenize(suffix) == tokenize(prefix+suffix)` still holds —
/// that's exactly what this test asserts, regardless of how trivially the non-special
/// segments tokenize.
const LLAMA31_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/mock-llama-3.1-8b-instruct/tokenizer.json"
);

/// In-tree mock DeepSeek-R1 fixture. Empty BPE vocab like mock-llama-3.1, but registers
/// DeepSeek's chat markers (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`,
/// `<｜end▁of▁sentence｜>`) and tool-call markers (`<｜tool▁calls▁begin｜>`,
/// `<｜tool▁call▁begin｜>`, `<｜tool▁sep｜>`, `<｜tool▁call▁end｜>`, `<｜tool▁calls▁end｜>`).
/// Exercises the cache on a third special-token family whose markers use multibyte code
/// points (`｜`=U+FF5C, `▁`=U+2581), confirming boundary detection and the merge invariant
/// hold for non-ASCII special tokens.
const DEEPSEEK_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/mock-deepseek-r1/tokenizer.json"
);

/// A tokenizer fixture together with the formatter that re-keys the chat corpus into
/// that family's native special-token markers, plus the list of those markers for the
/// `CachedTokenizer` constructor.
struct Setup {
    name: &'static str,
    path: &'static str,
    specials: &'static [&'static str],
    /// Rewrite a corpus turn from the canonical TinyLlama-format CHAT_TURNS entry
    /// (with `<s>role\n...</s>` markers) into this family's native chat format.
    render: fn(tinyllama_format: &str) -> String,
}

/// Identity: CHAT_TURNS entries are already in TinyLlama format.
fn render_llama2(s: &str) -> String {
    s.to_string()
}

/// Convert a TinyLlama-format chat string into Llama-3.1 instruction format.
///
/// `<s>role\ncontent</s>` → `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
/// with a single `<|begin_of_text|>` prepended once at the very start.
fn render_llama3(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 64);
    out.push_str("<|begin_of_text|>");
    let mut remaining = s;
    while let Some(rest) = remaining.strip_prefix("<s>") {
        let end = rest
            .find("</s>")
            .expect("CHAT_TURNS turn must end with </s>");
        let turn = &rest[..end];
        // The role is everything up to the first '\n'; the content is the rest.
        // Some corpus entries put no content after the role (e.g. "<s>system\n</s>"),
        // in which case split_once returns (role, "").
        let (role, content) = turn.split_once('\n').unwrap_or((turn, ""));
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(content);
        out.push_str("<|eot_id|>");
        remaining = &rest[end + "</s>".len()..];
    }
    out
}

/// Convert a TinyLlama-format chat string into DeepSeek-R1 format.
///
/// `<s>system\ncontent</s>`    → `content` (bare, immediately after the BOS)
/// `<s>user\ncontent</s>`      → `<｜User｜>content`
/// `<s>assistant\ncontent</s>` → `<｜Assistant｜>content<｜end▁of▁sentence｜>`
/// with a single `<｜begin▁of▁sentence｜>` prepended once at the very start. Any role
/// other than `system`/`assistant` opens with the `<｜User｜>` marker (matches the corpus,
/// which only uses system/user/assistant).
fn render_deepseek(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 64);
    out.push_str("<｜begin▁of▁sentence｜>");
    let mut remaining = s;
    while let Some(rest) = remaining.strip_prefix("<s>") {
        let end = rest
            .find("</s>")
            .expect("CHAT_TURNS turn must end with </s>");
        let turn = &rest[..end];
        let (role, content) = turn.split_once('\n').unwrap_or((turn, ""));
        match role {
            "system" => out.push_str(content),
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(content);
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {
                out.push_str("<｜User｜>");
                out.push_str(content);
            }
        }
        remaining = &rest[end + "</s>".len()..];
    }
    out
}

const SETUPS: &[Setup] = &[
    Setup {
        name: "tinyllama (Llama-2 family, <s>/</s>)",
        path: TINYLLAMA_PATH,
        specials: &["<s>", "</s>"],
        render: render_llama2,
    },
    Setup {
        name: "mock-llama-3.1 (Llama-3 family, <|begin_of_text|>/<|*_header_id|>/<|eot_id|>)",
        path: LLAMA31_PATH,
        specials: &[
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ],
        render: render_llama3,
    },
    Setup {
        name: "mock-deepseek-r1 (DeepSeek family, <｜begin▁of▁sentence｜>/<｜User｜>/<｜Assistant｜>/<｜end▁of▁sentence｜>)",
        path: DEEPSEEK_PATH,
        specials: &[
            "<｜begin▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
        ],
        render: render_deepseek,
    },
];

fn build_cached_setup(setup: &Setup) -> (Arc<dyn Tokenizer>, CachedTokenizer) {
    let base = Arc::new(
        HuggingFaceTokenizer::from_file(setup.path)
            .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", setup.path)),
    );
    let specials: Vec<String> = setup.specials.iter().map(|s| (*s).to_string()).collect();
    let cached = CachedTokenizer::new(base.clone(), specials, 50 * 1024 * 1024);
    (base, cached)
}

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

#[test]
fn cached_vs_uncached_first_pass() {
    // For each tokenizer family: re-key the corpus into that family's chat format,
    // then encode every turn twice (miss-path, hit-path) and assert byte-exact token
    // equality against an uncached baseline.
    for setup in SETUPS {
        let (base, cached) = build_cached_setup(setup);

        let mut hits_at_start = cached.cache_stats().hits;
        let mut hits_grew_at_least_once = false;

        for (i, raw_turn) in CHAT_TURNS.iter().enumerate() {
            let turn = (setup.render)(raw_turn);
            let plain = base
                .encode(&turn)
                .expect("uncached encode")
                .token_ids()
                .to_vec();

            // First call — miss path; cache populates at every special-token boundary.
            let cached_first = cached
                .encode(&turn)
                .expect("cached encode (miss)")
                .token_ids()
                .to_vec();
            assert_eq!(
                cached_first, plain,
                "[{}] turn {i}: miss-path cached encode != uncached encode",
                setup.name
            );

            // Second call — hit path; merged prefix + suffix must equal plain encode.
            let cached_second = cached
                .encode(&turn)
                .expect("cached encode (hit)")
                .token_ids()
                .to_vec();
            assert_eq!(
                cached_second, plain,
                "[{}] turn {i}: hit-path cached encode != uncached encode",
                setup.name
            );

            let now = cached.cache_stats().hits;
            if now > hits_at_start {
                hits_grew_at_least_once = true;
            }
            hits_at_start = now;
        }

        assert!(
            hits_grew_at_least_once,
            "[{}] expected at least one turn to produce an L1 hit",
            setup.name
        );
    }
}

#[test]
fn cross_turn_shared_prefix_hits() {
    // Turns 0/1/2 share progressively longer prefixes. Encoding them in order should
    // produce L1 hits on turns 1 and 2 (their prefixes were populated on turn 0).
    for setup in SETUPS {
        let (_base, cached) = build_cached_setup(setup);

        let before = cached.cache_stats();
        for raw_turn in &CHAT_TURNS[..3] {
            let _ = cached.encode(&(setup.render)(raw_turn)).unwrap();
        }
        let after = cached.cache_stats();

        assert!(
            after.hits > before.hits,
            "[{}] shared-prefix turns should produce L1 hits (before={}, after={})",
            setup.name,
            before.hits,
            after.hits
        );
    }
}

#[test]
fn cache_disabled_by_empty_specials_is_transparent() {
    // No specials registered -> L1 always misses, every encode goes through the inner
    // tokenizer. Output must still equal uncached encode. Tested per tokenizer family
    // because the relevant code path is the wrapper, not the tokenizer.
    for setup in SETUPS {
        let base = Arc::new(
            HuggingFaceTokenizer::from_file(setup.path)
                .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", setup.path)),
        );
        let cached = CachedTokenizer::new(base.clone(), Vec::new(), 4096);
        for (i, raw_turn) in CHAT_TURNS.iter().enumerate() {
            let turn = (setup.render)(raw_turn);
            let plain = base.encode(&turn).unwrap().token_ids().to_vec();
            let through = cached.encode(&turn).unwrap().token_ids().to_vec();
            assert_eq!(
                plain, through,
                "[{}] turn {i}: transparent encode mismatch",
                setup.name
            );
        }
        assert_eq!(
            cached.cache_stats().hits,
            0,
            "[{}] with no specials, L1 must produce zero hits",
            setup.name
        );
    }
}

/// Build an append-only multi-turn conversation in TinyLlama format. `turns[i]` is the
/// full history at turn `i`: the system prompt plus `i + 1` completed user/assistant
/// exchanges. Every turn is a well-formed sequence of `<s>role\ncontent</s>` blocks (so
/// the per-family `render` fns accept it), and `turns[i]` is a strict prefix of
/// `turns[i + 1]`.
fn growing_chat_turns(n: usize) -> Vec<String> {
    let mut convo = String::from("<s>system\nYou are a helpful assistant.</s>");
    let mut turns = Vec::with_capacity(n);
    for i in 0..n {
        convo.push_str(&format!(
            "<s>user\nQuestion {i} please answer it.</s><s>assistant\nDetailed answer {i} follows here.</s>"
        ));
        turns.push(convo.clone());
    }
    turns
}

#[test]
fn extend_on_hit_matches_uncached_across_growing_turns() {
    // With extend enabled, a growing conversation is encoded turn-by-turn: turn 0 is a
    // miss, every later turn is a partial hit that also deepens the cache. Each turn's
    // tokens must stay byte-exact against an uncached encode, on both tokenizer families
    // (mock-llama-3.1's empty-BPE vocab surfaces any seg_a/seg_b split off-by-one).
    let turns = growing_chat_turns(12);
    for setup in SETUPS {
        let base = Arc::new(
            HuggingFaceTokenizer::from_file(setup.path)
                .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", setup.path)),
        );
        let specials: Vec<String> = setup.specials.iter().map(|s| (*s).to_string()).collect();
        let cached =
            CachedTokenizer::new(base.clone(), specials, 50 * 1024 * 1024).with_extend(true);

        for (i, raw_turn) in turns.iter().enumerate() {
            let turn = (setup.render)(raw_turn);
            let plain = base.encode(&turn).unwrap().token_ids().to_vec();

            // First encode (miss path on turn 0, extend hit-path afterward).
            let first = cached.encode(&turn).unwrap().token_ids().to_vec();
            assert_eq!(
                first, plain,
                "[{}] turn {i}: extend-on first encode != uncached",
                setup.name
            );

            // Second encode exercises the fully-cached prefix + extend path.
            let second = cached.encode(&turn).unwrap().token_ids().to_vec();
            assert_eq!(
                second, plain,
                "[{}] turn {i}: extend-on second encode != uncached",
                setup.name
            );
        }

        assert!(
            cached.cache_stats().hits > 0,
            "[{}] expected L1 hits with extend on",
            setup.name
        );
    }
}

#[test]
fn extend_off_hits_never_insert() {
    // With extension disabled, partial hits must NOT mutate the cache — the original
    // hit-without-insert behavior. After turn 0 populates the cache, encoding the rest
    // of the growing conversation produces hits but adds zero entries.
    let turns = growing_chat_turns(10);
    for setup in SETUPS {
        let (_base, cached) = build_cached_setup(setup); // CachedTokenizer::new => extend off

        let _ = cached.encode(&(setup.render)(&turns[0])).unwrap();
        let entries_after_turn0 = cached.cache_stats().entries;
        let hits_after_turn0 = cached.cache_stats().hits;

        for raw_turn in &turns[1..] {
            let _ = cached.encode(&(setup.render)(raw_turn)).unwrap();
        }

        let after = cached.cache_stats();
        assert_eq!(
            after.entries, entries_after_turn0,
            "[{}] extend-off: hits must not add cache entries (was {}, now {})",
            setup.name, entries_after_turn0, after.entries
        );
        assert!(
            after.hits > hits_after_turn0,
            "[{}] later turns should have produced hits",
            setup.name
        );
    }
}

#[test]
fn extend_on_partial_hit_adds_exactly_one_entry() {
    // End-to-end deepest-only invariant at the `CachedTokenizer` level: turn 0 (miss)
    // populates an entry at every boundary; each later turn is a partial hit that, with
    // extend on, persists exactly ONE new entry (the suffix's deepest boundary) — never
    // the intermediate boundaries. Holds on every family.
    let turns = growing_chat_turns(6);
    for setup in SETUPS {
        let base = Arc::new(
            HuggingFaceTokenizer::from_file(setup.path)
                .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", setup.path)),
        );
        let specials: Vec<String> = setup.specials.iter().map(|s| (*s).to_string()).collect();
        let cached = CachedTokenizer::new(base, specials, 50 * 1024 * 1024).with_extend(true);

        // Turn 0: miss path populates the cache at every boundary.
        let _ = cached.encode(&(setup.render)(&turns[0])).unwrap();

        for (i, raw) in turns.iter().enumerate().skip(1) {
            let before = cached.cache_stats().entries;
            let _ = cached.encode(&(setup.render)(raw)).unwrap();
            let after = cached.cache_stats().entries;
            assert_eq!(
                after,
                before + 1,
                "[{}] turn {i}: partial-hit extend must add exactly one entry (before {before}, after {after})",
                setup.name
            );
        }
    }
}

/// Build an append-only DeepSeek-R1 tool-calling conversation. Each round appends a user
/// question, an assistant turn whose body is a real DeepSeek tool-call block (five nested
/// `<｜tool▁…｜>` special tokens around a JSON payload), a user-delivered tool result, and
/// an assistant answer. `turns[i]` is a strict prefix of `turns[i + 1]`.
fn growing_deepseek_tool_turns(n: usize) -> Vec<String> {
    let mut convo = String::from("<｜begin▁of▁sentence｜>You are a helpful assistant with tools.");
    let mut turns = Vec::with_capacity(n);
    for i in 0..n {
        convo.push_str(&format!(
            "<｜User｜>What is the weather in city {i}?\
             <｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{{\"city\": \"city {i}\"}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>\
             <｜User｜>Tool result: sunny in city {i}.\
             <｜Assistant｜>It is sunny in city {i}.<｜end▁of▁sentence｜>"
        ));
        turns.push(convo.clone());
    }
    turns
}

#[test]
fn extend_correct_with_deepseek_tool_calls() {
    // Tool-call markers that ARE special tokens (DeepSeek's `<｜tool▁…｜>` family) add real
    // cache boundaries. Encoding a growing tool conversation with extend on must stay
    // byte-exact vs an uncached encode straight through the nested multibyte tool tokens.
    let base = Arc::new(
        HuggingFaceTokenizer::from_file(DEEPSEEK_PATH)
            .unwrap_or_else(|e| panic!("load tokenizer {DEEPSEEK_PATH}: {e}")),
    );
    let specials: Vec<String> = [
        "<｜begin▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<｜end▁of▁sentence｜>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let cached = CachedTokenizer::new(base.clone(), specials, 8 * 1024 * 1024).with_extend(true);

    // The tool-call-begin marker must be an atomic special the cache can split on.
    let tool_begin = base
        .encode("<｜tool▁calls▁begin｜>")
        .unwrap()
        .token_ids()
        .to_vec();
    assert_eq!(
        tool_begin.len(),
        1,
        "tool-call-begin must encode atomically"
    );

    let turns = growing_deepseek_tool_turns(8);
    let mut saw_tool_token = false;
    for (i, turn) in turns.iter().enumerate() {
        let plain = base.encode(turn).unwrap().token_ids().to_vec();

        let first = cached.encode(turn).unwrap().token_ids().to_vec();
        assert_eq!(first, plain, "turn {i}: extend-on first encode != uncached");

        let second = cached.encode(turn).unwrap().token_ids().to_vec();
        assert_eq!(
            second, plain,
            "turn {i}: extend-on second encode != uncached"
        );

        if plain.contains(&tool_begin[0]) {
            saw_tool_token = true;
        }
    }
    assert!(
        saw_tool_token,
        "the tool-call special token must actually appear in the encoded turns"
    );
    assert!(
        cached.cache_stats().hits > 0,
        "expected L1 hits across the growing tool conversation"
    );
}

/// Build an append-only conversation whose assistant turns optionally wrap the tool call
/// in plain-text `<tool_call>…</tool_call>` markup (Hermes/Qwen style). When
/// `wrap_in_tool_call` is false the same JSON appears as ordinary assistant text. Both
/// variants share an identical special-token (`<s>`/`</s>`) structure — only the plain
/// text inside the assistant turns differs. Canonical `<s>role\ncontent</s>` form.
fn growing_tool_turns(n: usize, wrap_in_tool_call: bool) -> Vec<String> {
    let mut convo = String::from("<s>system\nYou are a helpful assistant with tools.</s>");
    let mut turns = Vec::with_capacity(n);
    for i in 0..n {
        let call_body =
            format!("{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"city {i}\"}}}}");
        let assistant_call = if wrap_in_tool_call {
            format!("<tool_call>{call_body}</tool_call>")
        } else {
            call_body
        };
        convo.push_str(&format!(
            "<s>user\nWeather in city {i}?</s>\
             <s>assistant\n{assistant_call}</s>\
             <s>tool\nsunny in city {i}</s>\
             <s>assistant\nIt is sunny in city {i}.</s>"
        ));
        turns.push(convo.clone());
    }
    turns
}

#[test]
fn extend_transparent_to_plaintext_tool_markup() {
    // Plain-text `<tool_call>…</tool_call>` markup is NOT a registered special token, so it
    // must be transparent to the cache: (1) encoding stays byte-exact under extend, and
    // (2) the markup adds no special-token boundary — a cache fed the markup variant ends
    // with the same number of entries as one fed the bare-JSON variant (identical `<s>`/
    // `</s>` structure). Run on the HF families where `<tool_call>` is genuinely text.
    for setup in &SETUPS[..2] {
        let markup = growing_tool_turns(8, true);
        let bare = growing_tool_turns(8, false);

        let build = || {
            let base = Arc::new(
                HuggingFaceTokenizer::from_file(setup.path)
                    .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", setup.path)),
            );
            let specials: Vec<String> = setup.specials.iter().map(|s| (*s).to_string()).collect();
            let cached =
                CachedTokenizer::new(base.clone(), specials, 50 * 1024 * 1024).with_extend(true);
            (base, cached)
        };

        // Markup variant: every turn must encode byte-exact vs uncached.
        let (base_m, cache_m) = build();
        for (i, raw) in markup.iter().enumerate() {
            let turn = (setup.render)(raw);
            let plain = base_m.encode(&turn).unwrap().token_ids().to_vec();
            let got = cache_m.encode(&turn).unwrap().token_ids().to_vec();
            assert_eq!(
                got, plain,
                "[{}] markup turn {i}: extend encode != uncached",
                setup.name
            );
        }

        // Bare variant: same special-token structure, no `<tool_call>` tags.
        let (_base_b, cache_b) = build();
        for raw in &bare {
            let _ = cache_b.encode(&(setup.render)(raw)).unwrap();
        }

        assert_eq!(
            cache_m.cache_stats().entries,
            cache_b.cache_stats().entries,
            "[{}] plain-text <tool_call> markup must add no special-token boundaries \
             (markup entries {}, bare entries {})",
            setup.name,
            cache_m.cache_stats().entries,
            cache_b.cache_stats().entries
        );
    }
}
