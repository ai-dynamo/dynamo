// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
//
// Portions adapted from sgl-project/llm-tokenizer v1.3.2 (Apache-2.0).
// Upstream: https://github.com/lightseekorg/smg
// Modifications: removed L0 layer, removed `add_special_tokens` plumbing (Dynamo's
// `Encoder::encode` has no such flag), dropped fingerprinting, retargeted onto
// `crate::traits::Tokenizer`.

//! Tokenizer caching layer (L1: prefix matching at special-token boundaries).
//!
//! Wraps any [`Tokenizer`] in a cache that records prefix tokenizations at every
//! special-token boundary. On a hit, the cached prefix tokens are merged with a
//! fresh encode of the trailing suffix only — turning O(N) tokenization work
//! into O(suffix_len) when prompts share a system prefix.
//!
//! # Correctness
//!
//! Boundaries are taken **only** at positions immediately following a registered
//! special token (e.g. `<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`). Special tokens
//! are atomic in BPE (`special: true, normalized: false`), so splitting there
//! preserves the invariant `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`.
//! No fallback to whitespace or punctuation — better to miss than to corrupt.
//!
//! # Storage normalization
//!
//! Cache hits return [`Encoding::Sp`] (token-ids only), even when the inner
//! tokenizer would have produced [`Encoding::Hf`] (rich offsets/attention/etc).
//! All current downstream consumers in Dynamo only call [`Encoding::token_ids`],
//! so this lossy normalization is safe; revisit if a caller starts reading
//! offsets or attention masks from cached encodings.
//!
//! # Configuration
//!
//! - `special_tokens: Vec<String>` — must be supplied at construction (the
//!   [`Tokenizer`] trait is intentionally minimal and does not expose them).
//!   An empty list disables L1: `encode`/`encode_batch` short-circuit straight
//!   to the inner tokenizer with no lookup, no miss-counter bump, and no
//!   insert attempt.
//! - `max_memory_bytes` — L1 byte budget; entries evicted via approximate LRU.
//!
//! # Provenance
//!
//! Adapted from `llm-tokenizer` v1.3.2 (`cache/l1.rs`, `cache/mod.rs`). L0 and
//! fingerprinting were dropped; L1 alone covers the headline multi-turn-chat
//! workload, and the in-memory cache lifetime is bound to a single tokenizer
//! instance so fingerprint-based invalidation is unnecessary.

mod l1;

use std::sync::Arc;

pub use l1::{CacheEventFn, L1Cache, L1CacheStats};

use crate::{
    Encoding, Result, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Caching wrapper around an inner tokenizer.
///
/// Implements [`Encoder`], [`Decoder`], and [`Tokenizer`]; decode calls pass
/// through to the inner tokenizer (decoding is fast and rarely repeated).
pub struct CachedTokenizer {
    inner: Arc<dyn Tokenizer>,
    l1: L1Cache,
    special_tokens: Vec<String>,
}

impl CachedTokenizer {
    /// Construct a cached tokenizer.
    ///
    /// `special_tokens` is the list of atomic special-token strings the inner
    /// tokenizer recognizes (typically extracted via the HuggingFace tokenizer's
    /// `get_added_tokens_decoder()` filtering by `special == true`). An empty list
    /// disables L1 — `encode`/`encode_batch` short-circuit to the inner tokenizer
    /// without touching the cache or its counters.
    ///
    /// `max_memory_bytes` is the L1 cache byte budget.
    pub fn new(
        inner: Arc<dyn Tokenizer>,
        special_tokens: Vec<String>,
        max_memory_bytes: usize,
    ) -> Self {
        Self {
            inner,
            l1: L1Cache::new(max_memory_bytes),
            special_tokens,
        }
    }

    /// Install hit/miss callbacks so each L1 lookup pushes an event into the
    /// supplied closures (e.g. `Prometheus::Counter::inc`). Replaces any
    /// previously-set observer.
    pub fn with_observer(mut self, on_hit: CacheEventFn, on_miss: CacheEventFn) -> Self {
        self.l1.set_observer(on_hit, on_miss);
        self
    }

    /// Snapshot of L1 cache statistics (cumulative hits/misses/entries/memory).
    pub fn cache_stats(&self) -> L1CacheStats {
        self.l1.stats()
    }

    /// Clear all cached entries and reset counters.
    pub fn clear_cache(&self) {
        self.l1.clear();
    }

    /// Access the underlying tokenizer (e.g. for downcasting to a concrete type).
    pub fn inner(&self) -> &Arc<dyn Tokenizer> {
        &self.inner
    }
}

fn special_token_refs(specials: &[String]) -> Vec<&str> {
    specials.iter().map(String::as_str).collect()
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // Empty specials => no boundaries are ever produced. Skip the lookup,
        // miss-counter bump, and insert attempt entirely — otherwise the
        // tiktoken wrapping path (which deliberately passes an empty list)
        // pays the scan cost on every call with zero chance of a hit.
        if self.special_tokens.is_empty() {
            return self.inner.encode(input);
        }

        let specials = special_token_refs(&self.special_tokens);

        if let Some((prefix_tokens, prefix_len)) = self.l1.longest_prefix_match(input, &specials) {
            let suffix = &input[prefix_len..];
            if suffix.is_empty() {
                return Ok(Encoding::Sp(prefix_tokens));
            }
            let suffix_enc = self.inner.encode(suffix)?;
            let mut merged: Vec<TokenIdType> = prefix_tokens;
            merged.extend_from_slice(suffix_enc.token_ids());
            return Ok(Encoding::Sp(merged));
        }

        // Miss path: full encode, then populate L1 at every boundary.
        let encoding = self.inner.encode(input)?;
        let _ = self
            .l1
            .insert_at_boundaries(input, self.inner.as_ref(), &specials);
        Ok(encoding)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // True passthrough when L1 is disabled — delegate to the inner's native
        // batch path (which may be rayon-parallel for HF) instead of falling
        // through per-item.
        if self.special_tokens.is_empty() {
            return self.inner.encode_batch(inputs);
        }

        // Per-item cache lookup — do NOT delegate to inner.encode_batch, which would
        // bypass the cache. Sequential iteration is fine; if rayon is added later it
        // belongs here, not inside `encode`.
        inputs.iter().map(|&i| self.encode(i)).collect()
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        // Decode is not cached — passthrough to inner.
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HuggingFaceTokenizer;

    const TINYLLAMA_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    fn inner() -> Arc<dyn Tokenizer> {
        Arc::new(HuggingFaceTokenizer::from_file(TINYLLAMA_PATH).expect("load TinyLlama"))
    }

    fn specials() -> Vec<String> {
        vec!["<s>".into(), "</s>".into()]
    }

    #[test]
    fn empty_specials_passes_through_correctly() {
        // L1 disabled by empty specials list — encode must produce correct ids
        // AND short-circuit to the inner tokenizer (no miss-counter bump, no
        // insert attempt). Otherwise the tiktoken integration would log a
        // miss per request with zero hits forever.
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), Vec::new(), 4096);
        let s = "<s>hello world</s>";
        let a = cached.encode(s).unwrap();
        let b = tok.encode(s).unwrap();
        assert_eq!(a.token_ids(), b.token_ids());
        let stats = cached.cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.misses, 0, "empty specials must not increment misses");
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn two_turn_chat_correctness_and_hit() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 64 * 1024);

        let template = "<s>system\nYou are helpful.</s><s>user\n";
        let first = format!("{template}First question?</s>");
        let second = format!("{template}Second different prompt entirely.</s>");

        // Warm the cache.
        let _ = cached.encode(&first).unwrap();

        // Second request: shared prefix → L1 hit, suffix-only fresh encode.
        let cached_second = cached.encode(&second).unwrap();
        let plain_second = tok.encode(&second).unwrap();
        assert_eq!(
            cached_second.token_ids(),
            plain_second.token_ids(),
            "cached encode must equal plain encode for second turn"
        );

        let stats = cached.cache_stats();
        assert!(stats.hits >= 1, "expected L1 hit on second request");
    }

    #[test]
    fn decode_passes_through() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 4096);
        let enc = cached.encode("<s>hello</s>").unwrap();
        let direct = tok.decode(enc.token_ids(), false).unwrap();
        let through = cached.decode(enc.token_ids(), false).unwrap();
        assert_eq!(direct, through);
    }

    #[test]
    fn encode_batch_uses_cache() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 64 * 1024);
        let shared = "<s>system\nShared persona.</s><s>user\n";
        let inputs = [
            format!("{shared}q1</s>"),
            format!("{shared}q2</s>"),
            format!("{shared}q3</s>"),
        ];
        let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let outs = cached.encode_batch(&refs).unwrap();
        assert_eq!(outs.len(), 3);
        // First call populates, second/third hit.
        assert!(cached.cache_stats().hits >= 2, "expected hits on q2 and q3");
    }
}
