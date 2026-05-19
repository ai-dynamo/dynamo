// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
// SPDX-License-Identifier: Apache-2.0
//
// Portions adapted from sgl-project/llm-tokenizer v1.3.2 (Apache-2.0).
// Upstream: https://github.com/lightseekorg/smg
// Modifications: removed `add_special_tokens` plumbing (Dynamo's Encoder has no such
// flag), bound `insert_at_boundaries` on `Encoder` rather than `Tokenizer`, retargeted
// imports onto `crate::traits`.

//! L1 Cache: Special-token boundary prefix cache
//!
//! Caches tokenization results at ALL special token boundaries.
//! Special tokens (like `<|im_start|>`, `<|im_end|>`) are atomic in BPE tokenizers
//! (`special: true, normalized: false`), making them the ONLY safe split points that
//! guarantee correctness: `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`.
//!
//! No fallback to whitespace/punctuation — better to not cache than risk corruption.

use std::{
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::DashMap;

use crate::{TokenIdType, traits::Encoder};

/// Hash type for cache keys
type Blake3Hash = [u8; 32];

/// Number of shards for concurrent access
const NUM_SHARDS: usize = 16;

/// Find ALL special token boundaries in the text.
///
/// **ONLY uses special tokens** — these are atomic (`special: true, normalized: false`)
/// in BPE, guaranteeing `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`.
///
/// Returns positions immediately after each special token (where prefixes can be cached).
/// Boundaries at the very end of the text are filtered out (no suffix left to tokenize).
fn find_special_token_boundaries(text: &str, special_tokens: &[&str]) -> Vec<usize> {
    if special_tokens.is_empty() {
        return Vec::new();
    }

    let mut boundaries = Vec::new();
    for &token in special_tokens {
        let mut start = 0;
        while let Some(pos) = text[start..].find(token) {
            let boundary = start + pos + token.len();
            if boundary < text.len() {
                boundaries.push(boundary);
            }
            start = boundary;
        }
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// A cached prefix entry. Tokens are held behind `Arc<[T]>` for zero-copy cloning.
#[derive(Debug, Clone)]
struct CachedPrefix {
    tokens: Arc<[TokenIdType]>,
    last_accessed: Arc<AtomicU64>,
    size_bytes: usize,
}

/// L1 cache: prefix matching at special-token boundaries.
pub struct L1Cache {
    /// Sharded maps for concurrent access. Key: Blake3 hash of `input[0..boundary]`.
    shards: Vec<Arc<DashMap<Blake3Hash, CachedPrefix>>>,
    max_memory: usize,
    current_memory: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    /// Monotonic counter for LRU timestamps.
    access_counter: AtomicU64,
}

impl L1Cache {
    pub fn new(max_memory: usize) -> Self {
        let shards = (0..NUM_SHARDS).map(|_| Arc::new(DashMap::new())).collect();

        Self {
            shards,
            max_memory,
            current_memory: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    /// Try to find the longest prefix match at a special-token boundary.
    ///
    /// Returns `(cached_tokens, byte_offset)` if found. Caller extends the cached
    /// tokens with a fresh encode of `input[byte_offset..]`.
    pub fn longest_prefix_match(
        &self,
        input: &str,
        special_tokens: &[&str],
    ) -> Option<(Vec<TokenIdType>, usize)> {
        let boundaries = find_special_token_boundaries(input, special_tokens);

        if boundaries.is_empty() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Build all prefix hashes incrementally — O(N).
        let mut hasher = blake3::Hasher::new();
        let mut prefix_hashes = Vec::with_capacity(boundaries.len());
        let mut last_pos = 0;
        let bytes = input.as_bytes();
        for &boundary_pos in &boundaries {
            hasher.update(&bytes[last_pos..boundary_pos]);
            prefix_hashes.push((boundary_pos, *hasher.clone().finalize().as_bytes()));
            last_pos = boundary_pos;
        }

        // Search from the longest boundary down — return first hit.
        for (boundary_pos, hash_bytes) in prefix_hashes.into_iter().rev() {
            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            if let Some(entry) = self.shards[shard_idx].get(&hash_bytes) {
                let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
                entry.last_accessed.store(timestamp, Ordering::Relaxed);

                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some((entry.tokens.to_vec(), boundary_pos));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert prefix entries at every special-token boundary.
    ///
    /// Uses incremental hashing and incremental tokenization (per-segment encode of the
    /// delta text between adjacent boundaries) so populating N entries costs one full
    /// re-tokenize total, split across the segments.
    pub fn insert_at_boundaries<E: Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
        special_tokens: &[&str],
    ) -> anyhow::Result<()> {
        let boundaries = find_special_token_boundaries(input, special_tokens);

        if boundaries.is_empty() {
            return Ok(());
        }

        let mut hasher = blake3::Hasher::new();
        let mut running_tokens: Vec<TokenIdType> = Vec::new();
        let mut last_pos = 0;
        let mut entries_to_insert = Vec::with_capacity(boundaries.len());
        let bytes = input.as_bytes();

        for &boundary_pos in boundaries.iter() {
            let delta_text = &input[last_pos..boundary_pos];

            // 1. Incremental hash.
            hasher.update(&bytes[last_pos..boundary_pos]);
            let hash_bytes: Blake3Hash = *hasher.clone().finalize().as_bytes();

            // 2. Incremental tokenization. Dynamo's Encoder has no `add_special_tokens`
            //    parameter — equivalent to upstream always passing `false` past the first
            //    segment (which is also what Dynamo's HF impl always does for the first).
            let segment_encoding = tokenizer.encode(delta_text)?;
            running_tokens.extend_from_slice(segment_encoding.token_ids());

            // 3. Snapshot prefix tokens as Arc<[T]> for cheap sharing on hits.
            let prefix_tokens: Arc<[TokenIdType]> = running_tokens.as_slice().into();
            let size_bytes = boundary_pos + prefix_tokens.len() * size_of::<TokenIdType>();

            entries_to_insert.push((hash_bytes, prefix_tokens, size_bytes));

            last_pos = boundary_pos;
        }

        if entries_to_insert.is_empty() {
            return Ok(());
        }

        let total_size_needed: usize = entries_to_insert.iter().map(|(_, _, size)| size).sum();

        // Evict if necessary.
        let current = self.current_memory.load(Ordering::Relaxed) as usize;
        if current + total_size_needed > self.max_memory {
            self.evict_lru(total_size_needed);
        }

        // Insert all entries, accounting for replaced entries in memory tracking.
        let current_timestamp = self.access_counter.load(Ordering::Relaxed);
        for (hash_bytes, prefix_tokens, size_bytes) in entries_to_insert {
            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            let cached = CachedPrefix {
                tokens: prefix_tokens,
                last_accessed: Arc::new(AtomicU64::new(current_timestamp)),
                size_bytes,
            };

            if let Some(old) = self.shards[shard_idx].insert(hash_bytes, cached) {
                // Replaced an existing entry — adjust delta only. The counter update is
                // not atomic with the shard insert, so concurrent replacements of the
                // same key can briefly skew the counter. Benign — eviction is best-effort
                // and the drift is bounded to a single entry's size per race.
                let old_size = old.size_bytes as u64;
                let new_size = size_bytes as u64;
                if new_size >= old_size {
                    self.current_memory
                        .fetch_add(new_size - old_size, Ordering::Relaxed);
                } else {
                    self.current_memory
                        .fetch_sub(old_size - new_size, Ordering::Relaxed);
                }
            } else {
                self.current_memory
                    .fetch_add(size_bytes as u64, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Approximate LRU via random sampling: sample K random entries, evict the oldest,
    /// repeat. O(samples) per eviction round — no full scan.
    fn evict_lru(&self, space_needed: usize) {
        const SAMPLE_SIZE: usize = 32;
        let mut freed = 0usize;
        let mut iteration = 0usize;

        while freed < space_needed {
            let mut samples: Vec<(usize, Blake3Hash, u64, usize)> = Vec::with_capacity(SAMPLE_SIZE);

            for i in 0..SAMPLE_SIZE {
                let shard_idx = (iteration * SAMPLE_SIZE + i) % NUM_SHARDS;
                if let Some(entry) = self.shards[shard_idx].iter().next() {
                    let hash = *entry.key();
                    let timestamp = entry.value().last_accessed.load(Ordering::Relaxed);
                    let size = entry.value().size_bytes;
                    samples.push((shard_idx, hash, timestamp, size));
                }
            }

            if samples.is_empty() {
                break;
            }

            if let Some((shard_idx, hash, _, _)) =
                samples.iter().min_by_key(|(_, _, ts, _)| ts).copied()
                && let Some((_, removed)) = self.shards[shard_idx].remove(&hash)
            {
                freed += removed.size_bytes;
                self.current_memory
                    .fetch_sub(removed.size_bytes as u64, Ordering::Relaxed);
            }

            iteration += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    pub fn stats(&self) -> L1CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        L1CacheStats {
            hits,
            misses,
            entries: self.len(),
            memory_bytes: self.current_memory.load(Ordering::Relaxed) as usize,
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
        self.current_memory.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct L1CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub memory_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{HuggingFaceTokenizer, traits::Tokenizer};

    // TinyLlama: real Llama BPE with `<s>` and `</s>` as added tokens with
    // `special: true, normalized: false` — atomic in BPE, safe boundary points.
    const TINYLLAMA_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    const SPECIALS: &[&str] = &["<s>", "</s>"];

    fn load_tokenizer() -> Arc<dyn Tokenizer> {
        Arc::new(HuggingFaceTokenizer::from_file(TINYLLAMA_PATH).expect("load TinyLlama"))
    }

    #[test]
    fn boundaries_are_after_each_special_token_occurrence() {
        let input = "<s>system\nHi</s><s>user\nHello</s>";
        let bounds = find_special_token_boundaries(input, SPECIALS);
        // Drop the trailing boundary (==text.len()), so 3 not 4 boundaries.
        assert_eq!(bounds.len(), 3);
        for w in bounds.windows(2) {
            assert!(w[0] < w[1], "boundaries must be strictly increasing");
        }
        assert!(bounds.iter().all(|&b| b < input.len()));
    }

    #[test]
    fn no_special_tokens_yields_no_boundaries() {
        assert!(find_special_token_boundaries("plain text", &[]).is_empty());
    }

    #[test]
    fn insert_then_lookup_finds_shared_prefix() {
        let cache = L1Cache::new(1024 * 1024);
        let tokenizer = load_tokenizer();

        let warm = "<s>system\nYou are helpful.</s><s>user\nHi</s>";
        cache
            .insert_at_boundaries(warm, tokenizer.as_ref(), SPECIALS)
            .unwrap();
        assert!(!cache.is_empty());

        let target = "<s>system\nYou are helpful.</s><s>user\nDifferent question</s>";
        let (tokens, offset) = cache
            .longest_prefix_match(target, SPECIALS)
            .expect("shared prefix should match");
        assert!(offset > 0);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn miss_increments_misses_counter() {
        let cache = L1Cache::new(1024 * 1024);
        assert!(
            cache
                .longest_prefix_match("plain text no specials", SPECIALS)
                .is_none()
        );
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn hit_increments_hits_counter() {
        let cache = L1Cache::new(1024 * 1024);
        let tokenizer = load_tokenizer();
        let warm = "<s>system\nA.</s><s>user\nB</s>";
        cache
            .insert_at_boundaries(warm, tokenizer.as_ref(), SPECIALS)
            .unwrap();
        let _ = cache.longest_prefix_match(warm, SPECIALS);
        assert!(cache.stats().hits >= 1);
    }

    #[test]
    fn merge_invariant_holds_against_uncached_encode() {
        // Load-bearing correctness check: cached prefix + fresh suffix encode must
        // equal plain encode of the full input. Relies on `<s>`/`</s>` being atomic
        // in TinyLlama's BPE (they are).
        let cache = L1Cache::new(1024 * 1024);
        let tokenizer = load_tokenizer();

        let template = "<s>system\nYou are helpful.</s><s>user\n";
        let warm = format!("{template}First.</s>");
        cache
            .insert_at_boundaries(&warm, tokenizer.as_ref(), SPECIALS)
            .unwrap();

        let target = format!("{template}A completely different second question.</s>");
        let (prefix_tokens, prefix_len) = cache
            .longest_prefix_match(&target, SPECIALS)
            .expect("should find prefix");

        let suffix = &target[prefix_len..];
        let suffix_enc = tokenizer.encode(suffix).unwrap();
        let mut merged = prefix_tokens.clone();
        merged.extend_from_slice(suffix_enc.token_ids());

        let plain = tokenizer.encode(&target).unwrap();
        assert_eq!(
            merged,
            plain.token_ids(),
            "merged tokens must equal plain encode"
        );
    }

    #[test]
    fn eviction_respects_memory_budget() {
        // 4 KB budget — tight enough to force eviction after a few inserts.
        let cache = L1Cache::new(4 * 1024);
        let tokenizer = load_tokenizer();
        for i in 0..50 {
            let input = format!(
                "<s>system\nPersona {i} chatty.</s><s>user\nTurn {i} content here.</s>"
            );
            cache
                .insert_at_boundaries(&input, tokenizer.as_ref(), SPECIALS)
                .unwrap();
        }
        let stats = cache.stats();
        assert!(
            stats.memory_bytes <= 4 * 1024,
            "memory_bytes={} exceeds budget",
            stats.memory_bytes
        );
    }

    #[test]
    fn concurrent_inserts_and_lookups_do_not_corrupt() {
        use std::thread;

        let cache = Arc::new(L1Cache::new(1024 * 1024));
        let tokenizer = load_tokenizer();

        let mut handles = vec![];
        for i in 0..10 {
            let cache_c = cache.clone();
            let tok = tokenizer.clone();
            handles.push(thread::spawn(move || {
                let input = format!("<s>system\nThread {i}.</s><s>user\nThread {i} body.</s>");
                cache_c
                    .insert_at_boundaries(&input, tok.as_ref(), SPECIALS)
                    .unwrap();
                let r = cache_c.longest_prefix_match(&input, SPECIALS);
                assert!(r.is_some(), "thread {i} expected match after insert");
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.stats().memory_bytes > 0);
        assert!(cache.stats().hits >= 10);
    }
}
