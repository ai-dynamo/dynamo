// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
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

/// Optional per-event observer. `on_hit` runs after each cache hit, `on_miss`
/// after each miss — wired by `CachedTokenizer::with_observer` to push events
/// straight into Prometheus counters without a periodic sampling step.
pub type CacheEventFn = Arc<dyn Fn() + Send + Sync>;

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
    on_hit: Option<CacheEventFn>,
    on_miss: Option<CacheEventFn>,
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
            on_hit: None,
            on_miss: None,
        }
    }

    /// Install hit/miss callbacks. Replaces any previously-set observers.
    pub fn set_observer(&mut self, on_hit: CacheEventFn, on_miss: CacheEventFn) {
        self.on_hit = Some(on_hit);
        self.on_miss = Some(on_miss);
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
            if let Some(cb) = &self.on_miss {
                cb();
            }
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
                if let Some(cb) = &self.on_hit {
                    cb();
                }
                return Some((entry.tokens.to_vec(), boundary_pos));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(cb) = &self.on_miss {
            cb();
        }
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
            // Budget by the resident token-vector bytes only. The prefix text is hashed
            // and discarded (never stored), and the 32 B key + Arc/bucket overhead is a
            // small per-entry constant — negligible against the token data for the
            // long-prefix workloads this cache targets.
            let size_bytes = prefix_tokens.len() * size_of::<TokenIdType>();

            entries_to_insert.push((hash_bytes, prefix_tokens, size_bytes));

            last_pos = boundary_pos;
        }

        self.insert_entries(entries_to_insert);
        Ok(())
    }

    /// Commit a batch of `(key, tokens, size_bytes)` entries: enforce the byte budget
    /// (skip if the batch can't fit an empty cache; otherwise evict only the deficit),
    /// then insert with memory accounting. Shared by the miss path
    /// ([`insert_at_boundaries`]) and the partial-hit path ([`extend_after_match`]).
    fn insert_entries(&self, entries: Vec<(Blake3Hash, Arc<[TokenIdType]>, usize)>) {
        if entries.is_empty() {
            return;
        }

        let total_size_needed: usize = entries.iter().map(|(_, _, size)| size).sum();

        // If this batch can't fit even into an empty cache, skip it rather than
        // evicting everything for a guaranteed overflow.
        if total_size_needed > self.max_memory {
            return;
        }

        // Evict only the deficit, not the full batch size — otherwise a large batch
        // against a near-empty cache would over-evict, and (worse) a large batch
        // against a populated cache could still leave us over budget after insert
        // because the eviction target was wrong.
        let current = self.current_memory.load(Ordering::Relaxed) as usize;
        let deficit = current
            .saturating_add(total_size_needed)
            .saturating_sub(self.max_memory);
        if deficit > 0 {
            self.evict_lru(deficit);
        }

        // Insert all entries, accounting for replaced entries in memory tracking.
        let current_timestamp = self.access_counter.load(Ordering::Relaxed);
        for (hash_bytes, prefix_tokens, size_bytes) in entries {
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
    }

    /// Extend the cache on a *partial* hit so the next turn of a growing conversation
    /// hits deeper. Given the `(prefix_tokens, prefix_len)` returned by
    /// [`longest_prefix_match`], tokenize the remaining suffix and cache the cumulative
    /// prefix at the suffix's **deepest** special-token boundary, then return the full
    /// merged token vector.
    ///
    /// Deepest-only is intentional: in an append-only conversation the next turn always
    /// reaches the deepest boundary, so caching it bounds per-turn work to the newest
    /// exchange; shallow/branching coverage already comes from the miss path's
    /// [`insert_at_boundaries`]. Splitting at special-token boundaries is correctness-safe
    /// because special tokens are atomic in BPE
    /// (`tokenize(a) + tokenize(b) == tokenize(a + b)`).
    ///
    /// Note: unlike the read-only fast path, this **writes** to the cache on a hit
    /// (one insert + possible eviction). It relies on the same best-effort memory
    /// accounting as [`insert_at_boundaries`].
    pub fn extend_after_match<E: Encoder + ?Sized>(
        &self,
        input: &str,
        prefix_tokens: Vec<TokenIdType>,
        prefix_len: usize,
        tokenizer: &E,
        special_tokens: &[&str],
    ) -> anyhow::Result<Vec<TokenIdType>> {
        // Deepest special-token boundary strictly past the matched prefix. Strict `>`
        // avoids re-inserting the entry we just matched. `find_special_token_boundaries`
        // excludes any boundary == input.len(), so `deepest < input.len()` and the
        // trailing segment below is always non-empty.
        let deepest = find_special_token_boundaries(input, special_tokens)
            .iter()
            .rev()
            .copied()
            .find(|&b| b > prefix_len);

        let Some(deepest) = deepest else {
            // No new boundary in the suffix — nothing worth caching. Encode the suffix
            // once and merge, identical to the non-extend hit path.
            let suffix_enc = tokenizer.encode(&input[prefix_len..])?;
            let mut merged = prefix_tokens;
            merged.extend_from_slice(suffix_enc.token_ids());
            return Ok(merged);
        };

        // Cumulative tokens up to `deepest` = matched prefix + the spanning segment.
        // Both `prefix_len` and `deepest` are special-token boundaries, so encoding the
        // span as one chunk and concatenating preserves the merge invariant.
        let seg_a = tokenizer.encode(&input[prefix_len..deepest])?;
        let mut cumulative = prefix_tokens;
        cumulative.extend_from_slice(seg_a.token_ids());

        // Key is blake3 of input[0..deepest]. Built with the same streaming idiom as
        // `longest_prefix_match`/`insert_at_boundaries` so the digest is byte-for-byte
        // identical to the incremental one a future lookup computes for this prefix.
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input.as_bytes()[..deepest]);
        let hash_bytes: Blake3Hash = *hasher.finalize().as_bytes();

        let tokens: Arc<[TokenIdType]> = cumulative.as_slice().into();
        // Budget by resident token bytes only (see `insert_at_boundaries`).
        let size_bytes = tokens.len() * size_of::<TokenIdType>();
        self.insert_entries(vec![(hash_bytes, tokens, size_bytes)]);

        // Tokenize the trailing segment after `deepest` for the returned result.
        let seg_b = tokenizer.encode(&input[deepest..])?;
        let mut merged = cumulative;
        merged.extend_from_slice(seg_b.token_ids());
        Ok(merged)
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
            let input =
                format!("<s>system\nPersona {i} chatty.</s><s>user\nTurn {i} content here.</s>");
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

    /// Build an append-only multi-turn conversation. `turns[i]` is the full prompt at
    /// turn `i`: the system prompt, `i + 1` completed user/assistant exchanges, and a
    /// diverging open user turn (no trailing special, so the deepest boundary is the
    /// `<s>` that opens it). Each `turns[i]` shares a strictly longer `</s>`-bounded
    /// prefix with `turns[i + 1]`.
    fn growing_chat_turns(n: usize) -> Vec<String> {
        let mut convo = String::from("<s>system\nYou are a helpful assistant.</s>");
        let mut turns = Vec::with_capacity(n);
        for i in 0..n {
            convo.push_str(&format!(
                "<s>user\nQuestion {i} please answer it.</s><s>assistant\nDetailed answer {i} follows here.</s>"
            ));
            turns.push(format!("{convo}<s>user\nFollow-up {i}"));
        }
        turns
    }

    #[test]
    fn extend_on_hit_advances_match_depth_each_turn() {
        // The load-bearing behavioral proof. Without extension the match offset is
        // pinned at turn-1 depth (hits never insert); with extension it advances every
        // turn, so the suffix re-tokenized per turn shrinks instead of growing.
        let tok = load_tokenizer();
        let turns = growing_chat_turns(5);

        // EXTEND OFF: seed turn 0 via the miss path, then only look up (never insert).
        let off = L1Cache::new(8 * 1024 * 1024);
        off.insert_at_boundaries(&turns[0], tok.as_ref(), SPECIALS)
            .unwrap();
        let pinned = off
            .longest_prefix_match(&turns[1], SPECIALS)
            .expect("hit")
            .1;
        for t in &turns[1..] {
            let (_toks, offset) = off.longest_prefix_match(t, SPECIALS).expect("hit");
            assert_eq!(
                offset, pinned,
                "extend-off offset must stay pinned at turn-1 depth"
            );
        }

        // EXTEND ON: each hit caches the deepest boundary, so the next turn hits deeper.
        let on = L1Cache::new(8 * 1024 * 1024);
        on.insert_at_boundaries(&turns[0], tok.as_ref(), SPECIALS)
            .unwrap();
        let mut prev = 0usize;
        for (i, t) in turns.iter().enumerate().skip(1) {
            let (prefix_tokens, offset) = on.longest_prefix_match(t, SPECIALS).expect("hit");
            assert!(
                offset > prev,
                "turn {i}: extend-on offset {offset} must exceed previous {prev}"
            );
            prev = offset;

            // Extending must also preserve byte-exact correctness vs an uncached encode.
            let merged = on
                .extend_after_match(t, prefix_tokens, offset, tok.as_ref(), SPECIALS)
                .unwrap();
            let plain = tok.encode(t).unwrap();
            assert_eq!(
                merged,
                plain.token_ids(),
                "turn {i}: extend merge must equal plain encode"
            );
        }

        assert!(
            prev > pinned,
            "extend-on frontier ({prev}) must reach deeper than pinned extend-off depth ({pinned})"
        );
    }

    #[test]
    fn extend_on_hit_respects_budget_and_stays_correct() {
        // Tiny budget forces eviction (and over-budget skips) while extending; every
        // turn's encode must stay correct and memory must stay within budget.
        let tok = load_tokenizer();
        let cache = L1Cache::new(4 * 1024);
        let turns = growing_chat_turns(20);
        cache
            .insert_at_boundaries(&turns[0], tok.as_ref(), SPECIALS)
            .unwrap();

        for t in &turns[1..] {
            let merged = match cache.longest_prefix_match(t, SPECIALS) {
                Some((prefix_tokens, offset)) => cache
                    .extend_after_match(t, prefix_tokens, offset, tok.as_ref(), SPECIALS)
                    .unwrap(),
                None => {
                    // Full miss under eviction pressure — mirror the miss path.
                    let enc = tok.encode(t).unwrap();
                    cache
                        .insert_at_boundaries(t, tok.as_ref(), SPECIALS)
                        .unwrap();
                    enc.token_ids().to_vec()
                }
            };
            let plain = tok.encode(t).unwrap();
            assert_eq!(
                merged,
                plain.token_ids(),
                "encode must stay correct under eviction pressure"
            );
            assert!(
                cache.stats().memory_bytes <= 4 * 1024,
                "memory_bytes={} exceeds budget",
                cache.stats().memory_bytes
            );
        }
    }

    #[test]
    fn concurrent_extend_on_hit_does_not_corrupt() {
        use std::thread;

        let tok = load_tokenizer();
        let cache = Arc::new(L1Cache::new(8 * 1024 * 1024));
        let turns = growing_chat_turns(8);
        // Seed turn 0 so every thread gets at least a partial hit.
        cache
            .insert_at_boundaries(&turns[0], tok.as_ref(), SPECIALS)
            .unwrap();

        let mut handles = vec![];
        for _ in 0..8 {
            let cache_c = cache.clone();
            let tok_c = tok.clone();
            let turns_c = turns.clone();
            handles.push(thread::spawn(move || {
                for t in &turns_c[1..] {
                    if let Some((prefix_tokens, offset)) = cache_c.longest_prefix_match(t, SPECIALS)
                    {
                        let merged = cache_c
                            .extend_after_match(t, prefix_tokens, offset, tok_c.as_ref(), SPECIALS)
                            .unwrap();
                        let plain = tok_c.encode(t).unwrap();
                        assert_eq!(
                            merged,
                            plain.token_ids(),
                            "concurrent extend must stay correct"
                        );
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.stats().memory_bytes > 0);
    }

    #[test]
    fn extend_after_match_persists_correct_deepest_entry() {
        // The *saved* entry on a partial hit — not just the returned merge — must be
        // byte-exact and retrievable: a fresh lookup hits at the just-cached deepest
        // boundary and returns exactly `encode(input[0..deepest])`, so the next turn
        // reuses a correct prefix. Also proves the deepest-only invariant: extend
        // persists exactly one new entry.
        let tok = load_tokenizer();
        let turns = growing_chat_turns(3);

        let cache = L1Cache::new(8 * 1024 * 1024);
        cache
            .insert_at_boundaries(&turns[0], tok.as_ref(), SPECIALS)
            .unwrap();

        let (prefix_tokens, prefix_len) = cache
            .longest_prefix_match(&turns[1], SPECIALS)
            .expect("partial hit on turns[1]");
        let entries_before = cache.stats().entries;

        let _merged = cache
            .extend_after_match(&turns[1], prefix_tokens, prefix_len, tok.as_ref(), SPECIALS)
            .unwrap();

        assert_eq!(
            cache.stats().entries,
            entries_before + 1,
            "extend must persist exactly one (deepest) entry"
        );

        // The deepest boundary strictly past the matched prefix is what extend cached.
        let deepest = find_special_token_boundaries(&turns[1], SPECIALS)
            .into_iter()
            .rev()
            .find(|&b| b > prefix_len)
            .expect("a deeper boundary must exist in the appended turn");

        // A fresh lookup must now hit AT that deepest boundary, and the stored tokens must
        // equal the uncached encode of exactly that prefix.
        let (saved_tokens, saved_offset) = cache
            .longest_prefix_match(&turns[1], SPECIALS)
            .expect("hit after extend");
        assert_eq!(
            saved_offset, deepest,
            "lookup must now hit at the just-saved deepest boundary"
        );
        let expected = tok.encode(&turns[1][..deepest]).unwrap();
        assert_eq!(
            saved_tokens,
            expected.token_ids(),
            "persisted entry tokens must equal the uncached encode of the cached prefix"
        );
    }

    #[test]
    fn boundaries_detected_for_multibyte_deepseek_tool_tokens() {
        // `find_special_token_boundaries` keys off byte offsets; DeepSeek's tool tokens use
        // multibyte code points (｜ = U+FF5C, ▁ = U+2581, 3 bytes each). A boundary must
        // land immediately after each occurrence at a valid char boundary, so the cache can
        // split a tool-call block at its special tokens without panicking on a slice.
        let specials = &["<｜tool▁calls▁begin｜>", "<｜tool▁call▁end｜>"];
        let text = "<｜tool▁calls▁begin｜>payload<｜tool▁call▁end｜>tail";
        let bounds = find_special_token_boundaries(text, specials);

        let after_begin = "<｜tool▁calls▁begin｜>".len();
        let after_end = text.find("<｜tool▁call▁end｜>").unwrap() + "<｜tool▁call▁end｜>".len();
        assert_eq!(bounds, vec![after_begin, after_end]);
        for &b in &bounds {
            assert!(
                text.is_char_boundary(b),
                "boundary {b} is not a char boundary"
            );
            let _ = &text[..b]; // must not panic
        }
    }
}
