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
//!
//! Storage and eviction are delegated to a weighted [`moka`] `sync::Cache` (W-TinyLFU):
//! entries are keyed by the blake3 digest of `input[0..boundary]` and weighed by their
//! resident token-vector bytes, so the byte budget is enforced — and recency/frequency
//! tracked — by moka rather than by hand.

use std::{
    hash::BuildHasherDefault,
    mem::size_of_val,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use aho_corasick::AhoCorasick;
use moka::sync::Cache;
use rustc_hash::FxHasher;

use crate::{TokenIdType, traits::Encoder};

/// Hash type for cache keys
type Blake3Hash = [u8; 32];

/// Keys are blake3 digests (already uniformly distributed), so a fast non-DoS-resistant
/// hasher suffices — no need for the default SipHash.
type PrefixHasher = BuildHasherDefault<FxHasher>;

/// Weighted W-TinyLFU cache mapping a prefix's blake3 digest to its cumulative tokens.
type PrefixCache = Cache<Blake3Hash, Arc<[TokenIdType]>, PrefixHasher>;

/// All special-token boundaries in `text`: positions immediately after each special-token
/// occurrence (where prefixes can be cached).
///
/// **ONLY uses special tokens** — these are atomic (`special: true, normalized: false`) in
/// BPE, so a boundary right after one is a safe split point:
/// `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`. A single overlapping
/// Aho-Corasick pass reports every occurrence of every pattern (matching the per-token scan
/// it replaces, for the non-self-overlapping special tokens real tokenizers use). Boundaries
/// at the very end of the text are dropped (no suffix left to tokenize). Match ends land on
/// char boundaries because the patterns are valid UTF-8 matched against valid UTF-8.
fn boundaries_with(text: &str, matcher: &AhoCorasick) -> Vec<usize> {
    let mut boundaries: Vec<usize> = matcher
        .find_overlapping_iter(text)
        .map(|m| m.end())
        .filter(|&end| end < text.len())
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// Test-only reference: build a one-off automaton and find boundaries. Production goes
/// through [`L1Cache::boundaries`], which reuses a process-once automaton.
#[cfg(test)]
fn find_special_token_boundaries(text: &str, special_tokens: &[&str]) -> Vec<usize> {
    if special_tokens.is_empty() {
        return Vec::new();
    }
    let matcher = AhoCorasick::new(special_tokens)
        .expect("special tokens form a valid Aho-Corasick automaton");
    boundaries_with(text, &matcher)
}

/// Optional per-event observer. `on_hit` runs after each cache hit, `on_miss`
/// after each miss — wired by `CachedTokenizer::with_observer` to push events
/// straight into Prometheus counters without a periodic sampling step.
pub type CacheEventFn = Arc<dyn Fn() + Send + Sync>;

/// L1 cache: prefix matching at special-token boundaries, backed by a weighted W-TinyLFU
/// [`moka`] cache that owns storage, recency/frequency tracking, and eviction. Hit/miss
/// counts (our notion of a *prefix* hit) are tracked separately for metrics.
pub struct L1Cache {
    /// Prefix entries keyed by the blake3 digest of `input[0..boundary]`.
    cache: PrefixCache,
    /// Aho-Corasick automaton over the special tokens, built once at construction (`None`
    /// when there are no special tokens). Lets boundary detection be a single pass.
    matcher: Option<AhoCorasick>,
    hits: AtomicU64,
    misses: AtomicU64,
    on_hit: Option<CacheEventFn>,
    on_miss: Option<CacheEventFn>,
}

impl L1Cache {
    /// `special_tokens` is the atomic special-token set whose boundaries the cache splits
    /// at; an empty set leaves L1 inert (no boundaries, no entries).
    pub fn new(max_memory: usize, special_tokens: Vec<String>) -> Self {
        // Capacity is the byte budget; each entry weighs its resident token-vector bytes
        // (the prefix text is hashed and discarded, never stored). moka's W-TinyLFU policy
        // admits/evicts to keep the weighted size within budget.
        let cache = Cache::builder()
            .max_capacity(max_memory as u64)
            .weigher(|_k: &Blake3Hash, tokens: &Arc<[TokenIdType]>| -> u32 {
                size_of_val(tokens.as_ref()).min(u32::MAX as usize) as u32
            })
            .build_with_hasher(PrefixHasher::default());

        // Build the boundary automaton once; `None` when there are no special tokens.
        let matcher = (!special_tokens.is_empty()).then(|| {
            AhoCorasick::new(&special_tokens)
                .expect("special tokens form a valid Aho-Corasick automaton")
        });

        Self {
            cache,
            matcher,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            on_hit: None,
            on_miss: None,
        }
    }

    /// Install hit/miss callbacks. Replaces any previously-set observers.
    pub fn set_observer(&mut self, on_hit: CacheEventFn, on_miss: CacheEventFn) {
        self.on_hit = Some(on_hit);
        self.on_miss = Some(on_miss);
    }

    /// Special-token boundaries in `text` via the process-once Aho-Corasick automaton built
    /// at construction — a single pass over the input rather than one `str::find` sweep per
    /// token. Empty when the cache has no special tokens.
    fn boundaries(&self, text: &str) -> Vec<usize> {
        match &self.matcher {
            Some(matcher) => boundaries_with(text, matcher),
            None => Vec::new(),
        }
    }

    /// Try to find the longest prefix match at a special-token boundary.
    ///
    /// Returns `(cached_tokens, byte_offset, deepest_boundary)` if found. The caller
    /// extends the cached tokens with a fresh encode of `input[byte_offset..]`;
    /// `deepest_boundary` is the deepest special-token boundary in `input` (end-exclusive),
    /// handed back so [`extend_after_match`] need not rescan the input for it.
    pub fn longest_prefix_match(&self, input: &str) -> Option<(Arc<[TokenIdType]>, usize, usize)> {
        let boundaries = self.boundaries(input);

        if boundaries.is_empty() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            if let Some(cb) = &self.on_miss {
                cb();
            }
            return None;
        }

        // Deepest boundary in the input — returned on a hit so the extend path can split
        // there without recomputing `find_special_token_boundaries`.
        let deepest_boundary = *boundaries.last().expect("boundaries is non-empty here");

        // Build all prefix hashes incrementally — O(N).
        let mut hasher = blake3::Hasher::new();
        let mut prefix_hashes = Vec::with_capacity(boundaries.len());
        let mut last_pos = 0;
        let bytes = input.as_bytes();
        for &boundary_pos in &boundaries {
            hasher.update(&bytes[last_pos..boundary_pos]);
            // `finalize(&self)` borrows — no need to clone the hasher to keep updating it.
            prefix_hashes.push((boundary_pos, *hasher.finalize().as_bytes()));
            last_pos = boundary_pos;
        }

        // Search from the longest boundary down — return first hit. moka updates recency
        // and frequency on `get`, so no manual timestamp bookkeeping is needed.
        for (boundary_pos, hash_bytes) in prefix_hashes.into_iter().rev() {
            if let Some(tokens) = self.cache.get(&hash_bytes) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                if let Some(cb) = &self.on_hit {
                    cb();
                }
                // Return the shared `Arc` directly — the caller decides whether to
                // materialize a `Vec` (and reserves exact capacity when it does),
                // avoiding a clone of the (large) cached prefix on every hit.
                return Some((tokens, boundary_pos, deepest_boundary));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(cb) = &self.on_miss {
            cb();
        }
        None
    }

    /// Insert prefix entries at every special-token boundary (e.g. to pre-seed the cache).
    ///
    /// Uses incremental hashing and incremental tokenization (per-segment encode of the
    /// delta text between adjacent boundaries) so populating N entries costs one full
    /// re-tokenize total, split across the segments. The miss path uses
    /// [`Self::populate_and_encode`] instead, which reuses this same work to *also* return
    /// the full token vector (avoiding a redundant second tokenization).
    pub fn insert_at_boundaries<E: Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
    ) -> anyhow::Result<()> {
        let boundaries = self.boundaries(input);
        if boundaries.is_empty() {
            return Ok(());
        }
        self.populate_boundaries(input, &boundaries, tokenizer)?;
        Ok(())
    }

    /// Miss-path encode: tokenize `input` exactly once, caching the cumulative prefix at
    /// every special-token boundary as we go, and return the full token-id vector. This
    /// replaces a separate full `encode` + [`Self::insert_at_boundaries`], which together
    /// tokenized the input ~twice (once for the result, once split across segments).
    ///
    /// The concatenation of the per-segment encodes equals an uncached `encode(input)`
    /// because special tokens are atomic in BPE — the same invariant the hit path relies
    /// on. Returns token-ids only; the caller wraps them in [`crate::Encoding::Sp`].
    pub fn populate_and_encode<E: Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
    ) -> anyhow::Result<Vec<TokenIdType>> {
        let boundaries = self.boundaries(input);
        if boundaries.is_empty() {
            // No special tokens present — nothing cacheable; a single plain encode.
            return Ok(tokenizer.encode(input)?.token_ids().to_vec());
        }

        // Tokenize + cache every boundary prefix; `running` covers input[0..last boundary].
        let mut running = self.populate_boundaries(input, &boundaries, tokenizer)?;

        // The trailing segment after the last boundary is not a cache key (boundaries
        // exclude input.len()); encoding it completes the full tokenization.
        let tail_start = *boundaries.last().expect("boundaries is non-empty here");
        let tail = tokenizer.encode(&input[tail_start..])?;
        running.extend_from_slice(tail.token_ids());
        Ok(running)
    }

    /// Shared core of the miss path: walk `boundaries`, hashing and tokenizing each
    /// inter-boundary segment, caching the cumulative prefix at each boundary, and return
    /// the running token vector (covering `input[0..boundaries.last()]`).
    fn populate_boundaries<E: Encoder + ?Sized>(
        &self,
        input: &str,
        boundaries: &[usize],
        tokenizer: &E,
    ) -> anyhow::Result<Vec<TokenIdType>> {
        let mut hasher = blake3::Hasher::new();
        let mut running_tokens: Vec<TokenIdType> = Vec::new();
        let mut last_pos = 0;
        let bytes = input.as_bytes();

        for &boundary_pos in boundaries {
            // 1. Incremental hash. `finalize(&self)` borrows, so no clone is needed.
            hasher.update(&bytes[last_pos..boundary_pos]);
            let hash_bytes: Blake3Hash = *hasher.finalize().as_bytes();

            // 2. Incremental tokenization. Dynamo's Encoder has no `add_special_tokens`
            //    parameter — equivalent to upstream always passing `false` past the first
            //    segment (which is also what Dynamo's HF impl always does for the first).
            let seg = tokenizer.encode(&input[last_pos..boundary_pos])?;
            running_tokens.extend_from_slice(seg.token_ids());

            // 3. Snapshot the cumulative prefix as Arc<[T]> and hand it to moka (the weigher
            //    charges its token bytes against the budget; eviction is moka's job).
            let prefix_tokens: Arc<[TokenIdType]> = running_tokens.as_slice().into();
            self.cache.insert(hash_bytes, prefix_tokens);

            last_pos = boundary_pos;
        }

        Ok(running_tokens)
    }

    /// Extend the cache on a *partial* hit so the next turn of a growing conversation
    /// hits deeper. Given the `(prefix_tokens, prefix_len, deepest_boundary)` returned by
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
        prefix_tokens: Arc<[TokenIdType]>,
        prefix_len: usize,
        deepest_boundary: usize,
        tokenizer: &E,
    ) -> anyhow::Result<Vec<TokenIdType>> {
        // `deepest_boundary` (from `longest_prefix_match`) is the deepest special-token
        // boundary in `input`; split there only if it lies strictly past the matched
        // prefix. Strict `>` avoids re-inserting the entry we just matched. Boundaries
        // exclude any position == input.len(), so `deepest < input.len()` and the trailing
        // segment below is always non-empty.
        let deepest = (deepest_boundary > prefix_len).then_some(deepest_boundary);

        let Some(deepest) = deepest else {
            // No new boundary in the suffix — nothing worth caching. Encode the suffix
            // once and merge, identical to the non-extend hit path. Reserve exact capacity
            // so the prefix isn't re-copied by a Vec grow-realloc.
            let suffix_enc = tokenizer.encode(&input[prefix_len..])?;
            let mut merged = Vec::with_capacity(prefix_tokens.len() + suffix_enc.token_ids().len());
            merged.extend_from_slice(&prefix_tokens);
            merged.extend_from_slice(suffix_enc.token_ids());
            return Ok(merged);
        };

        // Cumulative tokens up to `deepest` = matched prefix + the spanning segment.
        // Both `prefix_len` and `deepest` are special-token boundaries, so encoding the
        // span as one chunk and concatenating preserves the merge invariant.
        // Encode both segments up front so `cumulative` can be reserved to its final
        // size (prefix + seg_a + seg_b) — this eliminates the two grow-reallocs (each of
        // which re-copied the whole large prefix) the previous Vec-append path incurred.
        let seg_a = tokenizer.encode(&input[prefix_len..deepest])?;
        let seg_b = tokenizer.encode(&input[deepest..])?;
        let mut cumulative = Vec::with_capacity(
            prefix_tokens.len() + seg_a.token_ids().len() + seg_b.token_ids().len(),
        );
        cumulative.extend_from_slice(&prefix_tokens);
        cumulative.extend_from_slice(seg_a.token_ids());

        // Key is blake3 of input[0..deepest]. Built with the same streaming idiom as
        // `longest_prefix_match`/`insert_at_boundaries` so the digest is byte-for-byte
        // identical to the incremental one a future lookup computes for this prefix.
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input.as_bytes()[..deepest]);
        let hash_bytes: Blake3Hash = *hasher.finalize().as_bytes();

        // Snapshot prefix+seg_a (`as_slice().into()` copies only the populated len, not the
        // reserved capacity) and cache it.
        let tokens: Arc<[TokenIdType]> = cumulative.as_slice().into();
        self.cache.insert(hash_bytes, tokens);

        // Append the trailing segment for the returned result — no realloc, capacity was
        // reserved above.
        cumulative.extend_from_slice(seg_b.token_ids());
        Ok(cumulative)
    }

    /// Number of live entries. Flushes moka's deferred maintenance first so the count is
    /// exact rather than lagging behind pending inserts/evictions.
    pub fn len(&self) -> usize {
        self.cache.run_pending_tasks();
        self.cache.entry_count() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> L1CacheStats {
        // Flush moka's deferred maintenance so entry_count / weighted_size are accurate.
        self.cache.run_pending_tasks();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        L1CacheStats {
            hits,
            misses,
            entries: self.cache.entry_count() as usize,
            memory_bytes: self.cache.weighted_size() as usize,
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    pub fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks();
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

    /// An `L1Cache` over the TinyLlama [`SPECIALS`] with the given byte budget.
    fn test_cache(max_memory: usize) -> L1Cache {
        L1Cache::new(
            max_memory,
            SPECIALS.iter().map(|s| (*s).to_string()).collect(),
        )
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
        let cache = test_cache(1024 * 1024);
        let tokenizer = load_tokenizer();

        let warm = "<s>system\nYou are helpful.</s><s>user\nHi</s>";
        cache
            .insert_at_boundaries(warm, tokenizer.as_ref())
            .unwrap();
        assert!(!cache.is_empty());

        let target = "<s>system\nYou are helpful.</s><s>user\nDifferent question</s>";
        let (tokens, offset, _deepest) = cache
            .longest_prefix_match(target)
            .expect("shared prefix should match");
        assert!(offset > 0);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn miss_increments_misses_counter() {
        let cache = test_cache(1024 * 1024);
        assert!(
            cache
                .longest_prefix_match("plain text no specials")
                .is_none()
        );
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn hit_increments_hits_counter() {
        let cache = test_cache(1024 * 1024);
        let tokenizer = load_tokenizer();
        let warm = "<s>system\nA.</s><s>user\nB</s>";
        cache
            .insert_at_boundaries(warm, tokenizer.as_ref())
            .unwrap();
        let _ = cache.longest_prefix_match(warm);
        assert!(cache.stats().hits >= 1);
    }

    #[test]
    fn merge_invariant_holds_against_uncached_encode() {
        // Load-bearing correctness check: cached prefix + fresh suffix encode must
        // equal plain encode of the full input. Relies on `<s>`/`</s>` being atomic
        // in TinyLlama's BPE (they are).
        let cache = test_cache(1024 * 1024);
        let tokenizer = load_tokenizer();

        let template = "<s>system\nYou are helpful.</s><s>user\n";
        let warm = format!("{template}First.</s>");
        cache
            .insert_at_boundaries(&warm, tokenizer.as_ref())
            .unwrap();

        let target = format!("{template}A completely different second question.</s>");
        let (prefix_tokens, prefix_len, _deepest) = cache
            .longest_prefix_match(&target)
            .expect("should find prefix");

        let suffix = &target[prefix_len..];
        let suffix_enc = tokenizer.encode(suffix).unwrap();
        // longest_prefix_match returns the shared `Arc<[u32]>`; copy into a Vec to append the suffix.
        let mut merged = prefix_tokens.to_vec();
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
        let cache = test_cache(4 * 1024);
        let tokenizer = load_tokenizer();
        for i in 0..50 {
            let input =
                format!("<s>system\nPersona {i} chatty.</s><s>user\nTurn {i} content here.</s>");
            cache
                .insert_at_boundaries(&input, tokenizer.as_ref())
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

        let cache = Arc::new(test_cache(1024 * 1024));
        let tokenizer = load_tokenizer();

        let mut handles = vec![];
        for i in 0..10 {
            let cache_c = cache.clone();
            let tok = tokenizer.clone();
            handles.push(thread::spawn(move || {
                let input = format!("<s>system\nThread {i}.</s><s>user\nThread {i} body.</s>");
                cache_c.insert_at_boundaries(&input, tok.as_ref()).unwrap();
                let r = cache_c.longest_prefix_match(&input);
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
        let off = test_cache(8 * 1024 * 1024);
        off.insert_at_boundaries(&turns[0], tok.as_ref()).unwrap();
        let pinned = off.longest_prefix_match(&turns[1]).expect("hit").1;
        for t in &turns[1..] {
            let (_toks, offset, _deepest) = off.longest_prefix_match(t).expect("hit");
            assert_eq!(
                offset, pinned,
                "extend-off offset must stay pinned at turn-1 depth"
            );
        }

        // EXTEND ON: each hit caches the deepest boundary, so the next turn hits deeper.
        let on = test_cache(8 * 1024 * 1024);
        on.insert_at_boundaries(&turns[0], tok.as_ref()).unwrap();
        let mut prev = 0usize;
        for (i, t) in turns.iter().enumerate().skip(1) {
            let (prefix_tokens, offset, deepest) = on.longest_prefix_match(t).expect("hit");
            assert!(
                offset > prev,
                "turn {i}: extend-on offset {offset} must exceed previous {prev}"
            );
            prev = offset;

            // Extending must also preserve byte-exact correctness vs an uncached encode.
            let merged = on
                .extend_after_match(t, prefix_tokens, offset, deepest, tok.as_ref())
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
        let cache = test_cache(4 * 1024);
        let turns = growing_chat_turns(20);
        cache.insert_at_boundaries(&turns[0], tok.as_ref()).unwrap();

        for t in &turns[1..] {
            let merged = match cache.longest_prefix_match(t) {
                Some((prefix_tokens, offset, deepest)) => cache
                    .extend_after_match(t, prefix_tokens, offset, deepest, tok.as_ref())
                    .unwrap(),
                None => {
                    // Full miss under eviction pressure — mirror the miss path.
                    let enc = tok.encode(t).unwrap();
                    cache.insert_at_boundaries(t, tok.as_ref()).unwrap();
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
        let cache = Arc::new(test_cache(8 * 1024 * 1024));
        let turns = growing_chat_turns(8);
        // Seed turn 0 so every thread gets at least a partial hit.
        cache.insert_at_boundaries(&turns[0], tok.as_ref()).unwrap();

        let mut handles = vec![];
        for _ in 0..8 {
            let cache_c = cache.clone();
            let tok_c = tok.clone();
            let turns_c = turns.clone();
            handles.push(thread::spawn(move || {
                for t in &turns_c[1..] {
                    if let Some((prefix_tokens, offset, deepest)) = cache_c.longest_prefix_match(t)
                    {
                        let merged = cache_c
                            .extend_after_match(t, prefix_tokens, offset, deepest, tok_c.as_ref())
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

        let cache = test_cache(8 * 1024 * 1024);
        cache.insert_at_boundaries(&turns[0], tok.as_ref()).unwrap();

        let (prefix_tokens, prefix_len, deepest_boundary) = cache
            .longest_prefix_match(&turns[1])
            .expect("partial hit on turns[1]");
        let entries_before = cache.stats().entries;

        let _merged = cache
            .extend_after_match(
                &turns[1],
                prefix_tokens,
                prefix_len,
                deepest_boundary,
                tok.as_ref(),
            )
            .unwrap();

        assert_eq!(
            cache.stats().entries,
            entries_before + 1,
            "extend must persist exactly one (deepest) entry"
        );

        // The deepest boundary strictly past the matched prefix is what extend cached, and
        // `longest_prefix_match` must have handed back exactly that boundary (no rescan).
        let deepest = find_special_token_boundaries(&turns[1], SPECIALS)
            .into_iter()
            .rev()
            .find(|&b| b > prefix_len)
            .expect("a deeper boundary must exist in the appended turn");
        assert_eq!(
            deepest_boundary, deepest,
            "longest_prefix_match must return the deepest boundary used by extend"
        );

        // A fresh lookup must now hit AT that deepest boundary, and the stored tokens must
        // equal the uncached encode of exactly that prefix.
        let (saved_tokens, saved_offset, _deepest) = cache
            .longest_prefix_match(&turns[1])
            .expect("hit after extend");
        assert_eq!(
            saved_offset, deepest,
            "lookup must now hit at the just-saved deepest boundary"
        );
        let expected = tok.encode(&turns[1][..deepest]).unwrap();
        assert_eq!(
            &*saved_tokens,
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

    #[test]
    fn populate_and_encode_matches_uncached_and_seeds_cache() {
        // The fused miss path must (a) return ids byte-exact to an uncached encode and
        // (b) leave the cache populated at the boundaries, so a follow-up lookup hits.
        let tok = load_tokenizer();
        let cache = test_cache(8 * 1024 * 1024);
        let input = "<s>system\nYou are helpful.</s><s>user\nHello there, friend.</s>";

        let got = cache.populate_and_encode(input, tok.as_ref()).unwrap();
        let plain = tok.encode(input).unwrap();
        assert_eq!(
            got,
            plain.token_ids(),
            "fused miss encode must equal uncached encode"
        );

        // It also seeded the cache: a follow-up lookup hits at a boundary.
        assert!(
            !cache.is_empty(),
            "miss path must populate boundary entries"
        );
        let (_t, offset, _d) = cache
            .longest_prefix_match(input)
            .expect("hit after populate");
        assert!(offset > 0, "follow-up lookup should hit a cached boundary");
    }

    #[test]
    fn populate_and_encode_handles_inputs_without_special_tokens() {
        // No registered special appears in the input → no boundaries → one plain encode,
        // nothing cached, still byte-exact.
        let tok = load_tokenizer();
        let cache = test_cache(8 * 1024 * 1024);
        let input = "plain text with no special tokens at all";

        let got = cache.populate_and_encode(input, tok.as_ref()).unwrap();
        let plain = tok.encode(input).unwrap();
        assert_eq!(got, plain.token_ids());
        assert!(cache.is_empty(), "nothing cacheable without boundaries");
    }

    #[test]
    fn populate_and_encode_handles_trailing_special_token() {
        // Input ending in a special token: the final boundary == input.len() is excluded,
        // so the trailing `</s>` lands in the tail segment. The assembled ids must still
        // equal an uncached encode.
        let tok = load_tokenizer();
        let cache = test_cache(8 * 1024 * 1024);
        let input = "<s>system\nDone.</s>";

        let got = cache.populate_and_encode(input, tok.as_ref()).unwrap();
        let plain = tok.encode(input).unwrap();
        assert_eq!(
            got,
            plain.token_ids(),
            "tail-segment assembly must be exact"
        );
    }
}
