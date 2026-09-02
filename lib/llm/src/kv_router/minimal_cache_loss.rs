// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal request-level cache-loss accounting.
//!
//! The router already keeps the per-worker cache index used for routing. This
//! module deliberately adds no second cache-event stream. It does retain a
//! bounded, process-local history of *completed* canonical sequence hashes so
//! the funnel can distinguish "computed before" from "still resident now".

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Instant,
};

use parking_lot::Mutex;

use dynamo_kv_router::protocols::{BlockExtraInfo, TokensWithHashes};

pub const HISTORY_BYTES_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BYTES";
pub const CACHE_LOSS_FUNNEL_ENABLED_ENV: &str = "DYN_CACHE_LOSS_FUNNEL_ENABLED";
pub const DEFAULT_HISTORY_BYTES: usize = 256 * 1024 * 1024;
pub const HISTORY_CHUNK_RECORDS: usize = 4_096;

/// Conservative planning estimate: an 8-byte FIFO sequence hash plus the
/// amortized hash-map key, refcount, bucket slack, and allocator overhead. The
/// amortized chunk timestamp is negligible at the fixed chunk size.
/// This is deliberately larger than `size_of::<u64>()`; it is a capacity model,
/// not a promise about a particular Rust allocator build.
pub const ESTIMATED_BYTES_PER_HISTORY_RECORD: usize = 32;

/// Whether cache-loss accounting is enabled for this frontend process.
///
/// This is deliberately evaluated once during frontend construction. When it
/// is false, callers must not allocate the history ledger, copy prompt tokens,
/// hash blocks, or attach worker outcome metadata.
pub fn enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(CACHE_LOSS_FUNNEL_ENABLED_ENV)
}

/// A bounded history of cache identities that have definitely been computed.
///
/// The identity is the canonical rolling `SequenceHash`, not a bare token-block
/// hash. Thus equal token blocks under different preceding contexts remain
/// distinct, matching Dynamo's routing identity. Each record is retained in
/// arrival order; a refcount keeps a hash present while any retained record
/// still refers to it.
#[derive(Debug)]
pub struct CacheHistory {
    capacity_blocks: usize,
    capacity_bytes: usize,
    block_tokens: u64,
    chunk_records: usize,
    retained_records: usize,
    chunks: VecDeque<HistoryChunk>,
    retained: HashMap<u64, u32>,
}

#[derive(Debug)]
struct HistoryChunk {
    inserted_at: Instant,
    hashes: Vec<u64>,
}

impl CacheHistory {
    pub fn from_env(block_tokens: u32) -> Arc<Mutex<Self>> {
        let requested_bytes = std::env::var(HISTORY_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_HISTORY_BYTES);
        let capacity_bytes = requested_bytes.max(ESTIMATED_BYTES_PER_HISTORY_RECORD);
        Arc::new(Mutex::new(Self::new_with_budget(
            block_tokens,
            capacity_bytes,
        )))
    }

    #[cfg(test)]
    pub fn new(capacity_blocks: usize, block_tokens: u32) -> Self {
        Self::new_with_budget(
            block_tokens,
            capacity_blocks.saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
        )
    }

    #[cfg(test)]
    fn new_with_chunk(capacity_blocks: usize, block_tokens: u32, chunk_records: usize) -> Self {
        Self::new_with_chunk_budget(
            block_tokens,
            capacity_blocks.saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
            chunk_records,
        )
    }

    fn new_with_budget(block_tokens: u32, capacity_bytes: usize) -> Self {
        let record_budget = capacity_bytes / ESTIMATED_BYTES_PER_HISTORY_RECORD;
        Self::new_with_chunk_budget(
            block_tokens,
            capacity_bytes,
            record_budget.min(HISTORY_CHUNK_RECORDS),
        )
    }

    fn new_with_chunk_budget(
        block_tokens: u32,
        capacity_bytes: usize,
        chunk_records: usize,
    ) -> Self {
        let record_budget = capacity_bytes / ESTIMATED_BYTES_PER_HISTORY_RECORD;
        let chunk_records = chunk_records.max(1).min(record_budget.max(1));
        let capacity_blocks = (record_budget / chunk_records).max(1) * chunk_records;
        assert!(
            capacity_blocks > 0,
            "cache history capacity must be positive"
        );
        assert!(
            block_tokens > 0,
            "cache history block size must be positive"
        );
        Self {
            capacity_blocks,
            capacity_bytes,
            block_tokens: u64::from(block_tokens),
            chunk_records,
            retained_records: 0,
            chunks: VecDeque::new(),
            retained: HashMap::new(),
        }
    }

    /// Count the longest complete prefix whose canonical identities have been
    /// computed by a prior completed request within this process lifetime.
    pub fn previously_computed_tokens(&self, sequence_hashes: &[u64]) -> u64 {
        let blocks = sequence_hashes
            .iter()
            .take_while(|hash| self.retained.contains_key(hash))
            .count() as u64;
        blocks.saturating_mul(self.block_tokens)
    }

    /// Retain the supplied complete canonical sequence hashes. Duplicate hashes
    /// are records too: refreshing a repeated prefix keeps it in the recent
    /// window without losing an older retained occurrence prematurely.
    pub fn record_completed(&mut self, sequence_hashes: impl IntoIterator<Item = u64>) {
        for hash in sequence_hashes {
            if self.retained_records == self.capacity_blocks {
                self.evict_oldest_chunk();
            }
            if self
                .chunks
                .back()
                .is_none_or(|chunk| chunk.hashes.len() == self.chunk_records)
            {
                self.chunks.push_back(HistoryChunk {
                    inserted_at: Instant::now(),
                    hashes: Vec::with_capacity(self.chunk_records),
                });
            }
            self.chunks
                .back_mut()
                .expect("history chunk was just created")
                .hashes
                .push(hash);
            self.retained_records += 1;
            *self.retained.entry(hash).or_default() += 1;
        }
    }

    fn evict_oldest_chunk(&mut self) {
        let chunk = self.chunks.pop_front().expect("history was non-empty");
        self.retained_records -= chunk.hashes.len();
        for evicted in chunk.hashes {
            let count = self
                .retained
                .get_mut(&evicted)
                .expect("history refcount missing");
            *count -= 1;
            if *count == 0 {
                self.retained.remove(&evicted);
            }
        }
    }

    pub fn stats(&self) -> CacheHistoryStats {
        CacheHistoryStats {
            capacity_blocks: self.capacity_blocks,
            capacity_bytes: self.capacity_bytes,
            retained_records: self.retained_records,
            retained_unique_hashes: self.retained.len(),
            represented_tokens: (self.retained_records as u64).saturating_mul(self.block_tokens),
            estimated_retained_bytes: self
                .retained_records
                .saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
            oldest_chunk_age_seconds: self
                .chunks
                .front()
                .map(|chunk| chunk.inserted_at.elapsed().as_secs())
                .unwrap_or(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheHistoryStats {
    pub capacity_blocks: usize,
    pub capacity_bytes: usize,
    pub retained_records: usize,
    pub retained_unique_hashes: usize,
    pub represented_tokens: u64,
    pub estimated_retained_bytes: usize,
    pub oldest_chunk_age_seconds: u64,
}

/// Request-local state used to record canonical prompt and generated histories
/// only after the worker has supplied a complete cache-loss outcome.
///
/// Generated tokens are kept by output-choice index. At finalization the newest
/// sampled token is excluded: it was returned to the caller but has not yet
/// been fed back through the model, so it has no corresponding KV entry.
pub struct CacheHistoryRequest {
    prompt_tokens: Vec<u32>,
    prompt_hashes: Vec<u64>,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    block_size: u32,
    is_eagle: bool,
    output_branches: HashMap<u32, Vec<u32>>,
    prompt_recorded: bool,
    finalized: bool,
}

impl CacheHistoryRequest {
    pub fn new(
        prompt_tokens: Vec<u32>,
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        block_size: u32,
        is_eagle: bool,
    ) -> Self {
        let mut tokens_with_hashes =
            TokensWithHashes::new(prompt_tokens.clone(), block_size).with_is_eagle(is_eagle);
        if let Some(infos) = &block_mm_infos {
            tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.clone());
        }
        if let Some(lora_name) = &lora_name {
            tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name.clone());
        }
        if let Some(cache_namespace) = &cache_namespace {
            tokens_with_hashes = tokens_with_hashes.with_cache_namespace(cache_namespace.clone());
        }
        Self {
            prompt_tokens,
            prompt_hashes: tokens_with_hashes.get_or_compute_seq_hashes().to_vec(),
            block_mm_infos,
            lora_name,
            cache_namespace,
            block_size,
            is_eagle,
            output_branches: HashMap::new(),
            prompt_recorded: false,
            finalized: false,
        }
    }

    pub fn observe_output(&mut self, output_index: u32, token_ids: &[u32]) {
        if !token_ids.is_empty() {
            self.output_branches
                .entry(output_index)
                .or_default()
                .extend_from_slice(token_ids);
        }
    }

    pub fn prompt_hashes(&self) -> Vec<u64> {
        self.prompt_hashes.clone()
    }

    pub fn output_hashes(&self) -> Vec<Vec<u64>> {
        self.output_branches
            .values()
            .filter_map(|output| {
                let computed_output = &output[..output.len().saturating_sub(1)];
                (!computed_output.is_empty()).then(|| {
                    let mut sequence =
                        Vec::with_capacity(self.prompt_tokens.len() + computed_output.len());
                    sequence.extend_from_slice(&self.prompt_tokens);
                    sequence.extend_from_slice(computed_output);
                    self.sequence_hashes(&sequence)
                        .into_iter()
                        .skip(self.prompt_hashes.len())
                        .collect()
                })
            })
            .collect()
    }

    pub fn record_prompt(&mut self, history: &mut CacheHistory, hashes: Vec<u64>) {
        if self.prompt_recorded {
            return;
        }
        history.record_completed(hashes);
        self.prompt_recorded = true;
    }

    pub fn finalize(
        &mut self,
        history: &mut CacheHistory,
        prompt_hashes: Vec<u64>,
        output_hashes: Vec<Vec<u64>>,
    ) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        self.record_prompt(history, prompt_hashes);
        for hashes in output_hashes {
            history.record_completed(hashes);
        }
    }

    fn sequence_hashes(&self, tokens: &[u32]) -> Vec<u64> {
        let mut tokens_with_hashes =
            TokensWithHashes::new(tokens.to_vec(), self.block_size).with_is_eagle(self.is_eagle);
        if let Some(infos) = &self.block_mm_infos {
            tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.clone());
        }
        if let Some(lora_name) = &self.lora_name {
            tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name.clone());
        }
        if let Some(cache_namespace) = &self.cache_namespace {
            tokens_with_hashes = tokens_with_hashes.with_cache_namespace(cache_namespace.clone());
        }
        tokens_with_hashes.get_or_compute_seq_hashes().to_vec()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RouteObservation {
    pub prompt_tokens: u64,
    pub previously_computed_tokens: u64,
    pub best_router_tokens: u64,
    pub selected_router_tokens: u64,
}

impl RouteObservation {
    pub fn bounded(self) -> Self {
        let prompt_tokens = self.prompt_tokens;
        Self {
            prompt_tokens,
            previously_computed_tokens: self.previously_computed_tokens.min(prompt_tokens),
            best_router_tokens: self.best_router_tokens.min(prompt_tokens),
            selected_router_tokens: self.selected_router_tokens.min(prompt_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CACHE_LOSS_FUNNEL_ENABLED_ENV, CacheHistory, enabled};

    #[test]
    #[serial_test::serial]
    fn cache_loss_funnel_is_disabled_by_default() {
        temp_env::with_var_unset(CACHE_LOSS_FUNNEL_ENABLED_ENV, || {
            assert!(!enabled());
        });
    }

    #[test]
    #[serial_test::serial]
    fn cache_loss_funnel_accepts_truthy_enablement() {
        temp_env::with_var(CACHE_LOSS_FUNNEL_ENABLED_ENV, Some("true"), || {
            assert!(enabled());
        });
    }

    #[test]
    fn retains_recent_records_and_expires_the_oldest_chunk() {
        let mut history = CacheHistory::new_with_chunk(2, 16, 1);
        history.record_completed([10, 20]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 32);

        history.record_completed([30]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 0);
        assert_eq!(history.previously_computed_tokens(&[20, 30]), 32);
        assert_eq!(history.stats().represented_tokens, 32);
    }

    #[test]
    fn evicts_a_complete_oldest_chunk() {
        let mut history = CacheHistory::new_with_chunk(8, 16, 4);
        history.record_completed([1, 2, 3, 4]);
        history.record_completed([5, 6, 7, 8]);
        history.record_completed([9]);
        assert_eq!(history.stats().retained_records, 5);
        assert_eq!(history.previously_computed_tokens(&[1]), 0);
        assert_eq!(history.previously_computed_tokens(&[5, 6, 7, 8, 9]), 80);
    }

    #[test]
    fn duplicate_records_keep_a_hash_retained_until_all_expire() {
        let mut history = CacheHistory::new_with_chunk(2, 8, 1);
        history.record_completed([7, 7]);
        history.record_completed([9]);

        assert_eq!(history.previously_computed_tokens(&[7]), 8);
        history.record_completed([11]);
        assert_eq!(history.previously_computed_tokens(&[7]), 0);
    }

    #[test]
    fn generated_history_excludes_the_newest_sampled_token() {
        let mut request =
            super::CacheHistoryRequest::new(vec![1, 2, 3, 4], None, None, None, 2, false);
        request.observe_output(0, &[5, 6, 7, 8]);
        let mut history = CacheHistory::new(32, 2);
        let prompt_hashes = request.prompt_hashes();
        let output_hashes = request.output_hashes();
        request.finalize(&mut history, prompt_hashes, output_hashes);

        // The newest sampled token has no corresponding KV entry.
        assert_eq!(history.stats().retained_records, 3);
    }

    #[test]
    fn default_budget_derives_capacity_without_a_block_limit() {
        let history = CacheHistory::from_env(16);
        let history = history.lock();
        assert_eq!(
            history.stats().capacity_blocks,
            super::DEFAULT_HISTORY_BYTES / super::ESTIMATED_BYTES_PER_HISTORY_RECORD
        );
    }
}
