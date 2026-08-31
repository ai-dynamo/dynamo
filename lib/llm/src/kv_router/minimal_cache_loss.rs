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
};

use parking_lot::Mutex;

use dynamo_kv_router::protocols::{BlockExtraInfo, TokensWithHashes};

/// Default number of complete sequence-hash records retained by a frontend.
///
/// Each record represents one full KV block. The ledger is FIFO: once full,
/// the oldest observed record is removed before the newest is inserted.
pub const DEFAULT_HISTORY_BLOCK_CAPACITY: usize = 5_000_000;
pub const HISTORY_BLOCK_CAPACITY_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BLOCKS";
pub const HISTORY_BYTES_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BYTES";
pub const DEFAULT_HISTORY_BYTES: usize = 256 * 1024 * 1024;

/// Conservative planning estimate: an 8-byte FIFO sequence hash plus the
/// amortized hash-map key, refcount, bucket slack, and allocator overhead.
/// This is deliberately larger than `size_of::<u64>()`; it is a capacity model,
/// not a promise about a particular Rust allocator build.
pub const ESTIMATED_BYTES_PER_HISTORY_RECORD: usize = 32;

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
    records: VecDeque<u64>,
    retained: HashMap<u64, u32>,
}

impl CacheHistory {
    pub fn from_env(block_tokens: u32) -> Arc<Mutex<Self>> {
        let requested_blocks = std::env::var(HISTORY_BLOCK_CAPACITY_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_HISTORY_BLOCK_CAPACITY);
        let capacity_bytes = std::env::var(HISTORY_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_HISTORY_BYTES);
        let byte_limited_blocks = capacity_bytes / ESTIMATED_BYTES_PER_HISTORY_RECORD;
        let capacity_blocks = requested_blocks.min(byte_limited_blocks.max(1));
        Arc::new(Mutex::new(Self::new_with_budget(
            capacity_blocks,
            block_tokens,
            capacity_bytes,
        )))
    }

    pub fn new(capacity_blocks: usize, block_tokens: u32) -> Self {
        Self::new_with_budget(
            capacity_blocks,
            block_tokens,
            capacity_blocks.saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
        )
    }

    fn new_with_budget(capacity_blocks: usize, block_tokens: u32, capacity_bytes: usize) -> Self {
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
            records: VecDeque::with_capacity(capacity_blocks.min(65_536)),
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
            if self.records.len() == self.capacity_blocks {
                let evicted = self.records.pop_front().expect("history was non-empty");
                let count = self
                    .retained
                    .get_mut(&evicted)
                    .expect("history refcount missing");
                *count -= 1;
                if *count == 0 {
                    self.retained.remove(&evicted);
                }
            }
            self.records.push_back(hash);
            *self.retained.entry(hash).or_default() += 1;
        }
    }

    pub fn stats(&self) -> CacheHistoryStats {
        CacheHistoryStats {
            capacity_blocks: self.capacity_blocks,
            capacity_bytes: self.capacity_bytes,
            retained_records: self.records.len(),
            retained_unique_hashes: self.retained.len(),
            represented_tokens: (self.records.len() as u64).saturating_mul(self.block_tokens),
            estimated_retained_bytes: self
                .records
                .len()
                .saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
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
}

/// Request-local state used to record canonical prompt and generated histories
/// only after the worker has supplied a complete cache-loss outcome.
///
/// Generated tokens are kept by output-choice index. At finalization the newest
/// sampled token is excluded: it was returned to the caller but has not yet
/// been fed back through the model, so it has no corresponding KV entry.
pub struct CacheHistoryRequest {
    prompt_tokens: Vec<u32>,
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
        Self {
            prompt_tokens,
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

    pub fn previously_computed_tokens(&self, history: &CacheHistory) -> u64 {
        history.previously_computed_tokens(&self.sequence_hashes(&self.prompt_tokens))
    }

    pub fn observe_output(&mut self, output_index: u32, token_ids: &[u32]) {
        if !token_ids.is_empty() {
            self.output_branches
                .entry(output_index)
                .or_default()
                .extend_from_slice(token_ids);
        }
    }

    pub fn record_prompt(&mut self, history: &mut CacheHistory) {
        if self.prompt_recorded {
            return;
        }
        history.record_completed(self.sequence_hashes(&self.prompt_tokens));
        self.prompt_recorded = true;
    }

    pub fn finalize(&mut self, history: &mut CacheHistory) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        self.record_prompt(history);
        for output in self.output_branches.values() {
            let computed_output = &output[..output.len().saturating_sub(1)];
            if computed_output.is_empty() {
                continue;
            }
            let mut sequence = Vec::with_capacity(self.prompt_tokens.len() + computed_output.len());
            sequence.extend_from_slice(&self.prompt_tokens);
            sequence.extend_from_slice(computed_output);
            history.record_completed(self.sequence_hashes(&sequence));
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
    use super::CacheHistory;

    #[test]
    fn retains_recent_records_and_expires_the_oldest() {
        let mut history = CacheHistory::new(2, 16);
        history.record_completed([10, 20]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 32);

        history.record_completed([30]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 0);
        assert_eq!(history.previously_computed_tokens(&[20, 30]), 32);
        assert_eq!(history.stats().represented_tokens, 32);
    }

    #[test]
    fn duplicate_records_keep_a_hash_retained_until_all_expire() {
        let mut history = CacheHistory::new(2, 8);
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
        request.observe_output(0, &[5, 6, 7]);
        let mut history = CacheHistory::new(32, 2);
        request.finalize(&mut history);

        // Prompt has two complete blocks; prompt plus the first two generated
        // tokens has three. The final sampled token is deliberately absent.
        assert_eq!(history.stats().retained_records, 5);
    }

    #[test]
    fn default_budget_can_hold_the_requested_five_million_records() {
        let capacity_from_default_budget =
            super::DEFAULT_HISTORY_BYTES / super::ESTIMATED_BYTES_PER_HISTORY_RECORD;
        assert!(capacity_from_default_budget >= super::DEFAULT_HISTORY_BLOCK_CAPACITY);
    }
}
