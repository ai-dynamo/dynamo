// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashSet;

pub(crate) const CACHE_HISTORY_ENABLED_ENV: &str = "DYN_CACHE_REUSE_HISTORY_ENABLED";
pub(crate) const HISTORY_BYTES_ENV: &str = "DYN_CACHE_REUSE_HISTORY_BYTES";
pub(crate) const DEFAULT_HISTORY_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const ESTIMATED_BYTES_PER_ENTRY: usize = 32;
const HISTORY_SHARDS: usize = 64;

pub(crate) fn enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(CACHE_HISTORY_ENABLED_ENV)
}

/// Process-local FIFO membership history for canonical sequence hashes.
///
/// Reads touch only one shard at a time. The FIFO lock serializes insertions and
/// evictions so a recomputation cannot race an eviction and leave membership
/// inconsistent. Duplicate hashes neither consume capacity nor refresh age.
pub(crate) struct CacheHistory {
    block_tokens: u64,
    capacity_entries: usize,
    capacity_bytes: usize,
    fifo: Mutex<VecDeque<u64>>,
    shards: Box<[RwLock<FxHashSet<u64>>]>,
    retained_entries: AtomicUsize,
}

impl CacheHistory {
    pub(crate) fn from_env(block_tokens: u32) -> Arc<Self> {
        let requested_bytes = std::env::var(HISTORY_BYTES_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_HISTORY_BYTES);
        Arc::new(Self::with_byte_budget(block_tokens, requested_bytes))
    }

    fn with_byte_budget(block_tokens: u32, requested_bytes: usize) -> Self {
        assert!(
            block_tokens > 0,
            "cache history block size must be positive"
        );
        let capacity_bytes = requested_bytes.max(ESTIMATED_BYTES_PER_ENTRY);
        let capacity_entries = capacity_bytes / ESTIMATED_BYTES_PER_ENTRY;
        let shards = (0..HISTORY_SHARDS)
            .map(|_| RwLock::new(FxHashSet::default()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            block_tokens: u64::from(block_tokens),
            capacity_entries,
            capacity_bytes,
            fifo: Mutex::new(VecDeque::new()),
            shards,
            retained_entries: AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_capacity(capacity_entries: usize, block_tokens: u32) -> Self {
        Self::with_byte_budget(
            block_tokens,
            capacity_entries.saturating_mul(ESTIMATED_BYTES_PER_ENTRY),
        )
    }

    pub(crate) fn previously_computed_tokens(&self, sequence_hashes: &[u64]) -> u64 {
        let blocks = sequence_hashes
            .iter()
            .take_while(|&&hash| {
                let shard = self.shard(hash);
                self.shards[shard].read().contains(&hash)
            })
            .count() as u64;
        blocks.saturating_mul(self.block_tokens)
    }

    pub(crate) fn record_completed<I>(&self, sequence_hashes: I) -> Option<CacheHistoryStats>
    where
        I: Iterator<Item = u64> + Clone,
    {
        if sequence_hashes.clone().all(|hash| {
            let shard = self.shard(hash);
            self.shards[shard].read().contains(&hash)
        }) {
            return None;
        }

        let mut fifo = self.fifo.lock();
        let initial_len = fifo.len();
        for hash in sequence_hashes {
            let shard = self.shard(hash);
            let mut membership = self.shards[shard].write();
            if membership.contains(&hash) {
                continue;
            }

            if fifo.len() == self.capacity_entries {
                let evicted = fifo.pop_front().expect("history capacity is positive");
                let evicted_shard = self.shard(evicted);
                let removed = if evicted_shard == shard {
                    membership.remove(&evicted)
                } else {
                    self.shards[evicted_shard].write().remove(&evicted)
                };
                debug_assert!(removed, "FIFO and membership shards diverged");
            }

            let inserted = membership.insert(hash);
            debug_assert!(inserted, "FIFO writer must serialize membership updates");
            fifo.push_back(hash);
        }
        self.retained_entries.store(fifo.len(), Ordering::Relaxed);
        (fifo.len() != initial_len).then(|| self.stats_for_entries(fifo.len()))
    }

    pub(crate) fn stats(&self) -> CacheHistoryStats {
        let retained_entries = self.retained_entries.load(Ordering::Relaxed);
        self.stats_for_entries(retained_entries)
    }

    fn stats_for_entries(&self, retained_entries: usize) -> CacheHistoryStats {
        CacheHistoryStats {
            retained_entries,
            represented_tokens: (retained_entries as u64).saturating_mul(self.block_tokens),
            estimated_retained_bytes: retained_entries.saturating_mul(ESTIMATED_BYTES_PER_ENTRY),
            capacity_entries: self.capacity_entries,
            capacity_bytes: self.capacity_bytes,
        }
    }

    fn shard(&self, hash: u64) -> usize {
        let mut mixed = hash;
        mixed ^= mixed >> 33;
        mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
        mixed ^= mixed >> 33;
        (mixed as usize) & (HISTORY_SHARDS - 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheHistoryStats {
    pub(crate) retained_entries: usize,
    pub(crate) represented_tokens: u64,
    pub(crate) estimated_retained_bytes: usize,
    pub(crate) capacity_entries: usize,
    pub(crate) capacity_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn longest_prefix_stops_at_first_unseen_hash() {
        let history = CacheHistory::with_capacity(4, 16);
        assert!(history.record_completed([10, 20, 30].into_iter()).is_some());

        assert_eq!(history.previously_computed_tokens(&[10, 20, 99, 30]), 32);
    }

    #[test]
    fn duplicate_does_not_consume_capacity_or_refresh_fifo_age() {
        let history = CacheHistory::with_capacity(2, 8);
        assert!(history.record_completed([10, 20].into_iter()).is_some());
        history.record_completed([10, 30].into_iter());

        assert_eq!(history.previously_computed_tokens(&[10]), 0);
        assert_eq!(history.previously_computed_tokens(&[20, 30]), 16);
        assert_eq!(history.stats().retained_entries, 2);
    }

    #[test]
    fn duplicate_only_completion_reports_no_change() {
        let history = CacheHistory::with_capacity(2, 8);
        assert!(history.record_completed([10, 20].into_iter()).is_some());
        assert!(history.record_completed([10, 20, 10].into_iter()).is_none());
    }

    #[test]
    fn concurrent_recomputations_and_evictions_keep_fifo_consistent() {
        let history = Arc::new(CacheHistory::with_capacity(8, 8));
        let barrier = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            for thread in 0..8_u64 {
                let history = history.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    for offset in 0..64_u64 {
                        history
                            .record_completed([offset % 4, thread * 64 + offset + 10].into_iter());
                    }
                });
            }
        });

        let fifo = history.fifo.lock();
        assert_eq!(fifo.len(), history.stats().retained_entries);
        let unique = fifo.iter().copied().collect::<FxHashSet<_>>();
        assert_eq!(unique.len(), fifo.len());
        for &hash in fifo.iter() {
            assert!(history.shards[history.shard(hash)].read().contains(&hash));
        }
    }

    #[test]
    fn disabled_by_default() {
        temp_env::with_var_unset(CACHE_HISTORY_ENABLED_ENV, || assert!(!enabled()));
    }

    #[test]
    fn explicit_flag_enables_history() {
        temp_env::with_var(CACHE_HISTORY_ENABLED_ENV, Some("true"), || {
            assert!(enabled())
        });
    }
}
