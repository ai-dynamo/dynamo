// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Naive indexer implementations for benchmarking purposes only.
//!
//! These correspond to blog sections 2 and 3 and exist to show the performance
//! progression from naive approaches to the production indexers.
//!
//! - [`NaiveNestedMap`]: `worker -> { local_hash -> set<seq_hash> }`.  O(W × D)
//!   per `find_matches` call.  Blog section 2.
//! - [`InvertedIndex`]: `local_hash -> { seq_hash -> set<worker> }`.  O(D + W)
//!   per `find_matches` call.  Blog section 3.

use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::indexer::SyncIndexer;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores,
    RouterEvent, WorkerId, WorkerWithDpRank,
};

// ============================================================================
// Section 2 — Naive Nested Map
// ============================================================================

/// Naive per-worker nested `HashMap` index (blog section 2).
///
/// Structure: `worker -> { local_hash -> set<seq_hash> }`.
///
/// `find_matches` iterates every worker and walks the full query depth for each
/// one, giving O(W × D) per call.  A coarse `RwLock` serializes readers and
/// writers, mirroring the blog's single-threaded actor semantics.
pub struct NaiveNestedMap {
    index: RwLock<
        HashMap<WorkerWithDpRank, HashMap<LocalBlockHash, HashSet<ExternalSequenceBlockHash>>>,
    >,
    reverse: RwLock<HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, LocalBlockHash>>>,
}

impl NaiveNestedMap {
    pub fn new() -> Self {
        Self {
            index: RwLock::new(HashMap::new()),
            reverse: RwLock::new(HashMap::new()),
        }
    }
}

impl SyncIndexer for NaiveNestedMap {
    fn find_matches(&self, sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        let index = self.index.read().unwrap();

        for (worker, blocks) in index.iter() {
            let mut depth = 0u32;
            for local_hash in sequence {
                let Some(set) = blocks.get(local_hash) else {
                    break;
                };
                if set.is_empty() {
                    break;
                }
                depth += 1;
            }
            if depth > 0 {
                scores.scores.insert(*worker, depth);
            }
        }

        scores
    }

    fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let mut index = self.index.write().unwrap();
                let mut reverse = self.reverse.write().unwrap();
                let worker_map = index.entry(worker).or_default();
                let rev_map = reverse.entry(worker).or_default();

                for block in store_data.blocks {
                    worker_map
                        .entry(block.tokens_hash)
                        .or_default()
                        .insert(block.block_hash);
                    rev_map.insert(block.block_hash, block.tokens_hash);
                }

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                let mut index = self.index.write().unwrap();
                let mut reverse = self.reverse.write().unwrap();

                let Some(worker_map) = index.get_mut(&worker) else {
                    return Ok(());
                };
                let Some(rev_map) = reverse.get_mut(&worker) else {
                    return Ok(());
                };

                for seq_hash in &remove_data.block_hashes {
                    let Some(local_hash) = rev_map.remove(seq_hash) else {
                        continue;
                    };
                    if let Some(set) = worker_map.get_mut(&local_hash) {
                        set.remove(seq_hash);
                    }
                }

                Ok(())
            }
            KvCacheEventData::Cleared => {
                self.clear_worker(worker);
                Ok(())
            }
        }
    }

    fn remove_worker(&self, worker_id: WorkerId) {
        let mut index = self.index.write().unwrap();
        let mut reverse = self.reverse.write().unwrap();
        index.retain(|w, _| w.worker_id != worker_id);
        reverse.retain(|w, _| w.worker_id != worker_id);
    }

    fn dump_events(&self) -> Vec<RouterEvent> {
        Vec::new()
    }
}

impl NaiveNestedMap {
    fn clear_worker(&self, worker: WorkerWithDpRank) {
        let mut index = self.index.write().unwrap();
        let mut reverse = self.reverse.write().unwrap();
        index.remove(&worker);
        reverse.remove(&worker);
    }
}

// ============================================================================
// Section 3 — Inverted Index
// ============================================================================

/// Inverted index keyed by `LocalBlockHash` (blog section 3).
///
/// Structure: `local_hash -> { seq_hash -> set<worker> }`.
///
/// `find_matches` walks the query once and drains workers as they stop
/// matching, giving O(D + W) per call.
pub struct InvertedIndex {
    index: DashMap<LocalBlockHash, HashMap<ExternalSequenceBlockHash, HashSet<WorkerWithDpRank>>>,
    reverse: DashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, LocalBlockHash>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            index: DashMap::new(),
            reverse: DashMap::new(),
        }
    }
}

impl SyncIndexer for InvertedIndex {
    fn find_matches(&self, sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        // Collect active worker set from position 0.
        let Some(entry) = self.index.get(&sequence[0]) else {
            return scores;
        };
        let mut active: HashSet<WorkerWithDpRank> =
            entry.values().flat_map(|s| s.iter().copied()).collect();
        drop(entry);

        if active.is_empty() {
            return scores;
        }

        for (depth, local_hash) in sequence.iter().enumerate() {
            let workers_here: HashSet<WorkerWithDpRank> = self
                .index
                .get(local_hash)
                .map(|e| e.values().flat_map(|s| s.iter().copied()).collect())
                .unwrap_or_default();

            let drained: Vec<WorkerWithDpRank> = active
                .iter()
                .filter(|w| !workers_here.contains(w))
                .copied()
                .collect();

            for w in drained {
                active.remove(&w);
                scores.scores.insert(w, depth as u32);
            }

            if active.is_empty() {
                break;
            }
        }

        for w in active {
            scores.scores.insert(w, sequence.len() as u32);
        }

        scores
    }

    fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                if !self.reverse.contains_key(&worker) {
                    self.reverse.insert(worker, HashMap::new());
                }
                let mut rev = self.reverse.get_mut(&worker).unwrap();

                for block in store_data.blocks {
                    self.index
                        .entry(block.tokens_hash)
                        .or_default()
                        .entry(block.block_hash)
                        .or_default()
                        .insert(worker);
                    rev.insert(block.block_hash, block.tokens_hash);
                }

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                let Some(mut rev) = self.reverse.get_mut(&worker) else {
                    return Ok(());
                };

                for seq_hash in &remove_data.block_hashes {
                    let Some(local_hash) = rev.remove(seq_hash) else {
                        continue;
                    };
                    if let Some(mut entry) = self.index.get_mut(&local_hash) {
                        if let Some(workers) = entry.get_mut(seq_hash) {
                            workers.remove(&worker);
                        }
                    }
                }

                Ok(())
            }
            KvCacheEventData::Cleared => {
                self.clear_worker(worker);
                Ok(())
            }
        }
    }

    fn remove_worker(&self, worker_id: WorkerId) {
        let workers_to_remove: Vec<WorkerWithDpRank> = self
            .reverse
            .iter()
            .filter(|e| e.key().worker_id == worker_id)
            .map(|e| *e.key())
            .collect();

        for worker in workers_to_remove {
            self.clear_worker(worker);
        }
    }

    fn dump_events(&self) -> Vec<RouterEvent> {
        Vec::new()
    }
}

impl InvertedIndex {
    fn clear_worker(&self, worker: WorkerWithDpRank) {
        let Some((_, rev_map)) = self.reverse.remove(&worker) else {
            return;
        };

        for (seq_hash, local_hash) in rev_map {
            if let Some(mut entry) = self.index.get_mut(&local_hash) {
                if let Some(workers) = entry.get_mut(&seq_hash) {
                    workers.remove(&worker);
                }
            }
        }
    }
}
