// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Naive indexer implementations for benchmarking purposes only.
//!
//! These correspond to blog sections 2 and 3 and exist to show the performance
//! progression from naive approaches to the production indexers.
//!
//! - [`NaiveNestedMap`]: `worker -> { local_hash -> set<seq_hash> }`.  O(W × D)
//!   per `find_matches` call, behind a true single-threaded actor.  Blog section 2.
//! - [`InvertedIndex`]: `local_hash -> { seq_hash -> set<worker> }`.  O(D + W)
//!   per `find_matches` call.  Blog section 3.

use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use tokio::sync::{mpsc, oneshot};

use crate::indexer::{KvIndexerInterface, KvRouterError, SyncIndexer};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores,
    RouterEvent, TokensWithHashes, WorkerId, WorkerWithDpRank,
};

// ============================================================================
// Section 2 — Naive Nested Map + Actor
// ============================================================================

/// Plain nested `HashMap` index — no locks, owned exclusively by the actor thread.
///
/// Structure: `worker -> { local_hash -> set<seq_hash> }`.
struct NaiveNestedMapInner {
    index: HashMap<WorkerWithDpRank, HashMap<LocalBlockHash, HashSet<ExternalSequenceBlockHash>>>,
    reverse: HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, LocalBlockHash>>,
}

impl NaiveNestedMapInner {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    fn find_matches(&self, sequence: &[LocalBlockHash]) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        for (worker, blocks) in &self.index {
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

    fn apply_event(&mut self, event: RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let worker_map = self.index.entry(worker).or_default();
                let rev_map = self.reverse.entry(worker).or_default();

                for block in store_data.blocks {
                    worker_map
                        .entry(block.tokens_hash)
                        .or_default()
                        .insert(block.block_hash);
                    rev_map.insert(block.block_hash, block.tokens_hash);
                }
            }
            KvCacheEventData::Removed(remove_data) => {
                let Some(worker_map) = self.index.get_mut(&worker) else {
                    return;
                };
                let Some(rev_map) = self.reverse.get_mut(&worker) else {
                    return;
                };

                for seq_hash in &remove_data.block_hashes {
                    let Some(local_hash) = rev_map.remove(seq_hash) else {
                        continue;
                    };
                    if let Some(set) = worker_map.get_mut(&local_hash) {
                        set.remove(seq_hash);
                    }
                }
            }
            KvCacheEventData::Cleared => {
                self.index.remove(&worker);
                self.reverse.remove(&worker);
            }
        }
    }

    fn remove_worker(&mut self, worker_id: WorkerId) {
        self.index.retain(|w, _| w.worker_id != worker_id);
        self.reverse.retain(|w, _| w.worker_id != worker_id);
    }
}

struct MatchRequest {
    sequence: Vec<LocalBlockHash>,
    reply: oneshot::Sender<OverlapScores>,
}

enum ActorMessage {
    Event(RouterEvent),
    Match(MatchRequest),
    RemoveWorker(WorkerId),
}

/// Single-threaded actor wrapping [`NaiveNestedMapInner`] (blog section 2).
///
/// All reads and writes are serialized through a single OS thread via channels.
/// This is the pure actor pattern described in the blog — no concurrent access
/// to the data structure at all.
pub struct NaiveNestedMap {
    tx: mpsc::Sender<ActorMessage>,
}

impl NaiveNestedMap {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<ActorMessage>(2048);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async move {
                let mut inner = NaiveNestedMapInner::new();

                while let Some(msg) = rx.recv().await {
                    match msg {
                        ActorMessage::Event(event) => {
                            inner.apply_event(event);
                        }
                        ActorMessage::Match(req) => {
                            let scores = inner.find_matches(&req.sequence);
                            let _ = req.reply.send(scores);
                        }
                        ActorMessage::RemoveWorker(worker_id) => {
                            inner.remove_worker(worker_id);
                        }
                    }
                }
            });
        });

        Self { tx }
    }
}

#[async_trait]
impl KvIndexerInterface for NaiveNestedMap {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(ActorMessage::Match(MatchRequest {
                sequence,
                reply: reply_tx,
            }))
            .await
            .map_err(|_| KvRouterError::IndexerOffline)?;
        reply_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    async fn find_matches_for_request(
        &self,
        _tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        unimplemented!("not used in bench")
    }

    async fn apply_event(&self, event: RouterEvent) {
        let _ = self.tx.send(ActorMessage::Event(event)).await;
    }

    async fn remove_worker(&self, worker: WorkerId) {
        let _ = self.tx.send(ActorMessage::RemoveWorker(worker)).await;
    }

    fn shutdown(&self) {}

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        Ok(Vec::new())
    }

    async fn process_routing_decision_for_request(
        &self,
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        unimplemented!("not used in bench")
    }

    async fn flush(&self) -> usize {
        let curr_size = self.tx.max_capacity() - self.tx.capacity();
        loop {
            if self.tx.capacity() == self.tx.max_capacity() {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }
        curr_size
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
