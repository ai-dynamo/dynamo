// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! **DO NOT USE IN PRODUCTION.** These are intentionally simplified indexer
//! implementations for benchmarking and blog illustrations only. They cut
//! corners (no reverse lookup, Remove events are unimplemented) that make
//! them incorrect under real workloads with eviction pressure.
//!
//! They correspond to blog sections 2 and 3 and exist to show the performance
//! progression from naive approaches to the production indexers.
//!
//! - [`NaiveNestedMap`]: `worker -> set<local_hash>`.  O(W × D) per
//!   `find_matches` call, behind a single-threaded actor.  Blog section 2.
//! - [`InvertedIndex`]: `local_hash -> set<worker>`.  O(D + W) per
//!   `find_matches` call, single-threaded actor.  Blog section 3.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use tokio::sync::{mpsc, oneshot};

use crate::indexer::{KvIndexerInterface, KvRouterError};
use crate::protocols::{
    KvCacheEventData, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId,
    WorkerWithDpRank,
};

// ============================================================================
// Section 2 — Naive Nested Map + Actor
// ============================================================================

/// Plain nested `HashMap` index — no locks, owned exclusively by the actor thread.
///
/// Structure: `worker -> set<local_hash>`.
/// No reverse lookup — Remove is unimplemented (relies on large GPU block
/// budget to avoid evictions).
struct NaiveNestedMapInner {
    index: HashMap<WorkerWithDpRank, HashSet<LocalBlockHash>>,
}

impl NaiveNestedMapInner {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
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
                if !blocks.contains(local_hash) {
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
                let worker_set = self.index.entry(worker).or_default();
                for block in store_data.blocks {
                    worker_set.insert(block.tokens_hash);
                }
            }
            KvCacheEventData::Removed(_) => {
                unimplemented!(
                    "NaiveNestedMap does not support Remove events; increase --num-gpu-blocks to avoid evictions"
                );
            }
            KvCacheEventData::Cleared => {
                self.index.remove(&worker);
            }
        }
    }

    fn remove_worker(&mut self, worker_id: WorkerId) {
        self.index.retain(|w, _| w.worker_id != worker_id);
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

impl Default for NaiveNestedMap {
    fn default() -> Self {
        Self::new()
    }
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

/// Plain inverted index — no locks, owned exclusively by the actor thread.
///
/// Flat forward index: `local_hash -> set<worker>`.
/// No reverse lookup — Remove is a no-op (relies on large GPU block budget
/// to avoid evictions), Clear/remove_worker scan the forward index.
struct InvertedIndexInner {
    index: HashMap<LocalBlockHash, HashSet<WorkerWithDpRank>>,
}

impl InvertedIndexInner {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn find_matches(&self, sequence: &[LocalBlockHash]) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        let Some(workers) = self.index.get(&sequence[0]) else {
            return scores;
        };
        let mut active: HashSet<WorkerWithDpRank> = workers.clone();

        if active.is_empty() {
            return scores;
        }

        for (depth, local_hash) in sequence.iter().enumerate() {
            let empty = HashSet::new();
            let workers_here = self.index.get(local_hash).unwrap_or(&empty);

            active.retain(|w| {
                if workers_here.contains(w) {
                    true
                } else {
                    scores.scores.insert(*w, depth as u32);
                    false
                }
            });

            if active.is_empty() {
                break;
            }
        }

        for w in active {
            scores.scores.insert(w, sequence.len() as u32);
        }

        scores
    }

    fn apply_event(&mut self, event: RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                for block in store_data.blocks {
                    self.index
                        .entry(block.tokens_hash)
                        .or_default()
                        .insert(worker);
                }
            }
            KvCacheEventData::Removed(_) => {
                unimplemented!(
                    "InvertedIndex does not support Remove events; increase --num-gpu-blocks to avoid evictions"
                );
            }
            KvCacheEventData::Cleared => {
                self.clear_worker(worker);
            }
        }
    }

    fn remove_worker(&mut self, worker_id: WorkerId) {
        for workers in self.index.values_mut() {
            workers.retain(|w| w.worker_id != worker_id);
        }
    }

    fn clear_worker(&mut self, worker: WorkerWithDpRank) {
        for workers in self.index.values_mut() {
            workers.remove(&worker);
        }
    }
}

enum InvertedIndexMessage {
    Event(RouterEvent),
    Match(MatchRequest),
    RemoveWorker(WorkerId),
}

/// Single-threaded actor wrapping [`InvertedIndexInner`] (blog section 3).
///
/// Same actor pattern as [`NaiveNestedMap`] — all reads and writes are
/// serialized through a single OS thread via channels.
pub struct InvertedIndex {
    tx: mpsc::Sender<InvertedIndexMessage>,
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl InvertedIndex {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<InvertedIndexMessage>(2048);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async move {
                let mut inner = InvertedIndexInner::new();

                while let Some(msg) = rx.recv().await {
                    match msg {
                        InvertedIndexMessage::Event(event) => {
                            inner.apply_event(event);
                        }
                        InvertedIndexMessage::Match(req) => {
                            let scores = inner.find_matches(&req.sequence);
                            let _ = req.reply.send(scores);
                        }
                        InvertedIndexMessage::RemoveWorker(worker_id) => {
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
impl KvIndexerInterface for InvertedIndex {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(InvertedIndexMessage::Match(MatchRequest {
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
        let _ = self.tx.send(InvertedIndexMessage::Event(event)).await;
    }

    async fn remove_worker(&self, worker: WorkerId) {
        let _ = self
            .tx
            .send(InvertedIndexMessage::RemoveWorker(worker))
            .await;
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
