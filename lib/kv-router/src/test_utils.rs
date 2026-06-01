// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for radix tree tests.

use std::collections::HashSet;
use std::future;

use crate::indexer::KvIndexerInterface;
use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, OverlapScores,
    RouterEvent, WorkerConfigLike, WorkerId, WorkerWithDpRank, compute_seq_hash_for_block,
};
use crate::sequences::SequencePublisher;

pub fn router_event(
    worker_id: WorkerId,
    event_id: u64,
    dp_rank: u32,
    data: KvCacheEventData,
) -> RouterEvent {
    RouterEvent::new(
        worker_id,
        KvCacheEvent {
            event_id,
            data,
            dp_rank,
        },
    )
}

pub fn stored_blocks_with_sequence_hashes(
    local_hashes: &[LocalBlockHash],
    seq_hashes: &[u64],
) -> Vec<KvCacheStoredBlockData> {
    local_hashes
        .iter()
        .zip(seq_hashes.iter())
        .map(|(&local, &seq)| KvCacheStoredBlockData {
            tokens_hash: local,
            block_hash: ExternalSequenceBlockHash(seq),
            mm_extra_info: None,
        })
        .collect()
}

pub fn remove_event(
    worker_id: WorkerId,
    event_id: u64,
    dp_rank: u32,
    block_hashes: Vec<ExternalSequenceBlockHash>,
) -> RouterEvent {
    router_event(
        worker_id,
        event_id,
        dp_rank,
        KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
    )
}

/// Creates blocks with artificial hash mapping (hash * 100) for testing.
pub fn make_blocks(hashes: Vec<u64>) -> Vec<KvCacheStoredBlockData> {
    hashes
        .iter()
        .map(|i| KvCacheStoredBlockData {
            tokens_hash: LocalBlockHash(*i),
            block_hash: ExternalSequenceBlockHash(*i * 100),
            mm_extra_info: None,
        })
        .collect()
}

pub fn add_blocks(
    hashes: Vec<u64>,
    parent_hash: Option<ExternalSequenceBlockHash>,
) -> KvCacheEventData {
    add_blocks_with_start_position(hashes, parent_hash, None)
}

pub fn add_blocks_with_start_position(
    hashes: Vec<u64>,
    parent_hash: Option<ExternalSequenceBlockHash>,
    start_position: Option<u32>,
) -> KvCacheEventData {
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash,
        start_position,
        blocks: make_blocks(hashes),
    })
}

pub fn create_store_event(
    worker_id: WorkerId,
    event_id: u64,
    hashes: Vec<u64>,
    parent: Option<ExternalSequenceBlockHash>,
) -> RouterEvent {
    router_event(worker_id, event_id, 0, add_blocks(hashes, parent))
}

pub fn create_store_event_with_start_position(
    worker_id: WorkerId,
    event_id: u64,
    hashes: Vec<u64>,
    parent: Option<ExternalSequenceBlockHash>,
    start_position: Option<u32>,
) -> RouterEvent {
    router_event(
        worker_id,
        event_id,
        0,
        add_blocks_with_start_position(hashes, parent, start_position),
    )
}

pub fn create_remove_event(worker_id: WorkerId, event_id: u64, hashes: Vec<u64>) -> RouterEvent {
    remove_event(
        worker_id,
        event_id,
        0,
        hashes
            .iter()
            .map(|i| ExternalSequenceBlockHash(*i * 100))
            .collect(),
    )
}

/// Create a store event with proper sequence hashes computed from local hashes.
pub fn make_store_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
    make_store_event_with_dp_rank(worker_id, local_hashes, 0)
}

/// Create a store event with a specific dp_rank.
pub fn make_store_event_with_dp_rank(
    worker_id: u64,
    local_hashes: &[u64],
    dp_rank: u32,
) -> RouterEvent {
    make_store_event_full(worker_id, local_hashes, dp_rank, None, None)
}

/// Create a store event with parent hash for continuation sequences.
pub fn make_store_event_with_parent(
    worker_id: u64,
    prefix_hashes: &[u64],
    local_hashes: &[u64],
) -> RouterEvent {
    let prefix_block_hashes: Vec<LocalBlockHash> =
        prefix_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let prefix_seq_hashes = compute_seq_hash_for_block(&prefix_block_hashes);
    let parent_hash = prefix_seq_hashes
        .last()
        .map(|&h| ExternalSequenceBlockHash(h));

    let full_hashes: Vec<u64> = prefix_hashes
        .iter()
        .chain(local_hashes.iter())
        .copied()
        .collect();
    let full_block_hashes: Vec<LocalBlockHash> =
        full_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let full_seq_hashes = compute_seq_hash_for_block(&full_block_hashes);

    let new_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let new_seq_hashes = &full_seq_hashes[prefix_hashes.len()..];

    router_event(
        worker_id,
        0,
        0,
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(&new_block_hashes, new_seq_hashes),
        }),
    )
}

pub fn make_store_event_with_start_position(
    worker_id: u64,
    local_hashes: &[u64],
    start_position: u32,
) -> RouterEvent {
    make_store_event_full(worker_id, local_hashes, 0, None, Some(start_position))
}

/// Create a store event with all options.
pub fn make_store_event_full(
    worker_id: u64,
    local_hashes: &[u64],
    dp_rank: u32,
    parent_hash: Option<ExternalSequenceBlockHash>,
    start_position: Option<u32>,
) -> RouterEvent {
    let local_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

    router_event(
        worker_id,
        0,
        dp_rank,
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            start_position,
            blocks: stored_blocks_with_sequence_hashes(&local_block_hashes, &seq_hashes),
        }),
    )
}

/// Create a remove event for blocks with given local hashes.
pub fn make_remove_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
    make_remove_event_with_dp_rank(worker_id, local_hashes, 0)
}

/// Create a remove event with a specific dp_rank.
pub fn make_remove_event_with_dp_rank(
    worker_id: u64,
    local_hashes: &[u64],
    dp_rank: u32,
) -> RouterEvent {
    let local_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

    remove_event(
        worker_id,
        0,
        dp_rank,
        seq_hashes
            .iter()
            .map(|&h| ExternalSequenceBlockHash(h))
            .collect(),
    )
}

/// Create a remove event with parent hash for continuation sequences.
pub fn make_remove_event_with_parent(
    worker_id: u64,
    prefix_hashes: &[u64],
    local_hashes: &[u64],
) -> RouterEvent {
    let full_hashes: Vec<u64> = prefix_hashes
        .iter()
        .chain(local_hashes.iter())
        .copied()
        .collect();
    let full_block_hashes: Vec<LocalBlockHash> =
        full_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
    let full_seq_hashes = compute_seq_hash_for_block(&full_block_hashes);
    let suffix_seq_hashes = &full_seq_hashes[prefix_hashes.len()..];

    remove_event(
        worker_id,
        0,
        0,
        suffix_seq_hashes
            .iter()
            .map(|&h| ExternalSequenceBlockHash(h))
            .collect(),
    )
}

/// Create a clear event for a worker.
pub fn make_clear_event(worker_id: u64) -> RouterEvent {
    make_clear_event_with_dp_rank(worker_id, 0)
}

/// Create a clear event with a specific dp_rank.
pub fn make_clear_event_with_dp_rank(worker_id: u64, dp_rank: u32) -> RouterEvent {
    router_event(worker_id, 0, dp_rank, KvCacheEventData::Cleared)
}

/// Snapshot the tree state for deterministic comparison.
pub async fn snapshot_tree(index: &dyn KvIndexerInterface) -> Vec<RouterEvent> {
    snapshot_events(index.dump_events().await.unwrap())
}

pub fn snapshot_events(mut events: Vec<RouterEvent>) -> Vec<RouterEvent> {
    for ev in &mut events {
        ev.event.event_id = 0;
    }
    events.sort_by(|a, b| {
        a.worker_id.cmp(&b.worker_id).then_with(|| {
            a.event.dp_rank.cmp(&b.event.dp_rank).then_with(|| {
                let hash_a = match &a.event.data {
                    KvCacheEventData::Stored(s) => {
                        s.blocks.first().map(|b| b.block_hash.0).unwrap_or(0)
                    }
                    KvCacheEventData::Removed(r) => {
                        r.block_hashes.first().map(|h| h.0).unwrap_or(0)
                    }
                    KvCacheEventData::Cleared => 0,
                };
                let hash_b = match &b.event.data {
                    KvCacheEventData::Stored(s) => {
                        s.blocks.first().map(|b| b.block_hash.0).unwrap_or(0)
                    }
                    KvCacheEventData::Removed(r) => {
                        r.block_hashes.first().map(|h| h.0).unwrap_or(0)
                    }
                    KvCacheEventData::Cleared => 0,
                };
                hash_a.cmp(&hash_b)
            })
        })
    });
    events
}

pub async fn flush_and_settle(index: &dyn KvIndexerInterface) {
    index.flush().await;
}

pub async fn query_scores(index: &dyn KvIndexerInterface, query: &[u64]) -> OverlapScores {
    index
        .find_matches(query.iter().copied().map(LocalBlockHash).collect())
        .await
        .unwrap()
}

pub async fn assert_score(
    index: &dyn KvIndexerInterface,
    query: &[u64],
    worker: WorkerWithDpRank,
    expected_score: u32,
) {
    let scores = query_scores(index, query).await;
    assert_eq!(scores.scores.get(&worker), Some(&expected_score));
}

pub async fn assert_no_scores(index: &dyn KvIndexerInterface, query: &[u64]) {
    let scores = query_scores(index, query).await;
    assert!(scores.scores.is_empty());
}

pub async fn assert_exact_scores(
    index: &dyn KvIndexerInterface,
    query: &[u64],
    expected_scores: &[(WorkerWithDpRank, u32)],
) {
    let scores = query_scores(index, query).await;
    assert_eq!(scores.scores.len(), expected_scores.len());
    for (worker, expected_score) in expected_scores {
        assert_eq!(scores.scores.get(worker), Some(expected_score));
    }
}

/// No-op [`SequencePublisher`] for tests and benchmarks that don't need event transport.
pub struct NoopSequencePublisher;

impl SequencePublisher for NoopSequencePublisher {
    fn publish_event(
        &self,
        _event: &ActiveSequenceEvent,
    ) -> impl future::Future<Output = anyhow::Result<()>> + Send {
        future::ready(Ok(()))
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
}

/// Minimal [`WorkerConfigLike`] for scheduler/queue tests and benchmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleWorkerConfig {
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub taints: HashSet<String>,
}

impl Default for SimpleWorkerConfig {
    fn default() -> Self {
        Self {
            data_parallel_start_rank: 0,
            data_parallel_size: 1,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
            taints: HashSet::new(),
        }
    }
}

impl WorkerConfigLike for SimpleWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}
