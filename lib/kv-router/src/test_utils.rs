// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for radix tree tests.

use std::future;

use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    WorkerConfigLike, WorkerId, WorkerWithDpRank,
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
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash,
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
}

impl Default for SimpleWorkerConfig {
    fn default() -> Self {
        Self {
            data_parallel_start_rank: 0,
            data_parallel_size: 1,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
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
}
