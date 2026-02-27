// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::ConcurrentRadixTree;
use dynamo_kv_router::ThreadPoolIndexer;
use dynamo_kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
use dynamo_kv_router::protocols::{LocalBlockHash, OverlapScores, RouterEvent, WorkerId};

#[derive(Clone)]
pub enum Indexer {
    Single(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTree>>),
}

impl Indexer {
    pub async fn apply_event(&self, event: RouterEvent) {
        match self {
            Indexer::Single(idx) => idx.apply_event(event).await,
            Indexer::Concurrent(idx) => idx.apply_event(event).await,
        }
    }

    pub async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Indexer::Single(idx) => idx.remove_worker(worker_id).await,
            Indexer::Concurrent(idx) => idx.remove_worker(worker_id).await,
        }
    }

    pub async fn find_matches(&self, hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        match self {
            Indexer::Single(idx) => idx.find_matches(hashes).await.map_err(Into::into),
            Indexer::Concurrent(idx) => idx.find_matches(hashes).await.map_err(Into::into),
        }
    }

    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>> {
        match self {
            Indexer::Single(idx) => idx.dump_events().await.map_err(Into::into),
            Indexer::Concurrent(idx) => idx.dump_events().await.map_err(Into::into),
        }
    }
}

pub fn create_indexer(block_size: u32, num_threads: usize) -> Indexer {
    if num_threads > 1 {
        Indexer::Concurrent(Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTree::new(),
            num_threads,
            block_size,
        )))
    } else {
        Indexer::Single(KvIndexer::new_with_frequency(
            CancellationToken::new(),
            None,
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            None,
        ))
    }
}
