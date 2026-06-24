// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{DpRank, LocalBlockHash, SharedCacheHits, WorkerId, WorkerWithDpRank},
    scheduling::overlap::{
        CacheHitEstimates, cache_hit_estimates_from_tiered_matches,
        tier_overlap_blocks_from_tiered_matches,
    },
};
use dynamo_runtime::pipeline::async_trait;
use serde::Serialize;

use super::{
    indexer::Indexer,
    scheduler::{OverlapScoresRefresh, RefreshedOverlap},
};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WorkerCacheHitEstimate {
    pub effective_overlap_blocks: f64,
}

impl WorkerCacheHitEstimate {
    pub fn rounded_overlap_blocks(self) -> u32 {
        self.effective_overlap_blocks.round() as u32
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerOverlapScore {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub device_blocks: usize,
    pub host_pinned_blocks: usize,
    pub disk_blocks: usize,
    pub host_pinned_extension_blocks: usize,
    pub disk_extension_blocks: usize,
    pub shared_beyond_device_blocks: Option<u32>,
    pub router_credit_blocks: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedCacheOverlapScore {
    pub enabled: bool,
    pub total_hit_blocks: u32,
    pub ranges: Vec<(u32, u32)>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverlapScoresResponse {
    pub block_size: u32,
    pub num_blocks: usize,
    pub workers: Vec<WorkerOverlapScore>,
    pub shared_cache: SharedCacheOverlapScore,
}

pub(super) fn cache_hit_for_worker(
    cache_hit_estimates: &CacheHitEstimates,
    worker: WorkerWithDpRank,
) -> WorkerCacheHitEstimate {
    WorkerCacheHitEstimate {
        effective_overlap_blocks: cache_hit_estimates
            .effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0),
    }
}

pub(super) fn shared_cache_overlap_score(
    enabled: bool,
    hits: Option<&SharedCacheHits>,
    error: Option<String>,
) -> SharedCacheOverlapScore {
    let Some(hits) = hits else {
        return SharedCacheOverlapScore {
            enabled,
            total_hit_blocks: 0,
            ranges: Vec::new(),
            error,
        };
    };

    SharedCacheOverlapScore {
        enabled,
        total_hit_blocks: hits.total_hits,
        ranges: hits
            .ranges
            .iter()
            .map(|range| (range.start, range.end))
            .collect(),
        error,
    }
}

/// Re-queries the indexer to refresh stale overlap scores for requests that have been
/// waiting in the scheduler queue. Wired into the scheduler's
/// [`OverlapScoresRefresh`](dynamo_kv_router::scheduling::OverlapScoresRefresh) slot at
/// router startup.
///
/// `Remote` and `None` indexer variants intentionally don't get a refresher: remote
/// indexer queries are too expensive to repeat per request, and `None` has nothing to query.
pub(super) struct KvRouterOverlapRefresher {
    indexer: Indexer,
    kv_router_config: KvRouterConfig,
    block_size: u32,
}

impl KvRouterOverlapRefresher {
    pub(super) fn for_indexer(
        indexer: Indexer,
        kv_router_config: KvRouterConfig,
        block_size: u32,
    ) -> Option<Self> {
        match &indexer {
            Indexer::KvIndexer { .. } | Indexer::Concurrent { .. } => Some(Self {
                indexer,
                kv_router_config,
                block_size,
            }),
            Indexer::Remote { .. } | Indexer::None => None,
        }
    }
}

#[async_trait]
impl OverlapScoresRefresh for KvRouterOverlapRefresher {
    async fn refresh(&self, block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
        let tiered = match self.indexer.find_matches_by_tier_ref(block_hashes).await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = ?e, "overlap refresh: find_matches_by_tier failed");
                return None;
            }
        };
        let tier_overlap_blocks = tier_overlap_blocks_from_tiered_matches(&tiered);
        let estimates = cache_hit_estimates_from_tiered_matches(
            &self.kv_router_config,
            self.block_size,
            &tiered,
        );
        Some(RefreshedOverlap {
            tier_overlap_blocks,
            effective_overlap_blocks: estimates.effective_overlap_blocks.into_iter().collect(),
            effective_cached_tokens: estimates.cached_tokens.into_iter().collect(),
        })
    }
}
