// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{DpRank, LocalBlockHash, SharedCacheHits, StorageTier, WorkerId, WorkerWithDpRank},
    scheduling::TierOverlapBlocks,
};
use dynamo_runtime::pipeline::async_trait;
use serde::Serialize;

use super::{
    indexer::{Indexer, TieredMatchDetails},
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

#[derive(Debug, Clone, Default)]
pub(super) struct CacheHitEstimates {
    pub(super) effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub(super) cached_tokens: HashMap<WorkerWithDpRank, usize>,
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

fn cache_hit_weight_for_tier(kv_router_config: &KvRouterConfig, storage_tier: StorageTier) -> f64 {
    match storage_tier {
        StorageTier::Device => 1.0,
        StorageTier::HostPinned => kv_router_config.host_cache_hit_weight,
        StorageTier::Disk | StorageTier::External => kv_router_config.disk_cache_hit_weight,
    }
}

fn cached_tokens_from_effective_overlap(block_size: u32, effective_overlap_blocks: f64) -> usize {
    (effective_overlap_blocks * block_size as f64)
        .round()
        .max(0.0) as usize
}

pub(super) fn cache_hit_estimates_from_tiered_matches(
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    tiered_matches: &TieredMatchDetails,
) -> CacheHitEstimates {
    let mut effective_overlap_blocks = HashMap::new();

    for (worker, overlap) in &tiered_matches.device.overlap_scores.scores {
        effective_overlap_blocks.insert(*worker, *overlap as f64);
    }

    for (storage_tier, tier_matches) in &tiered_matches.lower_tier {
        let weight = cache_hit_weight_for_tier(kv_router_config, *storage_tier);
        if weight == 0.0 {
            continue;
        }

        for (worker, hits) in &tier_matches.hits {
            if *hits == 0 {
                continue;
            }
            *effective_overlap_blocks.entry(*worker).or_insert(0.0) += *hits as f64 * weight;
        }
    }

    let cached_tokens = effective_overlap_blocks
        .iter()
        .map(|(worker, overlap)| {
            (
                *worker,
                cached_tokens_from_effective_overlap(block_size, *overlap),
            )
        })
        .collect();

    CacheHitEstimates {
        effective_overlap_blocks,
        cached_tokens,
    }
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

pub(super) fn tier_overlap_blocks_from_tiered_matches(
    tiered_matches: &TieredMatchDetails,
) -> TierOverlapBlocks {
    let mut tier_overlap_blocks = TierOverlapBlocks::default();

    tier_overlap_blocks.device.extend(
        tiered_matches
            .device
            .overlap_scores
            .scores
            .iter()
            .map(|(worker, hits)| (*worker, *hits as usize)),
    );

    if let Some(host_matches) = tiered_matches.lower_tier.get(&StorageTier::HostPinned) {
        tier_overlap_blocks.host_pinned.extend(
            host_matches
                .hits
                .iter()
                .map(|(worker, hits)| (*worker, *hits)),
        );
    }

    // Disk and External share the same weighting, so accumulate both into the disk bucket.
    for tier in [StorageTier::Disk, StorageTier::External] {
        if let Some(matches) = tiered_matches.lower_tier.get(&tier) {
            for (worker, hits) in &matches.hits {
                *tier_overlap_blocks.disk.entry(*worker).or_default() += *hits;
            }
        }
    }

    tier_overlap_blocks
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
            effective_overlap_blocks: estimates.effective_overlap_blocks,
            effective_cached_tokens: estimates.cached_tokens,
        })
    }
}
