// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{StorageTier, WorkerWithDpRank};
use crate::scheduling::TierOverlapBlocks;
use crate::scheduling::config::RouterConfigOverride;

use super::types::{OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore};

#[derive(Default)]
pub(super) struct OverlapInputs {
    pub(super) tier_overlap_blocks: TierOverlapBlocks,
    pub(super) effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub(super) effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
}

#[derive(Default)]
pub(super) struct CacheHitEstimates {
    pub(super) effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub(super) cached_tokens: HashMap<WorkerWithDpRank, usize>,
}

fn cache_hit_weight_for_tier(config: &crate::config::KvRouterConfig, tier: StorageTier) -> f64 {
    match tier {
        StorageTier::Device => 1.0,
        StorageTier::HostPinned => config.host_cache_hit_weight,
        StorageTier::Disk | StorageTier::External => config.disk_cache_hit_weight,
    }
}

pub(super) fn cache_hit_estimates_from_tiered_matches(
    config: &crate::config::KvRouterConfig,
    block_size: u32,
    tiered_matches: &TieredMatchDetails,
) -> CacheHitEstimates {
    let mut effective_overlap_blocks = HashMap::new();

    for (worker, overlap) in &tiered_matches.device.overlap_scores.scores {
        effective_overlap_blocks.insert(*worker, *overlap as f64);
    }

    for (storage_tier, tier_matches) in &tiered_matches.lower_tier {
        let weight = cache_hit_weight_for_tier(config, *storage_tier);
        if weight == 0.0 {
            continue;
        }
        for (worker, hits) in &tier_matches.hits {
            *effective_overlap_blocks.entry(*worker).or_insert(0.0) += *hits as f64 * weight;
        }
    }

    let cached_tokens = effective_overlap_blocks
        .iter()
        .map(|(worker, overlap)| {
            (
                *worker,
                (*overlap * block_size as f64).round().max(0.0) as usize,
            )
        })
        .collect();

    CacheHitEstimates {
        effective_overlap_blocks,
        cached_tokens,
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

    for tier in [StorageTier::Disk, StorageTier::External] {
        if let Some(matches) = tiered_matches.lower_tier.get(&tier) {
            for (worker, hits) in &matches.hits {
                *tier_overlap_blocks.disk.entry(*worker).or_default() += *hits;
            }
        }
    }

    tier_overlap_blocks
}

pub(super) fn build_overlap_scores_response(
    config: &crate::config::KvRouterConfig,
    config_override: Option<&RouterConfigOverride>,
    tiered: &TieredMatchDetails,
    block_size: u32,
    schedulable_workers: impl IntoIterator<Item = WorkerWithDpRank>,
) -> OverlapScoresResponse {
    let mut all_workers = HashSet::new();
    all_workers.extend(schedulable_workers);
    for worker in tiered.device.overlap_scores.scores.keys() {
        all_workers.insert(*worker);
    }
    for matches in tiered.lower_tier.values() {
        for worker in matches.hits.keys() {
            all_workers.insert(*worker);
        }
    }

    let host = tiered.lower_tier.get(&StorageTier::HostPinned);
    let disk = tiered.lower_tier.get(&StorageTier::Disk);
    let external = tiered.lower_tier.get(&StorageTier::External);
    let overlap_score_credit = config_override
        .and_then(|cfg| cfg.overlap_score_credit)
        .unwrap_or(config.overlap_score_credit);

    let mut workers: Vec<_> = all_workers
        .into_iter()
        .map(|worker| {
            let device_blocks = tiered
                .device
                .overlap_scores
                .scores
                .get(&worker)
                .copied()
                .unwrap_or(0) as usize;
            let host_pinned_extension_blocks = host
                .and_then(|matches| matches.hits.get(&worker))
                .copied()
                .unwrap_or(0);
            let disk_extension_blocks = disk
                .and_then(|matches| matches.hits.get(&worker))
                .copied()
                .unwrap_or(0)
                + external
                    .and_then(|matches| matches.hits.get(&worker))
                    .copied()
                    .unwrap_or(0);
            let host_pinned_blocks = device_blocks + host_pinned_extension_blocks;
            let disk_blocks = host_pinned_blocks + disk_extension_blocks;
            let router_credit_blocks = overlap_score_credit * device_blocks as f64
                + config.host_cache_hit_weight * host_pinned_extension_blocks as f64
                + config.disk_cache_hit_weight * disk_extension_blocks as f64;

            WorkerOverlapScore {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                device_blocks,
                host_pinned_blocks,
                disk_blocks,
                host_pinned_extension_blocks,
                disk_extension_blocks,
                shared_beyond_device_blocks: None,
                router_credit_blocks,
            }
        })
        .collect();
    workers.sort_by_key(|worker| (worker.worker_id, worker.dp_rank));

    OverlapScoresResponse {
        block_size,
        num_blocks: tiered.device.overlap_scores.frequencies.len(),
        workers,
        shared_cache: SharedCacheOverlapScore {
            enabled: false,
            total_hit_blocks: 0,
            ranges: Vec::new(),
            error: None,
        },
    }
}
