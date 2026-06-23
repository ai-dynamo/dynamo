// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::KvRouterConfig;
use crate::indexer::TieredMatchDetails;
use crate::protocols::{StorageTier, WorkerWithDpRank};
use rustc_hash::FxHashMap;

use super::TierOverlapBlocks;

#[derive(Debug, Clone, Default)]
pub struct CacheHitEstimates {
    pub effective_overlap_blocks: FxHashMap<WorkerWithDpRank, f64>,
    pub cached_tokens: FxHashMap<WorkerWithDpRank, usize>,
}

pub fn cache_hit_estimates_from_tiered_matches(
    config: &KvRouterConfig,
    block_size: u32,
    tiered_matches: &TieredMatchDetails,
) -> CacheHitEstimates {
    let mut effective_overlap_blocks = FxHashMap::default();

    for (worker, overlap) in &tiered_matches.device.overlap_scores.scores {
        effective_overlap_blocks.insert(*worker, *overlap as f64);
    }

    for (storage_tier, tier_matches) in &tiered_matches.lower_tier {
        let weight = cache_hit_weight_for_tier(config, *storage_tier);
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
                (*overlap * block_size as f64).round().max(0.0) as usize,
            )
        })
        .collect();

    CacheHitEstimates {
        effective_overlap_blocks,
        cached_tokens,
    }
}

pub fn tier_overlap_blocks_from_tiered_matches(
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

fn cache_hit_weight_for_tier(config: &KvRouterConfig, tier: StorageTier) -> f64 {
    match tier {
        StorageTier::Device => 1.0,
        StorageTier::HostPinned => config.host_cache_hit_weight,
        StorageTier::Disk | StorageTier::External => config.disk_cache_hit_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::indexer::{LowerTierMatchDetails, MatchDetails};
    use crate::protocols::{OverlapScores, WorkerWithDpRank};

    #[test]
    fn converts_weighted_and_raw_tier_overlap_once() {
        let worker = WorkerWithDpRank::new(7, 1);
        let mut device = OverlapScores::new();
        device.scores.insert(worker, 2);
        let mut host = LowerTierMatchDetails::default();
        host.hits.insert(worker, 3);
        let mut disk = LowerTierMatchDetails::default();
        disk.hits.insert(worker, 4);
        let mut external = LowerTierMatchDetails::default();
        external.hits.insert(worker, 5);
        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: device,
                last_matched_hashes: Default::default(),
            },
            lower_tier: HashMap::from([
                (StorageTier::HostPinned, host),
                (StorageTier::Disk, disk),
                (StorageTier::External, external),
            ]),
        };
        let config = KvRouterConfig {
            host_cache_hit_weight: 0.5,
            disk_cache_hit_weight: 0.25,
            ..Default::default()
        };

        let estimates = cache_hit_estimates_from_tiered_matches(&config, 16, &tiered);
        let tiers = tier_overlap_blocks_from_tiered_matches(&tiered);

        assert_eq!(estimates.effective_overlap_blocks[&worker], 5.75);
        assert_eq!(estimates.cached_tokens[&worker], 92);
        assert_eq!(tiers.device[&worker], 2);
        assert_eq!(tiers.host_pinned[&worker], 3);
        assert_eq!(tiers.disk[&worker], 9);
    }
}
