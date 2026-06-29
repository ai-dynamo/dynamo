// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::config::{KvRouterConfig, RouterConfigOverride};
use crate::indexer::TieredMatchDetails;
use crate::protocols::{
    DpRank, SharedCacheHits, StorageTier, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use rustc_hash::FxHashMap;
use serde::Serialize;

use super::TierOverlapBlocks;

#[derive(Debug, Clone, Default)]
pub struct CacheHitEstimates {
    pub effective_overlap_blocks: FxHashMap<WorkerWithDpRank, f64>,
    pub cached_tokens: FxHashMap<WorkerWithDpRank, usize>,
}

/// Compact overlap state retained while a request waits for scheduling.
#[derive(Debug, Clone, Default)]
pub struct OverlapSignals {
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
}

impl OverlapSignals {
    pub fn selected_worker_tiers<C: WorkerConfigLike>(
        &self,
        worker: WorkerWithDpRank,
        config: &C,
    ) -> SelectedWorkerTierSnapshot {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        let mut dp_device_blocks = Vec::new();
        let mut gpu_blocks = 0;
        let mut host_pinned_blocks = 0;
        let mut disk_blocks = 0;

        for dp_rank in start..end {
            let rank = WorkerWithDpRank::new(worker.worker_id, dp_rank);
            let device = saturating_u32(
                self.tier_overlap_blocks
                    .device
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            );
            let host = device.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .host_pinned
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            ));
            let disk = host.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .disk
                    .get(&rank)
                    .copied()
                    .unwrap_or(0),
            ));

            dp_device_blocks.push((dp_rank, device));
            gpu_blocks = gpu_blocks.max(device);
            host_pinned_blocks = host_pinned_blocks.max(host);
            disk_blocks = disk_blocks.max(disk);
        }

        if dp_device_blocks.is_empty() {
            let device = saturating_u32(
                self.tier_overlap_blocks
                    .device
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            );
            let host = device.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .host_pinned
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            ));
            let disk = host.saturating_add(saturating_u32(
                self.tier_overlap_blocks
                    .disk
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
            ));
            dp_device_blocks.push((worker.dp_rank, device));
            gpu_blocks = device;
            host_pinned_blocks = host;
            disk_blocks = disk;
        }

        SelectedWorkerTierSnapshot {
            dp_device_blocks,
            gpu_blocks,
            host_pinned_blocks,
            disk_blocks,
        }
    }
}

/// Raw selected-worker overlap from the exact inputs used for final scheduling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectedWorkerTierSnapshot {
    pub dp_device_blocks: Vec<(DpRank, u32)>,
    pub gpu_blocks: u32,
    pub host_pinned_blocks: u32,
    pub disk_blocks: u32,
}

/// Borrowed analysis helper. It never owns indexer details or crosses an await.
pub struct OverlapAnalysis<'a> {
    config: &'a KvRouterConfig,
    block_size: u32,
    tiered: &'a TieredMatchDetails,
}

impl<'a> OverlapAnalysis<'a> {
    pub fn new(
        config: &'a KvRouterConfig,
        block_size: u32,
        tiered: &'a TieredMatchDetails,
    ) -> Self {
        Self {
            config,
            block_size,
            tiered,
        }
    }

    pub fn signals(&self) -> OverlapSignals {
        let estimates =
            cache_hit_estimates_from_tiered_matches(self.config, self.block_size, self.tiered);
        OverlapSignals {
            tier_overlap_blocks: tier_overlap_blocks_from_tiered_matches(self.tiered),
            effective_overlap_blocks: estimates.effective_overlap_blocks.into_iter().collect(),
            effective_cached_tokens: estimates.cached_tokens.into_iter().collect(),
        }
    }

    pub fn scores_response(
        &self,
        config_override: Option<&RouterConfigOverride>,
        num_blocks: usize,
        expected_workers: impl IntoIterator<Item = WorkerWithDpRank>,
        shared_cache_enabled: bool,
        shared_cache_hits: Option<&SharedCacheHits>,
        shared_cache_error: Option<String>,
    ) -> OverlapScoresResponse {
        build_overlap_scores_response(
            self.config,
            config_override,
            self.tiered,
            self.block_size,
            num_blocks,
            expected_workers,
            shared_cache_enabled,
            shared_cache_hits,
            shared_cache_error,
        )
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

#[expect(clippy::too_many_arguments)]
pub fn build_overlap_scores_response(
    config: &KvRouterConfig,
    config_override: Option<&RouterConfigOverride>,
    tiered: &TieredMatchDetails,
    block_size: u32,
    num_blocks: usize,
    expected_workers: impl IntoIterator<Item = WorkerWithDpRank>,
    shared_cache_enabled: bool,
    shared_cache_hits: Option<&SharedCacheHits>,
    shared_cache_error: Option<String>,
) -> OverlapScoresResponse {
    let mut all_workers: HashSet<_> = expected_workers.into_iter().collect();
    all_workers.extend(tiered.device.overlap_scores.scores.keys().copied());
    for matches in tiered.lower_tier.values() {
        all_workers.extend(matches.hits.keys().copied());
    }

    let host = tiered.lower_tier.get(&StorageTier::HostPinned);
    let disk = tiered.lower_tier.get(&StorageTier::Disk);
    let external = tiered.lower_tier.get(&StorageTier::External);
    let overlap_score_credit = config_override
        .and_then(|cfg| cfg.overlap_score_credit)
        .unwrap_or(config.overlap_score_credit);
    let shared_cache_multiplier = config_override
        .and_then(|cfg| cfg.shared_cache_multiplier)
        .unwrap_or(config.shared_cache_multiplier);

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
            let shared_beyond_device_blocks =
                shared_cache_hits.map(|hits| hits.hits_beyond(device_blocks as u32));
            let shared_credit_blocks =
                shared_beyond_device_blocks.unwrap_or(0) as f64 * shared_cache_multiplier;
            let router_credit_blocks = overlap_score_credit * device_blocks as f64
                + config.host_cache_hit_weight * host_pinned_extension_blocks as f64
                + config.disk_cache_hit_weight * disk_extension_blocks as f64
                + shared_credit_blocks;

            WorkerOverlapScore {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                device_blocks,
                host_pinned_blocks,
                disk_blocks,
                host_pinned_extension_blocks,
                disk_extension_blocks,
                shared_beyond_device_blocks,
                router_credit_blocks,
            }
        })
        .collect();
    workers.sort_by_key(|worker| (worker.worker_id, worker.dp_rank));

    let shared_cache = match shared_cache_hits {
        Some(hits) => SharedCacheOverlapScore {
            enabled: shared_cache_enabled,
            total_hit_blocks: hits.total_hits,
            ranges: hits
                .ranges
                .iter()
                .map(|range| (range.start, range.end))
                .collect(),
            error: shared_cache_error,
        },
        None => SharedCacheOverlapScore {
            enabled: shared_cache_enabled,
            total_hit_blocks: 0,
            ranges: Vec::new(),
            error: shared_cache_error,
        },
    };

    OverlapScoresResponse {
        block_size,
        num_blocks,
        workers,
        shared_cache,
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
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
    use crate::protocols::{OverlapScores, SharedCacheHits, WorkerWithDpRank};
    use crate::test_utils::SimpleWorkerConfig;

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

    #[test]
    fn score_response_uses_explicit_block_count_and_shared_credit() {
        let warm = WorkerWithDpRank::new(7, 1);
        let idle = WorkerWithDpRank::new(3, 0);
        let mut device = OverlapScores::new();
        device.scores.insert(warm, 2);
        let mut host = LowerTierMatchDetails::default();
        host.hits.insert(warm, 1);
        let mut external = LowerTierMatchDetails::default();
        external.hits.insert(warm, 2);
        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: device,
                ..Default::default()
            },
            lower_tier: HashMap::from([
                (StorageTier::HostPinned, host),
                (StorageTier::External, external),
            ]),
        };
        let config = KvRouterConfig {
            overlap_score_credit: 0.5,
            host_cache_hit_weight: 0.25,
            disk_cache_hit_weight: 0.1,
            shared_cache_multiplier: 0.5,
            ..Default::default()
        };
        #[allow(clippy::single_range_in_vec_init)]
        let shared = SharedCacheHits::from_ranges(vec![0..4]);

        let response = OverlapAnalysis::new(&config, 16, &tiered).scores_response(
            None,
            9,
            [warm, idle],
            true,
            Some(&shared),
            None,
        );

        assert_eq!(response.num_blocks, 9);
        assert_eq!(response.workers[0].worker_id, idle.worker_id);
        assert_eq!(response.workers[1].worker_id, warm.worker_id);
        let warm_score = &response.workers[1];
        assert_eq!(warm_score.host_pinned_blocks, 3);
        assert_eq!(warm_score.disk_blocks, 5);
        assert_eq!(warm_score.shared_beyond_device_blocks, Some(2));
        assert!((warm_score.router_credit_blocks - 2.45).abs() < f64::EPSILON);
        assert!(response.shared_cache.enabled);
        assert_eq!(response.shared_cache.total_hit_blocks, 4);
    }

    #[test]
    fn selected_worker_snapshot_includes_zero_ranks_and_saturates() {
        let worker_id = 9;
        let rank_2 = WorkerWithDpRank::new(worker_id, 2);
        let config = SimpleWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..Default::default()
        };
        let mut signals = OverlapSignals::default();
        signals
            .tier_overlap_blocks
            .device
            .insert(rank_2, usize::MAX);
        signals.tier_overlap_blocks.host_pinned.insert(rank_2, 1);
        signals.tier_overlap_blocks.disk.insert(rank_2, 1);

        let snapshot = signals.selected_worker_tiers(rank_2, &config);

        assert_eq!(snapshot.dp_device_blocks, vec![(2, u32::MAX), (3, 0)]);
        assert_eq!(snapshot.gpu_blocks, u32::MAX);
        assert_eq!(snapshot.host_pinned_blocks, u32::MAX);
        assert_eq!(snapshot.disk_blocks, u32::MAX);
    }
}
