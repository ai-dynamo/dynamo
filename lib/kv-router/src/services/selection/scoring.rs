// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use rustc_hash::FxHashMap;

use crate::indexer::TieredMatchDetails;
use crate::protocols::{StorageTier, WorkerId, WorkerWithDpRank};
use crate::scheduling::TierOverlapBlocks;
use crate::scheduling::config::RouterConfigOverride;
use crate::services::overlap::MooncakeOverlapSummary;

use super::types::{OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore};

#[derive(Default)]
pub(super) struct OverlapInputs {
    pub(super) tier_overlap_blocks: TierOverlapBlocks,
    pub(super) effective_overlap_blocks: FxHashMap<WorkerWithDpRank, f64>,
    pub(super) effective_cached_tokens: FxHashMap<WorkerWithDpRank, usize>,
    pub(super) mooncake_summaries: FxHashMap<WorkerId, MooncakeOverlapSummary>,
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
