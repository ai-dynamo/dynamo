// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::indexer::{LowerTierMatchDetails, TieredMatchDetails};
use crate::protocols::{StorageTier, WorkerId, WorkerWithDpRank};

/// Per-instance match summary aligned with Mooncake RFC #1403.
///
/// All counts are in matched tokens. CPU and disk counts are cumulative through
/// the preceding tiers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MooncakeOverlapSummary {
    pub longest_matched: u32,
    pub gpu: u32,
    pub dp: HashMap<String, u32>,
    pub cpu: u32,
    pub disk: u32,
}

pub(crate) fn build_mooncake_overlap_summaries(
    tiered: &TieredMatchDetails,
    block_size: u32,
    expected_workers: impl IntoIterator<Item = WorkerWithDpRank>,
) -> FxHashMap<WorkerId, MooncakeOverlapSummary> {
    let host = tiered.lower_tier.get(&StorageTier::HostPinned);
    let disk = tiered.lower_tier.get(&StorageTier::Disk);
    let external = tiered.lower_tier.get(&StorageTier::External);

    let mut all_workers: HashSet<_> = expected_workers.into_iter().collect();
    all_workers.extend(tiered.device.overlap_scores.scores.keys().copied());
    for matches in tiered.lower_tier.values() {
        all_workers.extend(matches.hits.keys().copied());
    }

    let mut summaries = FxHashMap::<WorkerId, MooncakeOverlapSummary>::default();
    for worker in all_workers {
        let gpu_blocks = tiered
            .device
            .overlap_scores
            .scores
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let cpu_blocks = gpu_blocks.saturating_add(extension_blocks(host, worker));
        let disk_blocks = cpu_blocks
            .saturating_add(extension_blocks(disk, worker))
            .saturating_add(extension_blocks(external, worker));

        let gpu_tokens = gpu_blocks.saturating_mul(block_size);
        let cpu_tokens = cpu_blocks.saturating_mul(block_size);
        let disk_tokens = disk_blocks.saturating_mul(block_size);
        let summary = summaries.entry(worker.worker_id).or_default();
        summary.dp.insert(worker.dp_rank.to_string(), gpu_tokens);
        summary.gpu = summary.gpu.max(gpu_tokens);
        summary.cpu = summary.cpu.max(cpu_tokens);
        summary.disk = summary.disk.max(disk_tokens);
    }

    for summary in summaries.values_mut() {
        summary.longest_matched = summary.gpu.max(summary.cpu).max(summary.disk);
    }
    summaries
}

fn extension_blocks(extension: Option<&LowerTierMatchDetails>, worker: WorkerWithDpRank) -> u32 {
    extension
        .and_then(|matches| matches.hits.get(&worker))
        .copied()
        .map(|blocks| u32::try_from(blocks).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{LowerTierMatchDetails, MatchDetails};
    use crate::protocols::OverlapScores;

    #[test]
    fn builds_cumulative_multi_rank_mooncake_summary() {
        let worker_0 = WorkerWithDpRank::new(7, 0);
        let worker_1 = WorkerWithDpRank::new(7, 1);
        let mut device = OverlapScores::new();
        device.scores.insert(worker_0, 2);
        device.scores.insert(worker_1, 1);
        let mut host = LowerTierMatchDetails::default();
        host.hits.insert(worker_0, 1);
        let mut disk = LowerTierMatchDetails::default();
        disk.hits.insert(worker_0, 2);
        disk.hits.insert(worker_1, 1);
        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: device,
                last_matched_hashes: Default::default(),
            },
            lower_tier: HashMap::from([(StorageTier::HostPinned, host), (StorageTier::Disk, disk)]),
        };

        let summaries = build_mooncake_overlap_summaries(&tiered, 16, [worker_0, worker_1]);
        let summary = summaries.get(&7).unwrap();

        assert_eq!(
            summary.dp,
            HashMap::from([("0".into(), 32), ("1".into(), 16)])
        );
        assert_eq!(summary.gpu, 32);
        assert_eq!(summary.cpu, 48);
        assert_eq!(summary.disk, 80);
        assert_eq!(summary.longest_matched, 80);
    }

    #[test]
    fn includes_expected_zero_overlap_rank() {
        let worker = WorkerWithDpRank::new(9, 3);
        let summaries =
            build_mooncake_overlap_summaries(&TieredMatchDetails::default(), 16, [worker]);

        assert_eq!(
            summaries.get(&9).unwrap().dp,
            HashMap::from([("3".into(), 0)])
        );
    }
}
