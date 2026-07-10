// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-generated hints that are attached to selected backend requests.

use serde::{Deserialize, Serialize};

use crate::{
    indexer::TieredMatchDetails,
    protocols::{ExternalSequenceBlockHash, LocalBlockHash, StorageTier, WorkerWithDpRank},
};

/// Extra-args key for router-generated backend hints.
pub const ROUTER_HINT_EXTRA_ARGS_KEY: &str = "router_hint";

/// Worker runtime_data key. Boolean true means the worker can consume router_hint extra args.
pub const ROUTER_HINT_RUNTIME_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime_data key for the advertised KVCC control endpoint.
pub const ROUTER_HINT_SOURCE_CONTROL_ENDPOINT_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoint";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterHint {
    pub request_id: String,
    pub source_control_endpoint: String,
    pub kv_block_hashes: Vec<ExternalSequenceBlockHash>,
    /// Position in the request's prefix where `kv_block_hashes[0]` lives.
    pub start_block_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterHintSource {
    pub source: WorkerWithDpRank,
    /// Position where the source's host-pinned chain starts.
    pub source_start_block_index: u32,
    /// Position where the target should start consuming the hint.
    pub start_block_index: u32,
    pub planned_prefix_blocks: u32,
}

pub struct RouterHintSourceSelectionInput<'a> {
    pub target: WorkerWithDpRank,
    pub local_block_hashes: &'a [LocalBlockHash],
    pub tiered_matches: &'a TieredMatchDetails,
}

pub fn select_router_hint_source(
    input: RouterHintSourceSelectionInput<'_>,
) -> Option<RouterHintSource> {
    let host_pinned_matches = input
        .tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)?;
    let request_blocks = input.local_block_hashes.len();
    if request_blocks == 0 {
        return None;
    }

    let target_device_match = input
        .tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&input.target)
        .copied()
        .unwrap_or(0) as usize;

    let mut best = None;
    for (&source, &hits) in &host_pinned_matches.hits {
        if source == input.target || hits == 0 {
            continue;
        }

        let source_start = host_pinned_matches
            .next_continuations
            .get(&source)
            .map(|continuation| continuation.start_pos.saturating_sub(hits))
            .unwrap_or(0)
            .min(request_blocks);
        let source_end = source_start.saturating_add(hits).min(request_blocks);
        let start = target_device_match.max(source_start).min(request_blocks);
        let planned = source_end.saturating_sub(start);
        if planned == 0 {
            continue;
        }

        let candidate = RouterHintSource {
            source,
            source_start_block_index: source_start as u32,
            start_block_index: start as u32,
            planned_prefix_blocks: planned as u32,
        };
        if best.is_none_or(|current: RouterHintSource| {
            candidate.planned_prefix_blocks > current.planned_prefix_blocks
                || (candidate.planned_prefix_blocks == current.planned_prefix_blocks
                    && candidate.source < current.source)
        }) {
            best = Some(candidate);
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        indexer::{LowerTierContinuation, LowerTierMatchDetails, MatchDetails, TieredMatchDetails},
        protocols::{
            ExternalSequenceBlockHash, LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank,
        },
        router_hint::{
            RouterHintSource, RouterHintSourceSelectionInput, select_router_hint_source,
        },
    };

    fn hashes(count: u64) -> Vec<LocalBlockHash> {
        (0..count).map(LocalBlockHash).collect()
    }

    fn tiered(
        device_hits: &[(WorkerWithDpRank, u32)],
        host_hits: &[(WorkerWithDpRank, usize)],
    ) -> TieredMatchDetails {
        let mut device = MatchDetails {
            overlap_scores: OverlapScores::new(),
            ..Default::default()
        };
        for &(worker, blocks) in device_hits {
            device.overlap_scores.scores.insert(worker, blocks);
            device
                .last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(blocks as u64));
        }

        let mut host = LowerTierMatchDetails::default();
        for &(worker, hits) in host_hits {
            host.hits.insert(worker, hits);
            let source_start = device_hits
                .iter()
                .find_map(|&(candidate, blocks)| (candidate == worker).then_some(blocks as usize))
                .unwrap_or(0);
            host.next_continuations.insert(
                worker,
                LowerTierContinuation {
                    start_pos: source_start + hits,
                    last_matched_hash: Some(ExternalSequenceBlockHash(100 + hits as u64)),
                },
            );
        }

        TieredMatchDetails {
            device,
            lower_tier: HashMap::from([(StorageTier::HostPinned, host)]),
        }
    }

    #[test]
    fn selects_longest_remote_host_pinned_source() {
        let target = WorkerWithDpRank::new(9, 0);
        let source_a = WorkerWithDpRank::new(7, 0);
        let source_b = WorkerWithDpRank::new(8, 0);
        let tiered = tiered(&[], &[(source_a, 2), (source_b, 3)]);

        let selected = select_router_hint_source(RouterHintSourceSelectionInput {
            target,
            local_block_hashes: &hashes(4),
            tiered_matches: &tiered,
        });

        assert_eq!(
            selected,
            Some(RouterHintSource {
                source: source_b,
                source_start_block_index: 0,
                start_block_index: 0,
                planned_prefix_blocks: 3,
            })
        );
    }

    #[test]
    fn skips_target_and_tie_breaks_by_source_identity() {
        let target = WorkerWithDpRank::new(9, 0);
        let source_a = WorkerWithDpRank::new(7, 1);
        let source_b = WorkerWithDpRank::new(7, 0);
        let tiered = tiered(&[], &[(target, 4), (source_a, 3), (source_b, 3)]);

        let selected = select_router_hint_source(RouterHintSourceSelectionInput {
            target,
            local_block_hashes: &hashes(4),
            tiered_matches: &tiered,
        });

        assert_eq!(selected.map(|source| source.source), Some(source_b));
    }

    #[test]
    fn starts_after_target_device_match() {
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let tiered = tiered(&[(target, 2)], &[(source, 4)]);

        let selected = select_router_hint_source(RouterHintSourceSelectionInput {
            target,
            local_block_hashes: &hashes(5),
            tiered_matches: &tiered,
        });

        assert_eq!(
            selected,
            Some(RouterHintSource {
                source,
                source_start_block_index: 0,
                start_block_index: 2,
                planned_prefix_blocks: 2,
            })
        );
    }

    #[test]
    fn returns_none_without_useful_remote_blocks() {
        let target = WorkerWithDpRank::new(9, 0);
        let tiered = tiered(&[], &[(target, 4)]);

        assert!(
            select_router_hint_source(RouterHintSourceSelectionInput {
                target,
                local_block_hashes: &hashes(4),
                tiered_matches: &tiered,
            })
            .is_none()
        );
    }
}
