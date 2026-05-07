// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{DpRank, LocalBlockHash, StorageTier, WorkerId, WorkerWithDpRank};

pub const REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_plan";
pub const REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY: &str =
    "remote_kv_reuse_no_plan_reason";
pub const REMOTE_KV_REUSE_PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteKvReusePlan {
    pub plan_id: String,
    pub request_id: String,
    pub target_worker_id: WorkerId,
    pub target_dp_rank: DpRank,
    pub source_worker_id: WorkerId,
    pub source_dp_rank: DpRank,
    pub source_tier: StorageTier,
    pub block_hashes: Vec<LocalBlockHash>,
    pub planned_prefix_blocks: u32,
    pub block_size_tokens: u32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub plan_version: u32,
}

// Compatibility identity is intentionally deferred in v1; source resolve remains authoritative.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteKvReuseNoPlanReason {
    Disabled,
    NoRemoteG2Candidate,
    NoContiguousPrefix,
    SourceIsTarget,
    IncompatibleBlockSize,
    PlanExpired,
    SerializationFailed,
}

impl RemoteKvReuseNoPlanReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoRemoteG2Candidate => "no_remote_g2_candidate",
            Self::NoContiguousPrefix => "no_contiguous_prefix",
            Self::SourceIsTarget => "source_is_target",
            Self::IncompatibleBlockSize => "incompatible_block_size",
            Self::PlanExpired => "plan_expired",
            Self::SerializationFailed => "serialization_failed",
        }
    }
}

pub struct RemoteKvReuseSelectionInput<'a> {
    pub request_id: &'a str,
    pub target: WorkerWithDpRank,
    pub block_hashes: &'a [LocalBlockHash],
    pub block_size_tokens: u32,
    pub tiered_matches: &'a TieredMatchDetails,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteKvReuseSelectionStats {
    pub rejected_g1_candidates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteKvReuseDecision {
    Plan {
        plan: RemoteKvReusePlan,
        stats: RemoteKvReuseSelectionStats,
    },
    NoPlan {
        reason: RemoteKvReuseNoPlanReason,
        stats: RemoteKvReuseSelectionStats,
    },
}

pub fn select_remote_g2_reuse_plan(
    input: RemoteKvReuseSelectionInput<'_>,
) -> RemoteKvReuseDecision {
    let stats = RemoteKvReuseSelectionStats {
        rejected_g1_candidates: input
            .tiered_matches
            .device
            .overlap_scores
            .scores
            .values()
            .filter(|&&overlap| overlap > 0)
            .count() as u32,
    };

    let Some(host_pinned_matches) = input
        .tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
    else {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            stats,
        };
    };

    let mut saw_remote_candidate = false;
    let mut best: Option<(WorkerWithDpRank, usize)> = None;
    for (&worker, &hits) in &host_pinned_matches.hits {
        if worker == input.target {
            continue;
        }
        saw_remote_candidate = true;
        if hits == 0 {
            continue;
        }
        match best {
            None => best = Some((worker, hits)),
            Some((best_worker, best_hits))
                if hits > best_hits || (hits == best_hits && worker < best_worker) =>
            {
                best = Some((worker, hits));
            }
            Some(_) => {}
        }
    }

    let Some((source, hits)) = best else {
        return RemoteKvReuseDecision::NoPlan {
            reason: if saw_remote_candidate {
                RemoteKvReuseNoPlanReason::NoContiguousPrefix
            } else {
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate
            },
            stats,
        };
    };

    let planned_prefix_blocks = hits.min(input.block_hashes.len()) as u32;
    if planned_prefix_blocks == 0 {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
            stats,
        };
    }

    RemoteKvReuseDecision::Plan {
        plan: RemoteKvReusePlan {
            plan_id: format!(
                "remote-g2:{}:{}:{}:{}",
                input.request_id, source.worker_id, source.dp_rank, input.created_at_ms
            ),
            request_id: input.request_id.to_string(),
            target_worker_id: input.target.worker_id,
            target_dp_rank: input.target.dp_rank,
            source_worker_id: source.worker_id,
            source_dp_rank: source.dp_rank,
            source_tier: StorageTier::HostPinned,
            block_hashes: input.block_hashes[..planned_prefix_blocks as usize].to_vec(),
            planned_prefix_blocks,
            block_size_tokens: input.block_size_tokens,
            created_at_ms: input.created_at_ms,
            expires_at_ms: input.expires_at_ms,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
        },
        stats,
    }
}

#[cfg(test)]
mod tests {
    use crate::indexer::{LowerTierMatchDetails, MatchDetails, TieredMatchDetails};
    use crate::protocols::{LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank};
    use crate::remote_g2_plan::{
        REMOTE_KV_REUSE_PLAN_VERSION, RemoteKvReuseDecision, RemoteKvReuseNoPlanReason,
        RemoteKvReusePlan, RemoteKvReuseSelectionInput, select_remote_g2_reuse_plan,
    };

    fn test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            plan_id: "plan-1".to_string(),
            request_id: "request-1".to_string(),
            target_worker_id: 9,
            target_dp_rank: 0,
            source_worker_id: 7,
            source_dp_rank: 1,
            source_tier: StorageTier::HostPinned,
            block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
        }
    }

    fn block_hashes(count: u64) -> Vec<LocalBlockHash> {
        (0..count).map(LocalBlockHash).collect()
    }

    fn tiered_matches(
        device_hits: &[(WorkerWithDpRank, u32)],
        host_pinned_hits: &[(WorkerWithDpRank, usize)],
    ) -> TieredMatchDetails {
        let mut device = MatchDetails {
            overlap_scores: OverlapScores::new(),
            ..Default::default()
        };
        device
            .overlap_scores
            .scores
            .extend(device_hits.iter().copied());

        let mut lower_tier = std::collections::HashMap::new();
        let mut host_pinned = LowerTierMatchDetails::default();
        host_pinned.hits.extend(host_pinned_hits.iter().copied());
        lower_tier.insert(StorageTier::HostPinned, host_pinned);

        TieredMatchDetails { device, lower_tier }
    }

    fn selection_input<'a>(
        target: WorkerWithDpRank,
        block_hashes: &'a [LocalBlockHash],
        tiered_matches: &'a TieredMatchDetails,
    ) -> RemoteKvReuseSelectionInput<'a> {
        RemoteKvReuseSelectionInput {
            request_id: "request-1",
            target,
            block_hashes,
            block_size_tokens: 16,
            tiered_matches,
            created_at_ms: 1000,
            expires_at_ms: 2000,
        }
    }

    #[test]
    fn remote_kv_reuse_plan_round_trips_json() {
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn remote_kv_reuse_plan_serialization_has_no_forbidden_router_truth() {
        let json = serde_json::to_string(&test_plan()).unwrap();
        for forbidden in [
            "virtual_address",
            "physical_address",
            "nixl_descriptor",
            "descriptor",
            "target_g1_block_id",
            "source_block_id",
            "block_ptr",
            "handle",
        ] {
            assert!(
                !json.contains(forbidden),
                "serialized plan contains forbidden router truth: {forbidden}"
            );
        }
    }

    #[test]
    fn no_plan_reason_is_low_cardinality_snake_case() {
        let json = serde_json::to_string(&RemoteKvReuseNoPlanReason::NoRemoteG2Candidate).unwrap();
        assert_eq!(json, "\"no_remote_g2_candidate\"");
    }

    #[test]
    fn selects_longest_remote_g2_prefix() {
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(7, 0), 2),
                (WorkerWithDpRank::new(8, 0), 4),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 8);
                assert_eq!(plan.source_dp_rank, 0);
                assert_eq!(plan.planned_prefix_blocks, 4);
                assert_eq!(plan.block_hashes, hashes[..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g2_tie_break_is_stable_by_worker_then_rank() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(
            &[],
            &[
                (WorkerWithDpRank::new(8, 1), 3),
                (WorkerWithDpRank::new(7, 3), 3),
                (WorkerWithDpRank::new(7, 1), 3),
            ],
        );

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 1);
                assert_eq!(plan.planned_prefix_blocks, 3);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn remote_g1_device_hits_are_rejected_not_selected() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = TieredMatchDetails {
            device: {
                let mut device = MatchDetails {
                    overlap_scores: OverlapScores::new(),
                    ..Default::default()
                };
                device
                    .overlap_scores
                    .scores
                    .insert(WorkerWithDpRank::new(7, 0), 2);
                device
            },
            lower_tier: Default::default(),
        };

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert!(stats.rejected_g1_candidates > 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn source_selection_does_not_change_target() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.target_worker_id, 42);
                assert_eq!(plan.target_dp_rank, 2);
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.source_dp_rank, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn zero_host_pinned_hits_return_no_contiguous_prefix() {
        let hashes = block_hashes(2);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 0)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }
}
