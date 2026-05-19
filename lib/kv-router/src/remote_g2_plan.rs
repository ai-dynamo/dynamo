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
    /// Position in the request's prefix where `block_hashes[0]` lives.
    /// Equals the source worker's device-tier match count at plan time.
    /// The target's connector uses this to verify alignment with its own
    /// `num_computed_tokens` before attaching descriptors.
    pub start_block_index: u32,
    pub planned_prefix_blocks: u32,
    pub block_size_tokens: u32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub plan_version: u32,
    /// Parallel to `block_hashes`, carrying each block's source-side
    /// KV-cache-manager hash (TRT-LLM splitmix). The TRT-LLM source side
    /// uses these values to look up blocks via `find_block_by_hash`;
    /// `block_hashes` (XXH3 tokens hash) remains the plan's identity.
    /// Empty when the producer has not been updated to populate the new
    /// field — TRT-LLM's source falls back to using `block_hashes` for
    /// the lookup (legacy behavior).
    #[serde(default)]
    pub kv_block_hashes: Vec<u64>,
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

    // The indexer returns `hits` as a chained continuation count: the
    // HostPinned chain for this source extends from the position where its
    // Device chain ended, not from position 0. Without offsetting, the plan
    // would reference request.block_hashes[..hits] (positions 0..hits), but
    // the actual HostPinned matches on the source cover positions
    // [device_match, device_match + hits). The wrong hashes would land in
    // the plan, the source's resolve_for_request would fail to find them,
    // and the plan would be silently useless.
    let device_match = input
        .tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&source)
        .copied()
        .unwrap_or(0) as usize;

    let request_blocks = input.block_hashes.len();
    let start = device_match.min(request_blocks);
    let available_after_device = request_blocks.saturating_sub(start);
    let planned_prefix_blocks = (hits as usize).min(available_after_device) as u32;
    if planned_prefix_blocks == 0 {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
            stats,
        };
    }
    let end = start + planned_prefix_blocks as usize;

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
            block_hashes: input.block_hashes[start..end].to_vec(),
            start_block_index: start as u32,
            planned_prefix_blocks,
            block_size_tokens: input.block_size_tokens,
            created_at_ms: input.created_at_ms,
            expires_at_ms: input.expires_at_ms,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            // Caller fills this in post-selection by walking the indexer for
            // the chosen source. Left empty here so the planner stays a pure
            // function of `tiered_matches` and does not depend on the indexer.
            kv_block_hashes: Vec::new(),
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
            start_block_index: 0,
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            kv_block_hashes: vec![],
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
    fn remote_kv_reuse_plan_round_trips_kv_block_hashes() {
        // Populated kv_block_hashes must appear in the JSON and survive
        // a serialize → deserialize round trip with the exact same values.
        let mut plan = test_plan();
        plan.kv_block_hashes = vec![
            0xAAAA_AAAA_AAAA_AAAA,
            0xBBBB_BBBB_BBBB_BBBB,
            0xCCCC_CCCC_CCCC_CCCC,
        ];
        let json = serde_json::to_string(&plan).unwrap();
        assert!(
            json.contains("\"kv_block_hashes\""),
            "serialized plan missing kv_block_hashes field: {json}"
        );
        // Big values must serialize as integers, not stringified
        assert!(json.contains("12297829382473034410"));
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.kv_block_hashes, plan.kv_block_hashes);
    }

    #[test]
    fn remote_kv_reuse_plan_accepts_legacy_payload_without_kv_block_hashes() {
        // A producer that has not been updated to populate kv_block_hashes
        // emits the field-less JSON; it must still deserialize, with the
        // new field defaulting to empty.
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        // Strip the kv_block_hashes field from the JSON to simulate a
        // legacy producer.
        let legacy = json.replace(",\"kv_block_hashes\":[]", "");
        assert!(
            !legacy.contains("kv_block_hashes"),
            "legacy payload should not contain kv_block_hashes"
        );
        let decoded: RemoteKvReusePlan = serde_json::from_str(&legacy).unwrap();
        assert!(decoded.kv_block_hashes.is_empty());
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

    #[test]
    fn plan_start_block_index_is_zero_when_source_has_no_device_match() {
        // Source A has 0 device-tier matches and 3 HostPinned hits → plan
        // covers request positions [0, 3) and start_block_index == 0.
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 3);
                assert_eq!(plan.block_hashes, hashes[..3].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn plan_start_block_index_equals_source_device_match() {
        // Source A has 2 device-tier matches and 2 HostPinned hits chained
        // past them → plan covers request positions [2, 4) and
        // start_block_index == 2 (skip past A's device chain).
        let hashes = block_hashes(6);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 2)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 2);
                assert_eq!(plan.planned_prefix_blocks, 2);
                assert_eq!(plan.block_hashes, hashes[2..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }
}
