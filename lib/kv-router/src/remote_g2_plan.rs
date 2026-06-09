// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::indexer::TieredMatchDetails;
use crate::protocols::{DpRank, LocalBlockHash, StorageTier, WorkerId, WorkerWithDpRank};

pub const REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_plan";
pub const REMOTE_KV_REUSE_SOURCE_ROUTE_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_source_route";
pub const REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY: &str = "remote_kv_reuse_no_plan_reason";
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
    pub router_block_hashes: Vec<LocalBlockHash>,
    /// Position in the request's prefix where `router_block_hashes[0]` lives.
    /// Equals the source worker's device-tier match count at plan time.
    /// The target's connector uses this to verify alignment with its own
    /// `num_computed_tokens` before attaching descriptors.
    pub start_block_index: u32,
    pub planned_prefix_blocks: u32,
    pub block_size_tokens: u32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub plan_version: u32,
    /// Parallel to `router_block_hashes`, carrying each block's framework
    /// KV-event hash. The source framework uses these values to look up actual
    /// HostPinned blocks; `router_block_hashes` remains the plan identity.
    pub engine_block_hashes: Vec<u64>,
}

// Compatibility identity is intentionally deferred in v1; source resolve remains authoritative.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteKvReuseSourceRoute {
    pub source_worker_id: WorkerId,
    pub source_host: String,
    pub source_bootstrap_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteKvReuseNoPlanReason {
    Disabled,
    NoRemoteG2Candidate,
    NoContiguousPrefix,
    BelowRemoteG2Cost,
    SourceIsTarget,
    IncompatibleBlockSize,
    PlanExpired,
    NoSourceBootstrapEndpoint,
    LocalOverlapGapTooLarge,
    SerializationFailed,
}

impl RemoteKvReuseNoPlanReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoRemoteG2Candidate => "no_remote_g2_candidate",
            Self::NoContiguousPrefix => "no_contiguous_prefix",
            Self::BelowRemoteG2Cost => "below_remote_g2_cost",
            Self::SourceIsTarget => "source_is_target",
            Self::IncompatibleBlockSize => "incompatible_block_size",
            Self::PlanExpired => "plan_expired",
            Self::NoSourceBootstrapEndpoint => "no_source_bootstrap_endpoint",
            Self::LocalOverlapGapTooLarge => "local_overlap_gap_too_large",
            Self::SerializationFailed => "serialization_failed",
        }
    }
}

#[derive(Clone, Copy)]
pub struct RemoteKvReuseSelectionInput<'a> {
    pub request_id: &'a str,
    pub target: WorkerWithDpRank,
    pub target_local_prefix_blocks: u32,
    pub best_local_prefix_blocks: u32,
    pub block_hashes: &'a [LocalBlockHash],
    pub block_size_tokens: u32,
    pub tiered_matches: &'a TieredMatchDetails,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub cost_model: Option<RemoteG2CostModel>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteKvReuseSelectionStats {
    pub rejected_g1_candidates: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RemoteG2CostModel {
    pub score_weight: f64,
    pub cost_blocks: f64,
    pub cost_per_block: f64,
    pub max_planned_blocks: Option<u32>,
    pub max_local_overlap_gap_blocks: Option<u32>,
}

impl RemoteG2CostModel {
    pub fn capped_planned_blocks(&self, planned_blocks: u32) -> u32 {
        self.max_planned_blocks
            .map(|max_blocks| planned_blocks.min(max_blocks))
            .unwrap_or(planned_blocks)
    }

    pub fn estimated_cost_blocks(&self, planned_blocks: u32) -> f64 {
        self.cost_blocks + self.cost_per_block * planned_blocks as f64
    }

    pub fn score_blocks(&self, incremental_blocks: u32, planned_blocks: u32) -> f64 {
        self.score_weight * incremental_blocks as f64 - self.estimated_cost_blocks(planned_blocks)
    }

    pub fn allows_local_overlap_gap(
        &self,
        target_local_blocks: u32,
        best_local_blocks: u32,
    ) -> bool {
        self.max_local_overlap_gap_blocks
            .map(|max_gap| target_local_blocks >= best_local_blocks.saturating_sub(max_gap))
            .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RemoteG2CandidateScore {
    pub target: WorkerWithDpRank,
    pub source: WorkerWithDpRank,
    pub start_block_index: u32,
    pub planned_blocks: u32,
    pub incremental_blocks: u32,
    pub cost_blocks: f64,
    pub score_blocks: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RemoteG2CandidateDecision {
    Candidate {
        candidate: RemoteG2CandidateScore,
        stats: RemoteKvReuseSelectionStats,
    },
    NoCandidate {
        reason: RemoteKvReuseNoPlanReason,
        stats: RemoteKvReuseSelectionStats,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteKvReuseDecision {
    Plan {
        plan: RemoteKvReusePlan,
        source_route: Option<RemoteKvReuseSourceRoute>,
        stats: RemoteKvReuseSelectionStats,
    },
    NoPlan {
        reason: RemoteKvReuseNoPlanReason,
        stats: RemoteKvReuseSelectionStats,
    },
}

fn choose_better_candidate_score(
    best: &mut Option<RemoteG2CandidateScore>,
    worker: WorkerWithDpRank,
    start: usize,
    hits: usize,
    input: RemoteKvReuseSelectionInput<'_>,
) {
    let start_block_index = start as u32;
    let planned_blocks = input
        .cost_model
        .map(|model| model.capped_planned_blocks(hits as u32))
        .unwrap_or(hits as u32);
    if planned_blocks == 0 {
        return;
    }
    let incremental_blocks = remote_g2_incremental_blocks(
        start_block_index,
        planned_blocks,
        input.target_local_prefix_blocks,
    );
    if incremental_blocks == 0 {
        return;
    }
    let score_incremental_blocks = remote_g2_incremental_blocks(
        start_block_index,
        planned_blocks,
        input.best_local_prefix_blocks,
    );
    // The transfer must help the target, but its scheduler credit is only the
    // prefix it adds beyond the best local-only worker. This matches L3-style
    // prefetch scoring: local cache remains the baseline.
    let (cost_blocks, score_blocks) =
        remote_g2_score_for_candidate(input.cost_model, score_incremental_blocks, planned_blocks);
    let candidate = RemoteG2CandidateScore {
        target: input.target,
        source: worker,
        start_block_index,
        planned_blocks,
        incremental_blocks,
        cost_blocks,
        score_blocks,
    };

    match best {
        None => *best = Some(candidate),
        Some(best_candidate)
            if candidate.score_blocks > best_candidate.score_blocks
                || (candidate.score_blocks == best_candidate.score_blocks
                    && (candidate.planned_blocks > best_candidate.planned_blocks
                        || (candidate.planned_blocks == best_candidate.planned_blocks
                            && candidate.source < best_candidate.source))) =>
        {
            *best = Some(candidate);
        }
        Some(_) => {}
    }
}

fn remote_g2_selection_stats(tiered_matches: &TieredMatchDetails) -> RemoteKvReuseSelectionStats {
    RemoteKvReuseSelectionStats {
        rejected_g1_candidates: tiered_matches
            .device
            .overlap_scores
            .scores
            .values()
            .filter(|&&overlap| overlap > 0)
            .count() as u32,
    }
}

fn remote_g2_incremental_blocks(start: u32, planned_blocks: u32, target_local_blocks: u32) -> u32 {
    let end = start.saturating_add(planned_blocks);
    end.saturating_sub(target_local_blocks.max(start))
}

fn remote_g2_score_for_candidate(
    cost_model: Option<RemoteG2CostModel>,
    incremental_blocks: u32,
    planned_blocks: u32,
) -> (f64, f64) {
    match cost_model {
        Some(model) => (
            model.estimated_cost_blocks(planned_blocks),
            model.score_blocks(incremental_blocks, planned_blocks),
        ),
        None => (0.0, incremental_blocks as f64),
    }
}

pub fn select_remote_g2_candidate(
    input: RemoteKvReuseSelectionInput<'_>,
) -> RemoteG2CandidateDecision {
    let stats = remote_g2_selection_stats(input.tiered_matches);

    if input.cost_model.is_some_and(|model| {
        !model.allows_local_overlap_gap(
            input.target_local_prefix_blocks,
            input.best_local_prefix_blocks,
        )
    }) {
        return RemoteG2CandidateDecision::NoCandidate {
            reason: RemoteKvReuseNoPlanReason::LocalOverlapGapTooLarge,
            stats,
        };
    }

    let Some(host_pinned_matches) = input
        .tiered_matches
        .lower_tier
        .get(&StorageTier::HostPinned)
    else {
        return RemoteG2CandidateDecision::NoCandidate {
            reason: RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            stats,
        };
    };

    let request_blocks = input.block_hashes.len();
    let mut saw_remote_candidate = false;
    let mut best_continuation: Option<RemoteG2CandidateScore> = None;
    let mut best_root_fallback: Option<RemoteG2CandidateScore> = None;
    for (&worker, &host_continuation_hits) in &host_pinned_matches.hits {
        if worker == input.target {
            continue;
        }
        saw_remote_candidate = true;

        let device_match = input
            .tiered_matches
            .device
            .overlap_scores
            .scores
            .get(&worker)
            .copied()
            .unwrap_or(0) as usize;

        // Normal lower-tier semantics report HostPinned hits as a continuation
        // after the source worker's Device match. With write-through HiCache,
        // the same blocks are present in both GPU and CPU, so that continuation
        // can be zero even though a valid CPU-pinned chain exists from root.
        // Prefer real HostPinned continuations first; only fall back to a root
        // candidate if no positive HostPinned continuation exists. Otherwise a
        // zero-hit write-through fallback can beat a smaller but real write-back
        // HostPinned chain, only to fail the later source-side chain walk.
        let (start, hits) = if host_continuation_hits > 0 {
            (device_match.min(request_blocks), host_continuation_hits)
        } else if device_match > 0 {
            (0, device_match.min(request_blocks))
        } else {
            continue;
        };

        let hits = hits.min(request_blocks.saturating_sub(start));
        if hits == 0 {
            continue;
        }

        if host_continuation_hits > 0 {
            choose_better_candidate_score(&mut best_continuation, worker, start, hits, input);
        } else {
            choose_better_candidate_score(&mut best_root_fallback, worker, start, hits, input);
        }
    }

    let best = best_continuation.or(best_root_fallback);
    let Some(candidate) = best else {
        return RemoteG2CandidateDecision::NoCandidate {
            reason: if saw_remote_candidate {
                RemoteKvReuseNoPlanReason::NoContiguousPrefix
            } else {
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate
            },
            stats,
        };
    };

    if candidate.planned_blocks == 0 {
        return RemoteG2CandidateDecision::NoCandidate {
            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
            stats,
        };
    }
    if candidate.score_blocks <= 0.0 {
        return RemoteG2CandidateDecision::NoCandidate {
            reason: RemoteKvReuseNoPlanReason::BelowRemoteG2Cost,
            stats,
        };
    }

    RemoteG2CandidateDecision::Candidate { candidate, stats }
}

pub fn materialize_remote_g2_reuse_plan(
    input: RemoteKvReuseSelectionInput<'_>,
    candidate: RemoteG2CandidateScore,
) -> RemoteKvReuseDecision {
    let stats = remote_g2_selection_stats(input.tiered_matches);
    if candidate.target != input.target {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            stats,
        };
    }
    if candidate.source == input.target {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::SourceIsTarget,
            stats,
        };
    }
    if candidate.score_blocks <= 0.0 {
        return RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::BelowRemoteG2Cost,
            stats,
        };
    }
    let start = candidate.start_block_index as usize;
    let planned_prefix_blocks = candidate.planned_blocks;
    let end = start + planned_prefix_blocks as usize;

    RemoteKvReuseDecision::Plan {
        plan: RemoteKvReusePlan {
            plan_id: format!(
                "remote-g2:{}:{}:{}:{}",
                input.request_id,
                candidate.source.worker_id,
                candidate.source.dp_rank,
                input.created_at_ms
            ),
            request_id: input.request_id.to_string(),
            target_worker_id: input.target.worker_id,
            target_dp_rank: input.target.dp_rank,
            source_worker_id: candidate.source.worker_id,
            source_dp_rank: candidate.source.dp_rank,
            source_tier: StorageTier::HostPinned,
            router_block_hashes: input.block_hashes[start..end].to_vec(),
            start_block_index: candidate.start_block_index,
            planned_prefix_blocks,
            block_size_tokens: input.block_size_tokens,
            created_at_ms: input.created_at_ms,
            expires_at_ms: input.expires_at_ms,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            // Caller fills this in post-selection by walking the indexer for
            // the chosen source. Left empty here so the planner stays a pure
            // function of `tiered_matches` and does not depend on the indexer.
            engine_block_hashes: Vec::new(),
        },
        source_route: None,
        stats,
    }
}

pub fn select_remote_g2_reuse_plan(
    input: RemoteKvReuseSelectionInput<'_>,
) -> RemoteKvReuseDecision {
    match select_remote_g2_candidate(input) {
        RemoteG2CandidateDecision::Candidate { candidate, .. } => {
            materialize_remote_g2_reuse_plan(input, candidate)
        }
        RemoteG2CandidateDecision::NoCandidate { reason, stats } => {
            RemoteKvReuseDecision::NoPlan { reason, stats }
        }
    }
}

#[cfg(test)]
mod tests {
    // Test naming convention:
    // - `serde_*` covers the wire-format contract.
    // - `select_*` covers the selection algorithm.
    // - `scenario_*` covers full plan-shape behavior for concrete inputs.

    use crate::indexer::{LowerTierMatchDetails, MatchDetails, TieredMatchDetails};
    use crate::protocols::{LocalBlockHash, OverlapScores, StorageTier, WorkerWithDpRank};
    use crate::remote_g2_plan::{
        REMOTE_KV_REUSE_PLAN_VERSION, RemoteG2CandidateDecision, RemoteG2CostModel,
        RemoteKvReuseDecision, RemoteKvReuseNoPlanReason, RemoteKvReusePlan,
        RemoteKvReuseSelectionInput, select_remote_g2_candidate, select_remote_g2_reuse_plan,
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
            router_block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            start_block_index: 0,
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            engine_block_hashes: vec![],
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
            target_local_prefix_blocks: 0,
            best_local_prefix_blocks: 0,
            block_hashes,
            block_size_tokens: 16,
            tiered_matches,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            cost_model: None,
        }
    }

    #[test]
    fn serde_plan_round_trips_basic() {
        let plan = test_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn serde_plan_round_trips_engine_block_hashes() {
        // Populated engine_block_hashes must appear in the JSON and survive
        // a serialize → deserialize round trip with the exact same values.
        let mut plan = test_plan();
        plan.engine_block_hashes = vec![
            0xAAAA_AAAA_AAAA_AAAA,
            0xBBBB_BBBB_BBBB_BBBB,
            0xCCCC_CCCC_CCCC_CCCC,
        ];
        let json = serde_json::to_string(&plan).unwrap();
        assert!(
            json.contains("\"engine_block_hashes\""),
            "serialized plan missing engine_block_hashes field: {json}"
        );
        // Big values must serialize as integers, not stringified
        assert!(json.contains("12297829382473034410"));
        let decoded: RemoteKvReusePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.engine_block_hashes, plan.engine_block_hashes);
    }

    #[test]
    fn serde_plan_has_no_router_truth_fields() {
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
    fn serde_no_plan_reason_uses_snake_case() {
        let json = serde_json::to_string(&RemoteKvReuseNoPlanReason::NoRemoteG2Candidate).unwrap();
        assert_eq!(json, "\"no_remote_g2_candidate\"");
    }

    #[test]
    fn select_longest_remote_g2_prefix() {
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
                assert_eq!(plan.router_block_hashes, hashes[..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn select_tie_break_by_worker_then_rank() {
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
    fn select_rejects_g1_only_device_hits() {
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
    fn select_preserves_target_identity() {
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
    fn select_distinguishes_target_by_dp_rank() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(9, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.source_worker_id, 9);
                assert_eq!(plan.source_dp_rank, 1);
                assert_eq!(plan.planned_prefix_blocks, 3);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn select_plan_metadata_propagates_from_input() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(42, 2);
        let source = WorkerWithDpRank::new(7, 1);
        let matches = tiered_matches(&[], &[(source, 3)]);
        let input = RemoteKvReuseSelectionInput {
            request_id: "req-meta",
            target,
            target_local_prefix_blocks: 0,
            best_local_prefix_blocks: 0,
            block_hashes: &hashes,
            block_size_tokens: 32,
            tiered_matches: &matches,
            created_at_ms: 1234,
            expires_at_ms: 5678,
            cost_model: None,
        };

        let decision = select_remote_g2_reuse_plan(input);

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.request_id, "req-meta");
                assert_eq!(plan.block_size_tokens, 32);
                assert_eq!(plan.created_at_ms, 1234);
                assert_eq!(plan.expires_at_ms, 5678);
                assert_eq!(plan.plan_version, REMOTE_KV_REUSE_PLAN_VERSION);
                assert_eq!(plan.source_tier, StorageTier::HostPinned);
                assert!(plan.engine_block_hashes.is_empty());
                assert_eq!(plan.plan_id, "remote-g2:req-meta:7:1:1234");
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_zero_host_pinned_hits_no_contiguous_prefix() {
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
    fn scenario_empty_request_no_contiguous_prefix() {
        let hashes: Vec<LocalBlockHash> = Vec::new();
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(WorkerWithDpRank::new(7, 0), 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_partial_g2_no_g1_matched_prefix_only() {
        // Source A has 0 device-tier matches and 3 HostPinned hits → plan
        // covers request positions [0, 3) and start_block_index == 0.
        let hashes = block_hashes(5);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, stats, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 3);
                assert_eq!(plan.router_block_hashes, hashes[..3].to_vec());
                assert!(plan.planned_prefix_blocks < hashes.len() as u32);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_g1_partial_g2_tail_start_at_device_match() {
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
                assert_eq!(plan.router_block_hashes, hashes[2..4].to_vec());
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_zero_overlap_no_remote_g2_candidate() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_target_only_host_pinned_no_remote_g2_candidate() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = tiered_matches(&[], &[(target, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_no_host_pinned_tier_no_remote_g2_candidate() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let matches = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: OverlapScores::new(),
                ..Default::default()
            },
            lower_tier: Default::default(),
        };

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoRemoteG2Candidate);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }

    #[test]
    fn scenario_full_g2_no_g1_full_coverage_plan() {
        let hashes = block_hashes(3);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 3)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::Plan { plan, stats, .. } => {
                assert_eq!(plan.source_worker_id, 7);
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, hashes.len() as u32);
                assert_eq!(plan.router_block_hashes, hashes);
                assert_eq!(stats.rejected_g1_candidates, 0);
            }
            other => panic!("expected plan, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_scores_only_incremental_blocks_beyond_target_local_prefix() {
        let hashes = block_hashes(8);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 8)]);
        let input = RemoteKvReuseSelectionInput {
            target_local_prefix_blocks: 3,
            best_local_prefix_blocks: 3,
            cost_model: Some(RemoteG2CostModel {
                score_weight: 0.5,
                cost_blocks: 1.0,
                cost_per_block: 0.0,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.planned_blocks, 8);
                assert_eq!(candidate.incremental_blocks, 5);
                assert_eq!(candidate.cost_blocks, 1.0);
                assert_eq!(candidate.score_blocks, 1.5);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_charges_per_planned_block_transfer_cost() {
        let hashes = block_hashes(8);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 8)]);
        let input = RemoteKvReuseSelectionInput {
            target_local_prefix_blocks: 3,
            best_local_prefix_blocks: 3,
            cost_model: Some(RemoteG2CostModel {
                score_weight: 0.5,
                cost_blocks: 1.0,
                cost_per_block: 0.1,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.planned_blocks, 8);
                assert_eq!(candidate.incremental_blocks, 5);
                assert!((candidate.cost_blocks - 1.8).abs() < f64::EPSILON);
                assert!((candidate.score_blocks - 0.7).abs() < f64::EPSILON);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_scores_remote_g2_against_best_local_prefix() {
        let hashes = block_hashes(8);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 8)]);
        let input = RemoteKvReuseSelectionInput {
            target_local_prefix_blocks: 1,
            best_local_prefix_blocks: 6,
            cost_model: Some(RemoteG2CostModel {
                score_weight: 1.0,
                cost_blocks: 0.0,
                cost_per_block: 0.0,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::Candidate { candidate, .. } => {
                assert_eq!(candidate.planned_blocks, 8);
                assert_eq!(candidate.incremental_blocks, 7);
                assert_eq!(candidate.score_blocks, 2.0);
            }
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_rejects_when_best_local_already_covers_remote_prefix() {
        let hashes = block_hashes(8);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 8)]);
        let input = RemoteKvReuseSelectionInput {
            target_local_prefix_blocks: 1,
            best_local_prefix_blocks: 8,
            cost_model: Some(RemoteG2CostModel {
                score_weight: 1.0,
                cost_blocks: 0.0,
                cost_per_block: 0.0,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::NoCandidate { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::BelowRemoteG2Cost);
            }
            other => panic!("expected no candidate, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_caps_planned_blocks_before_scoring_and_materializing() {
        let hashes = block_hashes(12);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 12)]);
        let input = RemoteKvReuseSelectionInput {
            cost_model: Some(RemoteG2CostModel {
                score_weight: 1.0,
                cost_blocks: 1.0,
                cost_per_block: 0.0,
                max_planned_blocks: Some(5),
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_reuse_plan(input);

        match decision {
            RemoteKvReuseDecision::Plan { plan, .. } => {
                assert_eq!(plan.start_block_index, 0);
                assert_eq!(plan.planned_prefix_blocks, 5);
                assert_eq!(plan.router_block_hashes, hashes[..5]);
            }
            other => panic!("expected capped plan, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_rejects_target_outside_local_overlap_gap() {
        let hashes = block_hashes(12);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 12)]);
        let input = RemoteKvReuseSelectionInput {
            target_local_prefix_blocks: 2,
            best_local_prefix_blocks: 8,
            cost_model: Some(RemoteG2CostModel {
                score_weight: 1.0,
                cost_blocks: 1.0,
                cost_per_block: 0.0,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: Some(4),
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::NoCandidate { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::LocalOverlapGapTooLarge);
            }
            other => panic!("expected local-overlap rejection, got {other:?}"),
        }
    }

    #[test]
    fn select_candidate_rejects_cost_negative_direct_g2() {
        let hashes = block_hashes(8);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[], &[(source, 8)]);
        let input = RemoteKvReuseSelectionInput {
            cost_model: Some(RemoteG2CostModel {
                score_weight: 0.5,
                cost_blocks: 16.0,
                cost_per_block: 0.0,
                max_planned_blocks: None,
                max_local_overlap_gap_blocks: None,
            }),
            ..selection_input(target, &hashes, &matches)
        };

        let decision = select_remote_g2_candidate(input);

        match decision {
            RemoteG2CandidateDecision::NoCandidate { reason, .. } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::BelowRemoteG2Cost);
            }
            other => panic!("expected no candidate, got {other:?}"),
        }
    }

    #[test]
    fn scenario_full_g1_extra_g2_no_contiguous_prefix() {
        let hashes = block_hashes(4);
        let target = WorkerWithDpRank::new(9, 0);
        let source = WorkerWithDpRank::new(7, 0);
        let matches = tiered_matches(&[(source, 4)], &[(source, 2)]);

        let decision = select_remote_g2_reuse_plan(selection_input(target, &hashes, &matches));

        match decision {
            RemoteKvReuseDecision::NoPlan { reason, stats } => {
                assert_eq!(reason, RemoteKvReuseNoPlanReason::NoContiguousPrefix);
                assert!(stats.rejected_g1_candidates > 0);
            }
            other => panic!("expected no plan, got {other:?}"),
        }
    }
}
