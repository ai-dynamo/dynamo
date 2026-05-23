// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-indexer remote-pull helpers.
//!
//! Two things live here, both about the **wire** between the composer and a
//! remote holder:
//!
//! - [`DiscoveryPlan`] — the decimation strategy for the hub indexer query
//!   plus the pin-target arithmetic. Pure data + arithmetic; unit-tested
//!   in isolation so the composer's `do_discover_and_pull` is just glue.
//! - [`pull_from`] — the per-candidate `open_session` → `pull_from_session`
//!   → `close_session` RPC dance.
//!
//! Orchestration (concurrency with G3 staging, candidate iteration, cancel,
//! terminal status emission) lives in [`super::composer::OnboardingComposer`].

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};

use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::transfer::{
    CloseTransferSessionRequest, FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PullFromSessionRequest, SearchMode, TierSelection,
};

use super::InstanceLeader;
use crate::{InstanceId, SequenceHash};

/// Wall-clock bound on the whole remote-search attempt. Sized to the holder's
/// transfer-session watchdog so a wedged pull degrades to the local match
/// rather than holding `get_num_new_matched_tokens` at `(None, false)`.
pub(super) const REMOTE_SEARCH_WATCHDOG: Duration = Duration::from_secs(30);

/// Maximum hashes sent to the hub indexer in a single discovery query.
/// The indexer's max-position semantics let us probe the remote depth with
/// a sparse, evenly-spaced sample of the remaining range — the holder's
/// `open_session` first-hole match then fills in the actual contiguous reach.
pub(super) const MAX_DISCOVERY_HASHES: usize = 64;

/// Plan for one hub-indexer discovery probe + the corresponding pin-target.
///
/// Constructed from a slice of "expected remaining" hashes (everything past
/// the synchronous local matches, before staging completes). Holds the
/// decimated query that goes on the wire and enough state to translate the
/// indexer's `deepest` reply back into a concrete pin range over the
/// composer's full sequence.
///
/// All arithmetic is pure; see the `tests` module for the contract under
/// every interesting input size + deepest position.
#[derive(Debug, Clone)]
pub(super) struct DiscoveryPlan {
    /// Decimated hashes to send to the indexer. Length ≤ [`MAX_DISCOVERY_HASHES`].
    query: Vec<SequenceHash>,
    /// Sampling stride used to build `query`: `query[k] == remaining[k * stride]`.
    stride: usize,
    /// Absolute index in the full sequence at which `remaining` starts.
    /// Pin ranges returned by [`Self::resolve_pin_target`] are expressed in
    /// the full-sequence frame.
    remaining_start: usize,
    /// Length of the underlying remaining slice, in hashes. Used both for
    /// the min-remote-blocks gate and as the upper bound when the deepest
    /// hit is the last sample in `query`.
    remaining_len: usize,
}

impl DiscoveryPlan {
    /// Build a plan over `sequence_hashes[remaining_start..]`. The query is
    /// decimated to at most [`MAX_DISCOVERY_HASHES`] samples by taking every
    /// `stride`-th hash, where `stride = ceil(remaining_len / MAX)`. An
    /// out-of-range `remaining_start` is clamped to `sequence_hashes.len()`.
    pub fn new(sequence_hashes: &[SequenceHash], remaining_start: usize) -> Self {
        let remaining_start = remaining_start.min(sequence_hashes.len());
        let remaining = &sequence_hashes[remaining_start..];
        let stride = remaining.len().div_ceil(MAX_DISCOVERY_HASHES).max(1);
        let query: Vec<SequenceHash> = remaining.iter().step_by(stride).copied().collect();
        Self {
            query,
            stride,
            remaining_start,
            remaining_len: remaining.len(),
        }
    }

    /// True when the plan has nothing to query (empty remaining range).
    pub fn is_empty(&self) -> bool {
        self.query.is_empty()
    }

    /// Hashes to send on the discovery wire.
    pub fn query_hashes(&self) -> &[SequenceHash] {
        &self.query
    }

    /// Length of the remaining slice (`remaining_len` = "blocks past local").
    /// Used by the composer's `min_remote_blocks` threshold check.
    pub fn remaining_len(&self) -> usize {
        self.remaining_len
    }

    /// Translate the indexer's `deepest` sampled hash into a concrete pin
    /// range over the composer's full sequence.
    ///
    /// Semantics: the indexer told us "deepest is the deepest *sampled* hash
    /// the cluster holds". Hashes strictly past the next sample after
    /// `deepest` are not safe to claim, so the pin extends from
    /// `current_prefix` up to (exclusive) the absolute position of that next
    /// sample (or the end of remaining when `deepest` is the last sample).
    /// The holder's `open_session` first-hole match then reports its actual
    /// contiguous reach within that target.
    ///
    /// Returns `None` when the pin would be empty, either because `deepest`
    /// isn't in our query (a malformed indexer reply) or because
    /// `current_prefix` has already reached past the indexer's deepest
    /// (staging covered the whole reach locally).
    pub fn resolve_pin_target(
        &self,
        deepest: SequenceHash,
        current_prefix: usize,
    ) -> Option<Range<usize>> {
        let query_idx = self.query.iter().position(|h| *h == deepest)?;
        let upper_in_remaining = (query_idx + 1)
            .checked_mul(self.stride)?
            .min(self.remaining_len);
        let pin_upper = self.remaining_start + upper_in_remaining;
        let pin_lower = current_prefix.min(pin_upper);
        if pin_lower >= pin_upper {
            return None;
        }
        Some(pin_lower..pin_upper)
    }
}

/// Open a session on `candidate`, pull its committed prefix into local G2,
/// and close the session.
///
/// Returns:
/// - `Ok(true)` if blocks were committed/pulled and the pull completed
/// - `Ok(false)` if the holder had nothing to offer
/// - `Err` on RPC/pull failure (partial chunks may still have landed in local G2)
///
/// The composer iterates a candidate list calling this helper, stopping on the
/// first `Ok(true)`. Final correctness is established by the composer's
/// post-pull re-match over the full sequence, not by the per-candidate result.
pub(super) async fn pull_from(
    leader: &Arc<InstanceLeader>,
    candidate: InstanceId,
    target: &[SequenceHash],
) -> Result<bool> {
    let client = LeaderControlClient::new(leader.messenger().clone(), candidate);

    let open = client
        .transfer()
        .open_session(OpenTransferSessionRequest {
            sequence_hashes: target.to_vec(),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .map_err(|e| anyhow!("open_session on {candidate}: {e}"))?;

    let (capability, committed) = match open {
        OpenTransferSessionResponse::Sync {
            capability,
            committed,
            ..
        } => (capability, committed),
        OpenTransferSessionResponse::NoBlocksFound => return Ok(false),
        OpenTransferSessionResponse::Async { capability } => {
            // We requested Sync; an Async reply means nothing usable inline.
            close(&client, capability.session_id, "unexpected async open").await;
            return Ok(false);
        }
    };

    if committed.is_empty() {
        close(&client, capability.session_id, "no committed blocks").await;
        return Ok(false);
    }

    // `selector: None` pulls every committed hash — i.e. the holder's
    // contiguous G2 prefix of `target` (its authoritative deepest match).
    let pull = leader
        .pull_from_session(PullFromSessionRequest {
            session_id: capability.session_id,
            source_instance_id: candidate,
            endpoint: Some(capability.endpoint),
            selector: None,
        })
        .await;

    close(&client, capability.session_id, "remote search complete").await;

    match pull {
        Ok(_) => Ok(true),
        Err(e) => Err(anyhow!("pull_from_session from {candidate}: {e}")),
    }
}

/// Best-effort holder-side session teardown. The holder watchdog reclaims on
/// failure, so this never propagates an error.
async fn close(client: &LeaderControlClient, session_id: uuid::Uuid, reason: &str) {
    if let Err(e) = client
        .transfer()
        .close_session(CloseTransferSessionRequest {
            session_id,
            reason: Some(reason.to_string()),
        })
        .await
    {
        tracing::debug!(error = %e, %session_id, "close_session failed (watchdog will reclaim)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic sequence of `n` distinct `SequenceHash`es with
    /// monotonically increasing `position()` so range arithmetic + lookups in
    /// the plan map back to the right slot.
    fn seq(n: usize) -> Vec<SequenceHash> {
        (0..n as u64)
            .map(|i| SequenceHash::new(i, None, i))
            .collect()
    }

    #[test]
    fn discovery_plan_under_cap_does_not_decimate() {
        let s = seq(10);
        let plan = DiscoveryPlan::new(&s, 0);
        assert_eq!(plan.stride, 1);
        assert_eq!(plan.query_hashes(), s.as_slice());
        assert_eq!(plan.remaining_len(), 10);
        assert!(!plan.is_empty());
    }

    #[test]
    fn discovery_plan_at_cap_does_not_decimate() {
        let s = seq(MAX_DISCOVERY_HASHES);
        let plan = DiscoveryPlan::new(&s, 0);
        assert_eq!(plan.stride, 1);
        assert_eq!(plan.query_hashes().len(), MAX_DISCOVERY_HASHES);
    }

    #[test]
    fn discovery_plan_just_over_cap_uses_stride_two() {
        // 65 hashes → stride ceil(65/64) = 2 → 33 samples ([0, 2, 4, ..., 64]).
        let s = seq(MAX_DISCOVERY_HASHES + 1);
        let plan = DiscoveryPlan::new(&s, 0);
        assert_eq!(plan.stride, 2);
        assert_eq!(plan.query_hashes().len(), 33);
        assert_eq!(plan.query_hashes()[0], s[0]);
        assert_eq!(plan.query_hashes()[1], s[2]);
        assert_eq!(*plan.query_hashes().last().unwrap(), s[64]);
    }

    #[test]
    fn discovery_plan_1024_hits_user_example() {
        // The user's worked example: 1024 remaining → stride 16 → 64 samples.
        let s = seq(1024);
        let plan = DiscoveryPlan::new(&s, 0);
        assert_eq!(plan.stride, 16);
        assert_eq!(plan.query_hashes().len(), MAX_DISCOVERY_HASHES);
        assert_eq!(plan.query_hashes()[0], s[0]);
        assert_eq!(plan.query_hashes()[1], s[16]);
        assert_eq!(*plan.query_hashes().last().unwrap(), s[1008]);
    }

    #[test]
    fn discovery_plan_remaining_start_offsets_pin_target() {
        // Remaining starts at position 4 in a 12-hash sequence.
        let s = seq(12);
        let plan = DiscoveryPlan::new(&s, 4);
        assert_eq!(plan.remaining_len(), 8);
        assert_eq!(plan.stride, 1);
        // Deepest is s[6] (third sample, query_idx 2 with stride 1).
        let pin = plan.resolve_pin_target(s[6], 4).expect("pin range");
        // Upper bound = remaining_start + (query_idx + 1) * stride = 4 + 3 = 7.
        // Lower bound = current_prefix = 4.
        assert_eq!(pin, 4..7);
    }

    #[test]
    fn discovery_plan_pin_extends_to_next_sample_after_deepest() {
        // The user's example: query = [x, y, z], deepest = y, pin reaches up
        // to (but not including) z. Construct a 5-hash sequence so the
        // natural stride is 1; force-decimate via a hand-rolled plan to
        // exercise the stride-2 arithmetic.
        let s = seq(5);
        let plan = DiscoveryPlan {
            query: vec![s[0], s[2], s[4]],
            stride: 2,
            remaining_start: 0,
            remaining_len: 5,
        };
        let pin = plan.resolve_pin_target(s[2], 0).expect("pin range");
        assert_eq!(pin, 0..4, "pin includes hashes [0,1,2,3]; excludes z=s[4]");
    }

    #[test]
    fn discovery_plan_pin_when_deepest_is_last_sample_caps_at_remaining_len() {
        // Deepest is the final sample — no "next" to bound against, so pin
        // extends to the end of remaining. Holder's first-hole match will
        // truncate to its actual reach.
        let s = seq(5);
        let plan = DiscoveryPlan {
            query: vec![s[0], s[2], s[4]],
            stride: 2,
            remaining_start: 0,
            remaining_len: 5,
        };
        let pin = plan.resolve_pin_target(s[4], 0).expect("pin range");
        // (2+1)*2 = 6, capped at remaining_len = 5. Pin = 0..5.
        assert_eq!(pin, 0..5);
    }

    #[test]
    fn discovery_plan_pin_none_when_current_prefix_covers_indexer_reach() {
        // Staging extended the local prefix past the indexer's deepest reach —
        // nothing remote to pull.
        let s = seq(8);
        let plan = DiscoveryPlan::new(&s, 2);
        let pin = plan.resolve_pin_target(s[2], 8);
        assert!(pin.is_none(), "current_prefix == seq end → no pin");
    }

    #[test]
    fn discovery_plan_pin_none_when_deepest_not_in_query() {
        // Malformed indexer reply (deepest is a hash we never sent). Return
        // None rather than panic so the composer degrades to local match.
        let s = seq(8);
        let plan = DiscoveryPlan::new(&s, 0);
        let unrelated = SequenceHash::new(9999, None, 9999);
        assert!(plan.resolve_pin_target(unrelated, 0).is_none());
    }

    #[test]
    fn discovery_plan_empty_remaining_is_empty() {
        let s = seq(4);
        let plan = DiscoveryPlan::new(&s, 4);
        assert!(plan.is_empty());
        assert_eq!(plan.remaining_len(), 0);
        assert!(plan.resolve_pin_target(s[0], 0).is_none());
    }

    #[test]
    fn discovery_plan_pin_uses_current_prefix_as_lower_bound() {
        // After staging, current_prefix grew past remaining_start. The pin
        // starts at current_prefix (not at remaining_start) so we don't try
        // to re-pull hashes staging already delivered locally.
        let s = seq(20);
        let plan = DiscoveryPlan::new(&s, 4);
        // Deepest at s[10] → query_idx 6 (stride 1) → upper = 4 + 7 = 11.
        // current_prefix = 8 (past remaining_start 4) → pin = 8..11.
        let pin = plan.resolve_pin_target(s[10], 8).expect("pin range");
        assert_eq!(pin, 8..11);
    }
}
