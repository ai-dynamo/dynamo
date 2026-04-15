// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Search reconciliation for `get_num_new_matched_tokens`.
//!
//! vLLM may re-poll `get_num_new_matched_tokens` for the same request between
//! forward passes with a different `num_computed_tokens`. Between polls the
//! scheduler may:
//!   * absorb more G1 blocks (`new > old`),
//!   * evict previously-computed G1 blocks (`new < old`), or
//!   * restore a prefix from eviction and thereby grow the visible sequence.
//!
//! Rather than discarding the saved search on mismatch (the previous behavior,
//! which panicked in debug and returned `(None, false)` in release per
//! issue #5285), we reconcile a list of `OnboardingShard`s covering contiguous
//! block-index ranges of the sequence. Each shard owns its own
//! `FindMatchesResult`. On a mismatch we:
//!   * Case B (`new > old`): update `num_computed_tokens`; the completion walk
//!     masks off the redundant prefix via `effective_start`.
//!   * Case C (`new < old`): prepend a new shard covering
//!     `[new/bs .. shards[0].start_block)`.
//!   * Case D (`total_tokens` grew, indicating eviction restore): append a new
//!     shard for the new upper range.
//!
//! On completion, the outcome is computed by walking shards in order and
//! short-circuiting at the first hole (first-hole first-match semantics).

use super::*;
use kvbm_common::SequenceHash;
use kvbm_engine::leader::Leader;
use slot::{OnboardingShard, OnboardingState};

/// Compute the exclusive upper-block-index of the search range for a given
/// `total_tokens`. Matches the invariant the original `process_match` used:
/// if the token count lands exactly on a block boundary, the last full block
/// is excluded from the search.
fn compute_last_block_index(total_tokens: usize, block_size: usize) -> usize {
    if total_tokens.is_multiple_of(block_size) {
        (total_tokens / block_size).saturating_sub(1)
    } else {
        total_tokens / block_size
    }
}

/// Issue a new find_matches for the given block-index range and return a shard.
///
/// `sequence_hashes` is the slot's full sequence-hash vector; we slice
/// `[start_block .. end_block_exclusive)`.
fn issue_shard(
    leader: &dyn Leader,
    sequence_hashes: &[SequenceHash],
    start_block: usize,
    end_block_exclusive: usize,
) -> Result<OnboardingShard> {
    debug_assert!(start_block < end_block_exclusive);
    debug_assert!(end_block_exclusive <= sequence_hashes.len());

    let slice = &sequence_hashes[start_block..end_block_exclusive];
    let options = FindMatchesOptions {
        search_remote: true,
        staging_mode: StagingMode::Full,
    };

    tracing::debug!(
        start_block,
        end_block_exclusive,
        num_hashes = slice.len(),
        "issuing new shard find_matches_with_options"
    );

    let find_session = leader
        .find_matches_with_options(slice, options)
        .map_err(|e| {
            tracing::error!("Failed to start find operation: {}", e);
            anyhow!("Failed to start find operation: {}", e)
        })?;

    Ok(OnboardingShard {
        start_block,
        num_queried_blocks: end_block_exclusive - start_block,
        find_session,
    })
}

/// Reconcile the stored onboarding state against the latest
/// `num_computed_tokens` and `total_tokens`, issuing new shards as needed,
/// and return the current match outcome.
///
/// Cases (let `old = state.num_computed_tokens`, `new = num_computed_tokens`,
/// `bs = block_size`):
///   * **A** — `new == old` and `total_tokens` unchanged: no-op.
///   * **B** — `new > old`: update `state.num_computed_tokens = new`. Shards
///     are untouched; the completion walk masks the prefix via `effective_start`.
///   * **C** — `new < old`: prepend a new shard covering
///     `[new/bs .. shards[0].start_block)`.
///   * **D** — `current_last_block_index > shards.last().end_block()`: append
///     a new shard for `[shards.last().end_block() .. current_last_block_index)`.
///
/// B+D and C+D may apply in the same call.
pub(crate) fn reconcile_state(
    state: &mut OnboardingState,
    num_computed_tokens: usize,
    total_tokens: usize,
    block_size: usize,
    sequence_hashes: &[SequenceHash],
    leader: &dyn Leader,
) -> Result<()> {
    debug_assert!(!state.shards.is_empty());
    state.debug_assert_contiguous();

    let old = state.num_computed_tokens;
    let new = num_computed_tokens;

    // Contract asserts (unchanged from legacy behavior): vLLM always passes
    // block-aligned values. A mismatch here is a caller-side contract breach.
    assert!(
        new.is_multiple_of(block_size),
        "num_computed_tokens {} must be a multiple of block_size {}",
        new,
        block_size
    );

    // Case B: new > old ----------------------------------------------------
    if new > old {
        let delta = new - old;
        assert!(
            delta.is_multiple_of(block_size),
            "num_computed_tokens delta {} -> {} must be a multiple of block_size {}",
            old,
            new,
            block_size,
        );
        tracing::debug!(
            old,
            new,
            "num_computed_tokens increased; masking prefix via effective_start"
        );
        state.num_computed_tokens = new;
    }

    // Case C: new < old ----------------------------------------------------
    if new < old {
        let delta = old - new;
        assert!(
            delta.is_multiple_of(block_size),
            "num_computed_tokens delta {} -> {} must be a multiple of block_size {}",
            old,
            new,
            block_size,
        );
        let new_start_block = new / block_size;
        let current_head_start = state.shards[0].start_block;
        debug_assert!(new_start_block <= current_head_start);

        if new_start_block < current_head_start {
            tracing::debug!(
                old,
                new,
                new_start_block,
                current_head_start,
                "num_computed_tokens decreased; prepending prefix shard"
            );
            let new_shard =
                issue_shard(leader, sequence_hashes, new_start_block, current_head_start)?;
            state.shards.insert(0, new_shard);
        }
        state.num_computed_tokens = new;
    }

    // Case D: total_tokens grew -------------------------------------------
    let current_last_block_index = compute_last_block_index(total_tokens, block_size);
    let existing_end = state.shards.last().unwrap().end_block();
    if current_last_block_index > existing_end {
        tracing::debug!(
            existing_end,
            current_last_block_index,
            "total_tokens grew (eviction restore); appending suffix shard"
        );
        let new_shard = issue_shard(
            leader,
            sequence_hashes,
            existing_end,
            current_last_block_index,
        )?;
        state.shards.push(new_shard);
    }

    state.debug_assert_contiguous();
    Ok(())
}

/// Compute the outcome from a reconciled onboarding state.
pub(crate) fn compute_outcome(state: &OnboardingState, block_size: usize) -> MatchCheckOutcome {
    // Step 1: any shard still working? Return InProgress.
    if !state.all_shards_terminal() {
        tracing::trace!("Find operation still in progress (some shards non-terminal)");
        return MatchCheckOutcome::InProgress;
    }

    // Step 2: walk contiguously with first-hole short-circuit.
    let (effective_start, final_end) = state.matched_span(block_size);
    let matched_tokens = final_end.saturating_sub(effective_start) * block_size;

    tracing::debug!(
        effective_start,
        final_end,
        matched_tokens,
        num_shards = state.shards.len(),
        "Find completed (walk)"
    );

    if matched_tokens == 0 {
        // No external blocks usable — either the prefix had a hole, or the
        // `effective_start` ate the whole range. Flow to Inactive via Found{0}.
        MatchCheckOutcome::Found { matched_tokens: 0 }
    } else {
        MatchCheckOutcome::Found { matched_tokens }
    }
}

impl ConnectorLeader {
    /// Internal helper to check match status and determine the outcome.
    /// This function contains all the logic for determining whether blocks match,
    /// but does not perform state transitions - that's handled by the caller.
    pub(crate) fn process_match(
        &self,
        slot: &mut RequestSlot,
        num_computed_tokens: usize,
    ) -> Result<MatchCheckOutcome> {
        let block_size = slot.block_size();
        let total_tokens = slot.sequence.total_tokens();

        // Early exit if we cannot match a full block
        if (total_tokens - num_computed_tokens) < block_size {
            // If a stale onboarding state exists for some reason, drop it via
            // finalize's txn_to_inactive path by returning NoMatch.
            return Ok(MatchCheckOutcome::NoMatch);
        }

        let instance_leader = self
            .instance_leader
            .get()
            .ok_or_else(|| anyhow!("InstanceLeader not set; called before initialized"))?;
        let leader: &dyn Leader = instance_leader;

        // Contract: num_computed_tokens must be block-aligned.
        assert!(
            num_computed_tokens.is_multiple_of(block_size),
            "num_computed_tokens {} must be a multiple of block_size {}",
            num_computed_tokens,
            block_size,
        );

        let sequence_hashes = slot.all_sequence_hashes();
        assert!(!sequence_hashes.is_empty());

        // Initial search path: no active onboarding state -> kick off a fresh
        // single-shard search.
        if !slot.has_onboarding_state() {
            let num_device_blocks = num_computed_tokens / block_size;
            let last_block_index = compute_last_block_index(total_tokens, block_size);

            // Defensive: if there's nothing left to search, return NoMatch.
            if num_device_blocks >= last_block_index {
                return Ok(MatchCheckOutcome::NoMatch);
            }

            let shard = issue_shard(
                leader,
                &sequence_hashes,
                num_device_blocks,
                last_block_index,
            )?;
            let num_queried_blocks = shard.num_queried_blocks;

            if let Err(e) = slot.txn_prepare_to_onboard(
                num_computed_tokens,
                total_tokens,
                num_device_blocks,
                num_queried_blocks,
                shard.find_session,
            ) {
                tracing::error!("Failed to set find session: {}", e);
                bail!("Failed to set find session: {}", e);
            }
        } else {
            // Existing onboarding state: reconcile against the new inputs.
            debug_assert!(matches!(
                slot.txn_state(),
                TransactionState::PreparingToOnboard(_)
            ));
            let state = slot.onboarding_state_mut().expect("state should exist");
            reconcile_state(
                state,
                num_computed_tokens,
                total_tokens,
                block_size,
                &sequence_hashes,
                leader,
            )?;
        }

        debug_assert!(matches!(
            slot.txn_state(),
            TransactionState::PreparingToOnboard(_)
        ));

        let state = slot.onboarding_state().expect("state should exist");
        Ok(compute_outcome(state, block_size))
    }

    /// Recover from a match error by transitioning to error state and extracting state for cleanup.
    pub(crate) fn recover_from_match_error(&self, slot: &mut RequestSlot) {
        // Transition to error state to preserve any active state data
        slot.txn_to_error();

        // Take the error state and clean up
        if let Ok(active_data) = slot.txn_take_error() {
            match active_data {
                slot::ActiveStateData::Onboarding(onboarding_state) => {
                    // Release every shard's session. For Ready variants this
                    // is a no-op; for AsyncSession variants this returns
                    // server-side session state to the pool.
                    if let Some(instance_leader) = self.instance_leader.get() {
                        onboarding_state.release_all(instance_leader);
                    } else {
                        tracing::warn!(
                            "recover_from_match_error: InstanceLeader not set; cannot release {} shard session(s)",
                            onboarding_state.shards.len()
                        );
                    }
                }
                slot::ActiveStateData::Offloading(_offloading_state) => {
                    // Offloading cleanup if needed
                    tracing::warn!("Offloading error recovery - cleanup may be needed");
                    todo!("implement offloading error recovery");
                }
            }
        }
        // Slot is now in Inactive state
    }
}

// ============================================================================
// Reconciliation unit tests
// ============================================================================
//
// These tests exercise `reconcile_state` and `compute_outcome` directly via a
// minimal `TestLeader` stub of the `Leader` trait. They cover Cases A/B/C/D
// from the design and the multi-shard walk + first-hole semantics. Async
// shard variants use `AsyncSessionResult::new_complete_for_test` /
// `new_pending_for_test` to construct terminal/pending states without a
// real session.

#[cfg(test)]
mod reconcile_tests {
    use super::*;
    use kvbm_engine::leader::{AsyncSessionResult, FindMatchesResult, MatchBreakdown, ReadyResult};
    use std::sync::Mutex as StdMutex;
    use tokio::sync::watch;

    const BS: usize = 4;

    /// A stub `Leader` that returns canned `FindMatchesResult`s in queue
    /// order. Each call consumes the next item from `responses`. If the queue
    /// is exhausted, the call panics — tests should provide exactly the
    /// number of canned results they expect.
    struct TestLeader {
        responses: StdMutex<Vec<FindMatchesResult>>,
        calls: StdMutex<Vec<usize>>,
    }

    impl TestLeader {
        fn new(responses: Vec<FindMatchesResult>) -> Self {
            Self {
                responses: StdMutex::new(responses),
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl Leader for TestLeader {
        fn find_matches_with_options(
            &self,
            sequence_hashes: &[SequenceHash],
            _options: FindMatchesOptions,
        ) -> Result<FindMatchesResult> {
            self.calls.lock().unwrap().push(sequence_hashes.len());
            let mut q = self.responses.lock().unwrap();
            assert!(
                !q.is_empty(),
                "TestLeader: unexpected find_matches_with_options call ({} hashes)",
                sequence_hashes.len()
            );
            Ok(q.remove(0))
        }
    }

    /// Build a vector of dummy `SequenceHash` values. The tests don't depend
    /// on actual hash content, only on slice indices.
    fn dummy_hashes(n: usize) -> Vec<SequenceHash> {
        (0..n as u64)
            .map(|i| SequenceHash::new(i, None, i))
            .collect()
    }

    /// Construct a Ready shard result with `g2_count` empty G2 placeholders.
    /// Since `ImmutableBlock<G2>` cannot be constructed outside kvbm-logical,
    /// this only works for `g2_count == 0`. Callers that need a non-zero
    /// match count should use `complete_async(matched)` instead.
    fn ready_zero() -> FindMatchesResult {
        FindMatchesResult::Ready(ReadyResult::new(vec![], MatchBreakdown::default()))
    }

    /// Construct a terminal AsyncSession with `matched_blocks` reported via
    /// the watch channel. The blocks vec is empty (only the count matters
    /// for reconciliation/walk logic).
    fn complete_async(matched: usize) -> FindMatchesResult {
        FindMatchesResult::AsyncSession(AsyncSessionResult::new_complete_for_test(
            matched,
            vec![],
            MatchBreakdown::default(),
        ))
    }

    /// Construct a pending AsyncSession (status = Searching). Returns the
    /// session and the watch sender so tests can transition it later.
    fn pending_async() -> (
        FindMatchesResult,
        watch::Sender<kvbm_engine::leader::OnboardingStatus>,
    ) {
        let (session, tx) = AsyncSessionResult::new_pending_for_test();
        (FindMatchesResult::AsyncSession(session), tx)
    }

    /// Build a fresh `OnboardingState` with a single shard.
    fn make_state(
        num_computed_tokens: usize,
        total_tokens_at_start: usize,
        start_block: usize,
        num_queried_blocks: usize,
        find_session: FindMatchesResult,
    ) -> slot::OnboardingState {
        slot::OnboardingState::new(
            num_computed_tokens,
            total_tokens_at_start,
            slot::OnboardingShard {
                start_block,
                num_queried_blocks,
                find_session,
            },
        )
    }

    // ------------------------------------------------------------------
    // Case A — no change
    // ------------------------------------------------------------------

    #[test]
    fn case_a_unchanged_in_progress() {
        // single shard, async pending. Expect InProgress, shards untouched.
        let (pending, _tx) = pending_async();
        let mut state = make_state(40, 64, 10, 5, pending);
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 0);
        assert_eq!(state.shards.len(), 1);
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    #[test]
    fn case_a_unchanged_terminal_full_match() {
        // single shard, async complete with all 5 blocks. Expect Found{5*BS}.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 0);
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 5 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    // ------------------------------------------------------------------
    // Case B — new > old (mask via effective_start)
    // ------------------------------------------------------------------

    #[test]
    fn case_b_mask_partial() {
        // Shard at start_block=10, num_queried=5, all 5 matched.
        // num_computed grows from 40 (=10*BS) to 48 (=12*BS): mask 2 blocks.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 48, 64, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 0);
        assert_eq!(state.num_computed_tokens, 48);
        assert_eq!(state.shards.len(), 1);

        // effective_start = max(10, 12) = 12; final_end = 15; matched = 3*BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 3 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_b_mask_consumes_all() {
        // Mask covers the entire matched range -> 0 effective external tokens.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 60, 64, BS, &hashes, &leader).unwrap();
        // effective_start = 15, final_end = 15 -> 0
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn case_b_unaligned_delta_panics() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        // 42 is not a multiple of BS=4 -> the top-level alignment assert fires.
        reconcile_state(&mut state, 42, 64, BS, &hashes, &leader).unwrap();
    }

    // ------------------------------------------------------------------
    // Case C — new < old (prepend prefix shard)
    // ------------------------------------------------------------------

    #[test]
    fn case_c_prepend_full_match() {
        // shard at start_block=10, num=5, complete with 5 matches.
        // num_computed drops from 40 to 32 -> prepend shard [8..10) of size 2.
        // Test leader returns a fully-matched (2/2) async response for the prefix.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(2)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.num_computed_tokens, 32);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 8);
        assert_eq!(state.shards[0].num_queried_blocks, 2);
        assert_eq!(state.shards[1].start_block, 10);

        // effective_start = max(8, 8) = 8; running_end walks 8 -> 10 -> 15.
        // matched = (15 - 8)*BS = 28.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 7 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_hole_zero_match() {
        // Prefix shard returns a hole (1 of 2 matched). Walk short-circuits at
        // running_end + 1 = 9. effective_start = 8. final_end = 9. matched = 1*BS.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(1)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, BS),
            other => panic!("expected Found{{BS}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_zero_match() {
        // Prefix shard returns 0 matches -> full hole at the very start.
        // running_end starts at 8, matched=0 -> final_end=8. effective_start=8.
        // matched = 0.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(0)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_pending_keeps_in_progress() {
        // Old suffix terminal, new prefix in progress -> InProgress.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let (pending, _tx) = pending_async();
        let leader = TestLeader::new(vec![pending]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader).unwrap();
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    #[test]
    fn case_c_full_preemption_no_special_logging() {
        // num_computed_tokens drops to 0. Should be handled identically to any
        // other Case C — prepend a shard for the full prefix.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(10)]);

        reconcile_state(&mut state, 0, 64, BS, &hashes, &leader).unwrap();
        assert_eq!(state.num_computed_tokens, 0);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 0);
        assert_eq!(state.shards[0].num_queried_blocks, 10);
        // effective_start = max(0, 0) = 0; full match across both shards = 15 * BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 15 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn case_c_unaligned_delta_panics() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);
        // 38 is not a multiple of BS=4.
        reconcile_state(&mut state, 38, 64, BS, &hashes, &leader).unwrap();
    }

    // ------------------------------------------------------------------
    // Case D — total_tokens grew (eviction restore)
    // ------------------------------------------------------------------

    #[test]
    fn case_d_appends_upper_shard() {
        // Original total_tokens=64 -> last_block_index=15; shard covers [10..15).
        // total_tokens grows to 96 -> last_block_index=23. Append [15..23).
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(8)]);

        reconcile_state(&mut state, 40, 96, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[1].start_block, 15);
        assert_eq!(state.shards[1].num_queried_blocks, 8);
        // walk: 10 -> 15 (shard0 full) -> 23 (shard1 full). effective_start=10. matched=13*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 13 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_d_pending_upper_shard() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let (pending, _tx) = pending_async();
        let leader = TestLeader::new(vec![pending]);

        reconcile_state(&mut state, 40, 96, BS, &hashes, &leader).unwrap();
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    // ------------------------------------------------------------------
    // Combinations: B+D, C+D
    // ------------------------------------------------------------------

    #[test]
    fn case_b_plus_d() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(8)]);

        // num grows by 8 (mask 2), total grows -> append shard [15..23) full
        reconcile_state(&mut state, 48, 96, BS, &hashes, &leader).unwrap();
        // effective_start = max(10, 12) = 12; final_end = 23 -> matched = 11*BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 11 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_plus_d() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        // First call: prefix [8..10) (Case C), then second: upper [15..23) (Case D).
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader).unwrap();
        assert_eq!(leader.call_count(), 2);
        assert_eq!(state.shards.len(), 3);
        // walk: 8 -> 10 -> 15 -> 23. effective_start=8. matched = 15*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 15 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    // ------------------------------------------------------------------
    // Walk semantics edge cases
    // ------------------------------------------------------------------

    #[test]
    fn walk_short_circuits_on_first_hole() {
        // Two terminal shards; first has a hole (matched < num_queried).
        // Second shard's match count is irrelevant past the hole.
        let mut state = make_state(40, 64, 10, 5, complete_async(3));
        // Pre-attach a second shard manually for this test.
        state.shards.push(slot::OnboardingShard {
            start_block: 15,
            num_queried_blocks: 3,
            find_session: complete_async(3),
        });
        // walk: 10 -> 13 (hole). effective_start = 10. matched = 3*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 3 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn walk_one_pending_shard_returns_in_progress() {
        // Two shards: first terminal (full), second pending. Step-1 returns
        // InProgress (no partial reporting).
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let (pending, _tx) = pending_async();
        state.shards.push(slot::OnboardingShard {
            start_block: 15,
            num_queried_blocks: 3,
            find_session: pending,
        });
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    // ------------------------------------------------------------------
    // Invariants and metric plumbing
    // ------------------------------------------------------------------

    #[test]
    fn shard_contiguity_invariant_after_reconcile() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader).unwrap();
        // debug_assert_contiguous would have panicked if the invariant was
        // violated; call it explicitly to be sure.
        state.debug_assert_contiguous();
        assert_eq!(state.shards[0].end_block(), state.shards[1].start_block);
        assert_eq!(state.shards[1].end_block(), state.shards[2].start_block);
    }

    #[test]
    fn total_query_blocks_sums_across_shards() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader).unwrap();
        assert_eq!(state.total_query_blocks(), 2 + 5 + 8);
    }

    /// Direct repro of the scenario reported in #5285: at bs=16, vLLM polled
    /// `get_num_new_matched_tokens` first with 21536 and then with 21520 (a
    /// 16-token == 1-block decrease, i.e. one G1 block was evicted between
    /// scheduler passes). The legacy code panicked in debug here. With
    /// reconciliation, we simply prepend a one-block prefix shard and report
    /// a (potentially) extended match without losing the suffix work.
    #[test]
    fn issue_5285_repro_one_block_eviction() {
        const REAL_BS: usize = 16;
        // total_tokens beyond the first poll's range; pick something that gives
        // headroom and a stable last_block_index.
        let total_tokens = 24_000_usize;
        let last_block_index = total_tokens / REAL_BS;
        // First poll: num_computed_tokens=21536 -> shard at 21536/16 = 1346,
        // covering [1346 .. last_block_index). Suppose all blocks matched.
        let num_blocks_first = last_block_index - 1346;
        let mut state = slot::OnboardingState::new(
            21536,
            total_tokens,
            slot::OnboardingShard {
                start_block: 1346,
                num_queried_blocks: num_blocks_first,
                find_session: complete_async(num_blocks_first),
            },
        );
        let hashes = dummy_hashes(last_block_index);

        // Second poll: 21520 = 1345 * 16. We expect a new prefix shard
        // [1345..1346) of size 1.
        let leader = TestLeader::new(vec![complete_async(1)]);
        reconcile_state(&mut state, 21520, total_tokens, REAL_BS, &hashes, &leader).unwrap();

        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 1345);
        assert_eq!(state.shards[0].num_queried_blocks, 1);

        // effective_start = max(1345, 21520/16=1345) = 1345
        // final_end = last_block_index
        // matched = (last_block_index - 1345) * 16
        let expected = (last_block_index - 1345) * REAL_BS;
        match compute_outcome(&state, REAL_BS) {
            MatchCheckOutcome::Found { matched_tokens } => {
                assert_eq!(matched_tokens, expected)
            }
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn ready_zero_match_returns_found_zero() {
        // A Ready variant with no matched blocks -> Found{0}. (This is the
        // legacy single-shard happy path with an empty result.)
        let mut state = make_state(40, 64, 10, 5, ready_zero());
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    fn other_repr(o: &MatchCheckOutcome) -> &'static str {
        match o {
            MatchCheckOutcome::InProgress => "InProgress",
            MatchCheckOutcome::NoMatch => "NoMatch",
            MatchCheckOutcome::Found { .. } => "Found",
        }
    }
}
