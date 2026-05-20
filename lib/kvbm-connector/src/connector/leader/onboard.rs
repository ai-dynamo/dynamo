// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use anyhow::Context;
use futures::future::{BoxFuture, Either, Ready};

use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::TransferOptions;

use super::*;
use crate::G3;

/// Future type returned by `FindMatchesResult::wait_for_completion()`.
type StagingCompletion = Either<Ready<Result<()>>, BoxFuture<'static, Result<()>>>;

/// Reset a slot back to `Inactive` after a failed onboarding transfer.
///
/// `start_onboarding` calls `txn_start_onboarding` *before* spawning the
/// transfer task, so a transfer error would otherwise wedge the slot in
/// `Onboarding` forever and block any future reuse of this `request_id`.
/// Mirrors the cancel-path pivot used by `prepare_intra_pass_onboarding`
/// (`txn_to_error` → `txn_take_error`), which is the only state path out
/// of `Onboarding` that does not require the worker callback we are not
/// going to receive on this failure.
fn reset_slot_after_failed_onboarding(slot: &mut slot::RequestSlot) {
    slot.txn_to_error(); // Onboarding → Error(Onboarding)
    let _ = slot.txn_take_error(); // Error(Onboarding) → Inactive (state dropped)
}

/// Collect the G2 blocks destined for onboarding from every shard, honoring
/// the `[effective_start .. final_end)` span (first-hole contiguous match).
///
/// This walks shards in order, calling `take_g2_blocks()` on each. It then
/// drops the leading `effective_start - shards[0].start_block` blocks (the
/// Case-B mask) from the head and truncates the tail to `final_end -
/// effective_start` elements. Any excess beyond `final_end` in the hole-shard
/// (the shard whose match count was < num_queried_blocks) comes pre-filtered
/// because terminal shards have `matched_count` g2 blocks available.
pub(crate) fn collect_g2_blocks_from_shards(
    state: &mut slot::OnboardingState,
    block_size: usize,
) -> Result<Vec<ImmutableBlock<G2>>> {
    debug_assert!(!state.shards.is_empty());
    debug_assert!(state.all_shards_terminal());

    let (effective_start, final_end) = state.matched_span(block_size);
    debug_assert!(effective_start <= final_end);
    let desired_blocks = final_end - effective_start;
    let leading_skip = effective_start - state.shards[0].start_block;

    let mut collected: Vec<ImmutableBlock<G2>> = Vec::new();
    for shard in state.shards.iter_mut() {
        // Stop early once we have enough blocks.
        if collected.len() >= leading_skip + desired_blocks {
            break;
        }
        let blocks = shard
            .find_session
            .take_g2_blocks()
            .ok_or_else(|| anyhow!("No G2 blocks found for shard at {}", shard.start_block))?;
        collected.extend(blocks);
    }

    // Apply head mask.
    if leading_skip > 0 {
        if leading_skip > collected.len() {
            bail!(
                "effective_start mask ({}) exceeds collected g2 blocks ({})",
                leading_skip,
                collected.len()
            );
        }
        collected.drain(..leading_skip);
    }

    // Truncate the tail to exactly desired_blocks.
    if collected.len() < desired_blocks {
        bail!(
            "collected {} g2 blocks but span requested {}",
            collected.len(),
            desired_blocks
        );
    }
    collected.truncate(desired_blocks);

    Ok(collected)
}

/// Select the G1 block IDs that correspond to externally-matched (onboard) blocks.
///
/// When onboarding, the `block_ids` list contains ALL blocks for the request:
///   [computed_blocks... | external_blocks... | new_blocks...]
///
/// The external (matched) blocks start right after the computed prefix and span
/// `num_external_tokens / block_size` entries.
///
/// # Arguments
/// * `block_ids` - All block IDs allocated for the request
/// * `num_computed_tokens` - Tokens already present in G1 (prefix cache hit)
/// * `num_external_tokens` - Tokens matched externally (to be onboarded)
/// * `block_size` - Tokens per block
fn select_onboard_block_ids(
    block_ids: &[BlockId],
    num_computed_tokens: usize,
    num_external_tokens: usize,
    block_size: usize,
) -> Vec<BlockId> {
    let num_computed_blocks = num_computed_tokens / block_size;
    let num_external_blocks = num_external_tokens / block_size;
    let floor_loss_tokens = num_external_tokens % block_size;
    let num_external_blocks_ceil = num_external_blocks + if floor_loss_tokens > 0 { 1 } else { 0 };

    // Diagnostic: surface floor-division loss so we can tell whether
    // num_external_tokens is a multiple of block_size for this workload.
    // floor_loss_tokens > 0 means the trailing remainder is silently dropped
    // and the actual byte count moved by the onboard transfer is smaller
    // than (num_external_tokens * per_token_kv_bytes) suggests.
    tracing::info!(
        num_computed_tokens,
        num_external_tokens,
        block_size,
        num_computed_blocks,
        num_external_blocks,
        num_external_blocks_ceil,
        floor_loss_tokens,
        "select_onboard_block_ids"
    );

    block_ids[num_computed_blocks..num_computed_blocks + num_external_blocks].to_vec()
}

impl ConnectorLeader {
    /// Prepare intra-pass onboarding by storing G2/G1 block pairs.
    ///
    /// Unlike `start_onboarding` which spawns an async task, this method:
    /// 1. Extracts G2 blocks from the find session and moves them to the leader's
    ///    `pending_intra_pass_g2_blocks` collection (preserving ownership)
    /// 2. Stores the G2/G1 block IDs in the slot's `pending_intra_pass` state
    ///
    /// The G2 blocks are held on the leader until the forward pass completes.
    /// A cleanup task (spawned in `process_scheduler_output`) releases them.
    pub(crate) fn prepare_intra_pass_onboarding(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        // Intra-pass onboarding currently only supports G2→G1 (the worker's
        // CudaStream layer-by-layer load expects G2 sources). Host-bypass
        // mode has no G2 to source from, so reject the combination loudly
        // rather than silently producing wrong results. Use Inter mode if
        // bypass is required.
        if self.runtime.config().cache.bypass_host_cache() {
            bail!(
                "Intra-pass onboarding is not supported when host (G2) cache is bypassed. \
                 Set DYN_KVBM_CPU_CACHE_GB to enable G2, or use Inter onboard mode \
                 (the default) which supports the bypass + GDS direct G3→G1 path."
            );
        }
        let shared_slot = self.get_slot(request_id)?;
        let mut slot = shared_slot.lock();

        let block_size = self.block_size();
        let num_computed_tokens = slot
            .onboarding_state()
            .expect("session should exist")
            .num_computed_tokens;

        // The g1 slice starts at `num_computed_blocks` (i.e. the effective
        // start used when computing matched_tokens; see `matched_span`) and
        // spans `num_external_blocks` entries. We take this slice up front so
        // we can own it without borrowing `block_ids` past `apply_new_blocks`.
        let g1_block_ids = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            block_size,
        );

        // Record the block_ids - this assigns them to the token_block sequence hashes
        slot.apply_new_blocks(block_ids);

        // Extract G2 blocks from every shard (with head mask + tail truncate)
        // and also capture the session IDs we now need to release.
        //
        // CD-only states have empty shards; intra-pass onboarding is not on the
        // CD path so an empty shards list here is a programmer error caught by
        // `collect_g2_blocks_from_shards`.
        let (g2_blocks, session_ids_to_release) = {
            let state = slot
                .onboarding_state_mut()
                .ok_or_else(|| anyhow!("Expected active onboarding state for {}", request_id))?;
            let g2_blocks = collect_g2_blocks_from_shards(state, block_size)?;
            let session_ids: Vec<_> = state
                .shards
                .iter()
                .filter_map(|s| s.find_session.session_id())
                .collect();
            (g2_blocks, session_ids)
        };

        let g2_block_ids: Vec<BlockId> = g2_blocks.iter().map(|b| b.block_id()).collect();

        // Validate block counts match
        if g2_block_ids.len() != g1_block_ids.len() {
            bail!(
                "G2/G1 block count mismatch: {} G2 blocks, {} G1 blocks",
                g2_block_ids.len(),
                g1_block_ids.len()
            );
        }

        tracing::debug!(
            request_id,
            g2_count = g2_block_ids.len(),
            g1_count = g1_block_ids.len(),
            "Prepared intra-pass onboarding data"
        );

        // Move G2 blocks to leader's collection (preserves ownership until forward pass completes)
        self.pending_intra_pass_g2_blocks.lock().extend(g2_blocks);

        // Store the block IDs in slot's pending state for aggregation
        slot.extend_pending_intra_pass(g2_block_ids, g1_block_ids);

        // Transition to inactive since we're not doing async onboarding
        slot.txn_to_error(); // Clear the PreparingToOnboard state
        let _ = slot.txn_take_error(); // Discard and transition to Inactive

        // Release server-side session state for every shard's async session
        // (Ready variants return None, so they're skipped).
        if let Some(instance_leader) = self.instance_leader() {
            for session_id in session_ids_to_release {
                instance_leader.release_session(session_id);
            }
        }

        Ok(())
    }

    pub(crate) fn start_onboarding(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let shared_slot = self.get_slot(request_id)?;

        // Extract a wait_for_completion future for every shard, then transition
        // to Onboarding.
        let (staging_futs, onboard_blocks_ids) = {
            let mut slot = shared_slot.lock();

            let num_computed_tokens = match slot.onboarding_state() {
                Some(state) => state.num_computed_tokens,
                None => 0,
            };
            let onboard_blocks_ids = select_onboard_block_ids(
                &block_ids,
                num_computed_tokens,
                num_external_tokens,
                self.block_size(),
            );

            // record the block_ids
            // this will assign the block_ids to the token_block sequence hashes
            slot.apply_new_blocks(block_ids);

            let staging_futs: Vec<_> = match slot.onboarding_state() {
                Some(onboarding_state) => onboarding_state
                    .shards
                    .iter()
                    .map(|shard| shard.find_session.wait_for_completion())
                    .collect(),
                None => bail!("Expected active onboarding state for {}", request_id),
            };

            if let Err(e) = slot.txn_start_onboarding() {
                tracing::error!("Failed to start onboarding: {}", e);
                bail!("Failed to start onboarding: {}", e);
            }

            (staging_futs, onboard_blocks_ids)
        };

        let leader = self.clone();
        let handle = self.runtime.tokio();
        let request_id = request_id.to_string();
        let block_size = self.block_size();

        let bypass_host = self.runtime.config().cache.bypass_host_cache();
        handle.spawn(async move {
            let transfer_result = execute_onboarding(
                leader.clone(),
                shared_slot.clone(),
                onboard_blocks_ids.clone(),
                staging_futs,
                block_size,
                bypass_host,
            )
            .await;

            if transfer_result.is_err() {
                tracing::error!(
                    "Onboarding failed: {}",
                    transfer_result.as_ref().err().unwrap()
                );
            } else {
                tracing::debug!("Onboarding completed successfully");
            }

            // -----------------------------------------------------------------
            // PHASE 1: local cleanup. These ops cannot fail and MUST complete
            // before any worker RPC that .expect()-panics on failure. If we
            // ever reorder a worker RPC above this block, a panicking RPC
            // would short-circuit the slot reset + session release and the
            // request_id would wedge in Onboarding (same shape as the
            // todo!() bug that preceded this fix). Keep RPCs in Phase 2.
            // -----------------------------------------------------------------

            // Collect session IDs for release before any state transition —
            // onboarding_state() returns Some only while the slot is in
            // Onboarding. On Err, also clear the slot's Onboarding state so
            // the request_id can be reused; without this the slot wedges in
            // Onboarding because txn_start_onboarding already transitioned
            // it before the transfer was awaited.
            let session_ids: Vec<_> = {
                let mut slot_guard = shared_slot.lock();
                let ids: Vec<_> = slot_guard
                    .onboarding_state()
                    .map(|state| {
                        state
                            .shards
                            .iter()
                            .filter_map(|s| s.find_session.session_id())
                            .collect()
                    })
                    .unwrap_or_default();
                if transfer_result.is_err() {
                    reset_slot_after_failed_onboarding(&mut slot_guard);
                }
                ids
            };

            // Release server-side state for every shard's async session
            // (Ready variants return None).
            if !session_ids.is_empty() {
                let instance_leader = leader.instance_leader().expect("InstanceLeader not set");
                for session_id in session_ids {
                    instance_leader.release_session(session_id);
                }
            }

            // -----------------------------------------------------------------
            // PHASE 2: worker RPCs. Both calls .expect() on failure because
            // a worker-side failure here is a CRITICAL signal we don't want
            // to silently swallow into a log line. Local state is already
            // consistent thanks to Phase 1, so a panic here unwinds the
            // spawn task without leaving the slot in Onboarding or leaking
            // shard sessions.
            // -----------------------------------------------------------------

            if transfer_result.is_err() {
                // Notify workers that these block_ids did NOT get the
                // expected values; the scheduler is responsible for handling
                // these errors. Clone request_id so it stays live for the
                // mark_onboarding_complete call below.
                leader
                    .workers
                    .get()
                    .unwrap()
                    .mark_failed_onboarding(request_id.clone(), onboard_blocks_ids)
                    .await
                    .expect("Failed to mark failed onboarding");
            }

            // Regardless of error, mark the onboarding as complete.
            // An error here is a CRITICAL failure: one or more workers have
            // been lost or cannot be reached. Not completing this transaction
            // will result in resources being leaked and the system will
            // eventually deadlock or fail.
            leader
                .workers
                .get()
                .unwrap()
                .mark_onboarding_complete(request_id)
                .await
                .expect("Failed to mark onboarding complete");
        });

        Ok(())
    }
}

async fn execute_onboarding(
    leader: Arc<ConnectorLeader>,
    slot: Arc<Mutex<RequestSlot>>,
    block_ids: Vec<BlockId>,
    staging_futs: Vec<StagingCompletion>,
    block_size: usize,
    bypass_host: bool,
) -> Result<()> {
    let g1_block_ids = block_ids;
    let start = Instant::now();

    // Wait for every shard's find_session to reach a terminal state.
    for fut in staging_futs {
        fut.await
            .context("Onboarding find_session operation failed")?;
    }
    let staging_complete = Instant::now();

    // Pull source blocks from the find session. In host-bypass mode the
    // Ready result carries G3 blocks (no staging happened); otherwise the
    // AsyncSession has already staged into G2 across one or more shards.
    //
    // The ImmutableBlock<G2/G3> Drop impl returns the block to the inactive
    // pool, so the source-blocks Vec must outlive `execute_local_transfer`
    // below — otherwise the source blocks could be evicted and overwritten
    // mid-transfer. Both branches bind their Vec at function scope.
    //
    // FIXME(kvbm-398 merge): multi-shard search + bypass_host is not yet
    // implemented — bypass_host paths still assume a single legacy shard.
    let _g2_source_hold: Vec<ImmutableBlock<G2>>;
    let _g3_source_hold: Vec<ImmutableBlock<G3>>;
    let (src_layout, src_block_ids) = {
        let mut slot_guard = slot.lock();
        let state = slot_guard
            .onboarding_state_mut()
            .expect("Onboarding state not found");
        if bypass_host {
            if state.shards.len() != 1 {
                bail!(
                    "G2-bypass onboarding with multi-shard search is not yet supported \
                     (shard_count={})",
                    state.shards.len()
                );
            }
            // Apply the same head-mask + tail-truncate as the G2 path so the
            // G3 source list aligns with g1_block_ids, which was sliced from
            // `matched_span`. Without this, a post-issue increase in
            // num_computed_tokens (leading_skip > 0) or a partial shard
            // (matched_count < num_queried_blocks) over-selects source blocks.
            let (effective_start, final_end) = state.matched_span(block_size);
            let desired = final_end.saturating_sub(effective_start);
            let leading_skip = effective_start - state.shards[0].start_block;

            let mut g3_blocks = state.shards[0]
                .find_session
                .take_g3_blocks()
                .ok_or_else(|| anyhow!("No G3 blocks found (bypass mode)"))?;

            if leading_skip > g3_blocks.len() {
                bail!(
                    "effective_start mask ({}) exceeds collected g3 blocks ({})",
                    leading_skip,
                    g3_blocks.len()
                );
            }
            g3_blocks.drain(..leading_skip);

            if g3_blocks.len() < desired {
                bail!(
                    "collected {} g3 blocks but span requested {}",
                    g3_blocks.len(),
                    desired
                );
            }
            g3_blocks.truncate(desired);

            let ids: Vec<BlockId> = g3_blocks.iter().map(|b| b.block_id()).collect();
            _g3_source_hold = g3_blocks;
            _g2_source_hold = Vec::new();
            (LogicalLayoutHandle::G3, ids)
        } else {
            let g2_blocks = collect_g2_blocks_from_shards(state, block_size)?;
            let ids: Vec<BlockId> = g2_blocks.iter().map(|b| b.block_id()).collect();
            _g2_source_hold = g2_blocks;
            _g3_source_hold = Vec::new();
            (LogicalLayoutHandle::G2, ids)
        }
    };

    let num_blocks = src_block_ids.len();
    assert_eq!(num_blocks, g1_block_ids.len());

    let instance_leader = leader.instance_leader().expect("InstanceLeader not set");
    let parallel_worker = instance_leader
        .parallel_worker()
        .ok_or_else(|| anyhow::anyhow!("No parallel worker available for local transfer"))?;

    // TODO: potential optimization would be to stream blocks to G1 as they
    // become ready. The current implementation awaits all source blocks
    // before issuing the transfer. The balance is when to acquire/allocate
    // G1 blocks (a precious commodity) vs. when to start onboarding.
    let start_xfer = Instant::now();
    parallel_worker
        .execute_local_transfer(
            src_layout,
            LogicalLayoutHandle::G1,
            Arc::from(src_block_ids),
            Arc::from(g1_block_ids),
            TransferOptions::default(),
        )?
        .await?;
    let end_xfer = Instant::now();

    tracing::info!(
        blocks = num_blocks,
        staging_us = staging_complete.duration_since(start).as_micros() as u64,
        xfer_us = end_xfer.duration_since(start_xfer).as_micros() as u64,
        total_us = end_xfer.duration_since(start).as_micros() as u64,
        src = if bypass_host {
            "kvbm_engine::G3"
        } else {
            "kvbm_engine::G2"
        },
        dst = "kvbm_engine::G1",
        "Onboard transfer complete"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_SIZE: usize = 16; // tokens per block

    /// Onboard blocks must come from after the computed prefix,
    /// not from the end of the block list.
    ///
    /// Scenario: 10 blocks total, first 2 are computed (already in G1),
    /// next 3 are external (to be onboarded from G2), remaining 5 are new.
    ///
    /// block_ids:  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    ///              ^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^
    ///              computed(2)  external(3)      new(5)
    #[test]
    fn test_select_onboard_blocks_after_computed_prefix() {
        let block_ids: Vec<BlockId> = (100..110).collect(); // 10 blocks
        let num_computed_tokens = 2 * BLOCK_SIZE; // 2 blocks already computed
        let num_external_tokens = 3 * BLOCK_SIZE; // 3 blocks to onboard

        let result = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            BLOCK_SIZE,
        );

        // Must select blocks [102, 103, 104] — the 3 blocks after the 2 computed ones
        assert_eq!(result, vec![102, 103, 104]);
    }

    /// When nothing is computed, external blocks start at the beginning.
    #[test]
    fn test_select_onboard_blocks_no_computed_prefix() {
        let block_ids: Vec<BlockId> = (100..108).collect(); // 8 blocks
        let num_computed_tokens = 0;
        let num_external_tokens = 5 * BLOCK_SIZE;

        let result = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            BLOCK_SIZE,
        );

        assert_eq!(result, vec![100, 101, 102, 103, 104]);
    }

    /// When all blocks are external (full cache hit from G2).
    #[test]
    fn test_select_onboard_blocks_all_external() {
        let block_ids: Vec<BlockId> = (100..106).collect(); // 6 blocks
        let num_computed_tokens = 0;
        let num_external_tokens = 6 * BLOCK_SIZE;

        let result = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            BLOCK_SIZE,
        );

        assert_eq!(result, vec![100, 101, 102, 103, 104, 105]);
    }

    /// When all blocks are computed (nothing to onboard).
    #[test]
    fn test_select_onboard_blocks_nothing_external() {
        let block_ids: Vec<BlockId> = (100..106).collect();
        let num_computed_tokens = 6 * BLOCK_SIZE;
        let num_external_tokens = 0;

        let result = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            BLOCK_SIZE,
        );

        assert_eq!(result, Vec::<BlockId>::new());
    }

    /// Single external block after a large computed prefix.
    #[test]
    fn test_select_onboard_blocks_single_external_after_large_prefix() {
        let block_ids: Vec<BlockId> = (100..120).collect(); // 20 blocks
        let num_computed_tokens = 15 * BLOCK_SIZE;
        let num_external_tokens = BLOCK_SIZE;

        let result = select_onboard_block_ids(
            &block_ids,
            num_computed_tokens,
            num_external_tokens,
            BLOCK_SIZE,
        );

        // Block at index 15 (after 15 computed blocks)
        assert_eq!(result, vec![115]);
    }

    // -------------------------------------------------------------------
    // reset_slot_after_failed_onboarding — regression coverage for the
    // multi-shard host-bypass + (any other) error path. start_onboarding
    // drives the slot into Onboarding before spawning the transfer task;
    // if the transfer returns Err and we don't pivot the slot back to
    // Inactive, the request_id is stuck until the slot is destroyed.
    //
    // A true end-to-end reproducer (panic in spawned task) needs the full
    // ConnectorLeader + WorkerClients harness; these tests guard the
    // narrower invariant the spawn-Err arm now relies on: the helper
    // must return *any* Onboarding slot to Inactive.
    // -------------------------------------------------------------------

    use crate::common::Request;
    use crate::connector::leader::slot::{RequestSlot, TransactionState};
    use kvbm_engine::leader::{FindMatchesResult, ReadyResult};

    fn build_test_slot() -> RequestSlot {
        let tokens: Vec<u32> = (0..16).collect(); // 4 complete blocks of size 4
        let request = Request::new("test-request", tokens, None, None, None);
        RequestSlot::new(request, 4).expect("Failed to create RequestSlot")
    }

    fn mock_onboarding_pair() -> (usize, FindMatchesResult) {
        let ready = ReadyResult::new(vec![], Default::default());
        (100, FindMatchesResult::Ready(ready))
    }

    #[test]
    fn test_reset_slot_after_failed_onboarding_clears_onboarding_state() {
        let mut slot = build_test_slot();
        let (num_computed_tokens, find_session) = mock_onboarding_pair();

        // Drive the slot to Onboarding, matching what start_onboarding does
        // before awaiting execute_onboarding.
        slot.txn_prepare_to_onboard_legacy(num_computed_tokens, find_session)
            .unwrap();
        slot.txn_start_onboarding().unwrap();
        assert!(
            matches!(slot.txn_state(), TransactionState::Onboarding(_)),
            "precondition: slot should be Onboarding, was {:?}",
            slot.txn_state()
        );

        reset_slot_after_failed_onboarding(&mut slot);

        assert!(
            slot.txn_state().is_inactive(),
            "slot must be Inactive after reset; was {:?}",
            slot.txn_state()
        );
    }

    #[test]
    fn test_reset_slot_after_failed_onboarding_is_idempotent() {
        // Calling reset on a slot already in Inactive must be a no-op,
        // not a panic — the spawn-Err arm runs reset under a lock and we
        // don't want a torn lock if reset is ever called twice.
        let mut slot = build_test_slot();
        assert!(slot.txn_state().is_inactive());

        reset_slot_after_failed_onboarding(&mut slot);

        assert!(slot.txn_state().is_inactive());
    }
}
