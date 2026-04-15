// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use futures::future::{BoxFuture, Either, Ready};

use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::TransferOptions;

use super::*;

/// Future type returned by `FindMatchesResult::wait_for_completion()`.
type StagingCompletion = Either<Ready<Result<()>>, BoxFuture<'static, Result<()>>>;

/// Collect the G2 blocks destined for onboarding from every shard, honoring
/// the `[effective_start .. final_end)` span (first-hole contiguous match).
///
/// This walks shards in order, calling `take_g2_blocks()` on each. It then
/// drops the leading `effective_start - shards[0].start_block` blocks (the
/// Case-B mask) from the head and truncates the tail to `final_end -
/// effective_start` elements. Any excess beyond `final_end` in the hole-shard
/// (the shard whose match count was < num_queried_blocks) comes pre-filtered
/// because terminal shards have `matched_count` g2 blocks available.
fn collect_g2_blocks_from_shards(
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
        let shared_slot = self.get_slot(request_id)?;
        let mut slot = shared_slot.lock();

        let block_size = self.block_size();
        let num_computed_tokens = slot
            .onboarding_state()
            .expect("session should exist")
            .num_computed_tokens;

        let num_computed_blocks = num_computed_tokens / block_size;
        let num_external_blocks = num_external_tokens / block_size;

        // The g1 slice starts at `num_computed_blocks` (i.e. the effective
        // start used when computing matched_tokens; see `matched_span`) and
        // spans `num_external_blocks` entries. We take this slice up front so
        // we can own it without borrowing `block_ids` past `apply_new_blocks`.
        let g1_block_ids =
            block_ids[num_computed_blocks..num_computed_blocks + num_external_blocks].to_vec();

        // Record the block_ids - this assigns them to the token_block sequence hashes
        slot.apply_new_blocks(block_ids);

        // Extract G2 blocks from every shard (with head mask + tail truncate)
        // and also capture the session IDs we now need to release.
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

            let num_external_blocks = num_external_tokens / self.block_size();
            let onboard_start_block_idx = block_ids.len().saturating_sub(num_external_blocks);
            let onboard_blocks_ids = block_ids[onboard_start_block_idx..].to_vec();

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

        handle.spawn(async move {
            match execute_onboarding(
                leader.clone(),
                shared_slot.clone(),
                onboard_blocks_ids.clone(),
                staging_futs,
                block_size,
            )
            .await
            {
                Ok(()) => {
                    tracing::debug!("Onboarding completed successfully");
                }
                Err(_e) => {
                    tracing::error!("Onboarding failed: {}", _e);
                    // we were unable to execute the local transfer, so we need to report to each worker which block_ids
                    // did not get the expected values; the scheduler will be responsible for handling these errors.
                    leader
                        .workers
                        .get()
                        .unwrap()
                        .mark_failed_onboarding(request_id, onboard_blocks_ids)
                        .await
                        .expect("Failed to mark failed onboarding");
                    todo!("clean up session and free resources")
                }
            }

            // Release server-side state for every shard's async session (Ready
            // variants return None).
            let session_ids: Vec<_> = shared_slot
                .lock()
                .onboarding_state()
                .map(|state| {
                    state
                        .shards
                        .iter()
                        .filter_map(|s| s.find_session.session_id())
                        .collect()
                })
                .unwrap_or_default();
            if !session_ids.is_empty() {
                let instance_leader = leader.instance_leader().expect("InstanceLeader not set");
                for session_id in session_ids {
                    instance_leader.release_session(session_id);
                }
            }

            // regardess of error, we mark the onboarding as complete
            // an error here is a CRITICAL failure, one or more workers have been lost or can not be reached.
            // not completeing this transaction will result in resources being leaked and the system will eventually
            // deadlock or fail.
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
) -> Result<()> {
    let g1_block_ids = block_ids;

    // Wait for every shard's find_session to reach a terminal state.
    for fut in staging_futs {
        fut.await
            .context("Onboarding find_session operation failed")?;
    }

    let g2_blocks = {
        let mut slot = slot.lock();
        let state = slot
            .onboarding_state_mut()
            .expect("Onboarding state not found");
        collect_g2_blocks_from_shards(state, block_size)?
    };

    let g2_block_ids: Vec<BlockId> = g2_blocks.iter().map(|b| b.block_id()).collect();

    assert_eq!(g2_block_ids.len(), g1_block_ids.len());

    // All blocks are now in G2
    let instance_leader = leader.instance_leader().expect("InstanceLeader not set");
    let parallel_worker = instance_leader
        .parallel_worker()
        .ok_or_else(|| anyhow::anyhow!("No parallel worker available for local transfer"))?;

    // TODO: potential optimization would be to stream G2 blocks to G1 blocks as G2 blocks are ready.
    // The current implementation awaits all G2 blocks to be ready before executing the transfer.
    // The balance here is when do we acquire/allocate G1 blocks as they are a precious commodity vs.,
    // when should we start onboarding. More analysis is needed here to determine the optimal strategy.
    // let start_time = Instant::now();
    parallel_worker
        .execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G1,
            Arc::from(g2_block_ids),
            Arc::from(g1_block_ids),
            TransferOptions::default(),
        )?
        .await?;
    // let end_time = Instant::now();
    // let duration = end_time.duration_since(start_time);
    // tracing::info!(
    //     "G2 to G1 transfer: blocks={}, duration={:?}"
    //     g2_block_ids.len(),
    //     duration,
    // );
    Ok(())
}
