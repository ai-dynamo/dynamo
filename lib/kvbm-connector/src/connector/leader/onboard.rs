// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use futures::future::{BoxFuture, Either, Ready};

use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::TransferOptions;

use super::*;

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
    let num_external_blocks = num_external_tokens / block_size;
    let onboard_start_block_idx = block_ids.len().saturating_sub(num_external_blocks);
    block_ids[onboard_start_block_idx..].to_vec()
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
        let num_computed_tokens = slot
            .onboarding_state()
            .expect("session should exist")
            .num_computed_tokens;

        let num_computed_blocks = num_computed_tokens / self.block_size();
        let num_external_blocks = num_external_tokens / self.block_size();
        let g1_block_ids =
            block_ids[num_computed_blocks..num_computed_blocks + num_external_blocks].to_vec();

        // Record the block_ids - this assigns them to the token_block sequence hashes
        slot.apply_new_blocks(block_ids);

        // Extract G2 blocks from the find session (takes ownership)
        let g2_blocks = slot
            .onboarding_state_mut()
            .ok_or_else(|| anyhow!("Expected active onboarding state for {}", request_id))?
            .find_session
            .take_g2_blocks()
            .ok_or_else(|| anyhow!("No G2 blocks found for intra-pass onboarding"))?;

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

        Ok(())
    }

    pub(crate) fn start_onboarding(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let shared_slot = self.get_slot(request_id)?;

        // Extract session_id and transition to Onboarding state
        let (staging_fut, onboard_blocks_ids) = {
            let mut slot = shared_slot.lock();

            let onboard_blocks_ids =
                select_onboard_block_ids(&block_ids, 0, num_external_tokens, self.block_size());

            // record the block_ids
            // this will assign the block_ids to the token_block sequence hashes
            slot.apply_new_blocks(block_ids);

            let staging_fut = match slot.onboarding_state() {
                Some(onboarding_state) => onboarding_state.find_session.wait_for_completion(),
                None => bail!("Expected active onboarding state for {}", request_id),
            };

            if let Err(e) = slot.txn_start_onboarding() {
                tracing::error!("Failed to start onboarding: {}", e);
                bail!("Failed to start onboarding: {}", e);
            }

            (staging_fut, onboard_blocks_ids)
        };

        let leader = self.clone();
        let handle = self.runtime.tokio();
        let request_id = request_id.to_string();

        handle.spawn(async move {
            match execute_onboarding(
                leader.clone(),
                shared_slot.clone(),
                onboard_blocks_ids.clone(),
                staging_fut,
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

            // This will clean up the find session if it exists, this is necessary to avoid resource leaks.
            if let Some(session_id) = shared_slot
                .lock()
                .onboarding_state()
                .and_then(|state| state.find_session.session_id())
            {
                let instance_leader = leader.instance_leader().expect("InstanceLeader not set");
                instance_leader.release_session(session_id);
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
    staging_fut: Either<Ready<Result<()>>, BoxFuture<'static, Result<()>>>,
) -> Result<()> {
    let g1_block_ids = block_ids;

    // Wait for find_session completion by accessing it through the slot
    staging_fut
        .await
        .context("Onboarding find_session operation failed")?;

    let g2_blocks = slot
        .lock()
        .onboarding_state_mut()
        .expect("Onboarding state not found")
        .find_session
        .take_g2_blocks()
        .ok_or_else(|| anyhow::anyhow!("No G2 blocks found"))?;

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

        let result =
            select_onboard_block_ids(&block_ids, num_computed_tokens, num_external_tokens, BLOCK_SIZE);

        // Must select blocks [102, 103, 104] — the 3 blocks after the 2 computed ones
        assert_eq!(result, vec![102, 103, 104]);
    }

    /// When nothing is computed, external blocks start at the beginning.
    #[test]
    fn test_select_onboard_blocks_no_computed_prefix() {
        let block_ids: Vec<BlockId> = (100..108).collect(); // 8 blocks
        let num_computed_tokens = 0;
        let num_external_tokens = 5 * BLOCK_SIZE;

        let result =
            select_onboard_block_ids(&block_ids, num_computed_tokens, num_external_tokens, BLOCK_SIZE);

        assert_eq!(result, vec![100, 101, 102, 103, 104]);
    }

    /// When all blocks are external (full cache hit from G2).
    #[test]
    fn test_select_onboard_blocks_all_external() {
        let block_ids: Vec<BlockId> = (100..106).collect(); // 6 blocks
        let num_computed_tokens = 0;
        let num_external_tokens = 6 * BLOCK_SIZE;

        let result =
            select_onboard_block_ids(&block_ids, num_computed_tokens, num_external_tokens, BLOCK_SIZE);

        assert_eq!(result, vec![100, 101, 102, 103, 104, 105]);
    }

    /// When all blocks are computed (nothing to onboard).
    #[test]
    fn test_select_onboard_blocks_nothing_external() {
        let block_ids: Vec<BlockId> = (100..106).collect();
        let num_computed_tokens = 6 * BLOCK_SIZE;
        let num_external_tokens = 0;

        let result =
            select_onboard_block_ids(&block_ids, num_computed_tokens, num_external_tokens, BLOCK_SIZE);

        assert_eq!(result, Vec::<BlockId>::new());
    }

    /// Single external block after a large computed prefix.
    #[test]
    fn test_select_onboard_blocks_single_external_after_large_prefix() {
        let block_ids: Vec<BlockId> = (100..120).collect(); // 20 blocks
        let num_computed_tokens = 15 * BLOCK_SIZE;
        let num_external_tokens = 1 * BLOCK_SIZE;

        let result =
            select_onboard_block_ids(&block_ids, num_computed_tokens, num_external_tokens, BLOCK_SIZE);

        // Block at index 15 (after 15 computed blocks)
        assert_eq!(result, vec![115]);
    }
}
