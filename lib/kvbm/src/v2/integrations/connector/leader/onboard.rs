// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use futures::future::{BoxFuture, Either, Ready};

use crate::{logical::LogicalLayoutHandle, physical::TransferOptions};
use std::time::Instant;

use super::*;

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

            let num_external_blocks = num_external_tokens / self.block_size();
            let onboard_start_block_idx = block_ids.len().saturating_sub(num_external_blocks);
            let onboard_blocks_ids = block_ids[onboard_start_block_idx..].to_vec();

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

    // TODO: potential optimization would be to stream G2 blocks to G1 blocks as G2 blocks are ready.
    // The current implementation awaits all G2 blocks to be ready before executing the transfer.
    // The balance here is when do we acquire/allocate G1 blocks as they are a precious commodity vs.,
    // when should we start onboarding. More analysis is needed here to determine the optimal strategy.
    let start_time = Instant::now();
    instance_leader
        .execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G1,
            g2_block_ids,
            g1_block_ids,
            TransferOptions::default(),
        )?
        .await?;
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    // tracing::info!(
    //     "G2 to G1 transfer: blocks={}, duration={:?}"
    //     g2_block_ids.len(),
    //     duration,
    // );
    Ok(())
}
