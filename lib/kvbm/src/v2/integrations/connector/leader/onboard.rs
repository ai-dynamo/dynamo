// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::future::{BoxFuture, Either, Ready};

use crate::{logical::LogicalLayoutHandle, physical::TransferOptions};

use super::*;

impl ConnectorLeader {
    pub(crate) fn start_onboarding(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        let shared_slot = self.get_slot(request_id)?;

        // Extract session_id and transition to Onboarding state
        let staging_fut = {
            let mut slot = shared_slot.lock();

            let staging_fut = match slot.onboarding_state() {
                Some(onboarding_state) => onboarding_state.find_session.wait_for_completion(),
                None => bail!("Expected active onboarding state for {}", request_id),
            };

            if let Err(e) = slot.txn_start_onboarding() {
                tracing::error!("Failed to start onboarding: {}", e);
                bail!("Failed to start onboarding: {}", e);
            }

            staging_fut
        };

        let leader = self.clone();
        let handle = self.runtime.tokio();

        handle.spawn(async move {
            match execute_onboarding(leader.clone(), shared_slot.clone(), block_ids, staging_fut)
                .await
            {
                Ok(()) => {
                    tracing::debug!("Onboarding completed successfully");
                }
                Err(_e) => {
                    tracing::error!("Onboarding failed: {}", _e);
                    todo!("clean up session and free resources")
                }
            }
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
    staging_fut.await?;

    let g2_blocks = slot
        .lock()
        .onboarding_state_mut()
        .expect("Onboarding state not found")
        .find_session
        .take_g2_blocks()
        .ok_or_else(|| anyhow::anyhow!("No G2 blocks found"))?;

    let g2_block_ids: Vec<BlockId> = g2_blocks.iter().map(|b| b.block_id()).collect();

    // All blocks are now in G2
    let instance_leader = leader.instance_leader().expect("InstanceLeader not set");

    instance_leader
        .execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G1,
            g2_block_ids,
            g1_block_ids,
            TransferOptions::default(),
        )?
        .await?;

    let onboarding_state = slot.lock().txn_take_onboarding()?;

    if let Some(session_id) = onboarding_state.find_session.session_id() {
        instance_leader.release_session(session_id);
    }

    Ok(())
}
