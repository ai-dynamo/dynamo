// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared G3→G2 staging logic.
//!
//! Extracts the common staging kernel used by InitiatorSession, ResponderSession,
//! and ServerSession. Each caller handles its own post-staging bookkeeping
//! (updating holders, sending messages, etc.).

use std::sync::Arc;

use anyhow::Result;

use crate::{BlockId, G2, G3, worker::group::ParallelWorkers};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::{blocks::ImmutableBlock, manager::BlockManager};
use kvbm_physical::transfer::TransferOptions;

use super::blocks::BlockHolder;

/// Result of staging G3 blocks to G2.
pub struct StagingResult {
    /// Newly created G2 blocks (registered with the G2 manager).
    pub new_g2_blocks: Vec<ImmutableBlock<G2>>,
}

/// Stage G3 blocks to G2.
///
/// Core staging kernel: allocate G2 destinations → execute local transfer (G3→G2)
/// → register new G2 blocks with the source sequence hashes → return new blocks.
///
/// The caller is responsible for:
/// - Clearing the G3 holder (`take_all()`)
/// - Adding new blocks to the G2 holder (`extend()`)
/// - Sending any notifications to peers
pub async fn stage_g3_to_g2(
    g3_blocks: &BlockHolder<G3>,
    g2_manager: &BlockManager<G2>,
    parallel_worker: &dyn ParallelWorkers,
) -> Result<StagingResult> {
    if g3_blocks.is_empty() {
        return Ok(StagingResult {
            new_g2_blocks: Vec::new(),
        });
    }

    let src_ids: Vec<BlockId> = g3_blocks.blocks().iter().map(|b| b.block_id()).collect();

    // Allocate destination G2 blocks
    let dst_blocks = g2_manager
        .allocate_blocks(src_ids.len())
        .ok_or_else(|| anyhow::anyhow!("Failed to allocate G2 blocks"))?;

    let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

    // Execute transfer
    let notification = parallel_worker.execute_local_transfer(
        LogicalLayoutHandle::G3,
        LogicalLayoutHandle::G2,
        Arc::from(src_ids),
        Arc::from(dst_ids),
        TransferOptions::default(),
    )?;

    // Wait for transfer to complete
    notification.await?;

    // Register new G2 blocks using the G3 blocks' metadata (sequence hashes)
    let new_g2_blocks: Vec<ImmutableBlock<G2>> = dst_blocks
        .into_iter()
        .zip(g3_blocks.blocks().iter())
        .map(|(dst, src)| {
            let complete = dst
                .stage(src.sequence_hash(), g2_manager.block_size())
                .expect("block size mismatch");
            g2_manager.register_block(complete)
        })
        .collect();

    Ok(StagingResult { new_g2_blocks })
}
