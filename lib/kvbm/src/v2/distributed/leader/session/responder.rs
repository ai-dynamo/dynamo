// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use tokio::sync::mpsc;

use std::collections::HashSet;
use std::sync::Arc;

use crate::{
    logical::{blocks::ImmutableBlock, manager::BlockManager},
    physical::transfer::TransferOptions,
    v2::{
        BlockId, G2, G3, InstanceId, SequenceHash, distributed::worker::Worker,
        logical::LogicalLayoutHandle, physical::manager::LayoutHandle,
    },
};

use super::{BlockHolder, SessionId, messages::OnboardMessage, transport::MessageTransport};

/// Responder-side session for handling block onboarding requests.
///
/// Lifecycle:
/// 1. Spawned when receiving CreateSession
/// 2. Searches local G2 for matches
/// 3. Holds ImmutableBlock<G2> references (RAII)
/// 4. Sends G2Results immediately
/// 5. Searches local G3 for remaining matches (if G3 available)
/// 6. Sends G3Results
/// 7. Receives HoldBlocks and filters held G2 blocks
/// 8. Receives StageBlocks and executes G3->G2 transfers
/// 9. Sends BlocksReady when staging completes
/// 10. Sends Acknowledged
/// 11. Completes and drops (releases blocks)
pub struct ResponderSession {
    session_id: SessionId,
    instance_id: InstanceId,
    requester: InstanceId,
    g2_manager: Arc<BlockManager<G2>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    worker: Option<Arc<dyn Worker>>,
    transport: Arc<MessageTransport>,
    // Held blocks using BlockHolder for RAII semantics
    // Blocks are automatically released when the session drops
    held_g2_blocks: BlockHolder<G2>,
    held_g3_blocks: BlockHolder<G3>,
}

impl ResponderSession {
    /// Create a new responder session.
    pub fn new(
        session_id: SessionId,
        instance_id: InstanceId,
        requester: InstanceId,
        g2_manager: Arc<BlockManager<G2>>,
        g3_manager: Option<Arc<BlockManager<G3>>>,
        worker: Option<Arc<dyn Worker>>,
        transport: Arc<MessageTransport>,
    ) -> Self {
        Self {
            session_id,
            instance_id,
            requester,
            g2_manager,
            g3_manager,
            worker,
            transport,
            held_g2_blocks: BlockHolder::empty(),
            held_g3_blocks: BlockHolder::empty(),
        }
    }

    /// Run the responder session task.
    ///
    /// This is the main session loop that processes messages from the channel.
    pub async fn run(
        mut self,
        mut rx: mpsc::Receiver<OnboardMessage>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<()> {
        // Phase 1: Immediate G2 search
        let g2_matches = self.g2_manager.match_blocks(&sequence_hashes);

        // Hold the G2 blocks using BlockHolder (RAII semantics)
        self.held_g2_blocks = BlockHolder::new(g2_matches);

        // Send G2 results immediately (fire-and-forget) with parallel arrays
        let g2_sequence_hashes: Vec<SequenceHash> = self.held_g2_blocks.sequence_hashes();
        let g2_block_ids: Vec<BlockId> = self
            .held_g2_blocks
            .blocks()
            .iter()
            .map(|b| b.block_id())
            .collect();

        // TODO: Get layout handle from G2 manager
        // Need to add layout_handle() method to BlockManager or store it
        let layout_handle = LayoutHandle::new(0, 0); // Placeholder

        let g2_msg = OnboardMessage::G2Results {
            responder: self.instance_id,
            session_id: self.session_id,
            layout_handle,
            sequence_hashes: g2_sequence_hashes,
            block_ids: g2_block_ids,
        };
        self.transport.send(self.requester, g2_msg).await?;

        // Phase 2: Search G3 for remaining hashes (if G3 available)
        let g2_matched_hashes: HashSet<SequenceHash> =
            self.held_g2_blocks.sequence_hashes().into_iter().collect();

        let remaining_hashes: Vec<SequenceHash> = sequence_hashes
            .iter()
            .filter(|h| !g2_matched_hashes.contains(h))
            .copied()
            .collect();

        if !remaining_hashes.is_empty()
            && let Some(ref g3_manager) = self.g3_manager
        {
            let g3_matches = g3_manager.match_blocks(&remaining_hashes);

            if !g3_matches.is_empty() {
                // Hold the G3 blocks using BlockHolder
                self.held_g3_blocks = BlockHolder::new(g3_matches);

                // Send G3 results (sequence hashes only, keep order)
                let g3_sequence_hashes: Vec<SequenceHash> = self.held_g3_blocks.sequence_hashes();

                let g3_msg = OnboardMessage::G3Results {
                    responder: self.instance_id,
                    session_id: self.session_id,
                    sequence_hashes: g3_sequence_hashes,
                };
                self.transport.send(self.requester, g3_msg).await?;
            }
        }

        // Send SearchComplete to signal we're done searching
        let complete_msg = OnboardMessage::SearchComplete {
            responder: self.instance_id,
            session_id: self.session_id,
        };
        self.transport.send(self.requester, complete_msg).await?;

        // Phase 3: Process incoming messages
        while let Some(msg) = rx.recv().await {
            match msg {
                OnboardMessage::HoldBlocks {
                    hold_hashes,
                    drop_hashes: _,
                    ..
                } => {
                    // Filter by sequence hash - BlockHolder's retain keeps only matching hashes
                    self.held_g2_blocks.retain(&hold_hashes);
                    self.held_g3_blocks.retain(&hold_hashes);

                    // Send acknowledgment
                    let ack = OnboardMessage::Acknowledged {
                        responder: self.instance_id,
                        session_id: self.session_id,
                    };
                    self.transport.send(self.requester, ack).await?;

                    // Check if we have G3 blocks to manage
                    if self.held_g3_blocks.is_empty() {
                        // No G3 blocks, session complete
                        break;
                    }
                    // Otherwise, wait for StageBlocks message
                }

                OnboardMessage::StageBlocks { stage_hashes, .. } => {
                    // Filter G3 blocks to only keep blocks to be staged
                    // BlockHolder's retain keeps only matching hashes
                    self.held_g3_blocks.retain(&stage_hashes);

                    if !self.held_g3_blocks.is_empty() && self.worker.is_some() {
                        // Execute G3->G2 transfer
                        self.stage_g3_to_g2().await?;
                    }

                    // Don't exit - wait for CloseSession in Hold/Prepare modes
                }

                OnboardMessage::ReleaseBlocks { release_hashes, .. } => {
                    // Release specific blocks by sequence hash
                    // BlockHolder's release removes blocks with given hashes
                    self.held_g2_blocks.release(&release_hashes);
                    self.held_g3_blocks.release(&release_hashes);
                }

                // todo: how does close session drop the session from the dashmap?
                // todo: do we need to handle this in the handler rather than the session responder loop?
                OnboardMessage::CloseSession { .. } => {
                    // Session complete - release all blocks and exit
                    // take_all() explicitly releases the blocks
                    let _ = self.held_g2_blocks.take_all();
                    let _ = self.held_g3_blocks.take_all();
                    break;
                }

                OnboardMessage::CreateSession { .. } => {
                    // Duplicate CreateSession - ignore
                }

                // todo: be explicit about what messages are expected and what messages are unexpected
                //       on the responder session - avoid using the wildcard match
                _ => {
                    // Unexpected message - log and ignore
                    eprintln!(
                        "ResponderSession: unexpected message for session {}: {:?}",
                        self.session_id, msg
                    );
                }
            }

            // TODO: Add heartbeat/TTL timeout handling
            // If no message received within TTL duration:
            // - Release all held blocks
            // - Exit session
            // Implementation:
            //   tokio::select! {
            //       msg = rx.recv() => { /* process message */ }
            //       _ = tokio::time::sleep_until(ttl_deadline) => {
            //           eprintln!("Session {} TTL expired, releasing blocks", self.session_id);
            //           break;
            //       }
            //   }
        }

        Ok(())
    }

    /// Stage G3 blocks to G2.
    async fn stage_g3_to_g2(&mut self) -> Result<()> {
        let worker = self
            .worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker required for G3->G2 staging"))?;

        let stage_block_ids: Vec<BlockId> = self
            .held_g3_blocks
            .blocks()
            .iter()
            .map(|b| b.block_id())
            .collect();

        // Allocate destination G2 blocks
        let dst_blocks = self
            .g2_manager
            .allocate_blocks(stage_block_ids.len())
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate G2 blocks"))?;

        let dst_block_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

        // Execute transfer
        let notification = worker.execute_local_transfer(
            LogicalLayoutHandle::G3,
            LogicalLayoutHandle::G2,
            Arc::from(stage_block_ids.clone()),
            Arc::from(dst_block_ids.clone()),
            TransferOptions::default(),
        )?;

        // Wait for transfer to complete
        notification.await?;

        // Register the new G2 blocks using the G3 blocks' metadata
        let new_g2_blocks: Vec<ImmutableBlock<G2>> = dst_blocks
            .into_iter()
            .zip(self.held_g3_blocks.blocks().iter())
            .map(|(dst, src)| {
                self.g2_manager
                    .register_mutable_block_from_existing(dst, src)
            })
            .collect();

        // Extract sequence hashes and block IDs for newly staged blocks
        let new_sequence_hashes: Vec<SequenceHash> =
            new_g2_blocks.iter().map(|b| b.sequence_hash()).collect();
        let new_block_ids: Vec<BlockId> = new_g2_blocks.iter().map(|b| b.block_id()).collect();

        // Release G3 blocks (take_all releases them) and hold new G2 blocks
        let _ = self.held_g3_blocks.take_all();
        self.held_g2_blocks.extend(new_g2_blocks);

        // TODO: Get layout handle from G2 manager
        let layout_handle = LayoutHandle::new(0, 0); // Placeholder

        // Send BlocksReady with only newly staged blocks
        let ready_msg = OnboardMessage::BlocksReady {
            responder: self.instance_id,
            session_id: self.session_id,
            layout_handle,
            sequence_hashes: new_sequence_hashes,
            block_ids: new_block_ids,
        };
        self.transport.send(self.requester, ready_msg).await?;

        Ok(())
    }
}
