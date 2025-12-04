// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ControllableSession: A session on Decode that can be controlled remotely by Prefill.
//!
//! This implements the "inverted control pattern" where:
//! 1. Decode creates a local session (finds local matches, holds blocks)
//! 2. Decode sends the session_id to Prefill (out-of-band)
//! 3. Prefill attaches to the session on Decode via Nova
//! 4. Prefill controls the session remotely (queries state, triggers staging, pulls blocks)

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::{
    logical::{blocks::ImmutableBlock, manager::BlockManager},
    physical::transfer::TransferOptions,
    v2::{
        BlockId, G2, G3, InstanceId, SequenceHash, distributed::parallelism::ParallelWorker,
        logical::LogicalLayoutHandle, physical::manager::LayoutHandle,
    },
};

use super::{
    BlockHolder, SessionId,
    messages::{
        ControllableSessionOptions, G2BlockInfo, G3BlockInfo, RemoteSessionMessage,
        RemoteSessionPhase,
    },
    transport::MessageTransport,
};

/// A session on Decode that can be controlled remotely by Prefill.
///
/// Created via `create_controllable_session()` on Decode, then Prefill attaches
/// by calling `attach_remote_session()` with the session_id.
///
/// ## Lifecycle
///
/// 1. Created with held G2/G3 blocks from local search
/// 2. If `auto_stage=true` (default), immediately starts G3→G2 staging
/// 3. Waits for Prefill to attach via `AttachSession`
/// 4. Sends current state to Prefill
/// 5. Responds to control messages (TriggerStaging, BlocksPulled, DetachSession)
/// 6. Sends push notifications when blocks are staged
/// 7. Completes when all blocks pulled or session detached
pub struct ControllableSession {
    session_id: SessionId,
    #[allow(dead_code)] // Reserved for future use (e.g., logging, state reporting)
    instance_id: InstanceId,
    phase: RemoteSessionPhase,
    options: ControllableSessionOptions,

    // Held blocks using BlockHolder for RAII semantics
    // Blocks are automatically released when the session drops
    g2_blocks: BlockHolder<G2>,
    g3_blocks: BlockHolder<G3>,

    // G2 layout handles from workers (for RDMA source descriptors)
    // Blocks are allocated round-robin across workers
    worker_g2_handles: Vec<LayoutHandle>,

    // Controller info (set on attach)
    controller: Option<InstanceId>,

    // Transport for sending messages to controller
    transport: Arc<MessageTransport>,

    // Parallel worker for G3->G2 staging (fans out to all workers)
    parallel_worker: Option<Arc<dyn ParallelWorker>>,

    // Managers
    g2_manager: Arc<BlockManager<G2>>,
    #[allow(dead_code)]
    g3_manager: Option<Arc<BlockManager<G3>>>,

    // Message receive channel
    rx: mpsc::Receiver<RemoteSessionMessage>,

    // Track if staging is in progress or complete
    staging_started: bool,
    staging_complete: bool,
}

impl ControllableSession {
    /// Create a new controllable session.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: SessionId,
        instance_id: InstanceId,
        g2_blocks: Vec<ImmutableBlock<G2>>,
        g3_blocks: Vec<ImmutableBlock<G3>>,
        worker_g2_handles: Vec<LayoutHandle>,
        g2_manager: Arc<BlockManager<G2>>,
        g3_manager: Option<Arc<BlockManager<G3>>>,
        parallel_worker: Option<Arc<dyn ParallelWorker>>,
        transport: Arc<MessageTransport>,
        rx: mpsc::Receiver<RemoteSessionMessage>,
        options: ControllableSessionOptions,
    ) -> Self {
        Self {
            session_id,
            instance_id,
            phase: RemoteSessionPhase::AwaitingAttachment,
            options,
            g2_blocks: BlockHolder::new(g2_blocks),
            g3_blocks: BlockHolder::new(g3_blocks),
            worker_g2_handles,
            controller: None,
            transport,
            parallel_worker,
            g2_manager,
            g3_manager,
            rx,
            staging_started: false,
            staging_complete: false,
        }
    }

    /// Run the controllable session.
    ///
    /// This is the main session loop that:
    /// 1. Optionally auto-stages G3→G2
    /// 2. Waits for controller attachment
    /// 3. Processes control messages until detachment or completion
    pub async fn run(mut self) -> Result<()> {
        // If auto_stage is enabled and we have G3 blocks, start staging immediately
        if self.options.auto_stage && !self.g3_blocks.is_empty() && self.parallel_worker.is_some() {
            self.phase = RemoteSessionPhase::Staging;
            self.staging_started = true;

            // Note: We start staging but don't send notifications yet
            // because there's no controller attached. We'll send the full
            // state when the controller attaches.
            self.stage_g3_to_g2_internal().await?;
        }

        // Update phase based on staging status
        self.update_phase();

        // Process messages
        while let Some(msg) = self.rx.recv().await {
            match msg {
                RemoteSessionMessage::AttachSession { controller, .. } => {
                    self.handle_attach(controller).await?;
                }
                RemoteSessionMessage::TriggerStaging { .. } => {
                    self.handle_trigger_staging().await?;
                }
                RemoteSessionMessage::BlocksPulled { pulled_hashes, .. } => {
                    self.handle_blocks_pulled(&pulled_hashes)?;
                    if self.phase == RemoteSessionPhase::Complete {
                        break;
                    }
                }
                RemoteSessionMessage::DetachSession { .. } => {
                    self.phase = RemoteSessionPhase::Complete;
                    break;
                }
                _ => {
                    // Ignore unexpected messages (SessionState, BlocksStaged, SessionError are outbound)
                    eprintln!(
                        "ControllableSession: unexpected message for session {}: {:?}",
                        self.session_id, msg
                    );
                }
            }
        }

        Ok(())
    }

    /// Handle controller attachment.
    async fn handle_attach(&mut self, controller: InstanceId) -> Result<()> {
        self.controller = Some(controller);

        // Update phase
        if self.phase == RemoteSessionPhase::AwaitingAttachment {
            self.phase = RemoteSessionPhase::Attached;
        }
        self.update_phase();

        // Send current state to controller
        let state = self.build_session_state();
        self.transport.send_remote_session(controller, state).await
    }

    /// Handle trigger staging request.
    ///
    /// This is idempotent - calling it when already staging or staged is a no-op.
    async fn handle_trigger_staging(&mut self) -> Result<()> {
        // Idempotent: skip if already started or complete
        if self.staging_started {
            return Ok(());
        }

        // Skip if no G3 blocks
        if self.g3_blocks.is_empty() {
            return Ok(());
        }

        // Skip if no parallel worker
        if self.parallel_worker.is_none() {
            if let Some(controller) = self.controller {
                let error_msg = RemoteSessionMessage::SessionError {
                    session_id: self.session_id,
                    error: "No parallel worker available for G3->G2 staging".to_string(),
                };
                self.transport
                    .send_remote_session(controller, error_msg)
                    .await?;
            }
            return Ok(());
        }

        self.phase = RemoteSessionPhase::Staging;
        self.staging_started = true;

        // Execute staging
        let staged_info = self.stage_g3_to_g2_internal().await?;

        // Update phase
        self.update_phase();

        // Notify controller of newly staged blocks (if attached)
        if let Some(controller) = self.controller {
            let msg = RemoteSessionMessage::BlocksStaged {
                session_id: self.session_id,
                staged_blocks: staged_info,
                g3_remaining_count: self.g3_blocks.count(),
            };
            self.transport.send_remote_session(controller, msg).await?;
        }

        Ok(())
    }

    /// Handle blocks pulled notification.
    fn handle_blocks_pulled(&mut self, pulled_hashes: &[SequenceHash]) -> Result<()> {
        // Release pulled blocks using BlockHolder's release method
        self.g2_blocks.release(pulled_hashes);

        // Check if session is complete
        if self.g2_blocks.is_empty() && self.g3_blocks.is_empty() {
            self.phase = RemoteSessionPhase::Complete;
        }

        Ok(())
    }

    /// Update phase based on current state.
    fn update_phase(&mut self) {
        // Don't change phase if already complete
        if self.phase == RemoteSessionPhase::Complete {
            return;
        }

        // If staging is complete and no G3 blocks remain, we're ready
        if self.staging_complete && self.g3_blocks.is_empty() {
            self.phase = RemoteSessionPhase::Ready;
        } else if self.staging_started && !self.staging_complete {
            self.phase = RemoteSessionPhase::Staging;
        }
    }

    /// Build current session state message.
    fn build_session_state(&self) -> RemoteSessionMessage {
        // Allocate blocks round-robin across workers
        let g2_blocks: Vec<G2BlockInfo> = self
            .g2_blocks
            .blocks()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // Use round-robin allocation if we have worker handles
                let layout_handle = if !self.worker_g2_handles.is_empty() {
                    self.worker_g2_handles[i % self.worker_g2_handles.len()]
                } else {
                    // Fallback to placeholder if no workers (shouldn't happen in RDMA mode)
                    LayoutHandle::new(0, 0)
                };

                G2BlockInfo {
                    block_id: b.block_id(),
                    sequence_hash: b.sequence_hash(),
                    layout_handle,
                }
            })
            .collect();

        let g3_blocks: Option<Vec<G3BlockInfo>> = if self.g3_blocks.is_empty() {
            None
        } else {
            Some(
                self.g3_blocks
                    .blocks()
                    .iter()
                    .map(|b| G3BlockInfo {
                        sequence_hash: b.sequence_hash(),
                    })
                    .collect(),
            )
        };

        RemoteSessionMessage::SessionState {
            session_id: self.session_id,
            g2_blocks,
            g3_pending_count: self.g3_blocks.count(),
            g3_blocks,
            phase: self.phase,
        }
    }

    /// Stage G3 blocks to G2 (internal implementation).
    ///
    /// Returns info about the newly staged blocks.
    async fn stage_g3_to_g2_internal(&mut self) -> Result<Vec<G2BlockInfo>> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorker required for G3->G2 staging"))?;

        if self.g3_blocks.is_empty() {
            self.staging_complete = true;
            return Ok(Vec::new());
        }

        let stage_block_ids: Vec<BlockId> = self
            .g3_blocks
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
        let notification = parallel_worker.execute_local_transfer(
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
            .zip(self.g3_blocks.blocks().iter())
            .map(|(dst, src)| {
                self.g2_manager
                    .register_mutable_block_from_existing(dst, src)
            })
            .collect();

        // Build result info for newly staged blocks
        // Continue round-robin allocation from where existing g2_blocks left off
        let starting_index = self.g2_blocks.count();
        let staged_info: Vec<G2BlockInfo> = new_g2_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // Use round-robin allocation if we have worker handles
                let layout_handle = if !self.worker_g2_handles.is_empty() {
                    let idx = (starting_index + i) % self.worker_g2_handles.len();
                    self.worker_g2_handles[idx]
                } else {
                    // Fallback to placeholder if no workers
                    LayoutHandle::new(0, 0)
                };

                G2BlockInfo {
                    block_id: b.block_id(),
                    sequence_hash: b.sequence_hash(),
                    layout_handle,
                }
            })
            .collect();

        // Clear G3 blocks (take_all releases them) and add new G2 blocks
        let _ = self.g3_blocks.take_all();
        self.g2_blocks.extend(new_g2_blocks);

        self.staging_complete = true;

        Ok(staged_info)
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }
}

/// Result of creating a controllable session.
#[derive(Debug, Clone)]
pub struct ControllableSessionResult {
    /// The unique session ID.
    pub session_id: SessionId,
    /// Number of G2 blocks found.
    pub local_g2_count: usize,
    /// Number of G3 blocks found.
    pub local_g3_count: usize,
}
