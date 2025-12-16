// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RemoteSessionHandle: Handle for Prefill to control a remote session on Decode.
//!
//! # Deprecation Notice
//!
//! This module is being replaced by [`super::handle::SessionHandle`].
//! For new code, use `SessionHandle` instead of `RemoteSessionHandle`.
//!
//! Key differences:
//! - `SessionHandle` uses unified [`SessionPhase`] instead of `RemoteSessionPhase`
//! - `SessionHandle` uses [`SessionStateSnapshot`] instead of `RemoteSessionState`
//! - `SessionHandle` supports bidirectional control transfer
//!
//! # Legacy Documentation
//!
//! This implements the "Prefill side" of the inverted control pattern where:
//! 1. Decode creates a local session and sends session_id to Prefill
//! 2. Prefill uses this handle to attach and control the remote session
//! 3. Prefill can query state, trigger staging, and pull blocks via RDMA

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::watch;

use crate::physical::transfer::TransferCompleteNotification;
use crate::v2::distributed::parallelism::ParallelWorker;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::{BlockId, InstanceId, SequenceHash};

use super::{
    SessionId,
    messages::{G2BlockInfo, G3BlockInfo, RemoteSessionMessage, RemoteSessionPhase},
    transport::MessageTransport,
};

/// Handle for controlling a remote session from Prefill.
///
/// # Deprecation Notice
///
/// This struct is being replaced by [`super::SessionHandle`].
/// For new code, use `SessionHandle` instead.
///
/// # Legacy Usage
///
/// Created by `attach_remote_session()` on Prefill's InstanceLeader.
/// Provides methods to query state, trigger staging, and notify of pulled blocks.
///
/// ## Example
///
/// ```ignore
/// // Attach to remote session
/// let mut handle = prefill_leader.attach_remote_session(decode_id, session_id).await?;
///
/// // Wait for initial state
/// let state = handle.wait_for_initial_state().await?;
///
/// // Trigger staging if needed (idempotent)
/// if state.g3_pending_count > 0 {
///     handle.trigger_staging().await?;
///     handle.wait_for_staging_complete().await?;
/// }
///
/// // Get G2 blocks and pull via RDMA
/// let blocks = handle.get_g2_blocks();
/// // ... RDMA pull logic ...
///
/// // Notify remote that blocks were pulled
/// handle.mark_blocks_pulled(pulled_hashes).await?;
///
/// // Detach when done
/// handle.detach().await?;
/// ```
pub struct RemoteSessionHandle {
    session_id: SessionId,
    remote_instance: InstanceId,
    local_instance: InstanceId,
    transport: Arc<MessageTransport>,

    // Receive channel for state updates from Decode
    state_rx: watch::Receiver<RemoteSessionState>,

    // RDMA transfer support (optional - set via with_rdma_support)
    /// Parallel worker abstraction that handles RDMA transfers and metadata mapping.
    /// This encapsulates all parallelism knowledge - the handle doesn't need to know
    /// about individual workers or handle mappings.
    parallel_worker: Option<Arc<dyn ParallelWorker>>,
}

/// Current known state of the remote session.
#[derive(Debug, Clone, Default)]
pub struct RemoteSessionState {
    /// Current session phase.
    pub phase: RemoteSessionPhase,
    /// Blocks currently in G2 (ready for RDMA pull).
    pub g2_blocks: Vec<G2BlockInfo>,
    /// Count of blocks pending G3→G2 staging.
    pub g3_pending_count: usize,
    /// Optional: Full info for G3 blocks.
    pub g3_blocks: Option<Vec<G3BlockInfo>>,
}

impl RemoteSessionHandle {
    /// Create a new remote session handle.
    pub(crate) fn new(
        session_id: SessionId,
        remote_instance: InstanceId,
        local_instance: InstanceId,
        transport: Arc<MessageTransport>,
        state_rx: watch::Receiver<RemoteSessionState>,
    ) -> Self {
        Self {
            session_id,
            remote_instance,
            local_instance,
            transport,
            state_rx,
            parallel_worker: None,
        }
    }

    /// Add RDMA support to this handle.
    ///
    /// This enables the `pull_blocks_rdma` methods by providing access to
    /// the parallel worker abstraction.
    ///
    /// # Arguments
    /// * `parallel_worker` - Parallel worker for executing RDMA transfers
    pub fn with_rdma_support(mut self, parallel_worker: Arc<dyn ParallelWorker>) -> Self {
        self.parallel_worker = Some(parallel_worker);
        self
    }

    /// Wait for the initial state after attachment.
    ///
    /// Call this before issuing any commands to ensure the session
    /// is properly attached and you have the initial state.
    pub async fn wait_for_initial_state(&mut self) -> Result<RemoteSessionState> {
        self.state_rx
            .wait_for(|s| s.phase != RemoteSessionPhase::AwaitingAttachment)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to receive initial state: {}", e))?;
        Ok(self.state_rx.borrow().clone())
    }

    /// Get the current known state (non-blocking).
    pub fn current_state(&self) -> RemoteSessionState {
        self.state_rx.borrow().clone()
    }

    /// Check if the state has changed since last read.
    ///
    /// Returns true if there's new state available.
    pub fn has_changed(&self) -> bool {
        self.state_rx.has_changed().unwrap_or(false)
    }

    /// Wait for state to change.
    ///
    /// Returns the new state when it changes.
    pub async fn wait_for_change(&mut self) -> Result<RemoteSessionState> {
        self.state_rx
            .changed()
            .await
            .map_err(|e| anyhow::anyhow!("State channel closed: {}", e))?;
        Ok(self.state_rx.borrow().clone())
    }

    /// Trigger G3→G2 staging on the remote session.
    ///
    /// This is idempotent - calling it when already staging or staged is a no-op.
    pub async fn trigger_staging(&self) -> Result<()> {
        let msg = RemoteSessionMessage::TriggerStaging {
            controller: self.local_instance,
            session_id: self.session_id,
        };
        self.transport
            .send_remote_session(self.remote_instance, msg)
            .await
    }

    /// Wait for all G3 blocks to be staged to G2.
    ///
    /// Returns when the session phase becomes Ready or Complete.
    pub async fn wait_for_staging_complete(&mut self) -> Result<RemoteSessionState> {
        self.state_rx
            .wait_for(|s| {
                s.phase == RemoteSessionPhase::Ready || s.phase == RemoteSessionPhase::Complete
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed waiting for staging: {}", e))?;
        Ok(self.state_rx.borrow().clone())
    }

    /// Notify remote that blocks have been pulled and can be released.
    ///
    /// Call this after successfully pulling blocks via RDMA so the
    /// remote can release them from its hold.
    pub async fn mark_blocks_pulled(&self, pulled_hashes: Vec<SequenceHash>) -> Result<()> {
        let msg = RemoteSessionMessage::BlocksPulled {
            controller: self.local_instance,
            session_id: self.session_id,
            pulled_hashes,
        };
        self.transport
            .send_remote_session(self.remote_instance, msg)
            .await
    }

    /// Detach from the session (closes it on Decode side).
    ///
    /// This consumes the handle. After detaching, the session on Decode
    /// will release all remaining blocks and complete.
    pub async fn detach(self) -> Result<()> {
        let msg = RemoteSessionMessage::DetachSession {
            controller: self.local_instance,
            session_id: self.session_id,
        };
        self.transport
            .send_remote_session(self.remote_instance, msg)
            .await
    }

    /// Get blocks ready for RDMA pull.
    ///
    /// Returns a clone of the current G2 block info.
    pub fn get_g2_blocks(&self) -> Vec<G2BlockInfo> {
        self.state_rx.borrow().g2_blocks.clone()
    }

    /// Get the count of G3 blocks pending staging.
    pub fn g3_pending_count(&self) -> usize {
        self.state_rx.borrow().g3_pending_count
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the remote instance ID (Decode).
    pub fn remote_instance(&self) -> InstanceId {
        self.remote_instance
    }

    /// Get the local instance ID (Prefill).
    pub fn local_instance(&self) -> InstanceId {
        self.local_instance
    }

    /// Check if the session is complete.
    pub fn is_complete(&self) -> bool {
        self.state_rx.borrow().phase == RemoteSessionPhase::Complete
    }

    /// Check if the session is ready (all blocks in G2).
    pub fn is_ready(&self) -> bool {
        self.state_rx.borrow().phase == RemoteSessionPhase::Ready
    }

    // ========================================================================
    // RDMA Transfer Methods
    // ========================================================================

    /// Check if remote metadata has been imported.
    pub fn has_remote_metadata(&self) -> bool {
        self.parallel_worker
            .as_ref()
            .map(|pw| pw.has_remote_metadata(self.remote_instance))
            .unwrap_or(false)
    }

    /// Ensure remote metadata is imported (lazy loading).
    ///
    /// This requests and imports metadata from the remote instance if not
    /// already loaded. The metadata is cached so subsequent calls are no-ops.
    ///
    /// # Errors
    /// Returns an error if:
    /// - RDMA support was not configured (no parallel worker)
    /// - Metadata request fails
    /// - Worker count mismatch
    pub async fn ensure_metadata_imported(&mut self) -> Result<()> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("RDMA support not configured"))?;

        // Check if already loaded
        if parallel_worker.has_remote_metadata(self.remote_instance) {
            return Ok(());
        }

        // Request metadata from remote leader
        let remote_metadata = self
            .transport
            .request_metadata(self.remote_instance)
            .await?;

        // Connect to remote - this imports metadata and stores handle mappings
        parallel_worker
            .connect_remote(self.remote_instance, remote_metadata)?
            .await?;

        Ok(())
    }

    /// Pull blocks from remote G2 to local G2 via RDMA (lazy metadata loading).
    ///
    /// This method:
    /// 1. Ensures remote metadata is imported (if not already)
    /// 2. Executes SPMD-aware transfer: worker N pulls from remote worker N
    /// 3. Returns a notification that completes when all transfers done
    ///
    /// # Arguments
    /// * `blocks` - G2 block info from the remote session state
    /// * `local_dst_block_ids` - Local block IDs to transfer into
    ///
    /// # Returns
    /// TransferCompleteNotification that completes when all workers finish
    ///
    /// # Errors
    /// Returns an error if:
    /// - RDMA support not configured
    /// - Metadata import fails
    /// - Transfer execution fails
    pub async fn pull_blocks_rdma(
        &mut self,
        blocks: &[G2BlockInfo],
        local_dst_block_ids: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        // Ensure metadata is imported
        self.ensure_metadata_imported().await?;

        // Delegate to explicit version
        self.pull_blocks_rdma_explicit(blocks, local_dst_block_ids)
    }

    /// Pull blocks with explicit metadata pre-import (no lazy loading).
    ///
    /// Caller must have already ensured metadata is imported (via
    /// `ensure_metadata_imported()` or `leader.import_remote_metadata()`).
    ///
    /// # SPMD Behavior
    ///
    /// In SPMD (tensor parallelism), each worker holds a different tensor shard
    /// of the same logical block. This method executes the SAME transfer on
    /// EVERY worker: worker N pulls from remote worker N.
    ///
    /// # Arguments
    /// * `blocks` - G2 block info from the remote session state
    /// * `local_dst_block_ids` - Local block IDs to transfer into
    ///
    /// # Returns
    /// TransferCompleteNotification that completes when ALL workers finish
    ///
    /// # Errors
    /// Returns an error if:
    /// - RDMA support not configured
    /// - Metadata not imported
    /// - Transfer execution fails
    pub fn pull_blocks_rdma_explicit(
        &self,
        blocks: &[G2BlockInfo],
        local_dst_block_ids: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("RDMA support not configured"))?;

        // Validate metadata is loaded
        if !parallel_worker.has_remote_metadata(self.remote_instance) {
            anyhow::bail!(
                "Remote metadata not imported for instance {}. Call ensure_metadata_imported() first.",
                self.remote_instance
            );
        }

        // Validate block counts match
        if blocks.len() != local_dst_block_ids.len() {
            anyhow::bail!(
                "Block count mismatch: source={}, destination={}",
                blocks.len(),
                local_dst_block_ids.len()
            );
        }

        // Extract source block IDs
        let src_block_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id).collect();

        // Use the parallel worker's SPMD-aware transfer method
        // This handles all the worker iteration and handle lookup internally
        parallel_worker.execute_remote_onboard_for_instance(
            self.remote_instance,
            LogicalLayoutHandle::G2,
            src_block_ids,
            LogicalLayoutHandle::G2,
            local_dst_block_ids.to_vec().into(),
            Default::default(),
        )
    }
}

/// Sender for state updates to RemoteSessionHandle.
///
/// This is held by the receiver task that processes incoming
/// RemoteSessionMessage from the remote session.
pub struct RemoteSessionStateTx {
    tx: watch::Sender<RemoteSessionState>,
}

impl RemoteSessionStateTx {
    /// Create a new state sender.
    pub fn new(tx: watch::Sender<RemoteSessionState>) -> Self {
        Self { tx }
    }

    /// Update state with a SessionState message.
    pub fn update_from_session_state(
        &self,
        g2_blocks: Vec<G2BlockInfo>,
        g3_pending_count: usize,
        g3_blocks: Option<Vec<G3BlockInfo>>,
        phase: RemoteSessionPhase,
    ) {
        self.tx.send_modify(|state| {
            state.phase = phase;
            state.g2_blocks = g2_blocks;
            state.g3_pending_count = g3_pending_count;
            state.g3_blocks = g3_blocks;
        });
    }

    /// Update state with a BlocksStaged message.
    pub fn update_from_blocks_staged(
        &self,
        staged_blocks: Vec<G2BlockInfo>,
        g3_remaining_count: usize,
    ) {
        self.tx.send_modify(|state| {
            state.g2_blocks.extend(staged_blocks);
            state.g3_pending_count = g3_remaining_count;
            if g3_remaining_count == 0 {
                state.phase = RemoteSessionPhase::Ready;
            }
        });
    }

    /// Set an error state.
    #[allow(dead_code)]
    pub fn set_error(&self, _error: &str) {
        self.tx.send_modify(|state| {
            state.phase = RemoteSessionPhase::Complete;
        });
    }
}

/// Create a new remote session state channel.
pub fn remote_session_state_channel() -> (RemoteSessionStateTx, watch::Receiver<RemoteSessionState>)
{
    let (tx, rx) = watch::channel(RemoteSessionState::default());
    (RemoteSessionStateTx::new(tx), rx)
}
