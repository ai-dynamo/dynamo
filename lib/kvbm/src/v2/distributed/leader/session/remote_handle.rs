// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RemoteSessionHandle: Handle for Prefill to control a remote session on Decode.
//!
//! This implements the "Prefill side" of the inverted control pattern where:
//! 1. Decode creates a local session and sends session_id to Prefill
//! 2. Prefill uses this handle to attach and control the remote session
//! 3. Prefill can query state, trigger staging, and pull blocks

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::watch;

use crate::v2::{InstanceId, SequenceHash};

use super::{
    SessionId,
    messages::{G2BlockInfo, G3BlockInfo, RemoteSessionMessage, RemoteSessionPhase},
    transport::MessageTransport,
};

/// Handle for controlling a remote session from Prefill.
///
/// Created by `attach_remote_session()` on Prefill's InstanceLeader.
/// Provides methods to query state, trigger staging, and notify of pulled blocks.
///
/// ## Usage
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
        }
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

    // Note: RDMA pull implementation would go here, but requires Worker integration.
    // For now, the caller can use the G2BlockInfo to perform RDMA pulls externally.
    //
    // Future implementation:
    // pub async fn pull_blocks(
    //     &self,
    //     block_hashes: &[SequenceHash],
    //     worker: &dyn Worker,
    //     local_g2_manager: &BlockManager<G2>,
    // ) -> Result<Vec<ImmutableBlock<G2>>> {
    //     // 1. Look up remote G2BlockInfo for each hash
    //     // 2. Allocate local G2 blocks
    //     // 3. Use Worker.execute_remote_onboard() with RemoteDescriptor::Layout
    //     // 4. Register and return the new local blocks
    //     todo!()
    // }
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
