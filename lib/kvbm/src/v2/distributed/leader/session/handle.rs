// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SessionHandle: Unified handle for controlling a remote session.
//!
//! This is the unified replacement for `RemoteSessionHandle` that uses the
//! new session model types (`SessionPhase`, `ControlRole`, `SessionStateSnapshot`).
//!
//! Key improvements over RemoteSessionHandle:
//! - Uses unified `SessionPhase` and `ControlRole` enums
//! - Supports bidirectional control transfer (yield/acquire)
//! - Uses `SessionStateSnapshot` for state observation
//! - Same RDMA support via `ParallelWorker`

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::watch;

use crate::physical::transfer::{TransferCompleteNotification, TransferOptions};
use crate::v2::distributed::parallelism::ParallelWorker;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::{BlockId, InstanceId, SequenceHash};

use super::{
    BlockInfo, ControlRole, SessionId, SessionMessage, SessionPhase, SessionStateSnapshot,
    transport::MessageTransport,
};

/// Handle for controlling a remote session.
///
/// Created by attaching to a remote session. Provides methods to:
/// - Query and observe session state
/// - Issue control commands (trigger staging, release blocks)
/// - Transfer control bidirectionally (yield/acquire)
/// - Pull blocks via RDMA
///
/// ## Usage
///
/// ```ignore
/// // Attach to remote session
/// let mut handle = leader.attach_session(remote_id, session_id).await?;
///
/// // Wait for initial state
/// let state = handle.wait_for_ready().await?;
///
/// // Trigger staging if needed
/// if state.g3_pending > 0 {
///     handle.trigger_staging().await?;
///     handle.wait_for_ready().await?;
/// }
///
/// // Pull blocks via RDMA
/// let notification = handle.pull_blocks_rdma(&state.g2_blocks, &local_block_ids).await?;
/// notification.await?;
///
/// // Notify remote and detach
/// handle.mark_blocks_pulled(hashes).await?;
/// handle.detach().await?;
/// ```
pub struct SessionHandle {
    session_id: SessionId,
    remote_instance: InstanceId,
    local_instance: InstanceId,
    transport: Arc<MessageTransport>,

    // State observation
    state_rx: watch::Receiver<SessionStateSnapshot>,

    // RDMA transfer support
    parallel_worker: Option<Arc<dyn ParallelWorker>>,
}

impl SessionHandle {
    /// Create a new session handle.
    ///
    /// Note: Currently unused during incremental migration. Will be used once
    /// existing session implementations are fully migrated to the new model.
    #[allow(dead_code)]
    pub(crate) fn new(
        session_id: SessionId,
        remote_instance: InstanceId,
        local_instance: InstanceId,
        transport: Arc<MessageTransport>,
        state_rx: watch::Receiver<SessionStateSnapshot>,
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
    pub fn with_rdma_support(mut self, parallel_worker: Arc<dyn ParallelWorker>) -> Self {
        self.parallel_worker = Some(parallel_worker);
        self
    }

    // =========================================================================
    // Identity
    // =========================================================================

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the remote instance ID.
    pub fn remote_instance(&self) -> InstanceId {
        self.remote_instance
    }

    /// Get the local instance ID.
    pub fn local_instance(&self) -> InstanceId {
        self.local_instance
    }

    // =========================================================================
    // State Observation
    // =========================================================================

    /// Get the current state snapshot (non-blocking).
    pub fn current_state(&self) -> SessionStateSnapshot {
        self.state_rx.borrow().clone()
    }

    /// Get the current phase.
    pub fn phase(&self) -> SessionPhase {
        self.state_rx.borrow().phase
    }

    /// Get the current control role of the remote session.
    pub fn remote_control_role(&self) -> ControlRole {
        self.state_rx.borrow().control_role
    }

    /// Check if state has changed since last read.
    pub fn has_changed(&self) -> bool {
        self.state_rx.has_changed().unwrap_or(false)
    }

    /// Wait for state to change.
    pub async fn wait_for_change(&mut self) -> Result<SessionStateSnapshot> {
        self.state_rx
            .changed()
            .await
            .map_err(|e| anyhow::anyhow!("State channel closed: {}", e))?;
        Ok(self.state_rx.borrow().clone())
    }

    /// Wait for the session to reach Ready phase (all blocks in G2).
    pub async fn wait_for_ready(&mut self) -> Result<SessionStateSnapshot> {
        self.state_rx
            .wait_for(|s| s.phase == SessionPhase::Ready || s.phase.is_terminal())
            .await
            .map_err(|e| anyhow::anyhow!("Failed waiting for ready: {}", e))?;

        let state = self.state_rx.borrow().clone();
        if state.phase == SessionPhase::Failed {
            anyhow::bail!("Session failed while waiting for ready");
        }
        Ok(state)
    }

    /// Wait for the session to complete.
    pub async fn wait_for_complete(&mut self) -> Result<SessionStateSnapshot> {
        self.state_rx
            .wait_for(|s| s.phase.is_terminal())
            .await
            .map_err(|e| anyhow::anyhow!("Failed waiting for complete: {}", e))?;
        Ok(self.state_rx.borrow().clone())
    }

    /// Check if the session is complete.
    pub fn is_complete(&self) -> bool {
        self.state_rx.borrow().phase.is_terminal()
    }

    /// Check if the session is ready (all blocks in G2).
    pub fn is_ready(&self) -> bool {
        self.state_rx.borrow().phase == SessionPhase::Ready
    }

    /// Get G2 blocks from current state.
    pub fn get_g2_blocks(&self) -> Vec<BlockInfo> {
        self.state_rx.borrow().g2_blocks.clone()
    }

    /// Get count of G3 blocks pending staging.
    pub fn g3_pending_count(&self) -> usize {
        self.state_rx.borrow().g3_pending
    }

    /// Get the layer range that is ready for transfer.
    ///
    /// Returns `None` if all layers are ready or layerwise tracking is not active.
    /// Returns `Some(range)` if only specific layers are ready.
    pub fn ready_layer_range(&self) -> Option<std::ops::Range<usize>> {
        self.state_rx.borrow().ready_layer_range.clone()
    }

    // =========================================================================
    // Control Commands
    // =========================================================================

    /// Trigger G3→G2 staging on the remote session.
    ///
    /// Idempotent - no-op if already staging or staged.
    pub async fn trigger_staging(&self) -> Result<()> {
        let msg = SessionMessage::TriggerStaging {
            session_id: self.session_id,
        };
        self.transport.send_session(self.remote_instance, msg).await
    }

    /// Notify remote that blocks have been pulled.
    ///
    /// Call after successfully pulling blocks via RDMA.
    pub async fn mark_blocks_pulled(&self, pulled_hashes: Vec<SequenceHash>) -> Result<()> {
        let msg = SessionMessage::BlocksPulled {
            session_id: self.session_id,
            pulled_hashes,
        };
        self.transport.send_session(self.remote_instance, msg).await
    }

    /// Detach from the session.
    ///
    /// Consumes the handle. The remote session will release remaining blocks.
    pub async fn detach(self) -> Result<()> {
        let msg = SessionMessage::Detach {
            peer: self.local_instance,
            session_id: self.session_id,
        };
        self.transport.send_session(self.remote_instance, msg).await
    }

    // =========================================================================
    // Control Transfer (Bidirectional)
    // =========================================================================

    /// Yield control to the remote peer.
    ///
    /// After yielding, this handle transitions to Neutral and the remote
    /// can acquire control if desired.
    pub async fn yield_control(&self) -> Result<()> {
        let msg = SessionMessage::YieldControl {
            peer: self.local_instance,
            session_id: self.session_id,
        };
        self.transport.send_session(self.remote_instance, msg).await
    }

    /// Attempt to acquire control from the remote peer.
    ///
    /// Valid when remote is in Neutral state.
    pub async fn acquire_control(&self) -> Result<()> {
        let msg = SessionMessage::AcquireControl {
            peer: self.local_instance,
            session_id: self.session_id,
        };
        self.transport.send_session(self.remote_instance, msg).await
    }

    // =========================================================================
    // RDMA Transfer Methods
    // =========================================================================

    /// Check if remote metadata has been imported.
    pub fn has_remote_metadata(&self) -> bool {
        self.parallel_worker
            .as_ref()
            .map(|pw| pw.has_remote_metadata(self.remote_instance))
            .unwrap_or(false)
    }

    /// Ensure remote metadata is imported (lazy loading).
    pub async fn ensure_metadata_imported(&mut self) -> Result<()> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("RDMA support not configured"))?;

        if parallel_worker.has_remote_metadata(self.remote_instance) {
            return Ok(());
        }

        let remote_metadata = self
            .transport
            .request_metadata(self.remote_instance)
            .await?;

        parallel_worker
            .connect_remote(self.remote_instance, remote_metadata)?
            .await?;

        Ok(())
    }

    /// Pull blocks from remote G2 to local G2 via RDMA.
    ///
    /// This method:
    /// 1. Ensures remote metadata is imported
    /// 2. Executes SPMD-aware transfer (worker N pulls from remote worker N)
    /// 3. Returns notification that completes when all transfers done
    pub async fn pull_blocks_rdma(
        &mut self,
        blocks: &[BlockInfo],
        local_dst_block_ids: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        self.ensure_metadata_imported().await?;
        self.pull_blocks_rdma_explicit(blocks, local_dst_block_ids)
    }

    /// Pull blocks with explicit metadata pre-import.
    ///
    /// Caller must have already ensured metadata is imported.
    pub fn pull_blocks_rdma_explicit(
        &self,
        blocks: &[BlockInfo],
        local_dst_block_ids: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("RDMA support not configured"))?;

        if !parallel_worker.has_remote_metadata(self.remote_instance) {
            anyhow::bail!(
                "Remote metadata not imported for instance {}",
                self.remote_instance
            );
        }

        if blocks.len() != local_dst_block_ids.len() {
            anyhow::bail!(
                "Block count mismatch: source={}, destination={}",
                blocks.len(),
                local_dst_block_ids.len()
            );
        }

        let src_block_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id).collect();

        parallel_worker.execute_remote_onboard_for_instance(
            self.remote_instance,
            LogicalLayoutHandle::G2,
            src_block_ids,
            LogicalLayoutHandle::G2,
            local_dst_block_ids.to_vec().into(),
            Default::default(),
        )
    }

    /// Pull blocks from remote G2 to local G2 via RDMA with transfer options.
    ///
    /// This method allows specifying transfer options like layer range for
    /// layerwise transfer. Use this when you only want to pull specific layers.
    ///
    /// # Example
    /// ```ignore
    /// // Pull only layer 0
    /// let notification = handle.pull_blocks_rdma_with_options(
    ///     &state.g2_blocks,
    ///     &local_block_ids,
    ///     TransferOptions::builder().layer_range(0..1).build(),
    /// ).await?;
    /// notification.await?;
    /// ```
    pub async fn pull_blocks_rdma_with_options(
        &mut self,
        blocks: &[BlockInfo],
        local_dst_block_ids: &[BlockId],
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.ensure_metadata_imported().await?;
        self.pull_blocks_rdma_with_options_explicit(blocks, local_dst_block_ids, options)
    }

    /// Pull blocks with options and explicit metadata pre-import.
    ///
    /// Caller must have already ensured metadata is imported.
    pub fn pull_blocks_rdma_with_options_explicit(
        &self,
        blocks: &[BlockInfo],
        local_dst_block_ids: &[BlockId],
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("RDMA support not configured"))?;

        if !parallel_worker.has_remote_metadata(self.remote_instance) {
            anyhow::bail!(
                "Remote metadata not imported for instance {}",
                self.remote_instance
            );
        }

        if blocks.len() != local_dst_block_ids.len() {
            anyhow::bail!(
                "Block count mismatch: source={}, destination={}",
                blocks.len(),
                local_dst_block_ids.len()
            );
        }

        let src_block_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id).collect();

        parallel_worker.execute_remote_onboard_for_instance(
            self.remote_instance,
            LogicalLayoutHandle::G2,
            src_block_ids,
            LogicalLayoutHandle::G2,
            local_dst_block_ids.to_vec().into(),
            options,
        )
    }
}

/// Sender for state updates to SessionHandle.
pub struct SessionHandleStateTx {
    tx: watch::Sender<SessionStateSnapshot>,
}

impl SessionHandleStateTx {
    /// Create a new state sender.
    pub fn new(tx: watch::Sender<SessionStateSnapshot>) -> Self {
        Self { tx }
    }

    /// Update state from a full snapshot.
    pub fn update(&self, state: SessionStateSnapshot) {
        let _ = self.tx.send(state);
    }

    /// Update phase only.
    pub fn set_phase(&self, phase: SessionPhase) {
        self.tx.send_modify(|state| {
            state.phase = phase;
        });
    }

    /// Update G2 blocks.
    pub fn set_g2_blocks(&self, blocks: Vec<BlockInfo>) {
        self.tx.send_modify(|state| {
            state.g2_blocks = blocks;
        });
    }

    /// Add newly staged blocks.
    ///
    /// # Arguments
    /// * `staged` - Blocks that have been staged
    /// * `g3_remaining` - Count of G3 blocks still pending
    /// * `layer_range` - Optional layer range that is ready for transfer
    pub fn add_staged_blocks(
        &self,
        staged: Vec<BlockInfo>,
        g3_remaining: usize,
        layer_range: Option<std::ops::Range<usize>>,
    ) {
        self.tx.send_modify(|state| {
            state.g2_blocks.extend(staged);
            state.g3_pending = g3_remaining;
            state.ready_layer_range = layer_range;
            if g3_remaining == 0 && state.ready_layer_range.is_none() {
                // All blocks staged and no layer tracking = fully ready
                state.phase = SessionPhase::Ready;
            }
        });
    }

    /// Set error/failed state.
    pub fn set_failed(&self) {
        self.tx.send_modify(|state| {
            state.phase = SessionPhase::Failed;
        });
    }
}

/// Create a new session handle state channel.
pub fn session_handle_state_channel()
-> (SessionHandleStateTx, watch::Receiver<SessionStateSnapshot>) {
    let initial = SessionStateSnapshot {
        phase: SessionPhase::Searching,
        control_role: ControlRole::Controllee,
        g2_blocks: Vec::new(),
        g3_pending: 0,
        ready_layer_range: None,
    };
    let (tx, rx) = watch::channel(initial);
    (SessionHandleStateTx::new(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashmap::DashMap;

    fn create_test_transport() -> Arc<MessageTransport> {
        Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        ))
    }

    #[test]
    fn test_session_handle_state_channel() {
        let (tx, rx) = session_handle_state_channel();

        // Initial state
        let state = rx.borrow().clone();
        assert_eq!(state.phase, SessionPhase::Searching);
        assert_eq!(state.control_role, ControlRole::Controllee);
        assert!(state.g2_blocks.is_empty());

        // Update state
        tx.set_phase(SessionPhase::Ready);
        let state = rx.borrow().clone();
        assert_eq!(state.phase, SessionPhase::Ready);
    }

    #[test]
    fn test_session_handle_creation() {
        let (_, rx) = session_handle_state_channel();
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let remote_id = InstanceId::new_v4();
        let local_id = InstanceId::new_v4();

        let handle = SessionHandle::new(session_id, remote_id, local_id, transport, rx);

        assert_eq!(handle.session_id(), session_id);
        assert_eq!(handle.remote_instance(), remote_id);
        assert_eq!(handle.local_instance(), local_id);
        assert_eq!(handle.phase(), SessionPhase::Searching);
        assert!(!handle.is_ready());
        assert!(!handle.is_complete());
        assert!(!handle.has_remote_metadata());
    }

    #[tokio::test]
    async fn test_wait_for_ready() {
        let (tx, rx) = session_handle_state_channel();
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();

        let mut handle = SessionHandle::new(
            session_id,
            InstanceId::new_v4(),
            InstanceId::new_v4(),
            transport,
            rx,
        );

        // Spawn task to update state
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            tx.set_phase(SessionPhase::Ready);
        });

        let state = handle.wait_for_ready().await.unwrap();
        assert_eq!(state.phase, SessionPhase::Ready);
    }
}
