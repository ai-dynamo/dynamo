// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer handle and status tracking for offload operations.
//!
//! The `TransferHandle` is the user-facing interface for tracking and controlling
//! an offload transfer. It provides:
//! - Status tracking (Evaluating, Queued, Transferring, Complete, Cancelled)
//! - Block visibility (passed, completed, remaining)
//! - Cancellation with confirmation

use std::collections::HashSet;

use anyhow::Result;
use tokio::sync::watch;
use uuid::Uuid;

use crate::v2::BlockId;

use super::cancel::{CancelConfirmation, CancelStateUpdater, CancellationToken};

/// Unique identifier for a transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(Uuid);

impl TransferId {
    /// Create a new random transfer ID.
    pub fn new() -> Self {
        TransferId(Uuid::new_v4())
    }

    /// Get the underlying UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TransferId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TransferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for TransferId {
    fn from(uuid: Uuid) -> Self {
        TransferId(uuid)
    }
}

/// Status of a transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    /// Policy/filter evaluation in progress
    Evaluating,
    /// Passed filters, waiting in batch queue
    Queued,
    /// Transfer operation in progress
    Transferring,
    /// Transfer completed successfully
    Complete,
    /// Transfer was cancelled
    Cancelled,
    /// Transfer failed with error
    Failed,
}

impl TransferStatus {
    /// Check if the transfer is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TransferStatus::Complete | TransferStatus::Cancelled | TransferStatus::Failed
        )
    }

    /// Check if the transfer is still in progress.
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

/// Result of a completed transfer.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Transfer ID
    pub id: TransferId,
    /// Final status
    pub status: TransferStatus,
    /// Blocks that passed all filters
    pub passed_blocks: Vec<BlockId>,
    /// Blocks successfully transferred
    pub completed_blocks: Vec<BlockId>,
    /// Blocks that were filtered out
    pub filtered_blocks: Vec<BlockId>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Handle for tracking and controlling an offload transfer.
///
/// Obtained from `OffloadEngine::enqueue()`. Use this to:
/// - Monitor transfer progress via `status()`, `passed_blocks()`, etc.
/// - Cancel the transfer via `cancel()` and await confirmation
/// - Wait for completion via `wait()`
#[derive(Clone)]
pub struct TransferHandle {
    id: TransferId,
    status_rx: watch::Receiver<TransferStatus>,
    passed_blocks_rx: watch::Receiver<Vec<BlockId>>,
    completed_rx: watch::Receiver<Vec<BlockId>>,
    remaining_rx: watch::Receiver<Vec<BlockId>>,
    cancel_token: CancellationToken,
    result_rx: watch::Receiver<Option<TransferResult>>,
}

impl TransferHandle {
    /// Get the transfer ID.
    pub fn id(&self) -> TransferId {
        self.id
    }

    /// Get the current transfer status.
    pub fn status(&self) -> TransferStatus {
        *self.status_rx.borrow()
    }

    /// Get blocks that passed all filter policies.
    pub fn passed_blocks(&self) -> Vec<BlockId> {
        self.passed_blocks_rx.borrow().clone()
    }

    /// Get blocks that have been successfully transferred.
    pub fn completed_blocks(&self) -> Vec<BlockId> {
        self.completed_rx.borrow().clone()
    }

    /// Get blocks remaining to be transferred.
    pub fn remaining_blocks(&self) -> Vec<BlockId> {
        self.remaining_rx.borrow().clone()
    }

    /// Check if the transfer is complete (success, cancelled, or failed).
    pub fn is_complete(&self) -> bool {
        self.status().is_terminal()
    }

    /// Cancel the transfer and await confirmation.
    ///
    /// Returns a future that resolves when all blocks are confirmed released
    /// with no outstanding operations.
    ///
    /// # Example
    /// ```ignore
    /// // Request cancellation and wait for confirmation
    /// handle.cancel().await;
    /// // All blocks are now released
    /// ```
    pub fn cancel(&self) -> CancelConfirmation {
        self.cancel_token.request();
        self.cancel_token.wait_confirmed()
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_requested()
    }

    /// Wait for the transfer to complete.
    ///
    /// Returns the final `TransferResult` when the transfer reaches a terminal state.
    pub async fn wait(&mut self) -> Result<TransferResult> {
        // Wait until we have a result
        loop {
            {
                let result = self.result_rx.borrow();
                if let Some(r) = result.as_ref() {
                    return Ok(r.clone());
                }
            }

            if self.result_rx.changed().await.is_err() {
                // Channel closed without result
                return Err(anyhow::anyhow!("Transfer channel closed unexpectedly"));
            }
        }
    }

    /// Subscribe to status changes.
    pub fn subscribe_status(&self) -> watch::Receiver<TransferStatus> {
        self.status_rx.clone()
    }
}

impl std::fmt::Debug for TransferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferHandle")
            .field("id", &self.id)
            .field("status", &self.status())
            .field("passed_count", &self.passed_blocks().len())
            .field("completed_count", &self.completed_blocks().len())
            .field("remaining_count", &self.remaining_blocks().len())
            .finish()
    }
}

/// Internal state for tracking a transfer through the pipeline.
#[allow(dead_code)]
pub(crate) struct TransferState {
    pub(crate) id: TransferId,
    /// Current phase
    pub(crate) status: TransferStatus,
    /// Original input block IDs
    pub(crate) input_blocks: Vec<BlockId>,
    /// Blocks that passed policy filters
    pub(crate) passed_blocks: Vec<BlockId>,
    /// Blocks currently in-flight (being transferred)
    pub(crate) in_flight: HashSet<BlockId>,
    /// Successfully transferred blocks
    pub(crate) completed: Vec<BlockId>,
    /// Blocks that failed filters
    pub(crate) filtered_out: Vec<BlockId>,
    /// Error message if failed
    pub(crate) error: Option<String>,
    /// Notifier channels
    pub(crate) notifiers: TransferNotifiers,
    /// Cancel state updater
    pub(crate) cancel_updater: CancelStateUpdater,
    /// Total blocks expected in this transfer (set by PolicyEvaluator)
    pub(crate) total_expected_blocks: usize,
    /// Blocks that have been processed through policy evaluation (for sentinel flush)
    pub(crate) blocks_processed: usize,
    /// Precondition event that must be satisfied before processing this transfer.
    /// Set by the caller when enqueuing offload operations. BatchCollector will
    /// attach this to the TransferBatch, and PreconditionAwaiter will await it
    /// before forwarding to TransferExecutor.
    pub(crate) precondition: Option<dynamo_nova::events::EventHandle>,
}

#[allow(dead_code)]
impl TransferState {
    /// Create transfer state and associated handle.
    pub(crate) fn new(id: TransferId, input_blocks: Vec<BlockId>) -> (Self, TransferHandle) {
        let (status_tx, status_rx) = watch::channel(TransferStatus::Evaluating);
        let (passed_tx, passed_rx) = watch::channel(Vec::new());
        let (completed_tx, completed_rx) = watch::channel(Vec::new());
        let (remaining_tx, remaining_rx) = watch::channel(input_blocks.clone());
        let (result_tx, result_rx) = watch::channel(None);
        let (cancel_token, cancel_updater) = CancellationToken::new();

        let notifiers = TransferNotifiers {
            status_tx,
            passed_tx,
            completed_tx,
            remaining_tx,
            result_tx,
        };

        let state = TransferState {
            id,
            status: TransferStatus::Evaluating,
            input_blocks: input_blocks.clone(),
            passed_blocks: Vec::new(),
            in_flight: HashSet::new(),
            completed: Vec::new(),
            filtered_out: Vec::new(),
            error: None,
            notifiers,
            cancel_updater,
            total_expected_blocks: 0, // Set by PolicyEvaluator when transfer starts
            blocks_processed: 0,
            precondition: None, // Set by caller via enqueue_with_precondition
        };

        let handle = TransferHandle {
            id,
            status_rx,
            passed_blocks_rx: passed_rx,
            completed_rx,
            remaining_rx,
            cancel_token,
            result_rx,
        };

        (state, handle)
    }

    /// Check if cancellation has been requested.
    pub(crate) fn is_cancel_requested(&self) -> bool {
        self.cancel_updater.is_requested()
    }

    /// Update status and notify.
    pub(crate) fn set_status(&mut self, status: TransferStatus) {
        self.status = status;
        let _ = self.notifiers.status_tx.send(status);
    }

    /// Add blocks that passed filters.
    pub(crate) fn add_passed(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.passed_blocks.extend(block_ids);
        let _ = self.notifiers.passed_tx.send(self.passed_blocks.clone());
        self.update_remaining();
    }

    /// Add blocks that were filtered out.
    pub(crate) fn add_filtered(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.filtered_out.extend(block_ids);
        self.update_remaining();
    }

    /// Mark blocks as in-flight (being transferred).
    pub(crate) fn mark_in_flight(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.in_flight.extend(block_ids);
    }

    /// Mark blocks as completed (transferred successfully).
    pub(crate) fn mark_completed(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        for id in block_ids {
            self.in_flight.remove(&id);
            self.completed.push(id);
        }
        let _ = self.notifiers.completed_tx.send(self.completed.clone());
        self.update_remaining();
    }

    /// Update remaining blocks notification.
    fn update_remaining(&self) {
        let remaining: Vec<BlockId> = self
            .passed_blocks
            .iter()
            .filter(|id| !self.completed.contains(id))
            .copied()
            .collect();
        let _ = self.notifiers.remaining_tx.send(remaining);
    }

    /// Set error and mark as failed.
    pub(crate) fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.set_status(TransferStatus::Failed);
        self.finalize();
    }

    /// Mark as cancelled.
    pub(crate) fn set_cancelled(&mut self) {
        self.set_status(TransferStatus::Cancelled);
        self.cancel_updater.set_confirmed();
        self.finalize();
    }

    /// Mark as complete (all blocks transferred).
    pub(crate) fn set_complete(&mut self) {
        self.set_status(TransferStatus::Complete);
        self.finalize();
    }

    /// Finalize and send result.
    fn finalize(&mut self) {
        let result = TransferResult {
            id: self.id,
            status: self.status,
            passed_blocks: self.passed_blocks.clone(),
            completed_blocks: self.completed.clone(),
            filtered_blocks: self.filtered_out.clone(),
            error: self.error.clone(),
        };
        let _ = self.notifiers.result_tx.send(Some(result));
    }

    /// Get current in-flight count (for draining).
    pub(crate) fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Begin draining (cancellation in progress).
    pub(crate) fn begin_draining(&self) {
        self.cancel_updater.set_draining(self.in_flight.len());
    }

    /// Update draining count.
    pub(crate) fn update_draining(&self) {
        self.cancel_updater.update_draining(self.in_flight.len());
    }
}

/// Internal notification channels for transfer state updates.
#[allow(dead_code)]
pub(crate) struct TransferNotifiers {
    pub(crate) status_tx: watch::Sender<TransferStatus>,
    pub(crate) passed_tx: watch::Sender<Vec<BlockId>>,
    pub(crate) completed_tx: watch::Sender<Vec<BlockId>>,
    pub(crate) remaining_tx: watch::Sender<Vec<BlockId>>,
    pub(crate) result_tx: watch::Sender<Option<TransferResult>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_id() {
        let id1 = TransferId::new();
        let id2 = TransferId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_transfer_status() {
        assert!(!TransferStatus::Evaluating.is_terminal());
        assert!(!TransferStatus::Queued.is_terminal());
        assert!(!TransferStatus::Transferring.is_terminal());
        assert!(TransferStatus::Complete.is_terminal());
        assert!(TransferStatus::Cancelled.is_terminal());
        assert!(TransferStatus::Failed.is_terminal());
    }

    #[test]
    fn test_transfer_state_creation() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3];
        let (state, handle) = TransferState::new(id, blocks.clone());

        assert_eq!(state.id, id);
        assert_eq!(state.status, TransferStatus::Evaluating);
        assert_eq!(state.input_blocks, blocks);
        assert!(state.passed_blocks.is_empty());
        assert!(state.completed.is_empty());

        assert_eq!(handle.id(), id);
        assert_eq!(handle.status(), TransferStatus::Evaluating);
        assert_eq!(handle.remaining_blocks(), blocks);
    }

    #[test]
    fn test_transfer_state_progress() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3, 4, 5];
        let (mut state, handle) = TransferState::new(id, blocks);

        // Some blocks pass filters
        state.add_passed(vec![1, 2, 3]);
        state.add_filtered(vec![4, 5]);
        assert_eq!(handle.passed_blocks(), vec![1, 2, 3]);

        // Start transferring
        state.set_status(TransferStatus::Transferring);
        state.mark_in_flight(vec![1, 2]);
        assert_eq!(handle.status(), TransferStatus::Transferring);

        // Complete some
        state.mark_completed(vec![1]);
        assert_eq!(handle.completed_blocks(), vec![1]);
        assert_eq!(state.in_flight_count(), 1);

        // Complete rest
        state.mark_completed(vec![2, 3]);
        state.set_complete();

        assert_eq!(handle.status(), TransferStatus::Complete);
        assert_eq!(handle.completed_blocks(), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_transfer_handle_wait() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3];
        let (mut state, mut handle) = TransferState::new(id, blocks);

        // Spawn task to complete the transfer
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            state.add_passed(vec![1, 2, 3]);
            state.mark_completed(vec![1, 2, 3]);
            state.set_complete();
        });

        // Wait for completion
        let result = tokio::time::timeout(tokio::time::Duration::from_millis(100), handle.wait())
            .await
            .expect("Should complete within timeout")
            .expect("Should succeed");

        assert_eq!(result.status, TransferStatus::Complete);
        assert_eq!(result.completed_blocks, vec![1, 2, 3]);
    }
}
