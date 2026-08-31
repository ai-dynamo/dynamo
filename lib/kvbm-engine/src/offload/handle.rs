// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer handle and status tracking for offload operations.
//!
//! The `TransferHandle` is the user-facing interface for tracking and controlling
//! an offload transfer. It provides:
//! - Status tracking (Evaluating, Queued, Transferring, Complete, Cancelled)
//! - Block visibility (passed, completed, remaining)
//! - Cancellation with confirmation

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use tokio::sync::watch;
use uuid::Uuid;

use crate::BlockId;

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
    /// Blocks that failed transfer
    pub failed_blocks: Vec<BlockId>,
    /// Blocks that were filtered out
    pub filtered_blocks: Vec<BlockId>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Monotonic block counts for a transfer.
///
/// Counts are published through a watch channel, while the corresponding block
/// IDs remain in shared progress storage. This keeps progress notifications
/// constant-size regardless of the number of blocks in the transfer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferProgressCounts {
    /// Blocks accepted by policy evaluation.
    pub passed: usize,
    /// Blocks transferred successfully.
    pub completed: usize,
    /// Blocks whose transfer failed.
    pub failed: usize,
}

impl TransferProgressCounts {
    /// Number of blocks that have reached either success or failure.
    pub fn settled(self) -> usize {
        self.completed.saturating_add(self.failed)
    }
}

/// Per-consumer cursor for incrementally reading transfer progress.
///
/// A cursor belongs to one transfer. Create it with
/// [`TransferHandle::new_progress_cursor`] and retain it at the consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferProgressCursor {
    transfer_id: TransferId,
    passed: usize,
    completed: usize,
    failed: usize,
}

/// Blocks appended since a [`TransferProgressCursor`] was last consumed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransferProgressDelta {
    /// Blocks newly accepted by policy evaluation.
    pub passed_blocks: Vec<BlockId>,
    /// Blocks newly transferred successfully.
    pub completed_blocks: Vec<BlockId>,
    /// Blocks whose transfer newly failed.
    pub failed_blocks: Vec<BlockId>,
}

impl TransferProgressDelta {
    /// Whether the cursor observed no new progress.
    pub fn is_empty(&self) -> bool {
        self.passed_blocks.is_empty()
            && self.completed_blocks.is_empty()
            && self.failed_blocks.is_empty()
    }
}

#[derive(Debug)]
struct TransferProgress {
    input_blocks: Vec<BlockId>,
    policy_evaluated: bool,
    passed_blocks: Vec<BlockId>,
    completed_blocks: Vec<BlockId>,
    failed_blocks: Vec<BlockId>,
}

impl TransferProgress {
    fn new(input_blocks: Vec<BlockId>) -> Self {
        Self {
            input_blocks,
            policy_evaluated: false,
            passed_blocks: Vec::new(),
            completed_blocks: Vec::new(),
            failed_blocks: Vec::new(),
        }
    }

    fn counts(&self) -> TransferProgressCounts {
        TransferProgressCounts {
            passed: self.passed_blocks.len(),
            completed: self.completed_blocks.len(),
            failed: self.failed_blocks.len(),
        }
    }
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
    progress: Arc<RwLock<TransferProgress>>,
    progress_rx: watch::Receiver<TransferProgressCounts>,
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
        self.progress
            .read()
            .expect("transfer progress lock poisoned")
            .passed_blocks
            .clone()
    }

    /// Get blocks that have been successfully transferred.
    pub fn completed_blocks(&self) -> Vec<BlockId> {
        self.progress
            .read()
            .expect("transfer progress lock poisoned")
            .completed_blocks
            .clone()
    }

    /// Get blocks that failed transfer.
    pub fn failed_blocks(&self) -> Vec<BlockId> {
        self.progress
            .read()
            .expect("transfer progress lock poisoned")
            .failed_blocks
            .clone()
    }

    /// Get blocks remaining to be transferred.
    pub fn remaining_blocks(&self) -> Vec<BlockId> {
        let progress = self
            .progress
            .read()
            .expect("transfer progress lock poisoned");
        if !progress.policy_evaluated {
            return progress.input_blocks.clone();
        }

        let settled: HashSet<_> = progress
            .completed_blocks
            .iter()
            .chain(&progress.failed_blocks)
            .copied()
            .collect();
        progress
            .passed_blocks
            .iter()
            .filter(|id| !settled.contains(id))
            .copied()
            .collect()
    }

    /// Return constant-size progress counts without cloning block vectors.
    pub fn progress_counts(&self) -> TransferProgressCounts {
        *self.progress_rx.borrow()
    }

    /// Create a cursor that consumes this transfer's progress from the start.
    pub fn new_progress_cursor(&self) -> TransferProgressCursor {
        TransferProgressCursor {
            transfer_id: self.id,
            passed: 0,
            completed: 0,
            failed: 0,
        }
    }

    /// Read each newly passed, completed, or failed block exactly once for this
    /// cursor, then advance the cursor to the current progress boundary.
    pub fn consume_progress(&self, cursor: &mut TransferProgressCursor) -> TransferProgressDelta {
        assert_eq!(
            cursor.transfer_id, self.id,
            "transfer progress cursor used with a different handle"
        );
        let progress = self
            .progress
            .read()
            .expect("transfer progress lock poisoned");
        assert!(cursor.passed <= progress.passed_blocks.len());
        assert!(cursor.completed <= progress.completed_blocks.len());
        assert!(cursor.failed <= progress.failed_blocks.len());

        let delta = TransferProgressDelta {
            passed_blocks: progress.passed_blocks[cursor.passed..].to_vec(),
            completed_blocks: progress.completed_blocks[cursor.completed..].to_vec(),
            failed_blocks: progress.failed_blocks[cursor.failed..].to_vec(),
        };
        cursor.passed = progress.passed_blocks.len();
        cursor.completed = progress.completed_blocks.len();
        cursor.failed = progress.failed_blocks.len();
        delta
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
    /// handle.cancel().wait().await;
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

    /// Subscribe to constant-size block progress counts.
    pub fn subscribe_progress(&self) -> watch::Receiver<TransferProgressCounts> {
        self.progress_rx.clone()
    }
}

impl std::fmt::Debug for TransferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let progress = self.progress_counts();
        f.debug_struct("TransferHandle")
            .field("id", &self.id)
            .field("status", &self.status())
            .field("passed_count", &progress.passed)
            .field("completed_count", &progress.completed)
            .field("failed_count", &progress.failed)
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
    /// Shared cumulative block progress. Consumers use monotonic cursors so
    /// scheduler ticks copy only newly settled IDs.
    progress: Arc<RwLock<TransferProgress>>,
    /// Blocks currently in-flight (being transferred)
    pub(crate) in_flight: HashSet<BlockId>,
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
    pub(crate) precondition: Option<velo::EventHandle>,
}

#[allow(dead_code)]
impl TransferState {
    /// Create transfer state and associated handle.
    pub(crate) fn new(id: TransferId, input_blocks: Vec<BlockId>) -> (Self, TransferHandle) {
        let (status_tx, status_rx) = watch::channel(TransferStatus::Evaluating);
        let progress = Arc::new(RwLock::new(TransferProgress::new(input_blocks)));
        let (progress_tx, progress_rx) = watch::channel(TransferProgressCounts::default());
        let (result_tx, result_rx) = watch::channel(None);
        let (cancel_token, cancel_updater) = CancellationToken::new();

        let notifiers = TransferNotifiers {
            status_tx,
            progress_tx,
            result_tx,
        };

        let state = TransferState {
            id,
            status: TransferStatus::Evaluating,
            progress: progress.clone(),
            in_flight: HashSet::new(),
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
            progress,
            progress_rx,
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
        let counts = {
            let mut progress = self
                .progress
                .write()
                .expect("transfer progress lock poisoned");
            progress.policy_evaluated = true;
            progress.passed_blocks.extend(block_ids);
            progress.counts()
        };
        let _ = self.notifiers.progress_tx.send(counts);
    }

    /// Add blocks that were filtered out.
    pub(crate) fn add_filtered(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.filtered_out.extend(block_ids);
    }

    /// Mark blocks as in-flight (being transferred).
    pub(crate) fn mark_in_flight(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.in_flight.extend(block_ids);
    }

    /// Mark blocks as completed (transferred successfully).
    pub(crate) fn mark_completed(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        let completed: Vec<_> = block_ids.into_iter().collect();
        for id in &completed {
            self.in_flight.remove(id);
        }
        let counts = {
            let mut progress = self
                .progress
                .write()
                .expect("transfer progress lock poisoned");
            progress.completed_blocks.extend(completed);
            progress.counts()
        };
        let _ = self.notifiers.progress_tx.send(counts);
    }

    /// Mark blocks as failed (transfer unsuccessful).
    pub(crate) fn mark_failed(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        let failed: Vec<_> = block_ids.into_iter().collect();
        for id in &failed {
            self.in_flight.remove(id);
        }
        let counts = {
            let mut progress = self
                .progress
                .write()
                .expect("transfer progress lock poisoned");
            progress.failed_blocks.extend(failed);
            progress.counts()
        };
        let _ = self.notifiers.progress_tx.send(counts);
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
        let progress = self
            .progress
            .read()
            .expect("transfer progress lock poisoned");
        let result = TransferResult {
            id: self.id,
            status: self.status,
            passed_blocks: progress.passed_blocks.clone(),
            completed_blocks: progress.completed_blocks.clone(),
            failed_blocks: progress.failed_blocks.clone(),
            filtered_blocks: self.filtered_out.clone(),
            error: self.error.clone(),
        };
        let _ = self.notifiers.result_tx.send(Some(result));
    }

    pub(crate) fn progress_counts(&self) -> TransferProgressCounts {
        self.progress
            .read()
            .expect("transfer progress lock poisoned")
            .counts()
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
    pub(crate) progress_tx: watch::Sender<TransferProgressCounts>,
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
        assert_eq!(state.progress_counts(), TransferProgressCounts::default());

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

    #[test]
    fn test_mark_failed_removes_from_in_flight() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3];
        let (mut state, handle) = TransferState::new(id, blocks);

        state.add_passed(vec![1, 2, 3]);
        state.mark_in_flight(vec![1, 2, 3]);
        assert_eq!(state.in_flight_count(), 3);

        state.mark_failed(vec![2]);
        assert_eq!(state.in_flight_count(), 2);
        assert_eq!(handle.failed_blocks(), vec![2]);
        assert!(handle.completed_blocks().is_empty());
    }

    #[test]
    fn test_mark_failed_updates_remaining() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3];
        let (mut state, handle) = TransferState::new(id, blocks);

        state.add_passed(vec![1, 2, 3]);
        state.mark_in_flight(vec![1, 2, 3]);

        // Fail block 2 — remaining should exclude it
        state.mark_failed(vec![2]);
        let remaining = handle.remaining_blocks();
        assert!(remaining.contains(&1));
        assert!(!remaining.contains(&2));
        assert!(remaining.contains(&3));
    }

    #[test]
    fn test_partial_failure_result() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3, 4, 5];
        let (mut state, handle) = TransferState::new(id, blocks);

        state.add_passed(vec![1, 2, 3]);
        state.add_filtered(vec![4, 5]);
        state.mark_in_flight(vec![1, 2, 3]);

        // Block 1 succeeds, block 2 fails, block 3 succeeds
        state.mark_completed(vec![1, 3]);
        state.mark_failed(vec![2]);

        assert_eq!(handle.completed_blocks(), vec![1, 3]);
        assert_eq!(handle.failed_blocks(), vec![2]);
        assert_eq!(state.in_flight_count(), 0);

        // Simulate the pipeline's terminal state logic
        let progress = state.progress_counts();
        let total = progress.passed + state.filtered_out.len();
        let done = progress.settled() + state.filtered_out.len();
        assert_eq!(done, total);

        // With failures, should set_error not set_complete
        let failed_count = progress.failed;
        assert!(failed_count > 0);
        state.set_error(format!(
            "{failed_count} blocks failed to transfer to object storage",
        ));
        assert_eq!(state.status, TransferStatus::Failed);
    }

    #[test]
    fn test_progress_cursor_consumes_each_block_once() {
        let id = TransferId::new();
        let (mut state, handle) = TransferState::new(id, vec![1, 2, 3]);
        let mut cursor = handle.new_progress_cursor();

        state.add_passed([1, 2, 3]);
        state.mark_completed([1]);
        let first = handle.consume_progress(&mut cursor);
        assert_eq!(first.passed_blocks, vec![1, 2, 3]);
        assert_eq!(first.completed_blocks, vec![1]);
        assert!(first.failed_blocks.is_empty());

        state.mark_completed([2]);
        state.mark_failed([3]);
        let second = handle.consume_progress(&mut cursor);
        assert!(second.passed_blocks.is_empty());
        assert_eq!(second.completed_blocks, vec![2]);
        assert_eq!(second.failed_blocks, vec![3]);
        assert!(handle.consume_progress(&mut cursor).is_empty());
        assert_eq!(
            handle.progress_counts(),
            TransferProgressCounts {
                passed: 3,
                completed: 2,
                failed: 1,
            }
        );
    }

    #[tokio::test]
    async fn test_partial_failure_wait_result() {
        let id = TransferId::new();
        let blocks = vec![1, 2, 3];
        let (mut state, mut handle) = TransferState::new(id, blocks);

        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            state.add_passed(vec![1, 2, 3]);
            state.mark_in_flight(vec![1, 2, 3]);
            state.mark_completed(vec![1, 3]);
            state.mark_failed(vec![2]);
            state.set_error("1 blocks failed to transfer to object storage".to_string());
        });

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(100), handle.wait())
            .await
            .expect("Should complete within timeout")
            .expect("Should succeed");

        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.completed_blocks, vec![1, 3]);
        assert_eq!(result.failed_blocks, vec![2]);
        assert!(result.error.is_some());
    }
}
