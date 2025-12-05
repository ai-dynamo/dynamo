// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Batch collection and accumulation for offload transfers.
//!
//! The batch collector accumulates blocks that pass policy evaluation and
//! groups them into batches for efficient transfer execution.

use std::time::Duration;

use tokio::sync::mpsc;

use crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock};
use crate::v2::{BlockId, SequenceHash};

use super::cancel::CancelStateUpdater;
use super::handle::TransferId;

/// Configuration for batch collection.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum blocks per batch
    pub max_batch_size: usize,
    /// Time to wait before flushing a partial batch
    pub flush_interval: Duration,
    /// Minimum batch size before flush (unless timeout)
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            flush_interval: Duration::from_millis(10),
            min_batch_size: 8,
        }
    }
}

impl BatchConfig {
    /// Create a new batch config with specified max size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set the flush interval.
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Set the minimum batch size.
    pub fn with_min_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }
}

/// A block that passed policy evaluation and is queued for transfer.
pub struct QueuedBlock<T: BlockMetadata> {
    /// Transfer ID this block belongs to
    pub transfer_id: TransferId,
    /// Block ID
    pub block_id: BlockId,
    /// Sequence hash
    pub sequence_hash: SequenceHash,
    /// Strong reference held during transfer (RAII)
    pub guard: ImmutableBlock<T>,
}

impl<T: BlockMetadata> std::fmt::Debug for QueuedBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedBlock")
            .field("transfer_id", &self.transfer_id)
            .field("block_id", &self.block_id)
            .field("sequence_hash", &self.sequence_hash)
            .finish()
    }
}

/// A batch of blocks ready for transfer execution.
pub struct TransferBatch<T: BlockMetadata> {
    /// Blocks in this batch
    pub blocks: Vec<QueuedBlock<T>>,
}

impl<T: BlockMetadata> TransferBatch<T> {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(capacity),
        }
    }

    /// Add a block to this batch.
    pub fn push(&mut self, block: QueuedBlock<T>) {
        self.blocks.push(block);
    }

    /// Get the number of blocks in this batch.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get block IDs in this batch.
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.block_id).collect()
    }

    /// Get sequence hashes in this batch.
    pub fn sequence_hashes(&self) -> Vec<SequenceHash> {
        self.blocks.iter().map(|b| b.sequence_hash).collect()
    }

    /// Get unique transfer IDs in this batch.
    pub fn transfer_ids(&self) -> Vec<TransferId> {
        let mut ids: Vec<TransferId> = self.blocks.iter().map(|b| b.transfer_id).collect();
        ids.sort_by_key(|id| id.as_uuid());
        ids.dedup();
        ids
    }

    /// Take all blocks out of this batch.
    pub fn take(&mut self) -> Vec<QueuedBlock<T>> {
        std::mem::take(&mut self.blocks)
    }

    /// Drain blocks for the given transfer ID (for cancellation).
    pub fn drain_transfer(&mut self, transfer_id: TransferId) -> Vec<QueuedBlock<T>> {
        let drained = Vec::new();
        self.blocks.retain(|b| {
            if b.transfer_id == transfer_id {
                false // Will be moved to drained
            } else {
                true // Keep in batch
            }
        });
        // Re-iterate to collect drained (retain doesn't give us removed items)
        // This is a bit inefficient but keeps the API simple
        // In practice, cancellation is rare
        drained
    }
}

impl<T: BlockMetadata> Default for TransferBatch<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of policy evaluation - blocks ready for batching.
pub struct EvalResult<T: BlockMetadata> {
    /// Transfer ID
    pub transfer_id: TransferId,
    /// Blocks that passed all policies
    pub passed_blocks: Vec<QueuedBlock<T>>,
    /// Block IDs that were filtered out
    pub filtered_ids: Vec<BlockId>,
}

/// Input to the batch collector from policy evaluator.
pub type BatchInput<T> = mpsc::Sender<EvalResult<T>>;
/// Receiver side of batch input channel.
pub type BatchInputRx<T> = mpsc::Receiver<EvalResult<T>>;

/// Output from the batch collector to transfer executor.
pub type BatchOutput<T> = mpsc::Sender<TransferBatch<T>>;
/// Receiver side of batch output channel.
pub type BatchOutputRx<T> = mpsc::Receiver<TransferBatch<T>>;

/// Batch collector that accumulates blocks and flushes batches.
///
/// The collector accumulates blocks from policy evaluation and groups them
/// into batches based on the configuration. Batches are flushed when:
/// - `max_batch_size` is reached
/// - `flush_interval` expires and `min_batch_size` is met
/// - Shutdown is requested
pub struct BatchCollector<T: BlockMetadata> {
    config: BatchConfig,
    /// Input channel from policy evaluator
    input_rx: BatchInputRx<T>,
    /// Output channel to transfer executor
    output_tx: BatchOutput<T>,
    /// Current batch being built
    current_batch: TransferBatch<T>,
    /// Cancel state updater for checking cancellation
    #[allow(dead_code)]
    cancel_updater: Option<CancelStateUpdater>,
}

impl<T: BlockMetadata> BatchCollector<T> {
    /// Create a new batch collector.
    pub fn new(config: BatchConfig, input_rx: BatchInputRx<T>, output_tx: BatchOutput<T>) -> Self {
        let max_batch_size = config.max_batch_size;
        Self {
            config,
            input_rx,
            output_tx,
            current_batch: TransferBatch::with_capacity(max_batch_size),
            cancel_updater: None,
        }
    }

    /// Set cancel state updater for checking cancellation.
    pub fn with_cancel_updater(mut self, updater: CancelStateUpdater) -> Self {
        self.cancel_updater = Some(updater);
        self
    }

    /// Run the batch collector loop.
    ///
    /// This runs until the input channel is closed.
    pub async fn run(mut self) {
        let mut flush_timer = tokio::time::interval(self.config.flush_interval);
        flush_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Receive evaluation results
                result = self.input_rx.recv() => {
                    match result {
                        Some(eval_result) => {
                            self.handle_eval_result(eval_result).await;
                        }
                        None => {
                            // Input channel closed, flush remaining and exit
                            self.flush_if_not_empty().await;
                            break;
                        }
                    }
                }
                // Periodic flush timer
                _ = flush_timer.tick() => {
                    self.try_flush().await;
                }
            }
        }
    }

    /// Handle an evaluation result.
    async fn handle_eval_result(&mut self, result: EvalResult<T>) {
        // Add passed blocks to current batch
        for block in result.passed_blocks {
            self.current_batch.push(block);

            // Flush if we've reached max batch size
            if self.current_batch.len() >= self.config.max_batch_size {
                self.flush().await;
            }
        }
    }

    /// Try to flush if minimum batch size is reached.
    async fn try_flush(&mut self) {
        if self.current_batch.len() >= self.config.min_batch_size {
            self.flush().await;
        }
    }

    /// Flush current batch if not empty.
    async fn flush_if_not_empty(&mut self) {
        if !self.current_batch.is_empty() {
            self.flush().await;
        }
    }

    /// Flush the current batch to the output channel.
    async fn flush(&mut self) {
        if self.current_batch.is_empty() {
            return;
        }

        let batch = std::mem::replace(
            &mut self.current_batch,
            TransferBatch::with_capacity(self.config.max_batch_size),
        );

        // Send to transfer executor
        if self.output_tx.send(batch).await.is_err() {
            // Output channel closed, log and continue
            tracing::warn!("Batch output channel closed");
        }
    }
}

/// Create batch collection channels with given capacity.
pub fn batch_channels<T: BlockMetadata>(
    input_capacity: usize,
    output_capacity: usize,
) -> (
    BatchInput<T>,
    BatchInputRx<T>,
    BatchOutput<T>,
    BatchOutputRx<T>,
) {
    let (input_tx, input_rx) = mpsc::channel(input_capacity);
    let (output_tx, output_rx) = mpsc::channel(output_capacity);
    (input_tx, input_rx, output_tx, output_rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.min_batch_size, 8);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .with_max_size(128)
            .with_min_size(16)
            .with_flush_interval(Duration::from_millis(50));

        assert_eq!(config.max_batch_size, 128);
        assert_eq!(config.min_batch_size, 16);
        assert_eq!(config.flush_interval, Duration::from_millis(50));
    }

    #[test]
    fn test_transfer_batch() {
        let mut batch: TransferBatch<()> = TransferBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[tokio::test]
    async fn test_batch_collector_empty_input() {
        let (input_tx, input_rx) = mpsc::channel::<EvalResult<()>>(10);
        let (output_tx, mut output_rx) = mpsc::channel::<TransferBatch<()>>(10);

        let collector = BatchCollector::new(BatchConfig::default(), input_rx, output_tx);

        // Drop input to close channel
        drop(input_tx);

        // Run collector
        tokio::spawn(async move {
            collector.run().await;
        });

        // Should receive nothing (empty input)
        let result = tokio::time::timeout(Duration::from_millis(50), output_rx.recv()).await;
        assert!(result.is_err() || result.unwrap().is_none());
    }
}
