// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline coordination for offload transfers.
//!
//! A pipeline connects three stages:
//! 1. **PolicyEvaluator**: Evaluates blocks against policies, filters out non-passing blocks
//! 2. **BatchCollector**: Accumulates passing blocks into batches
//! 3. **TransferExecutor**: Executes the actual data transfer
//!
//! ```text
//! enqueue() → [PolicyEvaluator] → [BatchCollector] → [TransferExecutor] → complete
//!                 ↓ cancel           ↓ cancel             ↓ cancel
//! ```

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::logical::blocks::{BlockMetadata, BlockRegistry, ImmutableBlock};
use crate::v2::logical::manager::BlockManager;
use crate::v2::physical::transfer::TransferOptions;
use crate::v2::{BlockId, SequenceHash};

use super::batch::{
    BatchCollector, BatchConfig, BatchInput, BatchOutputRx, EvalResult, QueuedBlock, TransferBatch,
};
use super::handle::{TransferId, TransferState, TransferStatus};
use super::policy::{EvalContext, OffloadPolicy};

/// Configuration for a pipeline.
#[derive(Clone)]
pub struct PipelineConfig<Src: BlockMetadata, Dst: BlockMetadata> {
    /// Policies to evaluate (all must pass)
    pub policies: Vec<Arc<dyn OffloadPolicy<Src>>>,
    /// Batch configuration
    pub batch_config: BatchConfig,
    /// Timeout for policy evaluation (fail-fast)
    pub policy_timeout: Duration,
    /// Whether arrivals from this pipeline auto-feed downstream
    pub auto_chain: bool,
    /// Channel capacity for evaluation input
    pub eval_input_capacity: usize,
    /// Channel capacity for batch input
    pub batch_input_capacity: usize,
    /// Channel capacity for transfer input
    pub transfer_input_capacity: usize,
    /// Marker
    _marker: PhantomData<(Src, Dst)>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> Default for PipelineConfig<Src, Dst> {
    fn default() -> Self {
        Self {
            policies: Vec::new(),
            batch_config: BatchConfig::default(),
            policy_timeout: Duration::from_millis(100),
            auto_chain: false,
            eval_input_capacity: 128,
            batch_input_capacity: 256,
            transfer_input_capacity: 8,
            _marker: PhantomData,
        }
    }
}

/// Builder for pipeline configuration.
pub struct PipelineBuilder<Src: BlockMetadata, Dst: BlockMetadata> {
    config: PipelineConfig<Src, Dst>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> PipelineBuilder<Src, Dst> {
    /// Create a new pipeline builder with defaults.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Add a policy to the pipeline.
    pub fn policy(mut self, policy: Arc<dyn OffloadPolicy<Src>>) -> Self {
        self.config.policies.push(policy);
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_config.max_batch_size = size;
        self
    }

    /// Set minimum batch size for flush.
    pub fn min_batch_size(mut self, size: usize) -> Self {
        self.config.batch_config.min_batch_size = size;
        self
    }

    /// Set batch flush interval.
    pub fn flush_interval(mut self, interval: Duration) -> Self {
        self.config.batch_config.flush_interval = interval;
        self
    }

    /// Set policy timeout.
    pub fn policy_timeout(mut self, timeout: Duration) -> Self {
        self.config.policy_timeout = timeout;
        self
    }

    /// Enable auto-chaining to downstream pipelines.
    pub fn auto_chain(mut self, enabled: bool) -> Self {
        self.config.auto_chain = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> PipelineConfig<Src, Dst> {
        self.config
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> Default for PipelineBuilder<Src, Dst> {
    fn default() -> Self {
        Self::new()
    }
}

/// Input to the pipeline (from enqueue).
pub(crate) struct PipelineInput<T: BlockMetadata> {
    pub(crate) transfer_id: TransferId,
    pub(crate) blocks: Vec<ImmutableBlock<T>>,
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
}

/// Output from the pipeline (completed transfer).
pub struct PipelineOutput {
    pub transfer_id: TransferId,
    pub completed_hashes: Vec<SequenceHash>,
}

/// A running pipeline instance.
pub struct Pipeline<Src: BlockMetadata, Dst: BlockMetadata> {
    config: PipelineConfig<Src, Dst>,
    /// Input channel for new blocks
    pub(crate) input_tx: mpsc::Sender<PipelineInput<Src>>,
    /// Output channel for completed blocks (may feed downstream)
    output_tx: Option<mpsc::Sender<PipelineOutput>>,
    /// Task handles for pipeline stages
    _task_handles: Vec<JoinHandle<()>>,
    /// Marker
    _marker: PhantomData<Dst>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> Pipeline<Src, Dst> {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Pipeline configuration
    /// * `registry` - Block registry for policy evaluation
    /// * `src_manager` - Source tier block manager
    /// * `dst_manager` - Destination tier block manager
    /// * `leader` - Instance leader for transfer execution
    /// * `src_layout` - Source logical layout handle
    /// * `dst_layout` - Destination logical layout handle
    pub fn new(
        config: PipelineConfig<Src, Dst>,
        registry: Arc<BlockRegistry>,
        _src_manager: Arc<BlockManager<Src>>,
        dst_manager: Arc<BlockManager<Dst>>,
        leader: Arc<InstanceLeader>,
        src_layout: LogicalLayoutHandle,
        dst_layout: LogicalLayoutHandle,
    ) -> Self {
        // Create channels
        let (input_tx, input_rx) = mpsc::channel(config.eval_input_capacity);
        let (eval_tx, eval_rx) = mpsc::channel(config.batch_input_capacity);
        let (batch_tx, batch_rx) = mpsc::channel(config.transfer_input_capacity);
        let (output_tx, _output_rx) = mpsc::channel(64);

        // Spawn policy evaluator
        let evaluator = PolicyEvaluator {
            policies: config.policies.clone(),
            timeout: config.policy_timeout,
            input_rx,
            output_tx: eval_tx,
        };
        let eval_handle = tokio::spawn(async move {
            evaluator.run().await;
        });

        // Spawn batch collector
        let collector = BatchCollector::new(config.batch_config.clone(), eval_rx, batch_tx);
        let batch_handle = tokio::spawn(async move {
            collector.run().await;
        });

        // Spawn transfer executor
        let executor = TransferExecutor {
            input_rx: batch_rx,
            leader,
            dst_manager,
            src_layout,
            dst_layout,
            registry,
            _src_marker: PhantomData::<Src>,
        };
        let transfer_handle = tokio::spawn(async move {
            executor.run().await;
        });

        Self {
            config,
            input_tx,
            output_tx: Some(output_tx),
            _task_handles: vec![eval_handle, batch_handle, transfer_handle],
            _marker: PhantomData,
        }
    }

    /// Enqueue blocks for offloading through this pipeline.
    pub(crate) async fn enqueue(
        &self,
        transfer_id: TransferId,
        blocks: Vec<ImmutableBlock<Src>>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> Result<(), mpsc::error::SendError<PipelineInput<Src>>> {
        let input = PipelineInput {
            transfer_id,
            blocks,
            state,
        };
        self.input_tx.send(input).await
    }

    /// Check if this pipeline auto-chains to downstream.
    pub fn auto_chain(&self) -> bool {
        self.config.auto_chain
    }

    /// Get a clone of the output channel sender.
    pub fn output_tx(&self) -> Option<mpsc::Sender<PipelineOutput>> {
        self.output_tx.clone()
    }
}

/// Policy evaluator stage.
struct PolicyEvaluator<T: BlockMetadata> {
    policies: Vec<Arc<dyn OffloadPolicy<T>>>,
    timeout: Duration,
    input_rx: mpsc::Receiver<PipelineInput<T>>,
    output_tx: BatchInput<T>,
}

impl<T: BlockMetadata> PolicyEvaluator<T> {
    async fn run(mut self) {
        while let Some(input) = self.input_rx.recv().await {
            self.evaluate(input).await;
        }
    }

    async fn evaluate(&self, input: PipelineInput<T>) {
        let transfer_id = input.transfer_id;
        let mut passed = Vec::new();
        let mut filtered = Vec::new();

        for block in input.blocks {
            let ctx = EvalContext::new(block);

            // Evaluate all policies
            let pass = self.evaluate_policies(&ctx).await;

            if pass {
                passed.push(QueuedBlock {
                    transfer_id,
                    block_id: ctx.block_id,
                    sequence_hash: ctx.sequence_hash,
                    guard: ctx.block,
                });
            } else {
                filtered.push(ctx.block_id);
            }
        }

        // Update state with evaluation results
        {
            let mut state = input.state.lock().unwrap();
            state.add_passed(passed.iter().map(|b| b.block_id));
            state.add_filtered(filtered.iter().copied());
            state.set_status(TransferStatus::Queued);
        }

        // Send to batch collector
        let result = EvalResult {
            transfer_id,
            passed_blocks: passed,
            filtered_ids: filtered,
        };

        if self.output_tx.send(result).await.is_err() {
            tracing::warn!("Batch collector channel closed");
        }
    }

    async fn evaluate_policies(&self, ctx: &EvalContext<T>) -> bool {
        for policy in &self.policies {
            match tokio::time::timeout(self.timeout, policy.evaluate(ctx)).await {
                Ok(Ok(true)) => continue,
                Ok(Ok(false)) => return false,
                Ok(Err(e)) => {
                    tracing::warn!("Policy {} error: {}", policy.name(), e);
                    return false;
                }
                Err(_) => {
                    tracing::warn!("Policy {} timed out", policy.name());
                    return false;
                }
            }
        }
        true
    }
}

/// Transfer executor stage.
struct TransferExecutor<Src: BlockMetadata, Dst: BlockMetadata> {
    input_rx: BatchOutputRx<Src>,
    leader: Arc<InstanceLeader>,
    dst_manager: Arc<BlockManager<Dst>>,
    src_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
    registry: Arc<BlockRegistry>,
    _src_marker: PhantomData<Src>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> TransferExecutor<Src, Dst> {
    async fn run(mut self) {
        while let Some(batch) = self.input_rx.recv().await {
            if let Err(e) = self.execute_batch(batch).await {
                tracing::error!("Transfer batch failed: {}", e);
            }
        }
    }

    async fn execute_batch(&self, batch: TransferBatch<Src>) -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let src_block_ids = batch.block_ids();
        let sequence_hashes = batch.sequence_hashes();

        // Allocate destination blocks
        let dst_blocks = self
            .dst_manager
            .allocate_blocks(batch.len())
            .ok_or_else(|| {
                anyhow::anyhow!("Failed to allocate {} destination blocks", batch.len())
            })?;

        let dst_block_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

        // Execute transfer via leader
        let notification = self.leader.execute_local_transfer(
            self.src_layout,
            self.dst_layout,
            src_block_ids.clone(),
            dst_block_ids.clone(),
            TransferOptions::default(),
        )?;

        // Wait for transfer completion
        notification.await?;

        // Register blocks in destination tier using transfer_registration
        // This avoids bumping LFU counts for offloaded blocks
        for seq_hash in sequence_hashes.iter() {
            // Get or create registration handle without touching frequency tracker
            let _handle = self.registry.transfer_registration(*seq_hash);
            // TODO: The actual block registration in the destination tier requires
            // integration with BlockManager's registration API. For offloading,
            // we need a method like `register_transferred_block` that takes a
            // MutableBlock and marks it as containing transferred data.
        }

        // Drop dst_blocks - they will be returned to the pool
        // In a full implementation, we would register them first
        drop(dst_blocks);

        tracing::debug!(
            "Transferred {} blocks from {:?} to {:?}",
            src_block_ids.len(),
            self.src_layout,
            self.dst_layout
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let config = PipelineBuilder::<(), ()>::new()
            .batch_size(32)
            .min_batch_size(8)
            .policy_timeout(Duration::from_millis(50))
            .auto_chain(true)
            .build();

        assert_eq!(config.batch_config.max_batch_size, 32);
        assert_eq!(config.batch_config.min_batch_size, 8);
        assert_eq!(config.policy_timeout, Duration::from_millis(50));
        assert!(config.auto_chain);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::<(), ()>::default();
        assert!(config.policies.is_empty());
        assert!(!config.auto_chain);
    }
}
