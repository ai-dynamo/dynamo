// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Main offload engine coordinating pipelines.
//!
//! The `OffloadEngine` is a standalone component that manages block offloading
//! between storage tiers (G1→G2, G2→G3, G2→G4).
//!
//! # Example
//! ```ignore
//! let engine = OffloadEngine::builder(leader.clone())
//!     .with_registry(registry.clone())
//!     .with_g1_to_g2_pipeline(
//!         PipelineBuilder::<G1, G2>::new()
//!             .policy(Arc::new(PresenceFilter::new(registry.clone())))
//!             .batch_size(32)
//!             .auto_chain(true)
//!             .build()
//!     )
//!     .with_g2_to_g3_pipeline(
//!         PipelineBuilder::<G2, G3>::new()
//!             .policy(Arc::new(PresenceAndLFUFilter::with_default_threshold(registry.clone())))
//!             .batch_size(64)
//!             .build()
//!     )
//!     .build()?;
//!
//! let handle = engine.enqueue_g2_to_g3(blocks);
//! handle.wait().await?;
//! ```

use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use tokio::task::JoinHandle;

use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::logical::blocks::{BlockMetadata, BlockRegistry, WeakBlock};
use crate::v2::logical::manager::BlockManager;
use crate::v2::{BlockId, G1, G2, G3, G4};

use super::handle::{TransferHandle, TransferId, TransferState};
use super::pipeline::{ChainOutput, ChainOutputRx, Pipeline, PipelineConfig, PipelineInput};
use super::queue::CancellableQueue;
use super::source::SourceBlocks;

/// Central coordinator for offload pipelines.
///
/// The engine manages multiple pipelines (G1→G2, G2→G3, G2→G4) and provides
/// a unified interface for enqueueing blocks for offload.
#[allow(dead_code)]
pub struct OffloadEngine {
    /// Reference to the instance leader for transfers
    leader: Arc<InstanceLeader>,
    /// Block registry for policy evaluation
    registry: Arc<BlockRegistry>,
    /// G1→G2 pipeline
    g1_to_g2: Option<Pipeline<G1, G2>>,
    /// G2→G3 pipeline
    g2_to_g3: Option<Pipeline<G2, G3>>,
    /// G2→G4 pipeline
    g2_to_g4: Option<Pipeline<G2, G4>>,
    /// Active transfer tracking
    transfers: Arc<DashMap<TransferId, Arc<std::sync::Mutex<TransferState>>>>,
    /// Chain router task handle (routes G1→G2 output to downstream pipelines)
    _chain_router_handle: Option<JoinHandle<()>>,
}

impl OffloadEngine {
    /// Create a new builder for the offload engine.
    pub fn builder(leader: Arc<InstanceLeader>) -> OffloadEngineBuilder {
        OffloadEngineBuilder::new(leader)
    }

    /// Enqueue blocks for G1→G2 offload.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g1_to_g2(&self, blocks: impl Into<SourceBlocks<G1>>) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G2 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Enqueue blocks for G1→G2 offload with a precondition event.
    ///
    /// The precondition event must be satisfied before the batch is processed
    /// by the transfer executor. This enables coordination with worker forward passes.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g1_to_g2_with_precondition(
        &self,
        blocks: impl Into<SourceBlocks<G1>>,
        precondition: Option<dynamo_nova::events::EventHandle>,
    ) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G2 pipeline not configured"))?;

        self.enqueue_to_pipeline_with_precondition(pipeline, blocks.into(), precondition)
    }

    /// Enqueue blocks for G2→G3 offload.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g2_to_g3(&self, blocks: impl Into<SourceBlocks<G2>>) -> Result<TransferHandle> {
        let pipeline = self
            .g2_to_g3
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G2→G3 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Enqueue blocks for G2→G4 offload.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g2_to_g4(&self, blocks: impl Into<SourceBlocks<G2>>) -> Result<TransferHandle> {
        let pipeline = self
            .g2_to_g4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G2→G4 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Internal: enqueue to a specific pipeline.
    fn enqueue_to_pipeline<Src: BlockMetadata, Dst: BlockMetadata>(
        &self,
        pipeline: &Pipeline<Src, Dst>,
        source: SourceBlocks<Src>,
    ) -> Result<TransferHandle> {
        // Extract block IDs for state tracking (External/Strong only, Weak has None)
        let input_block_ids = self.extract_block_ids(&source);

        // Create transfer state and handle
        let transfer_id = TransferId::new();
        let (state, handle) = TransferState::new(transfer_id, input_block_ids);
        let state = Arc::new(std::sync::Mutex::new(state));

        // Store transfer state
        self.transfers.insert(transfer_id, state.clone());

        // Enqueue source blocks directly to pipeline
        // PolicyEvaluator handles different source types
        // TransferExecutor does upgrade stage just before transfer
        let queued = pipeline.enqueue(transfer_id, source, state);
        if !queued {
            // Transfer was already cancelled before enqueueing
            tracing::warn!("Transfer {} was cancelled before enqueueing", transfer_id);
        }

        Ok(handle)
    }

    /// Internal: enqueue to a specific pipeline with a precondition.
    fn enqueue_to_pipeline_with_precondition<Src: BlockMetadata, Dst: BlockMetadata>(
        &self,
        pipeline: &Pipeline<Src, Dst>,
        source: SourceBlocks<Src>,
        precondition: Option<dynamo_nova::events::EventHandle>,
    ) -> Result<TransferHandle> {
        // Extract block IDs for state tracking (External/Strong only, Weak has None)
        let input_block_ids = self.extract_block_ids(&source);

        // Create transfer state and handle
        let transfer_id = TransferId::new();
        let (mut state, handle) = TransferState::new(transfer_id, input_block_ids);

        // Set precondition on the transfer state
        state.precondition = precondition;

        let state = Arc::new(std::sync::Mutex::new(state));

        // Store transfer state
        self.transfers.insert(transfer_id, state.clone());

        // Enqueue source blocks directly to pipeline
        // PolicyEvaluator handles different source types
        // BatchCollector will extract precondition and attach to batches
        // PreconditionAwaiter will await the event before processing
        let queued = pipeline.enqueue(transfer_id, source, state);
        if !queued {
            // Transfer was already cancelled before enqueueing
            tracing::warn!("Transfer {} was cancelled before enqueueing", transfer_id);
        }

        Ok(handle)
    }

    /// Extract block IDs from source blocks.
    ///
    /// For External/Strong blocks, returns the known block IDs.
    /// For Weak blocks, returns empty vec (IDs determined at upgrade time).
    fn extract_block_ids<T: BlockMetadata>(&self, source: &SourceBlocks<T>) -> Vec<BlockId> {
        match source {
            SourceBlocks::External(blocks) => blocks.iter().map(|b| b.block_id).collect(),
            SourceBlocks::Strong(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            SourceBlocks::Weak(_) => Vec::new(), // IDs not available without upgrade
        }
    }

    /// Release a completed transfer's resources.
    ///
    /// This is optional - transfers are automatically cleaned up,
    /// but call this to release resources earlier.
    pub fn release_transfer(&self, transfer_id: TransferId) {
        self.transfers.remove(&transfer_id);
    }

    /// Get the number of active transfers.
    pub fn active_transfer_count(&self) -> usize {
        self.transfers.len()
    }

    /// Check if G1→G2 pipeline is configured.
    pub fn has_g1_to_g2(&self) -> bool {
        self.g1_to_g2.is_some()
    }

    /// Check if G2→G3 pipeline is configured.
    pub fn has_g2_to_g3(&self) -> bool {
        self.g2_to_g3.is_some()
    }

    /// Check if G2→G4 pipeline is configured.
    pub fn has_g2_to_g4(&self) -> bool {
        self.g2_to_g4.is_some()
    }
}

/// Builder for OffloadEngine.
pub struct OffloadEngineBuilder {
    leader: Arc<InstanceLeader>,
    registry: Option<Arc<BlockRegistry>>,
    g1_manager: Option<Arc<BlockManager<G1>>>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    g4_manager: Option<Arc<BlockManager<G4>>>,
    g1_to_g2_config: Option<PipelineConfig<G1, G2>>,
    g2_to_g3_config: Option<PipelineConfig<G2, G3>>,
    g2_to_g4_config: Option<PipelineConfig<G2, G4>>,
    /// Optional runtime handle override (defaults to leader.runtime())
    runtime: Option<tokio::runtime::Handle>,
}

impl OffloadEngineBuilder {
    /// Create a new builder with the given instance leader.
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self {
            leader,
            registry: None,
            g1_manager: None,
            g2_manager: None,
            g3_manager: None,
            g4_manager: None,
            g1_to_g2_config: None,
            g2_to_g3_config: None,
            g2_to_g4_config: None,
            runtime: None,
        }
    }

    /// Set an explicit runtime handle for spawning pipeline tasks.
    ///
    /// If not set, defaults to `leader.runtime()`. Use this when you need
    /// pipeline tasks to run on a specific runtime (e.g., in tests).
    pub fn with_runtime(mut self, runtime: tokio::runtime::Handle) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Set the block registry.
    pub fn with_registry(mut self, registry: Arc<BlockRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set the G1 block manager.
    pub fn with_g1_manager(mut self, manager: Arc<BlockManager<G1>>) -> Self {
        self.g1_manager = Some(manager);
        self
    }

    /// Set the G2 block manager.
    pub fn with_g2_manager(mut self, manager: Arc<BlockManager<G2>>) -> Self {
        self.g2_manager = Some(manager);
        self
    }

    /// Set the G3 block manager.
    pub fn with_g3_manager(mut self, manager: Arc<BlockManager<G3>>) -> Self {
        self.g3_manager = Some(manager);
        self
    }

    /// Set the G4 block manager.
    pub fn with_g4_manager(mut self, manager: Arc<BlockManager<G4>>) -> Self {
        self.g4_manager = Some(manager);
        self
    }

    /// Configure G1→G2 pipeline.
    pub fn with_g1_to_g2_pipeline(mut self, config: PipelineConfig<G1, G2>) -> Self {
        self.g1_to_g2_config = Some(config);
        self
    }

    /// Configure G2→G3 pipeline.
    pub fn with_g2_to_g3_pipeline(mut self, config: PipelineConfig<G2, G3>) -> Self {
        self.g2_to_g3_config = Some(config);
        self
    }

    /// Configure G2→G4 pipeline.
    pub fn with_g2_to_g4_pipeline(mut self, config: PipelineConfig<G2, G4>) -> Self {
        self.g2_to_g4_config = Some(config);
        self
    }

    /// Build the offload engine.
    pub fn build(self) -> Result<OffloadEngine> {
        let registry = self
            .registry
            .ok_or_else(|| anyhow::anyhow!("Block registry required"))?;

        // Get the runtime handle for spawning background tasks
        // Use explicit override if provided, otherwise get from leader
        let runtime = self.runtime.unwrap_or_else(|| self.leader.runtime());

        // Build G1→G2 pipeline if configured
        // Note: G1 is externally owned (vLLM GPU cache), so no G1 manager needed.
        // Pipeline works with ExternalBlock<G1> which contains block_id + sequence_hash.
        let mut g1_to_g2 = if let Some(config) = self.g1_to_g2_config {
            let g2_manager = self
                .g2_manager
                .clone()
                .ok_or_else(|| anyhow::anyhow!("G2 manager required for G1→G2 pipeline"))?;

            Some(Pipeline::new(
                config,
                registry.clone(),
                g2_manager,
                self.leader.clone(),
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                runtime.clone(),
            ))
        } else {
            None
        };

        // Build G2→G3 pipeline if configured
        let g2_to_g3 = if let Some(config) = self.g2_to_g3_config {
            let g3_manager = self
                .g3_manager
                .ok_or_else(|| anyhow::anyhow!("G3 manager required for G2→G3 pipeline"))?;

            Some(Pipeline::new(
                config,
                registry.clone(),
                g3_manager,
                self.leader.clone(),
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                runtime.clone(),
            ))
        } else {
            None
        };

        // Build G2→G4 pipeline if configured
        let g2_to_g4 = if let Some(config) = self.g2_to_g4_config {
            let g4_manager = self
                .g4_manager
                .ok_or_else(|| anyhow::anyhow!("G4 manager required for G2→G4 pipeline"))?;

            Some(Pipeline::new(
                config,
                registry.clone(),
                g4_manager,
                self.leader.clone(),
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G4,
                runtime.clone(),
            ))
        } else {
            None
        };

        // Wire up auto-chaining from G1→G2 to downstream G2→G3/G2→G4 pipelines
        let chain_router_handle = if let Some(ref mut g1_to_g2_pipeline) = g1_to_g2 {
            if g1_to_g2_pipeline.auto_chain() {
                if let Some(chain_rx) = g1_to_g2_pipeline.take_chain_rx() {
                    // Get references to downstream pipeline queues
                    let g2_to_g3_queue = g2_to_g3.as_ref().map(|p| p.eval_queue.clone());
                    let g2_to_g4_queue = g2_to_g4.as_ref().map(|p| p.eval_queue.clone());

                    // Only spawn if there's at least one downstream pipeline
                    if g2_to_g3_queue.is_some() || g2_to_g4_queue.is_some() {
                        tracing::debug!(
                            has_g2_to_g3 = g2_to_g3_queue.is_some(),
                            has_g2_to_g4 = g2_to_g4_queue.is_some(),
                            "Spawning chain router for G1→G2 auto-chaining"
                        );
                        Some(runtime.spawn(chain_router_task(
                            chain_rx,
                            g2_to_g3_queue,
                            g2_to_g4_queue,
                        )))
                    } else {
                        tracing::debug!(
                            "G1→G2 auto_chain enabled but no downstream pipelines configured"
                        );
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(OffloadEngine {
            leader: self.leader,
            registry,
            g1_to_g2,
            g2_to_g3,
            g2_to_g4,
            transfers: Arc::new(DashMap::new()),
            _chain_router_handle: chain_router_handle,
        })
    }
}

/// Routes chain output from G1→G2 to downstream G2→G3 and G2→G4 pipelines.
///
/// Blocks are converted to WeakBlocks for best-effort offloading - if they're
/// evicted before the downstream pipeline processes them, that's acceptable.
/// This enables graceful degradation under memory pressure.
async fn chain_router_task(
    mut chain_rx: ChainOutputRx<G2>,
    g2_to_g3_queue: Option<Arc<CancellableQueue<PipelineInput<G2>>>>,
    g2_to_g4_queue: Option<Arc<CancellableQueue<PipelineInput<G2>>>>,
) {
    while let Some(output) = chain_rx.recv().await {
        let ChainOutput {
            transfer_id,
            blocks,
            state,
        } = output;

        if blocks.is_empty() {
            continue;
        }

        // Convert strong blocks to weak blocks for best-effort downstream processing
        // This allows blocks to be evicted if memory pressure requires it
        let weak_blocks: Vec<WeakBlock<G2>> =
            blocks.iter().map(|block| block.downgrade()).collect();

        // Drop strong references - blocks can now be evicted if needed
        drop(blocks);

        tracing::debug!(
            %transfer_id,
            num_blocks = weak_blocks.len(),
            "Routing chain output to downstream pipelines as WeakBlocks"
        );

        // Enqueue to G2→G3 if available
        if let Some(ref queue) = g2_to_g3_queue {
            let input = PipelineInput {
                transfer_id,
                source: SourceBlocks::Weak(weak_blocks.clone()),
                state: state.clone(),
            };
            if !queue.push(transfer_id, input) {
                tracing::debug!(%transfer_id, "G2→G3 chain enqueue skipped (cancelled)");
            }
        }

        // Enqueue to G2→G4 if available
        if let Some(ref queue) = g2_to_g4_queue {
            let input = PipelineInput {
                transfer_id,
                source: SourceBlocks::Weak(weak_blocks),
                state,
            };
            if !queue.push(transfer_id, input) {
                tracing::debug!(%transfer_id, "G2→G4 chain enqueue skipped (cancelled)");
            }
        }
    }

    tracing::debug!("Chain router task shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require complex infrastructure setup (InstanceLeader, BlockManagers, etc.)
    // Basic API tests here.

    #[test]
    fn test_transfer_id_generation() {
        let id1 = TransferId::new();
        let id2 = TransferId::new();
        assert_ne!(id1, id2);
    }
}
