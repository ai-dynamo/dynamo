// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline coordination for offload transfers.
//!
//! A pipeline connects these stages:
//! 1. **PolicyEvaluator**: Evaluates blocks against policies, filters out non-passing blocks
//! 2. **BatchCollector**: Accumulates passing blocks into batches
//! 3. **PreconditionAwaiter**: Awaits precondition events before processing
//! 4. **BlockUpgrader**: Upgrades `WeakBlock` → `ImmutableBlock` (via `upgrade_batch`)
//! 5. **Transfer Executor**: Executes the actual data transfer
//!    - `BlockTransferExecutor`: For BlockManager destinations (G2, G3)
//!    - `ObjectTransferExecutor`: For object storage destinations (G4)
//!
//! # Cancellation Architecture
//!
//! Unlike mpsc-based pipelines where cancellation only happens at dequeue boundaries,
//! this implementation uses [`CancellableQueue`] which enables a dedicated sweeper task
//! to actively remove items from cancelled transfers. This ensures that `ImmutableBlock`
//! guards are dropped promptly when a transfer is cancelled.
//!
//! ```text
//! enqueue() ─┬─► [CancellableQueue A] ──► PolicyEvaluator ──┬─► [CancellableQueue B] ──► ...
//!            │                                              │
//!            └──────────────► [CancelSweeper] ◄─────────────┘
//!                                    │
//!                              (iterates queues,
//!                               removes by TransferId,
//!                               drops ImmutableBlock guards)
//! ```

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::Either;
use tokio::sync::{Semaphore, mpsc, watch};
use tokio::task::JoinHandle;

use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::distributed::object::ObjectBlockOps;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::logical::blocks::{BlockMetadata, BlockRegistry, ImmutableBlock};
use crate::v2::logical::manager::BlockManager;
use crate::v2::physical::transfer::TransferOptions;
use crate::v2::{BlockId, SequenceHash};

use super::batch::{
    BatchCollector, BatchConfig, BatchOutputRx, EvalResult, QueuedBlock, TimingTrace, TransferBatch,
};
use super::handle::{TransferId, TransferState, TransferStatus};
use super::pending::PendingTracker;
use super::policy::{EvalContext, OffloadPolicy};
use super::queue::CancellableQueue;
use super::source::{SourceBlock, SourceBlocks};
use crate::v2::distributed::object::ObjectLockManager;

/// Helper macro to create an NVTX range when the nvtx feature is enabled.
/// The range automatically ends when the returned guard is dropped.
macro_rules! nvtx_range {
    ($name:expr) => {{
        #[cfg(feature = "nvtx")]
        let _range = nvtx::range!($name);
        #[cfg(not(feature = "nvtx"))]
        let _range = ();
        _range
    }};
}

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
    /// Sweep interval for cancellation task
    pub sweep_interval: Duration,
    /// Skip actual transfers (for testing)
    pub skip_transfers: bool,
    /// Maximum number of concurrent transfer batches.
    ///
    /// This controls how many batches can be transferred simultaneously.
    /// Setting this higher can improve throughput at the cost of memory.
    /// Default: 1 (sequential execution)
    pub max_concurrent_transfers: usize,
    /// Pending tracker for duplicate prevention.
    ///
    /// If provided, this tracker is used. If None, the pipeline creates its own.
    /// Share this tracker with presence-based policies to prevent duplicate transfers.
    pub pending_tracker: Option<Arc<PendingTracker>>,
    /// Maximum number of concurrent precondition awaits.
    ///
    /// This controls how many batches can be awaiting their preconditions simultaneously.
    /// Allows multiple iterations to be in-flight without blocking the pipeline.
    /// Default: 8 (allows ~8 iterations in-flight concurrently)
    pub max_concurrent_precondition_awaits: usize,
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
            sweep_interval: Duration::from_millis(10),
            skip_transfers: false,
            max_concurrent_transfers: 1,
            pending_tracker: None,
            max_concurrent_precondition_awaits: 8,
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

    /// Set the sweep interval for cancellation.
    pub fn sweep_interval(mut self, interval: Duration) -> Self {
        self.config.sweep_interval = interval;
        self
    }

    /// Skip actual transfers (for testing).
    ///
    /// When enabled, the transfer executor will mark blocks as completed
    /// without executing actual data transfers.
    pub fn skip_transfers(mut self, skip: bool) -> Self {
        self.config.skip_transfers = skip;
        self
    }

    /// Set maximum concurrent transfers.
    ///
    /// This controls how many batches can be transferred simultaneously.
    /// Must be at least 1.
    ///
    /// # Default
    /// 1 (sequential execution)
    pub fn max_concurrent_transfers(mut self, n: usize) -> Self {
        self.config.max_concurrent_transfers = n.max(1);
        self
    }

    /// Set the pending tracker for duplicate prevention.
    ///
    /// Share this tracker with presence-based policies (via `create_policy_from_config`)
    /// to prevent duplicate transfers when overlapping sequences are enqueued.
    pub fn pending_tracker(mut self, tracker: Arc<PendingTracker>) -> Self {
        self.config.pending_tracker = Some(tracker);
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
    /// Source blocks - can be External, Strong, or Weak
    pub(crate) source: SourceBlocks<T>,
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
}

/// Output from the pipeline (completed transfer).
pub struct PipelineOutput {
    pub transfer_id: TransferId,
    pub completed_hashes: Vec<SequenceHash>,
}

/// Chain output - carries registered blocks for downstream pipelines.
///
/// When `auto_chain` is enabled, the pipeline sends registered blocks
/// through this channel instead of dropping them. The receiving pipeline
/// can then process them through its own policy evaluation and transfer.
pub struct ChainOutput<T: BlockMetadata> {
    pub transfer_id: TransferId,
    pub blocks: Vec<ImmutableBlock<T>>,
    /// State for transfer tracking (used when feeding downstream pipelines)
    #[allow(dead_code)]
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
}

/// Receiver for chain output from a pipeline.
pub type ChainOutputRx<T> = mpsc::Receiver<ChainOutput<T>>;

/// A running pipeline instance.
pub struct Pipeline<Src: BlockMetadata, Dst: BlockMetadata> {
    config: PipelineConfig<Src, Dst>,
    /// Input queue for new blocks (CancellableQueue for sweep support)
    pub(crate) eval_queue: Arc<CancellableQueue<PipelineInput<Src>>>,
    /// Output channel for completed blocks (may feed downstream)
    output_tx: Option<mpsc::Sender<PipelineOutput>>,
    /// Chain output receiver - provides registered blocks for downstream pipelines
    chain_rx: Option<ChainOutputRx<Dst>>,
    /// Watch channel for cancelled transfer IDs (triggers sweep)
    cancel_tx: watch::Sender<HashSet<TransferId>>,
    /// Tracker for pending (in-flight) transfers to prevent duplicates
    pending_tracker: Arc<PendingTracker>,
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
    /// * `dst_manager` - Destination tier block manager
    /// * `leader` - Instance leader for transfer execution
    /// * `src_layout` - Source logical layout handle
    /// * `dst_layout` - Destination logical layout handle
    /// * `runtime` - Tokio runtime handle for spawning background tasks
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: PipelineConfig<Src, Dst>,
        _registry: Arc<BlockRegistry>,
        dst_manager: Arc<BlockManager<Dst>>,
        leader: Arc<InstanceLeader>,
        src_layout: LogicalLayoutHandle,
        dst_layout: LogicalLayoutHandle,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        // Create cancellable queues
        let eval_queue: Arc<CancellableQueue<PipelineInput<Src>>> =
            Arc::new(CancellableQueue::new());
        let batch_queue: Arc<CancellableQueue<EvalResult<Src>>> = Arc::new(CancellableQueue::new());

        // Create output channel (still mpsc for downstream chaining)
        let (output_tx, _output_rx) = mpsc::channel(64);

        // Create watch channel for cancelled transfer IDs
        let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());

        // Create batch output channel (BatchCollector → PreconditionAwaiter)
        let (batch_tx, batch_rx) = mpsc::channel(config.transfer_input_capacity);

        // Create precondition output channel (PreconditionAwaiter → TransferExecutor)
        let (precond_tx, precond_rx) = mpsc::channel(config.transfer_input_capacity);

        // Create chain output channel if auto_chain is enabled
        let (chain_tx, chain_rx) = if config.auto_chain {
            let (tx, rx) = mpsc::channel(64);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Use provided pending tracker or create a new one
        let pending_tracker = config
            .pending_tracker
            .clone()
            .unwrap_or_else(|| Arc::new(PendingTracker::new()));

        // Spawn policy evaluator
        let evaluator = PolicyEvaluator {
            policies: config.policies.clone(),
            timeout: config.policy_timeout,
            input_queue: eval_queue.clone(),
            output_queue: batch_queue.clone(),
            cancel_rx: cancel_rx.clone(),
            pending_tracker: pending_tracker.clone(),
        };
        let eval_handle = runtime.spawn(async move {
            evaluator.run().await;
        });

        // Spawn batch collector (reads from CancellableQueue, outputs to mpsc)
        let collector_input_queue = batch_queue.clone();
        let batch_config = config.batch_config.clone();
        let collector_cancel_rx = cancel_rx.clone();
        let batch_handle = runtime.spawn(async move {
            let collector = BatchCollector::new_with_queue(
                batch_config,
                collector_input_queue,
                batch_tx,
                collector_cancel_rx,
            );
            // BatchCollector::new_with_queue returns BatchCollectorQueue
            collector.run().await;
        });

        // Spawn precondition awaiter (reads from batch_rx, outputs to precond_tx)
        let awaiter_leader = leader.clone();
        let precond_handle = runtime.spawn(async move {
            let awaiter = PreconditionAwaiter {
                input_rx: batch_rx,
                output_tx: precond_tx,
                leader: awaiter_leader,
            };
            awaiter.run().await;
        });

        // Spawn block transfer executor (reads from precond_rx)
        let executor = BlockTransferExecutor {
            input_rx: precond_rx,
            leader,
            dst_manager,
            src_layout,
            dst_layout,
            skip_transfers: config.skip_transfers,
            max_concurrent_transfers: config.max_concurrent_transfers,
            chain_tx,
            _src_marker: PhantomData::<Src>,
        };
        let transfer_handle = runtime.spawn(async move {
            executor.run().await;
        });

        // Spawn cancel sweeper
        let sweeper_queues = vec![eval_queue.clone()];
        let sweeper_batch_queue = batch_queue;
        let sweeper_interval = config.sweep_interval;
        let sweeper_cancel_rx = cancel_rx;
        let sweeper_handle = runtime.spawn(async move {
            cancel_sweeper(
                sweeper_queues,
                sweeper_batch_queue,
                sweeper_cancel_rx,
                sweeper_interval,
            )
            .await;
        });

        Self {
            config,
            eval_queue,
            output_tx: Some(output_tx),
            chain_rx,
            cancel_tx,
            pending_tracker,
            _task_handles: vec![
                eval_handle,
                batch_handle,
                precond_handle,
                transfer_handle,
                sweeper_handle,
            ],
            _marker: PhantomData,
        }
    }

    /// Enqueue blocks for offloading through this pipeline.
    pub(crate) fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<Src>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        tracing::debug!(%transfer_id, num_blocks = source.len(), "Pipeline: enqueueing blocks");
        let input = PipelineInput {
            transfer_id,
            source,
            state,
        };
        self.eval_queue.push(transfer_id, input)
    }

    /// Request cancellation for a transfer.
    ///
    /// This marks the transfer as cancelled in all queues, triggering the sweeper
    /// to remove queued items and the evaluator/collector to skip them.
    pub fn request_cancel(&self, transfer_id: TransferId) {
        // Mark cancelled in queues
        self.eval_queue.mark_cancelled(transfer_id);

        // Notify sweeper via watch channel
        self.cancel_tx.send_modify(|set| {
            set.insert(transfer_id);
        });
    }

    /// Check if this pipeline auto-chains to downstream.
    pub fn auto_chain(&self) -> bool {
        self.config.auto_chain
    }

    /// Get a clone of the output channel sender.
    pub fn output_tx(&self) -> Option<mpsc::Sender<PipelineOutput>> {
        self.output_tx.clone()
    }

    /// Take the chain output receiver for downstream pipeline feeding.
    ///
    /// This transfers ownership of the receiver - can only be called once.
    /// When `auto_chain` is enabled, this receiver will yield `ChainOutput<Dst>`
    /// containing registered blocks that can be fed to a downstream pipeline.
    ///
    /// # Returns
    /// - `Some(rx)` if `auto_chain` is enabled and receiver hasn't been taken
    /// - `None` if `auto_chain` is false or receiver was already taken
    pub fn take_chain_rx(&mut self) -> Option<ChainOutputRx<Dst>> {
        self.chain_rx.take()
    }

    /// Get the pending tracker for this pipeline.
    ///
    /// This can be shared with presence policies to enable duplicate prevention
    /// for blocks currently in-flight through this pipeline.
    pub fn pending_tracker(&self) -> &Arc<PendingTracker> {
        &self.pending_tracker
    }
}

// ============================================================================
// Object Pipeline (for G4 / object storage destinations)
// ============================================================================

/// Configuration for an object storage pipeline.
///
/// Similar to `PipelineConfig` but designed for object storage destinations
/// that don't use a `BlockManager`. The destination is `ObjectBlockOps`.
#[derive(Clone)]
pub struct ObjectPipelineConfig<Src: BlockMetadata> {
    /// Policies to evaluate (all must pass)
    pub policies: Vec<Arc<dyn OffloadPolicy<Src>>>,
    /// Batch configuration
    pub batch_config: BatchConfig,
    /// Timeout for policy evaluation (fail-fast)
    pub policy_timeout: Duration,
    /// Channel capacity for evaluation input
    pub eval_input_capacity: usize,
    /// Channel capacity for batch input
    pub batch_input_capacity: usize,
    /// Channel capacity for transfer input
    pub transfer_input_capacity: usize,
    /// Sweep interval for cancellation task
    pub sweep_interval: Duration,
    /// Skip actual transfers (for testing)
    pub skip_transfers: bool,
    /// Maximum number of concurrent transfer batches
    pub max_concurrent_transfers: usize,
    /// Pending tracker for duplicate prevention
    pub pending_tracker: Option<Arc<PendingTracker>>,
    /// Maximum concurrent precondition awaits
    pub max_concurrent_precondition_awaits: usize,
    /// Lock manager for distributed locking (optional)
    ///
    /// When provided, the executor will:
    /// - Create `.meta` files after successful transfers
    /// - Release `.lock` files after transfer completion
    pub lock_manager: Option<Arc<dyn ObjectLockManager>>,
    /// Marker
    _marker: PhantomData<Src>,
}

impl<Src: BlockMetadata> Default for ObjectPipelineConfig<Src> {
    fn default() -> Self {
        Self {
            policies: Vec::new(),
            batch_config: BatchConfig::default(),
            policy_timeout: Duration::from_millis(100),
            eval_input_capacity: 128,
            batch_input_capacity: 256,
            transfer_input_capacity: 8,
            sweep_interval: Duration::from_millis(10),
            skip_transfers: false,
            max_concurrent_transfers: 1,
            pending_tracker: None,
            max_concurrent_precondition_awaits: 8,
            lock_manager: None,
            _marker: PhantomData,
        }
    }
}

/// Builder for object pipeline configuration.
pub struct ObjectPipelineBuilder<Src: BlockMetadata> {
    config: ObjectPipelineConfig<Src>,
}

impl<Src: BlockMetadata> ObjectPipelineBuilder<Src> {
    /// Create a new object pipeline builder with defaults.
    pub fn new() -> Self {
        Self {
            config: ObjectPipelineConfig::default(),
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

    /// Set the sweep interval for cancellation.
    pub fn sweep_interval(mut self, interval: Duration) -> Self {
        self.config.sweep_interval = interval;
        self
    }

    /// Skip actual transfers (for testing).
    pub fn skip_transfers(mut self, skip: bool) -> Self {
        self.config.skip_transfers = skip;
        self
    }

    /// Set maximum concurrent transfers.
    pub fn max_concurrent_transfers(mut self, n: usize) -> Self {
        self.config.max_concurrent_transfers = n.max(1);
        self
    }

    /// Set the pending tracker for duplicate prevention.
    pub fn pending_tracker(mut self, tracker: Arc<PendingTracker>) -> Self {
        self.config.pending_tracker = Some(tracker);
        self
    }

    /// Set the lock manager for distributed locking.
    ///
    /// When provided, the executor will create `.meta` files after successful
    /// transfers and release `.lock` files after completion.
    pub fn lock_manager(mut self, manager: Arc<dyn ObjectLockManager>) -> Self {
        self.config.lock_manager = Some(manager);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ObjectPipelineConfig<Src> {
        self.config
    }
}

impl<Src: BlockMetadata> Default for ObjectPipelineBuilder<Src> {
    fn default() -> Self {
        Self::new()
    }
}

/// A running pipeline instance for object storage destinations.
///
/// Similar to `Pipeline` but uses `ObjectTransferExecutor` for G4 (object storage)
/// instead of `BlockTransferExecutor`. There is no destination `BlockManager`.
#[allow(dead_code)]
pub struct ObjectPipeline<Src: BlockMetadata> {
    config: ObjectPipelineConfig<Src>,
    /// Input queue for new blocks (CancellableQueue for sweep support)
    pub(crate) eval_queue: Arc<CancellableQueue<PipelineInput<Src>>>,
    /// Output channel for completed blocks
    output_tx: Option<mpsc::Sender<PipelineOutput>>,
    /// Watch channel for cancelled transfer IDs (triggers sweep)
    cancel_tx: watch::Sender<HashSet<TransferId>>,
    /// Tracker for pending (in-flight) transfers to prevent duplicates
    pending_tracker: Arc<PendingTracker>,
    /// Task handles for pipeline stages
    _task_handles: Vec<JoinHandle<()>>,
}

impl<Src: BlockMetadata> ObjectPipeline<Src> {
    /// Create a new object pipeline with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Pipeline configuration
    /// * `object_ops` - Object storage operations (e.g., S3 client)
    /// * `src_layout` - Source physical layout for reading block data
    /// * `leader` - Instance leader for precondition events
    /// * `runtime` - Tokio runtime handle for spawning background tasks
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ObjectPipelineConfig<Src>,
        object_ops: Arc<dyn ObjectBlockOps>,
        src_layout: LogicalLayoutHandle,
        leader: Arc<InstanceLeader>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        // Create cancellable queues
        let eval_queue: Arc<CancellableQueue<PipelineInput<Src>>> =
            Arc::new(CancellableQueue::new());
        let batch_queue: Arc<CancellableQueue<EvalResult<Src>>> = Arc::new(CancellableQueue::new());

        // Create output channel
        let (output_tx, _output_rx) = mpsc::channel(64);

        // Create watch channel for cancelled transfer IDs
        let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());

        // Create batch output channel (BatchCollector → PreconditionAwaiter)
        let (batch_tx, batch_rx) = mpsc::channel(config.transfer_input_capacity);

        // Create precondition output channel (PreconditionAwaiter → ObjectTransferExecutor)
        let (precond_tx, precond_rx) = mpsc::channel(config.transfer_input_capacity);

        // Use provided pending tracker or create a new one
        let pending_tracker = config
            .pending_tracker
            .clone()
            .unwrap_or_else(|| Arc::new(PendingTracker::new()));

        // Spawn policy evaluator
        let evaluator = PolicyEvaluator {
            policies: config.policies.clone(),
            timeout: config.policy_timeout,
            input_queue: eval_queue.clone(),
            output_queue: batch_queue.clone(),
            cancel_rx: cancel_rx.clone(),
            pending_tracker: pending_tracker.clone(),
        };
        let eval_handle = runtime.spawn(async move {
            evaluator.run().await;
        });

        // Spawn batch collector
        let collector_input_queue = batch_queue.clone();
        let batch_config = config.batch_config.clone();
        let collector_cancel_rx = cancel_rx.clone();
        let batch_handle = runtime.spawn(async move {
            let collector = BatchCollector::new_with_queue(
                batch_config,
                collector_input_queue,
                batch_tx,
                collector_cancel_rx,
            );
            collector.run().await;
        });

        // Spawn precondition awaiter
        let awaiter_leader = leader.clone();
        let precond_handle = runtime.spawn(async move {
            let awaiter = PreconditionAwaiter {
                input_rx: batch_rx,
                output_tx: precond_tx,
                leader: awaiter_leader,
            };
            awaiter.run().await;
        });

        // Spawn object transfer executor
        let executor = ObjectTransferExecutor::new(
            precond_rx,
            object_ops,
            src_layout,
            config.skip_transfers,
            config.max_concurrent_transfers,
            config.lock_manager.clone(),
        );
        let transfer_handle = runtime.spawn(async move {
            executor.run().await;
        });

        // Spawn cancel sweeper
        let sweeper_queues = vec![eval_queue.clone()];
        let sweeper_batch_queue = batch_queue;
        let sweeper_interval = config.sweep_interval;
        let sweeper_cancel_rx = cancel_rx;
        let sweeper_handle = runtime.spawn(async move {
            cancel_sweeper(
                sweeper_queues,
                sweeper_batch_queue,
                sweeper_cancel_rx,
                sweeper_interval,
            )
            .await;
        });

        Self {
            config,
            eval_queue,
            output_tx: Some(output_tx),
            cancel_tx,
            pending_tracker,
            _task_handles: vec![
                eval_handle,
                batch_handle,
                precond_handle,
                transfer_handle,
                sweeper_handle,
            ],
        }
    }

    /// Enqueue blocks for offloading through this pipeline.
    pub(crate) fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<Src>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        tracing::debug!(%transfer_id, num_blocks = source.len(), "ObjectPipeline: enqueueing blocks");
        let input = PipelineInput {
            transfer_id,
            source,
            state,
        };
        self.eval_queue.push(transfer_id, input)
    }

    /// Request cancellation for a transfer.
    pub fn request_cancel(&self, transfer_id: TransferId) {
        self.eval_queue.mark_cancelled(transfer_id);
        self.cancel_tx.send_modify(|set| {
            set.insert(transfer_id);
        });
    }

    /// Get a clone of the output channel sender.
    #[allow(dead_code)]
    pub fn output_tx(&self) -> Option<mpsc::Sender<PipelineOutput>> {
        self.output_tx.clone()
    }

    /// Get the pending tracker for this pipeline.
    pub fn pending_tracker(&self) -> &Arc<PendingTracker> {
        &self.pending_tracker
    }
}

/// Sweeper task that removes cancelled items from queues.
async fn cancel_sweeper<Src: BlockMetadata>(
    input_queues: Vec<Arc<CancellableQueue<PipelineInput<Src>>>>,
    batch_queue: Arc<CancellableQueue<EvalResult<Src>>>,
    mut cancel_rx: watch::Receiver<HashSet<TransferId>>,
    interval: Duration,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Sweep all queues
                for queue in &input_queues {
                    let removed = queue.sweep();
                    if removed > 0 {
                        tracing::debug!("Sweeper removed {} cancelled input items", removed);
                    }
                }

                let batch_removed = batch_queue.sweep();
                if batch_removed > 0 {
                    tracing::debug!("Sweeper removed {} cancelled batch items", batch_removed);
                }
            }
            result = cancel_rx.changed() => {
                if result.is_err() {
                    // Channel closed, shutdown
                    break;
                }
                // New cancellation added, sweep immediately
                for queue in &input_queues {
                    queue.sweep();
                }
                batch_queue.sweep();
            }
        }
    }
}

/// Policy evaluator stage.
struct PolicyEvaluator<T: BlockMetadata> {
    policies: Vec<Arc<dyn OffloadPolicy<T>>>,
    timeout: Duration,
    input_queue: Arc<CancellableQueue<PipelineInput<T>>>,
    output_queue: Arc<CancellableQueue<EvalResult<T>>>,
    cancel_rx: watch::Receiver<HashSet<TransferId>>,
    /// Tracker for pending transfers - guards are created when blocks pass policy
    pending_tracker: Arc<PendingTracker>,
}

impl<T: BlockMetadata> PolicyEvaluator<T> {
    async fn run(self) {
        let mut poll_interval = tokio::time::interval(Duration::from_micros(100));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            // Poll for items
            if let Some(item) = self.input_queue.pop_valid() {
                self.evaluate(item.data).await;
            } else {
                // No items available, wait a bit
                poll_interval.tick().await;
            }

            // Check for shutdown (cancel channel closed)
            if self.cancel_rx.has_changed().is_err() {
                break;
            }
        }
    }

    async fn evaluate(&self, input: PipelineInput<T>) {
        let _nvtx = nvtx_range!("offload::policy");
        let transfer_id = input.transfer_id;

        // Set total_expected_blocks for per-transfer sentinel flush
        let total_blocks = input.source.len();
        {
            let mut state = input.state.lock().unwrap();
            state.total_expected_blocks = total_blocks;
        }

        // Check if already cancelled (via queue or via handle)
        {
            let state = input.state.lock().unwrap();
            if state.is_cancel_requested() {
                drop(state); // Release lock before calling set_cancelled
                tracing::debug!(%transfer_id, "Transfer cancelled before evaluation");
                let mut state = input.state.lock().unwrap();
                state.set_cancelled();
                return;
            }
        }

        let mut passed = Vec::new();
        let mut filtered = Vec::new();

        // Process blocks based on source type
        match input.source {
            SourceBlocks::External(external_blocks) => {
                // External blocks (e.g., G1 from vLLM) still need policy evaluation
                // to check presence in destination tier
                for ext in external_blocks {
                    // Check for cancellation between blocks
                    if self.check_cancelled(&input.state, transfer_id) {
                        return;
                    }

                    // Create context with sequence_hash - block_id is known for External
                    let ctx = EvalContext::from_external(ext.block_id, ext.sequence_hash);
                    let pass = self.evaluate_policies(&ctx).await;

                    if pass {
                        // Create pending guard for duplicate prevention
                        let pending_guard = self.pending_tracker.guard(ext.sequence_hash);
                        passed.push(QueuedBlock {
                            transfer_id,
                            block_id: Some(ext.block_id),
                            sequence_hash: ext.sequence_hash,
                            source: SourceBlock::External(ext),
                            state: input.state.clone(),
                            pending_guard: Some(pending_guard),
                        });
                    } else {
                        filtered.push(ext.block_id);
                    }
                }
                tracing::debug!(%transfer_id, passed = passed.len(), filtered = filtered.len(), "External blocks evaluated");
            }
            SourceBlocks::Strong(strong_blocks) => {
                // Strong blocks get full policy evaluation
                for block in strong_blocks {
                    // Check for cancellation between blocks
                    if self.check_cancelled(&input.state, transfer_id) {
                        return;
                    }

                    let ctx = EvalContext::new(block);
                    let pass = self.evaluate_policies(&ctx).await;

                    if pass {
                        let block = ctx.block.expect("Strong block context always has block");
                        // Create pending guard for duplicate prevention
                        let pending_guard = self.pending_tracker.guard(ctx.sequence_hash);
                        passed.push(QueuedBlock {
                            transfer_id,
                            block_id: Some(ctx.block_id),
                            sequence_hash: ctx.sequence_hash,
                            source: SourceBlock::Strong(block),
                            state: input.state.clone(),
                            pending_guard: Some(pending_guard),
                        });
                    } else {
                        filtered.push(ctx.block_id);
                    }
                }
            }
            SourceBlocks::Weak(weak_blocks) => {
                // Weak blocks get policy evaluation using metadata (deferred upgrade)
                // block_id is unknown until upgrade at transfer time
                for weak in weak_blocks {
                    // Check for cancellation between blocks
                    if self.check_cancelled(&input.state, transfer_id) {
                        return;
                    }

                    let sequence_hash = weak.sequence_hash();
                    let ctx = EvalContext::from_weak(BlockId::default(), sequence_hash);
                    let pass = self.evaluate_policies(&ctx).await;

                    if pass {
                        // Create pending guard for duplicate prevention
                        let pending_guard = self.pending_tracker.guard(sequence_hash);
                        passed.push(QueuedBlock {
                            transfer_id,
                            block_id: None, // Determined at upgrade time
                            sequence_hash,
                            source: SourceBlock::Weak(weak),
                            state: input.state.clone(),
                            pending_guard: Some(pending_guard),
                        });
                    } else {
                        // For weak blocks, we track by sequence_hash since block_id is unknown
                        // We'll add sequence_hash tracking in TransferState if needed
                        tracing::debug!(%transfer_id, ?sequence_hash, "Weak block filtered by policy");
                    }
                }
            }
        }

        // Check for cancellation after evaluation
        {
            let state = input.state.lock().unwrap();
            if state.is_cancel_requested() {
                drop(state);
                tracing::debug!(%transfer_id, "Transfer cancelled after evaluation");
                let mut state = input.state.lock().unwrap();
                state.set_cancelled();
                return;
            }
        }

        tracing::debug!(%transfer_id, passed = passed.len(), filtered = filtered.len(), "Policy evaluation complete");

        // Update state with evaluation results
        {
            let mut state = input.state.lock().unwrap();
            // Only track block_ids for blocks that have them (External/Strong)
            // Weak blocks don't have block_id until upgrade
            state.add_passed(passed.iter().filter_map(|b| b.block_id));
            state.add_filtered(filtered.iter().copied());
            state.set_status(TransferStatus::Queued);
        }

        // Check if all blocks were filtered (transfer complete with no transfers)
        if passed.is_empty() {
            tracing::debug!(%transfer_id, "All blocks filtered, completing transfer");
            let mut state = input.state.lock().unwrap();
            state.set_complete();
            return;
        }

        // Send to batch collector
        let result = EvalResult {
            transfer_id,
            passed_blocks: passed,
            filtered_ids: filtered,
            state: input.state,
        };

        if !self.output_queue.push(transfer_id, result) {
            tracing::debug!(%transfer_id, "Push to output queue failed (cancelled)");
        }
    }

    /// Check if transfer is cancelled and handle state update.
    fn check_cancelled(
        &self,
        state: &Arc<std::sync::Mutex<TransferState>>,
        transfer_id: TransferId,
    ) -> bool {
        let state_guard = state.lock().unwrap();
        if state_guard.is_cancel_requested() {
            drop(state_guard);
            tracing::debug!(%transfer_id, "Transfer cancelled mid-evaluation");
            let mut state_guard = state.lock().unwrap();
            state_guard.set_cancelled();
            true
        } else {
            false
        }
    }

    async fn evaluate_policies(&self, ctx: &EvalContext<T>) -> bool {
        for policy in &self.policies {
            let eval_future = policy.evaluate(ctx);
            let timed_result = tokio::time::timeout(self.timeout, async {
                match eval_future {
                    Either::Left(ready) => ready.await,
                    Either::Right(boxed) => boxed.await,
                }
            })
            .await;

            match timed_result {
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

// ============================================================================
// Block Upgrader Types
// ============================================================================

/// A resolved block ready for transfer execution.
///
/// Created during the upgrade stage when `WeakBlock` references are upgraded
/// to `ImmutableBlock` guards. This type is used by both `BlockTransferExecutor`
/// and `ObjectTransferExecutor`.
pub struct ResolvedBlock<T: BlockMetadata> {
    /// Transfer ID this block belongs to
    pub transfer_id: TransferId,
    /// Block ID in the source layout
    pub block_id: BlockId,
    /// Sequence hash identifying the block content
    pub sequence_hash: SequenceHash,
    /// Guard holding the block - Some for Strong/Weak, None for External.
    /// The guard is held to prevent eviction during transfer.
    #[allow(dead_code)]
    pub guard: Option<ImmutableBlock<T>>,
    /// Transfer state for progress tracking
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
}

/// A batch of resolved blocks ready for transfer.
///
/// This is the output of the block upgrade stage and input to transfer executors.
pub struct ResolvedBatch<T: BlockMetadata> {
    /// Resolved blocks ready for transfer
    pub blocks: Vec<ResolvedBlock<T>>,
    /// Sequence hashes of blocks that were evicted during upgrade
    #[allow(dead_code)]
    pub evicted: Vec<SequenceHash>,
    /// Timing trace from the original batch (batch-level, not per-block)
    pub timing: TimingTrace,
}

impl<T: BlockMetadata> ResolvedBatch<T> {
    /// Check if the batch has any resolved blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get the number of resolved blocks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// Upgrade a batch of queued blocks by resolving weak references.
///
/// This is the "block upgrader" stage that converts `TransferBatch` (containing
/// mixed `SourceBlock` types) into `ResolvedBatch` (containing only resolved
/// `ImmutableBlock` guards).
///
/// # Block Type Handling
///
/// - `Strong`: Already have a guard, pass through directly
/// - `External`: No guard needed, caller holds the reference
/// - `Weak`: Attempt upgrade; if evicted, record in `evicted` list
///
/// This function is synchronous CPU work that can run in an "on-deck" slot
/// while other transfers are executing.
pub fn upgrade_batch<T: BlockMetadata>(batch: TransferBatch<T>) -> ResolvedBatch<T> {
    let mut resolved: Vec<ResolvedBlock<T>> = Vec::with_capacity(batch.len());
    let mut evicted: Vec<SequenceHash> = Vec::new();

    // Copy timing from batch and mark transfer start (O(1), not per-block)
    let mut timing = batch.timing;
    timing.mark_transfer_start();

    for queued in batch.blocks {
        // Note: pending_guard is automatically dropped when QueuedBlock is processed,
        // which removes the sequence_hash from the pending set. This happens either
        // when the block is resolved and transferred, or when it's evicted/dropped.
        match queued.source {
            SourceBlock::Strong(block) => {
                resolved.push(ResolvedBlock {
                    transfer_id: queued.transfer_id,
                    block_id: block.block_id(),
                    sequence_hash: queued.sequence_hash,
                    guard: Some(block),
                    state: queued.state,
                });
            }
            SourceBlock::External(ext) => {
                resolved.push(ResolvedBlock {
                    transfer_id: queued.transfer_id,
                    block_id: ext.block_id,
                    sequence_hash: ext.sequence_hash,
                    guard: None,
                    state: queued.state,
                });
            }
            SourceBlock::Weak(weak) => match weak.upgrade() {
                Some(block) => {
                    resolved.push(ResolvedBlock {
                        transfer_id: queued.transfer_id,
                        block_id: block.block_id(),
                        sequence_hash: queued.sequence_hash,
                        guard: Some(block),
                        state: queued.state,
                    });
                }
                None => {
                    tracing::debug!(
                        sequence_hash = ?queued.sequence_hash,
                        "Weak block evicted before transfer"
                    );
                    evicted.push(queued.sequence_hash);
                }
            },
        }
    }

    ResolvedBatch {
        blocks: resolved,
        evicted,
        timing,
    }
}

// ============================================================================
// Precondition Awaiter
// ============================================================================

/// Precondition awaiter stage.
///
/// Sits between BatchCollector and the transfer executors, awaiting precondition events
/// before forwarding batches. Spawns unbounded tasks to ensure all preconditions
/// are awaited - event awaiting is cheap (just waiting, no compute), so we never
/// skip awaiting a precondition to prevent deadlock scenarios.
struct PreconditionAwaiter<T: BlockMetadata> {
    input_rx: BatchOutputRx<T>,
    output_tx: mpsc::Sender<TransferBatch<T>>,
    leader: Arc<InstanceLeader>,
}

impl<T: BlockMetadata> PreconditionAwaiter<T> {
    async fn run(mut self) {
        // NO SEMAPHORE - spawn unbounded tasks
        // Event awaiting is cheap, we must never skip awaiting a precondition
        while let Some(mut batch) = self.input_rx.recv().await {
            let output_tx = self.output_tx.clone();
            let nova = self.leader.nova().clone();

            // Spawn task for each batch - unbounded
            tokio::spawn(async move {
                let _nvtx = nvtx_range!("offload::precondition");
                if let Some(event_handle) = batch.precondition {
                    tracing::debug!(?event_handle, "Awaiting precondition for batch");

                    // Create awaiter (returns Result<LocalEventWaiter, Error>)
                    let awaiter_result = nova.events().awaiter(event_handle);

                    match awaiter_result {
                        Ok(awaiter) => {
                            // Now await the LocalEventWaiter with timeout
                            match tokio::time::timeout(Duration::from_secs(300), awaiter).await {
                                Ok(Ok(())) => {
                                    tracing::debug!(?event_handle, "Precondition satisfied");
                                }
                                Ok(Err(poison)) => {
                                    tracing::error!(
                                        ?event_handle,
                                        ?poison,
                                        "Precondition poisoned, marking all blocks as failed"
                                    );
                                    // Mark all blocks as failed
                                    for queued in batch.blocks {
                                        let mut state = queued.state.lock().unwrap();
                                        state.set_error(format!(
                                            "precondition poisoned: {:?}",
                                            poison
                                        ));
                                    }
                                    return;
                                }
                                Err(_) => {
                                    tracing::error!(
                                        ?event_handle,
                                        "Precondition timeout after 30s"
                                    );
                                    // Mark all blocks as failed
                                    for queued in batch.blocks {
                                        let mut state = queued.state.lock().unwrap();
                                        state.set_error("precondition timeout".to_string());
                                    }
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(?event_handle, ?e, "Failed to create awaiter");
                            // Mark all blocks as failed
                            for queued in batch.blocks {
                                let mut state = queued.state.lock().unwrap();
                                state.set_error(format!("failed to create awaiter: {}", e));
                            }
                            return;
                        }
                    }
                }

                // Mark precondition complete (batch-level, O(1))
                batch.timing.mark_precondition_complete();

                // Forward batch to transfer executor
                if let Err(e) = output_tx.send(batch).await {
                    tracing::error!("Failed to forward batch after precondition: {}", e);
                }
            });
        }
    }
}

// ============================================================================
// Block Transfer Executor (for G2, G3 destinations)
// ============================================================================

/// Block transfer executor stage for BlockManager-based destinations.
///
/// Executes transfers to destinations with a `BlockManager` (G2, G3).
/// Uses `leader.execute_local_transfer()` to copy block data between layouts.
///
/// For object storage destinations (G4), use `ObjectTransferExecutor` instead.
struct BlockTransferExecutor<Src: BlockMetadata, Dst: BlockMetadata> {
    input_rx: BatchOutputRx<Src>,
    leader: Arc<InstanceLeader>,
    dst_manager: Arc<BlockManager<Dst>>,
    src_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
    /// Skip actual transfers (for testing)
    skip_transfers: bool,
    /// Maximum concurrent transfers
    max_concurrent_transfers: usize,
    /// Channel to send registered blocks for chaining to downstream pipeline
    chain_tx: Option<mpsc::Sender<ChainOutput<Dst>>>,
    _src_marker: PhantomData<Src>,
}

/// Shared state for BlockTransferExecutor that can be cloned across concurrent tasks.
struct SharedBlockExecutorState<Dst: BlockMetadata> {
    leader: Arc<InstanceLeader>,
    dst_manager: Arc<BlockManager<Dst>>,
    src_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
    skip_transfers: bool,
    chain_tx: Option<mpsc::Sender<ChainOutput<Dst>>>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> BlockTransferExecutor<Src, Dst> {
    async fn run(mut self) {
        // N slots for active transfers
        let transfer_semaphore = Arc::new(Semaphore::new(self.max_concurrent_transfers));
        // 1 slot for preparation (upgrade) work - on-deck
        let prepare_semaphore = Arc::new(Semaphore::new(1));

        // Extract shared state for concurrent tasks
        let shared = Arc::new(SharedBlockExecutorState {
            leader: self.leader.clone(),
            dst_manager: self.dst_manager.clone(),
            src_layout: self.src_layout,
            dst_layout: self.dst_layout,
            skip_transfers: self.skip_transfers,
            chain_tx: self.chain_tx.take(),
        });

        while let Some(batch) = self.input_rx.recv().await {
            if batch.is_empty() {
                continue;
            }

            // Wait for prepare slot (only 1 batch preparing at a time)
            // This is the "on-deck" slot for preparing while transfers run
            let prepare_permit = prepare_semaphore.clone().acquire_owned().await;
            if prepare_permit.is_err() {
                break; // Semaphore closed
            }
            let prepare_permit = prepare_permit.unwrap();

            // Prepare stage: resolve/upgrade blocks (weak→strong)
            // This happens in the "on-deck" slot while other transfers may be running
            let upgraded = upgrade_batch(batch);

            // Done preparing, release prepare slot for next batch
            drop(prepare_permit);

            if upgraded.is_empty() {
                tracing::debug!("All blocks in batch evicted, skipping transfer");
                continue;
            }

            // Now wait for transfer slot
            let transfer_permit = transfer_semaphore.clone().acquire_owned().await;
            if transfer_permit.is_err() {
                break; // Semaphore closed
            }
            let transfer_permit = transfer_permit.unwrap();

            // Spawn transfer task
            let shared_clone = shared.clone();
            tokio::spawn(async move {
                let _permit = transfer_permit; // Hold permit until task completes
                if let Err(e) = Self::execute_transfer(&shared_clone, upgraded).await {
                    tracing::warn!("BlockTransferExecutor: transfer failed: {}", e);
                }
            });
        }

        // Wait for all in-flight transfers to complete by acquiring all permits
        let _ = transfer_semaphore
            .acquire_many(self.max_concurrent_transfers as u32)
            .await;
    }

    /// Execute the actual transfer for resolved blocks.
    ///
    /// This is async I/O work that runs concurrently with other transfers.
    async fn execute_transfer(
        shared: &SharedBlockExecutorState<Dst>,
        mut batch: ResolvedBatch<Src>,
    ) -> anyhow::Result<()> {
        let _nvtx = nvtx_range!("offload::transfer");
        if batch.is_empty() {
            return Ok(());
        }

        let resolved = &batch.blocks;

        // Collect block_ids and sequence_hashes from resolved blocks
        let src_block_ids: Vec<BlockId> = resolved.iter().map(|b| b.block_id).collect();
        let sequence_hashes: Vec<SequenceHash> = resolved.iter().map(|b| b.sequence_hash).collect();

        // Collect states for completion tracking (group by transfer_id)
        let mut transfer_states: std::collections::HashMap<
            TransferId,
            (Arc<std::sync::Mutex<TransferState>>, Vec<BlockId>),
        > = std::collections::HashMap::new();
        for block in resolved {
            transfer_states
                .entry(block.transfer_id)
                .or_insert_with(|| (block.state.clone(), Vec::new()))
                .1
                .push(block.block_id);
        }

        // Skip actual transfers when in test mode
        if !shared.skip_transfers {
            // Allocate destination blocks
            let dst_blocks = shared
                .dst_manager
                .allocate_blocks(resolved.len())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Failed to allocate {} {} blocks; this is an indication of high memory pressure and may impact performance",
                        resolved.len(),
                        std::any::type_name::<Dst>()
                    )
                })?;

            let dst_block_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

            // Execute transfer via leader
            let start_xfer = Instant::now();
            let notification = shared.leader.execute_local_transfer(
                shared.src_layout,
                shared.dst_layout,
                src_block_ids.clone(),
                dst_block_ids.clone(),
                TransferOptions::default(),
            )?;

            // Wait for transfer completion
            notification.await?;
            let end_xfer = Instant::now();

            // Register each transferred block in the destination tier
            let registered_blocks: Vec<ImmutableBlock<Dst>> = dst_blocks
                .into_iter()
                .zip(sequence_hashes.iter())
                .map(|(dst_block, seq_hash)| {
                    shared
                        .dst_manager
                        .register_mutable_block_with_hash(dst_block, *seq_hash)
                })
                .collect();

            let registration_timepoint = Instant::now();

            // Compute timing statistics from batch timing (O(1), not per-block)
            let unique_transfer_ids: std::collections::HashSet<_> =
                resolved.iter().map(|b| b.transfer_id).collect();

            let policy_ms = batch
                .timing
                .policy_duration()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let precondition_ms = batch
                .timing
                .precondition_duration()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let total_ms = batch
                .timing
                .total_duration()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            tracing::info!(
                blocks = resolved.len(),
                containers = unique_transfer_ids.len(),
                policy_ms,
                precondition_ms,
                xfer_ms = end_xfer.duration_since(start_xfer).as_millis() as u64,
                registration_ms =
                    registration_timepoint.duration_since(end_xfer).as_millis() as u64,
                total_ms,
                src = std::any::type_name::<Src>(),
                dst = std::any::type_name::<Dst>(),
                "Batch transfer complete"
            );

            // Send registered blocks to downstream pipeline if chaining is enabled
            if let Some(chain_tx) = &shared.chain_tx {
                #[allow(clippy::type_complexity)]
                let mut chain_outputs: std::collections::HashMap<
                    TransferId,
                    (
                        Arc<std::sync::Mutex<TransferState>>,
                        Vec<ImmutableBlock<Dst>>,
                    ),
                > = std::collections::HashMap::new();

                for (registered, resolved_block) in
                    registered_blocks.into_iter().zip(resolved.iter())
                {
                    chain_outputs
                        .entry(resolved_block.transfer_id)
                        .or_insert_with(|| (resolved_block.state.clone(), Vec::new()))
                        .1
                        .push(registered);
                }

                for (transfer_id, (state, blocks)) in chain_outputs {
                    let output = ChainOutput {
                        transfer_id,
                        blocks,
                        state,
                    };
                    if chain_tx.send(output).await.is_err() {
                        tracing::warn!(
                            %transfer_id,
                            "Chain channel closed, downstream pipeline unavailable"
                        );
                    } else {
                        tracing::debug!(
                            %transfer_id,
                            "Sent blocks to chain output for downstream processing"
                        );
                    }
                }
            }
        }

        // Mark transfer complete (batch-level, O(1))
        batch.timing.mark_transfer_complete();

        // Mark blocks as completed in each transfer state
        for (transfer_id, (state, block_ids)) in transfer_states {
            let mut state_guard = state.lock().unwrap();
            state_guard.mark_completed(block_ids);

            let total = state_guard.passed_blocks.len() + state_guard.filtered_out.len();
            let done = state_guard.completed.len() + state_guard.filtered_out.len();
            tracing::debug!(
                %transfer_id,
                total,
                done,
                passed = state_guard.passed_blocks.len(),
                filtered = state_guard.filtered_out.len(),
                completed = state_guard.completed.len(),
                "Transfer batch progress"
            );
            if done >= total && total > 0 {
                state_guard.set_complete();
            }
        }

        Ok(())
    }

    /// Legacy execute_batch method - kept for backwards compatibility
    /// This uses the sequential execution model.
    #[allow(dead_code)]
    async fn execute_batch(&self, batch: TransferBatch<Src>) -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // === UPGRADE STAGE ===
        // Resolve all source blocks - upgrade weak blocks, keep strong/external as-is
        // Blocks that fail upgrade (weak evicted) are counted as filtered
        let mut resolved: Vec<ResolvedBlock<Src>> = Vec::with_capacity(batch.len());
        let mut evicted_sequence_hashes: Vec<SequenceHash> = Vec::new();

        for queued in batch.blocks {
            match queued.source {
                SourceBlock::Strong(block) => {
                    // Strong: already have guard, ready for transfer
                    resolved.push(ResolvedBlock {
                        transfer_id: queued.transfer_id,
                        block_id: block.block_id(),
                        sequence_hash: queued.sequence_hash,
                        guard: Some(block),
                        state: queued.state,
                    });
                }
                SourceBlock::External(ext) => {
                    // External: caller holds reference, we just pass metadata through
                    // Guard is None - transfer uses block_id directly
                    resolved.push(ResolvedBlock {
                        transfer_id: queued.transfer_id,
                        block_id: ext.block_id,
                        sequence_hash: ext.sequence_hash,
                        guard: None,
                        state: queued.state,
                    });
                }
                SourceBlock::Weak(weak) => {
                    // Weak: upgrade at last moment
                    match weak.upgrade() {
                        Some(block) => {
                            resolved.push(ResolvedBlock {
                                transfer_id: queued.transfer_id,
                                block_id: block.block_id(),
                                sequence_hash: queued.sequence_hash,
                                guard: Some(block),
                                state: queued.state,
                            });
                        }
                        None => {
                            // Block was evicted - count as filtered
                            tracing::debug!(
                                sequence_hash = ?queued.sequence_hash,
                                "Weak block evicted before transfer"
                            );
                            evicted_sequence_hashes.push(queued.sequence_hash);
                        }
                    }
                }
            }
        }

        // If all blocks were evicted, nothing to transfer
        if resolved.is_empty() {
            tracing::debug!(
                evicted = evicted_sequence_hashes.len(),
                "All weak blocks evicted, skipping transfer"
            );
            return Ok(());
        }

        // Collect block_ids and sequence_hashes from resolved blocks
        let src_block_ids: Vec<BlockId> = resolved.iter().map(|b| b.block_id).collect();
        let sequence_hashes: Vec<SequenceHash> = resolved.iter().map(|b| b.sequence_hash).collect();

        // Collect states for completion tracking (group by transfer_id)
        let mut transfer_states: std::collections::HashMap<
            TransferId,
            (Arc<std::sync::Mutex<TransferState>>, Vec<BlockId>),
        > = std::collections::HashMap::new();
        for block in &resolved {
            transfer_states
                .entry(block.transfer_id)
                .or_insert_with(|| (block.state.clone(), Vec::new()))
                .1
                .push(block.block_id);
        }

        // Skip actual transfers when in test mode
        if !self.skip_transfers {
            // Allocate destination blocks
            let dst_blocks = self
                .dst_manager
                .allocate_blocks(resolved.len())
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to allocate {} destination blocks", resolved.len())
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

            // Register each transferred block in the destination tier
            // This converts MutableBlock -> ImmutableBlock using the sequence hash
            let registered_blocks: Vec<ImmutableBlock<Dst>> = dst_blocks
                .into_iter()
                .zip(sequence_hashes.iter())
                .map(|(dst_block, seq_hash)| {
                    self.dst_manager
                        .register_mutable_block_with_hash(dst_block, *seq_hash)
                })
                .collect();

            tracing::debug!(
                num_registered = registered_blocks.len(),
                "Registered transferred blocks in destination tier"
            );

            // Send registered blocks to downstream pipeline if chaining is enabled
            if let Some(chain_tx) = &self.chain_tx {
                // Group registered blocks by transfer_id for proper state tracking
                // We need to match registered blocks back to their original transfer contexts
                #[allow(clippy::type_complexity)]
                let mut chain_outputs: std::collections::HashMap<
                    TransferId,
                    (
                        Arc<std::sync::Mutex<TransferState>>,
                        Vec<ImmutableBlock<Dst>>,
                    ),
                > = std::collections::HashMap::new();

                // Match registered blocks with their transfer states
                // resolved and registered_blocks are in the same order
                for (registered, resolved_block) in
                    registered_blocks.into_iter().zip(resolved.iter())
                {
                    chain_outputs
                        .entry(resolved_block.transfer_id)
                        .or_insert_with(|| (resolved_block.state.clone(), Vec::new()))
                        .1
                        .push(registered);
                }

                // Send chain outputs for each transfer
                for (transfer_id, (state, blocks)) in chain_outputs {
                    let output = ChainOutput {
                        transfer_id,
                        blocks,
                        state,
                    };
                    if chain_tx.send(output).await.is_err() {
                        tracing::warn!(%transfer_id, "Chain channel closed, downstream pipeline unavailable");
                    } else {
                        tracing::debug!(%transfer_id, "Sent blocks to chain output for downstream processing");
                    }
                }
            }
        }

        // Mark blocks as completed in each transfer state
        for (transfer_id, (state, block_ids)) in transfer_states {
            let mut state_guard = state.lock().unwrap();
            state_guard.mark_completed(block_ids);

            // Check if transfer is complete (all passed blocks transferred)
            // passed_blocks.len() == filtered_out.len() + completed.len() means we're done
            let total = state_guard.passed_blocks.len() + state_guard.filtered_out.len();
            let done = state_guard.completed.len() + state_guard.filtered_out.len();
            tracing::debug!(
                %transfer_id,
                total,
                done,
                passed = state_guard.passed_blocks.len(),
                filtered = state_guard.filtered_out.len(),
                completed = state_guard.completed.len(),
                "Transfer batch progress"
            );
            if done >= total && total > 0 {
                state_guard.set_complete();
            }
        }

        Ok(())
    }
}

// ============================================================================
// Object Transfer Executor (for G4 / object storage destinations)
// ============================================================================

/// Object transfer executor stage for object storage destinations.
///
/// Executes transfers to object storage (G4) via `ObjectBlockOps::put_blocks()`.
/// Unlike `BlockTransferExecutor`, this does not require a destination `BlockManager`.
///
/// # Source Requirements
///
/// The source blocks must be `ImmutableBlock<Src>` (post-upgrade). The executor:
/// 1. Receives `ResolvedBlock<Src>` from the upgrade stage
/// 2. Extracts `SequenceHash` as the object key
/// 3. Calls `ObjectBlockOps::put_blocks()` with the source layout
///
/// # Lock Management
///
/// When a `lock_manager` is provided, after successful transfers:
/// 1. Creates `.meta` file to mark block as offloaded
/// 2. Releases `.lock` file to allow other instances to proceed
///
/// # No Destination Registration
///
/// Object storage is external - there's no local `BlockManager<G4>` to register with.
/// The object is simply stored at the key derived from `SequenceHash`.
pub struct ObjectTransferExecutor<Src: BlockMetadata> {
    /// Input channel from the batch/precondition stage
    input_rx: BatchOutputRx<Src>,
    /// Object storage operations
    object_ops: Arc<dyn ObjectBlockOps>,
    /// Source logical layout handle for reading block data
    /// The ObjectBlockOps implementation resolves this to a physical layout
    src_layout: LogicalLayoutHandle,
    /// Skip actual transfers (for testing)
    skip_transfers: bool,
    /// Maximum concurrent transfer batches
    max_concurrent_transfers: usize,
    /// Optional lock manager for creating meta files and releasing locks
    lock_manager: Option<Arc<dyn ObjectLockManager>>,
}

/// Shared state for ObjectTransferExecutor that can be cloned across concurrent tasks.
struct SharedObjectExecutorState {
    object_ops: Arc<dyn ObjectBlockOps>,
    src_layout: LogicalLayoutHandle,
    skip_transfers: bool,
    lock_manager: Option<Arc<dyn ObjectLockManager>>,
}

impl<Src: BlockMetadata> ObjectTransferExecutor<Src> {
    /// Create a new object transfer executor.
    #[allow(dead_code)]
    pub fn new(
        input_rx: BatchOutputRx<Src>,
        object_ops: Arc<dyn ObjectBlockOps>,
        src_layout: LogicalLayoutHandle,
        skip_transfers: bool,
        max_concurrent_transfers: usize,
        lock_manager: Option<Arc<dyn ObjectLockManager>>,
    ) -> Self {
        Self {
            input_rx,
            object_ops,
            src_layout,
            skip_transfers,
            max_concurrent_transfers,
            lock_manager,
        }
    }

    /// Run the executor loop.
    pub async fn run(mut self) {
        // N slots for active transfers
        let transfer_semaphore = Arc::new(Semaphore::new(self.max_concurrent_transfers));
        // 1 slot for preparation (upgrade) work - on-deck
        let prepare_semaphore = Arc::new(Semaphore::new(1));

        // Extract shared state for concurrent tasks
        let shared = Arc::new(SharedObjectExecutorState {
            object_ops: self.object_ops.clone(),
            src_layout: self.src_layout.clone(),
            skip_transfers: self.skip_transfers,
            lock_manager: self.lock_manager.clone(),
        });

        while let Some(batch) = self.input_rx.recv().await {
            if batch.is_empty() {
                continue;
            }

            // Wait for prepare slot (only 1 batch preparing at a time)
            let prepare_permit = prepare_semaphore.clone().acquire_owned().await;
            if prepare_permit.is_err() {
                break; // Semaphore closed
            }
            let prepare_permit = prepare_permit.unwrap();

            // Prepare stage: resolve/upgrade blocks (weak→strong)
            let upgraded = upgrade_batch(batch);

            // Done preparing, release prepare slot for next batch
            drop(prepare_permit);

            if upgraded.is_empty() {
                tracing::debug!("All blocks in batch evicted, skipping object transfer");
                continue;
            }

            // Now wait for transfer slot
            let transfer_permit = transfer_semaphore.clone().acquire_owned().await;
            if transfer_permit.is_err() {
                break; // Semaphore closed
            }
            let transfer_permit = transfer_permit.unwrap();

            // Spawn transfer task
            let shared_clone = shared.clone();
            tokio::spawn(async move {
                let _permit = transfer_permit; // Hold permit until task completes
                if let Err(e) = Self::execute_transfer(&shared_clone, upgraded).await {
                    tracing::warn!("ObjectTransferExecutor: transfer failed: {}", e);
                }
            });
        }

        // Wait for all in-flight transfers to complete by acquiring all permits
        let _ = transfer_semaphore
            .acquire_many(self.max_concurrent_transfers as u32)
            .await;
    }

    /// Execute the actual transfer for resolved blocks to object storage.
    async fn execute_transfer(
        shared: &SharedObjectExecutorState,
        mut batch: ResolvedBatch<Src>,
    ) -> anyhow::Result<()> {
        let _nvtx = nvtx_range!("offload::transfer");
        if batch.is_empty() {
            return Ok(());
        }

        let resolved = &batch.blocks;

        // Collect keys (sequence hashes) and block_ids from resolved blocks
        let keys: Vec<SequenceHash> = resolved.iter().map(|b| b.sequence_hash).collect();
        let block_ids: Vec<BlockId> = resolved.iter().map(|b| b.block_id).collect();

        // Collect states for completion tracking (group by transfer_id)
        let mut transfer_states: std::collections::HashMap<
            TransferId,
            (Arc<std::sync::Mutex<TransferState>>, Vec<BlockId>),
        > = std::collections::HashMap::new();
        for block in resolved {
            transfer_states
                .entry(block.transfer_id)
                .or_insert_with(|| (block.state.clone(), Vec::new()))
                .1
                .push(block.block_id);
        }

        // Track successfully transferred sequence hashes for lock management
        let mut successful_hashes: Vec<SequenceHash> = Vec::new();

        // Skip actual transfers when in test mode
        if !shared.skip_transfers {
            // Execute object put via ObjectBlockOps
            let results = shared
                .object_ops
                .put_blocks(keys.clone(), shared.src_layout.clone(), block_ids)
                .await;

            // Log results and track successful transfers
            let mut success_count = 0;
            let mut fail_count = 0;

            for result in results {
                match result {
                    Ok(hash) => {
                        success_count += 1;
                        successful_hashes.push(hash);
                    }
                    Err(hash) => {
                        fail_count += 1;
                        tracing::warn!(?hash, "Failed to transfer block to object storage");
                    }
                }
            }

            if fail_count > 0 {
                tracing::warn!(
                    success = success_count,
                    failed = fail_count,
                    "Object transfer partially failed"
                );
            } else {
                tracing::debug!(
                    num_blocks = success_count,
                    "Successfully transferred blocks to object storage"
                );
            }

            // todo: merge the else part of this conditional and perhaps add the event tap for the successful transfers
            // for block transfers we emit an event as part of registration; however, we don't register g4 blocks in the
            // same way; therefore, we need a new convention on how we inform the broader system of the object creation

            // Create meta files and release locks for successful transfers
            if let Some(lock_manager) = &shared.lock_manager {
                for hash in &successful_hashes {
                    // Create meta file to mark block as offloaded
                    if let Err(e) = lock_manager.create_meta(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to create meta file");
                    }

                    // Release lock
                    if let Err(e) = lock_manager.release_lock(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to release lock");
                    }
                }
                tracing::debug!(
                    num_blocks = successful_hashes.len(),
                    "Created meta files and released locks"
                );
            }
        } else {
            // In skip mode, still do lock management if configured
            if let Some(lock_manager) = &shared.lock_manager {
                for hash in &keys {
                    if let Err(e) = lock_manager.create_meta(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to create meta file");
                    }
                    if let Err(e) = lock_manager.release_lock(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to release lock");
                    }
                }
            }
        }

        // Mark transfer complete (batch-level, O(1))
        batch.timing.mark_transfer_complete();

        // Compute timing statistics from batch timing
        let unique_transfer_ids: std::collections::HashSet<_> =
            resolved.iter().map(|b| b.transfer_id).collect();

        let policy_ms = batch
            .timing
            .policy_duration()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let precondition_ms = batch
            .timing
            .precondition_duration()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let transfer_ms = batch
            .timing
            .transfer_duration()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let total_ms = batch
            .timing
            .total_duration()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        tracing::info!(
            blocks = resolved.len(),
            containers = unique_transfer_ids.len(),
            policy_ms,
            precondition_ms,
            transfer_ms,
            total_ms,
            src = std::any::type_name::<Src>(),
            dst = "G4-object",
            "Object batch transfer complete"
        );

        // Mark blocks as completed in each transfer state
        for (transfer_id, (state, block_ids)) in transfer_states {
            let mut state_guard = state.lock().unwrap();
            state_guard.mark_completed(block_ids);

            let total = state_guard.passed_blocks.len() + state_guard.filtered_out.len();
            let done = state_guard.completed.len() + state_guard.filtered_out.len();
            tracing::debug!(
                %transfer_id,
                total,
                done,
                passed = state_guard.passed_blocks.len(),
                filtered = state_guard.filtered_out.len(),
                completed = state_guard.completed.len(),
                "Object transfer batch progress"
            );
            if done >= total && total > 0 {
                state_guard.set_complete();
            }
        }

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
            .sweep_interval(Duration::from_millis(5))
            .build();

        assert_eq!(config.batch_config.max_batch_size, 32);
        assert_eq!(config.batch_config.min_batch_size, 8);
        assert_eq!(config.policy_timeout, Duration::from_millis(50));
        assert!(config.auto_chain);
        assert_eq!(config.sweep_interval, Duration::from_millis(5));
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::<(), ()>::default();
        assert!(config.policies.is_empty());
        assert!(!config.auto_chain);
        assert_eq!(config.sweep_interval, Duration::from_millis(10));
    }
}
