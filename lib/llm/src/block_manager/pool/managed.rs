// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Cache Block Pool Management
//!
//! This module provides the primary [`BlockPool`] structure for managing KV cache blocks.
//! It orchestrates the allocation, registration, and reuse of blocks by coordinating
//! between an [`ActiveBlockPool`] and an [`InactiveBlockPool`].
//!
//! ## Core Components:
//!
//! - **[`BlockPool`]**: The main entry point for interacting with the block management system.
//!   It holds the shared state containing both active and inactive pools.
//! - **[`ActiveBlockPool`]**: Manages blocks that are currently associated with active sequences.
//!   It primarily uses weak references to track these blocks, allowing them to be potentially
//!   reclaimed by the inactive pool if no strong references remain.
//! - **[`InactiveBlockPool`]**: Manages blocks that are not currently in active use. It supports
//!   block reuse by matching sequence hashes and employs a priority-based eviction strategy
//!   for acquiring free blocks.
//! - **[`BlockRegistry`]**: Manages the registration of blocks that have transitioned from the
//!   Complete to Registered state.
//! - **[`MutableBlock`]**: Represents a uniquely owned block, typically obtained from allocation.
//!   It allows modification and is returned to the inactive pool upon being dropped.
//! - **[`ImmutableBlock`]**: Represents a shared, immutable reference to a block, usually after
//!   it has been registered or matched. Ensures that multiple sequences can reference the
//!   same underlying block data.
//!
//! ## Workflow:
//!
//! 1.  Blocks are initially added to the [`BlockPool`] via [`BlockPool::add_blocks`], populating the
//!     [`InactiveBlockPool`].
//! 2.  Sequences request blocks via [`BlockPool::allocate_blocks`], which attempts to acquire them
//!     from the [`InactiveBlockPool`]. This returns [`MutableBlock`]s.
//! 3.  Once a [`MutableBlock`] is filled and ready, it's registered using [`BlockPool::register_block`].
//!     This process checks the both the [`ActiveBlockPool`] and the [`InactiveBlockPool`] for existing blocks
//!     with the same content hash. It returns an [`ImmutableBlock`] representing the canonical block
//!     (either the one provided or an existing one).
//! 4.  Sequences can also try to reuse blocks directly using [`BlockPool::match_sequence_hash`], which
//!     checks both the active and inactive pools.
//! 5.  When an [`ImmutableBlock`] is no longer needed by any sequence (its `Arc` count drops to zero),
//!     the underlying [`MutableBlock`] (if it still exists via the weak reference in the active pool)
//!     can eventually be returned to the [`InactiveBlockPool`] when its final strong reference (the `Arc`
//!     within `ImmutableBlock`) is dropped.
//! 6.  Dropped [`MutableBlock`]s are automatically returned to the [`InactiveBlockPool`].

use super::*;

pub mod active;
pub mod controller;
pub mod inactive;
pub mod priority_key;
pub mod state;

use active::ActiveBlockPool;
use inactive::InactiveBlockPool;

#[derive(Builder, Dissolve)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct ManagedBlockPoolArgs<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    #[builder(default = "NullEventManager::new()")]
    event_manager: Arc<dyn EventManager>,

    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,

    #[builder(default)]
    blocks: Vec<Block<S, L, M>>,

    #[builder(default)]
    global_registry: GlobalRegistry,

    #[builder(default = "Handle::current()")]
    async_runtime: Handle,

    #[builder(default = "BlockRegistrationDuplicationSetting::Disabled")]
    default_duplication_setting: BlockRegistrationDuplicationSetting,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ManagedBlockPoolArgsBuilder<S, L, M> {
    pub fn build(self) -> anyhow::Result<ManagedBlockPool<S, L, M>> {
        let args = self.build_internal()?;
        let (
            event_manager,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            default_duplication_setting,
        ) = args.dissolve();

        tracing::info!("building block pool");
        let pool = ManagedBlockPool::new(
            event_manager,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            default_duplication_setting,
        );

        Ok(pool)
    }
}

// Specific request type aliases for our use cases
type AllocateBlocksReq<S, L, M> = RequestResponse<usize, BlockPoolResult<MutableBlocks<S, L, M>>>;
type RegisterBlocksReq<S, L, M> = RequestResponse<
    (MutableBlocks<S, L, M>, BlockRegistrationDuplicationSetting),
    BlockPoolResult<ImmutableBlocks<S, L, M>>,
>;
type MatchHashesReq<S, L, M> =
    RequestResponse<Vec<SequenceHash>, BlockPoolResult<ImmutableBlocks<S, L, M>>>;
type TouchBlocksReq = RequestResponse<Vec<SequenceHash>, BlockPoolResult<()>>;
type AddBlocksReq<S, L, M> = RequestResponse<Vec<Block<S, L, M>>, ()>;
type ResetReq = RequestResponse<(), BlockPoolResult<()>>;
type ReturnBlockReq<S, L, M> = RequestResponse<Vec<Block<S, L, M>>, BlockPoolResult<()>>;
type StatusReq = RequestResponse<(), BlockPoolResult<BlockPoolStatus>>;
type ResetBlocksReq = RequestResponse<Vec<SequenceHash>, BlockPoolResult<ResetBlocksResponse>>;

// Update the request enums to use the cleaner types
pub enum PriorityRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    AllocateBlocks(AllocateBlocksReq<S, L, M>),
    RegisterBlocks(RegisterBlocksReq<S, L, M>),
    MatchSequenceHashes(MatchHashesReq<S, L, M>),
    TouchBlocks(TouchBlocksReq),
    Reset(ResetReq),
    ReturnBlock(ReturnBlockReq<S, L, M>),
}

pub enum ControlRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    AddBlocks(AddBlocksReq<S, L, M>),
    Status(StatusReq),
    ResetBlocks(ResetBlocksReq),
}

/// Manages the blocks in a specific storage backenda
pub struct ManagedBlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, L, M>>,
    ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, L, M>>,
    available_blocks_counter: Arc<AtomicU64>,
    total_blocks_counter: Arc<AtomicU64>,
    default_duplication_setting: BlockRegistrationDuplicationSetting,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Clone for ManagedBlockPool<S, L, M> {
    fn clone(&self) -> Self {
        Self {
            priority_tx: self.priority_tx.clone(),
            ctrl_tx: self.ctrl_tx.clone(),
            available_blocks_counter: self.available_blocks_counter.clone(),
            total_blocks_counter: self.total_blocks_counter.clone(),
            default_duplication_setting: self.default_duplication_setting,
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ManagedBlockPool<S, L, M> {
    pub fn builder() -> ManagedBlockPoolArgsBuilder<S, L, M> {
        ManagedBlockPoolArgsBuilder::default()
    }

    /// Creates a new [`ManagedBlockPool`] with the given [`EventManager`].
    ///
    /// The pool starts empty and requires blocks to be added via [`add_blocks`].
    ///
    /// # Arguments
    ///
    /// * `event_manager` - An [`Arc<dyn EventManager>`] used for publishing block registration/removal events.
    ///
    /// # Returns
    ///
    /// A new [`ManagedBlockPool`] instance.
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        cancel_token: CancellationToken,
        blocks: Vec<Block<S, L, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        default_duplication_setting: BlockRegistrationDuplicationSetting,
    ) -> Self {
        let (pool, progress_engine) = Self::with_progress_engine(
            event_manager,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            default_duplication_setting,
        );

        // pool.runtime.handle().spawn(async move {
        //     let mut progress_engine = progress_engine;
        //     tracing::debug!("starting progress engine");
        //     while progress_engine.step().await {
        //         tracing::trace!("progress engine step");
        //     }
        // });

        let thread_name = format!(
            "block-pool-{}-{}",
            short_type_name::<S>(),
            short_type_name::<L>()
        );

        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build Tokio runtime for block pool progress engine");

                runtime.block_on(async move {
                    let mut progress_engine = progress_engine;
                    tracing::debug!("starting progress engine");
                    while progress_engine.step().await {
                        tracing::trace!("progress engine step");
                    }
                });
            })
            .expect("Failed to spawn block pool progress engine thread");

        pool
    }

    fn with_progress_engine(
        event_manager: Arc<dyn EventManager>,
        cancel_token: CancellationToken,
        blocks: Vec<Block<S, L, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        default_duplication_setting: BlockRegistrationDuplicationSetting,
    ) -> (Self, ProgressEngine<S, L, M>) {
        let (priority_tx, priority_rx) = tokio::sync::mpsc::unbounded_channel();
        let (ctrl_tx, ctrl_rx) = tokio::sync::mpsc::unbounded_channel();

        let progress_engine = ProgressEngine::<S, L, M>::new(
            event_manager,
            priority_rx,
            ctrl_rx,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
        );

        let available_blocks_counter = progress_engine.available_blocks_counter.clone();
        let total_blocks_counter = progress_engine.total_blocks_counter.clone();

        (
            Self {
                priority_tx,
                ctrl_tx,
                available_blocks_counter,
                total_blocks_counter,
                default_duplication_setting,
            },
            progress_engine,
        )
    }

    pub fn default_duplication_setting(&self) -> BlockRegistrationDuplicationSetting {
        self.default_duplication_setting
    }

    fn _add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> AsyncResponse<()> {
        let (req, resp_rx) = AddBlocksReq::new(blocks);

        self.ctrl_tx
            .send(ControlRequest::AddBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _allocate_blocks(
        &self,
        count: usize,
    ) -> AsyncResponse<BlockPoolResult<Vec<MutableBlock<S, L, M>>>> {
        let (req, resp_rx) = AllocateBlocksReq::new(count);

        self.priority_tx
            .send(PriorityRequest::AllocateBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
        duplication_setting: BlockRegistrationDuplicationSetting,
    ) -> AsyncResponse<BlockPoolResult<ImmutableBlocks<S, L, M>>> {
        if blocks.is_empty() {
            return Err(BlockPoolError::NoBlocksToRegister);
        }

        let (req, resp_rx) = RegisterBlocksReq::new((blocks, duplication_setting));

        self.priority_tx
            .send(PriorityRequest::RegisterBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> AsyncResponse<BlockPoolResult<ImmutableBlocks<S, L, M>>> {
        let (req, resp_rx) = MatchHashesReq::new(sequence_hashes.into());

        self.priority_tx
            .send(PriorityRequest::MatchSequenceHashes(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _touch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> AsyncResponse<BlockPoolResult<()>> {
        let (req, resp_rx) = TouchBlocksReq::new(sequence_hashes.into());

        self.priority_tx
            .send(PriorityRequest::TouchBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _reset(&self) -> AsyncResponse<BlockPoolResult<()>> {
        let (req, resp_rx) = ResetReq::new(());

        self.priority_tx
            .send(PriorityRequest::Reset(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _try_return_block(&self, block: OwnedBlock<S, L, M>) -> AsyncResponse<BlockPoolResult<()>> {
        let raw_blocks = block
            .try_take_block(private::PrivateToken)
            .ok_or(BlockPoolError::NotReturnable)?;

        let (req, resp_rx) = ReturnBlockReq::new(raw_blocks);

        self.priority_tx
            .send(PriorityRequest::ReturnBlock(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }
}

#[async_trait]
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockPool<S, L, M>
    for ManagedBlockPool<S, L, M>
{
    /// Adds a vector of [`Block`]s to the [`InactiveBlockPool`].
    async fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    fn add_blocks_blocking(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    /// Attempts to allocate a specified number of free blocks from the [`InactiveBlockPool`].
    ///
    /// Blocks acquired this way are returned as [`MutableBlock`]s, granting unique ownership
    /// and allowing modification. Dropping a [`MutableBlock`] automatically returns it
    /// to the [`InactiveBlockPool`].
    async fn allocate_blocks(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        self._allocate_blocks(count)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn allocate_blocks_blocking(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        self._allocate_blocks(count)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    /// Registers a vector of [`MutableBlock`]s (presumably after filling them) with the pool,
    /// making them available for sharing via the [`ActiveBlockPool`].
    ///
    /// This function checks if any of the blocks have the same sequence hash as an existing block
    /// in the active pool. If so, it returns an [`ImmutableBlock`].
    ///
    /// Note: Depending on the [`BlockRegistrationDuplicationSetting`], the returned [`ImmutableBlock`] may
    /// not be the same block that was provided -- that is, it should hold the same content, but was the
    /// first block registered. If duplication is allowed, we will keep alive both the primary block and
    /// the duplicate block.
    async fn register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        self._register_blocks(blocks, self.default_duplication_setting)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn register_blocks_blocking(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        self._register_blocks(blocks, self.default_duplication_setting)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    /// Attempts to match the given [`SequenceHash`] to an existing block, checking
    /// both the active and inactive pools.
    ///
    /// Checks the [`ActiveBlockPool`] first. If a valid strong reference exists, it returns
    /// an [`ImmutableBlock`] cloned from it. If the weak reference exists but is stale,
    /// it's removed.
    ///
    /// If not found in the active pool, it checks the [`InactiveBlockPool`]. If found there,
    /// the block is moved to the active pool (tracked by a weak reference) and returned
    /// as a new [`ImmutableBlock`].
    async fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        self._match_sequence_hashes(sequence_hashes)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn match_sequence_hashes_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        tracing::debug!("find matching for sequence_hashes: {:?}", sequence_hashes);
        self._match_sequence_hashes(sequence_hashes)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn touch_blocks(&self, sequence_hashes: &[SequenceHash]) -> Result<(), BlockPoolError> {
        self._touch_blocks(sequence_hashes)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn touch_blocks_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<(), BlockPoolError> {
        self._touch_blocks(sequence_hashes)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn try_return_block(&self, block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        self._try_return_block(block)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn try_return_block_blocking(&self, block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        self._try_return_block(block)?
            .blocking_recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn total_blocks(&self) -> u64 {
        self.total_blocks_counter.load(Ordering::Relaxed)
    }

    fn available_blocks(&self) -> u64 {
        self.available_blocks_counter.load(Ordering::Relaxed)
    }
}

struct ProgressEngine<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, L, M>>,
    ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, L, M>>,
    cancel_token: CancellationToken,
    state: State<S, L, M>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    available_blocks_counter: Arc<AtomicU64>,
    total_blocks_counter: Arc<AtomicU64>,
}

pub struct State<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    active: ActiveBlockPool<S, L, M>,
    inactive: InactiveBlockPool<S, L, M>,
    registry: BlockRegistry,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
    event_manager: Arc<dyn EventManager>,
}

impl<S: Storage, L: LocalityProvider + 'static, M: BlockMetadata> ProgressEngine<S, L, M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        event_manager: Arc<dyn EventManager>,
        priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, L, M>>,
        ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, L, M>>,
        cancel_token: CancellationToken,
        blocks: Vec<Block<S, L, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
    ) -> Self {
        let (return_tx, return_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut state =
            State::<S, L, M>::new(event_manager, return_tx, global_registry, async_runtime);

        let count = blocks.len();

        tracing::debug!(count, "adding blocks to inactive pool");
        state.inactive.add_blocks(blocks);

        let available_blocks_counter = state.inactive.available_blocks_counter();
        let total_blocks_counter = state.inactive.total_blocks_counter();

        Self {
            priority_rx,
            ctrl_rx,
            cancel_token,
            state,
            return_rx,
            available_blocks_counter,
            total_blocks_counter,
        }
    }

    pub async fn step(&mut self) -> bool {
        tokio::select! {
            biased;

            Some(priority_req) = self.priority_rx.recv(), if !self.priority_rx.is_closed() => {
                self.state.handle_priority_request(priority_req, &mut self.return_rx).await;
            }

            Some(req) = self.ctrl_rx.recv(), if !self.ctrl_rx.is_closed() => {
                self.state.handle_control_request(req);
            }

            Some(block) = self.return_rx.recv() => {
                self.state.handle_return_block(block);
            }

            _ = self.cancel_token.cancelled() => {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use crate::block_manager::block::{BasicMetadata, Blocks};
    use crate::block_manager::layout::{FullyContiguous, LayoutConfig, tests::setup_layout};

    use crate::block_manager::locality::Local;
    use crate::tokens::{TokenBlockSequence, Tokens};

    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};

    use super::*;

    /// Helper method to build a [`ManagedBlockPool`] with a [`ProgressEngine`] for unit testing
    impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ManagedBlockPoolArgsBuilder<S, L, M> {
        #[allow(clippy::type_complexity)]
        fn build_with_progress_engine(
            self,
        ) -> anyhow::Result<(ManagedBlockPool<S, L, M>, ProgressEngine<S, L, M>)> {
            let args = self.build_internal()?;
            let (
                event_manager,
                cancel_token,
                blocks,
                global_registry,
                async_runtime,
                default_duplication_setting,
            ) = args.dissolve();

            let (pool, progress_engine) = ManagedBlockPool::with_progress_engine(
                event_manager,
                cancel_token,
                blocks,
                global_registry,
                async_runtime,
                default_duplication_setting,
            );

            Ok((pool, progress_engine))
        }
    }

    #[tokio::test]
    async fn test_block_pool_state() {
        let layout = setup_layout(None).unwrap();
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
            .unwrap()
            .into_blocks()
            .unwrap();

        let (_pool, mut progress) = ManagedBlockPool::builder()
            .blocks(blocks)
            .build_with_progress_engine()
            .unwrap();

        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let blocks = progress.state.allocate_blocks(1).unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        drop(blocks);
        progress.step().await;
        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let mut blocks = progress.state.allocate_blocks(1).unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        let mut block = blocks.pop().unwrap();

        block.init_sequence(1337).unwrap();
        block.add_token(1).unwrap();
        block.add_token(2).unwrap();
        block.add_token(3).unwrap();
        block.add_token(4).unwrap();

        assert!(block.add_token(5).is_err());
    }

    #[tokio::test]
    async fn test_block_pool() {
        let layout = setup_layout(None).unwrap();
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
            .unwrap()
            .into_blocks()
            .unwrap();

        let (pool, mut progress) = ManagedBlockPool::builder()
            .blocks(blocks)
            .build_with_progress_engine()
            .unwrap();

        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let pool_clone = pool.clone();
        let allocate_1_block =
            tokio::spawn(async move { pool_clone.allocate_blocks(1).await.unwrap() });
        progress.step().await;

        let blocks = allocate_1_block.await.unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        // drop the single block
        drop(blocks);

        // check before and after the progress engine step
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        progress.step().await;
        assert_eq!(progress.state.inactive.available_blocks(), 7);
    }

    #[test]
    fn test_block_pool_blocking() {
        const EXPECTED_SEQUENCE_HASH: u64 = 14643705804678351452;

        // Create a new layout
        let layout = setup_layout(None).unwrap();

        // Create the Blocks
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
            .unwrap()
            .into_blocks()
            .unwrap();

        let async_runtime = tokio::runtime::Runtime::new().unwrap();

        // Create the ManagedBlockPool and add the blocks
        let pool = ManagedBlockPool::builder()
            .blocks(blocks)
            .async_runtime(async_runtime.handle().clone())
            .build()
            .unwrap();

        // All blocks should be in the Reset/Empty state
        // No blocks should match the expected sequence hash
        let matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        assert_eq!(matched_blocks.len(), 0);

        // Allocate a single block from the pool
        let mut mutable_blocks = pool.allocate_blocks_blocking(1).unwrap();
        assert_eq!(mutable_blocks.len(), 1);
        let mut block = mutable_blocks.pop().unwrap();

        // Initialize the sequence on the block with a salt hash
        block.init_sequence(1337).unwrap();

        // Add some tokens to the block - our page_size is 4
        block.add_token(1).unwrap();
        block.add_token(2).unwrap();
        block.add_token(3).unwrap();
        block.add_token(4).unwrap();

        // Should fail because we don't have space in the block
        assert!(block.add_token(5).is_err());

        // Commit the block - this will generate a sequence hash
        // This will put the block in a Complete state
        block.commit().unwrap();
        assert!(block.state().is_complete()); // perhaps renamed to Commited

        let sequence_hash = block.sequence_hash().unwrap();
        assert_eq!(sequence_hash, EXPECTED_SEQUENCE_HASH);

        // Register the block
        // We provide a mutable block to the register_blocks function
        // This will take ownership of the block and return an immutable block
        let mut immutable_blocks = pool.register_blocks_blocking(vec![block]).unwrap();
        let block = immutable_blocks.pop().unwrap();
        assert!(block.state().is_registered());
        assert_eq!(block.sequence_hash(), sequence_hash);

        // Dropping the immutable block should return the block to the pool
        // However, the block should remain in the ManagedBlockPool as an inactive block until it is reused
        // or promoted back to an immutable block by being matched with a sequence hash
        drop(block);

        // Get the list of ImmutableBlocks that match the sequence hash
        let matched = pool
            .match_sequence_hashes_blocking(&[sequence_hash])
            .unwrap();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].sequence_hash(), sequence_hash);
    }

    async fn create_blocks<S: Storage, L: LocalityProvider, M: BlockMetadata>(
        pool: &ManagedBlockPool<S, L, M>,
        num_blocks: usize,
    ) -> anyhow::Result<(Vec<ImmutableBlock<S, L, M>>, Vec<SequenceHash>)> {
        let tokens = vec![0; num_blocks * 4];
        let token_blocks = TokenBlockSequence::new(Tokens::from(tokens), 4, None);
        assert_eq!(token_blocks.blocks().len(), num_blocks);

        let mut sequence_hashes = Vec::new();
        let mut mutable_blocks = Vec::new();

        for token_block in token_blocks.blocks().iter() {
            let mut block = pool.allocate_blocks(1).await?.pop().unwrap();
            block.apply_token_block(token_block.clone())?;

            sequence_hashes.push(block.sequence_hash().unwrap());
            mutable_blocks.push(block);
        }
        let immutable_blocks = pool.register_blocks(mutable_blocks).await?;

        Ok((immutable_blocks, sequence_hashes))
    }

    async fn make_simple_pool(
        num_blocks: usize,
    ) -> anyhow::Result<
        ManagedBlockPool<NullDeviceStorage, crate::block_manager::locality::Local, BasicMetadata>,
    > {
        let config = LayoutConfig {
            num_blocks,
            num_layers: 1,
            outer_dim: 1,
            page_size: 4,
            inner_dim: 1024,
            alignment: 1,
            dtype_width_bytes: 2,
        };

        let layout = FullyContiguous::<NullDeviceStorage>::allocate(config, &NullDeviceAllocator)?;

        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)?.into_blocks()?;

        let pool = ManagedBlockPool::builder().blocks(blocks).build()?;

        Ok(pool)
    }

    /// A test that ensures that we only ever evict leaves from the inactive pool.
    #[tokio::test]
    async fn test_block_pool_evict_leaves() -> anyhow::Result<()> {
        let pool = make_simple_pool(4).await?;

        let (_, sequence_hashes) = create_blocks(&pool, 4).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate 1 block. This should evict the leaf of our allocated sequence.
        pool.allocate_blocks(1).await?;

        // The leaf should be evicted, so we should have 3 matches.
        let matched = pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;
        assert_eq!(matched.len(), 3);
        drop(matched);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate 2 blocks. This should get the previously allocated block, as well as one more leaf.
        pool.allocate_blocks(2).await.unwrap();

        // The next leaf should be evicted, so we should have 2 matches.
        let matched = pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;
        assert_eq!(matched.len(), 2);

        drop(matched);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // If we allocate all the blocks, the entire remaining sequence should be evicted.
        let blocks = pool.allocate_blocks(4).await?;
        assert_eq!(blocks.len(), 4);

        Ok(())
    }

    /// When a block has two children, we need to ensure that we evict both children before
    /// adding the parent to the leaf set.
    #[tokio::test]
    async fn test_block_pool_parent_child() -> anyhow::Result<()> {
        let pool = make_simple_pool(3).await?;

        let tokens = vec![1, 2, 3, 4, 5];

        let sequence = TokenBlockSequence::new(Tokens::from(tokens.clone()), 4, None);

        // Create a root block, with two child blocks.
        let mut root_block = pool.allocate_blocks(1).await?.pop().unwrap();
        root_block.apply_token_block(sequence.blocks().first().unwrap().clone())?;

        let root_block_hash = root_block.sequence_hash().unwrap();

        let mut child_blocks = Vec::new();
        let mut child_block_hashes = Vec::new();

        for i in 0..2 {
            // Create a new token sequence using the common prefix.
            let mut tokens = tokens.clone();
            for _ in 0..4 {
                tokens.push(i);
            }
            let seq = TokenBlockSequence::new(Tokens::from(tokens), 4, None);

            // Allocate and apply the suffix to the child block.
            let mut child_block = pool.allocate_blocks(1).await?.pop().unwrap();
            child_block.apply_token_block(seq.blocks()[1].clone())?;

            child_block_hashes.push(child_block.sequence_hash().unwrap());
            child_blocks.push(child_block);
        }

        // Register the root block
        let root_block = pool.register_blocks(vec![root_block]).await?;

        // Register the children
        let child_blocks = pool.register_blocks(child_blocks).await?;

        // Drop both of them.
        drop(root_block);
        drop(child_blocks);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate two new blocks, which should evict both children.
        pool.allocate_blocks(2).await?;

        // Now, the root block should be the only block left.
        for child_block_hash in child_block_hashes {
            let matched = pool.match_sequence_hashes(&[child_block_hash]).await?;
            assert_eq!(matched.len(), 0);
        }

        // Check that the root block remains.
        let matched = pool.match_sequence_hashes(&[root_block_hash]).await?;
        assert_eq!(matched.len(), 1);

        Ok(())
    }

    /// Matching an entire sequence (moving it to the active pool), and returning it
    /// should not affect the parent-child relationships of the blocks.
    #[tokio::test]
    async fn test_block_pool_match_return() -> anyhow::Result<()> {
        let pool = make_simple_pool(4).await?;

        let (_, sequence_hashes) = create_blocks(&pool, 4).await?;

        // We match the root of the sequence (moving it to the active pool), then
        // immediately return it.
        assert_eq!(
            pool.match_sequence_hashes(vec![sequence_hashes[0]].as_slice())
                .await?
                .len(),
            1
        );

        let _alloc_blocks1 = pool.allocate_blocks(3).await?;

        // Allocating 3 blocks should evict all but the root of the sequence.
        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            1
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let _alloc_blocks2 = pool.allocate_blocks(1).await?;

        // Now, allocating one more block should evict the root.
        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            0
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_block_pool_touch() -> anyhow::Result<()> {
        let pool = make_simple_pool(4).await?;

        let (_, sequence_hashes) = create_blocks(&pool, 4).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let _block0 = pool.allocate_blocks(1).await?;

        // The leaf should be evicted.
        assert_eq!(
            pool.match_sequence_hashes(vec![sequence_hashes[3]].as_slice())
                .await?
                .len(),
            0
        );

        // Now, touch the new leaf.
        pool.touch_blocks(vec![sequence_hashes[2]].as_slice())
            .await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let _block1 = pool.allocate_blocks(1).await?;

        // Since we touched block 2, block 1 should have been evicted.
        assert_eq!(
            pool.match_sequence_hashes(vec![sequence_hashes[1]].as_slice())
                .await?
                .len(),
            0
        );

        pool.touch_blocks(vec![sequence_hashes[3]].as_slice())
            .await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        pool.allocate_blocks(1).await?;

        // Now block 0 was evicted, since it was the last to be touched.
        assert_eq!(
            pool.match_sequence_hashes(vec![sequence_hashes[0]].as_slice())
                .await?
                .len(),
            0
        );
        Ok(())
    }

    const EXPECTED_SEQUENCE_HASH: u64 = 14643705804678351452;

    fn create_block(
        pool: &ManagedBlockPool<NullDeviceStorage, Local, BasicMetadata>,
    ) -> ImmutableBlock<NullDeviceStorage, Local, BasicMetadata> {
        let count = pool.available_blocks();

        // Allocate a single block from the pool
        let mut mutable_blocks = pool.allocate_blocks_blocking(1).unwrap();
        assert_eq!(mutable_blocks.len(), 1);
        let mut block = mutable_blocks.pop().unwrap();

        assert_eq!(pool.available_blocks(), count - 1);

        // Initialize the sequence on the block with a salt hash
        block.init_sequence(1337).unwrap();

        // Add some tokens to the block - our page_size is 4
        block.add_token(1).unwrap();
        block.add_token(2).unwrap();
        block.add_token(3).unwrap();
        block.add_token(4).unwrap();

        // Should fail because we don't have space in the block
        assert!(block.add_token(5).is_err());

        // Commit the block - this will generate a sequence hash
        // This will put the block in a Complete state
        block.commit().unwrap();
        assert!(block.state().is_complete()); // perhaps renamed to Commited

        let sequence_hash = block.sequence_hash().unwrap();
        assert_eq!(sequence_hash, EXPECTED_SEQUENCE_HASH);

        // Register the block
        // We provide a mutable block to the register_blocks function
        // This will take ownership of the block and return an immutable block
        let mut immutable_blocks = pool.register_blocks_blocking(vec![block]).unwrap();
        let block = immutable_blocks.pop().unwrap();
        assert!(block.state().is_registered());
        assert_eq!(block.sequence_hash(), sequence_hash);

        block
    }

    #[test]
    fn test_block_registration_allow_duplicates() {
        // const EXPECTED_SEQUENCE_HASH: u64 = 14643705804678351452;

        // Create a new layout
        let layout = setup_layout(None).unwrap();

        // Create the Blocks
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
            .unwrap()
            .into_blocks()
            .unwrap();

        let count = blocks.len() as u64;

        let async_runtime = tokio::runtime::Runtime::new().unwrap();

        // Create the ManagedBlockPool and add the blocks
        let pool = ManagedBlockPool::builder()
            .blocks(blocks)
            .async_runtime(async_runtime.handle().clone())
            .default_duplication_setting(BlockRegistrationDuplicationSetting::Allowed)
            .build()
            .unwrap();

        assert_eq!(pool.total_blocks(), count);
        assert_eq!(pool.available_blocks(), count);
        assert_eq!(
            pool.default_duplication_setting(),
            BlockRegistrationDuplicationSetting::Allowed
        );

        // All blocks should be in the Reset/Empty state
        // No blocks should match the expected sequence hash
        let matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        assert_eq!(matched_blocks.len(), 0);

        let primary = create_block(&pool);
        let primary_id = primary.block_id();
        assert_eq!(pool.available_blocks(), count - 1);

        // Now allocate another and register it with the same sequence
        let duplicate = create_block(&pool);
        assert!(duplicate.is_duplicate());
        assert_ne!(duplicate.block_id(), primary_id);
        assert_eq!(pool.available_blocks(), count - 2);

        // Reset only succeeds if all the blocks have been returned to the pool
        let reset_result = pool.reset_blocking();
        assert!(reset_result.is_err());

        // we hold both the primary and the duplicate in the duplicate
        // since we hold the primary in the duplicate, we expect this to fail
        assert!(pool.try_return_block_blocking(primary.into()).is_err());
        assert_eq!(pool.available_blocks(), count - 2);

        assert!(pool.try_return_block_blocking(duplicate.into()).is_ok());
        assert_eq!(pool.available_blocks(), count);

        // we can still match the primary block because we have not reset the pool
        let mut matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        let primary = matched_blocks.pop().unwrap();
        assert!(pool.try_return_block_blocking(primary.into()).is_ok());
        assert_eq!(pool.available_blocks(), count);

        // we can still create a duplicate even if the block is inactive
        let duplicate = create_block(&pool);
        assert!(duplicate.is_duplicate());
        assert_ne!(duplicate.block_id(), primary_id);
        assert_eq!(pool.available_blocks(), count - 2);

        assert!(pool.try_return_block_blocking(duplicate.into()).is_ok());
        assert_eq!(pool.available_blocks(), count);

        // Reset the pool
        let reset_result = pool.reset_blocking();
        assert!(reset_result.is_ok());

        // Now we should not be able to match the primary block
        let matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        assert_eq!(matched_blocks.len(), 0);
    }

    #[test]
    fn test_block_registration_disable_duplicates() {
        const EXPECTED_SEQUENCE_HASH: u64 = 14643705804678351452;

        // Create a new layout
        let layout = setup_layout(None).unwrap();

        // Create the Blocks
        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
            .unwrap()
            .into_blocks()
            .unwrap();

        let count = blocks.len() as u64;

        let async_runtime = tokio::runtime::Runtime::new().unwrap();

        // Create the ManagedBlockPool and add the blocks
        let pool = ManagedBlockPoolArgsBuilder::default()
            .blocks(blocks)
            .async_runtime(async_runtime.handle().clone())
            .default_duplication_setting(BlockRegistrationDuplicationSetting::Disabled)
            .build()
            .unwrap();

        assert_eq!(pool.total_blocks(), count);
        assert_eq!(pool.available_blocks(), count);

        // All blocks should be in the Reset/Empty state
        // No blocks should match the expected sequence hash
        let matched_blocks = pool
            .match_sequence_hashes_blocking(&[EXPECTED_SEQUENCE_HASH])
            .unwrap();
        assert_eq!(matched_blocks.len(), 0);

        // allocate and register the primary block
        let primary = create_block(&pool);
        let primary_id = primary.block_id();
        assert_eq!(pool.available_blocks(), count - 1);

        // Now allocate another and register it with the same sequence
        let duplicate = create_block(&pool);
        assert_eq!(pool.available_blocks(), count - 1);
        assert_eq!(duplicate.block_id(), primary_id);
    }
}
