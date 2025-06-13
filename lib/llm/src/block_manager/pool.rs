// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

mod active;
mod inactive;
mod priority_key;
mod state;

use active::ActiveBlockPool;
use derive_builder::Builder;
use derive_getters::Dissolve;
use inactive::InactiveBlockPool;
use priority_key::PriorityKey;

pub use super::block::{ImmutableBlock, MutableBlock};

use super::block::{
    nixl::short_type_name, registry::BlockRegistry, Block, BlockError, BlockMetadata,
    GlobalRegistry,
};
use super::events::{EventManager, NullEventManager};
use super::metrics::{BlockManagerMetrics, PoolMetrics};
use super::storage::Storage;

use crate::tokens::{SequenceHash, TokenBlock};

use prometheus::Registry;
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{Arc, Weak},
};
use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::Result;

#[derive(Debug, thiserror::Error)]
pub enum BlockPoolError {
    #[error("Block is not complete")]
    BlockNotComplete,

    #[error("Not enough blocks available, requested: {0}, available: {1}")]
    NotEnoughBlocksAvailable(usize, usize),

    #[error("Invalid MutableBlock: {0}")]
    InvalidMutableBlock(String),

    #[error("Failed to register block: {0}")]
    FailedToRegisterBlock(String),

    #[error("Progress engine shutdown")]
    ProgressEngineShutdown,

    #[error(transparent)]
    BlockError(#[from] BlockError),
}

#[derive(Builder, Dissolve)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct BlockPoolArgs<S: Storage, M: BlockMetadata> {
    #[builder(default = "NullEventManager::new()")]
    event_manager: Arc<dyn EventManager>,

    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,

    #[builder(default)]
    blocks: Vec<Block<S, M>>,

    #[builder(default)]
    global_registry: GlobalRegistry,

    #[builder(default = "Handle::current()")]
    async_runtime: Handle,

    #[builder(
        default = "BlockManagerMetrics::new(&Arc::new(Registry::new())).unwrap().pool(\"pool\")"
    )]
    pool_metrics: Arc<PoolMetrics>,
}

impl<S: Storage, M: BlockMetadata> BlockPoolArgsBuilder<S, M> {
    pub fn build(self) -> anyhow::Result<BlockPool<S, M>> {
        let args = self.build_internal()?;
        let (event_manager, cancel_token, blocks, global_registry, async_runtime, metrics) =
            args.dissolve();

        tracing::info!("building block pool");
        let pool = BlockPool::new(
            event_manager,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            metrics,
        );

        Ok(pool)
    }
}
/// Manages the blocks in a specific storage backenda
pub struct BlockPool<S: Storage, M: BlockMetadata> {
    priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, M>>,
    ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, M>>,
}

impl<S: Storage, M: BlockMetadata> Clone for BlockPool<S, M> {
    fn clone(&self) -> Self {
        Self {
            priority_tx: self.priority_tx.clone(),
            ctrl_tx: self.ctrl_tx.clone(),
        }
    }
}

#[derive(Dissolve)]
struct Unary<Req, Resp> {
    request: Req,
    response_tx: oneshot::Sender<Resp>,
}

impl<Req, Resp> Unary<Req, Resp> {
    fn make_request(request: Req) -> (Self, oneshot::Receiver<Resp>) {
        let (response_tx, response_rx) = oneshot::channel();
        (
            Self {
                request,
                response_tx,
            },
            response_rx,
        )
    }
}

type UnaryResponse<T> = Result<oneshot::Receiver<T>, BlockPoolError>;

type ImmutableBlocksResult<S, M> = Result<Vec<ImmutableBlock<S, M>>, BlockPoolError>;

pub type MutableBlocks<S, M> = Vec<MutableBlock<S, M>>;
pub type ImmutableBlocks<S, M> = Vec<ImmutableBlock<S, M>>;

enum PriorityRequest<S: Storage, M: BlockMetadata> {
    AllocateBlocks(Unary<usize, Result<Vec<MutableBlock<S, M>>, BlockPoolError>>),
    RegisterBlocks(Unary<MutableBlocks<S, M>, Result<ImmutableBlocks<S, M>, BlockPoolError>>),
    MatchSequenceHashes(Unary<Vec<SequenceHash>, Vec<ImmutableBlock<S, M>>>),
}

enum ControlRequest<S: Storage, M: BlockMetadata> {
    AddBlocks(Unary<Vec<Block<S, M>>, ()>),
}

impl<S: Storage, M: BlockMetadata> BlockPool<S, M> {
    pub fn builder() -> BlockPoolArgsBuilder<S, M> {
        BlockPoolArgsBuilder::default()
    }

    /// Creates a new [`BlockPool`] with the given [`EventManager`].
    ///
    /// The pool starts empty and requires blocks to be added via [`add_blocks`].
    ///
    /// # Arguments
    ///
    /// * `event_manager` - An [`Arc<dyn EventManager>`] used for publishing block registration/removal events.
    ///
    /// # Returns
    ///
    /// A new [`BlockPool`] instance.
    fn new(
        event_manager: Arc<dyn EventManager>,
        cancel_token: CancellationToken,
        blocks: Vec<Block<S, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        metrics: Arc<PoolMetrics>,
    ) -> Self {
        let (pool, progress_engine) = Self::with_progress_engine(
            event_manager,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            metrics,
        );

        // pool.runtime.handle().spawn(async move {
        //     let mut progress_engine = progress_engine;
        //     tracing::debug!("starting progress engine");
        //     while progress_engine.step().await {
        //         tracing::trace!("progress engine step");
        //     }
        // });

        let thread_name = format!("block-pool-{}", short_type_name::<S>());

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
        blocks: Vec<Block<S, M>>,
        global_registry: GlobalRegistry,
        async_runtime: Handle,
        metrics: Arc<PoolMetrics>,
    ) -> (Self, ProgressEngine<S, M>) {
        let (priority_tx, priority_rx) = tokio::sync::mpsc::unbounded_channel();
        let (ctrl_tx, ctrl_rx) = tokio::sync::mpsc::unbounded_channel();

        let progress_engine = ProgressEngine::<S, M>::new(
            event_manager,
            priority_rx,
            ctrl_rx,
            cancel_token,
            blocks,
            global_registry,
            async_runtime,
            metrics,
        );

        (
            Self {
                priority_tx,
                ctrl_tx,
            },
            progress_engine,
        )
    }

    /// Adds a vector of [`Block`]s to the [`InactiveBlockPool`].
    ///
    /// These blocks are typically created from a [`super::block::Blocks`]
    /// and represent the initial set of available cache blocks.
    /// Blocks added this way are initially reset.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A [`Vec<Block<S, M>>`] to add to the inactive pool.
    #[expect(dead_code)]
    pub(crate) async fn add_blocks(&self, blocks: Vec<Block<S, M>>) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    /// Blocking version of [`BlockPool::add_blocks`].
    pub(crate) fn add_blocks_blocking(
        &self,
        blocks: Vec<Block<S, M>>,
    ) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?
            .recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    fn _add_blocks(&self, blocks: Vec<Block<S, M>>) -> UnaryResponse<()> {
        let (req, resp_rx) = Unary::<_, ()>::make_request(blocks);

        self.ctrl_tx
            .send(ControlRequest::AddBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    /// Attempts to allocate a specified number of free blocks from the [`InactiveBlockPool`].
    ///
    /// Blocks acquired this way are returned as [`MutableBlock`]s, granting unique ownership
    /// and allowing modification. Dropping a [`MutableBlock`] automatically returns it
    /// to the [`InactiveBlockPool`].
    ///
    /// # Arguments
    ///
    /// * `count` - The number of blocks to allocate.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(Vec<MutableBlock<S, M>>)`: If successful, a vector of allocated mutable blocks.
    /// - `Err(BlockPoolError)`: If not enough blocks are available in the inactive pool.
    pub async fn allocate_blocks(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        self._allocate_blocks(count)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    /// Blocking version of [`BlockPool::allocate_blocks`].
    pub fn allocate_blocks_blocking(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        self._allocate_blocks(count)?
            .recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn _allocate_blocks(
        &self,
        count: usize,
    ) -> UnaryResponse<Result<Vec<MutableBlock<S, M>>, BlockPoolError>> {
        // Create the request
        let (req, resp_rx) =
            Unary::<_, Result<Vec<MutableBlock<S, M>>, BlockPoolError>>::make_request(count);

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::AllocateBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        Ok(resp_rx)
    }

    /// Registers a vector of [`MutableBlock`]s (presumably after filling them) with the pool,
    /// making them available for sharing via the [`ActiveBlockPool`].
    ///
    /// This function checks if any of the blocks have the same sequence hash as an existing block
    /// in the active pool. If so, it returns an [`ImmutableBlock`] pointing to the existing block,
    /// and the provided `block` is implicitly dropped (returned to the [`InactiveBlockPool`]).
    pub async fn register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, M>>,
    ) -> ImmutableBlocksResult<S, M> {
        self._register_blocks(blocks)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    /// Blocking version of [`BlockPool::register_blocks`].
    pub fn register_blocks_blocking(
        &self,
        blocks: Vec<MutableBlock<S, M>>,
    ) -> ImmutableBlocksResult<S, M> {
        self._register_blocks(blocks)?
            .recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn _register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, M>>,
    ) -> UnaryResponse<ImmutableBlocksResult<S, M>> {
        // Make the request
        let (req, resp_rx) = Unary::<_, ImmutableBlocksResult<S, M>>::make_request(blocks);

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::RegisterBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        Ok(resp_rx)
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
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The [`SequenceHash`] to look for.
    ///
    /// # Returns
    ///
    /// An [`Option<ImmutableBlock<S, M>>`] containing the shared block if found, otherwise `None`.
    pub async fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> ImmutableBlocksResult<S, M> {
        self._match_sequence_hashes(sequence_hashes)?
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    /// Blocking version of [`BlockPool::match_sequence_hashes`].
    pub fn match_sequence_hashes_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> ImmutableBlocksResult<S, M> {
        self._match_sequence_hashes(sequence_hashes)?
            .recv()
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)
    }

    fn _match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> UnaryResponse<Vec<ImmutableBlock<S, M>>> {
        // Create the request
        let (req, resp_rx) =
            Unary::<_, Vec<ImmutableBlock<S, M>>>::make_request(sequence_hashes.into());

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::MatchSequenceHashes(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        Ok(resp_rx)
    }
}

struct State<S: Storage, M: BlockMetadata> {
    active: ActiveBlockPool<S, M>,
    inactive: InactiveBlockPool<S, M>,
    registry: BlockRegistry,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
    event_manager: Arc<dyn EventManager>,
    metrics: Arc<PoolMetrics>,
}

struct ProgressEngine<S: Storage, M: BlockMetadata> {
    priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
    ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
    cancel_token: CancellationToken,
    state: State<S, M>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
    metrics: Arc<PoolMetrics>,
}

#[cfg(test)]
mod tests {
    use super::super::block::{BasicMetadata, Blocks};
    use super::super::layout::{tests::setup_layout, FullyContiguous, LayoutConfig};
    use super::*;

    use crate::block_manager::block::BlockExt;
    use crate::block_manager::DType;
    use crate::tokens::{TokenBlockSequence, Tokens};

    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};

    /// Helper method to build a [`BlockPool`] with a [`ProgressEngine`] for unit testing
    impl<S: Storage, M: BlockMetadata> BlockPoolArgsBuilder<S, M> {
        fn build_with_progress_engine(
            self,
        ) -> anyhow::Result<(BlockPool<S, M>, ProgressEngine<S, M>)> {
            let args = self.build_internal()?;
            let (event_manager, cancel_token, blocks, global_registry, async_runtime, metrics) =
                args.dissolve();
            let (pool, progress_engine) = BlockPool::with_progress_engine(
                event_manager,
                cancel_token,
                blocks,
                global_registry,
                async_runtime,
                metrics,
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

        let (_pool, mut progress) = BlockPool::builder()
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

        let (pool, mut progress) = BlockPool::builder()
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

        // Create the BlockPool and add the blocks
        let pool = BlockPool::builder()
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
        assert_eq!(block.sequence_hash().unwrap(), sequence_hash);

        // Dropping the immutable block should return the block to the pool
        // However, the block should remain in the BlockPool as an inactive block until it is reused
        // or promoted back to an immutable block by being matched with a sequence hash
        drop(block);

        // Get the list of ImmutableBlocks that match the sequence hash
        let matched = pool
            .match_sequence_hashes_blocking(&[sequence_hash])
            .unwrap();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].sequence_hash().unwrap(), sequence_hash);
    }

    async fn create_blocks<S: Storage, M: BlockMetadata>(
        pool: &BlockPool<S, M>,
        num_blocks: usize,
    ) -> anyhow::Result<(Vec<ImmutableBlock<S, M>>, Vec<SequenceHash>)> {
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
    ) -> anyhow::Result<BlockPool<NullDeviceStorage, BasicMetadata>> {
        let config = LayoutConfig {
            num_blocks,
            num_layers: 1,
            outer_dim: 1,
            page_size: 4,
            inner_dim: 1024,
            alignment: 1,
            dtype: DType::FP16,
        };

        let layout = FullyContiguous::<NullDeviceStorage>::allocate(config, &NullDeviceAllocator)?;

        let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)?.into_blocks()?;

        let pool = BlockPool::builder().blocks(blocks).build()?;

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

        // Register the children first. This can happen with offloading.
        let child_blocks = pool.register_blocks(child_blocks).await?;

        // After the children are registered, we can register the root block.
        let root_block = pool.register_blocks(vec![root_block]).await?;

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

    /// When offloading, it's possible that the tail of a sequence in a pool is evicted before
    /// the entire sequence can be offloaded. This can happen in the following case:
    ///
    /// Assume a sequence of 4 blocks: [0, 1, 2, 3]
    /// 1. Blocks 0, 1, and 2 are offloaded to host memory.
    /// 2. Block 2 is evicted from the host.
    /// 3. Block 3 is offloaded to host memory.
    /// Now, the contents of the cache are [0, 1] and [3].
    /// We need to treat these as two separate sequences.
    #[tokio::test]
    async fn test_block_pool_fragmentation() -> anyhow::Result<()> {
        let pool = make_simple_pool(4).await?;

        let tokens = vec![0; 16];

        let token_blocks = TokenBlockSequence::new(Tokens::from(tokens), 4, None);
        assert_eq!(token_blocks.blocks().len(), 4);

        let mut sequence_hashes = Vec::new();

        // Allocate and register the first 3 blocks.
        for block in token_blocks.blocks()[..3].iter() {
            let mut mutable_block = pool.allocate_blocks(1).await?.pop().unwrap();
            mutable_block.apply_token_block(block.clone())?;

            sequence_hashes.push(mutable_block.sequence_hash()?);
            let _ = pool.register_blocks(vec![mutable_block]).await?;
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate 2 blocks. This should take the remaining uninitialized block as well as the
        // tail of the currently registered sequence.
        let _ = pool.allocate_blocks(2).await?;

        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            2
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate 1 more block for the leaf of the sequence.
        let mut mutable_block = pool.allocate_blocks(1).await?.into_iter().next().unwrap();

        mutable_block.apply_token_block(token_blocks.blocks()[3].clone())?;

        let _ = pool.register_blocks(vec![mutable_block]).await?;

        // We should still only match the first 2 blocks, since the 3rd block has been evicted.
        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            2
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Now, we should be able to allocate all 4 blocks.
        let _ = pool.allocate_blocks(4).await?;

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

    /// When we move a suffix of a sequence to the active pool (like what happens when onboarding),
    /// then return it to the inactive pool, we need to ensure that the parent-child relationships
    /// are still correct, and that the temporary leaf in the inactive pool can't be evicted.
    #[tokio::test]
    async fn test_block_pool_match_partial() -> anyhow::Result<()> {
        let pool = make_simple_pool(4).await?;

        let (_, sequence_hashes) = create_blocks(&pool, 4).await?;

        // Assert that all 4 blocks are in the pool.
        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            4
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Now, we match only the last 2 blocks
        let matched_suffix = pool.match_sequence_hashes(&sequence_hashes[2..]).await?;
        assert_eq!(matched_suffix.len(), 2);

        // This allocation should fail. Although there are 2 inactive blocks, the leaf is in the active pool.
        let new_alloc_block = pool.allocate_blocks(1).await?;
        assert_eq!(new_alloc_block.len(), 0);

        // Now, drop the leaf, and return it to the inactive pool.
        drop(matched_suffix);

        // All 4 blocks should still be in the pool.
        assert_eq!(
            pool.match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            4
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        Ok(())
    }
}
