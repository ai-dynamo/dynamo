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
};
use super::events::{EventManager, NullEventManager};
use super::storage::Storage;

use crate::tokens::{SequenceHash, TokenBlock};

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{Arc, Weak},
};
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
}

impl<S: Storage, M: BlockMetadata> BlockPoolArgsBuilder<S, M> {
    pub fn build(self) -> anyhow::Result<BlockPool<S, M>> {
        let args = self.build_internal()?;
        let (event_manager, cancel_token, blocks) = args.dissolve();

        tracing::info!("building block pool");
        let pool = BlockPool::new(event_manager, cancel_token, blocks);

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
    ) -> Self {
        let (pool, progress_engine) =
            Self::with_progress_engine(event_manager, cancel_token, blocks);

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
    ) -> (Self, ProgressEngine<S, M>) {
        let (priority_tx, priority_rx) = tokio::sync::mpsc::unbounded_channel();
        let (ctrl_tx, ctrl_rx) = tokio::sync::mpsc::unbounded_channel();

        let progress_engine =
            ProgressEngine::<S, M>::new(event_manager, priority_rx, ctrl_rx, cancel_token, blocks);

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
}

struct ProgressEngine<S: Storage, M: BlockMetadata> {
    priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
    ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
    cancel_token: CancellationToken,
    state: State<S, M>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
}

#[cfg(test)]
mod tests {
    use crate::block_manager::block::BlockExt;

    use super::super::block::{BasicMetadata, Blocks};
    use super::super::layout::tests::setup_layout;
    use super::*;

    /// Helper method to build a [`BlockPool`] with a [`ProgressEngine`] for unit testing
    impl<S: Storage, M: BlockMetadata> BlockPoolArgsBuilder<S, M> {
        fn build_with_progress_engine(
            self,
        ) -> anyhow::Result<(BlockPool<S, M>, ProgressEngine<S, M>)> {
            let args = self.build_internal()?;
            let (event_manager, cancel_token, blocks) = args.dissolve();
            let (pool, progress_engine) =
                BlockPool::with_progress_engine(event_manager, cancel_token, blocks);

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

        // Create the BlockPool and add the blocks
        let pool = BlockPool::builder().blocks(blocks).build().unwrap();

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
}
