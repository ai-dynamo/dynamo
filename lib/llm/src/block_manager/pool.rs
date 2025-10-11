// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod managed;
pub use managed::ManagedBlockPool;

use derive_builder::Builder;
use derive_getters::Dissolve;
use serde::{Deserialize, Serialize};

pub use super::block::{ImmutableBlock, MutableBlock};

use super::block::{
    Block, BlockError, BlockMetadata, GlobalRegistry, MaybeReturnableBlock, nixl::short_type_name,
    private, registry::BlockRegistry,
};
use super::events::{EventManager, NullEventManager};
use super::storage::Storage;

use crate::block_manager::CacheLevel;
use crate::block_manager::block::locality::LocalityProvider;
use crate::tokens::{SequenceHash, TokenBlock};

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{Arc, Weak},
};
use tokio::runtime::Handle;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::Result;

// Type aliases to reduce complexity across the module
type BlockPoolResult<T> = Result<T, BlockPoolError>;
type AsyncResponse<T> = Result<oneshot::Receiver<T>, BlockPoolError>;

// Collection type aliases
pub type MutableBlocks<S, L, M> = Vec<MutableBlock<S, L, M>>;
pub type ImmutableBlocks<S, L, M> = Vec<ImmutableBlock<S, L, M>>;

/// Enum representing either a mutable or immutable block that can be returned to the pool
#[derive(Debug)]
pub enum OwnedBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    Mutable(MutableBlock<S, L, M>),
    Immutable(ImmutableBlock<S, L, M>),
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> MaybeReturnableBlock<S, L, M>
    for OwnedBlock<S, L, M>
{
    fn is_returnable(&self) -> bool {
        match self {
            OwnedBlock::Mutable(block) => block.is_returnable(),
            OwnedBlock::Immutable(block) => block.is_returnable(),
        }
    }

    fn try_take_block(self, token: private::PrivateToken) -> Option<Vec<Block<S, L, M>>> {
        match self {
            OwnedBlock::Mutable(block) => block.try_take_block(token),
            OwnedBlock::Immutable(block) => block.try_take_block(token),
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> From<MutableBlock<S, L, M>>
    for OwnedBlock<S, L, M>
{
    fn from(block: MutableBlock<S, L, M>) -> Self {
        OwnedBlock::Mutable(block)
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> From<ImmutableBlock<S, L, M>>
    for OwnedBlock<S, L, M>
{
    fn from(block: ImmutableBlock<S, L, M>) -> Self {
        OwnedBlock::Immutable(block)
    }
}

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

    #[error("Reset error: {0}")]
    ResetError(String),

    #[error("Block is not returnable")]
    NotReturnable,

    #[error("Unsupported cache level: {0:?}")]
    UnsupportedCacheLevel(CacheLevel),

    #[error("No blocks to register")]
    NoBlocksToRegister,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRegistrationDuplicationSetting {
    /// On registration, if duplication is allowed, blocks with duplicate hashes cannot be registered directly,
    /// but instead can be held live with a strong arc to the primary block. This maintains the lifetime of
    /// the duplicate block.
    Allowed,

    /// On registration, if duplication is disabled, blocks with duplicate hashes will be returned immediately
    /// to the inactive pool and the primary block, the one first registered, will be returned to the caller,
    /// replacing the duplicate block.
    ///
    /// Note: If block duplication is disabled, then the implementation must always respect the fact that the
    /// mutable block that was registered, may not be the same block returned by the registration function, and
    /// thus be able to update any references that wish to use the block after registration.
    Disabled,
}

/// Generic request-response pattern for background task communication
#[derive(Dissolve)]
pub struct RequestResponse<Req, Resp> {
    pub request: Req,
    pub response_tx: oneshot::Sender<Resp>,
}

impl<Req, Resp> RequestResponse<Req, Resp> {
    /// Create a new request-response pair
    pub fn new(request: Req) -> (Self, oneshot::Receiver<Resp>) {
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

#[async_trait]
pub trait BlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata>:
    BlockPoolController + AsyncBlockPoolController + Send + Sync
{
    /// Add a vector of [`Block`]s to the pool.
    ///
    /// These blocks are typically created from a [`super::block::Blocks`]
    /// and represent the initial set of available cache blocks.
    /// Blocks added this way are initially reset.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A [`Vec<Block<S, M>>`] to add to the inactive pool.
    async fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> BlockPoolResult<()>;

    /// Blocking version of [`BlockPool::add_blocks`].
    fn add_blocks_blocking(&self, blocks: Vec<Block<S, L, M>>) -> BlockPoolResult<()>;

    /// Allocate a specified number of free blocks from the pool.
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
    async fn allocate_blocks(&self, count: usize) -> BlockPoolResult<MutableBlocks<S, L, M>>;

    /// Blocking version of [`BlockPool::allocate_blocks`].
    fn allocate_blocks_blocking(&self, count: usize) -> BlockPoolResult<MutableBlocks<S, L, M>>;

    /// Register a vector of [`MutableBlock`]s with the pool.
    async fn register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>>;

    /// Blocking version of [`BlockPool::register_blocks`].
    fn register_blocks_blocking(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>>;

    /// Match a set of [`SequenceHash`]s to existing blocks in the pool.
    ///
    /// # Arguments
    ///
    /// * `sequence_hashes` - A [`Vec<SequenceHash>`] to match.
    ///
    /// # Returns
    ///
    /// An [`Option<ImmutableBlock<S, M>>`] containing the shared block if found, otherwise `None`.
    async fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>>;

    /// Blocking version of [`BlockPool::match_sequence_hashes`].
    fn match_sequence_hashes_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>>;

    /// Touch a set of blocks. Equivalent to registering and then immediately dropping.
    async fn touch_blocks(&self, sequence_hashes: &[SequenceHash]) -> BlockPoolResult<()>;

    /// Blocking version of [`BlockPool::touch_blocks`].
    fn touch_blocks_blocking(&self, sequence_hashes: &[SequenceHash]) -> BlockPoolResult<()>;

    /// Attempt to return a block to the pool. Blocks will naturally be returned to the pool when they are dropped
    /// and their reference count drops to 0; however, for testing and to synchronize the block returning to the
    /// pool, this function can be used.
    async fn try_return_block(&self, block: OwnedBlock<S, L, M>) -> BlockPoolResult<()>;

    /// Blocking version of [`BlockPool::try_return_block`].
    fn try_return_block_blocking(&self, block: OwnedBlock<S, L, M>) -> BlockPoolResult<()>;

    fn total_blocks(&self) -> u64;

    fn available_blocks(&self) -> u64;
}

/// State of the pool when queried.
///
/// Provides a snapshot of the pool's current state including:
/// - Active blocks currently in use
/// - Inactive blocks ordered by reuse priority
/// - Number of empty blocks
#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct BlockPoolStatus {
    /// Active blocks currently in use
    pub active_blocks: usize,

    /// Inactive blocks ordered by reuse priority
    /// Blocks at the front of the list are more likely to be reused
    pub inactive_blocks: usize,

    /// Number of empty blocks
    pub empty_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct ResetBlocksResponse {
    /// Blocks that were reset
    pub reset_blocks: Vec<SequenceHash>,

    /// Blocks that were not found in the pool
    pub not_found: Vec<SequenceHash>,

    /// Blocks that were not reset
    pub not_reset: Vec<SequenceHash>,
}

pub trait BlockPoolController: Send + Sync {
    /// Returns the [`BlockPoolStatus`] of the pool.
    fn status_blocking(&self) -> Result<BlockPoolStatus, BlockPoolError>;

    /// Resets the pool to its initial state.
    ///
    /// This function will error unless all blocks have returned to the inactive pool.
    fn reset_blocking(&self) -> Result<(), BlockPoolError>;

    /// Attempt to reset a set of blocks.
    fn reset_blocks_blocking(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError>;
}

#[async_trait::async_trait]
pub trait AsyncBlockPoolController: Send + Sync {
    /// Returns the [`BlockPoolStatus`] of the pool.
    async fn status(&self) -> Result<BlockPoolStatus, BlockPoolError>;

    /// Resets the pool to its initial state.
    ///
    /// This function will error unless all blocks have returned to the inactive pool.
    async fn reset(&self) -> Result<(), BlockPoolError>;

    /// Attempt to reset a set of blocks.
    async fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError>;
}
