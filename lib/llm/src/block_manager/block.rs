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

pub mod factory;
pub mod locality;

pub mod data;
pub mod registry;
pub mod state;
pub mod transfer;

pub mod collections;

pub use data::{view, BlockData, BlockDataExt, BlockDataProvider, BlockDataProviderMut};
pub use locality::LocalityProvider;

pub use crate::tokens::TokenBlockError;
pub use anyhow::Result;

pub use registry::{GlobalRegistry, RegistrationHandle};
pub use state::{BlockState, BlockStateInvalid};

use crate::block_manager::{
    state::KvBlockManagerState as BlockManager,
    storage::{Local, Remote, Storage, StorageTypeProvider},
};
use crate::tokens::{SaltHash, SequenceHash, Token, TokenBlock, Tokens};

// TODO: Define these traits in transfer_v2 if needed
// use transfer::{Immutable, Mutable, Readable, Writable};

use super::{
    events::PublishHandle,
    layout::{BlockLayout, LayoutError, LayoutType},
    storage::StorageType,
    WorkerID,
};

use derive_getters::Getters;
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use thiserror::Error;

pub mod private {
    pub struct PrivateToken;
}

/// A unique identifier for a block
pub type BlockId = usize;

/// A unique identifier for a block set
pub type BlockSetId = usize;

/// Result type for Block operations
pub type BlockResult<T> = std::result::Result<T, BlockError>;

/// Errors specific to block storage operations
#[derive(Debug, Error)]
pub enum BlockError {
    #[error(transparent)]
    Layout(#[from] LayoutError),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid block ID: {0}")]
    InvalidBlockID(BlockId),

    #[error("Misconfigured block data parallelism: {0}")]
    MisconfiguredBlockDataParallelism(String),

    #[error("Incompatible storage type: {0}")]
    IncompatibleStorageType(String),

    #[error("Views are not available on logical blocks")]
    ViewsNotAvailableOnLogicalBlocks,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub trait BlockMetadata: Default + std::fmt::Debug + Clone + Ord + Send + Sync + 'static {
    /// Called when the block is acquired from the pool
    fn on_acquired(&mut self, tick: u64);

    /// Called when the block is returned to the pool
    fn on_returned(&mut self, tick: u64);

    /// Resets the metadata to the default value
    /// If called, the [BlockMetadata::is_reset()] should return true
    fn reset_metadata(&mut self);

    /// The offload priority of the block. Higher priority blocks are offloaded first.
    /// If the block should not be offloaded, return None.
    fn offload_priority(&self) -> Option<u64>;
}

/// Marker trait for types that are mutable blocks
pub trait WritableBlock: BlockDataProviderMut {}

/// Marker trait for types that are immutable blocks
pub trait ReadableBlock: BlockDataProvider {}

pub trait ReadableBlocks {}

impl<T: ReadableBlock> ReadableBlocks for Vec<T> {}
impl<T: ReadableBlock> ReadableBlocks for [T] {}
impl<T: ReadableBlock> ReadableBlocks for &[T] {}

pub trait WritableBlocks {}

impl<T: WritableBlock> WritableBlocks for Vec<T> {}
impl<T: WritableBlock> WritableBlocks for [T] {}
impl<T: WritableBlock> WritableBlocks for &[T] {}

/// Blanket trait for anything that can be viewed as a slice of blocks
pub trait AsBlockSlice<'a, B: 'a> {
    fn as_block_slice(&'a self) -> &'a [B];
}

/// Blanket trait for anything that can be viewed as a mutable slice of blocks
pub trait AsBlockMutSlice<'a, B: 'a> {
    fn as_block_mut_slice(&'a mut self) -> &'a mut [B];
}

/// Blanket trait for anything that can be converted into a mutable block
pub trait IntoWritableBlocks<Locality: LocalityProvider, M: BlockMetadata> {
    type Output: WritableBlocks;
    fn into_writable_blocks(self, manager: &BlockManager<Locality, M>)
        -> BlockResult<Self::Output>;
}

impl<T: WritableBlocks, Locality: LocalityProvider, M: BlockMetadata>
    IntoWritableBlocks<Locality, M> for T
{
    type Output = T;
    fn into_writable_blocks(
        self,
        _manager: &BlockManager<Locality, M>,
    ) -> BlockResult<Self::Output> {
        Ok(self)
    }
}

pub trait IntoReadableBlocks<Locality: LocalityProvider, M: BlockMetadata> {
    type Output: ReadableBlocks;
    fn into_readable_blocks(self, manager: &BlockManager<Locality, M>)
        -> BlockResult<Self::Output>;
}

impl<T: ReadableBlocks, Locality: LocalityProvider, M: BlockMetadata>
    IntoReadableBlocks<Locality, M> for T
{
    type Output = T;
    fn into_readable_blocks(
        self,
        _manager: &BlockManager<Locality, M>,
    ) -> BlockResult<Self::Output> {
        Ok(self)
    }
}

/// A block with storage and associated metadata/state
#[derive(Debug)]
pub struct Block<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    data: L::BlockData<S>,
    metadata: M,
    state: BlockState,
    manager: Option<Arc<BlockManager<L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Block<S, L, M> {
    /// Create a new block with default metadata/state
    pub fn new(data: L::BlockData<S>, metadata: M) -> BlockResult<Self> {
        Ok(Self {
            data,
            metadata,
            state: BlockState::Reset,
            manager: None,
        })
    }

    pub fn sequence_hash(&self) -> Result<SequenceHash, BlockError> {
        match self.state() {
            BlockState::Complete(state) => Ok(state.token_block().sequence_hash()),
            BlockState::Registered(state, _) => Ok(state.sequence_hash()),
            _ => Err(BlockError::InvalidState(
                "Block is not complete nor registered.".to_string(),
            )),
        }
    }

    pub fn parent_sequence_hash(&self) -> Result<Option<SequenceHash>, BlockError> {
        match self.state() {
            BlockState::Complete(state) => Ok(state.token_block().parent_sequence_hash()),
            BlockState::Registered(state, _) => Ok(state.parent_sequence_hash()),
            _ => Err(BlockError::InvalidState(
                "Block is not complete nor registered.".to_string(),
            )),
        }
    }

    /// Reset the state of the block (public method replacing old crate-only version)
    pub fn reset(&mut self) {
        self.state = BlockState::Reset;
        self.metadata.reset_metadata();
    }

    /// Initialize a sequence on the block using a [SaltHash]
    ///
    /// The block must be in the [BlockState::Reset] state.
    ///
    /// After initialization, the block will be in the [BlockState::Partial] state.
    pub fn init_sequence(&mut self, salt_hash: SaltHash) -> Result<()> {
        Ok(self
            .state
            .initialize_sequence(self.page_size(), salt_hash)?)
    }

    /// Appends a single token to the block if it is in the Partial state and not full.
    /// Returns `Err` if the block is not Partial or already full.
    pub fn add_token(&mut self, token: Token) -> Result<()> {
        self.state.add_token(token)
    }

    /// Appends multiple tokens to the block if it is in the Partial state
    /// and has enough remaining capacity for *all* provided tokens.
    /// The block must be in the [BlockState::Partial] state.
    /// Returns `Err` if the block is not Partial or if there isn't enough space.
    pub fn add_tokens(&mut self, tokens: Tokens) -> Result<Tokens> {
        self.state.add_tokens(tokens)
    }

    /// Removes the last token from the block.
    /// Requires the block to be in the Partial state and not empty.
    /// Returns `Err` otherwise.
    pub fn pop_token(&mut self) -> Result<()> {
        self.state.pop_token()
    }

    /// Removes the last `count` tokens from the block.
    /// Requires the block to be in the Partial state and have at least `count` tokens.
    /// Returns `Err` otherwise.
    pub fn pop_tokens(&mut self, count: usize) -> Result<()> {
        self.state.pop_tokens(count)
    }

    /// Commit the block
    /// Requires the block to be in the [BlockState::Partial] state and completely full.
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    pub fn commit(&mut self) -> Result<()> {
        self.state.commit()
    }

    /// Apply a [TokenBlock] to the block
    /// Requires the block to be in the [BlockState::Reset] state.
    ///
    /// Additionally, the [TokenBlock] must match the [BlockLayout::page_size()]
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    pub fn apply_token_block(&mut self, token_block: TokenBlock) -> Result<()> {
        if self.page_size() != token_block.tokens().len() {
            return Err(BlockStateInvalid(format!(
                "TokenBlock size ({}) does not match Block page size ({})",
                token_block.tokens().len(),
                self.page_size()
            ))
            .into());
        }
        self.state.apply_token_block(token_block)
    }

    /// Returns the number of tokens currently in the block.
    pub fn len(&self) -> usize {
        match self.state.len() {
            Some(len) => len,
            None => self.page_size(),
        }
    }

    /// Returns the number of additional tokens that can be added (only valid for Partial state).
    pub fn remaining(&self) -> usize {
        self.state.remaining()
    }

    /// Returns true if the block contains no tokens (only true for Reset or empty Partial state).
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// Returns true if the block is full.
    pub fn is_full(&self) -> bool {
        self.len() == self.page_size()
    }

    /// Returns a list of tokens in the block.
    pub fn tokens(&self) -> Option<&Tokens> {
        self.state.tokens()
    }

    pub(crate) fn set_manager(&mut self, manager: Arc<BlockManager<L, M>>) {
        self.manager = Some(manager);
    }

    pub(crate) fn manager(&self) -> Option<&Arc<BlockManager<L, M>>> {
        self.manager.as_ref()
    }

    /// Get the metadata of the block
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Update the metadata of the block
    pub fn update_metadata(&mut self, metadata: M) {
        self.metadata = metadata;
    }

    /// Update the state of the block
    #[allow(dead_code)]
    pub(crate) fn update_state(&mut self, state: BlockState) {
        self.state = state;
    }

    /// Get a reference to the state of the block
    pub fn state(&self) -> &BlockState {
        &self.state
    }

    /// Get a mutable reference to the state of the block
    pub fn state_mut(&mut self) -> &mut BlockState {
        &mut self.state
    }

    /// Get the number of blocks in the block
    /// todo(ryan): validate this can be removed
    pub fn num_blocks(&self) -> usize {
        1
    }

    /// Get the block ID of the block
    pub fn block_id(&self) -> BlockId {
        self.data.block_id()
    }

    /// Get the number of layers in the block
    pub fn num_layers(&self) -> usize {
        self.data.num_layers()
    }

    /// Get the size of each block in the block
    pub fn page_size(&self) -> usize {
        self.data.page_size()
    }

    /// Get the inner dimension of the block
    pub fn inner_dim(&self) -> usize {
        self.data.num_inner_dims()
    }

    /// Get the number of outer dimensions in this block
    /// Works for all localities through BlockLayoutConfig
    pub fn num_outer_dims(&self) -> usize {
        self.data.num_outer_dims()
    }

    pub(crate) fn metadata_on_acquired(&mut self, tick: u64) {
        self.metadata.on_acquired(tick);
    }

    pub(crate) fn metadata_on_returned(&mut self, tick: u64) {
        self.metadata.on_returned(tick);
    }
}

pub(crate) trait PrivateBlockExt {
    fn register(
        &mut self,
        registry: &mut registry::BlockRegistry,
    ) -> Result<Option<PublishHandle>, registry::BlockRegistrationError>;
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> PrivateBlockExt for Block<S, L, M> {
    fn register(
        &mut self,
        registry: &mut registry::BlockRegistry,
    ) -> Result<Option<PublishHandle>, registry::BlockRegistrationError> {
        registry.register_block(&mut self.state)
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Local for Block<S, L, M> {}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> StorageTypeProvider for Block<S, L, M> {
    type StorageType = S;
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockDataProvider for Block<S, L, M> {
    fn block_data(&self) -> &impl BlockDataExt<S> {
        &self.data
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockDataProviderMut for Block<S, L, M> {
    fn block_data_mut(&mut self) -> &mut impl BlockDataExt<S> {
        &mut self.data
    }
}

pub trait BlockExt {
    /// Reset the state of the block
    fn reset(&mut self);

    /// Initialize a sequence on the block using a [SaltHash]
    ///
    /// The block must be in the [BlockState::Reset] state.
    ///
    /// After initialization, the block will be in the [BlockState::Partial] state.
    fn init_sequence(&mut self, salt_hash: SaltHash) -> Result<()>;

    /// Appends a single token to the block if it is in the Partial state and not full.
    /// Returns `Err` if the block is not Partial or already full.
    fn add_token(&mut self, token: Token) -> Result<()>;

    /// Appends multiple tokens to the block if it is in the Partial state
    /// and has enough remaining capacity for *all* provided tokens.
    /// The block must be in the [BlockState::Partial] state.
    /// Returns `Err` if the block is not Partial or if there isn't enough space.
    fn add_tokens(&mut self, tokens: Tokens) -> Result<Tokens>;

    /// Removes the last token from the block.
    /// Requires the block to be in the Partial state and not empty.
    /// Returns `Err` otherwise.
    fn pop_token(&mut self) -> Result<()>;

    /// Removes the last `count` tokens from the block.
    /// Requires the block to be in the Partial state and have at least `count` tokens.
    /// Returns `Err` otherwise.
    fn pop_tokens(&mut self, count: usize) -> Result<()>;

    /// Commit the block
    /// Requires the block to be in the [BlockState::Partial] state and completely full.
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    fn commit(&mut self) -> Result<()>;

    /// Apply a [TokenBlock] to the block
    /// Requires the block to be in the [BlockState::Reset] state.
    ///
    /// Additionally, the [TokenBlock] must match the [BlockLayout::page_size()]
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    fn apply_token_block(&mut self, token_block: TokenBlock) -> Result<()>;

    /// Returns the number of tokens currently in the block.
    fn len(&self) -> usize;

    /// Returns the number of additional tokens that can be added (only valid for Partial state).
    fn remaining(&self) -> usize;

    /// Returns true if the block contains no tokens (only true for Reset or empty Partial state).
    fn is_empty(&self) -> bool;

    /// Returns true if the block is full.
    fn is_full(&self) -> bool;

    /// Returns a list of tokens in the block.
    fn tokens(&self) -> Option<&Tokens>;
}

// impl<S: Storage, M: BlockMetadata> BlockExt for Block<S, M> {
//     fn reset(&mut self) {
//         Block::reset(self);
//     }

//     fn init_sequence(&mut self, salt_hash: SaltHash) -> Result<()> {
//         Ok(self
//             .state
//             .initialize_sequence(self.page_size(), salt_hash)?)
//     }

//     fn add_token(&mut self, token: Token) -> Result<()> {
//         self.state.add_token(token)
//     }

//     /// Get a block view for reading (Local blocks only)
//     pub fn block_view(&self) -> BlockResult<view::BlockView<S>> {
//         self.data.block_view()
//     }

//     /// Get a block view for writing (Local blocks only)
//     pub fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>> {
//         self.data.block_view_mut()
//     }
// }

// BlockIdentifier trait removed - use Block methods directly

// pub(crate) trait PrivateBlockExt {
//     fn register(
//         &mut self,
//         registry: &mut registry::BlockRegistry,
//     ) -> Result<Option<PublishHandle>, registry::BlockRegistationError>;
// }

// impl<S: Storage, L: LocalityProvider, M: BlockMetadata> PrivateBlockExt for Block<S, L, M> {
//     fn register(
//         &mut self,
//         registry: &mut registry::BlockRegistry,
//     ) -> Result<Option<PublishHandle>, registry::BlockRegistationError> {
//         registry.register_block(&mut self.state)
//     }
// }

// BlockExt trait removed - methods now implemented directly on Block for better organization

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Getters)]
pub struct BasicMetadata {
    #[getter(copy)]
    priority: u32,
    #[getter(copy)]
    returned_tick: u64,
    #[getter(copy)]
    acquired_tick: u64,
}

impl BasicMetadata {
    pub fn update_priority(&self, priority: u32) -> Self {
        BasicMetadata {
            priority,
            returned_tick: self.returned_tick,
            acquired_tick: self.acquired_tick,
        }
    }
}

impl BlockMetadata for BasicMetadata {
    fn on_acquired(&mut self, tick: u64) {
        self.acquired_tick = tick;
    }

    fn on_returned(&mut self, tick: u64) {
        self.returned_tick = tick;
    }

    fn reset_metadata(&mut self) {
        self.priority = 0;
    }

    fn offload_priority(&self) -> Option<u64> {
        Some(self.priority as u64)
    }
}
/// Collection that holds shared storage and layout
#[derive(Debug)]
pub struct Blocks<L: BlockLayout, M: BlockMetadata> {
    layout: Box<L>,
    metadata: std::marker::PhantomData<M>,
    block_set_idx: usize,
    worker_id: WorkerID,
}

impl<L: BlockLayout + 'static, M: BlockMetadata> Blocks<L, M> {
    /// Create a new block storage collection
    pub fn new(layout: L, block_set_idx: usize, worker_id: WorkerID) -> BlockResult<Self> {
        let layout = Box::new(layout);

        Ok(Self {
            layout,
            metadata: std::marker::PhantomData,
            block_set_idx,
            worker_id,
        })
    }

    /// Convert collection into Vec<Block> with default metadata/state
    pub fn into_blocks(self) -> BlockResult<Vec<Block<L::StorageType, locality::Local, M>>> {
        // convert box to arc
        let layout: Arc<dyn BlockLayout<StorageType = L::StorageType>> = Arc::new(*self.layout);
        layout_to_blocks(layout, self.block_set_idx, self.worker_id)
    }
}

pub(crate) fn layout_to_blocks<S: Storage, M: BlockMetadata>(
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    block_set_idx: usize,
    worker_id: WorkerID,
) -> BlockResult<Vec<Block<S, locality::Local, M>>> {
    (0..layout.num_blocks())
        .map(|idx| {
            let data = BlockData::new(layout.clone(), idx, block_set_idx, worker_id);
            let data = data;
            Block::new(data, M::default())
        })
        .collect()
}

pub struct MutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Option<Block<S, L, M>>,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
    // Use to track parent relationship, as well as ensure that parents of registered blocks stay
    // alive as long as the child is alive.
    parent: Option<Arc<MutableBlock<S, L, M>>>,
}

// MutableBlock inherits identification methods from Block via Deref

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> StorageTypeProvider
    for MutableBlock<S, L, M>
{
    type StorageType = S;
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockDataProvider
    for MutableBlock<S, L, M>
{
    fn block_data(&self) -> &impl BlockDataExt<S> {
        &self.block.as_ref().expect("block was dropped").data
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockDataProviderMut
    for MutableBlock<S, L, M>
{
    fn block_data_mut(&mut self) -> &mut impl BlockDataExt<S> {
        &mut self.block.as_mut().expect("block was dropped").data
    }
}

// Marker trait implementations for MutableBlock
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Local for MutableBlock<S, L, M> {}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> MutableBlock<S, L, M> {
    pub(crate) fn new(
        block: Block<S, L, M>,
        return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
    ) -> Self {
        Self {
            block: Some(block),
            return_tx,
            parent: None,
        }
    }

    pub fn set_parent(&mut self, parent: Arc<MutableBlock<S, L, M>>) {
        self.parent = Some(parent);
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> std::fmt::Debug for MutableBlock<S, L, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MutableBlock {{ block: {:?} }}", self.block)
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Drop for MutableBlock<S, L, M> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            if self.return_tx.send(block).is_err() {
                tracing::warn!("block pool shutdown before block was returned");
            }
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Deref for MutableBlock<S, L, M> {
    type Target = Block<S, L, M>;

    fn deref(&self) -> &Self::Target {
        self.block.as_ref().expect("block was dropped")
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> DerefMut for MutableBlock<S, L, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.block.as_mut().expect("block was dropped")
    }
}

// MutableBlock provides access to block data through simpler methods
// Simplified MutableBlock API - direct delegation to underlying data
// MutableBlock inherits methods from Block via Deref - no need for separate implementations

// // Local-specific BlockDataProvider implementations
// impl<S: Storage + NixlDescriptor, M: BlockMetadata> BlockDataProvider
//     for MutableBlock<S, locality::Local, M>
// {
//     type StorageType = S;

//     fn block_data(&self, _: private::PrivateToken) -> &BlockData<S> {
//         &self.block.as_ref().expect("block was dropped").data
//     }
// }

// impl<S: Storage + NixlDescriptor, M: BlockMetadata> BlockDataProviderMut
//     for MutableBlock<S, locality::Local, M>
// {
//     fn block_data_mut(&mut self, _: private::PrivateToken) -> &mut BlockData<S> {
//         &mut self.block.as_mut().expect("block was dropped").data
//     }
// }

impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockSlice<'a, MutableBlock<S, L, M>> for [MutableBlock<S, L, M>]
{
    fn as_block_slice(&'a self) -> &'a [MutableBlock<S, L, M>] {
        self
    }
}
impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockSlice<'a, MutableBlock<S, L, M>> for Vec<MutableBlock<S, L, M>>
{
    fn as_block_slice(&'a self) -> &'a [MutableBlock<S, L, M>] {
        self.as_slice()
    }
}
impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockMutSlice<'a, MutableBlock<S, L, M>> for [MutableBlock<S, L, M>]
{
    fn as_block_mut_slice(&'a mut self) -> &'a mut [MutableBlock<S, L, M>] {
        self
    }
}
impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockMutSlice<'a, MutableBlock<S, L, M>> for Vec<MutableBlock<S, L, M>>
{
    fn as_block_mut_slice(&'a mut self) -> &'a mut [MutableBlock<S, L, M>] {
        self.as_mut_slice()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> IntoWritableBlocks<L, M>
    for MutableBlock<S, L, M>
{
    type Output = Vec<MutableBlock<S, L, M>>;
    fn into_writable_blocks(self, _manager: &BlockManager<L, M>) -> BlockResult<Self::Output> {
        Ok(vec![self])
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> IntoReadableBlocks<L, M>
    for MutableBlock<S, L, M>
{
    type Output = Vec<MutableBlock<S, L, M>>;
    fn into_readable_blocks(self, _manager: &BlockManager<L, M>) -> BlockResult<Self::Output> {
        Ok(vec![self])
    }
}

#[derive(Debug)]
pub struct ImmutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Arc<MutableBlock<S, L, M>>,
    sequence_hash: SequenceHash,
}

// ImmutableBlock inherits identification methods from Block via Deref

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Clone for ImmutableBlock<S, L, M> {
    fn clone(&self) -> Self {
        Self {
            block: self.block.clone(),
            sequence_hash: self.sequence_hash,
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ImmutableBlock<S, L, M> {
    pub(crate) fn new(block: Arc<MutableBlock<S, L, M>>) -> Self {
        let sequence_hash = block.sequence_hash().expect("block is in the wrong state");
        Self {
            block,
            sequence_hash,
        }
    }

    pub(crate) fn mutable_block(&self) -> &Arc<MutableBlock<S, L, M>> {
        &self.block
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> StorageTypeProvider
    for ImmutableBlock<S, L, M>
{
    type StorageType = S;
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockDataProvider
    for ImmutableBlock<S, L, M>
{
    fn block_data(&self) -> &impl BlockDataExt<S> {
        &self.block.block.as_ref().expect("block was dropped").data
    }
}

// Marker trait implementations for ImmutableBlock
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Local for ImmutableBlock<S, L, M> {}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Deref for ImmutableBlock<S, L, M> {
    type Target = Block<S, L, M>;
    fn deref(&self) -> &Self::Target {
        self.block
            .as_ref()
            .block
            .as_ref()
            .expect("block was dropped")
    }
}

// ImmutableBlock provides access to block data through simpler methods
// Simplified block API - direct delegation to underlying data
// ImmutableBlock inherits methods from Block via Deref - no need for separate implementations

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> IntoReadableBlocks<L, M>
    for ImmutableBlock<S, L, M>
{
    type Output = Vec<ImmutableBlock<S, L, M>>;
    fn into_readable_blocks(self, _manager: &BlockManager<L, M>) -> BlockResult<Self::Output> {
        Ok(vec![self])
    }
}

impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockSlice<'a, ImmutableBlock<S, L, M>> for [ImmutableBlock<S, L, M>]
{
    fn as_block_slice(&'a self) -> &'a [ImmutableBlock<S, L, M>] {
        self
    }
}
impl<'a, S: Storage + 'a, L: LocalityProvider + 'a, M: BlockMetadata>
    AsBlockSlice<'a, ImmutableBlock<S, L, M>> for Vec<ImmutableBlock<S, L, M>>
{
    fn as_block_slice(&'a self) -> &'a [ImmutableBlock<S, L, M>] {
        self.as_slice()
    }
}

impl<S: Storage + 'static, L: LocalityProvider, M: BlockMetadata> ImmutableBlock<S, L, M> {
    pub async fn enqueue_offload(&self, priority: u64) -> Result<()> {
        if let Some(manager) = self.manager() {
            manager.enqueue_offload_block(self, priority).await?;
        } else {
            tracing::warn!("Block is not managed. Unable to enqueue offload.");
        }
        Ok(())
    }
}

impl<B: BlockDataProvider> ReadableBlock for B {}
impl<B: BlockDataProviderMut> WritableBlock for B {}

pub mod nixl {
    use super::*;

    use super::view::{BlockKind, Kind, LayerKind};

    use super::super::{
        layout::nixl::{NixlLayout, SerializedNixlBlockLayout},
        storage::nixl::{MemType, NixlRegisterableStorage, NixlStorage},
        WorkerID,
    };

    use derive_getters::{Dissolve, Getters};
    use nixl_sys::{Agent as NixlAgent, MemoryRegion, NixlDescriptor, OptArgs};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    // --- Mutability Marker ---
    pub trait MutabilityKind: Debug + Clone + Copy + Send + Sync + 'static {}

    #[derive(Debug, Clone, Copy)]
    pub struct IsMutable;
    impl MutabilityKind for IsMutable {}

    #[derive(Debug, Clone, Copy)]
    pub struct IsImmutable;
    impl MutabilityKind for IsImmutable {}

    impl<L: NixlLayout, M: BlockMetadata> Blocks<L, M>
    where
        L::StorageType: NixlRegisterableStorage,
    {
        /// Register the blocks with an NIXL agent
        pub fn nixl_register(
            &mut self,
            agent: &NixlAgent,
            opt_args: Option<&OptArgs>,
        ) -> anyhow::Result<()> {
            self.layout.nixl_register(agent, opt_args)
        }
    }

    /// A unified, lifetime-bound descriptor containing information needed for NIXL operations.
    /// Typed by Kind (Block/Layer) and Mutability (IsMutable/IsImmutable).
    #[derive(Copy, Clone)] // Can be Copy/Clone as it holds basic data + markers
    pub struct NixlMemoryDescriptor<'a, K: Kind, M: MutabilityKind> {
        addr: u64,
        size: usize,
        mem_type: MemType,
        device_id: u64,
        _lifetime: std::marker::PhantomData<&'a ()>, // Binds the descriptor's lifetime to 'a
        _kind: std::marker::PhantomData<K>,          // Stores the Kind marker type
        _mutability: std::marker::PhantomData<M>,    // Stores the Mutability marker type
    }

    // Helper function to get the short type name
    pub(crate) fn short_type_name<T>() -> &'static str {
        let name = core::any::type_name::<T>();
        name.split("::").last().unwrap_or(name)
    }

    // Implement Debug manually to avoid bounds on K/M
    impl<K: Kind, M: MutabilityKind> Debug for NixlMemoryDescriptor<'_, K, M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("NixlMemoryDescriptor")
                .field("addr", &self.addr)
                .field("size", &self.size)
                .field("mem_type", &self.mem_type)
                .field("device_id", &self.device_id)
                .field("kind", &short_type_name::<K>()) // Show marker types
                .field("mutability", &short_type_name::<M>())
                .finish()
        }
    }

    impl<K: Kind, M: MutabilityKind> NixlMemoryDescriptor<'_, K, M> {
        /// Creates a new NixlMemoryDescriptor. Typically called via conversion methods.
        #[inline]
        pub(crate) fn new(addr: u64, size: usize, mem_type: MemType, device_id: u64) -> Self {
            Self {
                addr,
                size,
                mem_type,
                device_id,
                _lifetime: std::marker::PhantomData,
                _kind: std::marker::PhantomData,
                _mutability: std::marker::PhantomData,
            }
        }
    }

    impl<K: Kind, M: MutabilityKind> MemoryRegion for NixlMemoryDescriptor<'_, K, M> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size
        }
    }

    impl<K: Kind, M: MutabilityKind> NixlDescriptor for NixlMemoryDescriptor<'_, K, M> {
        fn mem_type(&self) -> MemType {
            self.mem_type
        }

        fn device_id(&self) -> u64 {
            self.device_id
        }
    }

    // Comment out Nixl-related code for now
    pub trait NixlBlockDataImmutable<S: Storage + NixlDescriptor>: BlockDataExt<S> {
        /// Get the NIXL memory descriptor for the entire block
        fn as_block_descriptor(
            &self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>>;

        /// Get the NIXL memory descriptor for a specific layer
        fn as_layer_descriptor(
            &self,
            layer_idx: usize,
            outer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>>;
    }

    pub trait NixlBlockDataMutable<S: Storage + NixlDescriptor>:
        BlockDataExt<S> + NixlBlockDataImmutable<S>
    {
        /// Get the NIXL memory descriptor for the entire block
        fn as_block_descriptor_mut(
            &mut self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>>;

        /// Get the NIXL memory descriptor for a specific layer
        fn as_layer_descriptor_mut(
            &mut self,
            layer_idx: usize,
            outer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>>;
    }

    impl<S: Storage + NixlDescriptor> NixlBlockDataImmutable<S> for BlockData<S> {
        fn as_block_descriptor(
            &self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>> {
            Ok(self.block_view()?.as_nixl_descriptor())
        }

        fn as_layer_descriptor(
            &self,
            layer_idx: usize,
            outer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>> {
            Ok(self.layer_view(layer_idx, outer_idx)?.as_nixl_descriptor())
        }
    }

    impl<S: Storage + NixlDescriptor> NixlBlockDataMutable<S> for BlockData<S> {
        fn as_block_descriptor_mut(
            &mut self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>> {
            Ok(self.block_view_mut()?.as_nixl_descriptor_mut())
        }

        fn as_layer_descriptor_mut(
            &mut self,
            layer_idx: usize,
            outer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>> {
            Ok(self
                .layer_view_mut(layer_idx, outer_idx)?
                .as_nixl_descriptor_mut())
        }
    }
    // ... existing code ...

    /// Error type for NixlBlockSet serialization/deserialization failures.
    #[derive(Debug, Error)]
    pub enum NixlSerializationError {
        #[error("Serialization failed: {0}")]
        Serialize(#[from] serde_json::Error),
    }

    /// A strongly-typed wrapper for serialized NixlBlockSet data.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializedNixlBlockSet(Vec<u8>);

    impl TryFrom<&NixlBlockSet> for SerializedNixlBlockSet {
        type Error = NixlSerializationError;

        /// Serializes a NixlBlockSet into SerializedNixlBlockSet.
        fn try_from(value: &NixlBlockSet) -> Result<Self, Self::Error> {
            let bytes = serde_json::to_vec(value)?;
            Ok(SerializedNixlBlockSet(bytes))
        }
    }

    impl TryFrom<NixlBlockSet> for SerializedNixlBlockSet {
        type Error = NixlSerializationError;

        /// Serializes a NixlBlockSet into SerializedNixlBlockSet, consuming the original.
        fn try_from(value: NixlBlockSet) -> Result<Self, Self::Error> {
            let bytes = serde_json::to_vec(&value)?;
            Ok(SerializedNixlBlockSet(bytes))
        }
    }

    impl TryFrom<&SerializedNixlBlockSet> for NixlBlockSet {
        type Error = NixlSerializationError;

        /// Deserializes SerializedNixlBlockSet into a NixlBlockSet.
        fn try_from(value: &SerializedNixlBlockSet) -> Result<Self, Self::Error> {
            let block_set = serde_json::from_slice(&value.0)?;
            Ok(block_set)
        }
    }

    impl TryFrom<SerializedNixlBlockSet> for NixlBlockSet {
        type Error = NixlSerializationError;

        /// Deserializes SerializedNixlBlockSet into a NixlBlockSet, consuming the original.
        fn try_from(value: SerializedNixlBlockSet) -> Result<Self, Self::Error> {
            let block_set = serde_json::from_slice(&value.0)?;
            Ok(block_set)
        }
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize, Dissolve)]
    pub struct NixlBlockSet {
        /// The block set index
        block_sets: HashMap<usize, SerializedNixlBlockLayout>,

        /// Captures the NIXL metadata from [nixl_sys::Agent::get_local_md]
        nixl_metadata: Vec<u8>,

        /// Worker ID
        worker_id: u64,
    }

    impl NixlBlockSet {
        pub fn new(worker_id: u64) -> Self {
            Self {
                block_sets: HashMap::new(),
                nixl_metadata: Vec::new(),
                worker_id,
            }
        }

        pub fn worker_id(&self) -> u64 {
            self.worker_id
        }

        /// Get the block set for a given block set index
        pub fn block_sets(&self) -> &HashMap<usize, SerializedNixlBlockLayout> {
            &self.block_sets
        }

        /// Add a block set to the block set
        pub fn add_block_set(
            &mut self,
            block_set_idx: usize,
            serialized_layout: SerializedNixlBlockLayout,
        ) {
            self.block_sets.insert(block_set_idx, serialized_layout);
        }

        /// Get the NIXL metadata
        pub fn get_nixl_metadata(&self) -> &Vec<u8> {
            &self.nixl_metadata
        }

        /// Set the NIXL metadata
        pub fn set_nixl_metadata(&mut self, nixl_metadata: Vec<u8>) {
            self.nixl_metadata = nixl_metadata;
        }
    }

    #[derive(Debug, Clone)]
    pub struct RemoteBlocks {
        layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
        block_set_idx: usize,
        worker_id: WorkerID,
    }

    impl RemoteBlocks {
        pub fn new(
            layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
            block_set_idx: usize,
            worker_id: WorkerID,
        ) -> Self {
            Self {
                layout,
                block_set_idx,
                worker_id,
            }
        }

        pub fn from_serialized(
            serialized: SerializedNixlBlockLayout,
            block_set_idx: usize,
            worker_id: WorkerID,
        ) -> BlockResult<Self> {
            let layout = serialized.deserialize()?;
            Ok(Self::new(layout, block_set_idx, worker_id))
        }

        pub fn block<M: MutabilityKind>(&self, block_idx: usize) -> BlockResult<RemoteBlock<M>> {
            if block_idx >= self.layout.num_blocks() {
                return Err(BlockError::InvalidState(format!(
                    "block index out of bounds: {} >= {}",
                    block_idx,
                    self.layout.num_blocks()
                )));
            }
            Ok(RemoteBlock::new(
                self.layout.clone(),
                block_idx,
                self.block_set_idx,
                self.worker_id,
            ))
        }

        /// Get the layout of the remote blocks
        pub fn layout(&self) -> &dyn BlockLayout<StorageType = NixlStorage> {
            self.layout.as_ref()
        }
    }

    pub type ImmutableRemoteBlock = RemoteBlock<IsImmutable>;
    pub type MutableRemoteBlock = RemoteBlock<IsMutable>;

    pub struct RemoteBlock<M: MutabilityKind> {
        data: BlockData<NixlStorage>,
        _mutability: std::marker::PhantomData<M>,
    }

    impl<M: MutabilityKind> Remote for RemoteBlock<M> {}

    // impl<M: MutabilityKind> ReadableBlock for RemoteBlock<M> {
    //     type StorageType = NixlStorage;
    // }

    // impl WritableBlock for RemoteBlock<IsMutable> {
    //     type StorageType = NixlStorage;
    // }

    impl<M: MutabilityKind> RemoteBlock<M> {
        pub fn new(
            layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
            block_idx: usize,
            block_set_idx: usize,
            worker_id: WorkerID,
        ) -> Self {
            let data = BlockData::new(layout, block_idx, block_set_idx, worker_id);
            Self {
                data,
                _mutability: std::marker::PhantomData,
            }
        }
    }

    // impl<M: MutabilityKind> BlockDataExt<NixlStorage> for RemoteBlock<M> {
    //     // fn is_fully_contiguous(&self) -> bool {
    //     //     self.data.is_fully_contiguous()
    //     // }

    //     fn num_layers(&self) -> usize {
    //         self.data.num_layers()
    //     }

    //     fn num_outer_dims(&self) -> usize {
    //         self.data.num_outer_dims()
    //     }

    //     fn layer_view(
    //         &self,
    //         layer_idx: usize,
    //         outer_idx: usize,
    //     ) -> BlockResult<view::LayerView<NixlStorage>> {
    //         self.data.layer_view(layer_idx, outer_idx)
    //     }

    //     fn layer_view_mut(
    //         &mut self,
    //         layer_idx: usize,
    //         outer_idx: usize,
    //     ) -> BlockResult<view::LayerViewMut<NixlStorage>> {
    //         self.data.layer_view_mut(layer_idx, outer_idx)
    //     }

    //     fn block_view(&self) -> BlockResult<view::BlockView<NixlStorage>> {
    //         self.data.block_view()
    //     }

    //     fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<NixlStorage>> {
    //         self.data.block_view_mut()
    //     }
    // }

    impl<M: MutabilityKind> StorageTypeProvider for RemoteBlock<M> {
        type StorageType = NixlStorage;
    }

    impl<M: MutabilityKind> BlockDataProvider for RemoteBlock<M> {
        fn block_data(&self) -> &impl BlockDataExt<NixlStorage> {
            &self.data
        }
    }

    // impl<M: MutabilityKind> NixlBlockDataImmutable<NixlStorage> for RemoteBlock<M> {
    //     fn as_block_descriptor(
    //         &self,
    //     ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>> {
    //         self.data.as_block_descriptor()
    //     }

    //     fn as_layer_descriptor(
    //         &self,
    //         layer_idx: usize,
    //         outer_idx: usize,
    //     ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>> {
    //         self.data.as_layer_descriptor(layer_idx, outer_idx)
    //     }
    // }

    impl BlockDataProviderMut for RemoteBlock<IsMutable> {
        fn block_data_mut(&mut self) -> &mut impl BlockDataExt<NixlStorage> {
            &mut self.data
        }
    }
    // impl NixlBlockDataMutable<NixlStorage> for RemoteBlock<IsMutable> {
    //     fn as_block_descriptor_mut(
    //         &mut self,
    //     ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>> {
    //         self.data.as_block_descriptor_mut()
    //     }

    //     fn as_layer_descriptor_mut(
    //         &mut self,
    //         layer_idx: usize,
    //         outer_idx: usize,
    //     ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>> {
    //         self.data.as_layer_descriptor_mut(layer_idx, outer_idx)
    //     }
    // }

    impl<'a, M: MutabilityKind> AsBlockSlice<'a, RemoteBlock<M>> for [RemoteBlock<M>] {
        fn as_block_slice(&'a self) -> &'a [RemoteBlock<M>] {
            self
        }
    }

    impl<'a, M: MutabilityKind> AsBlockSlice<'a, RemoteBlock<M>> for Vec<RemoteBlock<M>> {
        fn as_block_slice(&'a self) -> &'a [RemoteBlock<M>] {
            self.as_slice()
        }
    }

    impl<'a> AsBlockMutSlice<'a, RemoteBlock<IsMutable>> for [RemoteBlock<IsMutable>] {
        fn as_block_mut_slice(&'a mut self) -> &'a mut [RemoteBlock<IsMutable>] {
            self
        }
    }

    impl<'a> AsBlockMutSlice<'a, RemoteBlock<IsMutable>> for Vec<RemoteBlock<IsMutable>> {
        fn as_block_mut_slice(&'a mut self) -> &'a mut [RemoteBlock<IsMutable>] {
            self.as_mut_slice()
        }
    }

    /// Defines the intended access pattern for a block represented by a descriptor.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum BlockMutability {
        Immutable,
        Mutable,
    }

    /// Describes a single block for identification and potential remote access setup.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BlockDescriptor {
        pub worker_id: WorkerID,
        pub block_set_idx: usize,
        pub block_idx: usize,
        pub mutability: BlockMutability,
    }

    /// A validated, homogeneous, and serializable collection of BlockDescriptors.
    /// Primarily used to describe sets of remote blocks for transfer operations.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Getters)]
    pub struct BlockDescriptorList {
        #[getter(copy)]
        worker_id: WorkerID,

        #[getter(copy)]
        block_set_idx: usize,

        #[getter(copy)]
        mutability: BlockMutability,

        block_indices: Vec<usize>,
        // TODO: Consider storing MemType explicitly if it cannot be reliably
        // derived from block_set_idx via the NixlBlockSet on the receiving side.
    }

    // impl<L: LocalityProvider, M: BlockMetadata> IntoWritableBlocks<L, M> for BlockDescriptorList {
    //     type Output = Vec<RemoteBlock<IsMutable>>;
    //     fn into_writable_blocks(self, manager: &BlockManager<L, M>) -> BlockResult<Self::Output> {
    //         Ok(manager.get_remote_blocks_mutable(&self)?)
    //     }
    // }

    #[derive(Debug, Error)]
    pub enum BlockDescriptorSetError {
        #[error("Input block list cannot be empty")]
        EmptyInput,

        #[error(
            "Blocks in the input list are not homogeneous (worker_id, block_set_idx mismatch)"
        )]
        NotHomogeneous,

        #[error("Serialization failed: {0}")]
        SerializationError(#[from] serde_json::Error),
        #[error(
            "An invalid block handle was encountered (block may have been dropped prematurely)"
        )]
        InvalidBlockHandle,
    }

    // impl BlockDescriptorList {
    //     /// Creates a new validated BlockDescriptorList from a slice of block handles.
    //     /// Ensures all handles belong to the same worker and block set.
    //     fn new<S: Storage>(
    //         blocks: &[&BlockData<S>], // Use the generic trait bound
    //         mutability: BlockMutability,
    //     ) -> Result<Self, BlockDescriptorSetError> {
    //         if blocks.is_empty() {
    //             return Err(BlockDescriptorSetError::EmptyInput);
    //         }

    //         let first = blocks[0];
    //         let worker_id = first.worker_id();
    //         let block_set_idx = first.block_set_id();

    //         let mut block_indices = Vec::with_capacity(blocks.len());
    //         block_indices.push(first.block_id());

    //         for block in blocks.iter().skip(1) {
    //             // Validate homogeneity
    //             if block.worker_id() != worker_id || block.block_set_id() != block_set_idx {
    //                 return Err(BlockDescriptorSetError::NotHomogeneous);
    //             }
    //             block_indices.push(block.block_id());
    //         }

    //         // TODO: Potentially validate MemType derived from block_set_idx here if possible

    //         Ok(Self {
    //             worker_id,
    //             block_set_idx,
    //             mutability,
    //             block_indices,
    //         })
    //     }

    // /// Creates a BlockDescriptorList representing immutable blocks.
    // pub fn from_immutable_blocks<S: Storage, L: LocalityProvider, M: BlockMetadata>(
    //     blocks: &[ImmutableBlock<S, L, M>],
    // ) -> Result<Self, BlockDescriptorSetError> {
    //     // Map each block handle to Option<&BlockData>,
    //     // then convert Option to Result (treating None as an error),
    //     // finally collect into Result<Vec<&BlockData>, Error>.
    //     let data: Vec<&BlockData<S>> = blocks
    //         .iter()
    //         .map(|b| {
    //             b.block
    //                 .block
    //                 .as_ref()
    //                 .map(|inner_b| &inner_b.data.block_data())
    //         })
    //         .map(|opt| opt.ok_or(BlockDescriptorSetError::InvalidBlockHandle))
    //         .collect::<Result<Vec<&BlockData<S>>, _>>()?;
    //     Self::new(&data, BlockMutability::Immutable)
    // }

    // /// Creates a BlockDescriptorList representing mutable blocks.
    // pub fn from_mutable_blocks<S: Storage, L: LocalityProvider, M: BlockMetadata>(
    //     blocks: &[MutableBlock<S, L, M>],
    // ) -> Result<Self, BlockDescriptorSetError> {
    //     // Map each block handle to Option<&BlockData>,
    //     // then convert Option to Result (treating None as an error),
    //     // finally collect into Result<Vec<&BlockData>, Error>.
    //     let data: Vec<&LocalBlockData<S>> = blocks
    //         .iter()
    //         .map(|b| b.block.as_ref().map(|inner_b| &inner_b.data))
    //         .map(|opt| opt.ok_or(BlockDescriptorSetError::InvalidBlockHandle))
    //         .collect::<Result<Vec<&LocalBlockData<S>>, _>>()?;

    //     Self::new(&data, BlockMutability::Mutable)
    // }

    // /// Serializes the BlockDescriptorList into a byte vector.
    // pub fn serialize(&self) -> Result<Vec<u8>, BlockDescriptorSetError> {
    //     Ok(serde_json::to_vec(self)?)
    // }

    // /// Deserializes a BlockDescriptorList from a byte slice.
    // pub fn deserialize(data: &[u8]) -> Result<Self, BlockDescriptorSetError> {
    //     Ok(serde_json::from_slice(data)?)
    // }

    pub trait AsBlockDescriptorSet {
        type Block;
        fn as_block_descriptor_set(&self) -> Result<BlockDescriptorList, BlockDescriptorSetError>;
    }

    // impl<S, L, M> AsBlockDescriptorSet for [ImmutableBlock<S, L, M>]
    // where
    //     S: Storage,
    //     L: LocalityProvider,
    //     M: BlockMetadata,
    // {
    //     type Block = ImmutableBlock<S, locality::Local, M>;
    //     fn as_block_descriptor_set(&self) -> Result<BlockDescriptorList, BlockDescriptorSetError> {
    //         BlockDescriptorList::from_immutable_blocks(self)
    //     }
    // }

    // impl<S, M> AsBlockDescriptorSet for [MutableBlock<S, locality::Local, M>]
    // where
    //     S: Storage,
    //     M: BlockMetadata,
    // {
    //     type Block = MutableBlock<S, locality::Local, M>;
    //     fn as_block_descriptor_set(&self) -> Result<BlockDescriptorList, BlockDescriptorSetError> {
    //         BlockDescriptorList::from_mutable_blocks(self)
    //     }
    // }

    // impl<T> AsBlockDescriptorSet for Vec<T>
    // where
    //     [T]: AsBlockDescriptorSet<Block = T>,
    // {
    //     type Block = T;
    //     fn as_block_descriptor_set(&self) -> Result<BlockDescriptorList, BlockDescriptorSetError> {
    //         self.as_slice().as_block_descriptor_set()
    //     }
    // }

    // impl<T, const N: usize> AsBlockDescriptorSet for [T; N]
    // where
    //     [T]: AsBlockDescriptorSet<Block = T>,
    // {
    //     type Block = T;
    //     fn as_block_descriptor_set(&self) -> Result<BlockDescriptorList, BlockDescriptorSetError> {
    //         self.as_slice().as_block_descriptor_set()
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::layout::tests::setup_layout;

    use crate::tokens::{TokenBlockSequence, Tokens};

    const BLOCK_SIZE: usize = 4;
    const SALT_HASH: SaltHash = 12345;

    // Helper to create a default reset block
    fn create_reset_block() -> Block<impl Storage, locality::Local, BasicMetadata> {
        let layout = setup_layout(None).unwrap();
        let data = BlockData::new(Arc::new(layout), 0, 42, 0);
        Block::new(data, BasicMetadata::default()).unwrap()
    }

    // Helper to create a complete TokenBlock for testing apply_token_block
    fn create_full_token_block() -> TokenBlock {
        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let salt_hash = SALT_HASH;
        let block_size = BLOCK_SIZE;
        let (mut blocks, _) =
            TokenBlockSequence::split_tokens(tokens.as_ref(), block_size, salt_hash);
        blocks.pop().unwrap()
    }

    #[test]
    fn test_block_state_transitions_and_ops() {
        let mut block = create_reset_block();
        assert!(matches!(block.state(), BlockState::Reset));

        // --- Reset State --- //
        assert!(block.add_token(1).is_err(), "Append on Reset should fail");
        assert!(
            block.add_tokens(Tokens::from(vec![1])).is_err(),
            "Extend on Reset should fail"
        );
        assert!(block.commit().is_err(), "Commit on Reset should fail");
        assert!(block.pop_token().is_err(), "Pop on Reset should fail");
        assert!(
            block.pop_tokens(1).is_err(),
            "Pop tokens on Reset should fail"
        );

        // --- Reset -> Partial (via init_sequence) --- //
        assert!(block.init_sequence(SALT_HASH).is_ok());
        assert!(matches!(block.state(), BlockState::Partial(_)));

        // --- Partial State --- //
        let invalid_block = create_full_token_block();
        assert!(
            block.apply_token_block(invalid_block).is_err(),
            "Apply block on Partial should fail"
        );

        // Append tokens
        assert!(block.add_token(1).is_ok()); // 1
        assert!(block.add_token(2).is_ok()); // 1, 2
        assert!(block.add_tokens(Tokens::from(vec![3])).is_ok()); // 1, 2, 3
        assert_eq!(block.len(), 3);

        // Extend beyond capacity (should fail)
        let new_tokens = Tokens::from(vec![4, 5]);
        assert_eq!(block.add_tokens(new_tokens.clone()).unwrap().as_ref(), &[5]);

        // Extend to fill capacity
        assert!(block.add_tokens(Tokens::from(vec![4])).is_ok()); // 1, 2, 3, 4
        assert_eq!(block.len(), BLOCK_SIZE);

        // Append when full (should fail)
        assert!(block.add_token(5).is_err(), "Append on full Partial block");

        // Pop tokens
        assert!(block.pop_token().is_ok()); // After pop: 1, 2, 3
        assert_eq!(block.len(), 3);

        // Pop multiple tokens
        assert!(block.pop_tokens(2).is_ok()); // After pop: [1]
        assert_eq!(block.len(), 1);

        // Pop too many tokens (should fail)
        assert!(block.pop_tokens(2).is_err(), "Pop too many tokens");
        assert_eq!(block.len(), 1);

        // Pop last token
        assert!(block.pop_token().is_ok()); // empty
        assert_eq!(block.len(), 0);
        assert!(block.is_empty());

        // Fill block again for commit
        assert!(block.add_tokens(Tokens::from(vec![1, 2, 3, 4])).is_ok());
        assert_eq!(block.len(), BLOCK_SIZE);

        // --- Partial -> Complete (via commit) --- //
        assert!(block.commit().is_ok());
        assert!(matches!(block.state(), BlockState::Complete(_)));
        assert_eq!(block.tokens().unwrap().as_ref(), &[1, 2, 3, 4]);

        // --- Complete State --- //
        assert!(
            block.init_sequence(SALT_HASH).is_err(),
            "Init sequence on Complete should fail"
        );
        assert!(
            block.add_token(5).is_err(),
            "Append on Complete should fail"
        );
        assert!(
            block.add_tokens(Tokens::from(vec![5])).is_err(),
            "Extend on Complete should fail"
        );
        assert!(block.commit().is_err(), "Commit on Complete should fail");
        assert!(block.pop_token().is_err(), "Pop on Complete should fail");
        assert!(
            block.pop_tokens(1).is_err(),
            "Pop tokens on Complete should fail"
        );
        let invalid_block = create_full_token_block();
        assert!(
            block.apply_token_block(invalid_block).is_err(),
            "Apply block on Complete should fail"
        );

        // --- Complete -> Reset (via reset) --- //
        block.reset();
        assert!(matches!(block.state(), BlockState::Reset));

        // --- Reset -> Complete (via apply_token_block) --- //
        let full_block = create_full_token_block();
        assert!(block.apply_token_block(full_block.clone()).is_ok());
        assert!(matches!(block.state(), BlockState::Complete(_)));
        let applied_tokens = block.tokens().unwrap();
        assert_eq!(applied_tokens, full_block.tokens());

        // Testing applying to a non-reset state:
        let mut non_reset_block = create_reset_block();
        non_reset_block.init_sequence(SALT_HASH).unwrap(); // Put in Partial state
        assert!(
            non_reset_block.apply_token_block(full_block).is_err(),
            "Apply block to non-reset state"
        );
    }

    #[test]
    fn test_block_state_incomplete_commit() {
        // Commit incomplete block (should fail)
        let mut partial_block = create_reset_block();
        partial_block.init_sequence(SALT_HASH).unwrap();
        partial_block.add_token(1).unwrap();
        partial_block.add_tokens(Tokens::from(vec![2, 3])).unwrap();
        assert_eq!(partial_block.len(), 3);
        assert!(
            partial_block.commit().is_err(),
            "Commit on incomplete Partial block"
        );
    }

    #[test]
    fn test_error_types() {
        let mut block = create_reset_block();
        block.init_sequence(SALT_HASH).unwrap();

        // Fill the block
        block.add_tokens(Tokens::from(vec![1, 2, 3, 4])).unwrap();

        // Append when full
        let append_err = block.add_token(5).unwrap_err();
        assert!(append_err.is::<TokenBlockError>());
        assert_eq!(
            *append_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::Full
        );

        // .add_tokens will try to fill the block and return the remaining tokens in the Tokens passed in
        let new_tokens = Tokens::from(vec![5]);
        let ret_tokens = block.add_tokens(new_tokens.clone()).unwrap();
        assert_eq!(new_tokens, ret_tokens);

        // Commit when full (should succeed)
        block.commit().unwrap();

        // Commit when Complete
        let commit_err = block.commit().unwrap_err();
        assert!(commit_err.is::<BlockStateInvalid>());

        // Reset and test pop empty
        block.reset();
        block.init_sequence(SALT_HASH).unwrap();
        let pop_err = block.pop_token().unwrap_err();
        assert!(pop_err.is::<TokenBlockError>());
        assert_eq!(
            *pop_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::Empty
        );

        let pop_tokens_err = block.pop_tokens(1).unwrap_err();
        assert!(pop_tokens_err.is::<TokenBlockError>());
        assert_eq!(
            *pop_tokens_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::InsufficientTokens
        );

        // Test commit incomplete
        block.add_token(1).unwrap();
        let commit_incomplete_err = block.commit().unwrap_err();
        assert!(commit_incomplete_err.is::<TokenBlockError>());
        assert_eq!(
            *commit_incomplete_err
                .downcast_ref::<TokenBlockError>()
                .unwrap(),
            TokenBlockError::Incomplete
        );
    }

    // #[test]
    // fn test_nixl_block_data_ext() {
    //     init_logging();

    //     let config = LayoutConfig::builder()
    //         .num_blocks(10)
    //         .num_layers(3)
    //         .outer_dim(2)
    //         .page_size(4)
    //         .inner_dim(13)
    //         .build()
    //         .unwrap();

    //     let mut layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
    //     let agent = NixlAgent::new("test").unwrap();

    //     tracing::info!("Registering layout");
    //     layout.nixl_register(&agent, None).unwrap();
    //     tracing::info!("Layout registered");

    //     let serialized = layout.serialize().unwrap();
    //     let layout = Arc::new(layout);

    //     let data = BlockData::new(layout.clone(), 0, 42, 0);
    //     assert_eq!(data.block_id(), 0);
    //     assert_eq!(data.block_set_id(), 42);
    //     let block_desc = data.as_block_descriptor().unwrap();
    //     println!("Block descriptor: {:?}", block_desc);

    //     let data = BlockData::new(layout.clone(), 1, 42, 0);
    //     assert_eq!(data.block_id(), 1);
    //     assert_eq!(data.block_set_id(), 42);
    //     let block_desc = data.as_block_descriptor().unwrap();
    //     println!("Block descriptor: {:?}", block_desc);

    //     let remote_layout = SerializedNixlBlockLayout::deserialize(&serialized).unwrap();
    //     println!("Nixl layout: {:?}", remote_layout);

    //     let remote_block = RemoteBlock::<IsMutable>::new(remote_layout.clone(), 0, 42, 0);
    //     let remote_desc = remote_block.as_block_descriptor().unwrap();
    //     println!("Remote Descriptor: {:?}", remote_desc);

    //     // drop(layout);
    //     tracing::info!("Layout dropped");
    // }

    // #[test]
    // fn test_mutable_block_data_ext() {
    //     init_logging();

    //     // Create a layout with multiple layers and blocks for testing all methods
    //     let config = LayoutConfig::builder()
    //         .num_blocks(10)
    //         .num_layers(2)
    //         .outer_dim(1)
    //         .page_size(4)
    //         .inner_dim(13)
    //         .build()
    //         .unwrap();

    //     let layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
    //     let layout = Arc::new(layout);

    //     // Create a channel for returning blocks
    //     let (return_tx, _return_rx) = tokio::sync::mpsc::unbounded_channel();

    //     // Create a block and wrap it in a MutableBlock
    //     let block_data = BlockData::new(layout.clone(), 0, 42, 0);
    //     let block = Block::new(block_data.into(), BasicMetadata::default()).unwrap();
    //     let mut mutable_block = MutableBlock::new(block, return_tx.clone());

    //     // Test is_fully_contiguous()
    //     assert!(mutable_block.is_fully_contiguous());

    //     // Test num_layers()
    //     assert_eq!(mutable_block.num_layers(), 2);

    //     // Test layer_view()
    //     let layer_view = mutable_block.layer_view(0, 0).unwrap();
    //     assert_eq!(layer_view.size(), 4 * 13 * 2); // page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { layer_view.as_ptr() }.is_null());

    //     // Test layer_view_mut()
    //     let mut layer_view_mut = mutable_block.layer_view_mut(1, 0).unwrap();
    //     assert_eq!(layer_view_mut.size(), 4 * 13 * 2); // page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { layer_view_mut.as_mut_ptr() }.is_null());

    //     // Test block_view()
    //     let block_view = mutable_block.block_view().unwrap();
    //     assert_eq!(block_view.size(), 2 * 4 * 13 * 2); // num_layers x page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { block_view.as_ptr() }.is_null());

    //     // Test block_view_mut()
    //     let mut block_view_mut = mutable_block.block_view_mut().unwrap();
    //     assert_eq!(block_view_mut.size(), 2 * 4 * 13 * 2); // num_layers x page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { block_view_mut.as_mut_ptr() }.is_null());

    //     tracing::info!("MutableBlock BlockDataExt tests completed successfully");
    // }

    // #[test]
    // fn test_immutable_block_data_ext() {
    //     init_logging();

    //     // Create a layout with multiple layers and blocks for testing all methods
    //     let config = LayoutConfig::builder()
    //         .num_blocks(10)
    //         .num_layers(2)
    //         .outer_dim(1)
    //         .page_size(4)
    //         .inner_dim(13)
    //         .build()
    //         .unwrap();

    //     let layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
    //     let layout = Arc::new(layout);

    //     // Create a channel for returning blocks
    //     let (return_tx, _return_rx) = tokio::sync::mpsc::unbounded_channel();

    // // Create a block and wrap it in a MutableBlock
    // let block_data = BlockData::new(layout.clone(), 0, 42, 0);
    // let block = Block::new(block_data, BasicMetadata::default()).unwrap();
    // let mut mutable_block = MutableBlock::new(block, return_tx.clone());

    // let tbs = TokenBlockSequence::new(Tokens::from(vec![0, 0, 0, 0]), 4, None);
    // let token_block = tbs.blocks().iter().next().unwrap();

    // mutable_block
    //     .apply_token_block(token_block.clone())
    //     .unwrap();

    //     // Wrap the mutable block in an Arc and create an ImmutableBlock from it
    //     let arc_mutable_block = Arc::new(mutable_block);
    //     let immutable_block = ImmutableBlock::new(arc_mutable_block);

    //     // Test is_fully_contiguous()
    //     assert!(immutable_block.is_fully_contiguous());

    //     // Test num_layers()
    //     assert_eq!(immutable_block.num_layers(), 2);

    //     // Test layer_view()
    //     let layer_view = immutable_block.layer_view(0, 0).unwrap();
    //     assert_eq!(layer_view.size(), 4 * 13 * 2); // page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { layer_view.as_ptr() }.is_null());

    //     // Test block_view()
    //     let block_view = immutable_block.block_view().unwrap();
    //     assert_eq!(block_view.size(), 2 * 4 * 13 * 2); // num_layers x page_size x inner_dim x dtype_bytes
    //     assert!(!unsafe { block_view.as_ptr() }.is_null());

    //     // Test that mutable methods return errors
    //     let mut mut_immutable_block = immutable_block; // We need a mutable reference for these tests

    //     let layer_view_mut_res = mut_immutable_block.layer_view_mut(0, 0);
    //     assert!(layer_view_mut_res.is_err());
    //     if let Err(BlockError::InvalidState(msg)) = layer_view_mut_res {
    //         assert!(msg.contains("immutable block"));
    //     } else {
    //         panic!("Expected InvalidState error");
    //     }

    //     let block_view_mut_res = mut_immutable_block.block_view_mut();
    //     assert!(block_view_mut_res.is_err());
    //     if let Err(BlockError::InvalidState(msg)) = block_view_mut_res {
    //         assert!(msg.contains("immutable block"));
    //     } else {
    //         panic!("Expected InvalidState error");
    //     }

    //     tracing::info!("ImmutableBlock BlockDataExt tests completed successfully");
    // }
}
