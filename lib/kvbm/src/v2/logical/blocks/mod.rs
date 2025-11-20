// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard types for type-safe block management
//!
//! This module provides type-safe RAII guards that ensure automatic resource cleanup:
//! - `MutableBlock`: Guards blocks in Reset state
//! - `CompleteBlock`: Guards blocks in Complete state
//! - `ImmutableBlock`: Guards registered blocks with upgrade capability
//! - `WeakBlock`: Weak references to registered blocks
//! - `PrimaryBlock`, `DuplicateBlock`: Internal registered block types

mod complete;
mod immutable;
mod mutable;
mod registered;
pub(crate) mod registry;

pub use complete::CompleteBlock;
pub use immutable::{ImmutableBlock, WeakBlock};
pub use mutable::MutableBlock;

pub(crate) mod state;
pub(crate) use registered::{DuplicateBlock, PrimaryBlock};
pub(crate) use registry::{BlockRegistrationHandle, BlockRegistry};

pub trait BlockMetadata: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> BlockMetadata for T {}

/// Logical Block Identifier
pub use super::{BlockId, SequenceHash};
use dynamo_tokens::TokenBlock;

/// Error type for block operations that returns the block to prevent leaks
#[derive(Debug, thiserror::Error)]
pub enum BlockError<B> {
    #[error("Block size mismatch: expected {expected} tokens, got {actual}")]
    BlockSizeMismatch {
        expected: usize,
        actual: usize,
        block: B, // Return the block to prevent leaks
    },
}

/// Policy for handling duplicate blocks with the same sequence hash
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDuplicationPolicy {
    /// Allow duplicate blocks enables multiple physical blocks to hold the "same"
    /// data for a single logical block / sequence hash.
    ///
    /// Most LLM inference frameworks use allow block duplication for the G1 / device
    /// memory.
    Allow,
    /// Reject duplicates disables multiple physical blocks to hold the "same".
    /// This essentially dedups the blocks at the time of registration; however, this
    /// adds additional burden to the implementor throw away the duplicate and use the
    /// primary block insteads.
    ///
    /// Internally, KVBM will dedup for G2+ storage layers: host, disk, distributed and
    /// object store.
    Reject,
}

// Generic Block with marker and state markers
#[derive(Debug)]
pub(crate) struct Block<T, State> {
    block_id: BlockId,
    block_size: usize,
    state: State,
    marker: std::marker::PhantomData<T>,
}

/// Trait for types that can be registered and provide block information
pub(crate) trait RegisteredBlock<T>: Send + Sync {
    /// Get the block ID
    fn block_id(&self) -> BlockId;

    /// Get the sequence hash
    fn sequence_hash(&self) -> SequenceHash;

    /// Get the registration handle
    #[expect(dead_code)]
    fn registration_handle(&self) -> &BlockRegistrationHandle;
}
