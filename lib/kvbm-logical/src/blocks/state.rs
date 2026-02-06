// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type-state pattern for block lifecycle with compile-time state enforcement.
//!
//! Blocks transition through states: Reset -> Staged -> Registered -> Reset.
//! The type system prevents invalid state transitions at compile time.

use crate::KvbmSequenceHashProvider;

use super::{Block, BlockError, BlockId, BlockMetadata};
use crate::registry::BlockRegistrationHandle;

use super::{SequenceHash, TokenBlock};
use std::marker::PhantomData;

/// Block identifier type

// State marker types
#[derive(Debug)]
pub struct Reset;

// State-specific data holders
#[derive(Debug)]
pub struct Staged {
    sequence_hash: SequenceHash,
}

#[derive(Debug)]
pub struct Registered {
    sequence_hash: SequenceHash,
    registration_handle: BlockRegistrationHandle,
}

// Implementation for Reset state
impl<T> Block<T, Reset> {
    pub fn new(block_id: BlockId, block_size: usize) -> Self {
        Self {
            block_id,
            block_size,
            state: Reset,
            marker: PhantomData,
        }
    }

    /// Transition from Reset to Staged via a TokenBlock.
    /// Computes the sequence hash from the token block and stores only the hash.
    pub fn complete(
        self,
        token_block: &TokenBlock,
    ) -> Result<Block<T, Staged>, BlockError<Block<T, Reset>>> {
        if token_block.block_size() != self.block_size {
            return Err(BlockError::BlockSizeMismatch {
                expected: self.block_size,
                actual: token_block.block_size(),
                block: self, // Return the block to prevent leaks
            });
        }

        Ok(self.stage(token_block.kvbm_sequence_hash()))
    }

    /// Stage a block directly with a known sequence hash (no TokenBlock needed).
    /// Used by the mutable-block path.
    pub fn stage(self, sequence_hash: SequenceHash) -> Block<T, Staged> {
        Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Staged { sequence_hash },
            marker: PhantomData,
        }
    }

    pub fn reset(self) -> Block<T, Reset> {
        self // Already in reset state
    }
}

impl<T: BlockMetadata> Block<T, Reset> {
    pub(crate) fn register_with_handle(
        self,
        registration_handle: BlockRegistrationHandle,
    ) -> Block<T, Registered> {
        into_registered(self.block_id, self.block_size, registration_handle)
    }
}

// Implementation for Staged state
impl<T: BlockMetadata> Block<T, Staged> {
    pub(crate) fn register_with_handle(
        self,
        registration_handle: BlockRegistrationHandle,
    ) -> Block<T, Registered> {
        into_registered(self.block_id, self.block_size, registration_handle)
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.state.sequence_hash
    }

    pub fn reset(self) -> Block<T, Reset> {
        Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Reset,
            marker: PhantomData,
        }
    }
}

/// Single call site for mark_present - creates a Registered block from any prior state.
fn into_registered<T: BlockMetadata>(
    block_id: BlockId,
    block_size: usize,
    registration_handle: BlockRegistrationHandle,
) -> Block<T, Registered> {
    registration_handle.mark_present::<T>();

    Block {
        block_id,
        block_size,
        state: Registered {
            sequence_hash: registration_handle.seq_hash(),
            registration_handle,
        },
        marker: PhantomData,
    }
}

// Implementation for Registered state
impl<T: BlockMetadata> Block<T, Registered> {
    pub fn sequence_hash(&self) -> SequenceHash {
        self.state.sequence_hash
    }

    pub(crate) fn registration_handle(&self) -> &BlockRegistrationHandle {
        &self.state.registration_handle
    }

    pub fn reset(self) -> Block<T, Reset> {
        // Mark absence when destroying Block<T, Registered>
        self.state.registration_handle.mark_absent::<T>();

        // Drop the registration handle
        Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Reset,
            marker: PhantomData,
        }
    }
}

// Common methods for all states
impl<T, State> Block<T, State> {
    #[inline]
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    #[inline]
    #[allow(dead_code)]
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}
