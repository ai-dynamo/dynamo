// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type-state pattern for block lifecycle with compile-time state enforcement.
//!
//! Blocks transition through states: Reset → Complete → Registered → Reset.
//! The type system prevents invalid state transitions at compile time.

use super::registry::BlockRegistrationHandle;
use super::{Block, BlockError, BlockId, BlockMetadata};

use super::{SequenceHash, TokenBlock};
use std::marker::PhantomData;

/// Block identifier type

// State marker types
#[derive(Debug)]
pub struct Reset;

// State-specific data holders
#[derive(Debug)]
pub struct Complete {
    token_block: TokenBlock,
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

    pub fn complete(
        self,
        token_block: TokenBlock,
    ) -> Result<Block<T, Complete>, BlockError<Block<T, Reset>>> {
        if token_block.block_size() != self.block_size {
            return Err(BlockError::BlockSizeMismatch {
                expected: self.block_size,
                actual: token_block.block_size(),
                block: self, // Return the block to prevent leaks
            });
        }

        Ok(Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Complete { token_block },
            marker: PhantomData,
        })
    }

    #[expect(dead_code)]
    pub fn reset(self) -> Block<T, Reset> {
        self // Already in reset state
    }
}

impl<T: BlockMetadata> Block<T, Reset> {
    pub(crate) fn register_with_handle(
        self,
        registration_handle: BlockRegistrationHandle,
    ) -> Block<T, Registered> {
        registration_handle.mark_present::<T>();

        Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Registered {
                sequence_hash: registration_handle.seq_hash(),
                registration_handle,
            },
            marker: PhantomData,
        }
    }
}

// Implementation for Complete state
impl<T: BlockMetadata> Block<T, Complete> {
    pub fn register(self, registration_handle: BlockRegistrationHandle) -> Block<T, Registered> {
        // Mark presence when creating Block<T, Registered>
        registration_handle.mark_present::<T>();

        Block {
            block_id: self.block_id,
            block_size: self.block_size,
            state: Registered {
                sequence_hash: self.state.token_block.positional_sequence_hash(),
                registration_handle,
            },
            marker: PhantomData,
        }
    }

    pub fn token_block(&self) -> &TokenBlock {
        &self.state.token_block
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.state.token_block.positional_sequence_hash()
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
