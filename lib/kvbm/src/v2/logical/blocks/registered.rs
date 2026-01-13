// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for registered blocks (primary and duplicate)

use std::sync::Arc;

use super::{
    Block, BlockId, BlockMetadata, BlockRegistrationHandle, RegisteredBlock, SequenceHash,
    state::{Registered, Reset},
};

/// Type alias for primary block return function
type PrimaryReturnFn<T> = Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>;

/// Type alias for duplicate block return function
type DuplicateReturnFn<T> = Arc<dyn Fn(Block<T, Reset>) + Send + Sync>;

/// RAII guard for [`Block<T, Registered>`] that automatically returns to RegisteredPool on drop
pub(crate) struct PrimaryBlock<T: BlockMetadata> {
    pub(crate) block: Option<Arc<Block<T, Registered>>>,
    pub(crate) return_fn: PrimaryReturnFn<T>,
}

/// RAII guard for duplicate blocks that share the same sequence hash as a primary block
pub(crate) struct DuplicateBlock<T: BlockMetadata> {
    pub(crate) block: Option<Block<T, Registered>>,
    pub(crate) return_fn: DuplicateReturnFn<T>,
    pub(crate) _primary: Arc<PrimaryBlock<T>>,
}

impl<T: BlockMetadata> PrimaryBlock<T> {
    /// Create a new PrimaryBlock
    pub(crate) fn new(block: Arc<Block<T, Registered>>, return_fn: PrimaryReturnFn<T>) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }
}

impl<T: BlockMetadata> DuplicateBlock<T> {
    /// Create a new DuplicateBlock
    pub(crate) fn new(
        block: Block<T, Registered>,
        primary: Arc<PrimaryBlock<T>>,
        return_fn: DuplicateReturnFn<T>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
            _primary: primary,
        }
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for PrimaryBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for DuplicateBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> Drop for PrimaryBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block);
        }
    }
}

impl<T: BlockMetadata> Drop for DuplicateBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block.reset());
        }
    }
}
