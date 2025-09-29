// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for registered blocks (primary and duplicate)

use std::sync::Arc;

use super::{
    super::pools::{
        BlockMetadata,
        block::{Block, BlockId, Registered, Reset},
        registry::BlockRegistrationHandle,
    },
    RegisteredBlock,
};
use crate::tokens::SequenceHash;

/// RAII guard for [`Block<T, Registered>`] that automatically returns to RegisteredPool on drop
pub(crate) struct PrimaryBlock<T: BlockMetadata> {
    pub(crate) block: Option<Arc<Block<T, Registered>>>,
    pub(crate) return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
}

/// RAII guard for duplicate blocks that share the same sequence hash as a primary block
pub(crate) struct DuplicateBlock<T: BlockMetadata> {
    pub(crate) block: Option<Block<T, Registered>>,
    pub(crate) return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    pub(crate) _primary: Arc<PrimaryBlock<T>>,
}

impl<T: BlockMetadata> PrimaryBlock<T> {
    /// Create a new PrimaryBlock
    pub(crate) fn new(
        block: Arc<Block<T, Registered>>,
        return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    /// Register this block and get an Arc to the RegisteredBlock trait object
    pub(crate) fn register(self) -> Arc<dyn RegisteredBlock<T>> {
        let block = self.block.clone().unwrap();
        block.registration_handle().attach_block(self)
    }
}

impl<T: BlockMetadata> DuplicateBlock<T> {
    /// Create a new DuplicateBlock
    pub(crate) fn new(
        block: Block<T, Registered>,
        primary: Arc<PrimaryBlock<T>>,
        return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
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
