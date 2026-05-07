// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `BlockManagerApi` — public lifecycle surface of [`BlockManager`].
//!
//! Implemented by [`BlockManager`] via simple delegation to the inherent
//! methods. Consumers that want to abstract over the manager (mocks,
//! decorators, alternate orchestrators) can depend on the trait instead
//! of the concrete type.

use std::collections::HashMap;

use crate::blocks::{BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock};
use crate::pools::SequenceHash;

use super::BlockManager;

/// Public lifecycle surface of [`BlockManager`].
///
/// See the inherent methods on [`BlockManager`] for behaviour, semantics,
/// and complexity notes — the trait is a thin re-export of those signatures.
pub trait BlockManagerApi<T: BlockMetadata> {
    fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>>;

    fn register_block(&self, block: CompleteBlock<T>) -> ImmutableBlock<T>;

    fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>>;

    fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>>;

    fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>>;

    fn total_blocks(&self) -> usize;

    fn available_blocks(&self) -> usize;

    fn block_size(&self) -> usize;
}

impl<T: BlockMetadata + Sync> BlockManagerApi<T> for BlockManager<T> {
    fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        BlockManager::allocate_blocks(self, count)
    }

    fn register_block(&self, block: CompleteBlock<T>) -> ImmutableBlock<T> {
        BlockManager::register_block(self, block)
    }

    fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        BlockManager::register_blocks(self, blocks)
    }

    fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        BlockManager::match_blocks(self, seq_hash)
    }

    fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        BlockManager::scan_matches(self, seq_hashes, touch)
    }

    fn total_blocks(&self) -> usize {
        BlockManager::total_blocks(self)
    }

    fn available_blocks(&self) -> usize {
        BlockManager::available_blocks(self)
    }

    fn block_size(&self) -> usize {
        BlockManager::block_size(self)
    }
}
