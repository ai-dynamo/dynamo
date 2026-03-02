// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A no-op inactive pool backend where `should_reset` always returns `true`.
//!
//! When this backend is used, blocks dropped from the active pool are
//! immediately reset and returned to the reset pool — they never enter
//! the inactive pool. All storage methods are unreachable or return empty
//! results because no block can ever be inserted.

use super::super::{Block, BlockMetadata, InactivePoolBackend, Registered, SequenceHash};

/// A no-op [`InactivePoolBackend`] that resets every block on return.
///
/// Because [`should_reset`](InactivePoolBackend::should_reset) always returns
/// `true`, blocks are never inserted into this backend. This effectively
/// disables the inactive pool — every dropped `ImmutableBlock` goes straight
/// back to the reset pool.
///
/// # Example
///
/// ```rust,ignore
/// use kvbm_logical::ext::ResetInactiveBlocksBackend;
///
/// let manager = BlockManager::<MyMeta>::builder()
///     .block_count(64)
///     .block_size(16)
///     .registry(registry)
///     .with_inactive_backend(ResetInactiveBlocksBackend)
///     .build()
///     .expect("build failed");
/// ```
pub struct ResetInactiveBlocksBackend;

impl<T: BlockMetadata> InactivePoolBackend<T> for ResetInactiveBlocksBackend {
    fn find_matches(&mut self, _hashes: &[SequenceHash], _touch: bool) -> Vec<Block<T, Registered>> {
        Vec::new()
    }

    fn scan_matches(
        &mut self,
        _hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, Block<T, Registered>)> {
        Vec::new()
    }

    fn allocate(&mut self, _count: usize) -> Vec<Block<T, Registered>> {
        Vec::new()
    }

    fn insert(&mut self, _block: Block<T, Registered>) {
        unreachable!("ResetInactiveBlocksBackend::should_reset always returns true; insert should never be called")
    }

    fn len(&self) -> usize {
        0
    }

    fn has_block(&self, _seq_hash: SequenceHash) -> bool {
        false
    }

    fn should_reset(&self, _block: &Block<T, Registered>) -> bool {
        true
    }
}
