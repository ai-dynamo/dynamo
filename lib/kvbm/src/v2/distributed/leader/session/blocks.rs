// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII block holding for sessions.
//!
//! This module provides [`BlockHolder<T>`], a tier-agnostic container for
//! holding blocks during session operations. Blocks are automatically
//! released when the holder is dropped.
//!
//! # Design Philosophy
//!
//! `BlockHolder` is intentionally simple - it's pure RAII with no staging logic.
//! This allows flexibility for different staging patterns:
//! - G3→G2 staging
//! - G4→G2 staging
//! - G1→G2 staging
//! - G2→G3 offload
//!
//! The caller decides when and how to stage; `BlockHolder` just holds.

use crate::v2::SequenceHash;
use crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock};

/// RAII block holder - tier-agnostic, just holds blocks.
///
/// # Type Parameter
///
/// `T` is the tier metadata type (e.g., `G2`, `G3`). It must implement
/// `BlockMetadata` which is `Clone + Send + Sync + 'static`.
///
/// # RAII Semantics
///
/// When `BlockHolder` is dropped, all held blocks are released. This ensures
/// blocks don't leak even if session handling panics.
///
/// # Example
///
/// ```ignore
/// // Create holder with searched blocks
/// let mut holder = BlockHolder::new(g2_blocks);
///
/// // Check what we have
/// println!("Holding {} blocks", holder.count());
///
/// // Release some blocks (e.g., after RDMA pull)
/// holder.release(&pulled_hashes);
///
/// // Holder drops here, releasing any remaining blocks
/// ```
#[derive(Debug)]
pub struct BlockHolder<T: BlockMetadata> {
    blocks: Vec<ImmutableBlock<T>>,
}

impl<T: BlockMetadata> BlockHolder<T> {
    /// Create a new `BlockHolder` with the given blocks.
    pub fn new(blocks: Vec<ImmutableBlock<T>>) -> Self {
        Self { blocks }
    }

    /// Create an empty `BlockHolder`.
    pub fn empty() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Get a reference to the held blocks.
    pub fn blocks(&self) -> &[ImmutableBlock<T>] {
        &self.blocks
    }

    /// Get the number of held blocks.
    pub fn count(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the holder is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Add blocks to this holder.
    pub fn extend(&mut self, blocks: impl IntoIterator<Item = ImmutableBlock<T>>) {
        self.blocks.extend(blocks);
    }

    /// Release blocks matching the given sequence hashes.
    ///
    /// Removes blocks from the holder whose sequence hash is in `hashes`.
    /// The blocks are dropped, releasing their references.
    pub fn release(&mut self, hashes: &[SequenceHash]) {
        self.blocks.retain(|b| !hashes.contains(&b.sequence_hash()));
    }

    /// Retain only blocks matching the given sequence hashes.
    ///
    /// Removes blocks from the holder whose sequence hash is NOT in `hashes`.
    /// The removed blocks are dropped, releasing their references.
    pub fn retain(&mut self, hashes: &[SequenceHash]) {
        self.blocks.retain(|b| hashes.contains(&b.sequence_hash()));
    }

    /// Take all blocks out of this holder.
    ///
    /// The holder becomes empty. Useful for transferring blocks to another
    /// location or for processing before dropping.
    pub fn take_all(&mut self) -> Vec<ImmutableBlock<T>> {
        std::mem::take(&mut self.blocks)
    }

    /// Get sequence hashes of all held blocks.
    pub fn sequence_hashes(&self) -> Vec<SequenceHash> {
        self.blocks.iter().map(|b| b.sequence_hash()).collect()
    }

    /// Find a block by sequence hash.
    pub fn find(&self, hash: &SequenceHash) -> Option<&ImmutableBlock<T>> {
        self.blocks.iter().find(|b| &b.sequence_hash() == hash)
    }

    /// Check if a block with the given hash is held.
    pub fn contains(&self, hash: &SequenceHash) -> bool {
        self.blocks.iter().any(|b| &b.sequence_hash() == hash)
    }

    /// Iterate over held blocks.
    pub fn iter(&self) -> impl Iterator<Item = &ImmutableBlock<T>> {
        self.blocks.iter()
    }
}

impl<T: BlockMetadata> Default for BlockHolder<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: BlockMetadata> FromIterator<ImmutableBlock<T>> for BlockHolder<T> {
    fn from_iter<I: IntoIterator<Item = ImmutableBlock<T>>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<T: BlockMetadata> IntoIterator for BlockHolder<T> {
    type Item = ImmutableBlock<T>;
    type IntoIter = std::vec::IntoIter<ImmutableBlock<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.into_iter()
    }
}

impl<'a, T: BlockMetadata> IntoIterator for &'a BlockHolder<T> {
    type Item = &'a ImmutableBlock<T>;
    type IntoIter = std::slice::Iter<'a, ImmutableBlock<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require test infrastructure to create ImmutableBlock instances.
    // These tests verify the basic container operations.

    #[test]
    fn test_empty_holder() {
        let holder: BlockHolder<()> = BlockHolder::empty();
        assert!(holder.is_empty());
        assert_eq!(holder.count(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let holder: BlockHolder<()> = BlockHolder::default();
        assert!(holder.is_empty());
    }
}
