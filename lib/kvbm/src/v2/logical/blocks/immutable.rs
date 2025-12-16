// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for immutable and weak block references

use super::{BlockId, BlockMetadata, BlockRegistrationHandle, RegisteredBlock, SequenceHash};

use std::sync::{Arc, Weak};

/// Type alias for upgrade function used in ImmutableBlock and WeakBlock
type UpgradeFn<T> = Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>;

/// RAII guard for registered blocks with upgrade capability
pub struct ImmutableBlock<T: BlockMetadata> {
    block: Arc<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
}

/// Weak reference to a registered block with upgrade capability
#[derive(Clone)]
pub struct WeakBlock<T: BlockMetadata> {
    sequence_hash: SequenceHash,
    block: Weak<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
}

impl<T: BlockMetadata> ImmutableBlock<T> {
    /// Create a new ImmutableBlock with an upgrade function
    pub(crate) fn new(block: Arc<dyn RegisteredBlock<T>>, upgrade_fn: UpgradeFn<T>) -> Self {
        Self { block, upgrade_fn }
    }

    /// Downgrade to a WeakBlock
    pub fn downgrade(&self) -> WeakBlock<T> {
        WeakBlock {
            sequence_hash: self.sequence_hash(),
            block: Arc::downgrade(&self.block),
            upgrade_fn: self.upgrade_fn.clone(),
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block.block_id()
    }

    /// Get the sequence hash
    pub fn sequence_hash(&self) -> SequenceHash {
        self.block.sequence_hash()
    }

    pub(crate) fn registration_handle(&self) -> BlockRegistrationHandle {
        self.block.registration_handle().clone()
    }
}

impl<T: BlockMetadata> std::fmt::Debug for ImmutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImmutableBlock")
            .field("block_id", &self.block_id())
            .field("sequence_hash", &self.sequence_hash())
            .finish()
    }
}

impl<T: BlockMetadata> WeakBlock<T> {
    /// Try to upgrade this WeakBlock back to an ImmutableBlock
    pub fn upgrade(&self) -> Option<ImmutableBlock<T>> {
        // First try to upgrade the weak reference directly
        if let Some(block) = self.block.upgrade() {
            return Some(ImmutableBlock::new(block, self.upgrade_fn.clone()));
        }

        // If that fails, use the upgrade function to search for the block
        if let Some(block) = (self.upgrade_fn)(self.sequence_hash) {
            return Some(ImmutableBlock::new(block, self.upgrade_fn.clone()));
        }

        None
    }

    /// Get the sequence hash
    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl<T: BlockMetadata> std::fmt::Debug for WeakBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakBlock")
            .field("sequence_hash", &self.sequence_hash())
            .finish()
    }
}
