// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::locality::LocalityProvider;

use super::*;

/// Manages active blocks being used by sequences
pub struct ActiveBlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    pub(super) map: HashMap<SequenceHash, Weak<MutableBlock<S, L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Default for ActiveBlockPool<S, L, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ActiveBlockPool<S, L, M> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        mut block: MutableBlock<S, L, M>,
    ) -> Result<ImmutableBlock<S, L, M>, BlockPoolError> {
        if !block.state().is_registered() {
            return Err(BlockPoolError::InvalidMutableBlock(
                "block is not registered".to_string(),
            ));
        }

        let sequence_hash = block.sequence_hash().map_err(|_| {
            BlockPoolError::InvalidMutableBlock("block has no sequence hash".to_string())
        })?;

        // Set the parent of the block if it has one.
        // This is needed to ensure the lifetime of the parent is at least as long as the child.
        if let Ok(Some(parent)) = block.parent_sequence_hash()
            && let Some(parent_block) = self.match_sequence_hash(parent)
        {
            block.set_parent(parent_block.mutable_block().clone());
        }

        let shared = Arc::new(block);

        match self.map.entry(sequence_hash) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let weak = entry.get();
                if let Some(arc) = weak.upgrade() {
                    Ok(ImmutableBlock::new(arc))
                } else {
                    // Weak reference is no longer alive, update it in the map
                    entry.insert(Arc::downgrade(&shared));
                    Ok(ImmutableBlock::new(shared))
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Arc::downgrade(&shared));
                Ok(ImmutableBlock::new(shared))
            }
        }
    }

    pub fn remove(&mut self, block: &mut Block<S, L, M>) {
        if let Ok(sequence_hash) = block.sequence_hash()
            && let Some(weak) = self.map.get(&sequence_hash)
        {
            if let Some(_arc) = weak.upgrade() {
                block.reset();
                return;
            }
            self.map.remove(&sequence_hash);
        }
    }

    pub fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<ImmutableBlock<S, L, M>> {
        if let Some(weak) = self.map.get(&sequence_hash) {
            if let Some(arc) = weak.upgrade() {
                Some(ImmutableBlock::new(arc))
            } else {
                // Weak reference is no longer alive, remove it from the map
                self.map.remove(&sequence_hash);
                None
            }
        } else {
            None
        }
    }

    pub fn status(&self) -> usize {
        self.map.keys().len()
    }
}
