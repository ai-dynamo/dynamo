// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::DashMap;

use crate::PositionalSequenceHash;

/// Positionally sparse radix tree for efficient indexing of [PositionalSequenceHashes][`crate::PositionalSequenceHash`].
#[derive(Clone)]
pub struct PositionalRadixTree<T> {
    map: DashMap<u64, DashMap<PositionalSequenceHash, T>>,
}

impl<T> PositionalRadixTree<T> {
    /// Creates a new empty [`PositionalRadixTree`].
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Provides the entry for the [`PositionalSequenceHash`] at the given position.
    pub fn prefix(
        &self,
        seq_hash: &PositionalSequenceHash,
    ) -> dashmap::mapref::one::RefMut<'_, u64, DashMap<PositionalSequenceHash, T>> {
        let position = seq_hash.position();
        self.map.entry(position).or_default()
    }

    /// Provides the sub-map for all [`PositionalSequenceHash`] entries at the given position.
    pub fn position(
        &self,
        position: u64,
    ) -> Option<dashmap::mapref::one::RefMut<'_, u64, DashMap<PositionalSequenceHash, T>>> {
        self.map.get_mut(&position)
    }

    /// Returns the number of entries [`PositionalSequenceHashes`][`crate::PositionalSequenceHash`] in the [`PositionalRadixTree`].
    pub fn len(&self) -> usize {
        if self.map.is_empty() {
            return 0;
        }
        self.map.iter().map(|level| level.len()).sum()
    }

    /// Returns true if the [`PositionalRadixTree`] is empty of [`PositionalSequenceHashes`][`crate::PositionalSequenceHash`]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for PositionalRadixTree<T> {
    fn default() -> Self {
        Self {
            map: DashMap::new(),
        }
    }
}
