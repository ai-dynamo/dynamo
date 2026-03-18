// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::DashMap;
use std::hash::Hash;

use crate::{PositionalHash, PositionalSequenceHash};

/// Positionally sparse radix tree for efficient indexing of [PositionalSequenceHashes][`crate::PositionalSequenceHash`].
#[derive(Clone)]
pub struct PositionalRadixTree<V, K = PositionalSequenceHash>
where
    K: PositionalHash + Hash + Eq + Clone,
{
    map: DashMap<u64, DashMap<K, V>>,
}

impl<V, K> PositionalRadixTree<V, K>
where
    K: PositionalHash + Hash + Eq + Clone,
{
    /// Creates a new empty [`PositionalRadixTree`].
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Provides the entry for the key at the given position.
    pub fn prefix(&self, key: &K) -> dashmap::mapref::one::RefMut<'_, u64, DashMap<K, V>> {
        let position = key.position();
        self.map.entry(position).or_default()
    }

    /// Provides the sub-map for all entries at the given position.
    pub fn position(
        &self,
        position: u64,
    ) -> Option<dashmap::mapref::one::RefMut<'_, u64, DashMap<K, V>>> {
        self.map.get_mut(&position)
    }

    /// Returns the number of entries in the [`PositionalRadixTree`].
    pub fn len(&self) -> usize {
        if self.map.is_empty() {
            return 0;
        }
        self.map.iter().map(|level| level.len()).sum()
    }

    /// Returns true if the [`PositionalRadixTree`] is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<V, K> Default for PositionalRadixTree<V, K>
where
    K: PositionalHash + Hash + Eq + Clone,
{
    fn default() -> Self {
        Self {
            map: DashMap::new(),
        }
    }
}
