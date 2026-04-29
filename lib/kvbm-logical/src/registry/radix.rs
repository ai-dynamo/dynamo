// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Positionally-sparse radix tree used by [`super::BlockRegistry`].
//!
//! This was previously `dynamo_tokens::PositionalRadixTree`. Moved here as a
//! `pub(crate)` module because the registry is its only consumer — keeping it
//! in `dynamo-tokens` exposed two extra public types (`PositionalRadixTree`,
//! `PositionalHash`) that no other crate ever imported.

use dashmap::DashMap;
use std::hash::Hash;

/// Trait for hashes that include position information.
pub(crate) trait PositionalHash {
    /// Returns the position associated with the hash.
    fn position(&self) -> u64;
}

/// Positionally sparse radix tree for efficient indexing of position-keyed hashes.
#[derive(Clone)]
pub(crate) struct PositionalRadixTree<V, K>
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
    pub(crate) fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Provides the entry for the key at the given position.
    pub(crate) fn prefix(&self, key: &K) -> dashmap::mapref::one::RefMut<'_, u64, DashMap<K, V>> {
        let position = key.position();
        self.map.entry(position).or_default()
    }

    /// Provides the sub-map for all entries at the given position.
    #[allow(dead_code)]
    pub(crate) fn position(
        &self,
        position: u64,
    ) -> Option<dashmap::mapref::one::RefMut<'_, u64, DashMap<K, V>>> {
        self.map.get_mut(&position)
    }

    /// Returns the number of entries in the [`PositionalRadixTree`].
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        if self.map.is_empty() {
            return 0;
        }
        self.map.iter().map(|level| level.len()).sum()
    }

    /// Returns true if the [`PositionalRadixTree`] is empty.
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
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

impl PositionalHash for dynamo_tokens::PositionalLineageHash {
    fn position(&self) -> u64 {
        Self::position(self)
    }
}
