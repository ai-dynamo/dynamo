// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage trait and implementations.

use std::collections::HashMap;
use std::hash::Hash;

use dashmap::DashMap;
use parking_lot::RwLock;

/// Storage for key-value pairs.
pub trait Storage<K, V>: Send + Sync {
    fn insert(&self, key: K, value: V);
    fn get(&self, key: &K) -> Option<V>;
    fn contains(&self, key: &K) -> bool;
    fn remove(&self, key: &K) -> Option<V>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&self);
}

/// HashMap-based storage.
pub struct HashMapStorage<K, V> {
    map: RwLock<HashMap<K, V>>,
}

impl<K, V> HashMapStorage<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity)),
        }
    }
}

impl<K, V> Default for HashMapStorage<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Storage<K, V> for HashMapStorage<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    fn insert(&self, key: K, value: V) {
        self.map.write().insert(key, value);
    }

    fn get(&self, key: &K) -> Option<V> {
        self.map.read().get(key).cloned()
    }

    fn contains(&self, key: &K) -> bool {
        self.map.read().contains_key(key)
    }

    fn remove(&self, key: &K) -> Option<V> {
        self.map.write().remove(key)
    }

    fn len(&self) -> usize {
        self.map.read().len()
    }

    fn clear(&self) {
        self.map.write().clear();
    }
}

/// Trait for keys that have an extractable position for radix organization.
pub trait PositionalStorageKey: Eq + Hash + Clone + Send + Sync {
    /// Extract the position/prefix used for first-level partitioning.
    fn position(&self) -> u64;
}

/// Radix-style storage using a two-level DashMap for concurrent access.
///
/// Organizes entries by position first, then by full key within each position.
/// This provides:
/// - Efficient prefix/position-based lookups
/// - High concurrency via sharded DashMap structure
/// - Natural grouping for positional data (e.g., sequence positions)
///
/// # Example
/// ```text
/// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// struct MyKey { position: u64, hash: u64 }
///
/// impl PositionalStorageKey for MyKey {
///     fn position(&self) -> u64 { self.position }
/// }
///
/// let storage: RadixStorage<MyKey, String> = RadixStorage::new();
/// storage.insert(MyKey { position: 0, hash: 123 }, "value".to_string());
/// ```
pub struct RadixStorage<K, V> {
    /// Two-level map: position -> (key -> value)
    map: DashMap<u64, DashMap<K, V>>,
}

impl<K, V> RadixStorage<K, V>
where
    K: PositionalStorageKey,
{
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Get the sub-map for a specific position.
    pub fn position(
        &self,
        position: u64,
    ) -> Option<dashmap::mapref::one::Ref<'_, u64, DashMap<K, V>>> {
        self.map.get(&position)
    }

    /// Get or create the sub-map for a key's position.
    pub fn prefix(&self, key: &K) -> dashmap::mapref::one::RefMut<'_, u64, DashMap<K, V>> {
        self.map.entry(key.position()).or_default()
    }

    /// Iterate over all positions that have entries.
    pub fn positions(&self) -> Vec<u64> {
        self.map.iter().map(|entry| *entry.key()).collect()
    }

    /// Get the count of entries at a specific position.
    pub fn position_len(&self, position: u64) -> usize {
        self.map.get(&position).map(|m| m.len()).unwrap_or(0)
    }
}

impl<K, V> Default for RadixStorage<K, V>
where
    K: PositionalStorageKey,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Storage<K, V> for RadixStorage<K, V>
where
    K: PositionalStorageKey + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn insert(&self, key: K, value: V) {
        let position = key.position();
        self.map.entry(position).or_default().insert(key, value);
    }

    fn get(&self, key: &K) -> Option<V> {
        let position = key.position();
        self.map.get(&position)?.get(key).map(|v| v.clone())
    }

    fn contains(&self, key: &K) -> bool {
        let position = key.position();
        self.map
            .get(&position)
            .map(|m| m.contains_key(key))
            .unwrap_or(false)
    }

    fn remove(&self, key: &K) -> Option<V> {
        let position = key.position();
        let inner = self.map.get(&position)?;
        let removed = inner.remove(key).map(|(_, v)| v);

        // Clean up empty position maps
        if inner.is_empty() {
            drop(inner);
            self.map.remove(&position);
        }

        removed
    }

    fn len(&self) -> usize {
        self.map.iter().map(|entry| entry.len()).sum()
    }

    fn clear(&self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap_storage() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();

        storage.insert(1, 100);
        storage.insert(2, 200);

        assert_eq!(storage.get(&1), Some(100));
        assert_eq!(storage.get(&3), None);
        assert!(storage.contains(&1));
        assert!(!storage.contains(&3));
        assert_eq!(storage.len(), 2);

        storage.remove(&1);
        assert_eq!(storage.len(), 1);
        assert!(!storage.contains(&1));

        storage.clear();
        assert!(storage.is_empty());
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct TestPositionalKey {
        position: u64,
        hash: u64,
    }

    impl PositionalStorageKey for TestPositionalKey {
        fn position(&self) -> u64 {
            self.position
        }
    }

    #[test]
    fn test_radix_storage_basic() {
        let storage: RadixStorage<TestPositionalKey, u64> = RadixStorage::new();

        let key1 = TestPositionalKey {
            position: 0,
            hash: 100,
        };
        let key2 = TestPositionalKey {
            position: 0,
            hash: 200,
        };
        let key3 = TestPositionalKey {
            position: 1,
            hash: 100,
        };

        storage.insert(key1, 1);
        storage.insert(key2, 2);
        storage.insert(key3, 3);

        assert_eq!(storage.get(&key1), Some(1));
        assert_eq!(storage.get(&key2), Some(2));
        assert_eq!(storage.get(&key3), Some(3));
        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_radix_storage_position_grouping() {
        let storage: RadixStorage<TestPositionalKey, u64> = RadixStorage::new();

        // Insert keys at different positions
        for pos in 0..3 {
            for hash in 0..5 {
                storage.insert(
                    TestPositionalKey {
                        position: pos,
                        hash,
                    },
                    pos * 10 + hash,
                );
            }
        }

        assert_eq!(storage.len(), 15);
        assert_eq!(storage.position_len(0), 5);
        assert_eq!(storage.position_len(1), 5);
        assert_eq!(storage.position_len(2), 5);
        assert_eq!(storage.position_len(99), 0);

        let mut positions = storage.positions();
        positions.sort();
        assert_eq!(positions, vec![0, 1, 2]);
    }

    #[test]
    fn test_radix_storage_remove() {
        let storage: RadixStorage<TestPositionalKey, u64> = RadixStorage::new();

        let key1 = TestPositionalKey {
            position: 0,
            hash: 100,
        };
        let key2 = TestPositionalKey {
            position: 0,
            hash: 200,
        };

        storage.insert(key1, 1);
        storage.insert(key2, 2);

        assert_eq!(storage.remove(&key1), Some(1));
        assert_eq!(storage.len(), 1);
        assert!(!storage.contains(&key1));
        assert!(storage.contains(&key2));

        // Remove last key at position 0, should clean up the position map
        assert_eq!(storage.remove(&key2), Some(2));
        assert_eq!(storage.len(), 0);
        assert!(storage.position(0).is_none());
    }

    #[test]
    fn test_radix_storage_clear() {
        let storage: RadixStorage<TestPositionalKey, u64> = RadixStorage::new();

        for pos in 0..10 {
            storage.insert(
                TestPositionalKey {
                    position: pos,
                    hash: 0,
                },
                pos,
            );
        }

        assert_eq!(storage.len(), 10);
        storage.clear();
        assert!(storage.is_empty());
        assert!(storage.positions().is_empty());
    }
}
