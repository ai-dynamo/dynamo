// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage trait and implementations.

use std::collections::HashMap;
use std::hash::Hash;

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
}
