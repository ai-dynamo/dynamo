// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eviction policies.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::storage::Storage;

/// Eviction policy wrapping a storage backend.
pub trait Eviction<K, V>: Storage<K, V> {
    fn evict(&self, count: usize) -> Vec<K>;
    fn capacity(&self) -> usize;
}

/// No eviction - storage grows unbounded.
pub struct NoEviction<S> {
    inner: S,
}

impl<S> NoEviction<S> {
    pub fn new(storage: S) -> Self {
        Self { inner: storage }
    }
}

impl<K, V, S: Storage<K, V>> Storage<K, V> for NoEviction<S> {
    fn insert(&self, key: K, value: V) {
        self.inner.insert(key, value);
    }

    fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    fn remove(&self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) {
        self.inner.clear();
    }
}

impl<K, V, S: Storage<K, V>> Eviction<K, V> for NoEviction<S> {
    fn evict(&self, _count: usize) -> Vec<K> {
        Vec::new()
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }
}

/// Entry for ordering in eviction set.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EvictionEntry<K: Ord + Copy> {
    priority: i64,
    insertion_id: u64,
    key: K,
}

/// Tail-first eviction using parent tracking from metadata.
///
/// Evicts deepest nodes first to avoid orphaning children.
/// Parent relationships are tracked separately from storage.
pub struct TailEviction<K, V, S>
where
    K: Eq + Hash + Ord + Copy + Send + Sync,
    V: Clone + Send + Sync,
    S: Storage<K, V>,
{
    inner: S,
    capacity: usize,
    parents: RwLock<HashMap<K, Option<K>>>,
    children: RwLock<HashMap<K, HashSet<K>>>,
    depths: RwLock<HashMap<K, u32>>,
    leaves: RwLock<BTreeSet<EvictionEntry<K>>>,
    insertion_counter: AtomicU64,
    _phantom: PhantomData<V>,
}

impl<K, V, S> TailEviction<K, V, S>
where
    K: Eq + Hash + Ord + Copy + Send + Sync,
    V: Clone + Send + Sync,
    S: Storage<K, V>,
{
    pub fn new(storage: S, capacity: usize) -> Self {
        Self {
            inner: storage,
            capacity,
            parents: RwLock::new(HashMap::new()),
            children: RwLock::new(HashMap::new()),
            depths: RwLock::new(HashMap::new()),
            leaves: RwLock::new(BTreeSet::new()),
            insertion_counter: AtomicU64::new(0),
            _phantom: PhantomData,
        }
    }

    /// Insert with parent tracking for eviction ordering.
    pub fn insert_with_parent(&self, key: K, value: V, parent: Option<K>) {
        let mut parents = self.parents.write();
        let mut children = self.children.write();
        let mut depths = self.depths.write();
        let mut leaves = self.leaves.write();

        let depth = match parent {
            Some(parent_key) => {
                leaves.retain(|e| e.key != parent_key);
                children.entry(parent_key).or_default().insert(key);
                depths.get(&parent_key).copied().unwrap_or(0) + 1
            }
            None => 0,
        };

        parents.insert(key, parent);
        depths.insert(key, depth);
        leaves.insert(EvictionEntry {
            priority: -(depth as i64),
            insertion_id: self.insertion_counter.fetch_add(1, Ordering::Relaxed),
            key,
        });

        drop(parents);
        drop(children);
        drop(depths);
        drop(leaves);

        self.inner.insert(key, value);
        self.maybe_evict();
    }

    fn maybe_evict(&self) {
        while self.inner.len() > self.capacity {
            if self.evict_one().is_none() {
                break;
            }
        }
    }

    fn evict_one(&self) -> Option<K> {
        let key = {
            let leaves = self.leaves.read();
            leaves.iter().next().map(|e| e.key)?
        };

        self.remove(&key);
        Some(key)
    }
}

impl<K, V, S> Storage<K, V> for TailEviction<K, V, S>
where
    K: Eq + Hash + Ord + Copy + Send + Sync,
    V: Clone + Send + Sync,
    S: Storage<K, V>,
{
    fn insert(&self, key: K, value: V) {
        self.insert_with_parent(key, value, None);
    }

    fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    fn remove(&self, key: &K) -> Option<V> {
        let mut parents = self.parents.write();
        let mut children = self.children.write();
        let mut depths = self.depths.write();
        let mut leaves = self.leaves.write();

        let _depth = depths.remove(key).unwrap_or(0);
        let parent = parents.remove(key).flatten();

        leaves.retain(|e| &e.key != key);

        if let Some(parent_key) = parent {
            if let Some(parent_children) = children.get_mut(&parent_key) {
                parent_children.remove(key);
                if parent_children.is_empty() {
                    children.remove(&parent_key);
                    if let Some(&parent_depth) = depths.get(&parent_key) {
                        leaves.insert(EvictionEntry {
                            priority: -(parent_depth as i64),
                            insertion_id: self.insertion_counter.fetch_add(1, Ordering::Relaxed),
                            key: parent_key,
                        });
                    }
                }
            }
        }

        if let Some(my_children) = children.remove(key) {
            for child in my_children {
                if let Some(child_parent) = parents.get_mut(&child) {
                    *child_parent = None;
                }
                if let Some(child_depth) = depths.get_mut(&child) {
                    *child_depth = 0;
                }
            }
        }

        drop(parents);
        drop(children);
        drop(depths);
        drop(leaves);

        self.inner.remove(key)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) {
        self.parents.write().clear();
        self.children.write().clear();
        self.depths.write().clear();
        self.leaves.write().clear();
        self.inner.clear();
    }
}

impl<K, V, S> Eviction<K, V> for TailEviction<K, V, S>
where
    K: Eq + Hash + Ord + Copy + Send + Sync,
    V: Clone + Send + Sync,
    S: Storage<K, V>,
{
    fn evict(&self, count: usize) -> Vec<K> {
        let mut evicted = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(key) = self.evict_one() {
                evicted.push(key);
            } else {
                break;
            }
        }
        evicted
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::storage::HashMapStorage;

    #[test]
    fn test_tail_eviction_basic() {
        let storage = HashMapStorage::new();
        let evictable = TailEviction::new(storage, 100);

        evictable.insert(1, 100);
        evictable.insert(2, 200);

        assert_eq!(evictable.get(&1), Some(100));
        assert_eq!(evictable.get(&2), Some(200));
        assert_eq!(evictable.len(), 2);
    }

    #[test]
    fn test_tail_eviction_with_parents() {
        let storage = HashMapStorage::new();
        let evictable = TailEviction::new(storage, 100);

        evictable.insert_with_parent(1, 100, None);
        evictable.insert_with_parent(2, 200, Some(1));
        evictable.insert_with_parent(3, 300, Some(2));

        assert_eq!(evictable.len(), 3);

        let evicted = evictable.evict(1);
        assert_eq!(evicted, vec![3]);
        assert_eq!(evictable.len(), 2);
        assert!(evictable.contains(&1));
        assert!(evictable.contains(&2));
        assert!(!evictable.contains(&3));
    }

    #[test]
    fn test_tail_eviction_capacity() {
        let storage = HashMapStorage::new();
        let evictable = TailEviction::new(storage, 3);

        evictable.insert_with_parent(1, 100, None);
        evictable.insert_with_parent(2, 200, Some(1));
        evictable.insert_with_parent(3, 300, Some(2));
        assert_eq!(evictable.len(), 3);

        evictable.insert_with_parent(4, 400, Some(3));
        assert_eq!(evictable.len(), 3);

        assert!(evictable.contains(&1));
        assert!(evictable.contains(&2));
    }

    #[test]
    fn test_remove_makes_parent_leaf() {
        let storage = HashMapStorage::new();
        let evictable = TailEviction::new(storage, 100);

        evictable.insert_with_parent(1, 100, None);
        evictable.insert_with_parent(2, 200, Some(1));

        evictable.remove(&2);

        let evicted = evictable.evict(1);
        assert_eq!(evicted, vec![1]);
    }
}
