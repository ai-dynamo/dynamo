// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eviction policies.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::storage::{FlatStorage, PositionalStorageKey, RadixStorage, Storage};

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

        if let Some(parent_key) = parent
            && let Some(parent_children) = children.get_mut(&parent_key)
        {
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

        if let Some(my_children) = children.remove(key) {
            for child in my_children {
                if let Some(child_parent) = parents.get_mut(&child) {
                    *child_parent = None;
                }
                // Remove old eviction entry with stale priority
                leaves.retain(|e| e.key != child);
                if let Some(child_depth) = depths.get_mut(&child) {
                    *child_depth = 0;
                    // Re-insert with updated priority (depth 0)
                    leaves.insert(EvictionEntry {
                        priority: 0,
                        insertion_id: self.insertion_counter.fetch_add(1, Ordering::Relaxed),
                        key: child,
                    });
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

/// Position-aware eviction over a pluggable storage backend.
///
/// Evicts entries from highest positions first (tail of sequence),
/// with FIFO ordering within each position. This is ideal for KV cache
/// where newer sequence positions are less valuable for prefix matching.
///
/// The storage backend `S` must implement `Storage<K, V>`. Common choices:
/// - `RadixStorage<K, V>` — position-sharded DashMap (legacy)
/// - `FlatStorage<K, V>` — flat DashMap (avoids position-sharding bugs)
///
/// # Example
/// ```text
/// let evictable = PositionalEviction::<Key, Value, FlatStorage<Key, Value>>::with_flat_storage(1000);
/// evictable.insert(key_at_pos_0, value);
/// evictable.insert(key_at_pos_100, value);
/// // Eviction will remove from position 100 first
/// evictable.evict(1);
/// ```
pub struct PositionalEviction<K, V, S = RadixStorage<K, V>>
where
    K: PositionalStorageKey + Ord + Copy,
    V: Clone + Send + Sync,
    S: Storage<K, V>,
{
    inner: S,
    capacity: usize,
    /// Track insertion order within each position: position -> ordered keys
    insertion_order: RwLock<HashMap<u64, Vec<K>>>,
    /// Track which positions have entries, ordered by position (descending for eviction)
    positions: RwLock<BTreeSet<std::cmp::Reverse<u64>>>,
    /// Track last access time for each key (for LRU-style eviction)
    last_access: RwLock<HashMap<K, std::time::Instant>>,
    /// Track access count for each key (for LFU-style eviction)
    access_count: RwLock<HashMap<K, u64>>,
    _phantom: PhantomData<V>,
}

impl<K, V, S> PositionalEviction<K, V, S>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
    S: Storage<K, V>,
{
    pub fn new(storage: S, capacity: usize) -> Self {
        Self {
            inner: storage,
            capacity,
            insertion_order: RwLock::new(HashMap::new()),
            positions: RwLock::new(BTreeSet::new()),
            last_access: RwLock::new(HashMap::new()),
            access_count: RwLock::new(HashMap::new()),
            _phantom: PhantomData,
        }
    }
}

impl<K, V> PositionalEviction<K, V, RadixStorage<K, V>>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create with a `RadixStorage` backend (backward-compatible).
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(RadixStorage::new(), capacity)
    }
}

impl<K, V> PositionalEviction<K, V, FlatStorage<K, V>>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create with a `FlatStorage` backend (no position-sharding).
    pub fn with_flat_storage(capacity: usize) -> Self {
        Self::new(FlatStorage::with_capacity(capacity), capacity)
    }
}

impl<K, V, S> PositionalEviction<K, V, S>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
    S: Storage<K, V>,
{

    /// Record an access for a key (used for LRU/LFU tracking).
    ///
    /// Called when a cache hit occurs (e.g., onboard from G4, or G2 cache hit).
    /// Updates both last_access time and access_count.
    pub fn touch(&self, key: &K) {
        if self.inner.contains(key) {
            let now = std::time::Instant::now();
            self.last_access.write().insert(*key, now);
            *self.access_count.write().entry(*key).or_insert(0) += 1;
        }
    }

    /// Batch touch multiple keys for efficiency.
    pub fn touch_many(&self, keys: &[K]) -> usize {
        let now = std::time::Instant::now();
        let mut touched = 0;
        let mut last_access = self.last_access.write();
        let mut access_count = self.access_count.write();

        for key in keys {
            if self.inner.contains(key) {
                last_access.insert(*key, now);
                *access_count.entry(*key).or_insert(0) += 1;
                touched += 1;
            }
        }
        touched
    }

    /// Get access statistics for a key.
    pub fn get_access_stats(&self, key: &K) -> Option<(std::time::Instant, u64)> {
        let last = self.last_access.read().get(key).copied();
        let count = self.access_count.read().get(key).copied().unwrap_or(0);
        last.map(|t| (t, count))
    }

    fn maybe_evict(&self) {
        self.evict_to_capacity();
    }

    /// Find the index of the LRU key in a list, using access time and count.
    /// Falls back to insertion order (FIFO, index 0) if no access tracking.
    fn find_lru_key(
        keys: &[K],
        last_access: &HashMap<K, std::time::Instant>,
        access_count: &HashMap<K, u64>,
    ) -> usize {
        let mut best_idx = 0;
        let mut best_time: Option<std::time::Instant> = None;
        let mut best_count: u64 = u64::MAX;

        for (idx, key) in keys.iter().enumerate() {
            let time = last_access.get(key).copied();
            let count = access_count.get(key).copied().unwrap_or(0);

            let is_better = match (best_time, time) {
                (None, None) => count < best_count,
                (None, Some(_)) => false, // Prefer keys without access tracking
                (Some(_), None) => true,  // Keys without tracking are evicted first
                (Some(bt), Some(t)) => t < bt || (t == bt && count < best_count),
            };

            if is_better {
                best_idx = idx;
                best_time = time;
                best_count = count;
            }
        }

        best_idx
    }

    /// Evict entries down to capacity in a single batched operation.
    ///
    /// Acquires locks once for the entire eviction pass instead of per-entry.
    /// This reduces lock acquisitions from ~6×N to a constant number.
    fn evict_to_capacity(&self) {
        let excess = self.inner.len().saturating_sub(self.capacity);
        if excess == 0 {
            return;
        }

        // Phase 1: Find all victims under a single lock acquisition
        let victims = {
            let mut positions = self.positions.write();
            let mut insertion_order = self.insertion_order.write();
            let last_access = self.last_access.read();
            let access_count = self.access_count.read();

            let mut victims = Vec::with_capacity(excess);
            let mut remaining = excess;

            while remaining > 0 {
                let highest_pos = match positions.iter().next() {
                    Some(r) => r.0,
                    None => break,
                };

                let keys = match insertion_order.get_mut(&highest_pos) {
                    Some(keys) if !keys.is_empty() => keys,
                    _ => break,
                };

                // Take up to `remaining` keys from this position (LRU order)
                while remaining > 0 && !keys.is_empty() {
                    let best_idx = Self::find_lru_key(keys, &last_access, &access_count);
                    victims.push(keys.remove(best_idx));
                    remaining -= 1;
                }

                if keys.is_empty() {
                    insertion_order.remove(&highest_pos);
                    positions.remove(&std::cmp::Reverse(highest_pos));
                }
            }

            victims
        };
        // All read/write locks dropped here

        if victims.is_empty() {
            return;
        }

        // Phase 2: Clean up access tracking outside of position locks
        {
            let mut la = self.last_access.write();
            let mut ac = self.access_count.write();
            for key in &victims {
                la.remove(key);
                ac.remove(key);
            }
        }

        // Phase 3: Remove from storage backend
        for key in &victims {
            self.inner.remove(key);
        }
    }

    fn evict_one(&self) -> Option<K> {
        let mut positions = self.positions.write();
        let mut insertion_order = self.insertion_order.write();

        // Get highest position (due to Reverse wrapper, first() gives highest)
        let highest_pos = positions.iter().next().map(|r| r.0)?;

        // Get the keys at this position
        let keys = insertion_order.get_mut(&highest_pos)?;
        if keys.is_empty() {
            return None;
        }

        // Find the least-recently-used key at this position
        let key = {
            let last_access = self.last_access.read();
            let access_count = self.access_count.read();
            let best_idx = Self::find_lru_key(keys, &last_access, &access_count);
            keys.remove(best_idx)
        };

        // If position is now empty, remove it from tracking
        if keys.is_empty() {
            insertion_order.remove(&highest_pos);
            positions.remove(&std::cmp::Reverse(highest_pos));
        }

        drop(positions);
        drop(insertion_order);

        // Clean up access tracking for evicted key
        self.last_access.write().remove(&key);
        self.access_count.write().remove(&key);

        self.inner.remove(&key);
        Some(key)
    }
}

impl<K, V, S> Storage<K, V> for PositionalEviction<K, V, S>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
    S: Storage<K, V>,
{
    fn insert(&self, key: K, value: V) {
        let position = key.position();

        {
            let mut positions = self.positions.write();
            let mut insertion_order = self.insertion_order.write();

            positions.insert(std::cmp::Reverse(position));
            insertion_order.entry(position).or_default().push(key);
        }

        self.inner.insert(key, value);
        self.maybe_evict();
    }

    fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    fn remove(&self, key: &K) -> Option<V> {
        let position = key.position();

        {
            let mut positions = self.positions.write();
            let mut insertion_order = self.insertion_order.write();

            if let Some(keys) = insertion_order.get_mut(&position) {
                keys.retain(|k| k != key);
                if keys.is_empty() {
                    insertion_order.remove(&position);
                    positions.remove(&std::cmp::Reverse(position));
                }
            }
        }

        // Clean up access tracking
        self.last_access.write().remove(key);
        self.access_count.write().remove(key);

        self.inner.remove(key)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) {
        self.positions.write().clear();
        self.insertion_order.write().clear();
        self.last_access.write().clear();
        self.access_count.write().clear();
        self.inner.clear();
    }

    fn insert_batch(&self, entries: Vec<(K, V)>) {
        // Phase 1: Update tracking under a single lock acquisition
        {
            let mut positions = self.positions.write();
            let mut insertion_order = self.insertion_order.write();
            for &(ref key, _) in &entries {
                let position = key.position();
                positions.insert(std::cmp::Reverse(position));
                insertion_order.entry(position).or_default().push(*key);
            }
        }
        // Locks dropped here

        // Phase 2: Insert into storage backend
        for (key, value) in entries {
            self.inner.insert(key, value);
        }

        // Phase 3: Evict once for the whole batch
        self.maybe_evict();
    }
}

impl<K, V, S> Eviction<K, V> for PositionalEviction<K, V, S>
where
    K: PositionalStorageKey + Ord + Copy + 'static,
    V: Clone + Send + Sync + 'static,
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

    // PositionalEviction tests

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct TestPosKey {
        position: u64,
        id: u64,
    }

    impl PositionalStorageKey for TestPosKey {
        fn position(&self) -> u64 {
            self.position
        }
    }

    #[test]
    fn test_positional_eviction_basic() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(100);

        let key1 = TestPosKey { position: 0, id: 1 };
        let key2 = TestPosKey { position: 1, id: 2 };

        evictable.insert(key1, 100);
        evictable.insert(key2, 200);

        assert_eq!(evictable.get(&key1), Some(100));
        assert_eq!(evictable.get(&key2), Some(200));
        assert_eq!(evictable.len(), 2);
    }

    #[test]
    fn test_positional_eviction_evicts_highest_first() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(100);

        // Insert at positions 0, 1, 2
        let key0 = TestPosKey { position: 0, id: 1 };
        let key1 = TestPosKey { position: 1, id: 2 };
        let key2 = TestPosKey { position: 2, id: 3 };

        evictable.insert(key0, 100);
        evictable.insert(key1, 200);
        evictable.insert(key2, 300);

        // Evict should remove from position 2 first
        let evicted = evictable.evict(1);
        assert_eq!(evicted, vec![key2]);
        assert_eq!(evictable.len(), 2);
        assert!(evictable.contains(&key0));
        assert!(evictable.contains(&key1));
        assert!(!evictable.contains(&key2));

        // Next eviction removes from position 1
        let evicted = evictable.evict(1);
        assert_eq!(evicted, vec![key1]);
    }

    #[test]
    fn test_positional_eviction_fifo_within_position() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(100);

        // Insert multiple keys at same position
        let key1 = TestPosKey { position: 5, id: 1 };
        let key2 = TestPosKey { position: 5, id: 2 };
        let key3 = TestPosKey { position: 5, id: 3 };

        evictable.insert(key1, 100);
        evictable.insert(key2, 200);
        evictable.insert(key3, 300);

        // Should evict in FIFO order within position
        let evicted = evictable.evict(2);
        assert_eq!(evicted, vec![key1, key2]);
        assert!(evictable.contains(&key3));
    }

    #[test]
    fn test_positional_eviction_auto_evict_on_capacity() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(3);

        // Insert 3 keys at increasing positions
        for i in 0..3 {
            evictable.insert(TestPosKey { position: i, id: i }, i);
        }
        assert_eq!(evictable.len(), 3);

        // Insert 4th key - should auto-evict highest position (2)
        evictable.insert(TestPosKey { position: 3, id: 3 }, 3);
        assert_eq!(evictable.len(), 3);

        // Position 3 was just inserted, so highest remaining after eviction of 3
        // should still have pos 0, 1, and the new 3
        assert!(evictable.contains(&TestPosKey { position: 0, id: 0 }));
        assert!(evictable.contains(&TestPosKey { position: 1, id: 1 }));
    }

    #[test]
    fn test_positional_eviction_remove() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(100);

        let key1 = TestPosKey { position: 0, id: 1 };
        let key2 = TestPosKey { position: 0, id: 2 };

        evictable.insert(key1, 100);
        evictable.insert(key2, 200);

        evictable.remove(&key1);
        assert_eq!(evictable.len(), 1);
        assert!(!evictable.contains(&key1));
        assert!(evictable.contains(&key2));
    }

    #[test]
    fn test_positional_eviction_insert_batch() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(100);

        let entries: Vec<(TestPosKey, u64)> = (0..10)
            .map(|i| (TestPosKey { position: i % 3, id: i }, i * 100))
            .collect();

        evictable.insert_batch(entries);

        assert_eq!(evictable.len(), 10);
        assert_eq!(evictable.get(&TestPosKey { position: 0, id: 0 }), Some(0));
        assert_eq!(evictable.get(&TestPosKey { position: 1, id: 1 }), Some(100));
        assert_eq!(evictable.get(&TestPosKey { position: 2, id: 2 }), Some(200));
    }

    #[test]
    fn test_positional_eviction_insert_batch_triggers_eviction() {
        let evictable: PositionalEviction<TestPosKey, u64> = PositionalEviction::with_capacity(5);

        // Batch insert 10 entries — should evict down to capacity
        let entries: Vec<(TestPosKey, u64)> = (0..10)
            .map(|i| (TestPosKey { position: i, id: i }, i * 100))
            .collect();

        evictable.insert_batch(entries);

        assert_eq!(evictable.len(), 5);
        // Lowest positions should survive (highest evicted first)
        assert!(evictable.contains(&TestPosKey { position: 0, id: 0 }));
        assert!(evictable.contains(&TestPosKey { position: 1, id: 1 }));
    }
}
