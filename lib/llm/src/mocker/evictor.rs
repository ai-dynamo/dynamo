// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::{Eq, Ordering};
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

/// A wrapper for (T, counter) that implements Ord based only on counter
#[derive(Debug, Clone, Eq, PartialEq)]
struct PriorityItem<T> {
    item: T,
    counter: i64,
}

impl<T: Eq> Ord for PriorityItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.counter.cmp(&other.counter)
    }
}

impl<T: Eq> PartialOrd for PriorityItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An LRU evictor that maintains objects and evicts them based on their
/// priority counter. Lower counter values are evicted first.
#[derive(Debug)]
pub struct LRUEvictor<T: Clone + Eq + Hash> {
    free_table: HashMap<T, i64>,
    priority_queue: BTreeSet<PriorityItem<T>>,
    positive_counter: i64,
    negative_counter: i64,
}

impl<T: Clone + Eq + Hash> Default for LRUEvictor<T> {
    fn default() -> Self {
        Self {
            free_table: HashMap::new(),
            priority_queue: BTreeSet::new(),
            positive_counter: 0,
            negative_counter: 0,
        }
    }
}

impl<T: Clone + Eq + Hash> LRUEvictor<T> {
    pub fn new(_cleanup_threshold: usize) -> Self {
        Self::default()
    }

    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, i64> {
        self.free_table.keys()
    }

    fn update(&mut self, object: T, counter: i64) {
        self.free_table.insert(object.clone(), counter);
        self.priority_queue.insert(PriorityItem {
            item: object,
            counter,
        });
    }

    pub fn insert(&mut self, object: T) {
        // Remove old entry if it exists
        if let Some(&old_counter) = self.free_table.get(&object) {
            self.priority_queue.remove(&PriorityItem {
                item: object.clone(),
                counter: old_counter,
            });
        }

        // Increment positive counter and insert
        self.positive_counter += 1;
        let counter = self.positive_counter;

        self.update(object, counter);
    }

    /// Push an object to the front with negative counter (highest priority for eviction)
    pub fn push_front(&mut self, object: T) {
        // Remove old entry if it exists
        if let Some(&old_counter) = self.free_table.get(&object) {
            self.priority_queue.remove(&PriorityItem {
                item: object.clone(),
                counter: old_counter,
            });
        }

        // Decrement negative counter and insert
        self.negative_counter -= 1;
        let counter = self.negative_counter;

        self.update(object, counter);
    }

    pub fn contains(&self, object: &T) -> bool {
        self.free_table.contains_key(object)
    }

    /// Evict an object based on LRU policy (lowest counter value)
    /// Returns the evicted object or None if no objects are available
    pub fn evict(&mut self) -> Option<T> {
        self.priority_queue.pop_first().map(|item| {
            self.free_table.remove(&item.item);
            item.item
        })
    }

    pub fn remove(&mut self, object: &T) -> bool {
        let Some(&counter) = self.free_table.get(object) else {
            return false;
        };

        self.free_table.remove(object);
        self.priority_queue.remove(&PriorityItem {
            item: object.clone(),
            counter,
        });
        true
    }

    pub fn len(&self) -> usize {
        self.free_table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.free_table.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_evictor_eviction_order() {
        // Create a new LRUEvictor
        let mut evictor = LRUEvictor::<i32>::new(1); // threshold value doesn't matter anymore

        // Add items in the specified order
        evictor.insert(4);
        evictor.insert(3);
        evictor.insert(2);
        evictor.insert(1);
        evictor.insert(5);
        evictor.insert(1); // Updates counter for 1
        evictor.insert(4); // Updates counter for 4
        evictor.insert(2); // Updates counter for 2
        evictor.push_front(4);

        // Verify the eviction order
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 4);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 3);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 5);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 1);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 2);
        let evicted = evictor.evict();
        assert_eq!(evicted, None);
        assert_eq!(evictor.len(), 0);
    }

    // ... existing test_push_front test ...
}
