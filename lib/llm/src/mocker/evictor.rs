// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::cmp::Eq;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

/// An LRU evictor that maintains objects and evicts them based on their
/// last accessed time. Implements a "lazy" eviction mechanism where:
/// 1. The priority queue does not immediately reflect updates or removes
/// 2. Objects are pushed to the queue in order of increasing priority (older objects first)
/// 3. The user must ensure objects are added in correct priority (temporal order)
/// 4. Remove and update operations are lazy - entries remain in the queue until
///    they are either evicted or cleaned up during maintenance
#[derive(Debug)]
pub struct LRUEvictor<T: Clone + Eq + Hash> {
    free_table: HashMap<T, f64>,
    priority_queue: VecDeque<(T, f64)>,
    cleanup_threshold: usize,
    start_time: Instant,
}

impl<T: Clone + Eq + Hash> Default for LRUEvictor<T> {
    fn default() -> Self {
        Self {
            free_table: HashMap::new(),
            priority_queue: VecDeque::new(),
            cleanup_threshold: 50,
            start_time: Instant::now(),
        }
    }
}

impl<T: Clone + Eq + Hash> LRUEvictor<T> {
    /// Create a new LRUEvictor with the default cleanup threshold
    pub fn new(cleanup_threshold: usize) -> Self {
        Self {
            cleanup_threshold,
            ..Default::default()
        }
    }

    /// Get the current timestamp as seconds since initialization
    pub fn current_timestamp(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get an iterator over the keys in the evictor
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, T, f64> {
        self.free_table.keys()
    }

    /// Insert or update an object in the evictor with current timestamp
    pub fn insert(&mut self, object: T) {
        let timestamp = self.current_timestamp();
        self._insert(object, timestamp);
    }

    /// Check if the evictor contains the given object
    pub fn contains(&self, object: &T) -> bool {
        self.free_table.contains_key(object)
    }

    /// Evict an object based on LRU policy
    /// Returns the evicted object or None if no objects are available
    pub fn evict(&mut self) -> Option<T> {
        if self.free_table.is_empty() {
            return None;
        }

        while let Some((object, last_accessed)) = self.priority_queue.pop_front() {
            let Some(&current_last_accessed) = self.free_table.get(&object) else {
                continue; // entry is already removed
            };

            if current_last_accessed == last_accessed {
                self.free_table.remove(&object);
                return Some(object);
            } // otherwise entry is stale
        }

        None
    }

    /// Insert or update an object in the evictor
    fn _insert(&mut self, object: T, last_accessed: f64) {
        self.free_table.insert(object.clone(), last_accessed);
        self.priority_queue.push_back((object, last_accessed));
        self.cleanup_if_necessary();
    }

    /// Remove an object from the evictor
    /// We don't remove from the priority queue immediately, as that would be inefficient
    /// Outdated entries will be filtered out during eviction or cleanup
    pub fn remove(&mut self, object: &T) -> bool {
        self.free_table.remove(object).is_some()
    }

    /// Get the number of objects in the evictor
    pub fn len(&self) -> usize {
        self.free_table.len()
    }

    /// Check if the evictor is empty
    pub fn is_empty(&self) -> bool {
        self.free_table.is_empty()
    }

    /// Check if cleanup is necessary and perform it if needed
    fn cleanup_if_necessary(&mut self) {
        if self.priority_queue.len() > self.cleanup_threshold * self.free_table.len() {
            self.cleanup();
        }
    }

    /// Clean up the priority queue by removing outdated entries
    fn cleanup(&mut self) {
        let mut new_priority_queue = VecDeque::new();
        for (object, timestamp) in self.priority_queue.drain(..) {
            let Some(&current_timestamp) = self.free_table.get(&object) else {
                continue;
            };

            if current_timestamp == timestamp {
                new_priority_queue.push_back((object, timestamp));
            }
        }
        self.priority_queue = new_priority_queue;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    fn test_lru_evictor_eviction_order(#[case] threshold: usize) {
        // Create a new LRUEvictor with the given cleanup threshold
        let mut evictor = LRUEvictor::<i32>::new(threshold);

        // Add items in the specified order with small delays between each
        evictor.insert(4);
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(3);
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(2);
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(1);
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(5);
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(1); // Updates timestamp for 1
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(4); // Updates timestamp for 4
        std::thread::sleep(std::time::Duration::from_millis(1));
        evictor.insert(2); // Updates timestamp for 2

        // Verify the eviction order
        println!("Testing with threshold {}", threshold);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 3);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 5);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 1);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 4);
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 2);
        let evicted = evictor.evict();
        assert_eq!(evicted, None);
        assert_eq!(evictor.len(), 0);
    }
}
