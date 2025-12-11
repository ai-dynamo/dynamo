// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frequency tracking for block reuse policies using Count-Min Sketch.

use parking_lot::Mutex;
use xxhash_rust::const_xxh3::const_custom_default_secret;
use xxhash_rust::xxh3::xxh3_64_with_secret;

const SECRET_0: &[u8; 192] = &const_custom_default_secret(0);
const SECRET_1: &[u8; 192] = &const_custom_default_secret(1);
const SECRET_2: &[u8; 192] = &const_custom_default_secret(2);
const SECRET_3: &[u8; 192] = &const_custom_default_secret(3);

/// Trait for types that can be used as keys in the TinyLFU sketch.
pub trait SketchKey: Copy + Send + Sync + 'static {
    /// Convert the key to bytes for hashing.
    fn hash_with_secret(&self, secret: &[u8; 192]) -> u64;
}

impl SketchKey for u64 {
    fn hash_with_secret(&self, secret: &[u8; 192]) -> u64 {
        let bytes = self.to_le_bytes();
        xxh3_64_with_secret(&bytes, secret)
    }
}

impl SketchKey for u128 {
    fn hash_with_secret(&self, secret: &[u8; 192]) -> u64 {
        let bytes = self.to_le_bytes();
        xxh3_64_with_secret(&bytes, secret)
    }
}

pub struct TinyLFUSketch<K: SketchKey> {
    table: Vec<u64>,
    size: u32,
    sample_size: u32,
    _phantom: std::marker::PhantomData<K>,
}

impl<K: SketchKey> TinyLFUSketch<K> {
    const RESET_MASK: u64 = 0x7777_7777_7777_7777;
    const ONE_MASK: u64 = 0x1111_1111_1111_1111;

    pub fn new(capacity: usize) -> Self {
        let table_size = std::cmp::max(1, capacity / 4);
        let sample_size = capacity.saturating_mul(10).min(u32::MAX as usize) as u32;

        Self {
            table: vec![0; table_size],
            size: 0,
            sample_size,
            _phantom: std::marker::PhantomData,
        }
    }

    fn hash(key: &K, seed: u32) -> u64 {
        let secret = match seed {
            0 => SECRET_0,
            1 => SECRET_1,
            2 => SECRET_2,
            3 => SECRET_3,
            _ => SECRET_0,
        };
        key.hash_with_secret(secret)
    }

    pub fn increment(&mut self, key: K) {
        if self.table.is_empty() {
            return;
        }

        let mut added = false;

        for i in 0..4 {
            let hash = Self::hash(&key, i);
            let table_index = (hash as usize) % self.table.len();
            let counter_index = (hash & 15) as u8;

            if self.increment_at(table_index, counter_index) {
                added = true;
            }
        }

        if added {
            self.size += 1;
            if self.size >= self.sample_size {
                self.reset();
            }
        }
    }

    fn increment_at(&mut self, table_index: usize, counter_index: u8) -> bool {
        let offset = (counter_index as usize) * 4;
        let mask = 0xF_u64 << offset;

        if self.table[table_index] & mask != mask {
            self.table[table_index] += 1u64 << offset;
            true
        } else {
            false
        }
    }

    pub fn estimate(&self, key: K) -> u32 {
        if self.table.is_empty() {
            return 0;
        }

        let mut min_count = u32::MAX;

        for i in 0..4 {
            let hash = Self::hash(&key, i);
            let table_index = (hash as usize) % self.table.len();
            let counter_index = (hash & 15) as u8;
            let count = self.count_at(table_index, counter_index);
            min_count = min_count.min(count as u32);
        }

        min_count
    }

    fn count_at(&self, table_index: usize, counter_index: u8) -> u8 {
        let offset = (counter_index as usize) * 4;
        let mask = 0xF_u64 << offset;
        ((self.table[table_index] & mask) >> offset) as u8
    }

    fn reset(&mut self) {
        let mut count = 0u32;

        for entry in self.table.iter_mut() {
            count += (*entry & Self::ONE_MASK).count_ones();
            *entry = (*entry >> 1) & Self::RESET_MASK;
        }

        self.size = (self.size >> 1) - (count >> 2);
    }
}

pub trait FrequencyTracker<K: SketchKey>: Send + Sync {
    fn touch(&self, key: K);
    fn count(&self, key: K) -> u32;
}

pub struct TinyLFUTracker<K: SketchKey> {
    sketch: Mutex<TinyLFUSketch<K>>,
}

impl<K: SketchKey> TinyLFUTracker<K> {
    pub fn new(capacity: usize) -> Self {
        Self {
            sketch: Mutex::new(TinyLFUSketch::new(capacity)),
        }
    }
}

impl<K: SketchKey> FrequencyTracker<K> for TinyLFUTracker<K> {
    fn touch(&self, key: K) {
        self.sketch.lock().increment(key);
    }

    fn count(&self, key: K) -> u32 {
        self.sketch.lock().estimate(key)
    }
}

pub struct NoOpTracker<K: SketchKey> {
    _phantom: std::marker::PhantomData<K>,
}

impl<K: SketchKey> NoOpTracker<K> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K: SketchKey> Default for NoOpTracker<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: SketchKey> FrequencyTracker<K> for NoOpTracker<K> {
    fn touch(&self, _key: K) {}
    fn count(&self, _key: K) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinylfu_increment_and_estimate() {
        let mut sketch = TinyLFUSketch::<u64>::new(100);

        sketch.increment(42);
        assert_eq!(sketch.estimate(42), 1);

        sketch.increment(42);
        sketch.increment(42);
        assert_eq!(sketch.estimate(42), 3);

        assert_eq!(sketch.estimate(99), 0);
    }

    #[test]
    fn test_tinylfu_saturation() {
        let mut sketch = TinyLFUSketch::<u64>::new(100);

        for _ in 0..20 {
            sketch.increment(42);
        }

        assert!(sketch.estimate(42) <= 15);
    }

    #[test]
    fn test_tinylfu_reset() {
        let mut sketch = TinyLFUSketch::<u64>::new(10);

        for i in 0..100 {
            sketch.increment(i);
        }

        let estimate_before = sketch.estimate(5);
        assert!(estimate_before > 0);
    }

    #[test]
    fn test_frequency_tracker_trait() {
        let tracker = TinyLFUTracker::<u64>::new(100);

        tracker.touch(42);
        assert_eq!(tracker.count(42), 1);

        tracker.touch(42);
        tracker.touch(42);
        assert_eq!(tracker.count(42), 3);
    }

    #[test]
    fn test_noop_tracker() {
        let tracker = NoOpTracker::<u64>::new();

        tracker.touch(42);
        assert_eq!(tracker.count(42), 0);

        tracker.touch(42);
        assert_eq!(tracker.count(42), 0);
    }

    #[test]
    fn test_u128_keys() {
        let mut sketch = TinyLFUSketch::<u128>::new(100);

        let key: u128 = 0x0123_4567_89AB_CDEF_0123_4567_89AB_CDEF;

        sketch.increment(key);
        assert_eq!(sketch.estimate(key), 1);

        sketch.increment(key);
        sketch.increment(key);
        assert_eq!(sketch.estimate(key), 3);

        assert_eq!(sketch.estimate(0), 0);
    }

    #[test]
    fn test_u128_tracker() {
        let tracker = TinyLFUTracker::<u128>::new(100);

        let key: u128 = 0x0123_4567_89AB_CDEF_0123_4567_89AB_CDEF;

        tracker.touch(key);
        assert_eq!(tracker.count(key), 1);

        tracker.touch(key);
        tracker.touch(key);
        assert_eq!(tracker.count(key), 3);
    }
}
