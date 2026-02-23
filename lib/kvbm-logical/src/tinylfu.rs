// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frequency tracking for block reuse policies using Count-Min Sketch.

use derive_builder::Builder;
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

/// Policy that determines when the TinyLFU sketch should decay its counters.
pub trait DecayPolicy: Send + Sync {
    /// Returns `true` if the sketch should reset/decay given the number of
    /// increments since the last reset.
    fn should_decay(&self, increments_since_last_reset: u32) -> bool;
}

/// Fixed-threshold decay policy: decays after a fixed number of increments.
///
/// This is the default policy, matching the original `capacity * 10` behavior.
pub struct FixedDecayPolicy {
    sample_size: u32,
}

impl FixedDecayPolicy {
    /// Create a policy with a specific sample size threshold.
    pub fn new(sample_size: u32) -> Self {
        Self { sample_size }
    }

    /// Create a policy derived from capacity, using `capacity * 10` as the threshold.
    pub fn from_capacity(capacity: usize) -> Self {
        let sample_size = capacity.saturating_mul(10).min(u32::MAX as usize) as u32;
        Self { sample_size }
    }
}

impl DecayPolicy for FixedDecayPolicy {
    fn should_decay(&self, increments_since_last_reset: u32) -> bool {
        increments_since_last_reset >= self.sample_size
    }
}

/// Settings for constructing a [`TinyLFUSketch`] or [`TinyLFUTracker`].
///
/// # Example
///
/// ```ignore
/// // Simple with defaults (FixedDecayPolicy from capacity)
/// let sketch = TinyLFUSettings::builder()
///     .capacity(100)
///     .build()?
///     .into_sketch::<u64>();
///
/// // With custom decay policy
/// let sketch = TinyLFUSettings::builder()
///     .capacity(100)
///     .decay_policy(FixedDecayPolicy::new(500))
///     .build()?
///     .into_sketch::<u64>();
/// ```
#[derive(Builder)]
#[builder(setter(into), build_fn(error = "anyhow::Error"), pattern = "owned")]
pub struct TinyLFUSettings {
    capacity: usize,

    #[builder(default, setter(custom))]
    decay_policy: Option<Box<dyn DecayPolicy>>,
}

impl TinyLFUSettingsBuilder {
    pub fn decay_policy(mut self, policy: impl DecayPolicy + 'static) -> Self {
        self.decay_policy = Some(Some(Box::new(policy)));
        self
    }
}

impl TinyLFUSettings {
    /// Creates a new builder for TinyLFUSettings.
    pub fn builder() -> TinyLFUSettingsBuilder {
        TinyLFUSettingsBuilder::default()
    }

    /// Converts settings into a [`TinyLFUSketch`].
    pub fn into_sketch<K: SketchKey>(self) -> TinyLFUSketch<K> {
        let decay_policy = self
            .decay_policy
            .unwrap_or_else(|| Box::new(FixedDecayPolicy::from_capacity(self.capacity)));
        TinyLFUSketch::with_decay_policy(self.capacity, decay_policy)
    }

    /// Converts settings into a [`TinyLFUTracker`].
    pub fn into_tracker<K: SketchKey>(self) -> TinyLFUTracker<K> {
        TinyLFUTracker {
            sketch: Mutex::new(self.into_sketch()),
        }
    }
}

pub struct TinyLFUSketch<K: SketchKey> {
    table: Vec<u64>,
    size: u32,
    decay_policy: Box<dyn DecayPolicy>,
    _phantom: std::marker::PhantomData<K>,
}

impl<K: SketchKey> TinyLFUSketch<K> {
    const RESET_MASK: u64 = 0x7777_7777_7777_7777;
    const ONE_MASK: u64 = 0x1111_1111_1111_1111;

    pub fn new(capacity: usize) -> Self {
        let decay_policy = Box::new(FixedDecayPolicy::from_capacity(capacity));
        Self::with_decay_policy(capacity, decay_policy)
    }

    fn with_decay_policy(capacity: usize, decay_policy: Box<dyn DecayPolicy>) -> Self {
        let table_size = std::cmp::max(1, capacity / 4);

        Self {
            table: vec![0; table_size],
            size: 0,
            decay_policy,
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
            if self.decay_policy.should_decay(self.size) {
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

        let half = self.size >> 1;
        let dec = count >> 2;
        self.size = half.saturating_sub(dec);
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

    #[test]
    fn test_settings_builder_default_policy() {
        let sketch = TinyLFUSettings::builder()
            .capacity(100usize)
            .build()
            .unwrap()
            .into_sketch::<u64>();

        assert_eq!(sketch.estimate(42), 0);
    }

    #[test]
    fn test_settings_builder_custom_policy() {
        let mut sketch = TinyLFUSettings::builder()
            .capacity(100usize)
            .decay_policy(FixedDecayPolicy::new(500))
            .build()
            .unwrap()
            .into_sketch::<u64>();

        // With sample_size=500 (instead of default 1000), decay triggers sooner
        // Increment the same key many times and verify it tracks
        for _ in 0..10 {
            sketch.increment(42);
        }
        assert!(sketch.estimate(42) >= 5);
    }

    #[test]
    fn test_settings_builder_into_tracker() {
        let tracker = TinyLFUSettings::builder()
            .capacity(100usize)
            .build()
            .unwrap()
            .into_tracker::<u64>();

        tracker.touch(42);
        assert_eq!(tracker.count(42), 1);
    }

    #[test]
    fn test_fixed_decay_policy() {
        let policy = FixedDecayPolicy::new(100);
        assert!(!policy.should_decay(99));
        assert!(policy.should_decay(100));
        assert!(policy.should_decay(101));
    }

    #[test]
    fn test_fixed_decay_policy_from_capacity() {
        let policy = FixedDecayPolicy::from_capacity(10);
        assert!(!policy.should_decay(99));
        assert!(policy.should_decay(100)); // 10 * 10 = 100
    }

    #[test]
    fn test_manual_decay_via_atomic_policy() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ManualDecayPolicy {
            trigger: Arc<AtomicBool>,
        }

        impl DecayPolicy for ManualDecayPolicy {
            fn should_decay(&self, _increments_since_last_reset: u32) -> bool {
                self.trigger.swap(false, Ordering::Relaxed)
            }
        }

        let trigger = Arc::new(AtomicBool::new(false));
        let mut sketch = TinyLFUSettings::builder()
            .capacity(100usize)
            .decay_policy(ManualDecayPolicy {
                trigger: Arc::clone(&trigger),
            })
            .build()
            .unwrap()
            .into_sketch::<u64>();

        // Increment key 42 four times — no decay armed
        for _ in 0..4 {
            sketch.increment(42);
        }
        assert_eq!(sketch.estimate(42), 4);

        // Arm decay, then increment a *different* key to trigger it.
        // The next added increment will see should_decay → true and reset.
        trigger.store(true, Ordering::Relaxed);
        sketch.increment(99);

        // After reset, counters are halved: 4 → 2, and key 99's single
        // count also halves (1 → 0 or 1 depending on rounding).
        let estimate_after = sketch.estimate(42);
        assert_eq!(estimate_after, 2, "counter should be halved by decay");
        assert!(sketch.estimate(99) <= 1);
    }
}
