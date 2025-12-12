// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage block registry with TinyLFU eviction.
//!
//! Tracks which sequence hashes have been stored in object storage (S3/GCS).
//! Uses Moka cache for:
//! - Lock-free concurrent access (sharded internally)
//! - TinyLFU eviction policy (better than LRU for cache workloads)
//! - Bounded capacity (set via DYN_KVBM_OBJECT_NUM_BLOCKS)
//!
//! Note: Eviction from the registry does NOT delete the S3 object.
//! The object remains in S3 but the registry "forgets" about it.

use crate::tokens::SequenceHash;
use moka::sync::Cache;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::block_manager::ObjectStorageConfig;
use super::external::ExternalRegistry;

/// Object key for S3/GCS storage (u64 used by NIXL OBJ backend)
pub type ObjectKey = u64;

/// Default capacity when DYN_KVBM_OBJECT_NUM_BLOCKS is not set.
const DEFAULT_CAPACITY: u64 = 100_000;

/// Registry mapping sequence hashes to object storage keys.
///
/// Thread-safe, high-performance cache for tracking which KV cache blocks
/// have been offloaded to object storage.
///
/// Uses Moka with TinyLFU eviction policy:
/// - Combines frequency (how often accessed) with recency (when last accessed)
/// - Better hit rate than pure LRU for real-world cache workloads
/// - Automatically evicts when capacity is reached
///
/// Capacity is bounded by `DYN_KVBM_OBJECT_NUM_BLOCKS` environment variable.
pub struct ObjectRegistry {
    /// Moka cache: sequence hash → object key
    /// Uses TinyLFU eviction when capacity is reached
    cache: Cache<SequenceHash, ObjectKey>,
    /// Maximum capacity
    capacity: u64,
    /// Counter for total registrations (for metrics)
    total_registered: AtomicU64,
}

impl std::fmt::Debug for ObjectRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectRegistry")
            .field("len", &self.cache.entry_count())
            .field("capacity", &self.capacity)
            .field("total_registered", &self.total_registered.load(Ordering::Relaxed))
            .field("total_evicted", &self.total_evicted())
            .finish()
    }
}

impl Default for ObjectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectRegistry {
    /// Create a new registry with capacity from environment.
    ///
    /// Uses `DYN_KVBM_OBJECT_NUM_BLOCKS` if set and > 0,
    /// otherwise defaults to 100,000 entries.
    pub fn new() -> Self {
        let capacity = ObjectStorageConfig::num_blocks_from_env();
        let capacity = if capacity > 0 { capacity as u64 } else { DEFAULT_CAPACITY };
        Self::with_capacity(capacity)
    }

    /// Create a new registry with explicit capacity.
    pub fn with_capacity(capacity: u64) -> Self {
        let capacity = capacity.max(1); // Ensure at least 1

        // Build cache with TinyLFU eviction (Moka's default)
        // Note: We don't use eviction_listener to avoid complexity.
        // Eviction count can be computed as: total_registered - len()
        let cache = Cache::builder()
            .max_capacity(capacity)
            .build();

        tracing::info!(
            capacity = capacity,
            "ObjectRegistry initialized with TinyLFU eviction, capacity={}",
            capacity
        );

        Self {
            cache,
            capacity,
            total_registered: AtomicU64::new(0),
        }
    }

    /// Register a block stored in object storage.
    ///
    /// If at capacity, TinyLFU eviction will remove the least valuable entry.
    /// This operation is lock-free (only touches the relevant shard).
    pub fn register(&self, sequence_hash: SequenceHash, object_key: ObjectKey) {
        tracing::info!(
            target: "kvbm_local_registry",
            sequence_hash = %sequence_hash,
            object_key = %object_key,
            "LOCAL_REGISTER: hash={:#018x} key={:#018x}",
            sequence_hash, object_key
        );
        self.cache.insert(sequence_hash, object_key);
        self.total_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// Register using sequence hash as object key (common case).
    pub fn register_with_hash_as_key(&self, sequence_hash: SequenceHash) {
        self.register(sequence_hash, sequence_hash);
    }

    /// Check if a sequence hash exists in object storage.
    ///
    /// This is a lock-free read that also updates frequency tracking
    /// (for TinyLFU eviction decisions).
    pub fn contains(&self, sequence_hash: SequenceHash) -> bool {
        self.cache.contains_key(&sequence_hash)
    }

    /// Get the object key for a sequence hash.
    ///
    /// This updates frequency tracking (for TinyLFU eviction).
    pub fn get(&self, sequence_hash: SequenceHash) -> Option<ObjectKey> {
        self.cache.get(&sequence_hash)
    }

    /// Find matching sequence hashes in object storage.
    ///
    /// Returns longest contiguous prefix that exists.
    /// Each lookup is lock-free and updates frequency tracking.
    pub fn match_sequence_hashes(&self, hashes: &[SequenceHash]) -> Vec<(SequenceHash, ObjectKey)> {
        tracing::info!(
            target: "kvbm_local_registry",
            num_hashes = hashes.len(),
            first_hash = ?hashes.first().map(|h| format!("{:#018x}", h)),
            "LOCAL_MATCH_QUERY: count={} first={:#018x?}",
            hashes.len(), hashes.first()
        );

        let matched: Vec<(SequenceHash, ObjectKey)> = hashes
            .iter()
            .map_while(|hash| {
                self.cache.get(hash).map(|key| (*hash, key))
            })
            .collect();

        // Log each match
        for (hash, key) in &matched {
            tracing::info!(
                target: "kvbm_local_registry",
                sequence_hash = %hash,
                object_key = %key,
                "LOCAL_MATCH_HIT: hash={:#018x} key={:#018x}",
                hash, key
            );
        }

        tracing::info!(
            target: "kvbm_local_registry",
            matched = matched.len(),
            queried = hashes.len(),
            "LOCAL_MATCH_RESULT: matched={} of {}",
            matched.len(), hashes.len()
        );

        matched
    }

    /// Find ALL matching sequence hashes (non-contiguous).
    ///
    /// Unlike `match_sequence_hashes`, this doesn't stop on first miss.
    pub fn find_all(&self, hashes: &[SequenceHash]) -> Vec<(SequenceHash, ObjectKey)> {
        hashes
            .iter()
            .filter_map(|hash| {
                self.cache.get(hash).map(|key| (*hash, key))
            })
            .collect()
    }

    /// Remove a sequence hash from the registry.
    pub fn unregister(&self, sequence_hash: SequenceHash) {
        self.cache.invalidate(&sequence_hash);
    }

    /// Number of registered entries (approximate, may lag slightly).
    pub fn len(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cache.entry_count() == 0
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.cache.invalidate_all();
    }

    /// Total number of registrations ever made.
    pub fn total_registered(&self) -> u64 {
        self.total_registered.load(Ordering::Relaxed)
    }

    /// Estimated number of evictions due to capacity limit.
    /// Computed as: total_registered - current_len (approximate).
    pub fn total_evicted(&self) -> u64 {
        let registered = self.total_registered.load(Ordering::Relaxed);
        let current = self.cache.entry_count();
        registered.saturating_sub(current)
    }

    /// Get utilization as a fraction (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        self.cache.entry_count() as f64 / self.capacity as f64
    }

    /// Trigger pending maintenance tasks (optional, Moka does this automatically).
    pub fn run_pending_tasks(&self) {
        self.cache.run_pending_tasks();
    }
}

// ============================================================================
// ExternalRegistry Implementation
// ============================================================================

impl ExternalRegistry<SequenceHash> for ObjectRegistry {
    fn match_keys(&self, keys: &[SequenceHash]) -> Vec<SequenceHash> {
        tracing::debug!(
            target: "external_registry",
            num_keys = keys.len(),
            backend = self.backend_name(),
            "match_keys: querying {} keys",
            keys.len()
        );

        let matched: Vec<SequenceHash> = keys
            .iter()
            .copied()
            .take_while(|k| self.cache.contains_key(k))
            .collect();

        tracing::debug!(
            target: "external_registry",
            matched = matched.len(),
            queried = keys.len(),
            "match_keys: matched {} of {} keys",
            matched.len(),
            keys.len()
        );

        matched
    }

    fn register(&self, keys: &[SequenceHash]) -> usize {
        let mut registered = 0;
        for key in keys {
            // Use key as both sequence_hash and object_key (common pattern)
            self.cache.insert(*key, *key);
            self.total_registered.fetch_add(1, Ordering::Relaxed);
            registered += 1;

            tracing::trace!(
                target: "external_registry",
                key = %key,
                "register: key={:#018x}",
                key
            );
        }

        tracing::debug!(
            target: "external_registry",
            registered = registered,
            total = keys.len(),
            "register: registered {} keys",
            registered
        );

        registered
    }

    fn unregister(&self, keys: &[SequenceHash]) -> usize {
        let mut removed = 0;
        for key in keys {
            if self.cache.contains_key(key) {
                self.cache.invalidate(key);
                removed += 1;

                tracing::trace!(
                    target: "external_registry",
                    key = %key,
                    "unregister: key={:#018x}",
                    key
                );
            }
        }

        tracing::debug!(
            target: "external_registry",
            removed = removed,
            total = keys.len(),
            "unregister: removed {} of {} keys",
            removed,
            keys.len()
        );

        removed
    }

    fn backend_name(&self) -> &'static str {
        "object_storage"
    }

    fn can_register(&self) -> bool {
        // TinyLFU will evict entries if at capacity, so we can always accept registrations
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let registry = ObjectRegistry::with_capacity(100);

        registry.register_with_hash_as_key(1234);
        registry.register_with_hash_as_key(5678);

        assert!(registry.contains(1234));
        assert!(registry.contains(5678));
        assert!(!registry.contains(9999));
        assert_eq!(registry.get(1234), Some(1234));
    }

    #[test]
    fn test_capacity_eviction() {
        let registry = ObjectRegistry::with_capacity(10);

        // Fill beyond capacity
        for i in 0..20 {
            registry.register_with_hash_as_key(i);
        }

        // Force pending evictions to complete
        registry.run_pending_tasks();

        // Should have evicted some entries
        assert!(registry.len() <= 10, "len={} should be <= 10", registry.len());
    }

    #[test]
    fn test_match_contiguous() {
        let registry = ObjectRegistry::with_capacity(100);

        registry.register_with_hash_as_key(100);
        registry.register_with_hash_as_key(200);
        registry.register_with_hash_as_key(300);

        let hashes = vec![100, 200, 300, 400, 500];
        let matched = registry.match_sequence_hashes(&hashes);

        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0], (100, 100));
        assert_eq!(matched[1], (200, 200));
        assert_eq!(matched[2], (300, 300));
    }

    #[test]
    fn test_match_stops_on_miss() {
        let registry = ObjectRegistry::with_capacity(100);

        registry.register_with_hash_as_key(100);
        // 200 is missing
        registry.register_with_hash_as_key(300);

        let hashes = vec![100, 200, 300];
        let matched = registry.match_sequence_hashes(&hashes);

        // Should stop at 200 (first miss)
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0], (100, 100));
    }

    #[test]
    fn test_find_all_non_contiguous() {
        let registry = ObjectRegistry::with_capacity(100);

        registry.register_with_hash_as_key(100);
        // 200 is missing
        registry.register_with_hash_as_key(300);

        let hashes = vec![100, 200, 300];
        let found = registry.find_all(&hashes);

        // Should find both 100 and 300
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_unregister() {
        let registry = ObjectRegistry::with_capacity(100);

        registry.register_with_hash_as_key(111);
        assert!(registry.contains(111));

        registry.unregister(111);

        // Force pending tasks
        registry.run_pending_tasks();

        assert!(!registry.contains(111));
    }

    #[test]
    fn test_metrics() {
        let registry = ObjectRegistry::with_capacity(100);

        for i in 0..50 {
            registry.register_with_hash_as_key(i);
        }

        assert_eq!(registry.total_registered(), 50);
        assert!(registry.utilization() > 0.0);
        assert!(registry.utilization() <= 1.0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let registry = Arc::new(ObjectRegistry::with_capacity(10000));
        let mut handles = vec![];

        // Spawn 10 writer threads
        for t in 0..10 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let hash = (t * 10000 + i) as u64;
                    reg.register_with_hash_as_key(hash);
                }
            }));
        }

        // Spawn 10 reader threads
        for t in 0..10 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let hash = (t * 10000 + i) as u64;
                    let _ = reg.contains(hash);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(registry.total_registered(), 10000);
    }

    // ========================================================================
    // ExternalRegistry Trait Tests
    // ========================================================================

    mod external_registry_tests {
        use super::*;
        use crate::block_manager::distributed::registry::external::ExternalRegistry;

        #[test]
        fn test_trait_match_keys_contiguous() {
            let registry = ObjectRegistry::with_capacity(100);

            // Register some keys
            ExternalRegistry::register(&registry, &[100, 200, 300]);

            // Match contiguous prefix
            let matched = ExternalRegistry::match_keys(&registry, &[100, 200, 300, 400, 500]);
            assert_eq!(matched, vec![100, 200, 300]);
        }

        #[test]
        fn test_trait_match_keys_stops_on_miss() {
            let registry = ObjectRegistry::with_capacity(100);

            ExternalRegistry::register(&registry, &[100, 300]); // 200 is missing

            let matched = ExternalRegistry::match_keys(&registry, &[100, 200, 300]);
            assert_eq!(matched, vec![100]); // Stops at 200
        }

        #[test]
        fn test_trait_register_returns_count() {
            let registry = ObjectRegistry::with_capacity(100);

            let count = ExternalRegistry::register(&registry, &[1, 2, 3, 4, 5]);
            assert_eq!(count, 5);
        }

        #[test]
        fn test_trait_unregister_returns_count() {
            let registry = ObjectRegistry::with_capacity(100);

            ExternalRegistry::register(&registry, &[10, 20, 30]);

            // Unregister some (20 exists, 40 doesn't)
            let count = ExternalRegistry::unregister(&registry, &[20, 40]);
            assert_eq!(count, 1);

            // Verify 20 is gone
            let matched = ExternalRegistry::match_keys(&registry, &[10, 20, 30]);
            assert_eq!(matched, vec![10]); // Stops at missing 20
        }

        #[test]
        fn test_trait_backend_name() {
            let registry = ObjectRegistry::with_capacity(100);
            assert_eq!(ExternalRegistry::<u64>::backend_name(&registry), "object_storage");
        }

        #[test]
        fn test_trait_can_register() {
            let registry = ObjectRegistry::with_capacity(100);
            // TinyLFU always allows registration (evicts if needed)
            assert!(ExternalRegistry::<u64>::can_register(&registry));
        }
    }
}
