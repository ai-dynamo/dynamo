// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage implementations for the registry.
//!
//! This module provides the `MokaStorage` implementation for multi-bucket registry.
//! Each entry is keyed by (bucket_id, sequence_hash).

use std::time::Duration;

use dashmap::DashMap;
use moka::sync::Cache;

use super::protocol::{BucketId, SequenceHash};
use super::types::ObjectKey;

/// Lease information for a hash being offloaded.
///
/// Uses a simple monotonic counter approach for expiration checking
/// which is faster than Instant comparisons in hot paths.
#[derive(Debug, Clone, Copy)]
struct Lease {
    /// Expiration time as nanos since an arbitrary epoch (for fast comparison).
    expires_at_nanos: u64,
}

/// Global start time for lease expiration calculations.
static LEASE_EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn lease_now_nanos() -> u64 {
    let epoch = LEASE_EPOCH.get_or_init(std::time::Instant::now);
    epoch.elapsed().as_nanos() as u64
}

impl Lease {
    fn new(timeout: Duration) -> Self {
        Self {
            expires_at_nanos: lease_now_nanos() + timeout.as_nanos() as u64,
        }
    }

    #[inline]
    fn is_expired(&self) -> bool {
        lease_now_nanos() >= self.expires_at_nanos
    }
}

/// Registry key combining bucket_id and sequence_hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RegistryKey {
    bucket_id: BucketId,
    sequence_hash: SequenceHash,
}

impl RegistryKey {
    #[inline]
    fn new(bucket_id: BucketId, sequence_hash: SequenceHash) -> Self {
        Self {
            bucket_id,
            sequence_hash,
        }
    }
}

/// Moka-based storage with TinyLFU eviction policy.
///
/// This is the recommended storage for production use. It provides:
/// - O(1) average case operations
/// - Automatic eviction when capacity is reached
/// - Thread-safe concurrent access
/// - Lease-based claiming for deduplication
/// - Multi-bucket support via compound keys (bucket_id, hash)
///
/// Memory usage: ~128 bytes per entry (including Moka overhead)
pub struct MokaStorage {
    /// Permanent registry: (bucket_id, hash) → object_key
    cache: Cache<RegistryKey, ObjectKey>,
    /// Active leases: (bucket_id, hash) → Lease
    leases: DashMap<RegistryKey, Lease>,
    capacity: u64,
    /// Default lease timeout.
    lease_timeout: Duration,
}

impl MokaStorage {
    /// Create new storage with specified capacity.
    pub fn new(capacity: u64) -> Self {
        Self::with_lease_timeout(capacity, Duration::from_secs(30))
    }

    /// Create new storage with specified capacity and lease timeout.
    pub fn with_lease_timeout(capacity: u64, lease_timeout: Duration) -> Self {
        let cache = Cache::new(capacity);
        Self {
            cache,
            leases: DashMap::new(),
            capacity,
            lease_timeout,
        }
    }

    /// Sync pending tasks.
    ///
    /// Moka's entry_count() is eventually consistent. This method forces
    /// processing of pending tasks to ensure accurate counts.
    /// Primarily useful for tests.
    pub fn sync(&self) {
        self.cache.run_pending_tasks();
    }

    /// Insert entry for a specific bucket.
    pub fn insert(&self, bucket_id: BucketId, hash: SequenceHash, key: ObjectKey) {
        let rkey = RegistryKey::new(bucket_id, hash);
        self.cache.insert(rkey, key);
        self.leases.remove(&rkey);
    }

    /// Insert batch of entries for a specific bucket.
    pub fn insert_batch(&self, bucket_id: BucketId, entries: &[(SequenceHash, ObjectKey)]) {
        for (hash, key) in entries {
            let rkey = RegistryKey::new(bucket_id, *hash);
            self.cache.insert(rkey, *key);
            self.leases.remove(&rkey);
        }
    }

    /// Check if hash exists in specific bucket.
    pub fn contains(&self, bucket_id: BucketId, hash: SequenceHash) -> bool {
        let rkey = RegistryKey::new(bucket_id, hash);
        self.cache.contains_key(&rkey)
    }

    /// Get entry from specific bucket.
    pub fn get(&self, bucket_id: BucketId, hash: SequenceHash) -> Option<ObjectKey> {
        let rkey = RegistryKey::new(bucket_id, hash);
        self.cache.get(&rkey)
    }

    /// Match prefix in specific bucket.
    pub fn match_prefix(
        &self,
        bucket_id: BucketId,
        hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, ObjectKey)> {
        hashes
            .iter()
            .map_while(|&hash| {
                let rkey = RegistryKey::new(bucket_id, hash);
                self.cache.get(&rkey).map(|key| (hash, key))
            })
            .collect()
    }

    /// Claim leases for multiple hashes in a specific bucket.
    ///
    /// Note: This is NOT fully atomic across all hashes (each hash is processed
    /// independently for better concurrency), but each individual hash claim is atomic.
    ///
    /// Returns three vectors:
    /// - granted: Hashes where lease was granted
    /// - already_stored: Hashes already permanently registered
    /// - already_leased: Hashes leased by other workers
    pub fn try_claim_leases(
        &self,
        bucket_id: BucketId,
        hashes: &[SequenceHash],
    ) -> (Vec<SequenceHash>, Vec<SequenceHash>, Vec<SequenceHash>) {
        let mut granted = Vec::with_capacity(hashes.len());
        let mut already_stored = Vec::new();
        let mut already_leased = Vec::new();

        let new_lease = Lease::new(self.lease_timeout);

        for &hash in hashes {
            let rkey = RegistryKey::new(bucket_id, hash);

            // Fast path: check cache first
            if self.cache.contains_key(&rkey) {
                already_stored.push(hash);
                continue;
            }

            // Use entry API for atomic lease check/grant
            use dashmap::mapref::entry::Entry;
            match self.leases.entry(rkey) {
                Entry::Occupied(entry) => {
                    // Double-check cache
                    if self.cache.contains_key(&rkey) {
                        entry.remove();
                        already_stored.push(hash);
                        continue;
                    }

                    if entry.get().is_expired() {
                        *entry.into_ref() = new_lease;
                        granted.push(hash);
                    } else {
                        already_leased.push(hash);
                    }
                }
                Entry::Vacant(entry) => {
                    if self.cache.contains_key(&rkey) {
                        already_stored.push(hash);
                        continue;
                    }
                    entry.insert(new_lease);
                    granted.push(hash);
                }
            }
        }

        (granted, already_stored, already_leased)
    }

    /// Get total number of entries across all buckets.
    pub fn len(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get storage capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.cache.invalidate_all();
        self.leases.clear();
    }

    /// Clean up expired leases.
    pub fn cleanup_expired_leases(&self) -> usize {
        let before = self.leases.len();
        self.leases.retain(|_, lease| !lease.is_expired());
        before - self.leases.len()
    }

    /// Get the number of active leases.
    pub fn lease_count(&self) -> usize {
        self.leases.iter().filter(|e| !e.value().is_expired()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BUCKET: BucketId = 12345;

    fn test_storage_basic(storage: &MokaStorage) {
        // Test insert and get
        storage.insert(TEST_BUCKET, 100, 100);
        storage.insert(TEST_BUCKET, 200, 200);
        storage.insert(TEST_BUCKET, 300, 300);

        assert!(storage.contains(TEST_BUCKET, 100));
        assert!(storage.contains(TEST_BUCKET, 200));
        assert!(storage.contains(TEST_BUCKET, 300));
        assert!(!storage.contains(TEST_BUCKET, 400));

        assert_eq!(storage.get(TEST_BUCKET, 100), Some(100));
        assert_eq!(storage.get(TEST_BUCKET, 200), Some(200));
        assert_eq!(storage.get(TEST_BUCKET, 400), None);

        storage.sync();
        assert_eq!(storage.len(), 3);
    }

    fn test_storage_batch(storage: &MokaStorage) {
        let entries = vec![(100, 100), (200, 200), (300, 300)];
        storage.insert_batch(TEST_BUCKET, &entries);

        storage.sync();
        assert_eq!(storage.len(), 3);
        assert!(storage.contains(TEST_BUCKET, 100));
        assert!(storage.contains(TEST_BUCKET, 200));
        assert!(storage.contains(TEST_BUCKET, 300));
    }

    fn test_storage_match_prefix(storage: &MokaStorage) {
        storage.insert_batch(TEST_BUCKET, &[(100, 100), (200, 200), (300, 300)]);

        // Full match
        let result = storage.match_prefix(TEST_BUCKET, &[100, 200, 300]);
        assert_eq!(result.len(), 3);

        // Partial match (stops at 250 which doesn't exist)
        let result = storage.match_prefix(TEST_BUCKET, &[100, 200, 250, 300]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (100, 100));
        assert_eq!(result[1], (200, 200));

        // No match (first element doesn't exist)
        let result = storage.match_prefix(TEST_BUCKET, &[50, 100, 200]);
        assert_eq!(result.len(), 0);
    }

    fn test_storage_clear(storage: &MokaStorage) {
        storage.insert_batch(TEST_BUCKET, &[(100, 100), (200, 200)]);
        storage.sync();
        assert_eq!(storage.len(), 2);

        storage.clear();
        storage.sync();
        assert_eq!(storage.len(), 0);
        assert!(!storage.contains(TEST_BUCKET, 100));
    }

    fn test_multi_bucket(storage: &MokaStorage) {
        let bucket_a: BucketId = 1;
        let bucket_b: BucketId = 2;

        // Same hash in different buckets should be separate
        storage.insert(bucket_a, 100, 1000);
        storage.insert(bucket_b, 100, 2000);

        storage.sync();
        assert_eq!(storage.len(), 2);

        assert_eq!(storage.get(bucket_a, 100), Some(1000));
        assert_eq!(storage.get(bucket_b, 100), Some(2000));

        // Different buckets don't see each other's entries
        assert!(!storage.contains(bucket_a, 200));
        storage.insert(bucket_b, 200, 2200);
        assert!(!storage.contains(bucket_a, 200));
        assert!(storage.contains(bucket_b, 200));
    }

    #[test]
    fn test_moka_storage() {
        let storage = MokaStorage::new(1000);
        test_storage_basic(&storage);

        let storage = MokaStorage::new(1000);
        test_storage_batch(&storage);

        let storage = MokaStorage::new(1000);
        test_storage_match_prefix(&storage);

        let storage = MokaStorage::new(1000);
        test_storage_clear(&storage);

        let storage = MokaStorage::new(1000);
        test_multi_bucket(&storage);
    }

    #[test]
    fn test_moka_capacity() {
        let storage = MokaStorage::new(100);
        assert_eq!(storage.capacity(), 100);
    }

    #[test]
    fn test_lease_claiming() {
        let storage = MokaStorage::new(1000);

        // Claim some leases
        let (granted, stored, leased) = storage.try_claim_leases(TEST_BUCKET, &[100, 200, 300]);
        assert_eq!(granted.len(), 3);
        assert!(stored.is_empty());
        assert!(leased.is_empty());

        // Try to claim same hashes again - should be leased
        let (granted, stored, leased) = storage.try_claim_leases(TEST_BUCKET, &[100, 200, 400]);
        assert_eq!(granted.len(), 1); // Only 400 granted
        assert!(stored.is_empty());
        assert_eq!(leased.len(), 2); // 100, 200 already leased

        // Register one hash
        storage.insert(TEST_BUCKET, 100, 100);

        // Now 100 is stored, 200 is still leased, 500 is new
        let (granted, stored, leased) = storage.try_claim_leases(TEST_BUCKET, &[100, 200, 500]);
        assert_eq!(granted.len(), 1); // 500
        assert_eq!(stored.len(), 1); // 100
        assert_eq!(leased.len(), 1); // 200
    }
}


