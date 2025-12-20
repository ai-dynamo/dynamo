// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local in-process registry implementation.
//!
//! This is a passthrough implementation that uses an in-process Moka cache.
//! Useful for:
//! - Testing without network
//! - Single-node deployments
//! - Development and debugging

use anyhow::Result;
use async_trait::async_trait;
use tracing::trace;

use super::super::protocol::{bucket_id_from_name, SequenceHash};
use super::super::storage::MokaStorage;
use super::super::traits::DistributedRegistry;
use super::super::types::{ObjectKey, OffloadResult};

/// Local in-process registry.
///
/// Implements `DistributedRegistry` using a local Moka cache.
/// No network communication - all operations are in-process.
///
/// # Example
/// ```ignore
/// let registry = LocalRegistry::new(100_000);
///
/// // Register some hashes
/// registry.register("my-bucket", &[hash1, hash2]).await?;
///
/// // Query
/// let result = registry.can_offload("my-bucket", &[hash1, hash2, hash3]).await?;
/// assert_eq!(result.can_offload, vec![hash3]);
/// ```
pub struct LocalRegistry {
    storage: MokaStorage,
}

impl LocalRegistry {
    /// Create a new local registry with specified capacity.
    pub fn new(capacity: u64) -> Self {
        Self {
            storage: MokaStorage::new(capacity),
        }
    }

    /// Get the current number of entries.
    pub fn len(&self) -> u64 {
        self.storage.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get the capacity.
    pub fn capacity(&self) -> u64 {
        self.storage.capacity()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.storage.clear();
    }
}

#[async_trait]
impl DistributedRegistry for LocalRegistry {
    async fn register(&self, bucket_name: &str, sequence_hashes: &[SequenceHash]) -> Result<()> {
        let bucket_id = bucket_id_from_name(bucket_name);
        for &hash in sequence_hashes {
            self.storage.insert(bucket_id, hash, hash as u64);
        }
        trace!(
            "LocalRegistry: registered {} hashes in bucket {}",
            sequence_hashes.len(),
            bucket_name
        );
        Ok(())
    }

    async fn register_with_keys(
        &self,
        bucket_name: &str,
        entries: &[(SequenceHash, ObjectKey)],
    ) -> Result<()> {
        let bucket_id = bucket_id_from_name(bucket_name);
        self.storage.insert_batch(bucket_id, entries);
        trace!(
            "LocalRegistry: registered {} entries in bucket {}",
            entries.len(),
            bucket_name
        );
        Ok(())
    }

    async fn can_offload(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<OffloadResult> {
        let bucket_id = bucket_id_from_name(bucket_name);
        // Use atomic lease claiming just like the distributed version
        let (granted, already_stored, already_leased) =
            self.storage.try_claim_leases(bucket_id, hashes);

        let result = OffloadResult {
            can_offload: granted,
            already_stored,
            leased: already_leased,
        };

        trace!(
            "LocalRegistry: can_offload {} granted, {} stored, {} leased of {} in bucket {}",
            result.can_offload.len(),
            result.already_stored.len(),
            result.leased.len(),
            hashes.len(),
            bucket_name
        );

        Ok(result)
    }

    async fn match_sequence_hashes(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<Vec<(SequenceHash, ObjectKey)>> {
        let bucket_id = bucket_id_from_name(bucket_name);
        let matched = self.storage.match_prefix(bucket_id, hashes);
        trace!(
            "LocalRegistry: match_sequence_hashes {} of {} in bucket {}",
            matched.len(),
            hashes.len(),
            bucket_name
        );
        Ok(matched)
    }

    async fn flush(&self) -> Result<()> {
        // No-op for local registry
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BUCKET: &str = "test-bucket";

    impl LocalRegistry {
        /// Sync for tests - forces Moka to process pending tasks.
        /// Moka's entry_count() is eventually consistent; this ensures accurate counts in tests.
        fn sync(&self) {
            self.storage.sync();
        }
    }

    #[tokio::test]
    async fn test_local_registry_basic() {
        let registry = LocalRegistry::new(1000);

        // Register some hashes
        registry.register(TEST_BUCKET, &[100, 200, 300]).await.unwrap();

        registry.sync();
        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
    }

    #[tokio::test]
    async fn test_local_registry_can_offload() {
        let registry = LocalRegistry::new(1000);

        // Register some hashes
        registry.register(TEST_BUCKET, &[100, 200]).await.unwrap();

        // Check can_offload
        let result = registry
            .can_offload(TEST_BUCKET, &[100, 200, 300, 400])
            .await
            .unwrap();

        assert_eq!(result.already_stored.len(), 2);
        assert!(result.already_stored.contains(&100));
        assert!(result.already_stored.contains(&200));

        assert_eq!(result.can_offload.len(), 2);
        assert!(result.can_offload.contains(&300));
        assert!(result.can_offload.contains(&400));
    }

    #[tokio::test]
    async fn test_local_registry_match_sequence() {
        let registry = LocalRegistry::new(1000);

        // Register contiguous hashes
        registry.register(TEST_BUCKET, &[100, 200, 300]).await.unwrap();

        // Full match
        let matched = registry
            .match_sequence_hashes(TEST_BUCKET, &[100, 200, 300])
            .await
            .unwrap();
        assert_eq!(matched.len(), 3);

        // Partial match (stops at 250)
        let matched = registry
            .match_sequence_hashes(TEST_BUCKET, &[100, 200, 250, 300])
            .await
            .unwrap();
        assert_eq!(matched.len(), 2);
        assert_eq!(matched[0], (100, 100));
        assert_eq!(matched[1], (200, 200));

        // No match (first doesn't exist)
        let matched = registry
            .match_sequence_hashes(TEST_BUCKET, &[50, 100, 200])
            .await
            .unwrap();
        assert_eq!(matched.len(), 0);
    }

    #[tokio::test]
    async fn test_local_registry_register_with_keys() {
        let registry = LocalRegistry::new(1000);

        // Register with explicit keys
        registry
            .register_with_keys(TEST_BUCKET, &[(100, 1000), (200, 2000)])
            .await
            .unwrap();

        // Check keys
        let matched = registry
            .match_sequence_hashes(TEST_BUCKET, &[100, 200])
            .await
            .unwrap();
        assert_eq!(matched[0], (100, 1000));
        assert_eq!(matched[1], (200, 2000));
    }

    #[tokio::test]
    async fn test_local_registry_contains() {
        let registry = LocalRegistry::new(1000);

        registry.register(TEST_BUCKET, &[100]).await.unwrap();

        assert!(registry.contains(TEST_BUCKET, 100).await.unwrap());
        assert!(!registry.contains(TEST_BUCKET, 200).await.unwrap());
    }

    #[tokio::test]
    async fn test_local_registry_get() {
        let registry = LocalRegistry::new(1000);

        registry
            .register_with_keys(TEST_BUCKET, &[(100, 999)])
            .await
            .unwrap();

        assert_eq!(registry.get(TEST_BUCKET, 100).await.unwrap(), Some(999));
        assert_eq!(registry.get(TEST_BUCKET, 200).await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_local_registry_clear() {
        let registry = LocalRegistry::new(1000);

        registry.register(TEST_BUCKET, &[100, 200, 300]).await.unwrap();
        registry.sync();
        assert_eq!(registry.len(), 3);

        registry.clear();
        registry.sync();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_local_registry_dedup_ratio() {
        let registry = LocalRegistry::new(1000);

        // Register 3 hashes
        registry.register(TEST_BUCKET, &[100, 200, 300]).await.unwrap();

        // Check 5 hashes: 3 already stored, 2 can offload
        let result = registry
            .can_offload(TEST_BUCKET, &[100, 200, 300, 400, 500])
            .await
            .unwrap();

        assert_eq!(result.dedup_count(), 3);
        assert_eq!(result.offload_count(), 2);
        assert!((result.dedup_ratio() - 0.6).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_local_registry_multi_bucket() {
        let registry = LocalRegistry::new(1000);

        // Register same hash in different buckets
        registry.register("bucket-a", &[100]).await.unwrap();
        registry.register("bucket-b", &[100]).await.unwrap();

        registry.sync();
        assert_eq!(registry.len(), 2);

        // Each bucket has its own data
        assert!(registry.contains("bucket-a", 100).await.unwrap());
        assert!(registry.contains("bucket-b", 100).await.unwrap());

        // Different bucket doesn't have the entry
        assert!(!registry.contains("bucket-c", 100).await.unwrap());
    }
}
