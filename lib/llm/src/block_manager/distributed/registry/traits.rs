// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait definitions for the distributed object registry.
//!
//! These are the PUBLIC traits that consumers use. Implementation details
//! (storage backend, transport mechanism) are hidden behind these interfaces.

use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::protocol::SequenceHash;
use super::types::{HubStats, ObjectKey, OffloadResult};
use crate::block_manager::block::transfer::remote::RemoteKey;

/// Distributed registry client trait.
///
/// This is the main interface for workers to interact with the distributed
/// object registry. It provides methods for:
/// - Registering hashes after storing to object storage
/// - Checking which hashes can be offloaded (deduplication)
/// - Matching sequence hashes to find what can be loaded
///
/// All operations are scoped to a bucket_name (e.g., "instance-1", "worker-0").
/// This enables cross-instance deduplication while keeping data organized.
///
/// # Implementations
/// - `ZmqRegistryClient`: ZMQ REQ/REP + PUB/SUB transport
/// - `LocalRegistry`: In-process for testing/single-node
/// - `RdmaRegistryClient`: RDMA transport (future)
///
/// # Example
/// ```ignore
/// // Check what needs to be stored
/// let result = registry.can_offload("my-bucket", &hashes).await?;
/// for hash in &result.can_offload {
///     store_to_object(*hash).await?;
/// }
/// registry.register("my-bucket", &result.can_offload).await?;
///
/// // Find what can be loaded
/// let matched = registry.match_sequence_hashes("my-bucket", &hashes).await?;
/// for (hash, key) in matched {
///     load_from_object(key).await?;
/// }
/// ```
#[async_trait]
pub trait DistributedRegistry: Send + Sync {
    /// Register sequence hashes after storing to object storage.
    ///
    /// The object key is assumed to be the same as the sequence hash
    /// (common pattern: hash is used as object object key).
    ///
    /// This may be batched internally for efficiency.
    async fn register(&self, bucket_name: &str, sequence_hashes: &[SequenceHash]) -> Result<()>;

    /// Register with explicit object keys (when key != hash).
    async fn register_with_keys(
        &self,
        bucket_name: &str,
        entries: &[(SequenceHash, ObjectKey)],
    ) -> Result<()>;

    /// Check which sequence hashes can be offloaded to object storage.
    ///
    /// Returns the subset of hashes that are NOT already in the registry
    /// (i.e., safe to store without duplication).
    ///
    /// # Returns
    /// `OffloadResult` containing:
    /// - `can_offload`: Hashes NOT in registry (need to be stored)
    /// - `already_stored`: Hashes already in registry (skip storing)
    ///
    /// # Example
    /// ```ignore
    /// let result = registry.can_offload("my-bucket", &[A, B, C, D]).await?;
    /// // If A, C already in object:
    /// // result.can_offload = [B, D]      // Store these
    /// // result.already_stored = [A, C]   // Skip these
    /// ```
    async fn can_offload(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<OffloadResult>;

    /// Match sequence hashes against registry.
    ///
    /// Returns contiguous prefix that exists in object storage (stops at first miss).
    /// Each result includes the object key needed to fetch from object.
    ///
    /// This mirrors the local `ObjectRegistry::match_sequence_hashes` behavior.
    ///
    /// # Example
    /// ```ignore
    /// // Query: [A, B, C, D, E] where A, B, C exist in object
    /// let matched = registry.match_sequence_hashes("my-bucket", &[A, B, C, D, E]).await?;
    /// // Returns: [(A, key_a), (B, key_b), (C, key_c)]
    /// // Stops at D because it doesn't exist
    /// ```
    async fn match_sequence_hashes(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<Vec<(SequenceHash, ObjectKey)>>;

    // =========================================================================
    // CONVENIENCE METHODS (default implementations)
    // =========================================================================

    /// Check if a single hash exists in object storage.
    async fn contains(&self, bucket_name: &str, hash: SequenceHash) -> Result<bool> {
        let result = self.can_offload(bucket_name, &[hash]).await?;
        Ok(!result.can_offload.contains(&hash))
    }

    /// Get object key for a single hash.
    async fn get(&self, bucket_name: &str, hash: SequenceHash) -> Result<Option<ObjectKey>> {
        let results = self.match_sequence_hashes(bucket_name, &[hash]).await?;
        Ok(results.first().map(|(_, key)| *key))
    }

    /// Flush any pending registrations.
    ///
    /// For implementations that batch registrations, this ensures all
    /// pending registrations are sent immediately.
    async fn flush(&self) -> Result<()>;

    /// Blocking version of match_sequence_hashes for use in sync contexts.
    ///
    fn match_sequence_hashes_blocking(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<Vec<SequenceHash>>
    where
        Self: Sync,
    {
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available"))?;

        let bucket = bucket_name.to_string();
        let hashes_vec = hashes.to_vec();

        // Use catch_unwind to handle single-threaded runtime gracefully
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::task::block_in_place(|| {
                handle.block_on(async {
                    self.match_sequence_hashes(&bucket, &hashes_vec).await
                })
            })
        }))
        .map_err(|_| anyhow::anyhow!("block_in_place failed (single-threaded runtime?)"))?
        .map(|matches| matches.into_iter().map(|(h, _)| h).collect())
    }

    /// Match RemoteKeys against the registry.
    ///
    /// Unlike `match_sequence_hashes`, this searches across multiple buckets
    /// using full `RemoteKey`s (bucket + key pairs). Returns the subset that exist.
    ///
    /// # Example
    /// ```ignore
    /// // Generate candidate keys for all workers
    /// let candidates: Vec<RemoteKey> = hashes.iter()
    ///     .flat_map(|&hash| key_builder.build_all(hash))
    ///     .collect();
    ///
    /// // Find which actually exist
    /// let matched = registry.match_remote_keys(&candidates).await?;
    /// // matched contains RemoteKeys with correct bucket info for fetching
    /// ```
    async fn match_remote_keys(&self, keys: &[RemoteKey]) -> Result<Vec<RemoteKey>>;

    /// Match contiguous prefix of RemoteKeys.
    ///
    /// Stops at first key not found. Returns RemoteKeys that exist.
    async fn match_remote_keys_prefix(&self, keys: &[RemoteKey]) -> Result<Vec<RemoteKey>>;

    /// Register RemoteKeys as existing in the distributed registry.
    ///
    /// Called after successfully storing blocks to remote storage.
    async fn register_remote_keys(&self, keys: &[RemoteKey]) -> Result<()>;
}

/// Registry hub trait (runs on leader/coordinator).
///
/// This trait defines the server-side interface for the registry hub.
/// The hub is the single source of truth for the distributed registry.
///
/// # Implementations
/// - `ZmqRegistryHub`: ZMQ-based hub
/// - `RdmaRegistryHub`: RDMA-based hub (future)
#[async_trait]
pub trait RegistryHub: Send + Sync {
    /// Start serving requests.
    ///
    /// This method blocks until the cancellation token is triggered.
    async fn serve(&self, cancel: CancellationToken) -> Result<()>;

    /// Get current entry count.
    fn len(&self) -> u64;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics.
    fn stats(&self) -> HubStats;

    /// Get capacity.
    fn capacity(&self) -> u64;

    /// Get utilization (len / capacity).
    fn utilization(&self) -> f64 {
        let cap = self.capacity();
        if cap == 0 {
            0.0
        } else {
            self.len() as f64 / cap as f64
        }
    }
}


