// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage module for distributed block management.
//!
//! This module provides traits and implementations for storing KV cache blocks
//! in object storage systems like S3/MinIO.
//!
//! # Architecture
//!
//! Traits are defined here; implementations are in feature-gated submodules:
//! - [`ObjectBlockOps`](crate::object::ObjectBlockOps) - High-level block operations (put, get, has)
//! - [`ObjectLockManager`](crate::object::ObjectLockManager) - Distributed locking for coordinated offloads
//!
//! Consumers should use factory functions to obtain trait objects without
//! depending on specific feature flags.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;

use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::layout::LayoutConfig;
use kvbm_physical::transfer::PhysicalLayout;

#[cfg(feature = "s3")]
pub mod s3;

pub mod nixl;

// ============================================================================
// Key Formatting
// ============================================================================

/// Trait for converting SequenceHash to object storage keys.
///
/// Implementations can embed rank, namespace, or any other prefix/suffix
/// to ensure key uniqueness across SPMD workers or other contexts.
pub trait KeyFormatter: Send + Sync {
    /// Convert a sequence hash to an object storage key string.
    fn format_key(&self, hash: &SequenceHash) -> String;
}

/// Default key formatter - uses Display representation of PositionalLineageHash.
///
/// Produces keys like: `0:abc123` or `5:abc123:def456` (position:current\[:parent\])
/// using base58 encoding for hash fragments.
/// Suitable for single-worker scenarios or testing.
#[derive(Debug, Clone, Default)]
pub struct DefaultKeyFormatter;

impl KeyFormatter for DefaultKeyFormatter {
    fn format_key(&self, hash: &SequenceHash) -> String {
        hash.to_string()
    }
}

/// Rank-prefixed key formatter for SPMD workers.
///
/// Formats keys as `{rank}/{display_hash}` to ensure uniqueness across workers
/// writing the same logical blocks. The hash uses the Display representation
/// (e.g., `0/5:abc123:def456`).
#[derive(Debug, Clone)]
pub struct RankPrefixedKeyFormatter {
    rank: usize,
}

impl RankPrefixedKeyFormatter {
    /// Create a new rank-prefixed formatter.
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }

    /// Get the rank.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl KeyFormatter for RankPrefixedKeyFormatter {
    fn format_key(&self, hash: &SequenceHash) -> String {
        format!("{}/{}", self.rank, hash)
    }
}

/// Create a key formatter appropriate for the given rank.
///
/// Returns a `RankPrefixedKeyFormatter` if rank is provided,
/// otherwise returns a `DefaultKeyFormatter`.
pub fn create_key_formatter(rank: Option<usize>) -> Arc<dyn KeyFormatter> {
    match rank {
        Some(r) => Arc::new(RankPrefixedKeyFormatter::new(r)),
        None => Arc::new(DefaultKeyFormatter),
    }
}

/// Extension methods for LayoutConfig to support object storage operations.
pub trait LayoutConfigExt {
    /// Compute the size of a single block in bytes.
    fn block_size_bytes(&self) -> usize;

    /// Compute the size of a single memory region in bytes.
    fn region_size(&self) -> usize;
}

impl LayoutConfigExt for LayoutConfig {
    fn block_size_bytes(&self) -> usize {
        self.num_layers
            .saturating_mul(self.outer_dim)
            .saturating_mul(self.page_size)
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }

    fn region_size(&self) -> usize {
        self.page_size
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }
}

/// Low-level object storage client trait.
pub trait ObjectClient: Send + Sync {
    /// Check if an object exists.
    fn has_object(&self, key: &[u8]) -> anyhow::Result<bool>;

    /// Put an object.
    fn put_object(&self, key: &[u8], data: &[&[u8]]) -> anyhow::Result<()>;

    /// Get an object.
    fn get_object(&self, key: &[u8], data: &mut [&mut [u8]]) -> anyhow::Result<()>;
}

/// Unified object block operations trait.
///
/// This trait provides high-level operations for storing and retrieving
/// KV cache blocks in object storage (e.g., S3, MinIO).
///
/// Uses `LogicalLayoutHandle` to identify source/destination layouts. In distributed
/// mode, workers resolve the logical handle to their own physical layouts. This allows
/// the leader (which doesn't have physical layouts) to use the same trait.
///
/// Uses `'static` BoxFuture for runtime flexibility - implementations clone/Arc
/// what they need from self. Takes owned Vecs for simplicity; keys are returned
/// in results so callers can correlate success/failure.
///
/// Implemented by:
/// - `S3ObjectBlockClient` - direct S3 operations (has_blocks only; put/get require physical layout)
/// - `DirectWorker` - resolves logical handle to physical layout, then delegates
/// - `CoordinatedWorker` - delegates to inner worker
/// - `LeaderObjectClient` - coordinates workers for distributed uploads
pub trait ObjectBlockOps: Send + Sync {
    /// Check if blocks exist in object storage.
    ///
    /// Returns a vector of (hash, size_option) pairs where:
    /// - Some(size) indicates the block exists with the given size in bytes
    /// - None indicates the block does not exist or an error occurred
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>>;

    /// Put blocks to object storage.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `src_layout` - Logical layout handle identifying the source (workers resolve to physical)
    /// * `block_ids` - Block IDs within the layout to upload
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully stored
    /// - Err(hash) indicates the block failed to store
    ///
    /// # Note
    /// For `S3ObjectBlockClient`, this will error - use `put_blocks_with_layout` instead.
    /// Workers should resolve the logical handle to their physical layout first.
    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>>;

    /// Get blocks from object storage.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `dst_layout` - Logical layout handle identifying the destination (workers resolve to physical)
    /// * `block_ids` - Block IDs within the layout to download into
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully retrieved
    /// - Err(hash) indicates the block failed to retrieve
    ///
    /// # Note
    /// For `S3ObjectBlockClient`, this will error - use `get_blocks_with_layout` instead.
    /// Workers should resolve the logical handle to their physical layout first.
    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>>;

    // =========================================================================
    // Physical Layout Methods (for workers that can resolve handles)
    // =========================================================================

    /// Put blocks to object storage using a resolved physical layout.
    ///
    /// This method is called by workers after resolving a logical handle to
    /// their physical layout. The default implementation errors; storage backends
    /// like `S3ObjectBlockClient` override this with actual upload logic.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout containing the block data
    /// * `block_ids` - Block IDs within the layout to upload
    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    /// Get blocks from object storage into a resolved physical layout.
    ///
    /// This method is called by workers after resolving a logical handle to
    /// their physical layout. The default implementation errors; storage backends
    /// like `S3ObjectBlockClient` override this with actual download logic.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout to write the block data into
    /// * `block_ids` - Block IDs within the layout to download into
    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

// ============================================================================
// Object Lock Manager Trait
// ============================================================================

/// Lock file content structure for distributed locking.
///
/// The lock file is stored as JSON in object storage at `{sequence_hash}.lock`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockFileContent {
    /// Unique identifier of the instance that holds the lock
    pub instance_id: String,
    /// When the lock was acquired (ISO 8601 timestamp)
    pub acquired_at: String,
    /// When the lock expires (ISO 8601 timestamp)
    pub deadline: String,
}

/// Object lock manager trait for distributed locking in object storage.
///
/// This trait provides the locking semantics for the object offload pipeline:
/// 1. Check `.meta` file to see if block is already offloaded
/// 2. Try to acquire `.lock` file with conditional PUT
/// 3. Create `.meta` file after successful offload
/// 4. Release `.lock` file after completion
///
/// # Locking Flow
///
/// ```text
/// has_meta() -> false -> try_acquire_lock() -> true -> execute transfer -> create_meta() -> release_lock()
///                                           -> false -> skip (another instance owns it)
///            -> true -> skip (already offloaded)
/// ```
pub trait ObjectLockManager: Send + Sync {
    /// Check if meta file exists (block already offloaded).
    ///
    /// Returns `true` if `{hash}.meta` exists, meaning the block has been
    /// successfully offloaded and should be skipped.
    fn has_meta(&self, hash: SequenceHash) -> BoxFuture<'static, Result<bool>>;

    /// Try to acquire a lock for the given block.
    ///
    /// This method:
    /// 1. Attempts conditional PUT of `{hash}.lock` with `If-None-Match: *`
    /// 2. If lock exists, reads it to check deadline
    /// 3. If deadline is breached (> timeout), overwrites the lock
    ///
    /// Returns:
    /// - `Ok(true)` if lock was acquired or overwritten
    /// - `Ok(false)` if another instance owns a valid lock
    /// - `Err(...)` for other errors
    fn try_acquire_lock(&self, hash: SequenceHash) -> BoxFuture<'static, Result<bool>>;

    /// Create the meta file after successful offload.
    ///
    /// This marks the block as successfully offloaded by creating `{hash}.meta`.
    fn create_meta(&self, hash: SequenceHash) -> BoxFuture<'static, Result<()>>;

    /// Release the lock by deleting the lock file.
    ///
    /// Deletes `{hash}.lock` after the transfer is complete.
    fn release_lock(&self, hash: SequenceHash) -> BoxFuture<'static, Result<()>>;
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create an object client from configuration.
///
/// Returns a trait object so consumers don't need to depend on the `s3` feature.
/// The implementation is selected based on the configuration type.
///
/// # Arguments
/// * `config` - Object storage configuration
/// * `rank` - Optional worker rank for key prefixing (None for leader)
/// * `nixl_agent` - NIXL agent (required when `config.client` is `Nixl`; the
///   OBJ backend must already be registered on the agent via
///   [`nixl::add_obj_backend`]).
///
/// # Errors
/// Returns an error if the object client cannot be initialized or if the
/// required feature is not enabled.
#[cfg(feature = "s3")]
pub async fn create_object_client(
    config: &kvbm_config::ObjectConfig,
    rank: Option<usize>,
    nixl_agent: Option<dynamo_memory::nixl::NixlAgent>,
) -> Result<Arc<dyn ObjectBlockOps>> {
    use kvbm_config::ObjectClientConfig;
    use s3::{S3Config, S3ObjectBlockClient};

    let key_formatter = create_key_formatter(rank);

    match &config.client {
        ObjectClientConfig::S3(s3_config) => {
            let config = S3Config::from_object_config(s3_config);
            let client = S3ObjectBlockClient::with_key_formatter(config, key_formatter).await?;
            Ok(Arc::new(client))
        }
        ObjectClientConfig::Nixl(nixl_config) => {
            let agent = nixl_agent.ok_or_else(|| {
                anyhow::anyhow!(
                    "Nixl object storage requires a NixlAgent with the OBJ backend initialised. \
                     Pass the agent from KvbmRuntime::nixl_agent()."
                )
            })?;
            let client =
                nixl::NixlObjectBlockClient::from_config(agent, nixl_config, rank).await?;
            Ok(Arc::new(client))
        }
    }
}

/// Fallback when S3 feature is disabled.
#[cfg(not(feature = "s3"))]
pub async fn create_object_client(
    _config: &kvbm_config::ObjectConfig,
    _rank: Option<usize>,
    _nixl_agent: Option<dynamo_memory::nixl::NixlAgent>,
) -> Result<Arc<dyn ObjectBlockOps>> {
    anyhow::bail!("Object storage requires the 's3' feature to be enabled")
}

/// Create a lock manager from configuration.
///
/// Returns a trait object so consumers don't need to depend on the `s3` feature.
///
/// # Arguments
/// * `config` - Object storage configuration
/// * `instance_id` - Unique identifier for this instance (used in lock files)
///
/// # Errors
/// Returns an error if the lock manager cannot be initialized or if the
/// required feature is not enabled.
#[cfg(feature = "s3")]
pub async fn create_lock_manager(
    config: &kvbm_config::ObjectConfig,
    instance_id: String,
) -> Result<Arc<dyn ObjectLockManager>> {
    use kvbm_config::ObjectClientConfig;
    use s3::{S3Config, S3LockManager, S3ObjectBlockClient};

    match &config.client {
        ObjectClientConfig::S3(s3_config) => {
            let config = S3Config::from_object_config(s3_config);
            // Lock manager uses default key formatter (no rank prefix for lock/meta files)
            let client = Arc::new(S3ObjectBlockClient::new(config).await?);
            let manager = S3LockManager::new(client, instance_id);
            Ok(Arc::new(manager))
        }
        ObjectClientConfig::Nixl(nixl_config) => {
            // Distributed locking relies on conditional PUT semantics (If-None-Match).
            // NIXL's OBJ backend does not expose conditional writes, so we fall back
            // to an SDK-backed lock manager derived from the embedded S3 config.
            use kvbm_config::NixlObjectConfig;
            match nixl_config {
                NixlObjectConfig::S3(nixl_s3) => {
                    let s3_config = nixl::nixl_s3_to_sdk_config(nixl_s3);
                    let client = Arc::new(s3::S3ObjectBlockClient::new(s3_config).await?);
                    let manager = s3::S3LockManager::new(client, instance_id);
                    Ok(Arc::new(manager))
                }
            }
        }
    }
}

/// Fallback when S3 feature is disabled.
#[cfg(not(feature = "s3"))]
pub async fn create_lock_manager(
    _config: &kvbm_config::ObjectConfig,
    _instance_id: String,
) -> Result<Arc<dyn ObjectLockManager>> {
    anyhow::bail!("Object storage requires the 's3' feature to be enabled")
}

// ============================================================================
// Tests
// ============================================================================

/// Verify that `create_lock_manager` falls back to the SDK-backed `S3LockManager`
/// for `ObjectClientConfig::Nixl` variants.
///
/// The NIXL OBJ backend has no conditional-PUT primitive, so distributed locking
/// must always use the embedded S3 config regardless of the outer `client` variant.
#[cfg(feature = "testing-s3")]
#[cfg(test)]
mod tests {
    use super::*;

    /// `create_lock_manager` for a Nixl config must succeed and return a working
    /// lock manager backed by S3 (not NIXL).
    ///
    /// We verify the "happy path" only — that the factory function completes
    /// without error and the returned lock manager can perform basic operations.
    #[tokio::test]
    async fn test_create_lock_manager_nixl_fallback_to_s3() {
        let endpoint =
            std::env::var("S3_TEST_ENDPOINT").unwrap_or_else(|_| "http://localhost:9876".into());

        let nixl_s3 = kvbm_config::NixlS3Config::with_endpoint(endpoint, "kvbm-lock-test");
        let config = kvbm_config::ObjectConfig {
            client: kvbm_config::ObjectClientConfig::Nixl(kvbm_config::NixlObjectConfig::S3(
                nixl_s3,
            )),
        };

        let manager = create_lock_manager(&config, "test-instance-id".into())
            .await
            .expect("create_lock_manager should succeed for Nixl config (falls back to S3)");

        // Exercise a round-trip: acquire lock, check meta, create meta, release.
        let hash = SequenceHash::new(0xFADE_FADE_u64, None, 0);

        // Initially no meta file.
        let has_meta = manager.has_meta(hash).await.unwrap();
        assert!(!has_meta, "meta should not exist before any offload");

        // Acquire lock.
        let acquired = manager.try_acquire_lock(hash).await.unwrap();
        assert!(acquired, "should acquire lock on first attempt");

        // Re-acquire with the same instance_id should succeed (we own it).
        // Note: with a different instance_id this would fail; we verify the
        // lock is held by checking that a second factory with a different id
        // cannot steal an active (non-expired) lock.
        let config2 = kvbm_config::ObjectConfig {
            client: kvbm_config::ObjectClientConfig::Nixl(kvbm_config::NixlObjectConfig::S3(
                kvbm_config::NixlS3Config::with_endpoint(
                    std::env::var("S3_TEST_ENDPOINT")
                        .unwrap_or_else(|_| "http://localhost:9876".into()),
                    "kvbm-lock-test",
                ),
            )),
        };
        let manager2 = create_lock_manager(&config2, "other-instance-id".into())
            .await
            .unwrap();
        let stolen = manager2.try_acquire_lock(hash).await.unwrap();
        assert!(!stolen, "non-expired lock should not be stolen by another instance");

        // Create meta file and release lock.
        manager.create_meta(hash).await.unwrap();
        manager.release_lock(hash).await.unwrap();

        // Meta should now be visible.
        let has_meta_after = manager.has_meta(hash).await.unwrap();
        assert!(has_meta_after, "meta should exist after create_meta");

        // Cleanup — S3ObjectBlockClient isn't exposed here so we use the manager's
        // internal client indirectly: just verify the state is consistent.
        // (Real cleanup is left to the test environment teardown.)
    }
}
