// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # M3: NIXL-Enabled Key-Value Storage
//!
//! M3 provides a high-performance key-value storage system with:
//! - Two-tier architecture: disk (RocksDB) + optional fast memory tier
//! - Byte-weighted LRU eviction for capacity management
//! - NIXL integration for RDMA access
//! - Atomic batch operations
//!
//! ## Example
//!
//! ```no_run
//! use dynamo_m3::{M3ConfigBuilder, M3Store};
//! use dynamo_memory::nixl::NixlAgent;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = Arc::new(NixlAgent::new("my_agent")?);
//!
//! let config = M3ConfigBuilder::default()
//!     .disk_tier_size("100GiB")
//!     .fast_tier_size("4GiB")
//!     .chunk_size("8MiB")
//!     .nixl_agent(agent)
//!     .build()?;
//!
//! let store = M3Store::new(config).await?;
//!
//! // Store data
//! store.put("my_key", b"my_value").await?;
//!
//! // Retrieve data
//! let buffer = store.get("my_key").await?;
//! println!("Retrieved: {:?}", buffer.as_slice());
//! # Ok(())
//! # }
//! ```

mod config;
mod disk_tier;
mod error;
mod fast_tier;
mod types;

pub use config::{M3Config, M3ConfigBuilder, SizeInput};
pub use error::{M3Error, Result};
pub use types::{HistogramSnapshot, M3Buffer, M3Store, SizeStatistics};

use disk_tier::DiskTier;
use fast_tier::FastTier;
use std::sync::Arc;

impl M3Store {
    /// Create a new M3Store instance
    pub async fn new(config: M3Config) -> Result<Self> {
        // Create disk tier
        let disk_tier = DiskTier::new(&config.db_path, config.disk_tier_size, config.chunk_size)?;

        // Create fast tier if enabled
        let fast_tier = if config.is_fast_tier_enabled() {
            let size = config.get_fast_tier_size()?;
            Some(FastTier::new(size, &config.nixl_agent)?)
        } else {
            None
        };

        let stats = Arc::new(SizeStatistics::default());

        Ok(Self {
            disk_tier,
            fast_tier,
            stats,
            config,
        })
    }

    /// Create a new key-value pair (fails if key exists)
    pub async fn create(&self, key: &str, data: &[u8]) -> Result<()> {
        if self.contains(key).await {
            return Err(M3Error::AlreadyExists(key.to_string()));
        }
        self.put(key, data).await
    }

    /// Update an existing key-value pair (fails if key doesn't exist)
    pub async fn update(&self, key: &str, data: &[u8]) -> Result<()> {
        if !self.contains(key).await {
            return Err(M3Error::NotFound(key.to_string()));
        }
        self.put(key, data).await
    }

    /// Put a key-value pair (create or update)
    pub async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        // Record statistics
        self.stats.record(data.len());

        // Check for eager eviction
        let current = self.disk_tier.current_bytes();
        let max = self.disk_tier.max_bytes();
        if self
            .stats
            .should_eager_evict(current, max, self.config.eager_eviction_threshold)
        {
            // Eager eviction happens automatically via Moka's capacity management
            tracing::debug!(
                "Disk tier at {}% capacity, eviction active",
                (current * 100 / max)
            );
        }

        // Write to disk tier
        self.disk_tier.put(key, data).await?;

        // Populate fast tier if enabled (best-effort)
        if let Some(ref fast_tier) = self.fast_tier {
            let _ = fast_tier.put(key, data).await;
        }

        Ok(())
    }

    /// Get value by key
    pub async fn get(&self, key: &str) -> Result<M3Buffer> {
        // Try fast tier first
        if let Some(ref fast_tier) = self.fast_tier
            && let Some(buffer) = fast_tier.get(key).await
        {
            return Ok(M3Buffer::FastTier(buffer));
        }

        // Fallback to disk tier
        let data = self.disk_tier.get(key).await?;

        // Populate fast tier on cache miss (best-effort)
        if let Some(ref fast_tier) = self.fast_tier {
            let _ = fast_tier.put(key, &data).await;
        }

        Ok(M3Buffer::Owned(data))
    }

    /// Get value and copy into provided buffer
    pub async fn get_in_place(&self, key: &str, dst: &mut [u8]) -> Result<()> {
        let buffer = self.get(key).await?;
        buffer.copy_to(dst)?;
        Ok(())
    }

    /// Check if key exists
    pub async fn contains(&self, key: &str) -> bool {
        self.disk_tier.contains(key).await
    }

    /// Remove key and return value if it existed
    pub async fn remove(&self, key: &str) -> Option<Vec<u8>> {
        // Invalidate fast tier
        if let Some(ref fast_tier) = self.fast_tier {
            fast_tier.invalidate(key).await;
        }

        // Remove from disk
        self.disk_tier.remove(key).await.ok().flatten()
    }

    /// Batch create (atomic all-or-nothing)
    pub async fn create_batch(&self, items: &[(&str, &[u8])]) -> Result<()> {
        // Check none exist
        for (key, _) in items {
            if self.contains(key).await {
                return Err(M3Error::BatchFailed(format!("Key already exists: {}", key)));
            }
        }
        self.put_batch(items).await
    }

    /// Batch update (atomic all-or-nothing)
    pub async fn update_batch(&self, items: &[(&str, &[u8])]) -> Result<()> {
        // Check all exist
        for (key, _) in items {
            if !self.contains(key).await {
                return Err(M3Error::BatchFailed(format!("Key not found: {}", key)));
            }
        }
        self.put_batch(items).await
    }

    /// Batch put (atomic all-or-nothing)
    pub async fn put_batch(&self, items: &[(&str, &[u8])]) -> Result<()> {
        // Record statistics
        for (_, data) in items {
            self.stats.record(data.len());
        }

        // Write to disk tier (atomic)
        self.disk_tier.put_batch(items).await?;

        // Populate fast tier (best-effort)
        if let Some(ref fast_tier) = self.fast_tier {
            for (key, data) in items {
                let _ = fast_tier.put(key, data).await;
            }
        }

        Ok(())
    }

    /// Batch get (atomic all-or-nothing)
    pub async fn get_batch(&self, keys: &[&str]) -> Result<Vec<M3Buffer>> {
        let mut results = Vec::with_capacity(keys.len());

        // Try to get all from fast tier first
        let mut missing_keys = Vec::new();
        let mut missing_indices = Vec::new();

        if let Some(ref fast_tier) = self.fast_tier {
            for (idx, key) in keys.iter().enumerate() {
                if let Some(buffer) = fast_tier.get(key).await {
                    results.push((idx, M3Buffer::FastTier(buffer)));
                } else {
                    missing_keys.push(*key);
                    missing_indices.push(idx);
                }
            }
        } else {
            missing_keys = keys.to_vec();
            missing_indices = (0..keys.len()).collect();
        }

        // Get missing keys from disk tier
        if !missing_keys.is_empty() {
            let disk_results = self.disk_tier.get_batch(&missing_keys).await?;

            // Populate fast tier for cache misses
            if let Some(ref fast_tier) = self.fast_tier {
                for (key, data) in missing_keys.iter().zip(&disk_results) {
                    let _ = fast_tier.put(key, data).await;
                }
            }

            for (idx, data) in missing_indices.into_iter().zip(disk_results) {
                results.push((idx, M3Buffer::Owned(data)));
            }
        }

        // Sort by original index and extract buffers
        results.sort_by_key(|(idx, _)| *idx);
        Ok(results.into_iter().map(|(_, buf)| buf).collect())
    }

    /// Get statistics
    pub fn stats(&self) -> &SizeStatistics {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &M3Config {
        &self.config
    }

    /// Get disk tier usage
    pub fn disk_usage(&self) -> usize {
        self.disk_tier.current_bytes()
    }

    /// Get fast tier usage (if enabled)
    pub fn fast_tier_usage(&self) -> Option<u64> {
        self.fast_tier.as_ref().map(|t| t.current_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_memory::nixl::NixlAgent;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_config() -> M3Config {
        let agent = Arc::new(NixlAgent::new("test_m3").expect("failed to create agent"));
        let dir = tempdir().unwrap();

        M3ConfigBuilder::default()
            .disk_tier_size(100 << 20) // 100MB
            .fast_tier_size(10 << 20) // 10MB
            .chunk_size(8 << 20) // 8MB
            .eager_eviction_threshold(0.85)
            .nixl_agent(agent)
            .db_path(dir.into_path())
            .build()
            .expect("failed to build config")
    }

    #[tokio::test]
    async fn test_m3_store_put_get() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        let data = vec![42u8; 1000];
        store.put("test_key", &data).await.unwrap();

        let buffer = store.get("test_key").await.unwrap();
        assert_eq!(buffer.as_slice(), data.as_slice());
    }

    #[tokio::test]
    async fn test_m3_store_create_already_exists() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        store.create("test_key", b"value1").await.unwrap();

        let result = store.create("test_key", b"value2").await;
        assert!(matches!(result, Err(M3Error::AlreadyExists(_))));
    }

    #[tokio::test]
    async fn test_m3_store_update_not_found() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        let result = store.update("nonexistent", b"value").await;
        assert!(matches!(result, Err(M3Error::NotFound(_))));
    }

    #[tokio::test]
    async fn test_m3_store_remove() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        let data = vec![42u8; 1000];
        store.put("test_key", &data).await.unwrap();

        let removed = store.remove("test_key").await;
        assert_eq!(removed, Some(data));

        assert!(!store.contains("test_key").await);
    }

    #[tokio::test]
    async fn test_m3_store_batch_operations() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        let items = vec![
            ("key1", &b"value1"[..]),
            ("key2", &b"value2"[..]),
            ("key3", &b"value3"[..]),
        ];

        store.put_batch(&items).await.unwrap();

        let keys = vec!["key1", "key2", "key3"];
        let buffers = store.get_batch(&keys).await.unwrap();

        assert_eq!(buffers[0].as_slice(), b"value1");
        assert_eq!(buffers[1].as_slice(), b"value2");
        assert_eq!(buffers[2].as_slice(), b"value3");
    }

    #[tokio::test]
    async fn test_m3_store_get_in_place() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        let data = vec![42u8; 1000];
        store.put("test_key", &data).await.unwrap();

        let mut dst = vec![0u8; 1000];
        store.get_in_place("test_key", &mut dst).await.unwrap();

        assert_eq!(dst, data);
    }

    #[tokio::test]
    async fn test_m3_store_statistics() {
        let config = create_test_config();
        let store = M3Store::new(config).await.unwrap();

        store.put("key1", &vec![0u8; 1000]).await.unwrap();
        store.put("key2", &vec![0u8; 2000]).await.unwrap();
        store.put("key3", &vec![0u8; 3000]).await.unwrap();

        let stats = store.stats();
        assert_eq!(stats.count(), 3);
        assert_eq!(stats.total_bytes(), 6000);
        assert_eq!(stats.average_size(), 2000);
    }
}
