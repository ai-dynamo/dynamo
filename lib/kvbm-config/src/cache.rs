// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache tier configuration for KVBM.
//!
//! Defines configuration for G2 (host/pinned memory) and G3 (disk) cache tiers.
//! The leader uses this configuration to coordinate cache tier creation on workers.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Host cache configuration (G2 tier - pinned CPU memory).
///
/// The host cache provides a staging area for KV blocks between GPU and disk.
/// Memory is allocated as pinned (page-locked) for efficient DMA transfers.
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct HostCacheConfig {
    /// Cache size in gigabytes.
    /// Used to compute num_blocks if not explicitly set.
    pub cache_size_gb: Option<f64>,

    /// Explicit number of blocks for the host cache.
    /// Takes priority over cache_size_gb if set.
    pub num_blocks: Option<usize>,
}

impl HostCacheConfig {
    /// Compute the number of blocks based on configuration and block size.
    ///
    /// Priority: explicit num_blocks > computed from cache_size_gb
    ///
    /// # Arguments
    /// * `bytes_per_block` - Size of each block in bytes
    ///
    /// # Returns
    /// Number of blocks, or None if neither num_blocks nor cache_size_gb is set.
    pub fn compute_num_blocks(&self, bytes_per_block: usize) -> Option<usize> {
        self.num_blocks.or_else(|| {
            self.cache_size_gb.map(|gb| {
                // Convert GB to bytes and divide by block size
                ((gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
            })
        })
    }

    /// Check if host cache is enabled (has any configuration).
    pub fn is_enabled(&self) -> bool {
        self.num_blocks.is_some() || self.cache_size_gb.is_some()
    }
}

/// Disk cache configuration (G3 tier - persistent storage).
///
/// The disk cache provides extended capacity for KV blocks beyond GPU and host memory.
/// Can use either GPU Direct Storage (GDS) for direct GPU-disk transfers or POSIX
/// for regular file I/O.
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct DiskCacheConfig {
    /// Cache size in gigabytes.
    /// Used to compute num_blocks if not explicitly set.
    pub cache_size_gb: Option<f64>,

    /// Explicit number of blocks for the disk cache.
    /// Takes priority over cache_size_gb if set.
    pub num_blocks: Option<usize>,

    /// Use GPU Direct Storage (GDS) if available.
    /// When true, enables GDS_MT backend for direct GPU-disk transfers.
    /// When false or GDS unavailable, falls back to POSIX backend.
    #[serde(default)]
    pub use_gds: bool,

    /// Storage path for disk cache files.
    /// If None, a default path will be used.
    pub storage_path: Option<PathBuf>,
}

impl DiskCacheConfig {
    /// Compute the number of blocks based on configuration and block size.
    ///
    /// Priority: explicit num_blocks > computed from cache_size_gb
    ///
    /// # Arguments
    /// * `bytes_per_block` - Size of each block in bytes
    ///
    /// # Returns
    /// Number of blocks, or None if neither num_blocks nor cache_size_gb is set.
    pub fn compute_num_blocks(&self, bytes_per_block: usize) -> Option<usize> {
        self.num_blocks.or_else(|| {
            self.cache_size_gb.map(|gb| {
                // Convert GB to bytes and divide by block size
                ((gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
            })
        })
    }

    /// Check if disk cache is enabled (has any configuration).
    pub fn is_enabled(&self) -> bool {
        self.num_blocks.is_some() || self.cache_size_gb.is_some()
    }
}

/// Top-level cache configuration.
///
/// Groups host (G2) and disk (G3) cache configurations together.
/// Use Figment profiles to configure different cache settings for leader vs worker.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct CacheConfig {
    /// Host cache (G2 tier) - pinned CPU memory.
    #[serde(default)]
    #[validate(nested)]
    pub host: HostCacheConfig,

    /// Disk cache (G3 tier) - persistent storage.
    /// Optional - only configure if disk caching is needed.
    #[validate(nested)]
    pub disk: Option<DiskCacheConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_cache_default() {
        let config = HostCacheConfig::default();
        assert!(config.cache_size_gb.is_none());
        assert!(config.num_blocks.is_none());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_host_cache_explicit_blocks() {
        let config = HostCacheConfig {
            num_blocks: Some(1000),
            cache_size_gb: Some(10.0), // Should be ignored
        };

        // With 1MB blocks, explicit num_blocks takes priority
        let bytes_per_block = 1_000_000;
        assert_eq!(config.compute_num_blocks(bytes_per_block), Some(1000));
        assert!(config.is_enabled());
    }

    #[test]
    fn test_host_cache_from_size_gb() {
        let config = HostCacheConfig {
            num_blocks: None,
            cache_size_gb: Some(10.0), // 10 GB
        };

        // With 1MB blocks: 10GB / 1MB = 10,000 blocks
        let bytes_per_block = 1_000_000;
        assert_eq!(config.compute_num_blocks(bytes_per_block), Some(10_000));
        assert!(config.is_enabled());
    }

    #[test]
    fn test_disk_cache_default() {
        let config = DiskCacheConfig::default();
        assert!(config.cache_size_gb.is_none());
        assert!(config.num_blocks.is_none());
        assert!(!config.use_gds);
        assert!(config.storage_path.is_none());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_disk_cache_with_gds() {
        let config = DiskCacheConfig {
            num_blocks: Some(5000),
            cache_size_gb: None,
            use_gds: true,
            storage_path: Some(PathBuf::from("/mnt/nvme/kv_cache")),
        };

        assert!(config.use_gds);
        assert_eq!(
            config.storage_path,
            Some(PathBuf::from("/mnt/nvme/kv_cache"))
        );
        assert!(config.is_enabled());
    }
}
