// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Role-specific configuration for leader and worker components.

use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{DiskCacheConfig, HostCacheConfig, NovaBackendConfig};

/// Leader-specific configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct LeaderConfig {
    /// Override Nova backend configuration for leader.
    /// If None, uses the base nova.backend configuration.
    #[validate(nested)]
    pub nova: Option<NovaBackendConfig>,

    /// Host cache (G2 tier) configuration.
    /// Used to configure pinned host memory cache on workers.
    #[validate(nested)]
    #[serde(default)]
    pub host_cache: HostCacheConfig,

    /// Disk cache (G3 tier) configuration.
    /// Used to configure persistent disk cache on workers.
    #[validate(nested)]
    pub disk_cache: Option<DiskCacheConfig>,
}

/// Worker-specific configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct WorkerConfig {
    /// Override Nova backend configuration for worker.
    /// If None, uses the base nova.backend configuration.
    #[validate(nested)]
    pub nova: Option<NovaBackendConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_leader_config() {
        let config = LeaderConfig::default();
        assert!(config.nova.is_none());
    }

    #[test]
    fn test_default_worker_config() {
        let config = WorkerConfig::default();
        assert!(config.nova.is_none());
    }

    #[test]
    fn test_leader_with_override() {
        let config = LeaderConfig {
            nova: Some(NovaBackendConfig {
                tcp_addr: Some("10.0.0.1".to_string()),
                tcp_interface: None,
                tcp_port: 9000,
            }),
            host_cache: HostCacheConfig::default(),
            disk_cache: None,
        };
        assert!(config.nova.is_some());
        let backend = config.nova.unwrap();
        assert_eq!(backend.tcp_addr, Some("10.0.0.1".to_string()));
        assert_eq!(backend.tcp_port, 9000);
    }

    #[test]
    fn test_leader_with_cache_configs() {
        let config = LeaderConfig {
            nova: None,
            host_cache: HostCacheConfig {
                cache_size_gb: Some(16.0),
                num_blocks: None,
            },
            disk_cache: Some(DiskCacheConfig {
                cache_size_gb: Some(100.0),
                num_blocks: None,
                use_gds: true,
                storage_path: Some("/mnt/nvme".into()),
            }),
        };

        assert!(config.host_cache.is_enabled());
        assert_eq!(config.host_cache.cache_size_gb, Some(16.0));

        let disk = config.disk_cache.unwrap();
        assert!(disk.is_enabled());
        assert!(disk.use_gds);
    }
}
