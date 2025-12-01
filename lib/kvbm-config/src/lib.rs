// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM Configuration Library
//!
//! Provides centralized configuration for Tokio, Rayon, Nova, and NixL runtimes.
//! Supports role-specific configuration for leader and worker components.

mod cache;
mod nixl;
mod nova;
mod rayon;
mod role;
mod tokio;

pub use cache::{DiskCacheConfig, HostCacheConfig};
pub use nixl::NixlConfig;
pub use nova::{NovaBackendConfig, NovaConfig, NovaDiscoveryConfig};
pub use rayon::RayonConfig;
pub use role::{LeaderConfig, WorkerConfig};
pub use tokio::TokioConfig;

use figment::{
    Figment, Metadata, Profile, Provider,
    providers::{Env, Format, Serialized, Toml},
    value::{Dict, Map},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::{Validate, ValidationErrors};

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to extract configuration: {0}")]
    Extraction(#[from] Box<figment::Error>),

    #[error("Configuration validation failed: {0}")]
    Validation(#[from] ValidationErrors),

    #[error("Configuration error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Top-level KVBM configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct KvbmConfig {
    #[validate(nested)]
    pub tokio: TokioConfig,

    #[validate(nested)]
    pub rayon: RayonConfig,

    #[validate(nested)]
    pub nova: NovaConfig,

    #[validate(nested)]
    pub nixl: NixlConfig,

    #[validate(nested)]
    pub leader: LeaderConfig,

    #[validate(nested)]
    pub worker: WorkerConfig,
}

impl KvbmConfig {
    /// Create a Figment configuration with all sources merged.
    ///
    /// Configuration sources in priority order (lowest to highest):
    /// 1. Code defaults
    /// 2. System config file at /opt/dynamo/etc/kvbm.toml
    /// 3. TOML file from KVBM_CONFIG_PATH environment variable
    /// 4. Environment variables (KVBM_* prefixed)
    pub fn figment() -> Figment {
        let config_path = std::env::var("KVBM_CONFIG_PATH").unwrap_or_default();

        Figment::new()
            .merge(Serialized::defaults(KvbmConfig::default()))
            .merge(Toml::file("/opt/dynamo/etc/kvbm.toml"))
            .merge(Toml::file(&config_path))
            // Tokio config: KVBM_TOKIO_WORKER_THREADS, KVBM_TOKIO_MAX_BLOCKING_THREADS
            .merge(
                Env::prefixed("KVBM_TOKIO_")
                    .map(|k| format!("tokio.{}", k.as_str().to_lowercase()).into()),
            )
            // Rayon config: KVBM_RAYON_NUM_THREADS
            .merge(
                Env::prefixed("KVBM_RAYON_")
                    .map(|k| format!("rayon.{}", k.as_str().to_lowercase()).into()),
            )
            // Nova backend config: KVBM_NOVA_BACKEND_TCP_ADDR, etc.
            .merge(
                Env::prefixed("KVBM_NOVA_BACKEND_")
                    .map(|k| format!("nova.backend.{}", k.as_str().to_lowercase()).into()),
            )
            // Nova discovery config: KVBM_NOVA_DISCOVERY_CLUSTER_ID, etc.
            .merge(
                Env::prefixed("KVBM_NOVA_DISCOVERY_")
                    .map(|k| format!("nova.discovery.{}", k.as_str().to_lowercase()).into()),
            )
            // Leader-specific nova overrides: KVBM_LEADER_NOVA_TCP_ADDR, etc.
            .merge(
                Env::prefixed("KVBM_LEADER_NOVA_")
                    .map(|k| format!("leader.nova.{}", k.as_str().to_lowercase()).into()),
            )
            // Worker-specific nova overrides: KVBM_WORKER_NOVA_TCP_ADDR, etc.
            .merge(
                Env::prefixed("KVBM_WORKER_NOVA_")
                    .map(|k| format!("worker.nova.{}", k.as_str().to_lowercase()).into()),
            )
            // NixL config: KVBM_NIXL_BACKENDS (comma-separated list)
            .merge(
                Env::prefixed("KVBM_NIXL_")
                    .map(|k| format!("nixl.{}", k.as_str().to_lowercase()).into()),
            )
            // Host cache config: KVBM_HOST_CACHE_SIZE_GB, KVBM_HOST_CACHE_NUM_BLOCKS
            .merge(
                Env::prefixed("KVBM_HOST_CACHE_")
                    .map(|k| format!("leader.host_cache.{}", k.as_str().to_lowercase()).into()),
            )
            // Disk cache config: KVBM_DISK_CACHE_SIZE_GB, KVBM_DISK_CACHE_NUM_BLOCKS, etc.
            .merge(
                Env::prefixed("KVBM_DISK_CACHE_")
                    .map(|k| format!("leader.disk_cache.{}", k.as_str().to_lowercase()).into()),
            )
    }

    /// Load configuration from default figment (env and files).
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment())
    }

    /// Extract configuration from any provider.
    ///
    /// Use this to load config from custom sources or to add programmatic overrides.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Merge tuple pairs for programmatic overrides (figment best practice)
    /// let config = KvbmConfig::extract_from(
    ///     KvbmConfig::figment()
    ///         .merge(("nova.backend.tcp_port", 8080u16))
    ///         .merge(("tokio.worker_threads", 4usize))
    /// )?;
    /// ```
    pub fn extract_from<T: Provider>(provider: T) -> Result<Self, ConfigError> {
        let config: Self = Figment::from(provider)
            .extract()
            .map_err(|e| ConfigError::Extraction(Box::new(e)))?;
        config.validate()?;
        Ok(config)
    }

    /// Build a figment from defaults, then merge a custom provider.
    ///
    /// Convenience method for adding programmatic overrides with highest priority.
    ///
    /// # Example
    /// ```rust,ignore
    /// let figment = KvbmConfig::figment_with(("nova.backend.tcp_port", 8080u16));
    /// let config = KvbmConfig::extract_from(figment)?;
    /// ```
    pub fn figment_with<T: Provider>(extra: T) -> Figment {
        Self::figment().merge(extra)
    }

    /// Get effective Nova config for leader (with overrides applied).
    pub fn nova_for_leader(&self) -> NovaConfig {
        let mut nova = self.nova.clone();
        if let Some(override_backend) = &self.leader.nova {
            nova.backend = override_backend.clone();
        }
        nova
    }

    /// Get effective Nova config for worker (with overrides applied).
    pub fn nova_for_worker(&self) -> NovaConfig {
        let mut nova = self.nova.clone();
        if let Some(override_backend) = &self.worker.nova {
            nova.backend = override_backend.clone();
        }
        nova
    }
}

/// Implement Provider trait for KvbmConfig.
///
/// This allows KvbmConfig to be used as a configuration source itself,
/// enabling composition with other providers. Dependent libraries can
/// extract their own config from the same Figment.
impl Provider for KvbmConfig {
    fn metadata(&self) -> Metadata {
        Metadata::named("KvbmConfig")
    }

    fn data(&self) -> Result<Map<Profile, Dict>, figment::Error> {
        Serialized::defaults(self).data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KvbmConfig::default();
        assert!(config.tokio.worker_threads.is_none());
        assert!(config.tokio.max_blocking_threads.is_none());
        assert!(config.rayon.num_threads.is_none());
    }

    #[test]
    fn test_figment_defaults() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_TOKIO_WORKER_THREADS",
                "KVBM_RAYON_NUM_THREADS",
                "KVBM_NOVA_BACKEND_TCP_ADDR",
                "KVBM_NOVA_DISCOVERY_CLUSTER_ID",
            ],
            || {
                let figment = KvbmConfig::figment();
                let config: KvbmConfig = figment.extract().unwrap();
                assert!(config.tokio.worker_threads.is_none());
            },
        );
    }

    #[test]
    fn test_env_override_tokio() {
        temp_env::with_vars(
            vec![
                ("KVBM_TOKIO_WORKER_THREADS", Some("8")),
                ("KVBM_TOKIO_MAX_BLOCKING_THREADS", Some("256")),
            ],
            || {
                let figment = KvbmConfig::figment();
                let config: KvbmConfig = figment.extract().unwrap();
                assert_eq!(config.tokio.worker_threads, Some(8));
                assert_eq!(config.tokio.max_blocking_threads, Some(256));
            },
        );
    }

    #[test]
    fn test_nova_for_leader_no_override() {
        let config = KvbmConfig::default();
        let leader_nova = config.nova_for_leader();
        assert_eq!(leader_nova.backend.tcp_port, config.nova.backend.tcp_port);
    }

    #[test]
    fn test_nova_for_leader_with_override() {
        let mut config = KvbmConfig::default();
        config.leader.nova = Some(NovaBackendConfig {
            tcp_addr: Some("192.168.1.1".to_string()),
            tcp_interface: None,
            tcp_port: 8080,
        });

        let leader_nova = config.nova_for_leader();
        assert_eq!(
            leader_nova.backend.tcp_addr,
            Some("192.168.1.1".to_string())
        );
        assert_eq!(leader_nova.backend.tcp_port, 8080);

        // Worker should still use defaults
        let worker_nova = config.nova_for_worker();
        assert!(worker_nova.backend.tcp_addr.is_none());
    }

    #[test]
    fn test_extract_from_with_tuple_override() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_TOKIO_WORKER_THREADS",
                "KVBM_NOVA_BACKEND_TCP_PORT",
            ],
            || {
                // Use tuple pair for programmatic override (figment best practice)
                let figment = KvbmConfig::figment()
                    .merge(("tokio.worker_threads", 16usize))
                    .merge(("nova.backend.tcp_port", 9090u16));

                let config = KvbmConfig::extract_from(figment).unwrap();
                assert_eq!(config.tokio.worker_threads, Some(16));
                assert_eq!(config.nova.backend.tcp_port, 9090);
            },
        );
    }

    #[test]
    fn test_figment_with_helper() {
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH", "KVBM_RAYON_NUM_THREADS"], || {
            let figment = KvbmConfig::figment_with(("rayon.num_threads", 8usize));
            let config = KvbmConfig::extract_from(figment).unwrap();
            assert_eq!(config.rayon.num_threads, Some(8));
        });
    }

    #[test]
    fn test_config_as_provider() {
        // KvbmConfig implements Provider, so it can be used as a source
        let original = KvbmConfig {
            tokio: TokioConfig {
                worker_threads: Some(4),
                max_blocking_threads: Some(128),
            },
            ..Default::default()
        };

        // Use the config as a provider to create a new figment
        let figment = Figment::from(&original);
        let extracted: KvbmConfig = figment.extract().unwrap();

        assert_eq!(extracted.tokio.worker_threads, Some(4));
        assert_eq!(extracted.tokio.max_blocking_threads, Some(128));
    }
}
