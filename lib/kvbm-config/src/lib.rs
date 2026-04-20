// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM Configuration Library
//!
//! Provides centralized configuration for Tokio, Rayon, Messenger, and NixL runtimes.
//! Supports role-specific configuration for leader and worker components.

mod cache;
mod debug;
mod discovery;
mod events;
mod messenger;
mod metrics;
mod nixl;
mod object;
mod offload;
mod onboard;
mod rayon;
mod tokio;
mod v1_compat;

pub use cache::{CacheConfig, DiskCacheConfig, HostCacheConfig, ParallelismMode};
pub use debug::DebugConfig;
pub use discovery::{
    DiscoveryConfig, EtcdDiscoveryConfig, FilesystemDiscoveryConfig, P2pDiscoveryConfig,
};
pub use events::{BatchingConfig as EventsBatchingConfig, EventPolicyConfig, EventsConfig};
pub use messenger::{MessengerBackendConfig, MessengerConfig};
pub use metrics::MetricsConfig;
pub use nixl::NixlConfig;
pub use object::{NixlObjectConfig, ObjectClientConfig, ObjectConfig, S3ObjectConfig};
pub use offload::{
    OffloadConfig, PolicyType, PresenceFilterConfig, PresenceLfuFilterConfig, TierOffloadConfig,
};
pub use onboard::{OnboardConfig, OnboardMode};
pub use rayon::RayonConfig;
pub use tokio::TokioConfig;
pub use v1_compat::V1EnvCompat;

use figment::{
    Figment, Metadata, Profile, Provider,
    providers::{Env, Format, Json, Serialized, Toml},
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

/// Top-level KVBM configuration.
///
/// Use Figment profiles to configure role-specific settings. For example,
/// leader and worker can have different `tokio.worker_threads` values by
/// putting them under `"leader"` and `"worker"` profile keys in JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct KvbmConfig {
    #[validate(nested)]
    pub tokio: TokioConfig,

    #[validate(nested)]
    pub rayon: RayonConfig,

    #[validate(nested)]
    pub messenger: MessengerConfig,

    /// NixL configuration. None = NixL disabled.
    #[validate(nested)]
    #[serde(default)]
    pub nixl: Option<NixlConfig>,

    /// Cache configuration (host G2 tier and disk G3 tier).
    #[validate(nested)]
    #[serde(default)]
    pub cache: CacheConfig,

    /// Offload policy configuration (G1→G2, G2→G3 transitions).
    #[validate(nested)]
    #[serde(default)]
    pub offload: OffloadConfig,

    /// Onboard configuration (G2→G1 loading strategy).
    #[serde(default)]
    pub onboard: OnboardConfig,

    /// Object storage configuration (G4 tier).
    /// None = object storage disabled.
    #[validate(nested)]
    #[serde(default)]
    pub object: Option<ObjectConfig>,

    /// Event publishing configuration for distributed coordination.
    #[validate(nested)]
    #[serde(default)]
    pub events: EventsConfig,

    /// Metrics and cache statistics configuration.
    #[validate(nested)]
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Debug configuration (recording, etc.).
    #[validate(nested)]
    #[serde(default)]
    pub debug: DebugConfig,
}

impl KvbmConfig {
    /// Create a Figment configuration with all sources merged.
    ///
    /// Configuration sources in priority order (lowest to highest):
    /// 1. Code defaults
    /// 2. V1 `DYN_KVBM_*` environment variables (compat layer)
    /// 3. System config file at /opt/dynamo/etc/kvbm.toml
    /// 4. TOML file from KVBM_CONFIG_PATH environment variable
    /// 5. V2 environment variables (KVBM_* prefixed)
    /// 6. JSON overrides from Python (via `from_figment_with_json`)
    pub fn figment() -> Figment {
        let config_path = std::env::var("KVBM_CONFIG_PATH").unwrap_or_default();

        Figment::new()
            // 1. Code defaults (lowest priority)
            .merge(Serialized::defaults(KvbmConfig::default()))
            // 2. V1 DYN_KVBM_* env vars (compat layer)
            .merge(V1EnvCompat)
            // 3-4. TOML files
            .merge(Toml::file("/opt/dynamo/etc/kvbm.toml"))
            .merge(Toml::file(&config_path))
            // 5. V2 KVBM_* env vars (override v1 and files)
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
            // Messenger backend config: KVBM_MESSENGER_BACKEND_TCP_ADDR, etc.
            .merge(
                Env::prefixed("KVBM_MESSENGER_BACKEND_")
                    .map(|k| format!("messenger.backend.{}", k.as_str().to_lowercase()).into()),
            )
            // Messenger discovery config: KVBM_MESSENGER_DISCOVERY_CLUSTER_ID, etc.
            .merge(
                Env::prefixed("KVBM_MESSENGER_DISCOVERY_")
                    .map(|k| format!("messenger.discovery.{}", k.as_str().to_lowercase()).into()),
            )
            // Messenger init timeout: KVBM_MESSENGER_INIT_TIMEOUT_SECS
            .merge(
                Env::prefixed("KVBM_MESSENGER_INIT_TIMEOUT_SECS")
                    .map(|_| "messenger.init_timeout_secs".into()),
            )
            // NixL config: KVBM_NIXL_BACKENDS (comma-separated list)
            .merge(
                Env::prefixed("KVBM_NIXL_")
                    .map(|k| format!("nixl.{}", k.as_str().to_lowercase()).into()),
            )
            // Cache host config: KVBM_CACHE_HOST_SIZE_GB, KVBM_CACHE_HOST_NUM_BLOCKS
            .merge(
                Env::prefixed("KVBM_CACHE_HOST_")
                    .map(|k| format!("cache.host.{}", k.as_str().to_lowercase()).into()),
            )
            // Cache disk config: KVBM_CACHE_DISK_SIZE_GB, KVBM_CACHE_DISK_NUM_BLOCKS, etc.
            .merge(
                Env::prefixed("KVBM_CACHE_DISK_")
                    .map(|k| format!("cache.disk.{}", k.as_str().to_lowercase()).into()),
            )
            // Cache parallelism mode: KVBM_CACHE_PARALLELISM=tensor_parallel|replicated_data
            .merge(Env::prefixed("KVBM_CACHE_PARALLELISM").map(|_| "cache.parallelism".into()))
            // Events config: KVBM_EVENTS_ENABLED, KVBM_EVENTS_SUBJECT
            .merge(
                Env::prefixed("KVBM_EVENTS_")
                    .map(|k| format!("events.{}", k.as_str().to_lowercase()).into()),
            )
            // Events batching config: KVBM_EVENTS_BATCHING_WINDOW_DURATION_MS, etc.
            .merge(
                Env::prefixed("KVBM_EVENTS_BATCHING_")
                    .map(|k| format!("events.batching.{}", k.as_str().to_lowercase()).into()),
            )
            // Metrics config: KVBM_METRICS_ENABLED, KVBM_METRICS_PORT, etc.
            .merge(
                Env::prefixed("KVBM_METRICS_")
                    .map(|k| format!("metrics.{}", k.as_str().to_lowercase()).into()),
            )
            // Debug config: KVBM_DEBUG_RECORDING
            .merge(
                Env::prefixed("KVBM_DEBUG_")
                    .map(|k| format!("debug.{}", k.as_str().to_lowercase()).into()),
            )
            // Offload config: KVBM_OFFLOAD_G1_TO_G2_*, KVBM_OFFLOAD_G2_TO_G3_*
            .merge(
                Env::prefixed("KVBM_OFFLOAD_G1_TO_G2_")
                    .map(|k| format!("offload.g1_to_g2.{}", k.as_str().to_lowercase()).into()),
            )
            .merge(
                Env::prefixed("KVBM_OFFLOAD_G2_TO_G3_")
                    .map(|k| format!("offload.g2_to_g3.{}", k.as_str().to_lowercase()).into()),
            )
            // Onboard config: KVBM_ONBOARD_MODE
            .merge(Env::prefixed("KVBM_ONBOARD_MODE").map(|_| "onboard.mode".into()))
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
    ///         .merge(("messenger.backend.tcp_port", 8080u16))
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
    /// let figment = KvbmConfig::figment_with(("messenger.backend.tcp_port", 8080u16));
    /// let config = KvbmConfig::extract_from(figment)?;
    /// ```
    pub fn figment_with<T: Provider>(extra: T) -> Figment {
        Self::figment().merge(extra)
    }

    /// Load configuration merging JSON overrides from Python.
    ///
    /// JSON has highest priority - overrides env vars, TOML files, and defaults.
    /// This is the primary entrypoint for vLLM's `kv_connector_extra_config` dict.
    ///
    /// # Example
    /// ```rust,ignore
    /// let json = r#"{"tokio": {"worker_threads": 8}, "messenger": {"backend": {"tcp_port": 9000}}}"#;
    /// let config = KvbmConfig::from_figment_with_json(json)?;
    /// ```
    pub fn from_figment_with_json(json: &str) -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment().merge(Json::string(json)))
    }

    // ==================== Profile-based Configuration ====================
    //
    // Figment profiles allow role-specific configuration. The `profile` key
    // in TOML/JSON is special - values under it are stored in named profiles
    // and overlaid when that profile is selected.
    //
    // Example JSON:
    // {
    //   "tokio": {"worker_threads": 4},           // default profile (all roles)
    //   "profile": {
    //     "leader": {"tokio": {"worker_threads": 2}},  // leader-only overlay
    //     "worker": {"tokio": {"worker_threads": 8}}   // worker-only overlay
    //   }
    // }
    //
    // When `build_leader()` selects "leader" profile:
    // - tokio.worker_threads = 2 (from leader profile overlay)
    //
    // When `build_worker()` selects "worker" profile:
    // - tokio.worker_threads = 8 (from worker profile overlay)

    /// Figment with leader profile selected.
    ///
    /// This merges `profile.leader.*` values over the defaults.
    /// If no `profile.leader` section exists, defaults are used.
    pub fn figment_for_leader() -> Figment {
        Self::figment().select(Profile::new("leader"))
    }

    /// Figment with worker profile selected.
    ///
    /// This merges `profile.worker.*` values over the defaults.
    /// If no `profile.worker` section exists, defaults are used.
    pub fn figment_for_worker() -> Figment {
        Self::figment().select(Profile::new("worker"))
    }

    /// Load leader config from env/files with leader profile selected.
    pub fn from_env_for_leader() -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment_for_leader())
    }

    /// Load worker config from env/files with worker profile selected.
    pub fn from_env_for_worker() -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment_for_worker())
    }

    /// Load leader config with JSON overrides and leader profile selected.
    ///
    /// JSON top-level keys are treated as profile names when using `.nested()`.
    /// Keys under `default` apply to all profiles, keys under `leader` only to leader.
    ///
    /// Example JSON:
    /// ```json
    /// {
    ///   "leader": {
    ///     "cache": {"host": {"cache_size_gb": 1.0}},
    ///     "tokio": {"worker_threads": 2}
    ///   }
    /// }
    /// ```
    pub fn from_figment_with_json_for_leader(json: &str) -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment_for_leader().merge(Json::string(json).nested()))
    }

    /// Load worker config with JSON overrides and worker profile selected.
    ///
    /// JSON top-level keys are treated as profile names when using `.nested()`.
    /// Keys under `default` apply to all profiles, keys under `worker` only to worker.
    ///
    /// Example JSON:
    /// ```json
    /// {
    ///   "worker": {"tokio": {"worker_threads": 8}}
    /// }
    /// ```
    pub fn from_figment_with_json_for_worker(json: &str) -> Result<Self, ConfigError> {
        Self::extract_from(Self::figment_for_worker().merge(Json::string(json).nested()))
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
        // TokioConfig defaults to 1 worker thread
        assert_eq!(config.tokio.worker_threads, Some(1));
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
                "KVBM_MESSENGER_BACKEND_TCP_ADDR",
                "KVBM_MESSENGER_DISCOVERY_CLUSTER_ID",
            ],
            || {
                let figment = KvbmConfig::figment();
                let config: KvbmConfig = figment.extract().unwrap();
                // TokioConfig defaults to 1 worker thread
                assert_eq!(config.tokio.worker_threads, Some(1));
            },
        );
    }

    #[test]
    fn test_env_override_tokio() {
        temp_env::with_vars(
            vec![
                ("KVBM_TOKIO_WORKER_THREADS", Some("2")),
                ("KVBM_TOKIO_MAX_BLOCKING_THREADS", Some("32")),
            ],
            || {
                let figment = KvbmConfig::figment();
                let config: KvbmConfig = figment.extract().unwrap();
                assert_eq!(config.tokio.worker_threads, Some(2));
                assert_eq!(config.tokio.max_blocking_threads, Some(32));
            },
        );
    }

    #[test]
    fn test_extract_from_with_tuple_override() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_TOKIO_WORKER_THREADS",
                "KVBM_MESSENGER_BACKEND_TCP_PORT",
            ],
            || {
                // Use tuple pair for programmatic override (figment best practice)
                let figment = KvbmConfig::figment()
                    .merge(("tokio.worker_threads", 2usize))
                    .merge(("messenger.backend.tcp_port", 9090u16));

                let config = KvbmConfig::extract_from(figment).unwrap();
                assert_eq!(config.tokio.worker_threads, Some(2));
                assert_eq!(config.messenger.backend.tcp_port, 9090);
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

    #[test]
    fn test_from_figment_with_json() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_TOKIO_WORKER_THREADS",
                "KVBM_MESSENGER_BACKEND_TCP_PORT",
            ],
            || {
                let json = r#"{"tokio": {"worker_threads": 2}, "messenger": {"backend": {"tcp_port": 9090}}}"#;
                let config = KvbmConfig::from_figment_with_json(json).unwrap();

                assert_eq!(config.tokio.worker_threads, Some(2));
                assert_eq!(config.messenger.backend.tcp_port, 9090);
            },
        );
    }

    #[test]
    fn test_from_figment_with_json_overrides_env() {
        // JSON should override env vars (highest priority)
        temp_env::with_vars(vec![("KVBM_TOKIO_WORKER_THREADS", Some("1"))], || {
            let json = r#"{"tokio": {"worker_threads": 2}}"#;
            let config = KvbmConfig::from_figment_with_json(json).unwrap();

            // JSON (2) should override env var (1)
            assert_eq!(config.tokio.worker_threads, Some(2));
        });
    }

    #[test]
    fn test_from_figment_with_empty_json() {
        // Empty JSON object should not cause errors and should use env/defaults
        // We just verify it doesn't fail - the actual values depend on environment
        let config = KvbmConfig::from_figment_with_json("{}");
        assert!(config.is_ok(), "Empty JSON should not cause errors");
    }

    // ==================== Profile Selection Tests ====================
    //
    // Figment profiles work with `.nested()` JSON provider - top-level keys
    // become profile names. Use "default" for values that apply to all profiles.

    #[test]
    fn test_profile_selection_leader_vs_worker() {
        // Test that leader and worker profiles get different values
        // JSON top-level keys are profile names when using .nested()
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                // JSON with nested profiles - top-level keys are profile names
                let json = r#"{
                    "default": {"tokio": {"worker_threads": 4}},
                    "leader": {"tokio": {"worker_threads": 2}},
                    "worker": {"tokio": {"worker_threads": 8}}
                }"#;

                // Leader should get 2 threads (from leader profile)
                let leader_config = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(
                    leader_config.tokio.worker_threads,
                    Some(2),
                    "Leader should get leader profile's tokio.worker_threads"
                );

                // Worker should get 8 threads (from worker profile)
                let worker_config = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(
                    worker_config.tokio.worker_threads,
                    Some(8),
                    "Worker should get worker profile's tokio.worker_threads"
                );
            },
        );
    }

    #[test]
    fn test_profile_no_override_uses_default() {
        // When no profile-specific section exists, default profile values are used
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                // JSON with only default profile
                let json = r#"{"default": {"tokio": {"worker_threads": 4}}}"#;

                // Both leader and worker should get the default (4)
                let leader_config = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(
                    leader_config.tokio.worker_threads,
                    Some(4),
                    "Leader should use default when no leader profile exists"
                );

                let worker_config = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(
                    worker_config.tokio.worker_threads,
                    Some(4),
                    "Worker should use default when no worker profile exists"
                );
            },
        );
    }

    #[test]
    fn test_profile_with_defaults_and_overlay() {
        // Test that default profile values apply to all roles, profile-specific overlay on top
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                // cache.host in default applies to all profiles
                // leader profile adds tokio override
                let json = r#"{
                    "default": {"cache": {"host": {"cache_size_gb": 2.0}}},
                    "leader": {"tokio": {"worker_threads": 2}}
                }"#;

                // Leader: gets cache.host from default + tokio from leader profile
                let leader_config = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(leader_config.cache.host.cache_size_gb, Some(2.0));
                assert_eq!(leader_config.tokio.worker_threads, Some(2));

                // Worker: gets cache.host from default, uses default tokio (not leader's override)
                let worker_config = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(worker_config.cache.host.cache_size_gb, Some(2.0));
                // Worker gets default tokio.worker_threads (1), NOT leader's override (2)
                assert_eq!(
                    worker_config.tokio.worker_threads,
                    Some(1),
                    "Worker should get default tokio, not leader's override"
                );
            },
        );
    }

    #[test]
    fn test_from_env_for_leader_and_worker() {
        // Test from_env_for_leader and from_env_for_worker work without error
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                // Both should succeed with default values
                let leader_config = KvbmConfig::from_env_for_leader();
                assert!(leader_config.is_ok(), "from_env_for_leader should succeed");

                let worker_config = KvbmConfig::from_env_for_worker();
                assert!(worker_config.is_ok(), "from_env_for_worker should succeed");
            },
        );
    }
}
