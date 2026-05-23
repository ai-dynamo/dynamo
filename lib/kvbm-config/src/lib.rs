// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM Configuration Library
//!
//! Provides centralized configuration for Tokio, Rayon, Messenger, and NixL runtimes.
//! Supports role-specific configuration for leader and worker components.
//!
//! # Settings deliberately *not* owned by this crate
//!
//! A handful of switches are resolved *outside* `kvbm-config` because they
//! must be answered before a `KvbmConfig` (or even a `KvbmRuntime`) can be
//! instantiated. They are intentionally surfaced as plain process-level
//! environment variables and are listed here so the inventory is complete:
//!
//! - **`KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS`** *(default `true`)* — gates
//!   `DynamoConnector.prefer_cross_layer_blocks` in
//!   `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/connector.py`. vLLM
//!   queries that property during connector construction (before the
//!   connector hands any JSON off to Rust), so the answer has to come from
//!   somewhere that doesn't depend on a parsed `KvbmConfig`. Truthy values
//!   (`true`/`1`/`yes`/`y`, case-insensitive) keep the cross-layer / FC
//!   path; anything else falls back to the per-layer `register_kv_caches`
//!   path.

mod cache;
mod control;
mod debug;
mod disagg;
mod discovery;
mod events;
mod hub;
mod messenger;
mod metrics;
mod nixl;
mod object;
mod offload;
mod onboard;
pub mod overrides;
mod rayon;
mod remote_search;
mod tokio;
mod v1_compat;

pub use cache::{CacheConfig, DiskCacheConfig, HostCacheConfig, ParallelismMode};
pub use control::ControlConfig;
pub use debug::DebugConfig;
pub use disagg::{DisaggConfig, DisaggregationRole};
pub use discovery::{
    DiscoveryConfig, EtcdDiscoveryConfig, FilesystemDiscoveryConfig, P2pDiscoveryConfig,
};
pub use events::{BatchingConfig as EventsBatchingConfig, EventPolicyConfig, EventsConfig};
pub use hub::LeaderHubConfig;
pub use kvbm_common::BlockLayoutMode;
pub use messenger::{MessengerBackendConfig, MessengerConfig};
pub use metrics::MetricsConfig;
pub use nixl::NixlConfig;
pub use object::{NixlObjectConfig, ObjectClientConfig, ObjectConfig, S3ObjectConfig};
pub use offload::{
    OffloadConfig, PolicyType, PresenceFilterConfig, PresenceLfuFilterConfig, TierOffloadConfig,
};
pub use onboard::{OnboardConfig, OnboardMode};
pub use rayon::RayonConfig;
pub use remote_search::RemoteSearch;
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

    /// Connector→hub configuration. The sole way the connector reaches a
    /// `kvbm-hub`. `None` = no hub features (normal hub-less connector work);
    /// the hub is currently required for `indexer`, `p2p`, and
    /// `disagg`.
    #[validate(nested)]
    #[serde(default)]
    pub hub: Option<LeaderHubConfig>,

    /// Maximum sequence length (tokens). Sourced from vLLM's `max_model_len`
    /// by the connector binding and reported to the hub for KV-index
    /// must-match validation. `None` when not supplied.
    #[serde(default)]
    pub max_seq_len: Option<usize>,

    /// Disaggregation configuration (P/D role). `None` = not a disagg
    /// participant. The hub URL is no longer here — it comes from
    /// [`hub`](Self::hub); this block only carries the per-instance role and
    /// admission budget.
    #[validate(nested)]
    #[serde(default)]
    pub disagg: Option<DisaggConfig>,

    /// Remote-search configuration. `None` = disabled. When `Some`, the leader
    /// captures a hub `IndexerLookupClient` at startup — which requires the hub
    /// to be enabled AND offering the `indexer` feature, else startup fails
    /// with an invalid-configuration error.
    #[validate(nested)]
    #[serde(default)]
    pub remote_search: Option<RemoteSearch>,

    /// Leader control-plane module configuration. `core` + `transfer` are
    /// always on; `dev` / `test` are opt-in and both default to `false`.
    #[validate(nested)]
    #[serde(default)]
    pub control: ControlConfig,

    /// Block-layout compatibility mode applied at cross-leader metadata
    /// import (conditional disagg, future leader-to-leader exchange).
    ///
    /// - `operational` (default): per-worker `(KvBlockLayout, LayoutConfig)`
    ///   must match exactly on import. Strict / bit-for-bit.
    /// - `universal`: canonical aggregate tensor must match; per-worker
    ///   permutation and shard extents may differ. Requires every local
    ///   worker's `KvBlockLayout` to be labeled (not `Unknown`) at
    ///   registration.
    ///
    /// See [`BlockLayoutMode`] docs for full semantics. Env override:
    /// `KVBM_BLOCK_LAYOUT={operational|universal}`.
    #[serde(default)]
    pub block_layout: BlockLayoutMode,
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
            // Control config: KVBM_CONTROL_ENABLED, KVBM_CONTROL_BIND_ADDR, KVBM_CONTROL_PORT
            .merge(
                Env::prefixed("KVBM_CONTROL_")
                    .map(|k| format!("control.{}", k.as_str().to_lowercase()).into()),
            )
            // Block-layout compat mode: KVBM_BLOCK_LAYOUT=operational|universal
            .merge(Env::prefixed("KVBM_BLOCK_LAYOUT").map(|_| "block_layout".into()))
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
        let mut config: Self = Figment::from(provider)
            .extract()
            .map_err(|e| ConfigError::Extraction(Box::new(e)))?;
        config.auto_enable_nixl_backends_for_tiers();
        config.validate()?;
        Ok(config)
    }

    /// Ensure `nixl.backends` includes whatever the configured cache tiers need.
    ///
    /// - Host cache (G2) enabled → UCX (used for inter-worker / remote transfers)
    /// - Disk cache (G3) enabled → POSIX, or GDS_MT when `disk.use_gds = true`
    /// - Host-bypass mode (`bypass_host_cache() == true`) → force GDS_MT for
    ///   direct G1↔G3 transfers, and UCX for the cross-process metadata
    ///   exchange path (`getLocalMD`) that registers G1 (VRAM) segments. POSIX
    ///   alone cannot register VRAM, so without these the worker fails at
    ///   `registerMem` for `VRAM_SEG`.
    ///
    /// Idempotent and additive: only fills gaps, never removes a backend the
    /// user explicitly enabled. If no tier is enabled, leaves `nixl` untouched
    /// (so a user with NIXL fully disabled stays that way).
    fn auto_enable_nixl_backends_for_tiers(&mut self) {
        let host_enabled = self.cache.host.is_enabled();
        let disk_cfg = self.cache.disk.as_ref();
        let disk_enabled = disk_cfg.is_some_and(|d| d.is_enabled());
        let bypass_host = self.cache.bypass_host_cache();
        let prefer_gds = disk_cfg.is_some_and(|d| d.use_gds) || bypass_host;

        if !host_enabled && !disk_enabled {
            return;
        }

        let nixl = self.nixl.get_or_insert_with(NixlConfig::empty);

        // UCX is needed both when G2 is real (inter-worker transfers) and in
        // host-bypass mode (the worker still needs UCX for the metadata
        // exchange that exposes its VRAM segments to the leader).
        if (host_enabled || bypass_host) && !nixl.has_backend("UCX") {
            let reason = if bypass_host {
                "host-bypass mode requires UCX for cross-process metadata exchange"
            } else {
                "host cache requires UCX for inter-worker transfers"
            };
            tracing::info!(reason, "Auto-enabling NIXL backend UCX");
            *nixl = nixl.clone().with_backend("UCX");
        }

        if disk_enabled && !nixl.has_backend("POSIX") && !nixl.has_backend("GDS_MT") {
            let backend = if prefer_gds { "GDS_MT" } else { "POSIX" };
            tracing::info!(
                backend,
                bypass_host,
                "Auto-enabling NIXL backend for disk cache"
            );
            *nixl = nixl.clone().with_backend(backend);
        }
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
    /// Accepts either of two equivalent shapes (or any mix of both):
    ///
    /// 1. **Flat config** — top-level keys are real `KvbmConfig` fields and
    ///    apply to every role:
    ///    ```json
    ///    {"cache": {"host": {"cache_size_gb": 1.0}}, "tokio": {"worker_threads": 2}}
    ///    ```
    /// 2. **Role overlay** — top-level `leader` / `worker` keys hold
    ///    role-specific overlays that override the flat layer for that role:
    ///    ```json
    ///    {"leader": {"tokio": {"worker_threads": 2}}}
    ///    ```
    /// 3. **Mixed** — flat as the default plus role overlays on top:
    ///    ```json
    ///    {
    ///      "cache": {"host": {"cache_size_gb": 1.0}},
    ///      "leader": {"tokio": {"worker_threads": 2}}
    ///    }
    ///    ```
    ///
    /// Precedence (highest to lowest): selected-role overlay → flat JSON →
    /// TOML / env → defaults.
    pub fn from_figment_with_json_for_leader(json: &str) -> Result<Self, ConfigError> {
        Self::from_role_json(json, Role::Leader)
    }

    /// Load worker config with JSON overrides and worker profile selected.
    ///
    /// See [`Self::from_figment_with_json_for_leader`] for the accepted JSON
    /// shapes.
    pub fn from_figment_with_json_for_worker(json: &str) -> Result<Self, ConfigError> {
        Self::from_role_json(json, Role::Worker)
    }

    /// Shared loader for role-specific JSON overrides.
    ///
    /// Splits the JSON object into a flat remainder (applied to the default
    /// profile so every role sees it) and role overlays (applied via
    /// figment's `.nested()` so the selected role's overlay overrides the
    /// flat layer at extract time). This is the behaviour Codex flagged as
    /// missing: previously a flat `{"cache": {...}}` was treated as a
    /// profile named `cache` and silently ignored by the leader/worker
    /// profiles.
    fn from_role_json(json: &str, role: Role) -> Result<Self, ConfigError> {
        use serde_json::Value;

        let parsed: Value = serde_json::from_str(json)
            .map_err(|e| ConfigError::Other(anyhow::anyhow!(e).context("invalid JSON")))?;
        let Value::Object(mut obj) = parsed else {
            return Err(ConfigError::Other(anyhow::anyhow!(
                "kv_connector_extra_config must be a JSON object"
            )));
        };

        // Pop figment profile keys — these go through `.nested()` so figment
        // routes them to the matching profile. `default` is figment's special
        // base profile (visible to every selected profile); `leader`/`worker`
        // overlay it. Anything left in `obj` is treated as flat config and
        // also lands in the default profile.
        let mut overlays = serde_json::Map::new();
        for key in ["default", "leader", "worker"] {
            if let Some(v) = obj.remove(key) {
                overlays.insert(key.to_string(), v);
            }
        }

        let mut figment = match role {
            Role::Leader => Self::figment_for_leader(),
            Role::Worker => Self::figment_for_worker(),
        };

        if !obj.is_empty() {
            let flat_json = Value::Object(obj).to_string();
            figment = figment.merge(Json::string(&flat_json));
        }
        if !overlays.is_empty() {
            let overlays_json = Value::Object(overlays).to_string();
            figment = figment.merge(Json::string(&overlays_json).nested());
        }

        Self::extract_from(figment)
    }
}

#[derive(Debug, Clone, Copy)]
enum Role {
    Leader,
    Worker,
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
    fn test_disagg_json_deserializes_via_leader_profile() {
        // User prompt example: disagg block nested in leader profile.
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH"], || {
            let json = r#"{
                "leader": {
                    "hub": { "url": "http://127.0.0.1:1337", "features": ["disagg"] },
                    "disagg": { "role": "prefill" }
                },
                "worker": {}
            }"#;

            let leader_cfg = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
            let disagg = leader_cfg.disagg.expect("leader should have disagg config");
            assert_eq!(disagg.role, DisaggregationRole::Prefill);
            let hub = leader_cfg.hub.expect("leader should have hub config");
            assert_eq!(hub.url, "http://127.0.0.1:1337");
            assert_eq!(hub.features, vec!["disagg"]);

            // Worker profile should not pick up the leader-only disagg block.
            let worker_cfg = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
            assert!(
                worker_cfg.disagg.is_none(),
                "worker should not inherit leader's disagg block"
            );
        });
    }

    #[test]
    fn test_disagg_role_decode_via_leader_profile() {
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH"], || {
            let json = r#"{
                "leader": {
                    "disagg": { "role": "decode" }
                }
            }"#;
            let cfg = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
            let disagg = cfg.disagg.expect("disagg present");
            assert_eq!(disagg.role, DisaggregationRole::Decode);
        });
    }

    #[test]
    fn test_control_default_is_disabled() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_CONTROL_DEV",
                "KVBM_CONTROL_METRICS",
            ],
            || {
                let cfg = KvbmConfig::from_env().unwrap();
                assert!(!cfg.control.dev);
                assert!(!cfg.control.metrics);
            },
        );
    }

    #[test]
    fn test_control_env_vars_apply() {
        temp_env::with_vars(
            vec![
                ("KVBM_CONTROL_DEV", Some("true")),
                ("KVBM_CONTROL_METRICS", Some("true")),
            ],
            || {
                let cfg = KvbmConfig::from_env().unwrap();
                assert!(cfg.control.dev);
                assert!(cfg.control.metrics);
            },
        );
    }

    #[test]
    fn block_layout_defaults_to_operational() {
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH", "KVBM_BLOCK_LAYOUT"], || {
            let cfg = KvbmConfig::from_env().unwrap();
            assert_eq!(cfg.block_layout, BlockLayoutMode::Operational);
        });
    }

    #[test]
    fn block_layout_env_universal_parses() {
        temp_env::with_vars(vec![("KVBM_BLOCK_LAYOUT", Some("universal"))], || {
            let cfg = KvbmConfig::from_env().unwrap();
            assert_eq!(cfg.block_layout, BlockLayoutMode::Universal);
        });
    }

    #[test]
    fn block_layout_env_operational_parses() {
        temp_env::with_vars(vec![("KVBM_BLOCK_LAYOUT", Some("operational"))], || {
            let cfg = KvbmConfig::from_env().unwrap();
            assert_eq!(cfg.block_layout, BlockLayoutMode::Operational);
        });
    }

    #[test]
    fn block_layout_json_universal_parses() {
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH", "KVBM_BLOCK_LAYOUT"], || {
            let json = r#"{"block_layout": "universal"}"#;
            let cfg = KvbmConfig::from_figment_with_json(json).unwrap();
            assert_eq!(cfg.block_layout, BlockLayoutMode::Universal);
        });
    }

    #[test]
    fn test_control_via_json_leader_profile() {
        temp_env::with_vars_unset(
            vec![
                "KVBM_CONFIG_PATH",
                "KVBM_CONTROL_DEV",
                "KVBM_CONTROL_METRICS",
            ],
            || {
                let json = r#"{
                    "leader": {
                        "control": { "dev": true }
                    }
                }"#;
                let cfg = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert!(cfg.control.dev);
                assert!(!cfg.control.metrics);
                // Worker profile gets the default (all-disabled) since `control`
                // was declared only under `leader`.
                let wcfg = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert!(!wcfg.control.dev);
            },
        );
    }

    #[test]
    fn test_disagg_absent_means_none() {
        temp_env::with_vars_unset(vec!["KVBM_CONFIG_PATH"], || {
            let cfg = KvbmConfig::from_figment_with_json_for_leader("{}").unwrap();
            assert!(cfg.disagg.is_none());
        });
    }

    #[test]
    fn test_flat_json_applies_to_both_roles() {
        // Top-level keys that are real KvbmConfig fields (not "leader" /
        // "worker" / "default") should be treated as flat config and apply
        // to both roles. Regression for Codex's P2 finding that flat
        // overrides were silently dropped by the role loaders.
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                let json = r#"{"cache": {"host": {"cache_size_gb": 2.5}}}"#;

                let leader = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(
                    leader.cache.host.cache_size_gb,
                    Some(2.5),
                    "flat JSON config should reach leader"
                );

                let worker = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(
                    worker.cache.host.cache_size_gb,
                    Some(2.5),
                    "flat JSON config should reach worker"
                );
            },
        );
    }

    #[test]
    fn test_mixed_flat_and_role_overlay() {
        // Flat layer applies to every role; role overlay refines for the
        // selected role only.
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                let json = r#"{
                    "cache": {"host": {"cache_size_gb": 1.0}},
                    "leader": {"tokio": {"worker_threads": 2}},
                    "worker": {"tokio": {"worker_threads": 8}}
                }"#;

                let leader = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(leader.cache.host.cache_size_gb, Some(1.0));
                assert_eq!(leader.tokio.worker_threads, Some(2));

                let worker = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(worker.cache.host.cache_size_gb, Some(1.0));
                assert_eq!(worker.tokio.worker_threads, Some(8));
            },
        );
    }

    #[test]
    fn test_role_overlay_overrides_flat_for_same_key() {
        // When both flat and role overlay touch the same key, role overlay
        // wins for the selected role; the other role sees the flat value.
        temp_env::with_vars_unset(
            vec!["KVBM_CONFIG_PATH", "KVBM_TOKIO_WORKER_THREADS"],
            || {
                let json = r#"{
                    "tokio": {"worker_threads": 4},
                    "leader": {"tokio": {"worker_threads": 2}}
                }"#;

                let leader = KvbmConfig::from_figment_with_json_for_leader(json).unwrap();
                assert_eq!(
                    leader.tokio.worker_threads,
                    Some(2),
                    "leader overlay should beat flat tokio.worker_threads"
                );

                let worker = KvbmConfig::from_figment_with_json_for_worker(json).unwrap();
                assert_eq!(
                    worker.tokio.worker_threads,
                    Some(4),
                    "worker has no overlay → flat value applies"
                );
            },
        );
    }

    #[test]
    fn test_non_object_json_rejected() {
        let err = KvbmConfig::from_figment_with_json_for_leader("[]").unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
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

    fn config_with_cache(host_gb: Option<f64>, disk_gb: Option<f64>, use_gds: bool) -> KvbmConfig {
        let mut config = KvbmConfig {
            cache: CacheConfig {
                host: HostCacheConfig {
                    cache_size_gb: host_gb,
                    num_blocks: None,
                },
                disk: disk_gb.map(|gb| DiskCacheConfig {
                    cache_size_gb: Some(gb),
                    num_blocks: None,
                    use_gds,
                    storage_path: None,
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        config.auto_enable_nixl_backends_for_tiers();
        config
    }

    #[test]
    fn test_auto_enable_no_tiers_leaves_nixl_none() {
        let config = config_with_cache(None, None, false);
        assert!(config.nixl.is_none());
    }

    #[test]
    fn test_auto_enable_host_adds_ucx() {
        let config = config_with_cache(Some(4.0), None, false);
        let nixl = config.nixl.expect("nixl auto-created");
        assert!(nixl.has_backend("UCX"));
        assert!(!nixl.has_backend("POSIX"));
    }

    #[test]
    fn test_auto_enable_disk_only_bypass_mode_adds_gds_mt_and_ucx() {
        // Disk-only (no host) triggers bypass_host_cache() == true, which forces
        // GDS_MT even when use_gds=false, and requires UCX for VRAM segment metadata.
        let config = config_with_cache(None, Some(10.0), false);
        let nixl = config.nixl.expect("nixl auto-created");
        assert!(nixl.has_backend("GDS_MT"));
        assert!(!nixl.has_backend("POSIX"));
        assert!(nixl.has_backend("UCX"));
    }

    #[test]
    fn test_auto_enable_disk_with_gds_adds_gds_mt() {
        let config = config_with_cache(None, Some(10.0), true);
        let nixl = config.nixl.expect("nixl auto-created");
        assert!(nixl.has_backend("GDS_MT"));
        assert!(!nixl.has_backend("POSIX"));
    }

    #[test]
    fn test_auto_enable_host_and_disk_adds_both() {
        let config = config_with_cache(Some(4.0), Some(10.0), false);
        let nixl = config.nixl.expect("nixl auto-created");
        assert!(nixl.has_backend("UCX"));
        assert!(nixl.has_backend("POSIX"));
    }

    #[test]
    fn test_auto_enable_preserves_user_backends() {
        // User explicitly configured GDS_MT for disk; auto-enable must not add POSIX too.
        let mut config = KvbmConfig {
            cache: CacheConfig {
                host: HostCacheConfig {
                    cache_size_gb: Some(4.0),
                    num_blocks: None,
                },
                disk: Some(DiskCacheConfig {
                    cache_size_gb: Some(10.0),
                    num_blocks: None,
                    use_gds: false, // even with use_gds=false, user's existing GDS_MT wins
                    storage_path: None,
                }),
                ..Default::default()
            },
            nixl: Some(NixlConfig::empty().with_backend("GDS_MT")),
            ..Default::default()
        };
        config.auto_enable_nixl_backends_for_tiers();
        let nixl = config.nixl.expect("nixl present");
        assert!(nixl.has_backend("GDS_MT"));
        assert!(!nixl.has_backend("POSIX"));
        // UCX still gets added because host cache is enabled
        assert!(nixl.has_backend("UCX"));
    }

    #[test]
    fn test_auto_enable_idempotent() {
        let mut config = config_with_cache(Some(4.0), Some(10.0), false);
        let snapshot: Vec<String> = config
            .nixl
            .as_ref()
            .unwrap()
            .enabled_backends()
            .into_iter()
            .cloned()
            .collect();
        config.auto_enable_nixl_backends_for_tiers();
        let mut after: Vec<String> = config
            .nixl
            .as_ref()
            .unwrap()
            .enabled_backends()
            .into_iter()
            .cloned()
            .collect();
        let mut before = snapshot;
        before.sort();
        after.sort();
        assert_eq!(before, after);
    }
}
