// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! V1 environment variable compatibility layer for KVBM configuration.
//!
//! This module provides a [`figment::Provider`] implementation that reads
//! legacy `DYN_KVBM_*` environment variables and maps them to the v2 config
//! structure. It is inserted into the Figment merge chain at low priority
//! (after defaults, before TOML files and v2 env vars), so v2-native
//! configuration always takes precedence.
//!
//! # Mapping Categories
//!
//! 1. **Direct**: env var value is parsed (TOML-like) and inserted at a v2 path
//! 2. **Semantic**: value is transformed before insertion (e.g., bool → enum variant)
//! 3. **Composite**: multiple env vars are combined into a struct, then serialized
//! 4. **Deprecated**: env var is recognized but only produces a warning

use std::collections::HashMap;

use figment::value::{Dict, Map, Value};
use figment::{Metadata, Profile, Provider};

use crate::nixl::NixlConfig;
use crate::object::{ObjectClientConfig, ObjectConfig, S3ObjectConfig};

/// Direct env var → config path mappings.
///
/// Each entry is (env_var_name, figment_dotted_path).
/// Values are parsed using figment's TOML-like string parsing
/// (bools, ints, floats auto-detected; everything else is a string).
const DIRECT_MAPPINGS: &[(&str, &str)] = &[
    // Cache host (G2)
    ("DYN_KVBM_CPU_CACHE_GB", "cache.host.cache_size_gb"),
    (
        "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS",
        "cache.host.num_blocks",
    ),
    // Cache disk (G3)
    ("DYN_KVBM_DISK_CACHE_GB", "cache.disk.cache_size_gb"),
    (
        "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS",
        "cache.disk.num_blocks",
    ),
    // Metrics
    ("DYN_KVBM_METRICS", "metrics.enabled"),
    ("DYN_KVBM_METRICS_PORT", "metrics.port"),
    (
        "DYN_KVBM_CACHE_STATS_MAX_REQUESTS",
        "metrics.cache_stats_max_requests",
    ),
    (
        "DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS",
        "metrics.cache_stats_log_interval_secs",
    ),
    // Debug
    ("DYN_KVBM_ENABLE_RECORD", "debug.recording"),
    // Messenger
    (
        "DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS",
        "messenger.init_timeout_secs",
    ),
    // Offload
    (
        "DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY",
        "offload.g1_to_g2.min_priority",
    ),
    (
        "DYN_KVBM_MAX_CONCURRENT_TRANSFERS",
        "offload.g1_to_g2.max_concurrent_transfers",
    ),
    // Events
    ("DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR", "events.enabled"),
];

/// Env vars for transfer batch size — first match wins.
/// `DYN_KVBM_MAX_TRANSFER_BATCH_SIZE` takes priority over the shorter name.
const BATCH_SIZE_VARS: &[&str] = &[
    "DYN_KVBM_MAX_TRANSFER_BATCH_SIZE",
    "DYN_KVBM_TRANSFER_BATCH_SIZE",
];

/// Deprecated v1 env vars that have no v2 equivalent (ZMQ replaced by Velo).
const DEPRECATED_ZMQ_VARS: &[&str] = &[
    "DYN_KVBM_LEADER_ZMQ_HOST",
    "DYN_KVBM_LEADER_ZMQ_PUB_PORT",
    "DYN_KVBM_LEADER_ZMQ_ACK_PORT",
    "DYN_KVBM_TRTLLM_ZMQ_PORT",
];

/// V1 `DYN_KVBM_*` environment variable compatibility provider.
///
/// Reads legacy environment variables and maps them into the v2 config
/// structure. Designed to be merged at low priority in the Figment chain
/// so that v2 env vars, TOML files, and JSON overrides take precedence.
///
/// # Example
///
/// ```rust,ignore
/// use figment::Figment;
/// use figment::providers::Serialized;
/// use kvbm_config::{KvbmConfig, V1EnvCompat};
///
/// let figment = Figment::new()
///     .merge(Serialized::defaults(KvbmConfig::default()))
///     .merge(V1EnvCompat)  // reads DYN_KVBM_* env vars
///     // ... higher-priority sources ...
///     ;
/// ```
pub struct V1EnvCompat;

impl Provider for V1EnvCompat {
    fn metadata(&self) -> Metadata {
        Metadata::named("V1 DYN_KVBM_* env vars (compat)")
    }

    fn data(&self) -> Result<Map<Profile, Dict>, figment::Error> {
        let mut dict = Dict::new();

        // Section 1: Direct mappings
        self.apply_direct_mappings(&mut dict);

        // Section 2: Transfer batch size (first-match priority)
        self.apply_batch_size(&mut dict);

        // Section 3: Disk cache dir (string, not auto-parsed)
        self.apply_disk_cache_dir(&mut dict);

        // Section 4: Parallelism mode (semantic: bool → enum string)
        self.apply_parallelism_mode(&mut dict);

        // Section 5: Offload filter disable (semantic: bool → policy array)
        self.apply_offload_filter(&mut dict);

        // Section 6: NixL backends (composite: scan DYN_KVBM_NIXL_BACKEND_*)
        self.apply_nixl_backends(&mut dict).map_err(|e| *e)?;

        // Section 7: Object storage (composite: multiple env vars → ObjectConfig)
        self.apply_object_storage(&mut dict).map_err(|e| *e)?;

        // Section 8: Deprecation warnings
        Self::warn_deprecated();

        Ok(Profile::Default.collect(dict))
    }
}

impl V1EnvCompat {
    /// Apply all direct env var → config path mappings.
    fn apply_direct_mappings(&self, dict: &mut Dict) {
        for &(env_var, config_path) in DIRECT_MAPPINGS {
            if let Ok(val) = std::env::var(env_var) {
                // Use figment's TOML-like parsing: "true"→Bool, "42"→Num, "3.14"→Num, etc.
                let parsed: Value = val.parse().expect("Value::from_str is infallible");
                merge_nested(dict, config_path, parsed);
            }
        }
    }

    /// Apply transfer batch size with priority fallback.
    fn apply_batch_size(&self, dict: &mut Dict) {
        for &env_var in BATCH_SIZE_VARS {
            if let Ok(val) = std::env::var(env_var) {
                let parsed: Value = val.parse().expect("Value::from_str is infallible");
                merge_nested(dict, "offload.g1_to_g2.max_batch_size", parsed);
                break; // first match wins
            }
        }
    }

    /// Disk cache dir is always a string path, never auto-parsed as a number.
    fn apply_disk_cache_dir(&self, dict: &mut Dict) {
        if let Ok(val) = std::env::var("DYN_KVBM_DISK_CACHE_DIR") {
            // Force string value (don't let figment parse "/tmp/" as something else)
            merge_nested(dict, "cache.disk.storage_path", Value::from(val));
        }
    }

    /// Map `DYN_KVBM_NCCL_MLA_MODE=true` → `cache.parallelism = "replicated_data"`.
    fn apply_parallelism_mode(&self, dict: &mut Dict) {
        if let Ok(val) = std::env::var("DYN_KVBM_NCCL_MLA_MODE")
            && parse_bool(&val) == Some(true)
        {
            merge_nested(
                dict,
                "cache.parallelism",
                Value::from("replicated_data".to_string()),
            );
        }
    }

    /// Map `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` → `offload.g1_to_g2.policies = ["pass_all"]`.
    fn apply_offload_filter(&self, dict: &mut Dict) {
        if let Ok(val) = std::env::var("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER")
            && parse_bool(&val) == Some(true)
        {
            merge_nested(
                dict,
                "offload.g1_to_g2.policies",
                Value::from(vec![Value::from("pass_all".to_string())]),
            );
        }
    }

    /// Scan `DYN_KVBM_NIXL_BACKEND_*` env vars and build a NixlConfig.
    fn apply_nixl_backends(&self, dict: &mut Dict) -> Result<(), Box<figment::Error>> {
        let prefix = "DYN_KVBM_NIXL_BACKEND_";
        let mut backends: HashMap<String, HashMap<String, String>> = HashMap::new();

        for (key, val) in std::env::vars() {
            if let Some(backend_name) = key.strip_prefix(prefix)
                && parse_bool(&val) == Some(true)
            {
                backends.insert(backend_name.to_uppercase(), HashMap::new());
            }
        }

        if !backends.is_empty() {
            let nixl_config = NixlConfig::new(backends);
            let value = Value::serialize(nixl_config)?;
            merge_nested(dict, "nixl", value);
        }

        Ok(())
    }

    /// Synthesize an ObjectConfig from v1 object storage env vars.
    fn apply_object_storage(&self, dict: &mut Dict) -> Result<(), Box<figment::Error>> {
        let enabled = std::env::var("DYN_KVBM_OBJECT_ENABLED")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false);

        if !enabled {
            return Ok(());
        }

        let bucket = std::env::var("DYN_KVBM_OBJECT_BUCKET").unwrap_or_default();
        let endpoint = std::env::var("DYN_KVBM_OBJECT_ENDPOINT").ok();
        let region =
            std::env::var("DYN_KVBM_OBJECT_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        let s3_config = S3ObjectConfig {
            endpoint_url: endpoint.clone(),
            bucket,
            region,
            // v1 uses custom endpoints for MinIO-style services
            force_path_style: endpoint.is_some(),
            ..Default::default()
        };

        let object_config = ObjectConfig {
            client: ObjectClientConfig::S3(s3_config),
        };

        let value = Value::serialize(object_config)?;
        merge_nested(dict, "object", value);

        Ok(())
    }

    /// Emit warnings for deprecated v1 env vars that have no v2 equivalent.
    fn warn_deprecated() {
        let mut zmq_found = Vec::new();
        for &var in DEPRECATED_ZMQ_VARS {
            if std::env::var(var).is_ok() {
                zmq_found.push(var);
            }
        }
        if !zmq_found.is_empty() {
            tracing::warn!(
                vars = ?zmq_found,
                "DYN_KVBM_LEADER_ZMQ_* env vars are deprecated in v2; \
                 ZMQ transport has been replaced by Velo messenger. \
                 These variables are ignored."
            );
        }
    }
}

/// Parse a boolean from a string, matching v1 conventions.
fn parse_bool(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Insert a value at a dotted path into a Dict, creating intermediate
/// dicts as needed.
///
/// For example, `insert_nested(dict, "cache.host.cache_size_gb", val)` creates
/// `dict["cache"]["host"]["cache_size_gb"] = val`.
fn merge_nested(dict: &mut Dict, path: &str, value: Value) {
    let parts: Vec<&str> = path.split('.').collect();
    let (leaf, parents) = parts
        .split_last()
        .expect("path must have at least one segment");

    let mut current = dict;
    for &part in parents {
        // Get or create intermediate dict
        let entry = current
            .entry(part.to_string())
            .or_insert_with(|| Value::Dict(figment::value::Tag::Default, Dict::new()));
        current = match entry {
            Value::Dict(_, d) => d,
            // If the existing value is not a dict, replace it with one
            other => {
                *other = Value::Dict(figment::value::Tag::Default, Dict::new());
                match other {
                    Value::Dict(_, d) => d,
                    _ => unreachable!(),
                }
            }
        };
    }
    current.insert(leaf.to_string(), value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KvbmConfig, ParallelismMode, PolicyType};

    /// Helper: extract config from figment with all external env vars cleared.
    fn extract_with_v1_vars(vars: Vec<(&str, Option<&str>)>) -> KvbmConfig {
        // Unset all v2 env vars that could interfere
        let mut unset_vars: Vec<&str> = vec![
            "KVBM_CONFIG_PATH",
            "KVBM_TOKIO_WORKER_THREADS",
            "KVBM_RAYON_NUM_THREADS",
            "KVBM_MESSENGER_BACKEND_TCP_ADDR",
            "KVBM_MESSENGER_DISCOVERY_CLUSTER_ID",
            "KVBM_CACHE_HOST_CACHE_SIZE_GB",
            "KVBM_CACHE_HOST_NUM_BLOCKS",
            "KVBM_CACHE_DISK_CACHE_SIZE_GB",
            "KVBM_CACHE_DISK_NUM_BLOCKS",
            "KVBM_CACHE_PARALLELISM",
            "KVBM_METRICS_ENABLED",
            "KVBM_METRICS_PORT",
            "KVBM_DEBUG_RECORDING",
            "KVBM_EVENTS_ENABLED",
            "KVBM_MESSENGER_INIT_TIMEOUT_SECS",
            "KVBM_ONBOARD_MODE",
        ];
        // Also unset all v1 vars not in the test set
        for &(env_var, _) in DIRECT_MAPPINGS {
            unset_vars.push(env_var);
        }
        for &var in BATCH_SIZE_VARS {
            unset_vars.push(var);
        }
        unset_vars.extend_from_slice(&[
            "DYN_KVBM_DISK_CACHE_DIR",
            "DYN_KVBM_NCCL_MLA_MODE",
            "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER",
            "DYN_KVBM_OBJECT_ENABLED",
            "DYN_KVBM_OBJECT_BUCKET",
            "DYN_KVBM_OBJECT_ENDPOINT",
            "DYN_KVBM_OBJECT_REGION",
        ]);

        // Dedup
        unset_vars.sort_unstable();
        unset_vars.dedup();

        // Build combined: set test vars, unset everything else
        let combined: Vec<(&str, Option<&str>)> = unset_vars
            .into_iter()
            .map(|k| {
                // Check if this var is in the test set
                vars.iter()
                    .find(|(vk, _)| *vk == k)
                    .copied()
                    .unwrap_or((k, None))
            })
            .chain(
                // Add test vars not already in unset list
                vars.iter()
                    .copied()
                    .filter(|(k, _)| !k.starts_with("KVBM_") && !k.starts_with("DYN_KVBM_")),
            )
            .collect();

        // Also collect explicitly set test vars that might not be in the unset list
        let mut final_vars = combined;
        for &(k, v) in &vars {
            if !final_vars.iter().any(|(fk, _)| *fk == k) {
                final_vars.push((k, v));
            }
        }

        temp_env::with_vars(final_vars, || {
            KvbmConfig::from_env().expect("config extraction should succeed")
        })
    }

    #[test]
    fn test_no_v1_vars_produces_defaults() {
        let config = extract_with_v1_vars(vec![]);
        assert!(config.cache.host.cache_size_gb.is_none());
        assert!(config.cache.host.num_blocks.is_none());
        assert!(config.cache.disk.is_none());
        assert!(!config.metrics.enabled);
        assert!(!config.debug.recording);
        assert_eq!(config.cache.parallelism, ParallelismMode::TensorParallel);
    }

    #[test]
    fn test_direct_mapping_cpu_cache() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_CPU_CACHE_GB", Some("4.5"))]);
        assert_eq!(config.cache.host.cache_size_gb, Some(4.5));
    }

    #[test]
    fn test_direct_mapping_cpu_cache_num_blocks() {
        let config = extract_with_v1_vars(vec![(
            "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS",
            Some("2000"),
        )]);
        assert_eq!(config.cache.host.num_blocks, Some(2000));
    }

    #[test]
    fn test_direct_mapping_disk_cache() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_DISK_CACHE_GB", Some("10.0"))]);
        let disk = config.cache.disk.expect("disk config should be present");
        assert_eq!(disk.cache_size_gb, Some(10.0));
    }

    #[test]
    fn test_direct_mapping_metrics() {
        let config = extract_with_v1_vars(vec![
            ("DYN_KVBM_METRICS", Some("true")),
            ("DYN_KVBM_METRICS_PORT", Some("9090")),
            ("DYN_KVBM_CACHE_STATS_MAX_REQUESTS", Some("500")),
            ("DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS", Some("10")),
        ]);
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.port, 9090);
        assert_eq!(config.metrics.cache_stats_max_requests, 500);
        assert_eq!(config.metrics.cache_stats_log_interval_secs, 10);
    }

    #[test]
    fn test_direct_mapping_debug_recording() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_ENABLE_RECORD", Some("true"))]);
        assert!(config.debug.recording);
    }

    #[test]
    fn test_direct_mapping_init_timeout() {
        let config = extract_with_v1_vars(vec![(
            "DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS",
            Some("600"),
        )]);
        assert_eq!(config.messenger.init_timeout_secs, 600);
    }

    #[test]
    fn test_direct_mapping_offload_min_priority() {
        let config = extract_with_v1_vars(vec![(
            "DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY",
            Some("5"),
        )]);
        assert_eq!(config.offload.g1_to_g2.min_priority, 5);
    }

    #[test]
    fn test_transfer_batch_size_max_takes_priority() {
        let config = extract_with_v1_vars(vec![
            ("DYN_KVBM_MAX_TRANSFER_BATCH_SIZE", Some("32")),
            ("DYN_KVBM_TRANSFER_BATCH_SIZE", Some("16")),
        ]);
        assert_eq!(config.offload.g1_to_g2.max_batch_size, Some(32));
    }

    #[test]
    fn test_transfer_batch_size_fallback() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_TRANSFER_BATCH_SIZE", Some("16"))]);
        assert_eq!(config.offload.g1_to_g2.max_batch_size, Some(16));
    }

    #[test]
    fn test_events_consolidator() {
        let config = extract_with_v1_vars(vec![(
            "DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR",
            Some("true"),
        )]);
        assert!(config.events.enabled);
    }

    #[test]
    fn test_disk_cache_dir() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_DISK_CACHE_DIR", Some("/mnt/nvme"))]);
        let disk = config.cache.disk.expect("disk config should be present");
        assert_eq!(
            disk.storage_path,
            Some(std::path::PathBuf::from("/mnt/nvme"))
        );
    }

    #[test]
    fn test_nccl_mla_mode_true() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_NCCL_MLA_MODE", Some("true"))]);
        assert_eq!(config.cache.parallelism, ParallelismMode::ReplicatedData);
    }

    #[test]
    fn test_nccl_mla_mode_false_uses_default() {
        let config = extract_with_v1_vars(vec![("DYN_KVBM_NCCL_MLA_MODE", Some("false"))]);
        assert_eq!(config.cache.parallelism, ParallelismMode::TensorParallel);
    }

    #[test]
    fn test_disable_disk_offload_filter() {
        let config =
            extract_with_v1_vars(vec![("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER", Some("true"))]);
        assert_eq!(config.offload.g1_to_g2.policies, vec![PolicyType::PassAll]);
    }

    #[test]
    fn test_object_storage_enabled() {
        let config = extract_with_v1_vars(vec![
            ("DYN_KVBM_OBJECT_ENABLED", Some("1")),
            ("DYN_KVBM_OBJECT_BUCKET", Some("test-bucket")),
            ("DYN_KVBM_OBJECT_ENDPOINT", Some("http://minio:9000")),
            ("DYN_KVBM_OBJECT_REGION", Some("us-west-2")),
        ]);
        let object = config.object.expect("object config should be present");
        match &object.client {
            ObjectClientConfig::S3(s3) => {
                assert_eq!(s3.bucket, "test-bucket");
                assert_eq!(s3.endpoint_url, Some("http://minio:9000".to_string()));
                assert_eq!(s3.region, "us-west-2");
                assert!(s3.force_path_style);
            }
            _ => panic!("expected S3 client config"),
        }
    }

    #[test]
    fn test_object_storage_disabled_by_default() {
        let config = extract_with_v1_vars(vec![]);
        assert!(config.object.is_none());
    }

    #[test]
    fn test_nixl_backends() {
        let config = extract_with_v1_vars(vec![
            ("DYN_KVBM_NIXL_BACKEND_UCX", Some("true")),
            ("DYN_KVBM_NIXL_BACKEND_GDS", Some("true")),
            ("DYN_KVBM_NIXL_BACKEND_POSIX", Some("false")),
        ]);
        let nixl = config.nixl.expect("nixl config should be present");
        assert!(nixl.has_backend("UCX"));
        assert!(nixl.has_backend("GDS"));
        assert!(!nixl.has_backend("POSIX"));
    }

    #[test]
    fn test_v2_env_overrides_v1() {
        // V2 env vars have higher priority than v1
        let config = extract_with_v1_vars(vec![
            ("DYN_KVBM_CPU_CACHE_GB", Some("4.0")),
            ("KVBM_CACHE_HOST_CACHE_SIZE_GB", Some("8.0")),
        ]);
        // V2 should win: 8.0
        assert_eq!(config.cache.host.cache_size_gb, Some(8.0));
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("True"), Some(true));
        assert_eq!(parse_bool("TRUE"), Some(true));
        assert_eq!(parse_bool("1"), Some(true));
        assert_eq!(parse_bool("yes"), Some(true));
        assert_eq!(parse_bool("on"), Some(true));

        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("False"), Some(false));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("no"), Some(false));
        assert_eq!(parse_bool("off"), Some(false));

        assert_eq!(parse_bool("maybe"), None);
        assert_eq!(parse_bool(""), None);
    }
}
