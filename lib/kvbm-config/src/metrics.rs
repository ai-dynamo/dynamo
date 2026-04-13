// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics and cache statistics configuration for KVBM.
//!
//! Controls the metrics endpoint, cache hit rate tracking, and logging intervals.

use serde::{Deserialize, Serialize};
use validator::Validate;

fn default_port() -> u16 {
    6880
}

fn default_cache_stats_max_requests() -> usize {
    1000
}

fn default_cache_stats_log_interval_secs() -> u64 {
    5
}

/// Metrics and cache statistics configuration.
///
/// Controls Prometheus-style metrics endpoint and the sliding-window
/// cache hit rate tracker.
///
/// # V1 Compatibility
///
/// These fields are populated from v1 env vars when the v1 compat layer is active:
/// - `DYN_KVBM_METRICS` → `enabled`
/// - `DYN_KVBM_METRICS_PORT` → `port`
/// - `DYN_KVBM_CACHE_STATS_MAX_REQUESTS` → `cache_stats_max_requests`
/// - `DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS` → `cache_stats_log_interval_secs`
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricsConfig {
    /// Enable the metrics endpoint.
    /// Default: false
    #[serde(default)]
    pub enabled: bool,

    /// Port for the metrics endpoint.
    /// Default: 6880
    #[serde(default = "default_port")]
    #[validate(range(min = 1))]
    pub port: u16,

    /// Maximum number of recent requests tracked in the sliding window
    /// for cache hit rate calculation.
    /// Default: 1000
    #[serde(default = "default_cache_stats_max_requests")]
    #[validate(range(min = 1))]
    pub cache_stats_max_requests: usize,

    /// Interval in seconds between cache statistics log messages.
    /// Default: 5
    #[serde(default = "default_cache_stats_log_interval_secs")]
    #[validate(range(min = 1))]
    pub cache_stats_log_interval_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: default_port(),
            cache_stats_max_requests: default_cache_stats_max_requests(),
            cache_stats_log_interval_secs: default_cache_stats_log_interval_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MetricsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.port, 6880);
        assert_eq!(config.cache_stats_max_requests, 1000);
        assert_eq!(config.cache_stats_log_interval_secs, 5);
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{
            "enabled": true,
            "port": 9090,
            "cache_stats_max_requests": 500,
            "cache_stats_log_interval_secs": 10
        }"#;

        let config: MetricsConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.port, 9090);
        assert_eq!(config.cache_stats_max_requests, 500);
        assert_eq!(config.cache_stats_log_interval_secs, 10);

        let serialized = serde_json::to_string(&config).unwrap();
        let roundtrip: MetricsConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(roundtrip.port, config.port);
    }

    #[test]
    fn test_partial_json_uses_defaults() {
        let json = r#"{"enabled": true}"#;
        let config: MetricsConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.port, 6880);
        assert_eq!(config.cache_stats_max_requests, 1000);
    }

    #[test]
    fn test_validation() {
        let config = MetricsConfig::default();
        assert!(config.validate().is_ok());
    }
}
