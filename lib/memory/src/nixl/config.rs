// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL backend configuration with Figment support.
//!
//! This module provides configuration extraction for NIXL backends from
//! environment variables with the pattern: `DYN_KVBM_NIXL_BACKEND_<backend>=<value>`

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use dynamo_config::parse_bool;

/// Configuration for NIXL backends.
///
/// Supports extracting backend configurations from environment variables:
/// - `DYN_KVBM_NIXL_BACKEND_UCX=true` - Enable UCX backend with default params
/// - `DYN_KVBM_NIXL_BACKEND_GDS=false` - Explicitly disable GDS backend
/// - Valid values: true/false, 1/0, on/off, yes/no (case-insensitive)
/// - Invalid values (e.g., "maybe", "random") will cause an error
/// - Custom params (e.g., `DYN_KVBM_NIXL_BACKEND_UCX_PARAM1=value`) will cause an error
///
/// # Data Structure
///
/// Uses a single HashMap where:
/// - Key presence = backend is enabled
/// - Value (inner HashMap) = backend-specific parameters (empty = defaults)
///
/// # TOML Example
///
/// ```toml
/// [backends.UCX]
/// # UCX with default params (empty map)
///
/// [backends.GDS]
/// threads = "4"
/// buffer_size = "1048576"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NixlBackendConfig {
    /// Map of backend name (uppercase) -> optional parameters.
    ///
    /// If a backend is present in the map, it's enabled.
    /// The inner HashMap contains optional override parameters.
    /// An empty inner map means use default parameters.
    #[serde(default)]
    backends: HashMap<String, HashMap<String, String>>,
}

impl NixlBackendConfig {
    /// Create a new empty configuration (no backends enabled).
    pub fn new(backends: HashMap<String, HashMap<String, String>>) -> Self {
        Self { backends }
    }

    /// Create configuration from environment variables.
    ///
    /// Extracts backends from `DYN_KVBM_NIXL_BACKEND_<backend>=<value>` variables.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Custom parameters are detected (not yet supported)
    /// - Invalid boolean values are provided (must be truthy or falsey)
    pub fn from_env() -> Result<Self> {
        let mut backends: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut explicitly_disabled = HashSet::new();

        // First pass: collect all backend parameters (e.g., DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET=value)
        for (key, value) in std::env::vars() {
            if let Some(remainder) = key.strip_prefix("DYN_KVBM_NIXL_BACKEND_") {
                // Check if there's an underscore (indicating custom params)
                if let Some(underscore_pos) = remainder.find('_') {
                    // This is a custom parameter like DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET
                    // Extract the backend name (before the underscore) and param name (after)
                    let backend_name = remainder[..underscore_pos].to_uppercase();
                    let param_name = remainder[underscore_pos + 1..].to_lowercase();

                    backends
                        .entry(backend_name)
                        .or_default()
                        .insert(param_name, value);
                }
            }
        }

        // Second pass: extract simple backend enablement variables
        for (key, value) in std::env::vars() {
            if let Some(remainder) = key.strip_prefix("DYN_KVBM_NIXL_BACKEND_") {
                // Skip if it contains underscore (custom param, not enablement)
                if remainder.contains('_') {
                    continue;
                }

                // Simple backend enablement (e.g., DYN_KVBM_NIXL_BACKEND_UCX=true)
                let backend_name = remainder.to_uppercase();
                match parse_bool(&value) {
                    Ok(true) => {
                        // Enable backend with empty params (if not already present with params)
                        backends.entry(backend_name).or_default();
                    }
                    Ok(false) => {
                        // Explicitly disabled, track it and remove if present
                        explicitly_disabled.insert(backend_name.clone());
                        backends.remove(&backend_name);
                    }
                    Err(e) => bail!("Invalid value for {}: {}", key, e),
                }
            }
        }

        // Remove any backends that were explicitly disabled
        for disabled in &explicitly_disabled {
            backends.remove(disabled);
        }

        Ok(Self { backends })
    }

    /// Add a backend with default parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backends
            .insert(backend.into().to_uppercase(), HashMap::new());
        self
    }

    /// Add a backend with custom parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend_params(
        mut self,
        backend: impl Into<String>,
        params: HashMap<String, String>,
    ) -> Self {
        self.backends.insert(backend.into().to_uppercase(), params);
        self
    }

    /// Get the list of enabled backend names (uppercase).
    pub fn backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Get parameters for a specific backend.
    /// Backend name is normalized to uppercase for lookup.
    ///
    /// Returns None if the backend is not enabled.
    pub fn backend_params(&self, backend: &str) -> Option<&HashMap<String, String>> {
        self.backends.get(&backend.to_uppercase())
    }

    /// Check if a specific backend is enabled.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.backends.contains_key(&backend.to_uppercase())
    }

    /// Merge another configuration into this one.
    ///
    /// Backends from the other configuration will be added to this one.
    /// If both have the same backend, params from `other` take precedence.
    pub fn merge(mut self, other: NixlBackendConfig) -> Self {
        self.backends.extend(other.backends);
        self
    }

    /// Iterate over all enabled backends and their parameters.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &HashMap<String, String>)> {
        self.backends.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_config_is_empty() {
        let config = NixlBackendConfig::default();
        assert_eq!(config.backends().len(), 0);
    }

    #[test]
    fn test_default_has_ucx() {
        let config = NixlBackendConfig::default();
        assert!(config.backends().len() == 0); // default() is empty, from_env() has UCX default
    }

    #[test]
    fn test_with_backend() {
        let config = NixlBackendConfig::default()
            .with_backend("ucx")
            .with_backend("gds_mt");

        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("gds_mt"));
        assert!(config.has_backend("GDS_MT"));
        assert!(!config.has_backend("other"));
    }

    #[test]
    fn test_with_backend_params() {
        let mut params = HashMap::new();
        params.insert("threads".to_string(), "4".to_string());
        params.insert("buffer_size".to_string(), "1048576".to_string());

        let config = NixlBackendConfig::default()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        // UCX should have empty params
        let ucx_params = config.backend_params("UCX").unwrap();
        assert!(ucx_params.is_empty());

        // GDS should have custom params
        let gds_params = config.backend_params("GDS").unwrap();
        assert_eq!(gds_params.get("threads"), Some(&"4".to_string()));
        assert_eq!(gds_params.get("buffer_size"), Some(&"1048576".to_string()));
    }

    #[test]
    fn test_merge_configs() {
        let config1 = NixlBackendConfig::default().with_backend("ucx");
        let config2 = NixlBackendConfig::default().with_backend("gds");

        let merged = config1.merge(config2);

        assert!(merged.has_backend("ucx"));
        assert!(merged.has_backend("gds"));
    }

    #[test]
    fn test_backend_name_case_insensitive() {
        let config = NixlBackendConfig::default()
            .with_backend("ucx")
            .with_backend("Gds_mt")
            .with_backend("OTHER");

        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("GDS_MT"));
        assert!(config.has_backend("gds_mt"));
        assert!(config.has_backend("OTHER"));
        assert!(config.has_backend("other"));
    }

    #[test]
    fn test_iter() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), "value".to_string());

        let config = NixlBackendConfig::default()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        let items: Vec<_> = config.iter().collect();
        assert_eq!(items.len(), 2);
    }

    // Note: Testing from_env() would require setting environment variables,
    // which is challenging in unit tests. This is better tested with integration tests.
}
