// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL backend configuration with Figment support.
//!
//! This module provides configuration extraction for NIXL backends from
//! environment variables with the pattern: `DYN_KVBM_NIXL_BACKEND_<backend>_<key>=<value>`

use anyhow::{Result, bail};
use std::collections::HashSet;

use dynamo_config::parse_bool;

/// Configuration for NIXL backends.
///
/// Supports extracting backend configurations from environment variables:
/// - `DYN_KVBM_NIXL_BACKEND_UCX=true` - Enable UCX backend with default params
/// - `DYN_KVBM_NIXL_BACKEND_GDS=false` - Explicitly disable GDS backend
/// - Valid values: true/false, 1/0, on/off, yes/no (case-insensitive)
/// - Invalid values (e.g., "maybe", "random") will cause an error
/// - Custom params (e.g., `DYN_KVBM_NIXL_BACKEND_UCX_PARAM1=value`) will cause an error
#[derive(Debug, Clone, Default)]
pub struct NixlBackendConfig {
    /// Set of enabled backends (just backend names, no custom params yet)
    backends: HashSet<String>,
}

impl NixlBackendConfig {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self::default()
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
        let mut backends = HashSet::new();

        // Extract all environment variables that match our pattern
        for (key, value) in std::env::vars() {
            if let Some(remainder) = key.strip_prefix("DYN_KVBM_NIXL_BACKEND_") {
                // Check if there's an underscore (indicating custom params)
                if remainder.contains('_') {
                    bail!(
                        "Custom NIXL backend parameters are not yet supported. \
                         Found: {}. Please use only DYN_KVBM_NIXL_BACKEND_<backend>=true \
                         to enable backends with default parameters.",
                        key
                    );
                }

                // Simple backend enablement (e.g., DYN_KVBM_NIXL_BACKEND_UCX=true)
                let backend_name = remainder.to_uppercase();
                match parse_bool(&value) {
                    Ok(true) => {
                        backends.insert(backend_name);
                    }
                    Ok(false) => {
                        // Explicitly disabled, don't add to backends
                        continue;
                    }
                    Err(e) => bail!("Invalid value for {}: {}", key, e),
                }
            }
        }

        // Default to UCX if no backends specified
        if backends.is_empty() {
            backends.insert("UCX".to_string());
        }

        Ok(Self { backends })
    }

    /// Add a backend to the configuration.
    ///
    /// Backend names will be converted to uppercase for consistency.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backends.insert(backend.into().to_uppercase());
        self
    }

    /// Get the set of enabled backends.
    pub fn backends(&self) -> &HashSet<String> {
        &self.backends
    }

    /// Check if a specific backend is enabled.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.backends.contains(&backend.to_uppercase())
    }

    /// Merge another configuration into this one.
    ///
    /// Backends from the other configuration will be added to this one.
    pub fn merge(mut self, other: NixlBackendConfig) -> Self {
        self.backends.extend(other.backends);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_config_is_empty() {
        let config = NixlBackendConfig::new();
        assert!(config.backends().is_empty());
    }

    #[test]
    fn test_with_backend() {
        let config = NixlBackendConfig::new()
            .with_backend("ucx")
            .with_backend("gds_mt");

        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("gds_mt"));
        assert!(config.has_backend("GDS_MT"));
        assert!(!config.has_backend("other"));
    }

    #[test]
    fn test_merge_configs() {
        let config1 = NixlBackendConfig::new().with_backend("ucx");
        let config2 = NixlBackendConfig::new().with_backend("gds");

        let merged = config1.merge(config2);

        assert!(merged.has_backend("ucx"));
        assert!(merged.has_backend("gds"));
    }

    #[test]
    fn test_backend_name_case_insensitive() {
        let config = NixlBackendConfig::new()
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

    // Note: Testing from_env() would require setting environment variables,
    // which is challenging in unit tests. This is better tested with integration tests.
}
