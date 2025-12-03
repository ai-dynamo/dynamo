// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL backend configuration with Figment support.
//!
//! This module provides configuration extraction for NIXL backends from
//! environment variables with the pattern: `DYN_KVBM_NIXL_BACKEND_<backend>_<key>=<value>`

use anyhow::{Result, bail};
use dynamo_runtime::config::parse_bool;
use std::collections::HashSet;

/// Configuration for NIXL backends.
///
/// Supports extracting backend configurations from environment variables:
/// - `DYN_KVBM_NIXL_BACKEND_UCX=true` - Enable UCX backend with default params
/// - `DYN_KVBM_NIXL_BACKEND_GDS=false` - Explicitly disable GDS backend
/// - `DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET=my-bucket` - Custom parameter for OBJ backend
/// - Valid enable values: true/false, 1/0, on/off, yes/no (case-insensitive)
///
/// Custom parameters use the pattern: `DYN_KVBM_NIXL_BACKEND_{BACKEND}_{PARAM}=value`
/// These are parsed by `NixlAgent::create_backend_params_from_env()` at agent creation time.
///
/// # Examples
///
/// ```rust,ignore
/// // Extract from environment
/// let config = NixlBackendConfig::from_env()?;
///
/// // Or combine with builder overrides
/// let config = NixlBackendConfig::from_env()?
///     .with_backend("ucx")
///     .with_backend("gds");
/// ```
#[derive(Debug, Clone, Default)]
pub struct NixlBackendConfig {
    /// Set of enabled backends
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
    /// Custom parameters like `DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET=...` are allowed
    /// and will be parsed by `NixlAgent::create_backend_params_from_env()` at
    /// agent creation time.
    ///
    /// # Errors
    /// Returns an error if invalid boolean values are provided for backend enablement.
    pub fn from_env() -> Result<Self> {
        let mut backends = HashSet::new();
        let mut backends_with_params = HashSet::new();
        let mut explicitly_disabled = HashSet::new();

        // Extract all environment variables that match our pattern
        for (key, _value) in std::env::vars() {
            if let Some(remainder) = key.strip_prefix("DYN_KVBM_NIXL_BACKEND_") {
                // Check if there's an underscore (indicating custom params)
                if let Some(underscore_pos) = remainder.find('_') {
                    // This is a custom parameter like DYN_KVBM_NIXL_BACKEND_OBJ_BUCKET
                    // Extract the backend name (before the underscore)
                    let backend_name = remainder[..underscore_pos].to_uppercase();
                    backends_with_params.insert(backend_name);
                }
            }
        }

        // Now extract simple backend enablement variables
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
                        backends.insert(backend_name);
                    }
                    Ok(false) => {
                        // Explicitly disabled, track it
                        explicitly_disabled.insert(backend_name);
                    }
                    Err(e) => bail!("Invalid value for {}: {}", key, e),
                }
            }
        }

        // Add backends that have custom parameters, but only if not explicitly disabled
        for backend in backends_with_params {
            if !explicitly_disabled.contains(&backend) {
                backends.insert(backend);
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
