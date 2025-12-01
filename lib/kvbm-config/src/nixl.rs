// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NixL backend configuration.
//!
//! Configures which NixL backends (UCX, GDS, etc.) are enabled for RDMA transfers,
//! along with optional parameters for each backend.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::Validate;

/// NixL backend configuration.
///
/// Controls which NixL backends are enabled for RDMA memory transfers
/// and their optional parameters.
///
/// # Backends
///
/// Common backends include:
/// - `UCX` - Unified Communication X (default)
/// - `GDS` - GPUDirect Storage
/// - `GDS_MT` - GPUDirect Storage (multi-threaded)
///
/// All backend names are normalized to uppercase.
///
/// # Configuration
///
/// Each backend can have optional parameters as key-value pairs.
/// If a backend has no parameters, use an empty map.
///
/// ## TOML Example
///
/// ```toml
/// [nixl.backends.UCX]
/// # UCX with default params (empty map)
///
/// [nixl.backends.GDS]
/// threads = "4"
/// buffer_size = "1048576"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct NixlConfig {
    /// Map of backend name (uppercase) -> optional parameters.
    ///
    /// If a backend is present in the map, it's enabled.
    /// The inner HashMap contains optional override parameters.
    /// An empty inner map means use default parameters.
    #[serde(default = "default_backends")]
    pub backends: HashMap<String, HashMap<String, String>>,
}

fn default_backends() -> HashMap<String, HashMap<String, String>> {
    let mut backends = HashMap::new();
    backends.insert("UCX".to_string(), HashMap::new());
    backends
}

impl Default for NixlConfig {
    fn default() -> Self {
        Self {
            backends: default_backends(),
        }
    }
}

impl NixlConfig {
    /// Create a new config with no backends enabled.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Add a backend with default parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend(mut self, name: impl Into<String>) -> Self {
        self.backends
            .insert(name.into().to_uppercase(), HashMap::new());
        self
    }

    /// Add a backend with custom parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend_params(
        mut self,
        name: impl Into<String>,
        params: HashMap<String, String>,
    ) -> Self {
        self.backends.insert(name.into().to_uppercase(), params);
        self
    }

    /// Get the list of enabled backend names (uppercase).
    pub fn enabled_backends(&self) -> Vec<&String> {
        self.backends.keys().collect()
    }

    /// Check if a specific backend is enabled.
    /// Backend name is normalized to uppercase for lookup.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.backends.contains_key(&backend.to_uppercase())
    }

    /// Get parameters for a specific backend.
    /// Backend name is normalized to uppercase for lookup.
    ///
    /// Returns None if the backend is not enabled.
    pub fn backend_params(&self, backend: &str) -> Option<&HashMap<String, String>> {
        self.backends.get(&backend.to_uppercase())
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
    fn test_default_config() {
        let config = NixlConfig::default();
        assert!(config.has_backend("UCX"));
        assert!(!config.has_backend("GDS"));
    }

    #[test]
    fn test_new_empty() {
        let config = NixlConfig::new();
        assert!(!config.has_backend("UCX"));
        assert!(config.backends.is_empty());
    }

    #[test]
    fn test_with_backend() {
        let config = NixlConfig::new().with_backend("ucx").with_backend("gds");

        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("GDS"));
        assert!(!config.has_backend("OTHER"));

        // Keys are stored uppercase
        assert!(config.backends.contains_key("UCX"));
        assert!(config.backends.contains_key("GDS"));
    }

    #[test]
    fn test_with_backend_params() {
        let mut params = HashMap::new();
        params.insert("threads".to_string(), "4".to_string());
        params.insert("buffer_size".to_string(), "1048576".to_string());

        let config = NixlConfig::new()
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
    fn test_lookup_normalizes_to_uppercase() {
        let config = NixlConfig::new().with_backend("ucx");

        // All lookups normalize to uppercase
        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("Ucx"));

        assert!(config.backend_params("ucx").is_some());
        assert!(config.backend_params("UCX").is_some());
    }

    #[test]
    fn test_enabled_backends() {
        let config = NixlConfig::new().with_backend("ucx").with_backend("gds");

        let backends = config.enabled_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&&"UCX".to_string()));
        assert!(backends.contains(&&"GDS".to_string()));
    }

    #[test]
    fn test_iter() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), "value".to_string());

        let config = NixlConfig::new()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        let items: Vec<_> = config.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut params = HashMap::new();
        params.insert("threads".to_string(), "4".to_string());

        let config = NixlConfig::new()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: NixlConfig = serde_json::from_str(&json).unwrap();

        assert!(parsed.has_backend("UCX"));
        assert!(parsed.has_backend("GDS"));
        assert_eq!(
            parsed.backend_params("GDS").unwrap().get("threads"),
            Some(&"4".to_string())
        );
    }
}
