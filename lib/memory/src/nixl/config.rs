// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL backend configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for NIXL backends.
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
    /// Creates a new configuration with the given backends.
    ///
    /// For an empty configuration with no backends, use [`Default::default()`].
    pub fn new(backends: HashMap<String, HashMap<String, String>>) -> Self {
        Self { backends }
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
    fn test_default_is_empty() {
        let config = NixlBackendConfig::default();
        assert!(config.backends().is_empty()); // default() has no backends
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
}
