// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL agent wrapper and configuration.
//!
//! This module provides:
//! - `NixlAgent`: Wrapper around nixl_sys::Agent that tracks initialized backends
//! - `NixlBackendConfig`: Configuration for NIXL backends from environment variables

mod config;

pub use config::NixlBackendConfig;

use anyhow::Result;
use nixl_sys::{Agent as RawNixlAgent, Params};
use std::collections::HashSet;

/// A NIXL agent wrapper that tracks which backends were successfully initialized.
///
/// This wrapper provides:
/// - Runtime validation of backend availability
/// - Clear error messages when operations need unavailable backends
/// - Single source of truth for backend state in tests and production
///
/// # Backend Tracking
///
/// Since `nixl_sys::Agent` doesn't provide a method to query active backends,
/// we track them during initialization. The `available_backends` set is populated
/// based on successful `create_backend()` calls.
#[derive(Clone, Debug)]
pub struct NixlAgent {
    agent: RawNixlAgent,
    available_backends: HashSet<String>,
}

impl NixlAgent {
    /// Generic helper function to create backend parameters from environment variables.
    ///
    /// Parses environment variables with the pattern:
    /// `DYN_KVBM_NIXL_BACKEND_{BACKEND_NAME}_{PARAM}`
    ///
    /// For example:
    /// - `DYN_KVBM_NIXL_BACKEND_POSIX_PATH=/my/path` -> `path` parameter for POSIX backend
    /// - `DYN_KVBM_NIXL_BACKEND_UCX_DEVICE=mlx5_0`   -> `device` parameter for UCX backend
    /// - `DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY=...`  -> `access_key` parameter for OBJ backend
    ///
    /// Returns `None` if no custom parameters are found for this backend.
    ///
    /// This is public so it can be used when constructing custom configurations before
    /// calling `new_with_backends_and_params()`.
    pub fn create_backend_params_from_env(backend_name: &str) -> Result<Option<Params>> {
        let prefix = format!("DYN_KVBM_NIXL_BACKEND_{}_", backend_name);
        let mut param_pairs: Vec<(String, String)> = Vec::new();

        // Parse DYN_KVBM_NIXL_BACKEND_{BACKEND_NAME}_* variables
        for (env_key, env_value) in std::env::vars() {
            if let Some(param_name) = env_key.strip_prefix(&prefix) {
                let param_key = param_name.to_lowercase();
                param_pairs.push((param_key, env_value));
            }
        }

        if param_pairs.is_empty() {
            return Ok(None);
        }

        tracing::debug!(
            "{} backend parameters configured from environment: {:?}",
            backend_name,
            param_pairs
        );

        // Create Params object from the collected key-value pairs
        let params =
            Params::from(param_pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())))?;
        Ok(Some(params))
    }

    /// Create a new NIXL agent with the specified backends and explicit parameters.
    ///
    /// This method allows you to provide explicit `Params` for each backend. If a backend fails,
    /// it logs a warning but continues with remaining backends. At least one backend must
    /// succeed or this returns an error.
    ///
    /// # Arguments
    /// * `name` - Agent name
    /// * `backends` - List of (backend_name, params) tuples
    ///
    /// # Returns
    /// A `NixlAgent` that tracks which backends were successfully initialized.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Agent creation fails
    /// - All backend initialization attempts fail
    ///
    /// # Example
    /// ```ignore
    /// let obj_params = Params::try_from_iter([
    ///     ("bucket", "my-bucket"),
    ///     ("region", "us-west-2"),
    /// ])?;
    ///
    /// let agent = NixlAgent::new_with_backends_and_params(
    ///     "my-agent",
    ///     &[("OBJ", obj_params)]
    /// )?;
    /// ```
    pub fn new_with_backends_and_params(name: &str, backends: &[(&str, Params)]) -> Result<Self> {
        let agent = RawNixlAgent::new(name)?;
        let mut available_backends = HashSet::new();

        for (backend, params) in backends {
            let backend_upper = backend.to_uppercase();

            match agent.create_backend(&backend_upper, params) {
                Ok(_) => {
                    available_backends.insert(backend_upper.clone());
                    tracing::debug!("{} backend created with provided parameters", backend_upper);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create {} backend with provided params: {}. Operations requiring this backend will fail.",
                        backend_upper, e
                    );
                }
            }
        }

        if available_backends.is_empty() {
            let backend_names: Vec<_> = backends.iter().map(|(name, _)| *name).collect();
            anyhow::bail!(
                "Failed to initialize any NIXL backends from {:?}",
                backend_names
            );
        }

        Ok(Self {
            agent,
            available_backends,
        })
    }

    /// Create a new NIXL agent with the specified backends.
    ///
    /// Attempts to initialize all requested backends. If a backend fails, it logs
    /// a warning but continues with remaining backends. At least one backend must
    /// succeed or this returns an error.
    ///
    /// This method will first check for custom parameters in environment variables
    /// (DYN_KVBM_NIXL_BACKEND_{BACKEND_NAME}_{PARAM}), and if not found, will use
    /// the default plugin parameters.
    ///
    /// # Arguments
    /// * `name` - Agent name
    /// * `backends` - List of backend names to try (e.g., `&["UCX", "GDS_MT", "POSIX"]`)
    ///
    /// # Returns
    /// A `NixlAgent` that tracks which backends were successfully initialized.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Agent creation fails
    /// - All backend initialization attempts fail
    pub fn new_with_backends(name: &str, backends: &[&str]) -> Result<Self> {
        let agent = RawNixlAgent::new(name)?;
        let mut available_backends = HashSet::new();

        for backend in backends {
            let backend_upper = backend.to_uppercase();

            // Try to get custom parameters from environment first
            match Self::create_backend_params_from_env(&backend_upper) {
                Ok(Some(custom_params)) => {
                    // Custom parameters found - use them
                    match agent.create_backend(&backend_upper, &custom_params) {
                        Ok(_) => {
                            available_backends.insert(backend_upper.clone());
                            tracing::info!(
                                "{} backend created with custom configuration from environment",
                                backend_upper
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create {} backend with custom params: {}. Check your DYN_KVBM_NIXL_BACKEND_{}_* environment variables.",
                                backend_upper, e, backend_upper
                            );
                            eprintln!(
                                "Failed to create {} backend with custom params: {}. Check your DYN_KVBM_NIXL_BACKEND_{}_* environment variables.",
                                backend_upper, e, backend_upper
                            );
                        }
                    }
                }
                Ok(None) => {
                    // No custom parameters - fall back to default plugin parameters
                    tracing::debug!(
                        "Attempting to create {} backend with default params",
                        backend_upper
                    );
                    match agent.get_plugin_params(&backend_upper) {
                        Ok((_, params)) => match agent.create_backend(&backend_upper, &params) {
                            Ok(_) => {
                                available_backends.insert(backend_upper.clone());
                                tracing::info!("Successfully created {} backend", backend_upper);
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to create {} backend: {}. Operations requiring this backend will fail.",
                                    backend_upper, e
                                );
                                eprintln!(
                                    "Failed to create {} backend: {}. Operations requiring this backend will fail.",
                                    backend_upper, e
                                );
                            }
                        },
                        Err(e) => {
                            tracing::error!(
                                "No {} plugin found: {:?}. Operations requiring this backend will fail.",
                                backend_upper, e
                            );
                            eprintln!(
                                "No {} plugin found. Operations requiring this backend will fail.",
                                backend_upper
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to parse {} backend parameters from environment: {}",
                        backend_upper,
                        e
                    );
                    eprintln!(
                        "Failed to parse {} backend parameters from environment: {}",
                        backend_upper, e
                    );
                }
            }
        }

        if available_backends.is_empty() {
            anyhow::bail!("Failed to initialize any NIXL backends from {:?}", backends);
        }

        Ok(Self {
            agent,
            available_backends,
        })
    }

    /// Create a NIXL agent requiring ALL specified backends with explicit parameters.
    ///
    /// Unlike `new_with_backends_and_params()` which continues if some backends fail,
    /// this method will return an error if ANY backend fails to initialize. Use this
    /// in production when specific backends with specific configurations are mandatory.
    ///
    /// # Arguments
    /// * `name` - Agent name
    /// * `backends` - List of (backend_name, params) tuples that MUST be available
    ///
    /// # Returns
    /// A `NixlAgent` with all requested backends initialized.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Agent creation fails
    /// - Any backend fails to initialize
    ///
    /// # Example
    /// ```ignore
    /// let obj_params = Params::try_from_iter([
    ///     ("bucket", "my-bucket"),
    /// ])?;
    ///
    /// let agent = NixlAgent::require_backends_with_params(
    ///     "worker-0",
    ///     &[("OBJ", obj_params)]
    /// )?;
    /// ```
    pub fn require_backends_with_params(name: &str, backends: &[(&str, Params)]) -> Result<Self> {
        let agent = RawNixlAgent::new(name)?;
        let mut available_backends = HashSet::new();
        let mut failed_backends = Vec::new();

        for (backend, params) in backends {
            let backend_upper = backend.to_uppercase();

            match agent.create_backend(&backend_upper, params) {
                Ok(_) => {
                    available_backends.insert(backend_upper.clone());
                    tracing::debug!("{} backend created with provided parameters", backend_upper);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create {} backend with provided params: {}",
                        backend_upper, e
                    );
                    failed_backends.push((
                        backend_upper.clone(),
                        format!("create with provided params failed: {}", e),
                    ));
                }
            }
        }

        if !failed_backends.is_empty() {
            let error_details: Vec<String> = failed_backends
                .iter()
                .map(|(name, reason)| format!("{}: {}", name, reason))
                .collect();
            anyhow::bail!(
                "Failed to initialize required backends: [{}]",
                error_details.join(", ")
            );
        }

        Ok(Self {
            agent,
            available_backends,
        })
    }

    /// Create a NIXL agent requiring ALL specified backends to be available.
    ///
    /// Unlike `new_with_backends()` which continues if some backends fail, this method
    /// will return an error if ANY backend fails to initialize. Use this in production
    /// when specific backends are mandatory.
    ///
    /// This method will first check for custom parameters in environment variables
    /// (DYN_KVBM_NIXL_BACKEND_{BACKEND_NAME}_{PARAM}), and if not found, will use
    /// the default plugin parameters.
    ///
    /// # Arguments
    /// * `name` - Agent name
    /// * `backends` - List of backend names that MUST be available
    ///
    /// # Returns
    /// A `NixlAgent` with all requested backends initialized.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Agent creation fails
    /// - Any backend fails to initialize
    ///
    /// # Example
    /// ```ignore
    /// // In production: require both UCX and GDS, fail if either is missing
    /// let agent = NixlAgent::require_backends("worker-0", &["UCX", "GDS_MT"])?;
    /// ```
    pub fn require_backends(name: &str, backends: &[&str]) -> Result<Self> {
        let agent = RawNixlAgent::new(name)?;
        let mut available_backends = HashSet::new();
        let mut failed_backends = Vec::new();

        for backend in backends {
            let backend_upper = backend.to_uppercase();

            // Try to get custom parameters from environment first
            match Self::create_backend_params_from_env(&backend_upper) {
                Ok(Some(custom_params)) => {
                    // Custom parameters found - use them
                    match agent.create_backend(&backend_upper, &custom_params) {
                        Ok(_) => {
                            available_backends.insert(backend_upper.clone());
                            tracing::debug!(
                                "{} backend created with custom configuration from environment",
                                backend_upper
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "Failed to create {} backend with custom params: {}",
                                backend_upper, e
                            );
                            failed_backends.push((
                                backend_upper.clone(),
                                format!("create with custom params failed: {}", e),
                            ));
                        }
                    }
                }
                Ok(None) => {
                    // No custom parameters - fall back to default plugin parameters
                    match agent.get_plugin_params(&backend_upper) {
                        Ok((_, params)) => match agent.create_backend(&backend_upper, &params) {
                            Ok(_) => {
                                available_backends.insert(backend_upper);
                            }
                            Err(e) => {
                                eprintln!("Failed to create {} backend: {}", backend_upper, e);
                                failed_backends
                                    .push((backend_upper.clone(), format!("create failed: {}", e)));
                            }
                        },
                        Err(e) => {
                            eprintln!("No {} plugin found", backend_upper);
                            failed_backends
                                .push((backend_upper.clone(), format!("plugin not found: {}", e)));
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Failed to parse {} backend parameters from environment: {}",
                        backend_upper, e
                    );
                    failed_backends.push((
                        backend_upper.clone(),
                        format!("params parsing failed: {}", e),
                    ));
                }
            }
        }

        if !failed_backends.is_empty() {
            let error_details: Vec<String> = failed_backends
                .iter()
                .map(|(name, reason)| format!("{}: {}", name, reason))
                .collect();
            anyhow::bail!(
                "Failed to initialize required backends: [{}]",
                error_details.join(", ")
            );
        }

        Ok(Self {
            agent,
            available_backends,
        })
    }

    /// Create a NIXL agent with default backends for testing/development.
    ///
    /// Attempts to initialize UCX, GDS, and POSIX backends. If some are unavailable,
    /// continues with whatever succeeds. This ensures code works in various environments.
    pub fn new_default(name: &str) -> Result<Self> {
        Self::new_with_backends(name, &["UCX", "GDS_MT", "POSIX"])
    }

    /// Get a reference to the underlying raw NIXL agent.
    pub fn raw_agent(&self) -> &RawNixlAgent {
        &self.agent
    }

    /// Consume and return the underlying raw NIXL agent.
    ///
    /// **Warning**: Once consumed, backend tracking is lost. Use this only when
    /// interfacing with code that requires `nixl_sys::Agent` directly.
    pub fn into_raw_agent(self) -> RawNixlAgent {
        self.agent
    }

    /// Check if a specific backend is available.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.available_backends.contains(&backend.to_uppercase())
    }

    /// Get all available backends.
    pub fn backends(&self) -> &HashSet<String> {
        &self.available_backends
    }

    /// Require a specific backend, returning an error if unavailable.
    ///
    /// Use this at the start of operations that need specific backends.
    ///
    /// # Example
    /// ```ignore
    /// agent.require_backend("GDS_MT)?;
    /// // Proceed with GDS-specific operations
    /// ```
    pub fn require_backend(&self, backend: &str) -> Result<()> {
        let backend_upper = backend.to_uppercase();
        if self.has_backend(&backend_upper) {
            Ok(())
        } else {
            anyhow::bail!(
                "Operation requires {} backend, but it was not initialized. Available backends: {:?}",
                backend_upper,
                self.available_backends
            )
        }
    }
}

// Delegate common methods to the underlying agent
impl std::ops::Deref for NixlAgent {
    type Target = RawNixlAgent;

    fn deref(&self) -> &Self::Target {
        &self.agent
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;

    #[test]
    fn test_agent_backend_tracking() {
        // Try to create agent with UCX
        let agent = NixlAgent::new_with_backends("test", &["UCX"]);

        // Should succeed if UCX is available
        if let Ok(agent) = agent {
            assert!(agent.has_backend("UCX"));
            assert!(agent.has_backend("ucx")); // Case insensitive
        }
    }

    #[test]
    fn test_require_backend() {
        let agent = NixlAgent::new_with_backends("test", &["UCX"]).expect("Need UCX for test");

        // Should succeed for available backend
        assert!(agent.require_backend("UCX").is_ok());

        // Should fail for unavailable backend
        assert!(agent.require_backend("GDS_MT").is_err());
    }

    #[test]
    fn test_require_backends_strict() {
        // Should succeed if UCX is available
        let agent = NixlAgent::require_backends("test_strict", &["UCX"])
            .expect("Failed to require backends");
        assert!(agent.has_backend("UCX"));

        // Should fail if any backend is missing (GDS likely not available)
        let result = NixlAgent::require_backends("test_strict_fail", &["UCX", "DUDE"]);
        assert!(result.is_err());
    }
}
