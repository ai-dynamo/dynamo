// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL agent wrapper and configuration.
//!
//! This module provides:
//! - `NixlAgent`: Wrapper around nixl_sys::Agent that tracks initialized backends
//! - `NixlBackendConfig`: Configuration for NIXL backends from environment variables

use anyhow::Result;
use nixl_sys::Agent;
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
    agent: Agent,
    available_backends: HashSet<String>,
}

impl NixlAgent {
    /// Create a NIXL agent without any backends.
    pub fn new(name: &str) -> Result<Self> {
        let agent = Agent::new(name)?;

        Ok(Self {
            agent,
            available_backends: HashSet::new(),
        })
    }

    /// Add a backend to the agent.
    pub fn add_backend(&mut self, backend: &str) -> Result<()> {
        if self.available_backends.contains(&backend.to_uppercase()) {
            return Ok(());
        }
        let backend_upper = backend.to_uppercase();
        match self.agent.get_plugin_params(&backend_upper) {
            Ok((_, params)) => match self.agent.create_backend(&backend_upper, &params) {
                Ok(_) => {
                    self.available_backends.insert(backend_upper);
                }
                Err(e) => {
                    anyhow::bail!("Failed to create nixl backend: {}", e);
                }
            },
            Err(_) => {
                anyhow::bail!("No {} plugin found", backend_upper);
            }
        }
        Ok(())
    }

    /// Create a NIXL agent requiring ALL specified backends to be available.
    ///
    /// Unlike `new_with_backends()` which continues if some backends fail, this method
    /// will return an error if ANY backend fails to initialize. Use this in production
    /// when specific backends are mandatory.
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
    pub fn with_backends(name: &str, backends: &[&str]) -> Result<Self> {
        let mut agent = Self::new(name)?;
        let mut failed_backends = Vec::new();

        for backend in backends {
            let backend_upper = backend.to_uppercase();
            match agent.add_backend(&backend_upper) {
                Ok(_) => {
                    tracing::debug!("Initialized NIXL backend: {}", backend_upper);
                }
                Err(e) => {
                    tracing::error!("Failed to initialize {} backend: {}", backend_upper, e);
                    failed_backends.push((backend_upper, e.to_string()));
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

        Ok(agent)
    }

    /// Get a reference to the underlying raw NIXL agent.
    pub fn raw_agent(&self) -> &Agent {
        &self.agent
    }

    /// Consume and return the underlying raw NIXL agent.
    ///
    /// **Warning**: Once consumed, backend tracking is lost. Use this only when
    /// interfacing with code that requires `nixl_sys::Agent` directly.
    pub fn into_raw_agent(self) -> Agent {
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
    /// Note: In general, you want to instantiate all your backends before you start registering memory.
    /// We may change this to a builder pattern in the future to enforce all backends are instantiated
    /// before you start registering memory.
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
    type Target = Agent;

    fn deref(&self) -> &Self::Target {
        &self.agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_backend_tracking() {
        // Try to create agent with UCX
        let agent = NixlAgent::with_backends("test", &["UCX"]).expect("Need UCX for test");

        // Should succeed if UCX is available
        assert!(agent.has_backend("UCX"));
        assert!(agent.has_backend("ucx")); // Case insensitive
    }

    #[test]
    fn test_require_backend() {
        let agent = NixlAgent::with_backends("test", &["UCX"]).expect("Need UCX for test");

        // Should succeed for available backend
        assert!(agent.require_backend("UCX").is_ok());

        // Should fail for unavailable backend
        assert!(agent.require_backend("GDS_MT").is_err());
    }

    #[test]
    fn test_require_backends_strict() {
        // Should succeed if UCX is available
        let agent =
            NixlAgent::with_backends("test_strict", &["UCX"]).expect("Failed to require backends");
        assert!(agent.has_backend("UCX"));

        // Should fail if any backend is missing (GDS likely not available)
        let result = NixlAgent::with_backends("test_strict_fail", &["UCX", "DUDE"]);
        assert!(result.is_err());
    }
}
