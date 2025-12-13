// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder for KvbmRuntime with optional pre-built components.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kvbm_config::KvbmConfig;
use dynamo_memory::nixl::NixlAgent;
use dynamo_nova::Nova;
use tokio::runtime::{Handle, Runtime};

/// Runtime handle - either owned or borrowed.
pub enum RuntimeHandle {
    /// Owned runtime (created by builder).
    Owned(Arc<Runtime>),
    /// Borrowed handle (external runtime).
    Handle(Handle),
}

impl RuntimeHandle {
    /// Get a handle to the runtime.
    pub fn handle(&self) -> Handle {
        match self {
            RuntimeHandle::Owned(rt) => rt.handle().clone(),
            RuntimeHandle::Handle(h) => h.clone(),
        }
    }
}

/// Builder for KvbmRuntime with optional pre-built components.
///
/// The builder allows injecting pre-built components or building them from config:
/// - If a component is provided, it's used directly
/// - If not provided, the component is built from the config
///
/// # Example
/// ```rust,ignore
/// // Build everything from environment
/// let runtime = KvbmRuntimeBuilder::from_env()?
///     .build_leader()
///     .await?;
///
/// // Inject existing tokio handle
/// let runtime = KvbmRuntimeBuilder::from_env()?
///     .with_runtime_handle(Handle::current())
///     .build_leader()
///     .await?;
/// ```
pub struct KvbmRuntimeBuilder {
    config: KvbmConfig,
    runtime: Option<RuntimeHandle>,
    nova: Option<Arc<Nova>>,
    nixl_agent: Option<NixlAgent>,
}

impl KvbmRuntimeBuilder {
    /// Create builder from config.
    pub fn new(config: KvbmConfig) -> Self {
        Self {
            config,
            runtime: None,
            nova: None,
            nixl_agent: None,
        }
    }

    /// Create builder from environment.
    pub fn from_env() -> Result<Self, dynamo_kvbm_config::ConfigError> {
        Ok(Self::new(KvbmConfig::from_env()?))
    }

    /// Create builder from JSON config string (merged with env/files).
    ///
    /// JSON has highest priority - overrides env vars, TOML files, and defaults.
    /// This is the primary entrypoint for vLLM's `kv_connector_extra_config` dict.
    pub fn from_json(json: &str) -> Result<Self, dynamo_kvbm_config::ConfigError> {
        Ok(Self::new(KvbmConfig::from_figment_with_json(json)?))
    }

    /// Use an existing tokio Runtime (takes ownership via Arc).
    pub fn with_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime = Some(RuntimeHandle::Owned(runtime));
        self
    }

    /// Use an existing tokio Handle (borrowed).
    pub fn with_runtime_handle(mut self, handle: Handle) -> Self {
        self.runtime = Some(RuntimeHandle::Handle(handle));
        self
    }

    /// Use an existing Nova instance.
    pub fn with_nova(mut self, nova: Arc<Nova>) -> Self {
        self.nova = Some(nova);
        self
    }

    /// Use an existing NixlAgent instance.
    pub fn with_nixl_agent(mut self, agent: NixlAgent) -> Self {
        self.nixl_agent = Some(agent);
        self
    }

    /// Build runtime for leader role.
    ///
    /// Uses the `nova` config from the KvbmConfig. Role-specific Nova settings
    /// should be provided via Figment profiles (e.g., `profile.leader.nova.*`).
    pub async fn build_leader(self) -> Result<super::KvbmRuntime> {
        self.build_internal().await
    }

    /// Build runtime for worker role.
    ///
    /// Uses the `nova` config from the KvbmConfig. Role-specific Nova settings
    /// should be provided via Figment profiles (e.g., `profile.worker.nova.*`).
    pub async fn build_worker(self) -> Result<super::KvbmRuntime> {
        self.build_internal().await
    }

    async fn build_internal(self) -> Result<super::KvbmRuntime> {
        // 1. Tokio runtime - use provided or build from config
        let runtime = match self.runtime {
            Some(rt) => rt,
            None => RuntimeHandle::Owned(Arc::new(self.config.tokio.build_runtime()?)),
        };

        // 2. Nova - use provided or build from config (BEFORE NixL)
        let nova = match self.nova {
            Some(nova) => nova,
            None => self.config.nova.build_nova().await?,
        };

        // 3. NixL - use provided or build from config (AFTER Nova)
        //    Only build if config.nixl is Some (NixL enabled)
        let nixl_agent = match self.nixl_agent {
            Some(agent) => Some(agent),
            None => match &self.config.nixl {
                Some(nixl_config) => {
                    let agent_name = format!("nixl-{}", nova.instance_id());
                    let backend_config = nixl_config.clone().into();
                    Some(NixlAgent::from_nixl_backend_config(
                        &agent_name,
                        backend_config,
                    )?)
                }
                None => None, // NixL disabled
            },
        };

        Ok(super::KvbmRuntime {
            config: self.config,
            runtime,
            nova,
            nixl_agent,
        })
    }
}
