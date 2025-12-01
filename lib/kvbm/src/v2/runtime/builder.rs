// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder for KvbmRuntime with optional pre-built components.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kvbm_config::KvbmConfig;
use dynamo_kvbm_config::NovaConfig;
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
    nixl_agent: Option<NixlAgent>,
    nova: Option<Arc<Nova>>,
}

impl KvbmRuntimeBuilder {
    /// Create builder from config.
    pub fn new(config: KvbmConfig) -> Self {
        Self {
            config,
            runtime: None,
            nixl_agent: None,
            nova: None,
        }
    }

    /// Create builder from environment.
    pub fn from_env() -> Result<Self, dynamo_kvbm_config::ConfigError> {
        Ok(Self::new(KvbmConfig::from_env()?))
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

    /// Use an existing NixlAgent.
    pub fn with_nixl_agent(mut self, agent: NixlAgent) -> Self {
        self.nixl_agent = Some(agent);
        self
    }

    /// Use an existing Nova instance.
    pub fn with_nova(mut self, nova: Arc<Nova>) -> Self {
        self.nova = Some(nova);
        self
    }

    /// Build runtime for leader role.
    ///
    /// Uses `config.nova_for_leader()` for Nova configuration.
    pub async fn build_leader(self) -> Result<super::KvbmRuntime> {
        self.build_internal(|config| config.nova_for_leader()).await
    }

    /// Build runtime for worker role.
    ///
    /// Uses `config.nova_for_worker()` for Nova configuration.
    pub async fn build_worker(self) -> Result<super::KvbmRuntime> {
        self.build_internal(|config| config.nova_for_worker()).await
    }

    async fn build_internal<F>(self, nova_config_fn: F) -> Result<super::KvbmRuntime>
    where
        F: FnOnce(&KvbmConfig) -> NovaConfig,
    {
        // Tokio runtime - use provided or build from config
        let runtime = match self.runtime {
            Some(rt) => rt,
            None => RuntimeHandle::Owned(Arc::new(self.config.tokio.build_runtime()?)),
        };

        // Nova - use provided or build from config (role-specific)
        let nova = match self.nova {
            Some(nova) => nova,
            None => nova_config_fn(&self.config).build_nova().await?,
        };

        Ok(super::KvbmRuntime {
            config: self.config,
            runtime,
            nova,
        })
    }
}
