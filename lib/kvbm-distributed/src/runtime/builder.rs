// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder for KvbmRuntime with optional pre-built components.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kvbm_config::KvbmConfig;
use dynamo_memory::nixl::NixlAgent;
use tokio::runtime::{Handle, Runtime};
use velo::Messenger;

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
pub struct KvbmRuntimeBuilder {
    config: KvbmConfig,
    runtime: Option<RuntimeHandle>,
    messenger: Option<Arc<Messenger>>,
    nixl_agent: Option<NixlAgent>,
}

impl KvbmRuntimeBuilder {
    /// Create builder from config.
    pub fn new(config: KvbmConfig) -> Self {
        Self {
            config,
            runtime: None,
            messenger: None,
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

    /// Use an existing Messenger instance.
    pub fn with_messenger(mut self, messenger: Arc<Messenger>) -> Self {
        self.messenger = Some(messenger);
        self
    }

    /// Use an existing NixlAgent instance.
    pub fn with_nixl_agent(mut self, agent: NixlAgent) -> Self {
        self.nixl_agent = Some(agent);
        self
    }

    /// Build runtime for leader role.
    pub async fn build_leader(self) -> Result<super::KvbmRuntime> {
        self.build_internal().await
    }

    /// Build runtime for worker role.
    pub async fn build_worker(self) -> Result<super::KvbmRuntime> {
        self.build_internal().await
    }

    async fn build_internal(self) -> Result<super::KvbmRuntime> {
        // 1. Tokio runtime - use provided or build from config
        let runtime = match self.runtime {
            Some(rt) => rt,
            None => RuntimeHandle::Owned(Arc::new(self.config.tokio.build_runtime()?)),
        };

        // 2. Messenger - use provided or build from config (BEFORE NixL)
        let messenger = match self.messenger {
            Some(m) => m,
            None => self.config.messenger.build_messenger().await?,
        };

        // 3. NixL - use provided or build from config (AFTER Messenger)
        //    Only build if config.nixl is Some (NixL enabled)
        let nixl_agent = match self.nixl_agent {
            Some(agent) => Some(agent),
            None => match &self.config.nixl {
                Some(nixl_config) => {
                    let agent_name = format!("nixl-{}", messenger.instance_id());
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
            messenger,
            nixl_agent,
        })
    }
}
