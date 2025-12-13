// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM Runtime - composed infrastructure for kvbm operations.
//!
//! The runtime contains the minimal shared components needed to construct
//! all downstream managers and services:
//! - Tokio runtime (for async execution)
//! - NixlAgent (for RDMA/UCX transfers)
//! - Nova (for distributed RPC)
//!
//! # Usage
//!
//! ```rust,ignore
//! // Build from environment (leader role)
//! let runtime = KvbmRuntime::from_env_leader().await?;
//!
//! // Build with custom config and injected components
//! let config = KvbmConfig::extract_from(
//!     KvbmConfig::figment()
//!         .merge(("nova.backend.tcp_port", 8080u16))
//! )?;
//! let runtime = KvbmRuntime::builder(config)
//!     .with_runtime_handle(Handle::current())
//!     .build_leader()
//!     .await?;
//!
//! // Use runtime components
//! let transfer_mgr = TransferManager::builder()
//!     .nixl_agent(runtime.nixl_agent().clone())
//!     .event_system(runtime.event_system().clone())
//!     .build()?;
//! ```

mod builder;

pub use builder::{KvbmRuntimeBuilder, RuntimeHandle};

use std::sync::Arc;

use dynamo_kvbm_config::KvbmConfig;
use dynamo_memory::nixl::NixlAgent;
use dynamo_nova::Nova;
use tokio::runtime::Handle;

/// KVBM Runtime - composed infrastructure for kvbm operations.
///
/// Contains the minimal shared components needed to construct
/// all downstream managers and services:
/// - Tokio runtime (for async execution)
/// - NixlAgent (for RDMA/UCX transfers)
/// - Nova (for distributed RPC)
///
/// The `LocalEventSystem` is available via `event_system()` which
/// returns the system from Nova.
pub struct KvbmRuntime {
    pub(crate) config: KvbmConfig,
    pub(crate) runtime: RuntimeHandle,
    pub(crate) nova: Arc<Nova>,
    pub(crate) nixl_agent: Option<NixlAgent>,
}

impl KvbmRuntime {
    /// Create a builder for customized construction.
    pub fn builder(config: KvbmConfig) -> KvbmRuntimeBuilder {
        KvbmRuntimeBuilder::new(config)
    }

    /// Quick construction from environment (for leader role).
    pub async fn from_env_leader() -> anyhow::Result<Self> {
        KvbmRuntimeBuilder::from_env()?.build_leader().await
    }

    /// Quick construction from environment (for worker role).
    pub async fn from_env_worker() -> anyhow::Result<Self> {
        KvbmRuntimeBuilder::from_env()?.build_worker().await
    }

    /// Get the configuration.
    pub fn config(&self) -> &KvbmConfig {
        &self.config
    }

    /// Get the tokio runtime handle.
    pub fn handle(&self) -> Handle {
        self.runtime.handle()
    }

    /// Get the tokio runtime handle (convenience alias for handle()).
    pub fn tokio(&self) -> Handle {
        self.handle()
    }

    /// Get Nova.
    pub fn nova(&self) -> &Arc<Nova> {
        &self.nova
    }

    /// Get NixlAgent for RDMA/UCX transfers.
    /// Returns None if NixL is disabled in config.
    pub fn nixl_agent(&self) -> Option<&NixlAgent> {
        self.nixl_agent.as_ref()
    }

    /// Get LocalEventSystem from Nova.
    ///
    /// The event system is used for worker coordination and
    /// transfer notifications.
    pub fn event_system(&self) -> Arc<dynamo_nova::events::LocalEventSystem> {
        self.nova.events().local().clone()
    }
}
