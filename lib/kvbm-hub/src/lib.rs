// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KVBM Hub
//!
//! Central coordination service for KVBM velo clients (connector / engine).
//! The hub is a single HTTP service that velo clients use as a shared source
//! of truth for peer discovery and registration, plus a velo participant
//! itself for control-plane messaging (heartbeat and, in the future,
//! scheduling / shard placement / etc).
//!
//! # Topology
//!
//! A hub server binds two axum listeners:
//!
//! - **Discovery port** (`1337` default) — HTTP implementation of the
//!   `velo::discovery::PeerDiscovery` protocol.
//! - **Control port** (`8337` default) — registration, heartbeat, health.
//!
//! The server is also a velo node: it exposes velo handlers (bidirectional
//! messaging with clients) in addition to the HTTP surface.
//!
//! # Client usage
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use std::sync::Arc;
//! use kvbm_hub::HubClient;
//!
//! let hub: Arc<HubClient> = kvbm_hub::create_client_builder()
//!     .host("name-or-ip")
//!     .port(1337)
//!     .build()?;
//!
//! let velo = velo::Velo::builder()
//!     .discovery(hub.clone())
//!     .build()
//!     .await?;
//!
//! hub.register_handlers(&velo)?;
//! hub.register_instance(velo.peer_info()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! When the last `Arc<HubClient>` is dropped, the held registration guard
//! issues an HTTP `DELETE` against the hub so the instance is promptly
//! removed from discovery.

pub mod client;
pub mod config;
pub mod features;
pub mod handlers;
pub mod protocol;
pub mod registry;
pub mod server;

pub use client::{HubClient, HubClientBuilder, HubClientConfig, HubRegistrationGuard};
pub use config::HubConfig;
pub use features::conditional_disagg::{ConditionalDisaggClient, ConditionalDisaggManager};
pub use features::{FeatureError, FeatureManager, HubContext};
pub use handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
pub use protocol::{
    CD_PREFILL_QUEUE, ConditionalDisaggConfig, ConditionalDisaggInstancesResponse,
    ConditionalDisaggRole, DEFAULT_CONTROL_PORT, DEFAULT_DISCOVERY_PORT, Feature, FeatureKey,
    PrefillRequest, ProbeResponse,
};
pub use registry::{EvictionCallback, InMemoryRegistry, PeerRegistry, RegistryError};
pub use server::{HubServer, HubServerBuilder, HubServerState};

/// Shorthand for [`HubClientBuilder::new`].
pub fn create_client_builder() -> HubClientBuilder {
    HubClientBuilder::new()
}

/// Shorthand for [`HubServerBuilder::new`].
pub fn create_server_builder() -> HubServerBuilder {
    HubServerBuilder::new()
}
