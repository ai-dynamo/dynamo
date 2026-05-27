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
/// `kvbmctl` config rendering. Gated behind the `kvbmctl` feature because it
/// depends on `kvbm-config`, which transitively pulls CUDA (cudarc) — kept out
/// of the default CPU-only hub build.
#[cfg(feature = "kvbmctl")]
pub mod render;
pub mod server;
pub mod web;

pub use client::{HubClient, HubClientBuilder, HubClientConfig, HubRegistrationGuard};
pub use config::{HubConfig, IndexerConfig};
#[cfg(feature = "kvbmctl")]
pub use features::cli::{FeatureCli, feature_clis, hub_arg};
pub use features::control_plane::ControlPlaneManager;
pub use features::disagg::{
    ConditionalDisaggClient, ConditionalDisaggInstancesResponse, ConditionalDisaggManager,
};
pub use features::indexer::{
    FindBlocksHit, IndexerConfigResponse, IndexerLookupClient, IndexerManager, InstancesResponse,
    PositionalIndex, QueryRequest, QueryResponse,
};
#[cfg(feature = "kvbmctl")]
pub use features::p2p::cli::{p2p_command, run_p2p};
pub use features::p2p::{P2pClient, P2pManager};
pub use features::prefill_router::{
    CALIBRATE_HANDLER, CalibrationDefaults, CalibrationRequest, CalibrationResponse,
    CalibrationResults, CalibrationSnapshot, DispatchOutcome, HttpExecutionBackend,
    PREFILL_DISPATCH_HANDLER, PerformanceModel, PrefillDispatchRequest, PrefillDispatchResponse,
    PrefillExecutionBackend, PrefillRequestDispatcher, PrefillRouter, PrefillRouterManager,
    RawCalibrationPayload, RawTrace, RecordingDispatcher, ResolvedCalibrationRequest, ScatterData,
    Selector, SelectorConfig, VeloExecutionBackend, analyze_calibration,
};
pub use features::{FeatureConfigRequirements, FeatureError, FeatureManager, HubContext};
pub use handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
pub use kvbm_common::BlockLayoutMode;
pub use protocol::{
    CD_PREFILL_QUEUE, ConditionalDisaggConfig, ConditionalDisaggRole, DEFAULT_CONTROL_PORT,
    DEFAULT_DISCOVERY_PORT, Feature, FeatureDescriptor, FeatureKey, HubConfigResponse,
    IndexerFeatureConfig, P2pConfig, PrefillBackendAdvertisement, PrefillRequest,
    PrefillRouterConfig, PrimaryConfig, ProbeResponse, RuntimeConfigSummary, VllmHttpEndpoint,
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
