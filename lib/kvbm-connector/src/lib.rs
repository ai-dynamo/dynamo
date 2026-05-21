// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration modules for external frameworks.
//!
//! This module provides trait-based abstractions for integrating with
//! external serving frameworks like vLLM, allowing pure Rust code to
//! remain independent of framework-specific types.

pub mod common;
pub mod config;
pub mod connector;
pub mod vllm;

// Re-export key types for convenience
pub use common::{
    CachedRequestData, NewRequestData, Request, RequestMetadata, SchedulerConnectorState,
    SchedulerOutput,
};
pub use config::{AttentionConfig, IntegrationsConfig, ParallelConfig};

// Re-export workspace types used throughout this crate
pub use kvbm_common::{BlockId, SequenceHash};
pub use kvbm_engine::{G1, G2, G3, G4, InstanceId, KvbmRuntime};

// Re-exports for bindings — runtime construction
pub use kvbm_config::KvbmConfig;
pub use kvbm_engine::{KvbmRuntimeBuilder, PeerInfo, WorkerAddress};
// Re-export for bindings — consolidator source tagging
pub use kvbm_engine::leader::EventSource;

/// If the config carries `leader.hub`, construct a [`kvbm_hub::HubClient`]
/// and pre-seed the runtime builder so velo uses it as its peer-discovery
/// backend.
///
/// Leaders with no `leader.hub` return the builder unmodified — velo falls
/// back to whatever static discovery the messenger config specifies (or none).
///
/// This is the wiring the leader's `InstanceLeader::messenger`
/// `discover_and_register_peer` flow needs. Without it, control-plane calls
/// like `core/register_leader` fail with "No discovery backend configured"
/// even though the hub knows every registered peer. The CD coordinator's
/// internal `HubPeerResolver` uses the hub directly and is unaffected — but
/// the public control-plane surface (which the P2P flow drives) goes
/// through velo.
///
/// Workers do not need this — cross-worker data transfer rides NIXL, not
/// velo, so only leaders configure a velo peer registry.
pub fn seed_leader_builder_with_hub_discovery(
    config: &KvbmConfig,
    builder: KvbmRuntimeBuilder,
) -> anyhow::Result<KvbmRuntimeBuilder> {
    let Some(hub) = config.hub.as_ref() else {
        return Ok(builder);
    };
    let hub_client = connector::leader::disagg::build_hub_client(&hub.url)?;
    // Coerce Arc<HubClient> → Arc<dyn PeerDiscovery>. HubClient's
    // `impl PeerDiscovery` (in kvbm-hub) makes this an upcast.
    let discovery: std::sync::Arc<dyn velo::discovery::PeerDiscovery> = hub_client;
    Ok(builder.with_discovery(discovery))
}

// Re-exports for bindings — memory/tensor types (already in public API via ConnectorWorkerInterface)
pub use dynamo_memory::{MemoryDescriptor, StorageKind, TensorDescriptor};
pub mod memory {
    //! Re-exports from `dynamo-memory` for bindings convenience.
    pub use dynamo_memory::nixl;
}

#[cfg(feature = "testing")]
pub mod testing;
