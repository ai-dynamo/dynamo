// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-scoped KV-cache Relay with endpoint-local CKF pools.

mod actor;
mod discovery;
mod host;
mod identity;
mod load;
mod pool_registry;
#[cfg(feature = "kv-dc-relay-proto")]
pub mod protocol;
#[cfg(feature = "kv-dc-relay-wan")]
mod publication_codec;
#[cfg(feature = "kv-dc-relay-wan")]
mod publication_hub;
mod resolution;
mod topology;
#[cfg(feature = "kv-dc-relay-wan")]
mod transport;
#[cfg(feature = "kv-dc-relay-wan")]
mod transport_config;

pub use discovery::KvDcRelayDiscoveryConfig;
pub use host::{
    DEFAULT_EXPECTED_UNIQUE_BLOCKS, KvDcRelay, KvDcRelayConfig, KvDcRelayError, KvDcRelayHealth,
    KvDcRelayProducerConfig,
};
#[cfg(feature = "ckf-diagnostics")]
pub use host::{
    KvDcRelayActorStats, KvDcRelayAggregationStats, KvDcRelayCacheDomainStats,
    KvDcRelayDiagnosticSnapshot, KvDcRelayEndpointStats, KvDcRelayIdentityStats,
    KvDcRelayMemberStats, KvDcRelayMemoryStats, KvDcRelayPublicationStats, KvDcRelayRecoveryStats,
    KvDcRelayStats,
};
pub use identity::{
    CanonicalModelId, CanonicalModelIdError, CanonicalModelRegistration, DcPoolCatalog,
    DcPoolDescriptor, DcRelayIdentity, KvQueryHashFormat, KvQuerySemantics, KvQuerySemanticsError,
    ModelAlias, ModelAliasError, ModelTarget, PoolIdentitySources, WorkerRole,
};
pub use load::PoolLoadSnapshot;
pub use topology::{
    AdapterReadiness, TopologyEntry, TopologyMember, TopologyReadinessState, TopologySnapshot,
};
#[cfg(feature = "kv-dc-relay-wan")]
pub use transport_config::KvDcRelayTransportConfig;
