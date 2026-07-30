// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-scoped KV-cache Relay and endpoint-independent CKF identity boundary.

mod actor;
mod discovery;
mod host;
mod identity;
#[cfg(feature = "kv-dc-relay-wan")]
mod load;
mod pool_registry;
#[cfg(feature = "kv-dc-relay-proto")]
pub mod protocol;
#[cfg(feature = "kv-dc-relay-wan")]
mod publication_codec;
#[cfg(feature = "kv-dc-relay-wan")]
mod publication_hub;
#[cfg(feature = "kv-dc-relay-wan")]
mod readiness;
mod resolution;

pub use host::{
    DEFAULT_EXPECTED_UNIQUE_BLOCKS, KvDcRelay, KvDcRelayConfig, KvDcRelayError, KvDcRelayHealth,
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
    DcPoolDescriptor, DcRelayIdentity, ModelAlias, ModelAliasError, ModelTarget,
    PoolIdentitySources,
};
