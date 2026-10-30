// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool Relay with endpoint-local CKF pools.

pub use dynamo_global_common::PoolKey as RoutingPoolKey;

mod actor;
mod discovery;
mod global_routers;
mod host;
mod identity;
mod load;
mod membership_watch;
mod namespace_source;
mod pool_registry;
mod publication;
mod resolution;
mod serving_load;
mod topology;
pub mod wan;

pub use host::{
    DEFAULT_EXPECTED_UNIQUE_BLOCKS, PoolRelay, PoolRelayConfig, PoolRelayError, PoolRelayHealth,
    PoolRelayProducerConfig, PoolRelaySources,
};
#[cfg(feature = "ckf-diagnostics")]
pub use host::{
    PoolRelayActorStats, PoolRelayAggregationStats, PoolRelayCacheDomainStats,
    PoolRelayDiagnosticSnapshot, PoolRelayEndpointStats, PoolRelayIdentityStats,
    PoolRelayMemberStats, PoolRelayMemoryStats, PoolRelayPublicationStats, PoolRelayRecoveryStats,
    PoolRelayStats,
};
pub use identity::{
    CanonicalModelId, CanonicalModelIdError, CanonicalModelRegistration, KvQueryHashFormat,
    KvQuerySemantics, KvQuerySemanticsError, ModelAlias, ModelAliasError, ModelTarget,
    PoolCatalogSnapshot, PoolDescriptor, PoolIdentitySources, PoolRelayIdentity, WorkerRole,
};
pub use load::PoolLoadSnapshot;
pub use namespace_source::discovery::PoolRelayDiscoveryConfig;
pub use namespace_source::{PoolRelaySourcesStatus, file::PoolRelaySourcesFile};
pub use publication::{
    PoolPublicationStream, PublicationError, PublicationErrorKind, PublicationFrame,
    PublicationFrameKind, RelayPublicationSource,
};
pub use topology::{
    AdapterReadiness, TopologyEntry, TopologyMember, TopologyReadinessState, TopologySnapshot,
};

/// Whether this process publishes the serving-load streams a Relay
/// aggregates. Off unless `DYN_SERVING_LOAD_PUBLISH` is truthy.
pub(crate) fn serving_load_publishing_enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(
        dynamo_runtime::config::environment_names::llm::DYN_SERVING_LOAD_PUBLISH,
    )
}
