// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared identities for routing partitions, logical KV indexers, and cluster-local producer pools.
//!
//! Hashed indexer identity material is resolved on control paths. Mutation and query paths carry
//! only those fixed-size values or physical lane indices.

/// Routing group used when a request does not specify one.
pub use dynamo_router_protocol::identity::{
    CacheOwnerId, CacheSemanticsId, ClusterId, DEFAULT_ROUTING_GROUP, IdentityParseError,
    IdentitySource, IndexerDomainId, PoolId, RoutingPartitionId, RoutingPartitionRef,
    RoutingScopeId, StableDpSlotId,
};

pub use dynamo_kv_indexer::identity::{
    CanonicalIdentityMaterial, ExplicitIdentityMap, IdentitySpecError, IndexerIdentitySpec,
    MAX_EXPLICIT_IDENTITY_ENTRIES, MAX_EXPLICIT_IDENTITY_KEY_BYTES,
    MAX_EXPLICIT_IDENTITY_VALUE_BYTES,
};
