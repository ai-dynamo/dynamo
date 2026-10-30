// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identities every router crate names: partitions, pools, owners, domains, and their sources.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub const DEFAULT_ROUTING_GROUP: &str = "default";

/// Logical routing partition shared by selection, indexing, and active-sequence tracking.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingPartitionId {
    pub model_name: String,
    pub routing_group: String,
}

impl RoutingPartitionId {
    pub fn new(model_name: impl Into<String>, routing_group: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            routing_group: routing_group.into(),
        }
    }

    pub fn as_ref(&self) -> RoutingPartitionRef<'_> {
        RoutingPartitionRef::new(&self.model_name, &self.routing_group)
    }
}

impl fmt::Display for RoutingPartitionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(formatter)
    }
}

/// Borrowed view of a [`RoutingPartitionId`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RoutingPartitionRef<'a> {
    pub model_name: &'a str,
    pub routing_group: &'a str,
}

impl<'a> RoutingPartitionRef<'a> {
    pub const fn new(model_name: &'a str, routing_group: &'a str) -> Self {
        Self {
            model_name,
            routing_group,
        }
    }

    pub fn into_owned(self) -> RoutingPartitionId {
        RoutingPartitionId::new(self.model_name, self.routing_group)
    }
}

impl fmt::Display for RoutingPartitionRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "model={} routing_group={}",
            self.model_name, self.routing_group
        )
    }
}

impl<'a> From<&'a RoutingPartitionId> for RoutingPartitionRef<'a> {
    fn from(partition: &'a RoutingPartitionId) -> Self {
        partition.as_ref()
    }
}

impl From<RoutingPartitionRef<'_>> for RoutingPartitionId {
    fn from(partition: RoutingPartitionRef<'_>) -> Self {
        partition.into_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentitySource {
    DefaultDerived,
    Explicit,
}

macro_rules! digest_identity {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name {
            digest: [u8; 16],
            source: IdentitySource,
        }

        impl $name {
            pub const fn new(digest: [u8; 16], source: IdentitySource) -> Self {
                Self { digest, source }
            }

            pub const fn digest(self) -> [u8; 16] {
                self.digest
            }

            pub const fn source(self) -> IdentitySource {
                self.source
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_digest(formatter, &self.digest)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("digest", &format_args!("{}", self))
                    .field("source", &self.source)
                    .finish()
            }
        }

        impl FromStr for $name {
            type Err = IdentityParseError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Ok(Self::new(
                    parse_fixed_hex::<16>(value)?,
                    IdentitySource::Explicit,
                ))
            }
        }
    };
}

digest_identity!(CacheSemanticsId);
digest_identity!(RoutingScopeId);
digest_identity!(StableDpSlotId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexerDomainId {
    cache_semantics: CacheSemanticsId,
    routing_scope: RoutingScopeId,
}

impl IndexerDomainId {
    pub const fn new(cache_semantics: CacheSemanticsId, routing_scope: RoutingScopeId) -> Self {
        Self {
            cache_semantics,
            routing_scope,
        }
    }

    pub const fn cache_semantics(self) -> CacheSemanticsId {
        self.cache_semantics
    }

    pub const fn routing_scope(self) -> RoutingScopeId {
        self.routing_scope
    }

    pub const fn relies_on_defaults(self) -> bool {
        matches!(
            self.cache_semantics.source(),
            IdentitySource::DefaultDerived
        ) || matches!(self.routing_scope.source(), IdentitySource::DefaultDerived)
    }
}

impl fmt::Display for IndexerDomainId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.cache_semantics, self.routing_scope)
    }
}

impl FromStr for IndexerDomainId {
    type Err = IdentityParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (cache_semantics, routing_scope) = value
            .split_once(':')
            .ok_or(IdentityParseError::InvalidIndexerDomain)?;
        if routing_scope.contains(':') {
            return Err(IdentityParseError::InvalidIndexerDomain);
        }
        Ok(Self::new(cache_semantics.parse()?, routing_scope.parse()?))
    }
}

/// Control-plane-stable identity for one discovery cluster inside a routing federation.
///
/// NOTE: This value survives process restarts, scaling, endpoint replacement, and producer
/// generations. It is meaningful only as the cluster dimension of [`PoolId`], not as a globally
/// unique identifier. Do not derive it from endpoint identity or fold it into routing scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClusterId(u64);

impl ClusterId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ClusterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}", self.0)
    }
}

impl FromStr for ClusterId {
    type Err = IdentityParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(u64::from_be_bytes(parse_fixed_hex::<8>(value)?)))
    }
}

/// One cluster-local producer/publication stream within an indexer domain.
///
/// NOTE: A global indexer has one domain and distinct pool lanes whose `cluster_id` values differ.
/// Runtime endpoint resolution happens only after a query selects a pool lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PoolId {
    indexer_domain: IndexerDomainId,
    cluster_id: ClusterId,
}

impl PoolId {
    pub const fn new(indexer_domain: IndexerDomainId, cluster_id: ClusterId) -> Self {
        Self {
            indexer_domain,
            cluster_id,
        }
    }

    pub const fn indexer_domain(self) -> IndexerDomainId {
        self.indexer_domain
    }

    pub const fn cluster_id(self) -> ClusterId {
        self.cluster_id
    }
}

impl fmt::Display for PoolId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.indexer_domain, self.cluster_id)
    }
}

impl FromStr for PoolId {
    type Err = IdentityParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (indexer_domain, cluster_id) = value
            .split_once('/')
            .ok_or(IdentityParseError::InvalidPool)?;
        if cluster_id.contains('/') {
            return Err(IdentityParseError::InvalidPool);
        }
        Ok(Self::new(indexer_domain.parse()?, cluster_id.parse()?))
    }
}

/// Stable identity for one durable data-parallel slot inside a cache pool.
///
/// The digest is resolved from explicit configuration. It must not be derived
/// from a worker ID, DRT connection ID, rank ordering, or a transport endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CacheOwnerId {
    pool: PoolId,
    slot: StableDpSlotId,
}

impl CacheOwnerId {
    pub const fn new(pool: PoolId, slot: StableDpSlotId) -> Self {
        Self { pool, slot }
    }

    pub const fn pool(self) -> PoolId {
        self.pool
    }

    pub const fn slot(self) -> StableDpSlotId {
        self.slot
    }
}

impl fmt::Display for CacheOwnerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.pool, self.slot)
    }
}

impl FromStr for CacheOwnerId {
    type Err = IdentityParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (pool, slot) = value
            .rsplit_once('/')
            .ok_or(IdentityParseError::InvalidCacheOwner)?;
        if slot.contains('/') {
            return Err(IdentityParseError::InvalidCacheOwner);
        }
        Ok(Self::new(pool.parse()?, slot.parse()?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IdentityParseError {
    #[error("identity must use lowercase hexadecimal with the exact encoded width")]
    InvalidHex,
    #[error("indexer domain must be `<cache-semantics-id>:<routing-scope-id>`")]
    InvalidIndexerDomain,
    #[error("pool ID must be `<indexer-domain-id>/<cluster-id>`")]
    InvalidPool,
    #[error("cache owner ID must be `<pool-id>/<stable-dp-slot-id>`")]
    InvalidCacheOwner,
}

fn parse_fixed_hex<const N: usize>(value: &str) -> Result<[u8; N], IdentityParseError> {
    if value.len() != N * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(IdentityParseError::InvalidHex);
    }

    let mut result = [0; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]);
        let low = hex_nibble(pair[1]);
        result[index] = high << 4 | low;
    }
    Ok(result)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("parse_fixed_hex validates every nibble"),
    }
}

fn write_digest(formatter: &mut fmt::Formatter<'_>, digest: &[u8; 16]) -> fmt::Result {
    for byte in digest {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}
