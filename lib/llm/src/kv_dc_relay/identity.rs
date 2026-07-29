// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::identity::{IdentitySource, PoolId};
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_runtime::protocols::EndpointId;
use serde::{Deserialize, Deserializer, Serialize};

fn validate_identity_text<E>(
    value: impl Into<String>,
    empty: E,
    surrounding_whitespace: E,
) -> Result<String, E> {
    let value = value.into();
    if value.is_empty() {
        return Err(empty);
    }
    if value.trim() != value {
        return Err(surrounding_whitespace);
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CanonicalModelId(String);

impl CanonicalModelId {
    pub fn new(value: impl Into<String>) -> Result<Self, CanonicalModelIdError> {
        validate_identity_text(
            value,
            CanonicalModelIdError::Empty,
            CanonicalModelIdError::SurroundingWhitespace,
        )
        .map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CanonicalModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for CanonicalModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CanonicalModelIdError {
    #[error("canonical model ID must not be empty")]
    Empty,
    #[error("canonical model ID must not contain leading or trailing whitespace")]
    SurroundingWhitespace,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ModelAlias(String);

impl ModelAlias {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelAliasError> {
        validate_identity_text(
            value,
            ModelAliasError::Empty,
            ModelAliasError::SurroundingWhitespace,
        )
        .map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelAlias {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ModelAliasError {
    #[error("model alias must not be empty")]
    Empty,
    #[error("model alias must not contain leading or trailing whitespace")]
    SurroundingWhitespace,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct CanonicalModelRegistration {
    model: CanonicalModelId,
    target: ModelTarget,
    aliases: Vec<ModelAlias>,
}

impl CanonicalModelRegistration {
    pub fn new(model: CanonicalModelId, aliases: Vec<ModelAlias>) -> Self {
        let target = ModelTarget::Base {
            base_model: model.clone(),
        };
        Self::with_target(model, target, aliases)
    }

    pub fn with_target(
        model: CanonicalModelId,
        target: ModelTarget,
        mut aliases: Vec<ModelAlias>,
    ) -> Self {
        aliases.retain(|alias| alias.as_str() != model.as_str());
        aliases.sort_unstable();
        aliases.dedup();
        Self {
            model,
            target,
            aliases,
        }
    }

    pub const fn model(&self) -> &CanonicalModelId {
        &self.model
    }

    pub const fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn aliases(&self) -> &[ModelAlias] {
        &self.aliases
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModelTarget {
    Base {
        base_model: CanonicalModelId,
    },
    Lora {
        base_model: CanonicalModelId,
        adapter: CanonicalModelId,
    },
}

impl ModelTarget {
    pub const fn base_model(&self) -> &CanonicalModelId {
        match self {
            Self::Base { base_model } | Self::Lora { base_model, .. } => base_model,
        }
    }

    pub const fn adapter(&self) -> Option<&CanonicalModelId> {
        match self {
            Self::Base { .. } => None,
            Self::Lora { adapter, .. } => Some(adapter),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DcPoolDescriptor {
    producer: ProducerIdentity,
    serving_endpoint: EndpointId,
    registrations: Arc<[CanonicalModelRegistration]>,
}

impl DcPoolDescriptor {
    pub(crate) fn new(
        producer: ProducerIdentity,
        serving_endpoint: EndpointId,
        registrations: Arc<[CanonicalModelRegistration]>,
    ) -> Self {
        Self {
            producer,
            serving_endpoint,
            registrations,
        }
    }

    pub const fn producer(&self) -> ProducerIdentity {
        self.producer
    }

    pub const fn pool_id(&self) -> PoolId {
        self.producer.pool_id()
    }

    pub const fn serving_endpoint(&self) -> &EndpointId {
        &self.serving_endpoint
    }

    pub fn registrations(&self) -> &[CanonicalModelRegistration] {
        &self.registrations
    }
}

/// Identity of one DC Relay runtime.
///
/// `drt_instance_id` identifies the backing Dynamo runtime and can remain stable across an
/// in-process Relay restart. `relay_incarnation` is generated for every [`KvDcRelay::start`]
/// and fences producer generations created by different Relay lifetimes.
///
/// [`KvDcRelay::start`]: super::KvDcRelay::start
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct DcRelayIdentity {
    drt_instance_id: u64,
    relay_incarnation: u64,
}

impl DcRelayIdentity {
    pub const fn new(drt_instance_id: u64, relay_incarnation: u64) -> Self {
        Self {
            drt_instance_id,
            relay_incarnation,
        }
    }

    pub const fn drt_instance_id(self) -> u64 {
        self.drt_instance_id
    }

    pub const fn relay_incarnation(self) -> u64 {
        self.relay_incarnation
    }
}

#[derive(Debug)]
struct DcPoolCatalogPools {
    by_id: BTreeMap<PoolId, DcPoolDescriptor>,
    ordered: OnceLock<Vec<DcPoolDescriptor>>,
}

impl Clone for DcPoolCatalogPools {
    fn clone(&self) -> Self {
        Self {
            by_id: self.by_id.clone(),
            ordered: OnceLock::new(),
        }
    }
}

#[derive(Clone)]
pub struct DcPoolCatalog {
    identity: DcRelayIdentity,
    revision: u64,
    pools: DcPoolCatalogPools,
}

impl DcPoolCatalog {
    pub(crate) fn new(
        identity: DcRelayIdentity,
        revision: u64,
        pools: Vec<DcPoolDescriptor>,
    ) -> Self {
        let by_id = pools
            .into_iter()
            .map(|descriptor| (descriptor.pool_id(), descriptor))
            .collect();
        Self {
            identity,
            revision,
            pools: DcPoolCatalogPools {
                by_id,
                ordered: OnceLock::new(),
            },
        }
    }

    pub(crate) fn upsert(&mut self, revision: u64, descriptor: DcPoolDescriptor) {
        self.revision = revision;
        self.pools.by_id.insert(descriptor.pool_id(), descriptor);
        self.pools.ordered.take();
    }

    pub(crate) fn remove(&mut self, revision: u64, pool_id: PoolId) {
        self.revision = revision;
        self.pools.by_id.remove(&pool_id);
        self.pools.ordered.take();
    }

    pub(crate) fn clear(&mut self, revision: u64) {
        self.revision = revision;
        self.pools.by_id.clear();
        self.pools.ordered.take();
    }

    pub const fn identity(&self) -> DcRelayIdentity {
        self.identity
    }

    pub const fn drt_instance_id(&self) -> u64 {
        self.identity.drt_instance_id()
    }

    pub const fn relay_incarnation(&self) -> u64 {
        self.identity.relay_incarnation()
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub fn pools(&self) -> &[DcPoolDescriptor] {
        self.pools
            .ordered
            .get_or_init(|| self.pools.by_id.values().cloned().collect())
    }

    #[cfg(test)]
    pub(crate) fn is_materialized(&self) -> bool {
        self.pools.ordered.get().is_some()
    }
}

impl fmt::Debug for DcPoolCatalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DcPoolCatalog")
            .field("identity", &self.identity)
            .field("revision", &self.revision)
            .field("pools", &self.pools())
            .finish()
    }
}

impl PartialEq for DcPoolCatalog {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
            && self.revision == other.revision
            && self.pools.by_id == other.pools.by_id
    }
}

impl Eq for DcPoolCatalog {}

impl Serialize for DcPoolCatalog {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Catalog<'a> {
            drt_instance_id: u64,
            relay_incarnation: u64,
            revision: u64,
            pools: &'a [DcPoolDescriptor],
        }

        Catalog {
            drt_instance_id: self.identity.drt_instance_id(),
            relay_incarnation: self.identity.relay_incarnation(),
            revision: self.revision,
            pools: self.pools(),
        }
        .serialize(serializer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolIdentitySources {
    cache_semantics: IdentitySource,
    routing_scope: IdentitySource,
}

impl PoolIdentitySources {
    pub const fn from_pool(pool_id: PoolId) -> Self {
        Self {
            cache_semantics: pool_id.indexer_domain().cache_semantics().source(),
            routing_scope: pool_id.indexer_domain().routing_scope().source(),
        }
    }

    pub const fn cache_semantics(self) -> IdentitySource {
        self.cache_semantics
    }

    pub const fn routing_scope(self) -> IdentitySource {
        self.routing_scope
    }

    pub const fn is_derived(self) -> bool {
        matches!(self.cache_semantics, IdentitySource::DefaultDerived)
            || matches!(self.routing_scope, IdentitySource::DefaultDerived)
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{CacheSemanticsId, DcId, IndexerDomainId, RoutingScopeId};

    use super::*;

    fn pool(cache_source: IdentitySource, routing_source: IdentitySource) -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], cache_source),
                RoutingScopeId::new([2; 16], routing_source),
            ),
            DcId::new(3),
        )
    }

    #[test]
    fn canonical_model_id_rejects_ambiguous_text() {
        assert_eq!(CanonicalModelId::new(""), Err(CanonicalModelIdError::Empty));
        assert_eq!(
            CanonicalModelId::new(" llama"),
            Err(CanonicalModelIdError::SurroundingWhitespace)
        );
    }

    #[test]
    fn canonical_registration_normalizes_aliases_without_creating_self_alias() {
        let model = CanonicalModelId::new("llama").unwrap();
        let registration = CanonicalModelRegistration::new(
            model.clone(),
            vec![
                ModelAlias::new("chat").unwrap(),
                ModelAlias::new("llama").unwrap(),
                ModelAlias::new("chat").unwrap(),
            ],
        );

        assert_eq!(registration.model(), &model);
        assert_eq!(registration.aliases(), &[ModelAlias::new("chat").unwrap()]);
    }

    #[test]
    fn pool_identity_sources_report_derived_components() {
        let explicit = PoolIdentitySources::from_pool(pool(
            IdentitySource::Explicit,
            IdentitySource::Explicit,
        ));
        assert_eq!(explicit.cache_semantics(), IdentitySource::Explicit);
        assert_eq!(explicit.routing_scope(), IdentitySource::Explicit);
        assert!(!explicit.is_derived());

        let derived = PoolIdentitySources::from_pool(pool(
            IdentitySource::Explicit,
            IdentitySource::DefaultDerived,
        ));
        assert_eq!(derived.cache_semantics(), IdentitySource::Explicit);
        assert_eq!(derived.routing_scope(), IdentitySource::DefaultDerived);
        assert!(derived.is_derived());
    }
}
