// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};

use dynamo_kv_router::identity::{
    CacheSemanticsId, CanonicalIdentityMaterial, DcId, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_runtime::protocols::EndpointId;

use crate::model_card::ModelDeploymentCard;

#[derive(Debug, Clone)]
pub(crate) struct ResolvedIndexerDomain {
    pub(crate) id: IndexerDomainId,
    pub(crate) diagnostic_model_artifact: String,
    pub(crate) kv_block_size: u32,
    pub(crate) event_hash_format: u16,
}

impl PartialEq for ResolvedIndexerDomain {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.kv_block_size == other.kv_block_size
            && self.event_hash_format == other.event_hash_format
    }
}

impl Eq for ResolvedIndexerDomain {}

impl Hash for ResolvedIndexerDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.kv_block_size.hash(state);
        self.event_hash_format.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct EndpointLocator {
    dc_id: DcId,
    endpoint_id: EndpointId,
}

impl EndpointLocator {
    pub(crate) fn new(dc_id: DcId, endpoint_id: EndpointId) -> Self {
        Self { dc_id, endpoint_id }
    }

    pub(crate) fn endpoint_id(&self) -> &EndpointId {
        &self.endpoint_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PoolBinding {
    pool_id: PoolId,
    serving_endpoint: EndpointLocator,
    // Retained for the runtime's serving-to-KV-state resolution boundary; Relay currently keeps
    // the discovery binding separately while supervising the actor.
    #[allow(dead_code)]
    kv_state_endpoint: Option<EndpointLocator>,
}

impl PoolBinding {
    pub(crate) fn new(
        pool_id: PoolId,
        serving_endpoint: EndpointLocator,
        kv_state_endpoint: Option<EndpointLocator>,
    ) -> Self {
        debug_assert_eq!(serving_endpoint.dc_id, pool_id.dc_id());
        debug_assert!(
            kv_state_endpoint
                .as_ref()
                .is_none_or(|endpoint| endpoint.dc_id == pool_id.dc_id())
        );
        Self {
            pool_id,
            serving_endpoint,
            kv_state_endpoint,
        }
    }

    pub(crate) const fn pool_id(&self) -> PoolId {
        self.pool_id
    }

    pub(crate) const fn serving_endpoint(&self) -> &EndpointLocator {
        &self.serving_endpoint
    }
}

pub(crate) fn resolve_indexer_domain(
    card: &ModelDeploymentCard,
    serving_endpoint: &EndpointId,
    event_hash_format: u16,
) -> ResolvedIndexerDomain {
    let spec = card.indexer_identity.as_ref();
    let semantic_material = CanonicalIdentityMaterial::cache_semantics(
        &[card.source_path()],
        spec.and_then(|spec| spec.semantics()),
        card.kv_cache_block_size,
        event_hash_format,
    );
    let routing_material = CanonicalIdentityMaterial::routing_scope(
        &[
            serving_endpoint.namespace.as_str(),
            serving_endpoint.component.as_str(),
            serving_endpoint.name.as_str(),
        ],
        spec.and_then(|spec| spec.routing_scope()),
    );
    let cache_semantics = CacheSemanticsId::new(
        digest16(semantic_material.bytes()),
        semantic_material.source(),
    );
    let routing_scope = RoutingScopeId::new(
        digest16(routing_material.bytes()),
        routing_material.source(),
    );
    ResolvedIndexerDomain {
        id: IndexerDomainId::new(cache_semantics, routing_scope),
        diagnostic_model_artifact: card.source_path().to_string(),
        kv_block_size: card.kv_cache_block_size,
        event_hash_format,
    }
}

pub(crate) fn stable_dc_id(value: &str) -> DcId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dynamo/indexer-dc/v1");
    hasher.update(&(value.len() as u32).to_le_bytes());
    hasher.update(value.as_bytes());
    let hash = hasher.finalize();
    DcId::new(u64::from_le_bytes(
        hash.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 output is 32 bytes"),
    ))
}

fn digest16(bytes: &[u8]) -> [u8; 16] {
    blake3::hash(bytes).as_bytes()[..16]
        .try_into()
        .expect("BLAKE3 output is 32 bytes")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dynamo_kv_router::identity::{ExplicitIdentityMap, IdentitySource, IndexerIdentitySpec};

    use super::*;

    fn card(name: &str, source_path: &str) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(name);
        card.source_path = Some(source_path.to_string());
        card.kv_cache_block_size = 512;
        card
    }

    #[test]
    fn explicit_dimensions_replace_different_defaults() {
        let explicit = ExplicitIdentityMap::new(BTreeMap::from([(
            "authority".to_string(),
            "shared".to_string(),
        )]))
        .unwrap();
        let spec = IndexerIdentitySpec::new(Some(explicit.clone()), Some(explicit));
        let endpoint_a = EndpointId::from("dc-a/router/generate-a");
        let endpoint_b = EndpointId::from("dc-b/router/generate-b");
        let mut a = card("a", "repo/a");
        a.indexer_identity = Some(spec.clone());
        let mut b = card("b", "repo/b");
        b.indexer_identity = Some(spec);

        let a = resolve_indexer_domain(&a, &endpoint_a, 1);
        let b = resolve_indexer_domain(&b, &endpoint_b, 1);
        assert_eq!(a.id, b.id);
        let pool_a = PoolId::new(a.id, stable_dc_id("dc-a"));
        let pool_b = PoolId::new(b.id, stable_dc_id("dc-b"));
        assert_ne!(pool_a, pool_b);
        assert_eq!(a.id.cache_semantics().source(), IdentitySource::Explicit);
        assert_eq!(a.id.routing_scope().source(), IdentitySource::Explicit);
    }

    #[test]
    fn default_and_explicit_sources_never_alias() {
        let endpoint = EndpointId::from("ns/router/generate");
        let default = resolve_indexer_domain(&card("model", "same"), &endpoint, 1);
        let explicit =
            ExplicitIdentityMap::new(BTreeMap::from([("model".to_string(), "same".to_string())]))
                .unwrap();
        let mut explicit_card = card("model", "same");
        explicit_card.indexer_identity = Some(IndexerIdentitySpec::new(Some(explicit), None));
        let explicit = resolve_indexer_domain(&explicit_card, &endpoint, 1);
        assert_ne!(default.id, explicit.id);
    }

    #[test]
    fn mandatory_semantics_cannot_be_replaced_by_explicit_material() {
        let explicit = ExplicitIdentityMap::new(BTreeMap::from([(
            "weights".to_string(),
            "revision-a".to_string(),
        )]))
        .unwrap();
        let endpoint = EndpointId::from("ns/router/generate");
        let mut first = card("model", "ignored-a");
        first.indexer_identity = Some(IndexerIdentitySpec::new(Some(explicit.clone()), None));
        let mut second = card("model", "ignored-b");
        second.kv_cache_block_size = 1024;
        second.indexer_identity = Some(IndexerIdentitySpec::new(Some(explicit), None));

        let first = resolve_indexer_domain(&first, &endpoint, 1);
        let second = resolve_indexer_domain(&second, &endpoint, 1);

        assert_ne!(first.id.cache_semantics(), second.id.cache_semantics());
    }

    #[test]
    fn relay_derivation_has_frozen_golden_vectors() {
        let endpoint = EndpointId::from("prod/router/generate");
        let resolved = resolve_indexer_domain(&card("display", "meta/llama"), &endpoint, 1);
        assert_eq!(
            resolved.id.cache_semantics().to_string(),
            "7d31eb9019357572470605f4a8be687e"
        );
        assert_eq!(
            resolved.id.routing_scope().to_string(),
            "18270d3ba03effaec8d167ba02c7752d"
        );
    }
}
