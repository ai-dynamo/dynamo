// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arc_swap::{ArcSwap, Guard};
use dynamo_kv_router::identity::{
    CacheSemanticsId, CanonicalIdentityMaterial, DcId, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_kv_router::indexer::cuckoo::{CKF_LANE_COUNT, GlobalCkfIndexer};
use dynamo_runtime::protocols::EndpointId;

use super::identity::{KvQueryHashFormat, KvQuerySemantics, KvQuerySemanticsError};
use crate::model_card::ModelDeploymentCard;

#[derive(Debug, Clone)]
pub(crate) struct ResolvedIndexerDomain {
    pub(crate) id: IndexerDomainId,
    // Test fixtures construct this field, but only the diagnostics feature reads it.
    #[cfg(any(test, feature = "ckf-diagnostics"))]
    #[cfg_attr(all(test, not(feature = "ckf-diagnostics")), allow(dead_code))]
    pub(crate) diagnostic_model_artifact: String,
    pub(crate) query_semantics: KvQuerySemantics,
}

impl PartialEq for ResolvedIndexerDomain {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.query_semantics == other.query_semantics
    }
}

impl Eq for ResolvedIndexerDomain {}

impl Hash for ResolvedIndexerDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.query_semantics.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct EndpointLocator {
    dc_id: DcId,
    endpoint_id: EndpointId,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum PublishedIndexerBundleError {
    #[error("global CKF lane {lane} for pool {pool_id} has no runtime binding")]
    MissingBinding { lane: usize, pool_id: PoolId },

    #[error("runtime binding for pool {pool_id} targets unconfigured global CKF lane {lane}")]
    UnexpectedBinding { lane: usize, pool_id: PoolId },

    #[error(
        "global CKF lane {lane} has pool {expected}, but its runtime binding has pool {actual}"
    )]
    WrongPool {
        lane: usize,
        expected: PoolId,
        actual: PoolId,
    },
}

/// One immutable query generation: CKF storage and its lane-to-runtime resolution.
///
/// Keeping both in one publication unit prevents a replacement from exposing a new consumer with
/// old endpoint bindings, or vice versa. Physical lanes are fixed, so direct array indexing keeps
/// endpoint resolution off hash maps on the query path.
#[derive(Debug)]
pub(crate) struct PublishedIndexerBundle {
    consumer: Arc<GlobalCkfIndexer>,
    lane_bindings: [Option<PoolBinding>; CKF_LANE_COUNT],
}

#[allow(dead_code)]
impl PublishedIndexerBundle {
    pub(crate) fn new(
        consumer: Arc<GlobalCkfIndexer>,
        lane_bindings: [Option<PoolBinding>; CKF_LANE_COUNT],
    ) -> Result<Self, PublishedIndexerBundleError> {
        let manifest = consumer.manifest();
        for (lane, binding) in lane_bindings.iter().enumerate() {
            match (manifest.pool_id(lane), binding) {
                (Some(expected), None) => {
                    return Err(PublishedIndexerBundleError::MissingBinding {
                        lane,
                        pool_id: expected,
                    });
                }
                (None, Some(binding)) => {
                    return Err(PublishedIndexerBundleError::UnexpectedBinding {
                        lane,
                        pool_id: binding.pool_id(),
                    });
                }
                (Some(expected), Some(binding)) if expected != binding.pool_id() => {
                    return Err(PublishedIndexerBundleError::WrongPool {
                        lane,
                        expected,
                        actual: binding.pool_id(),
                    });
                }
                _ => {}
            }
        }

        Ok(Self {
            consumer,
            lane_bindings,
        })
    }

    pub(crate) fn consumer(&self) -> &GlobalCkfIndexer {
        &self.consumer
    }

    pub(crate) fn binding(&self, lane: usize) -> Option<&PoolBinding> {
        self.lane_bindings.get(lane).and_then(Option::as_ref)
    }
}

/// Atomically publishes a complete consumer generation with its matching runtime bindings.
///
/// Query callers should use `load`, not `load_full`: the ArcSwap guard captures one coherent
/// generation without an unconditional `Arc` refcount operation. A query holding an old guard may
/// finish against the old consumer and old bindings after replacement.
#[allow(dead_code)]
pub(crate) struct PublishedGlobalCkfIndexer {
    active: ArcSwap<PublishedIndexerBundle>,
}

#[allow(dead_code)]
impl PublishedGlobalCkfIndexer {
    pub(crate) fn new(initial: PublishedIndexerBundle) -> Self {
        Self {
            active: ArcSwap::from_pointee(initial),
        }
    }

    pub(crate) fn load(&self) -> Guard<Arc<PublishedIndexerBundle>> {
        self.active.load()
    }

    pub(crate) fn replace(&self, next: PublishedIndexerBundle) -> Arc<PublishedIndexerBundle> {
        self.active.swap(Arc::new(next))
    }
}

pub(crate) fn resolve_indexer_domain(
    card: &ModelDeploymentCard,
    serving_endpoint: &EndpointId,
) -> Result<ResolvedIndexerDomain, KvQuerySemanticsError> {
    let hash_format = KvQueryHashFormat::from_enable_eagle(card.runtime_config.enable_eagle);
    let query_semantics = KvQuerySemantics::new(card.kv_cache_block_size, hash_format)?;
    let spec = card.indexer_identity.as_ref();
    let semantic_material = CanonicalIdentityMaterial::cache_semantics(
        &[card.source_path()],
        spec.and_then(|spec| spec.semantics()),
        query_semantics.kv_block_size(),
        query_semantics.hash_format().identity_version(),
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
    Ok(ResolvedIndexerDomain {
        id: IndexerDomainId::new(cache_semantics, routing_scope),
        #[cfg(any(test, feature = "ckf-diagnostics"))]
        diagnostic_model_artifact: card.source_path().to_string(),
        query_semantics,
    })
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
    use dynamo_kv_router::indexer::cuckoo::{
        CkfConfig, ConsumerInstanceId, DcCkfState, GlobalCkfIngestOutcome, GlobalCkfManifest,
        GlobalCkfSnapshot, LaneLease, PrefixSearchConfig, ProducerIdentity,
    };
    use dynamo_kv_router::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, compute_block_hash_for_seq,
        compute_seq_hash_for_block,
    };

    use super::*;

    fn card(name: &str, source_path: &str) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(name);
        card.source_path = Some(source_path.to_string());
        card.kv_cache_block_size = 512;
        card
    }

    fn domain(seed: u8) -> IndexerDomainId {
        IndexerDomainId::new(
            CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
            RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
        )
    }

    fn published_bundle(
        domain: IndexerDomainId,
        dc: u64,
        endpoint: &str,
        consumer_instance: u64,
    ) -> PublishedIndexerBundle {
        let state = DcCkfState::new(CkfConfig::new(16)).unwrap();
        let pool_id = PoolId::new(domain, DcId::new(dc));
        let mut lanes = [None; CKF_LANE_COUNT];
        lanes[0] = Some(pool_id);
        let manifest = GlobalCkfManifest::new(
            ConsumerInstanceId::new(consumer_instance),
            domain,
            state.format(),
            lanes,
        )
        .unwrap();
        let consumer =
            Arc::new(GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default()).unwrap());
        let mut bindings = std::array::from_fn(|_| None);
        bindings[0] = Some(PoolBinding::new(
            pool_id,
            EndpointLocator::new(DcId::new(dc), EndpointId::from(endpoint)),
            None,
        ));
        PublishedIndexerBundle::new(consumer, bindings).unwrap()
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

        let a = resolve_indexer_domain(&a, &endpoint_a).unwrap();
        let b = resolve_indexer_domain(&b, &endpoint_b).unwrap();
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
        let default = resolve_indexer_domain(&card("model", "same"), &endpoint).unwrap();
        let explicit =
            ExplicitIdentityMap::new(BTreeMap::from([("model".to_string(), "same".to_string())]))
                .unwrap();
        let mut explicit_card = card("model", "same");
        explicit_card.indexer_identity = Some(IndexerIdentitySpec::new(Some(explicit), None));
        let explicit = resolve_indexer_domain(&explicit_card, &endpoint).unwrap();
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

        let first = resolve_indexer_domain(&first, &endpoint).unwrap();
        let second = resolve_indexer_domain(&second, &endpoint).unwrap();

        assert_ne!(first.id.cache_semantics(), second.id.cache_semantics());
    }

    #[test]
    fn relay_derivation_has_frozen_golden_vectors() {
        let endpoint = EndpointId::from("prod/router/generate");
        let resolved = resolve_indexer_domain(&card("display", "meta/llama"), &endpoint).unwrap();
        assert_eq!(
            resolved.id.cache_semantics().to_string(),
            "7d31eb9019357572470605f4a8be687e"
        );
        assert_eq!(
            resolved.id.routing_scope().to_string(),
            "18270d3ba03effaec8d167ba02c7752d"
        );
    }

    #[test]
    fn eagle_is_a_distinct_cache_semantics_pipeline() {
        let endpoint = EndpointId::from("prod/router/generate");
        let standard = resolve_indexer_domain(&card("llama", "meta/llama"), &endpoint).unwrap();
        let mut eagle_card = card("llama", "meta/llama");
        eagle_card.runtime_config.enable_eagle = true;
        let eagle = resolve_indexer_domain(&eagle_card, &endpoint).unwrap();

        assert_eq!(
            standard.query_semantics.hash_format(),
            KvQueryHashFormat::DynamoStandardV1
        );
        assert_eq!(
            eagle.query_semantics.hash_format(),
            KvQueryHashFormat::DynamoEagleV1
        );
        assert_ne!(standard.id.cache_semantics(), eagle.id.cache_semantics());
    }

    #[test]
    fn zero_block_size_cannot_form_a_queryable_domain() {
        let endpoint = EndpointId::from("prod/router/generate");
        let mut invalid = card("llama", "meta/llama");
        invalid.kv_cache_block_size = 0;

        assert_eq!(
            resolve_indexer_domain(&invalid, &endpoint).unwrap_err(),
            KvQuerySemanticsError::ZeroBlockSize
        );
    }

    fn assert_query_pipeline_parity(hash_format: KvQueryHashFormat, lora_name: Option<&str>) {
        let block_size = 4;
        let tokens: Vec<u32> = (0..13).collect();
        let hash_options = BlockHashOptions {
            lora_name,
            cache_namespace: Some("tenant-ns"),
            is_eagle: Some(hash_format.is_eagle()),
            ..Default::default()
        };
        let local_hashes = compute_block_hash_for_seq(&tokens, block_size, hash_options);
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        assert!(!local_hashes.is_empty());

        let endpoint = EndpointId::from("prod/router/generate");
        let mut deployment = card("llama", "meta/llama");
        deployment.kv_cache_block_size = block_size;
        deployment.runtime_config.enable_eagle = hash_format.is_eagle();
        let resolved = resolve_indexer_domain(&deployment, &endpoint).unwrap();
        assert_eq!(resolved.query_semantics.hash_format(), hash_format);

        let mut producer = DcCkfState::new(CkfConfig::new(64)).unwrap();
        producer.apply_event(RouterEvent::new(
            1,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: local_hashes
                        .iter()
                        .zip(&sequence_hashes)
                        .map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(
                                *sequence_hash ^ 0xE771_6E00_5A17_CAFE,
                            ),
                            tokens_hash: *local_hash,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        ));
        let (_, buckets) = producer.barrier_snapshot().unwrap();

        let pool_id = PoolId::new(resolved.id, DcId::new(7));
        let identity = ProducerIdentity::new(pool_id, 11, 1, producer.format());
        let consumer_instance = ConsumerInstanceId::new(13);
        let mut lanes = [None; CKF_LANE_COUNT];
        lanes[0] = Some(pool_id);
        let manifest =
            GlobalCkfManifest::new(consumer_instance, resolved.id, producer.format(), lanes)
                .unwrap();
        let consumer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default()).unwrap();
        let mut ingestor = consumer.claim_lane(0).unwrap();
        let lease = LaneLease::new(consumer_instance, 0, 1);
        ingestor.assign(identity, lease).unwrap();
        assert_eq!(
            ingestor.install_snapshot(&GlobalCkfSnapshot::new(identity, lease, 1, buckets)),
            GlobalCkfIngestOutcome::SnapshotInstalled { sequence: 1 }
        );

        let result = consumer.find_prefix_matches(&local_hashes).unwrap();
        let lane = result.lanes()[0].expect("pool lane must be queryable");
        assert_eq!(lane.pool_id(), pool_id);
        assert_eq!(lane.prefix_depth(), local_hashes.len() as u32);
    }

    #[test]
    fn declared_query_semantics_match_dc_snapshot_and_global_query() {
        assert_query_pipeline_parity(KvQueryHashFormat::DynamoStandardV1, None);
        assert_query_pipeline_parity(KvQueryHashFormat::DynamoStandardV1, Some("tenant-a"));
        assert_query_pipeline_parity(KvQueryHashFormat::DynamoEagleV1, Some("tenant-a"));
    }

    #[test]
    fn published_consumer_and_bindings_replace_as_one_generation() {
        let published =
            PublishedGlobalCkfIndexer::new(published_bundle(domain(1), 11, "ns/router/old", 101));
        let old_query = published.load();

        let retired = published.replace(published_bundle(domain(2), 22, "ns/router/new", 202));
        let new_query = published.load();

        assert_eq!(
            old_query.consumer().manifest().consumer_instance(),
            ConsumerInstanceId::new(101)
        );
        assert_eq!(
            old_query
                .binding(0)
                .unwrap()
                .serving_endpoint()
                .endpoint_id(),
            &EndpointId::from("ns/router/old")
        );
        assert_eq!(
            new_query.consumer().manifest().consumer_instance(),
            ConsumerInstanceId::new(202)
        );
        assert_eq!(
            new_query
                .binding(0)
                .unwrap()
                .serving_endpoint()
                .endpoint_id(),
            &EndpointId::from("ns/router/new")
        );
        assert!(Arc::ptr_eq(&retired, &*old_query));
    }

    #[test]
    fn published_bundle_rejects_binding_from_another_pool() {
        let bundle = published_bundle(domain(3), 33, "ns/router/expected", 303);
        let consumer = Arc::new(bundle.consumer().clone());
        let wrong_pool = PoolId::new(domain(3), DcId::new(44));
        let mut bindings = std::array::from_fn(|_| None);
        bindings[0] = Some(PoolBinding::new(
            wrong_pool,
            EndpointLocator::new(DcId::new(44), EndpointId::from("ns/router/wrong")),
            None,
        ));

        assert!(matches!(
            PublishedIndexerBundle::new(consumer, bindings),
            Err(PublishedIndexerBundleError::WrongPool {
                lane: 0,
                expected,
                actual,
            }) if expected == PoolId::new(domain(3), DcId::new(33)) && actual == wrong_pool
        ));
    }
}
