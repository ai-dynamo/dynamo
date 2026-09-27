// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary from the existing KV relay's validated snapshots into Global View.
//!
//! One configured DGD owns one routing pool. Its Dynamo runtime namespace is
//! used only to scope relay entries; the routing PoolId is derived separately
//! from site ID, Kubernetes namespace, and DGD name in dynamo-kv-router.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use dynamo_kv_router::global_view::source::PoolObservation;
use dynamo_kv_router::global_view::state::{PoolRole, ServingReadiness, SignalState, SignalStatus};
use thiserror::Error;

use crate::kv_dc_relay::wan::grpc::protocol::{
    KvPoolCatalogUpdate, KvPoolDescriptor, ProducerKey, ServingReadinessState,
    ServingReadinessUpdate, WireIdentityError, WorkerRole, validate_pool_descriptor,
    validate_protocol_envelope, validate_topology_entry,
};

/// Configured local frontend is distinct from a relay's worker KV endpoint.
pub struct RelayPoolScope {
    pub runtime_namespace: String,
    pub frontend_endpoint: String,
}

pub struct CatalogProjection {
    pub observation: PoolObservation,
    /// Validated wire descriptors retained by the transport adapter for exact
    /// producer subscriptions and per-producer query semantics.
    pub producers: Vec<KvPoolDescriptor>,
}

impl CatalogProjection {
    /// The first POC scores aggregated requests only when exactly one
    /// aggregated KV producer advertises this model. Other roles are retained
    /// in the catalog, but their cache state has no agreed DGD-level meaning
    /// for an aggregated request. Multiple matching producers are ambiguous
    /// until local worker selection can be aligned with the overlap estimate.
    pub fn sole_aggregated_overlap_producer(&self, model: &str) -> Option<&KvPoolDescriptor> {
        let mut matching = self.producers.iter().filter(|descriptor| {
            descriptor
                .pool_roles
                .contains(&(WorkerRole::Aggregated as i32))
                && descriptor
                    .registrations
                    .iter()
                    .any(|registration| registration.canonical_model_id == model)
        });
        let producer = matching.next()?;
        matching.next().is_none().then_some(producer)
    }
}

#[derive(Debug, Error)]
pub enum RelayProjectionError {
    #[error(transparent)]
    Wire(#[from] WireIdentityError),
    #[error("relay update is missing {0}")]
    MissingField(&'static str),
    #[error("configured runtime namespace or frontend endpoint is empty")]
    InvalidScope,
    #[error("duplicate readiness for model {0}")]
    DuplicateModel(String),
    #[error("duplicate KV producer in scoped catalog")]
    DuplicateProducer,
}

fn validate_scope(scope: &RelayPoolScope) -> Result<(), RelayProjectionError> {
    if scope.runtime_namespace.trim().is_empty() || scope.frontend_endpoint.trim().is_empty() {
        return Err(RelayProjectionError::InvalidScope);
    }
    Ok(())
}

fn status(received_at_unix_ms: u64) -> SignalStatus {
    SignalStatus {
        state: SignalState::Complete,
        received_at_unix_ms: Some(received_at_unix_ms),
        ..Default::default()
    }
}

/// Each relay catalog update is a complete snapshot; an empty scoped result
/// withdraws all advertised models and KV producers from this DGD.
pub fn project_catalog(
    update: &KvPoolCatalogUpdate,
    scope: &RelayPoolScope,
    received_at_unix_ms: u64,
) -> Result<CatalogProjection, RelayProjectionError> {
    validate_scope(scope)?;
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    update
        .relay
        .as_ref()
        .ok_or(RelayProjectionError::MissingField("relay identity"))?;
    let snapshot = update
        .snapshot
        .as_ref()
        .ok_or(RelayProjectionError::MissingField("catalog snapshot"))?;
    let mut models = BTreeSet::new();
    let mut roles = BTreeSet::new();
    let mut producers = Vec::new();
    let mut producer_keys = HashSet::new();
    for descriptor in &snapshot.pools {
        let endpoint = descriptor
            .serving_endpoint
            .as_ref()
            .ok_or(RelayProjectionError::MissingField("serving endpoint"))?;
        if endpoint.namespace != scope.runtime_namespace {
            continue;
        }
        validate_pool_descriptor(descriptor)?;
        let producer = descriptor
            .producer
            .as_ref()
            .ok_or(RelayProjectionError::MissingField("pool producer"))?;
        if !producer_keys.insert(ProducerKey::try_from(producer)?) {
            return Err(RelayProjectionError::DuplicateProducer);
        }
        for registration in &descriptor.registrations {
            models.insert(registration.canonical_model_id.clone());
        }
        for value in &descriptor.pool_roles {
            let role =
                WorkerRole::try_from(*value).map_err(|_| WireIdentityError::WorkerRole(*value))?;
            roles.insert(match role {
                WorkerRole::Aggregated => PoolRole::Aggregated,
                WorkerRole::Prefill => PoolRole::Prefill,
                WorkerRole::Decode => PoolRole::Decode,
                WorkerRole::Encode => PoolRole::Encode,
                WorkerRole::Legacy => PoolRole::Legacy,
                WorkerRole::Unspecified => return Err(WireIdentityError::WorkerRole(*value).into()),
            });
        }
        producers.push(descriptor.clone());
    }
    Ok(CatalogProjection {
        observation: PoolObservation::Catalog {
            models: models.into_iter().collect(),
            roles: roles.into_iter().collect(),
            frontend_endpoint: Some(scope.frontend_endpoint.clone()),
            hardware: Vec::new(),
            status: status(received_at_unix_ms),
        },
        producers,
    })
}

/// Readiness is a separate complete namespace snapshot. Adapter readiness for
/// LoRA targets is omitted from the text-only base-model POC.
pub fn project_readiness(
    update: &ServingReadinessUpdate,
    scope: &RelayPoolScope,
    received_at_unix_ms: u64,
) -> Result<PoolObservation, RelayProjectionError> {
    validate_scope(scope)?;
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    update
        .relay
        .as_ref()
        .ok_or(RelayProjectionError::MissingField("relay identity"))?;
    let mut models = BTreeMap::new();
    for entry in &update.entries {
        if entry.namespace != scope.runtime_namespace {
            continue;
        }
        validate_topology_entry(entry)?;
        let state = match ServingReadinessState::try_from(entry.state)
            .map_err(|_| WireIdentityError::ReadinessState(entry.state))?
        {
            ServingReadinessState::Ready => ServingReadiness::Ready,
            ServingReadinessState::Unavailable => ServingReadiness::Unavailable,
            ServingReadinessState::Unknown => ServingReadiness::Unknown,
        };
        if models
            .insert(entry.canonical_model_id.clone(), state)
            .is_some()
        {
            return Err(RelayProjectionError::DuplicateModel(
                entry.canonical_model_id.clone(),
            ));
        }
    }
    Ok(PoolObservation::Readiness {
        models,
        status: status(received_at_unix_ms),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_dc_relay::wan::grpc::protocol::v1::model_target;
    use crate::kv_dc_relay::wan::grpc::protocol::{
        BaseModelTarget, CkfFormat, DigestIdentity, DynamoEndpointId, IdentitySource,
        IndexerDomainId, KvPoolCatalogSnapshot, KvPoolId, KvQueryHashFormat, KvQuerySemantics,
        ModelRegistration, ModelTarget, ProducerIdentity, RELAY_CONTRACT_MARKER,
        RELAY_PROTOCOL_VERSION, RelayIdentity, TopologyEntry, TopologyMember,
    };
    use bytes::Bytes;

    fn scope() -> RelayPoolScope {
        RelayPoolScope {
            runtime_namespace: "dynamo-mocker".into(),
            frontend_endpoint: "mocker.frontend.generate".into(),
        }
    }

    fn descriptor(namespace: &str) -> KvPoolDescriptor {
        KvPoolDescriptor {
            producer: Some(ProducerIdentity {
                pool_id: Some(KvPoolId {
                    identity_version: 1,
                    indexer_domain: Some(IndexerDomainId {
                        cache_semantics: Some(DigestIdentity {
                            digest: Bytes::from_static(&[1; 16]),
                            source: IdentitySource::Explicit as i32,
                        }),
                        routing_scope: Some(DigestIdentity {
                            digest: Bytes::from_static(&[2; 16]),
                            source: IdentitySource::Explicit as i32,
                        }),
                    }),
                    dc_id: 7,
                }),
                producer_incarnation: 1,
                layout_generation: 1,
                ckf_format: Some(CkfFormat {
                    format_version: 1,
                    seed: 42,
                    bucket_count: 1024,
                    fingerprint_bits: 16,
                    slots_per_bucket: 4,
                }),
            }),
            serving_endpoint: Some(DynamoEndpointId {
                namespace: namespace.into(),
                component: "mocker".into(),
                endpoint: "generate".into(),
            }),
            registrations: vec![ModelRegistration {
                canonical_model_id: "model".into(),
                target: Some(ModelTarget {
                    target: Some(model_target::Target::Base(BaseModelTarget {
                        base_model: "model".into(),
                    })),
                }),
                aliases: Vec::new(),
            }],
            query_semantics: Some(KvQuerySemantics {
                kv_block_size: 16,
                hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
            }),
            pool_roles: vec![WorkerRole::Aggregated as i32],
        }
    }

    #[test]
    fn scopes_catalog_to_one_dgd_and_retains_producer_identity() {
        let update = KvPoolCatalogUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(RelayIdentity {
                drt_instance_id: 1,
                relay_incarnation: 2,
            }),
            snapshot: Some(KvPoolCatalogSnapshot {
                pools: vec![descriptor("other-dgd"), descriptor("dynamo-mocker")],
            }),
            contract_marker: RELAY_CONTRACT_MARKER,
            ..Default::default()
        };
        let projected = project_catalog(&update, &scope(), 123).unwrap();
        assert_eq!(projected.producers.len(), 1);
        match projected.observation {
            PoolObservation::Catalog {
                models,
                roles,
                frontend_endpoint,
                status,
                ..
            } => {
                assert_eq!(models, vec!["model"]);
                assert_eq!(roles, vec![PoolRole::Aggregated]);
                assert_eq!(
                    frontend_endpoint.as_deref(),
                    Some("mocker.frontend.generate")
                );
                assert_eq!(status.received_at_unix_ms, Some(123));
            }
            _ => panic!("expected catalog"),
        }
    }

    #[test]
    fn aggregated_overlap_requires_one_matching_producer() {
        let aggregated = descriptor("dynamo-mocker");
        let mut decode = descriptor("dynamo-mocker");
        decode.pool_roles = vec![WorkerRole::Decode as i32];
        let mut second_aggregated = descriptor("dynamo-mocker");
        second_aggregated
            .serving_endpoint
            .as_mut()
            .unwrap()
            .component = "other".into();

        let mut projection = CatalogProjection {
            observation: PoolObservation::Catalog {
                models: vec!["model".into()],
                roles: vec![PoolRole::Aggregated, PoolRole::Decode],
                frontend_endpoint: None,
                hardware: Vec::new(),
                status: status(123),
            },
            producers: vec![aggregated, decode],
        };
        assert!(
            projection
                .sole_aggregated_overlap_producer("model")
                .is_some()
        );
        assert!(
            projection
                .sole_aggregated_overlap_producer("other")
                .is_none()
        );
        projection.producers.push(second_aggregated);
        assert!(
            projection
                .sole_aggregated_overlap_producer("model")
                .is_none()
        );
    }

    #[test]
    fn duplicate_producer_fails_scoped_catalog() {
        let same = descriptor("dynamo-mocker");
        let update = KvPoolCatalogUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(RelayIdentity {
                drt_instance_id: 1,
                relay_incarnation: 2,
            }),
            snapshot: Some(KvPoolCatalogSnapshot {
                pools: vec![same.clone(), same],
            }),
            contract_marker: RELAY_CONTRACT_MARKER,
            ..Default::default()
        };
        assert!(matches!(
            project_catalog(&update, &scope(), 123),
            Err(RelayProjectionError::DuplicateProducer)
        ));
    }

    #[test]
    fn scopes_readiness_and_rejects_duplicate_model() {
        let entry = TopologyEntry {
            namespace: "dynamo-mocker".into(),
            canonical_model_id: "model".into(),
            state: ServingReadinessState::Ready as i32,
            members: vec![TopologyMember {
                endpoint: Some(DynamoEndpointId {
                    namespace: "dynamo-mocker".into(),
                    component: "mocker".into(),
                    endpoint: "generate".into(),
                }),
                roles: vec![WorkerRole::Aggregated as i32],
                ..Default::default()
            }],
            ..Default::default()
        };
        let update = ServingReadinessUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(RelayIdentity {
                drt_instance_id: 1,
                relay_incarnation: 2,
            }),
            entries: vec![entry.clone()],
            contract_marker: RELAY_CONTRACT_MARKER,
            ..Default::default()
        };
        match project_readiness(&update, &scope(), 123).unwrap() {
            PoolObservation::Readiness { models, status } => {
                assert_eq!(models.get("model"), Some(&ServingReadiness::Ready));
                assert_eq!(status.received_at_unix_ms, Some(123));
            }
            _ => panic!("expected readiness"),
        }
        let duplicate = ServingReadinessUpdate {
            entries: vec![entry.clone(), entry],
            ..update
        };
        assert!(matches!(
            project_readiness(&duplicate, &scope(), 123),
            Err(RelayProjectionError::DuplicateModel(_))
        ));
    }
}
