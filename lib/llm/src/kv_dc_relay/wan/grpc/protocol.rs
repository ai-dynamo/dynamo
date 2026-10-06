// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned WAN contract for DC-local Relay pool publications.

pub mod v1 {
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.kvrelay.v1");
}

/// Compact transport encodings and validation shared by producers and consumers.
pub mod wire;

mod errors;
pub use errors::{RELAY_ERROR_REASON_HEADER, relay_error_reason};

pub use v1::{
    AdapterReadiness, BaseModelTarget, CkfFormat, DigestIdentity, DynamoEndpointId, FilterUpdate,
    FilterUpdateKind, IdentitySource, IndexerDomainId, KvPoolCatalogSnapshot, KvPoolCatalogUpdate,
    KvPoolDescriptor, KvPoolId, KvPoolLoadEntry, KvPoolLoadUpdate, KvQueryHashFormat,
    KvQuerySemantics, LoraModelTarget, ModelRegistration, ModelTarget, ProducerIdentity,
    RelayErrorReason, RelayIdentity, RelayInfo, RelayInfoRequest, ServingReadinessState,
    ServingReadinessUpdate, SubscribeKvPoolLoadRequest, SubscribeKvPoolRequest,
    SubscribeServingReadinessRequest, TopologyEntry, TopologyMember, WatchKvPoolCatalogRequest,
    WorkerRole, kv_event_relay_client::KvEventRelayClient,
};
pub use v1::{
    DataStatus, LoadView, ModelDeploymentStatus, ModelServingLoad, PoolDeploymentStatus,
    PoolServingLoad, RequestLifecycleStats, ServingLoadUpdate, SubscribeServingLoadRequest,
    TokenLoadStats,
};

pub use v1::kv_event_relay_server::{KvEventRelay, KvEventRelayServer};
pub use wire::{
    ProducerKey, WireIdentityError, validate_ckf_format, validate_contract_marker,
    validate_endpoint_id, validate_model_registration, validate_pool_descriptor, validate_pool_id,
    validate_producer_identity, validate_protocol_envelope, validate_query_semantics,
    validate_topology_entry, validate_worker_roles,
};
pub use wire::{
    validate_model_serving_load, validate_pool_serving_load, validate_serving_load_update,
};

/// Compatibility major; additive v1 changes do not increment this value.
pub const RELAY_PROTOCOL_VERSION: u32 = 1;
/// Marker carried by every top-level message for the Relay WAN v1 contract.
pub const RELAY_CONTRACT_MARKER: u32 = 0x4B56_5231;
/// Current composition of cache semantics, routing scope, and DC identity.
pub const POOL_IDENTITY_VERSION: u32 = 1;

/// Descriptor set used by the WAN reflection service.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/relay_descriptor.bin"));

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use prost::Message as _;

    use super::*;

    fn relay_identity() -> RelayIdentity {
        RelayIdentity {
            drt_instance_id: 17,
            relay_incarnation: 23,
        }
    }

    fn pool_id() -> KvPoolId {
        KvPoolId {
            identity_version: POOL_IDENTITY_VERSION,
            indexer_domain: Some(IndexerDomainId {
                cache_semantics: Some(DigestIdentity {
                    digest: Bytes::from_static(&[1; 16]),
                    source: IdentitySource::Explicit as i32,
                }),
                routing_scope: Some(DigestIdentity {
                    digest: Bytes::from_static(&[2; 16]),
                    source: IdentitySource::DefaultDerived as i32,
                }),
            }),
            dc_id: 7,
        }
    }

    fn producer() -> ProducerIdentity {
        ProducerIdentity {
            pool_id: Some(pool_id()),
            producer_incarnation: 23,
            layout_generation: 5,
            ckf_format: Some(CkfFormat {
                format_version: 1,
                seed: 42,
                bucket_count: 1 << 10,
                fingerprint_bits: 16,
                slots_per_bucket: 4,
            }),
        }
    }

    // Independent schemas model additive evolution without changing the v1 package/marker.
    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedInfoRequest {
        #[prost(fixed32, tag = "127")]
        contract_marker: u32,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedRelayInfo {
        #[prost(uint32, tag = "1")]
        protocol_version: u32,
        #[prost(message, optional, tag = "2")]
        relay: Option<RelayIdentity>,
        #[prost(fixed32, tag = "127")]
        contract_marker: u32,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedDescriptor {
        #[prost(message, optional, tag = "1")]
        producer: Option<ProducerIdentity>,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[test]
    fn additive_v1_fields_work_in_both_directions_without_entering_identity() {
        let future_request = ExtendedInfoRequest {
            contract_marker: RELAY_CONTRACT_MARKER,
            diagnostic_label: Some("future client".into()),
        };
        let old_request =
            RelayInfoRequest::decode(future_request.encode_to_vec().as_slice()).unwrap();
        validate_contract_marker(old_request.contract_marker).unwrap();
        let new_request =
            ExtendedInfoRequest::decode(old_request.encode_to_vec().as_slice()).unwrap();
        assert_eq!(new_request.contract_marker, RELAY_CONTRACT_MARKER);
        assert_eq!(new_request.diagnostic_label, None);

        let future_info = ExtendedRelayInfo {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            contract_marker: RELAY_CONTRACT_MARKER,
            diagnostic_label: Some("future server".into()),
        };
        let old_info = RelayInfo::decode(future_info.encode_to_vec().as_slice()).unwrap();
        validate_protocol_envelope(old_info.protocol_version, old_info.contract_marker).unwrap();
        assert_eq!(old_info.relay, future_info.relay);
        let new_info = ExtendedRelayInfo::decode(old_info.encode_to_vec().as_slice()).unwrap();
        assert_eq!(new_info.protocol_version, future_info.protocol_version);
        assert_eq!(new_info.contract_marker, future_info.contract_marker);
        assert_eq!(new_info.relay, future_info.relay);
        assert_eq!(new_info.diagnostic_label, None);

        let future_descriptor = ExtendedDescriptor {
            producer: Some(producer()),
            diagnostic_label: Some("not a generation dimension".into()),
        };
        let old_descriptor =
            KvPoolDescriptor::decode(future_descriptor.encode_to_vec().as_slice()).unwrap();
        let key = ProducerKey::try_from(future_descriptor.producer.as_ref().unwrap()).unwrap();
        assert_eq!(
            ProducerKey::try_from(old_descriptor.producer.as_ref().unwrap()).unwrap(),
            key
        );
        let new_descriptor =
            ExtendedDescriptor::decode(old_descriptor.encode_to_vec().as_slice()).unwrap();
        assert_eq!(
            ProducerKey::try_from(new_descriptor.producer.as_ref().unwrap()).unwrap(),
            key
        );
        assert_eq!(new_descriptor.diagnostic_label, None);
    }

    #[test]
    fn producer_key_fences_every_v1_identity_dimension() {
        let original = producer();
        let key = ProducerKey::try_from(&original).unwrap();
        let mutations: &[fn(&mut ProducerIdentity)] = &[
            |p| p.pool_id.as_mut().unwrap().identity_version += 1,
            |p| p.pool_id.as_mut().unwrap().dc_id += 1,
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .cache_semantics
                    .as_mut()
                    .unwrap()
                    .digest = Bytes::from_static(&[8; 16])
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .routing_scope
                    .as_mut()
                    .unwrap()
                    .digest = Bytes::from_static(&[9; 16])
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .cache_semantics
                    .as_mut()
                    .unwrap()
                    .source = IdentitySource::DefaultDerived as i32
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .routing_scope
                    .as_mut()
                    .unwrap()
                    .source = IdentitySource::Explicit as i32
            },
            |p| p.producer_incarnation += 1,
            |p| p.layout_generation += 1,
            |p| p.ckf_format.as_mut().unwrap().format_version += 1,
            |p| p.ckf_format.as_mut().unwrap().seed += 1,
            |p| p.ckf_format.as_mut().unwrap().bucket_count *= 2,
            |p| p.ckf_format.as_mut().unwrap().fingerprint_bits += 1,
            |p| p.ckf_format.as_mut().unwrap().slots_per_bucket += 1,
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut changed = original.clone();
            mutate(&mut changed);
            assert_ne!(
                ProducerKey::try_from(&changed),
                Ok(key),
                "key dimension {index} was ignored"
            );
        }
    }

    #[test]
    fn unknown_semantics_are_distinguished_from_malformed_known_values() {
        for value in [99, -1] {
            let error = validate_query_semantics(&KvQuerySemantics {
                kv_block_size: 64,
                hash_format: value,
            })
            .unwrap_err();
            assert!(error.is_unsupported());
            assert!(
                validate_worker_roles(&[value])
                    .unwrap_err()
                    .is_unsupported()
            );
        }
        assert!(
            !validate_query_semantics(&KvQuerySemantics {
                kv_block_size: 64,
                hash_format: 0,
            })
            .unwrap_err()
            .is_unsupported()
        );
        assert!(!validate_worker_roles(&[0]).unwrap_err().is_unsupported());
        let known = WorkerRole::Prefill as i32;
        assert!(
            !validate_worker_roles(&[known, known])
                .unwrap_err()
                .is_unsupported()
        );

        // A future target alternative at tag 3 decodes as an unset oneof in v1.
        let target = ModelTarget::decode(&[0x1a, 0x00][..]).unwrap();
        let registration = ModelRegistration {
            canonical_model_id: "future".into(),
            target: Some(target),
            aliases: vec![],
        };
        assert_eq!(
            validate_model_registration(&registration),
            Err(WireIdentityError::UnsupportedModelTarget)
        );
        let topology = TopologyEntry {
            namespace: "ns".into(),
            canonical_model_id: "model".into(),
            state: 99,
            ..Default::default()
        };
        assert!(
            validate_topology_entry(&topology)
                .unwrap_err()
                .is_unsupported()
        );
        assert!(
            !validate_topology_entry(&TopologyEntry {
                state: ServingReadinessState::Unknown as i32,
                ..topology
            })
            .unwrap_err()
            .is_unsupported()
        ); // Missing members, not an unknown readiness enum.
        validate_query_semantics(&KvQuerySemantics {
            kv_block_size: 64,
            hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
        })
        .unwrap();
    }

    #[test]
    fn catalog_round_trips_typed_identity_endpoint_and_lora_registration() {
        let update = KvPoolCatalogUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            revision: 9,
            snapshot: Some(KvPoolCatalogSnapshot {
                pools: vec![KvPoolDescriptor {
                    producer: Some(producer()),
                    serving_endpoint: Some(DynamoEndpointId {
                        namespace: "ns".into(),
                        component: "backend".into(),
                        endpoint: "generate".into(),
                    }),
                    registrations: vec![ModelRegistration {
                        canonical_model_id: "llama-lora".into(),
                        target: Some(ModelTarget {
                            target: Some(v1::model_target::Target::Lora(LoraModelTarget {
                                base_model: "llama".into(),
                                adapter: "llama-lora".into(),
                            })),
                        }),
                        aliases: vec!["chat".into()],
                    }],
                    query_semantics: Some(KvQuerySemantics {
                        kv_block_size: 64,
                        hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
                    }),
                    pool_roles: vec![WorkerRole::Decode as i32],
                }],
            }),
            contract_marker: RELAY_CONTRACT_MARKER,
        };

        let decoded = KvPoolCatalogUpdate::decode(update.encode_to_vec().as_slice())
            .expect("catalog update must decode");
        assert_eq!(decoded, update);
        let descriptor = &decoded.snapshot.as_ref().expect("snapshot").pools[0];
        validate_pool_descriptor(descriptor).expect("pool descriptor");
    }

    #[test]
    fn load_window_round_trips_as_one_complete_update() {
        let update = KvPoolLoadUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            window_sequence: 12,
            observed_ms: 1_000,
            window_ms: 500,
            pools: vec![KvPoolLoadEntry {
                producer: Some(producer()),
                kv_used_blocks: 40,
                total_kv_blocks: 100,
                kv_observed_ranks: 3,
                kv_expected_ranks: 4,
            }],
            contract_marker: RELAY_CONTRACT_MARKER,
        };
        let decoded = KvPoolLoadUpdate::decode(update.encode_to_vec().as_slice())
            .expect("load update must decode");
        assert_eq!(decoded, update);

        let heartbeat = KvPoolLoadUpdate {
            pools: Vec::new(),
            window_sequence: 13,
            ..update
        };
        let decoded = KvPoolLoadUpdate::decode(heartbeat.encode_to_vec().as_slice())
            .expect("idle heartbeat must decode");
        assert!(decoded.pools.is_empty());
    }

    fn serving_load_update() -> ServingLoadUpdate {
        let tokens = |prefill, decode| TokenLoadStats {
            active_prefill_tokens: prefill,
            active_decode_blocks: decode,
            ..Default::default()
        };
        ServingLoadUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            window_sequence: 1,
            observed_ms: 1_000,
            window_ms: 1_000,
            pools: vec![PoolServingLoad {
                producer: Some(producer()),
                load: Some(LoadView {
                    requests: None,
                    tokens: Some(tokens(Some(3), Some(4))),
                    status: DataStatus::Complete as i32,
                    source_observed_ms: 900,
                }),
                deployment: Some(PoolDeploymentStatus::default()),
            }],
            models: vec![ModelServingLoad {
                namespace: "ns".into(),
                canonical_model_id: "llama".into(),
                load: Some(LoadView {
                    requests: Some(RequestLifecycleStats {
                        requests_started_total: 5,
                        ..Default::default()
                    }),
                    tokens: Some(TokenLoadStats {
                        input_tokens_total: Some(10),
                        output_tokens_total: Some(20),
                        ..Default::default()
                    }),
                    status: DataStatus::Degraded as i32,
                    source_observed_ms: 900,
                }),
                deployment: Some(ModelDeploymentStatus {
                    serving_pools: vec![pool_id()],
                    ..Default::default()
                }),
            }],
            contract_marker: RELAY_CONTRACT_MARKER,
        }
    }

    #[test]
    fn serving_load_window_rejects_bad_envelopes_and_duplicate_keys() {
        let update = serving_load_update();
        let decoded = ServingLoadUpdate::decode(update.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, update);
        validate_serving_load_update(&update).unwrap();
        validate_pool_serving_load(&update.pools[0]).unwrap();
        validate_model_serving_load(&update.models[0]).unwrap();

        let mut wrong_version = update.clone();
        wrong_version.protocol_version += 1;
        assert_eq!(
            validate_serving_load_update(&wrong_version),
            Err(WireIdentityError::ProtocolVersion(2))
        );
        let mut no_relay = update.clone();
        no_relay.relay = None;
        assert!(validate_serving_load_update(&no_relay).is_err());

        let mut duplicate_pool = update.clone();
        duplicate_pool.pools.push(update.pools[0].clone());
        assert_eq!(
            validate_serving_load_update(&duplicate_pool),
            Err(WireIdentityError::DuplicateLoadPool)
        );
        let mut duplicate_model = update.clone();
        duplicate_model.models.push(update.models[0].clone());
        assert!(matches!(
            validate_serving_load_update(&duplicate_model),
            Err(WireIdentityError::DuplicateLoadModel { .. })
        ));
        let mut duplicate_link = update.models[0].clone();
        duplicate_link
            .deployment
            .as_mut()
            .unwrap()
            .serving_pools
            .push(pool_id());
        assert_eq!(
            validate_model_serving_load(&duplicate_link),
            Err(WireIdentityError::DuplicateServingPool)
        );

        // A pool with an unsupported identity is quarantined, not fatal to the
        // window; a pool that cannot be identified is.
        let mut future_pool = update.clone();
        future_pool.pools[0]
            .producer
            .as_mut()
            .unwrap()
            .pool_id
            .as_mut()
            .unwrap()
            .identity_version += 1;
        validate_serving_load_update(&future_pool).unwrap();
        assert!(
            validate_pool_serving_load(&future_pool.pools[0])
                .unwrap_err()
                .is_unsupported()
        );
        let mut anonymous = update.clone();
        anonymous.pools[0].producer = None;
        assert!(validate_serving_load_update(&anonymous).is_err());
    }

    #[test]
    fn serving_load_entries_quarantine_unknown_status_and_enforce_presence_rules() {
        let update = serving_load_update();
        for status in [99, -1] {
            let mut pool = update.pools[0].clone();
            pool.load.as_mut().unwrap().status = status;
            assert!(
                validate_pool_serving_load(&pool)
                    .unwrap_err()
                    .is_unsupported()
            );
        }
        let mut unspecified = update.models[0].clone();
        unspecified.load.as_mut().unwrap().status = DataStatus::Unspecified as i32;
        assert!(
            !validate_model_serving_load(&unspecified)
                .unwrap_err()
                .is_unsupported()
        );

        let presence_violations: &[fn(&mut LoadView)] = &[
            // DEGRADED gauges would read as a partial sum.
            |view| view.requests.as_mut().unwrap().requests_generating = Some(1),
            |view| view.tokens.as_mut().unwrap().inflight_input_tokens = Some(1),
            // COMPLETE requires every gauge.
            |view| view.status = DataStatus::Complete as i32,
            // Totals are always set; pool fields never.
            |view| view.tokens.as_mut().unwrap().output_tokens_total = None,
            |view| view.tokens.as_mut().unwrap().active_decode_blocks = Some(1),
        ];
        for (index, violate) in presence_violations.iter().enumerate() {
            let mut model = update.models[0].clone();
            violate(model.load.as_mut().unwrap());
            assert!(
                matches!(
                    validate_model_serving_load(&model),
                    Err(WireIdentityError::LoadViewField(_))
                ),
                "model presence violation {index} was accepted"
            );
        }

        let mut incomplete = update.pools[0].clone();
        incomplete.load.as_mut().unwrap().status = DataStatus::Degraded as i32;
        assert!(matches!(
            validate_pool_serving_load(&incomplete),
            Err(WireIdentityError::LoadViewField("active_prefill_tokens"))
        ));
        let mut with_requests = update.pools[0].clone();
        with_requests.load.as_mut().unwrap().requests = Some(RequestLifecycleStats::default());
        assert!(matches!(
            validate_pool_serving_load(&with_requests),
            Err(WireIdentityError::LoadViewField(_))
        ));
    }
}
