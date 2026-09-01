// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned WAN contract for DC-local Relay pool publications.

pub mod v1 {
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.kvrelay.v1");
}

/// Compact transport encodings and validation shared by producers and consumers.
pub mod wire;

pub use v1::{
    AdapterReadiness, BaseModelTarget, CkfFormat, DigestIdentity, DynamoEndpointId, FilterUpdate,
    FilterUpdateKind, IdentitySource, IndexerDomainId, KvPoolCatalogSnapshot, KvPoolCatalogUpdate,
    KvPoolDescriptor, KvPoolId, KvPoolLoadEntry, KvPoolLoadUpdate, KvQueryHashFormat,
    KvQuerySemantics, LoraModelTarget, ModelRegistration, ModelTarget, ProducerIdentity,
    RelayIdentity, RelayInfo, RelayInfoRequest, ServingReadinessState, ServingReadinessUpdate,
    SubscribeKvPoolLoadRequest, SubscribeKvPoolRequest, SubscribeServingReadinessRequest,
    TopologyEntry, TopologyMember, WatchKvPoolCatalogRequest, WorkerRole,
    kv_event_relay_client::KvEventRelayClient,
};

#[cfg(feature = "kv-dc-relay-wan")]
pub use v1::kv_event_relay_server::{KvEventRelay, KvEventRelayServer};
pub use wire::{
    WireIdentityError, validate_ckf_format, validate_contract_marker, validate_endpoint_id,
    validate_model_registration, validate_pool_descriptor, validate_pool_id,
    validate_producer_identity, validate_protocol_envelope, validate_query_semantics,
    validate_topology_entry, validate_worker_roles,
};

/// Current Relay protocol revision.
pub const RELAY_PROTOCOL_VERSION: u32 = 1;
/// Marker carried by every top-level message for the Relay WAN v1 contract.
pub const RELAY_CONTRACT_MARKER: u32 = 0x4B56_5231;
/// Current composition of cache semantics, routing scope, and DC identity.
pub const POOL_IDENTITY_VERSION: u32 = 1;

/// Descriptor set used by the WAN reflection service.
#[cfg(feature = "kv-dc-relay-wan")]
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
}
