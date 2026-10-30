// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_common::grpc;
use dynamo_llm::pool_relay::wan::grpc::protocol as relay;
use prost::Message;

fn producer(generation: u64) -> relay::ProducerIdentity {
    relay::ProducerIdentity {
        pool_id: Some(relay::KvPoolId {
            identity_version: relay::POOL_IDENTITY_VERSION,
            indexer_domain: Some(relay::IndexerDomainId {
                cache_semantics: Some(relay::DigestIdentity {
                    digest: vec![1; 16].into(),
                    source: 1,
                }),
                routing_scope: Some(relay::DigestIdentity {
                    digest: vec![2; 16].into(),
                    source: 1,
                }),
            }),
            dc_id: 1,
        }),
        producer_incarnation: 1,
        layout_generation: generation,
        ckf_format: Some(relay::CkfFormat {
            format_version: 1,
            seed: 1,
            bucket_count: 32,
            fingerprint_bits: 16,
            slots_per_bucket: 4,
        }),
    }
}

pub fn state(
    ready: bool,
    catalog_generation: u64,
    load_generation: u64,
    window: u64,
) -> grpc::PoolState {
    let identity = relay::RelayIdentity {
        drt_instance_id: 1,
        relay_incarnation: 1,
    };
    let catalog = relay::KvPoolCatalogUpdate {
        protocol_version: relay::RELAY_PROTOCOL_VERSION,
        contract_marker: relay::RELAY_CONTRACT_MARKER,
        relay: Some(identity),
        revision: catalog_generation,
        snapshot: Some(relay::KvPoolCatalogSnapshot {
            pools: vec![relay::KvPoolDescriptor {
                producer: Some(producer(catalog_generation)),
                serving_endpoint: Some(relay::DynamoEndpointId {
                    namespace: "runtime-blue".into(),
                    component: "backend".into(),
                    endpoint: "generate".into(),
                }),
                registrations: vec![relay::ModelRegistration {
                    canonical_model_id: "qwen-canonical".into(),
                    aliases: vec!["qwen".into()],
                    target: Some(relay::ModelTarget {
                        target: Some(relay::v1::model_target::Target::Base(
                            relay::BaseModelTarget {
                                base_model: "qwen-canonical".into(),
                            },
                        )),
                    }),
                }],
                query_semantics: Some(relay::KvQuerySemantics {
                    kv_block_size: 64,
                    hash_format: relay::v1::KvQueryHashFormat::DynamoStandardV1 as i32,
                }),
                pool_roles: vec![relay::v1::WorkerRole::Legacy as i32],
            }],
        }),
    };
    let readiness = relay::ServingReadinessUpdate {
        protocol_version: relay::RELAY_PROTOCOL_VERSION,
        contract_marker: relay::RELAY_CONTRACT_MARKER,
        relay: Some(identity),
        entries: vec![relay::TopologyEntry {
            // Runtime namespace deliberately differs from the DGD's Kubernetes namespace.
            namespace: "runtime-blue".into(),
            canonical_model_id: "qwen-canonical".into(),
            state: if ready {
                relay::ServingReadinessState::Ready
            } else {
                relay::ServingReadinessState::Unavailable
            } as i32,
            present_roles: vec![relay::v1::WorkerRole::Legacy as i32],
            members: vec![relay::v1::TopologyMember {
                endpoint: Some(relay::DynamoEndpointId {
                    namespace: "runtime-blue".into(),
                    component: "backend".into(),
                    endpoint: "generate".into(),
                }),
                roles: vec![relay::v1::WorkerRole::Legacy as i32],
                pool_id: producer(catalog_generation).pool_id,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    let serving = relay::ServingLoadUpdate {
        protocol_version: relay::RELAY_PROTOCOL_VERSION,
        contract_marker: relay::RELAY_CONTRACT_MARKER,
        relay: Some(identity),
        window_sequence: window,
        window_ms: 60_000,
        pools: vec![relay::PoolServingLoad {
            producer: Some(producer(load_generation)),
            load: Some(relay::LoadView {
                status: relay::DataStatus::Unavailable as i32,
                tokens: Some(relay::TokenLoadStats::default()),
                ..Default::default()
            }),
            deployment: Some(relay::PoolDeploymentStatus {
                live_workers: Some(1),
                max_concurrency: Some(64),
            }),
        }],
        models: vec![relay::ModelServingLoad {
            namespace: "runtime-blue".into(),
            canonical_model_id: "qwen-canonical".into(),
            load: Some(relay::LoadView {
                status: relay::DataStatus::Complete as i32,
                requests: Some(relay::RequestLifecycleStats {
                    requests_started_total: 8,
                    requests_awaiting_first_token: Some(0),
                    requests_generating: Some(8),
                    ..Default::default()
                }),
                tokens: Some(relay::TokenLoadStats {
                    awaiting_first_token_input_tokens: Some(0),
                    inflight_input_tokens: Some(0),
                    input_tokens_total: Some(0),
                    output_tokens_total: Some(0),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            deployment: Some(relay::ModelDeploymentStatus {
                expected_frontends: 1,
                observed_frontends: 1,
                ready_frontends: 1,
                serving_pools: vec![producer(load_generation).pool_id.unwrap()],
            }),
        }],
        ..Default::default()
    };
    let kv = relay::KvPoolLoadUpdate {
        protocol_version: relay::RELAY_PROTOCOL_VERSION,
        contract_marker: relay::RELAY_CONTRACT_MARKER,
        relay: Some(identity),
        window_sequence: window,
        window_ms: 60_000,
        pools: vec![relay::KvPoolLoadEntry {
            producer: Some(producer(load_generation)),
            kv_used_blocks: 20,
            total_kv_blocks: 100,
            kv_observed_ranks: 1,
            kv_expected_ranks: 1,
        }],
        ..Default::default()
    };
    relay::validate_pool_descriptor(&catalog.snapshot.as_ref().unwrap().pools[0]).unwrap();
    relay::validate_topology_entry(&readiness.entries[0]).unwrap();
    relay::validate_serving_load_update(&serving).unwrap();
    relay::validate_pool_serving_load(&serving.pools[0]).unwrap();
    relay::validate_model_serving_load(&serving.models[0]).unwrap();
    grpc::PoolState {
        catalog: catalog.encode_to_vec().into(),
        readiness: readiness.encode_to_vec().into(),
        serving_load: serving.encode_to_vec().into(),
        kv_load: kv.encode_to_vec().into(),
    }
}
