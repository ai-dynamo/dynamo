// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};
use aiperf_simulate::aisimulate::{
    OfflineEngineConfig, OfflineEngineFactory, OfflinePlacement, OfflineTopology,
};
use aisimulate_core::replay::DirectRequest;
use dynamo_aiperf_sim::{DynamoAISimulateExtension, DynamoKvRouterEngineFactory};

fn kv_router_config(selector_seed: u64) -> OfflineEngineConfig {
    OfflineEngineConfig {
        topology: OfflineTopology::Aggregated,
        workers: 2,
        placement: OfflinePlacement::KvRouter {
            selector_seed: Some(selector_seed),
        },
        ..OfflineEngineConfig::default()
    }
}

#[test]
fn dynamo_provider_builds_a_seeded_kv_router_engine() {
    let mut engine = DynamoKvRouterEngineFactory::default()
        .build(&kv_router_config(7))
        .expect("seeded KV-router engine builds");
    engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3, 4],
            max_output_tokens: 1,
            output_token_ids: Some(vec![5]),
            ..DirectRequest::default()
        })
        .expect("request is admitted through the KV router");

    assert!(engine.next_event_ms().is_some());
}

#[test]
fn dynamo_extension_registers_the_static_provider() {
    let mut registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("builtin registry builds");
    registry
        .register_extension(&DynamoAISimulateExtension)
        .expect("Dynamo static simulation provider registers");

    assert!(registry.alternate_execution("aisimulate").is_some());
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        vec!["dynamo.kv_router"]
    );
}
