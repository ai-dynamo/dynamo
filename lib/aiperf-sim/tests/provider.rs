// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use aiperf_runtime::coordinator::ResponseV2;
use aiperf_runtime::extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};
use aiperf_runtime::protocol_v2::{EnvelopeV2, OperationV2, PROTOCOL_V2, Sequencer};
use aiperf_simulate::aisimulate::{
    OfflineEngineConfig, OfflineEngineFactory, OfflinePlacement, OfflineTopology,
};
use aisimulate_core::replay::DirectRequest;
use dynamo_aiperf_sim::{
    DynamoAIPerfRegistryFactory, DynamoAISimulateExtension, DynamoKvRouterEngineFactory,
};

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

struct TestArtifactDirectory(PathBuf);

impl TestArtifactDirectory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "dynamo-aiperf-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestArtifactDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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
fn dynamo_provider_rejects_an_unseeded_kv_router() {
    let config = OfflineEngineConfig {
        topology: OfflineTopology::Aggregated,
        workers: 2,
        placement: OfflinePlacement::KvRouter {
            selector_seed: None,
        },
        ..OfflineEngineConfig::default()
    };

    let error = DynamoKvRouterEngineFactory::default()
        .validate(&config)
        .expect_err("the static Dynamo bundle must be reproducible")
        .to_string();
    assert!(
        error.contains("requires a deterministic selector seed"),
        "{error}"
    );
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

#[test]
fn dynamo_registry_factory_selects_the_static_provider() {
    let registry = DynamoAIPerfRegistryFactory
        .build()
        .expect("Dynamo AIPerf bundle registry builds");

    assert!(registry.alternate_execution("aisimulate").is_some());
    assert!(
        registry.extension_names().next().is_none(),
        "the linked bundle provider is a distribution builtin, not a runtime add-on"
    );
}

#[test]
fn dynamo_bundle_executes_kv_router_replay_with_provenance() {
    let artifacts = TestArtifactDirectory::new("bundle-replay");
    let mut run =
        serde_json::to_value(aiperf_runtime::graph_input::graph_cycle_test_support::run())
            .expect("serialize typed base run");
    run["artifact_dir"] = serde_json::json!(artifacts.path());
    run["cfg"]["runtime"]["workers"] = serde_json::json!(1);
    run["cfg"]["transport"] = serde_json::json!({
        "type": "aisimulate",
        "mode": "offline",
        "topology": "aggregated",
        "placement": "kv_router",
        "placement_selector_seed": 7,
        "artifacts": {"provider_provenance_json": "aiperf/provider.json"}
    });
    run["cfg"]["tokenizer"] = serde_json::json!({"name": "builtin"});
    run["cfg"]["datasets"] = serde_json::json!([{
        "type": "file",
        "format": "dag_jsonl",
        "sampling": "sequential",
        "records": [{
            "session_id": "root",
            "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
        }],
        "osl": {"value": 1.0},
        "options": {}
    }]);
    run["cfg"]["phases"] = serde_json::json!([{
        "type": "concurrency",
        "name": "profiling",
        "exclude_from_results": false,
        "concurrency": 1,
        "requests": 1
    }]);

    let application =
        DynamoAIPerfRegistryFactory::application(format!("blake3:{}", "a".repeat(64)))
            .expect("Dynamo AIPerf application composes");
    let result = application.handle_v2(
        EnvelopeV2 {
            protocol_version: PROTOCOL_V2,
            operation: OperationV2::Execute,
            run: serde_json::from_value(run).expect("decode full bundle run"),
        },
        &Sequencer::new(),
        None,
    );

    assert_eq!(result.exit_code, 0, "{result:?}");
    let ResponseV2::Terminal(terminal) = result.response else {
        panic!("bundle execution must return a terminal response")
    };
    assert!(terminal.success, "{terminal:?}");
    assert_eq!(
        terminal.run_metadata["simulation_provider_id"],
        "dynamo.kv_router"
    );
    assert_eq!(
        terminal.run_metadata["simulation_provider_version"],
        "1.5.0"
    );
    assert_eq!(
        terminal.run_metadata["simulation_bundle_identity"],
        "dynamo-aiperf-sim"
    );

    let provenance: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifacts.path().join("aiperf/provider.json"))
            .expect("provider provenance sidecar exists"),
    )
    .expect("provider provenance is JSON");
    assert_eq!(
        provenance,
        serde_json::json!({
            "simulation_provider_id": "dynamo.kv_router",
            "simulation_provider_version": "1.5.0",
            "simulation_bundle_identity": "dynamo-aiperf-sim"
        })
    );
}
