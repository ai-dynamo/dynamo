// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use aiperf_runtime::coordinator::ResponseV2;
use aiperf_runtime::extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};
use aiperf_runtime::protocol_v2::{EnvelopeV2, OperationV2, PROTOCOL_V2, Sequencer};
use aiperf_simulate::aisimulate::{
    OfflineEngineConfig, OfflineEngineFactory, OfflinePlacement, OfflineTopology,
};
use aiperf_steppable_abi::{
    ByteSliceV1, CreateRequestV1, DirectRequestV1, REQUEST_FACT_FLAG_ADMISSION,
    REQUEST_FACT_FLAG_LATENCIES, REQUEST_FACT_FLAG_OUTPUT_LENGTH, REQUEST_FLAG_ARRIVAL_TIMESTAMP,
    REQUEST_FLAG_UUID, ReplayContextV1, ReplayHandleV1, StatusV1, StepRequestV1, StepResultV1,
    U32SliceV1, validate_descriptor_v1,
};
use aisimulate_core::replay::{DirectRequest, ReplayTerminalStatus};
use dynamo_aiperf_sim::{
    DynamoAIPerfRegistryFactory, DynamoAISimulateExtension, DynamoKvRouterEngineFactory,
};
use uuid::Uuid;

type RequestFactTrace = ([u8; 16], u32, u64, u64, f64, f64, f64);

#[derive(Debug, PartialEq)]
struct StepTrace {
    end_ms: f64,
    events: Vec<([u8; 16], u32, u32)>,
    facts: Vec<RequestFactTrace>,
}

fn request(request: u8) -> DirectRequest {
    let prefix = if request.is_multiple_of(2) {
        1_000
    } else {
        2_000
    };
    DirectRequest {
        tokens: (0..64).map(|offset| prefix + offset).collect(),
        max_output_tokens: 8,
        uuid: Some(Uuid::from_bytes([request; 16])),
        dp_rank: 0,
        arrival_timestamp_ms: Some(0.0),
        ..DirectRequest::default()
    }
}

fn terminal_status(status: Option<ReplayTerminalStatus>) -> u32 {
    match status {
        Some(ReplayTerminalStatus::Completed) => 1,
        Some(ReplayTerminalStatus::Rejected) => 2,
        Some(ReplayTerminalStatus::Canceled) => 3,
        Some(ReplayTerminalStatus::Failed) => 4,
        None => 0,
    }
}

fn native_trace(config: &OfflineEngineConfig) -> Vec<StepTrace> {
    let mut engine = DynamoKvRouterEngineFactory::default()
        .build(config)
        .expect("source-linked KV-router engine builds");
    for request_id in 0_u8..24 {
        engine
            .submit(request(request_id))
            .expect("request is accepted");
    }

    let mut trace = Vec::new();
    while !engine.is_idle() {
        let outcome = engine
            .step_until(1_000_000.0)
            .expect("source-linked replay advances");
        let facts = outcome
            .events
            .iter()
            .filter_map(|event| {
                let mut flags = 0;
                let mut reused_input_tokens = 0;
                let mut admission_ms = 0.0;
                if let Some((at_ms, reused)) = engine.request_admission(event.uuid) {
                    flags |= REQUEST_FACT_FLAG_ADMISSION;
                    admission_ms = at_ms;
                    reused_input_tokens = reused as u64;
                }
                let mut ttft_ms = 0.0;
                let mut mean_itl_ms = 0.0;
                if let Some((ttft, mean_itl)) = engine.request_latencies(event.uuid) {
                    flags |= REQUEST_FACT_FLAG_LATENCIES;
                    ttft_ms = ttft;
                    mean_itl_ms = mean_itl;
                }
                let mut output_length = 0;
                if let Some(length) = engine.actual_output_length(event.uuid) {
                    flags |= REQUEST_FACT_FLAG_OUTPUT_LENGTH;
                    output_length = length as u64;
                }
                (flags != 0).then_some((
                    *event.uuid.as_bytes(),
                    flags,
                    reused_input_tokens,
                    output_length,
                    admission_ms,
                    ttft_ms,
                    mean_itl_ms,
                ))
            })
            .collect();
        trace.push(StepTrace {
            end_ms: outcome.end_ms,
            events: outcome
                .events
                .into_iter()
                .map(|event| {
                    (
                        *event.uuid.as_bytes(),
                        u32::from(event.emitted_token)
                            | (u32::from(event.terminal_status.is_some()) << 1),
                        terminal_status(event.terminal_status),
                    )
                })
                .collect(),
            facts,
        });
    }
    trace
}

fn monolithic_trace(config: &OfflineEngineConfig) -> Vec<StepTrace> {
    let descriptor = dynamo_steppable_provider::aiperf_steppable_plugin_v1();
    let table = unsafe { &*validate_descriptor_v1(descriptor).expect("complete V1 descriptor") };
    let backend_config = dynamo_steppable_provider::BackendConfig {
        topology: dynamo_steppable_provider::BackendTopology::Aggregated,
        engine: config
            .aggregate_replay_engine_config()
            .expect("aggregate replay config"),
        workers: config.workers,
        dynamic_placement: Some(dynamo_steppable_provider::DynamicPlacementLocator {
            library_path: PathBuf::from("unused-by-monolith"),
            selector_seed: [7; 32],
            options_namespace: Vec::new(),
            provider_options: Vec::new(),
            limits: dynamo_steppable_provider::DynamicPlacementLimits::default(),
        }),
        ..dynamo_steppable_provider::BackendConfig::default()
    };
    let payload = serde_json::to_vec(&backend_config).expect("serializable monolith config");
    let mut handle = ReplayHandleV1(std::ptr::null_mut());
    let mut error = ByteSliceV1::EMPTY;
    assert_eq!(
        unsafe {
            table.create.expect("create")(
                CreateRequestV1 {
                    struct_size: std::mem::size_of::<CreateRequestV1>() as u32,
                    flags: 0,
                    provider_payload: ByteSliceV1 {
                        data: payload.as_ptr(),
                        len: payload.len() as u64,
                    },
                },
                &mut handle,
                &mut error,
            )
        },
        StatusV1::OK
    );

    for request_id in 0_u8..24 {
        let request = request(request_id);
        let mut submitted_id = [0; 16];
        assert_eq!(
            unsafe {
                table.submit.expect("submit")(
                    handle,
                    DirectRequestV1 {
                        struct_size: std::mem::size_of::<DirectRequestV1>() as u32,
                        flags: REQUEST_FLAG_UUID | REQUEST_FLAG_ARRIVAL_TIMESTAMP,
                        tokens: U32SliceV1 {
                            data: request.tokens.as_ptr(),
                            len: request.tokens.len() as u64,
                        },
                        output_token_ids: U32SliceV1::EMPTY,
                        max_output_tokens: request.max_output_tokens as u64,
                        uuid: *request.uuid.expect("explicit UUID").as_bytes(),
                        dp_rank: request.dp_rank,
                        preferred_dp_rank: 0,
                        preferred_prefill_dp_rank: 0,
                        arrival_timestamp_ms: request.arrival_timestamp_ms.expect("arrival"),
                        priority: 0,
                        strict_priority: 0,
                        policy_class: ByteSliceV1::EMPTY,
                        replay_context: ReplayContextV1::EMPTY,
                    },
                    &mut submitted_id,
                )
            },
            StatusV1::OK
        );
    }

    let mut trace = Vec::new();
    loop {
        let mut result = StepResultV1::EMPTY;
        assert_eq!(
            unsafe {
                table.step.expect("step")(
                    handle,
                    StepRequestV1 {
                        struct_size: std::mem::size_of::<StepRequestV1>() as u32,
                        flags: 0,
                        until_ms: 1_000_000.0,
                    },
                    &mut result,
                )
            },
            StatusV1::OK
        );
        let events = if result.events.len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.events.data, result.events.len as usize) }
                .iter()
                .map(|event| (event.request_id, event.flags, event.terminal_status))
                .collect()
        };
        let facts = if result.request_facts.len == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    result.request_facts.data,
                    result.request_facts.len as usize,
                )
            }
            .iter()
            .map(|fact| {
                (
                    fact.request_id,
                    fact.flags,
                    fact.reused_input_tokens,
                    fact.output_length,
                    fact.admission_ms,
                    fact.ttft_ms,
                    fact.mean_itl_ms,
                )
            })
            .collect()
        };
        let is_idle = result.is_idle != 0;
        trace.push(StepTrace {
            end_ms: result.end_ms,
            events,
            facts,
        });
        unsafe {
            table.release_events.expect("release events")(result.events);
            table.release_request_facts.expect("release facts")(result.request_facts);
        }
        if is_idle {
            break;
        }
    }
    unsafe { table.destroy.expect("destroy")(handle) };
    trace
}

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
fn dynamo_provider_matches_the_monolithic_router_admission_schedule() {
    let config = OfflineEngineConfig {
        topology: OfflineTopology::Aggregated,
        workers: 8,
        placement: OfflinePlacement::KvRouter {
            selector_seed: Some(0x0707_0707_0707_0707),
        },
        ..OfflineEngineConfig::default()
    };
    assert_eq!(
        native_trace(&config),
        monolithic_trace(&config),
        "source-linked and monolithic engines must schedule the same replay"
    );
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
