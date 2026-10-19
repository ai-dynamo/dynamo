// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use uuid::Uuid;

use super::{
    NATIVE_DYNAMO_PROVIDER, NATIVE_DYNAMO_VERSION, NativeDynamoReplayComposition,
    NativeDynamoRouterConfig, ReplayAffinityConfig, ReplayAffinityMode,
};
use aisimulate_core::engine::{Backend, EngineConfig, NativeHostOffloadConfig, TimingModelConfig};
use aisimulate_core::replay::{
    AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1, AgenticConversationLineage, AgenticRuntimeIdentity,
    DirectRequest, ProviderSpec, ReplayAdapters, ReplayCaptureOptions, ReplayComposition,
    ReplayEngineConfig, ReplayEngineFactory, ReplayRequestContext, ReplayRuntimeInput,
    ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec,
    ReplayTelemetryObserver, ReplayTelemetrySampleKind, ReplayTelemetrySnapshot, ReplayTopology,
    WorkerPoolSpec, run_replay_with_composition,
};

fn fixture(
    backend: Backend,
    disaggregated: bool,
    mode: ReplayAffinityMode,
) -> (ReplaySpec, NativeDynamoRouterConfig) {
    let config = NativeDynamoRouterConfig {
        affinity: Some(ReplayAffinityConfig {
            mode,
            ttl_seconds: 1.0,
        }),
        ..Default::default()
    };
    let pool = WorkerPoolSpec {
        initial_workers: 2,
        startup_delay_ms: 0.0,
    };
    let spec = ReplaySpec {
        version: 1,
        topology: if disaggregated {
            ReplayTopology::Disaggregated {
                prefill: pool.clone(),
                decode: pool,
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::Aggregated { workers: pool }
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            dp_size: 2,
            rank: EngineConfig {
                block_size: 4,
                num_gpu_blocks: 64,
                max_num_batched_tokens: 32,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 10.0,
                    decode_ms: 1.0,
                },
                ..EngineConfig::for_backend(backend)
            },
            ..Default::default()
        })
        .unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec {
                provider: NATIVE_DYNAMO_PROVIDER.into(),
                config: serde_json::to_value(&config).unwrap(),
            },
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    (spec, config)
}

fn request(id: u128, conversation: &str, at_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
        output_token_ids: Some(vec![9, 10]),
        max_output_tokens: 2,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms: Some(at_ms),
        replay_context: Some(ReplayRequestContext {
            authored_id: format!("turn-{id}"),
            session_id: Some(conversation.into()),
            turn_index: Some(id as usize - 1),
            metadata: Value::Null,
            prompt_token_source: Default::default(),
            agentic: Some(AgenticRuntimeIdentity {
                request_id: format!("turn-{id}"),
                play_id: "play-a".into(),
                conversation_id: conversation.into(),
                lane_id: None,
                root_id: Some("root-turn".into()),
                parent_id: Some("parent-turn".into()),
                cache_id: None,
                lineage: Some(AgenticConversationLineage {
                    schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                    root_conversation_id: "root-conversation".into(),
                    parent_conversation_id: Some("parent-conversation".into()),
                }),
            }),
        }),
        ..Default::default()
    }
}

#[test]
fn native_policy_runs_inside_aisim_with_cache_feedback_lineage_and_virtual_ttl() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        for disaggregated in [false, true] {
            for mode in [
                ReplayAffinityMode::Session,
                ReplayAffinityMode::SiblingGroup,
            ] {
                let (spec, config) = fixture(backend, disaggregated, mode);
                let composition = NativeDynamoReplayComposition::from_config(config, None)
                    .with_capture_decisions(true);
                let evidence = composition.routing_evidence();
                let conversations = match mode {
                    ReplayAffinityMode::Session => ["same", "same", "same"],
                    ReplayAffinityMode::SiblingGroup => ["child-a", "child-b", "child-c"],
                };
                let requests = VecDeque::from([
                    request(1, conversations[0], 0.0),
                    request(2, conversations[1], 50.0),
                    request(3, conversations[2], 1100.0),
                ]);
                let report = run_replay_with_composition(
                    spec,
                    ReplayEngineFactory::new(),
                    Some(ReplayRuntimeInput::Requests(requests)),
                    composition,
                    ReplayCaptureOptions {
                        capture_per_request: true,
                        capture_lifecycle_evidence: true,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap_or_else(|error| {
                    panic!("{backend:?}, disaggregated={disaggregated}, {mode:?}: {error}")
                });
                assert_eq!(report.request_counts.completed_requests, 3);
                assert_eq!(report.per_request.len(), 3);
                let second = report
                    .per_request
                    .iter()
                    .find(|record| record.uuid == Uuid::from_u128(2).to_string())
                    .unwrap();
                assert!(
                    second.reused_input_tokens > 0,
                    "engine must reuse the prefix"
                );
                assert_eq!(
                    second.agentic.as_ref().unwrap().conversation_id,
                    conversations[1]
                );
                let evidence = evidence.snapshot().unwrap();
                let roles = if disaggregated {
                    vec!["prefill", "decode"]
                } else {
                    vec!["aggregated"]
                };
                for role in roles {
                    let role_evidence = &evidence["roles"][role];
                    assert_eq!(role_evidence["dynamo_version"], NATIVE_DYNAMO_VERSION);
                    assert_eq!(
                        role_evidence["native_policy"],
                        "dynamo.DefaultWorkerSelector"
                    );
                    assert_eq!(role_evidence["decision_count"], 3);
                    assert_eq!(role_evidence["post_dispatch_checks"], 3);
                    assert_eq!(role_evidence["dispatch_aborts"], 0);
                    let decisions = role_evidence["decisions"].as_array().unwrap();
                    assert_eq!(decisions[0]["binding_reused"], false);
                    assert_eq!(decisions[1]["binding_reused"], true);
                    assert_eq!(decisions[2]["binding_reused"], false);
                    assert_eq!(decisions[0]["group_key"], decisions[1]["group_key"]);
                    assert_eq!(decisions[0]["worker_id"], decisions[1]["worker_id"]);
                    assert_eq!(decisions[0]["dp_rank"], decisions[1]["dp_rank"]);
                    if role == "decode" {
                        assert_eq!(role_evidence["physical_kv_events"], 0);
                    } else {
                        assert!(role_evidence["physical_kv_events"].as_u64().unwrap() > 0);
                        assert!(decisions[1]["overlap_blocks"].as_u64().unwrap() > 0);
                    }
                }
            }
        }
    }
}

#[test]
fn sibling_group_rejects_missing_lineage_before_dispatch() {
    let (spec, config) = fixture(Backend::Vllm, false, ReplayAffinityMode::SiblingGroup);
    let mut request = request(1, "child-a", 0.0);
    request
        .replay_context
        .as_mut()
        .unwrap()
        .agentic
        .as_mut()
        .unwrap()
        .lineage = None;
    let composition =
        NativeDynamoReplayComposition::from_config(config, None).with_capture_decisions(true);
    let evidence = composition.routing_evidence();
    let error = run_replay_with_composition(
        spec,
        ReplayEngineFactory::new(),
        Some(ReplayRuntimeInput::Requests(VecDeque::from([request]))),
        composition,
        ReplayCaptureOptions::default(),
        None,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unambiguous conversation lineage")
    );
    assert_eq!(
        evidence.snapshot().unwrap()["roles"]["aggregated"]["decision_count"],
        0
    );
}

#[test]
fn affinity_rejects_authored_placement_constraints_before_dispatch() {
    for disaggregated in [false, true] {
        for constraint in ["decode_dp", "prefill_dp", "policy_class"] {
            let (spec, config) = fixture(Backend::Vllm, disaggregated, ReplayAffinityMode::Session);
            let mut request = request(1, "conversation", 0.0);
            match constraint {
                "decode_dp" => request.preferred_dp_rank = Some(1),
                "prefill_dp" => request.preferred_prefill_dp_rank = Some(1),
                "policy_class" => request.policy_class = Some("latency".into()),
                _ => unreachable!(),
            }
            let composition = NativeDynamoReplayComposition::from_config(config, None)
                .with_capture_decisions(true);
            let evidence = composition.routing_evidence();
            let error = run_replay_with_composition(
                spec,
                ReplayEngineFactory::new(),
                Some(ReplayRuntimeInput::Requests(VecDeque::from([request]))),
                composition,
                ReplayCaptureOptions::default(),
                None,
            )
            .unwrap_err();
            let expected = if constraint == "policy_class" {
                "does not support authored policy classes"
            } else {
                "does not support authored DP-rank pins"
            };
            assert!(
                error.to_string().contains(expected),
                "disaggregated={disaggregated}, {constraint}: {error}"
            );
            for role in evidence.snapshot().unwrap()["roles"]
                .as_object()
                .unwrap()
                .values()
            {
                assert_eq!(role["decision_count"], 0);
                assert_eq!(role["post_dispatch_checks"], 0);
            }
        }
    }
}

#[test]
fn affinity_rejects_unsupported_engine_and_router_options() {
    let (base, config) = fixture(Backend::Vllm, false, ReplayAffinityMode::Session);
    for (field, value, message) in [
        ("backend", json!("trtllm"), "supports vllm or sglang"),
        ("aic_nextn", json!(2), "speculative decoding"),
        (
            "native_host_offload",
            serde_json::to_value(NativeHostOffloadConfig::new(8)).unwrap(),
            "device KV cache only",
        ),
    ] {
        let mut spec = base.clone();
        spec.engine["rank"][field] = value;
        let composition = NativeDynamoReplayComposition::from_config(config.clone(), None);
        assert!(
            composition
                .validate_spec(&spec)
                .unwrap_err()
                .to_string()
                .contains(message)
        );
    }
    let mut unsupported = config;
    unsupported.router_config = Some(super::KvRouterConfig {
        router_queue_threshold: Some(0.5),
        ..Default::default()
    });
    let mut spec = base;
    spec.adapters.placement.config = serde_json::to_value(&unsupported).unwrap();
    let composition = NativeDynamoReplayComposition::from_config(unsupported, None);
    assert!(
        composition
            .validate_spec(&spec)
            .unwrap_err()
            .to_string()
            .contains("default native KV router configuration")
    );
}

#[test]
fn ordinary_routing_rejects_unmodeled_services_and_cache_sources() {
    let (base, _) = fixture(Backend::Vllm, false, ReplayAffinityMode::Session);
    for (field, value) in [
        ("use_remote_indexer", json!(true)),
        ("conditional_disagg_enabled", json!(true)),
        ("use_kv_events", json!(false)),
    ] {
        let config = NativeDynamoRouterConfig::from_value(&json!({
            "router_config": {(field): value}
        }))
        .unwrap();
        assert!(config.affinity.is_none());
        let mut spec = base.clone();
        spec.adapters.placement.config = serde_json::to_value(&config).unwrap();
        let error = NativeDynamoReplayComposition::from_config(config, None)
            .validate_spec(&spec)
            .unwrap_err();
        assert!(
            error.to_string().contains(&format!(
                "does not support overriding router_config.{field}"
            )),
            "{field}: {error}"
        );
    }
}

#[test]
fn native_config_and_provider_mismatches_are_not_silently_ignored() {
    for config in [
        json!({"unsupported": true}),
        json!({"affinity": {"mode": "session", "unsupported": true}}),
        json!({"router_config": {"unsupported": true}}),
    ] {
        assert!(NativeDynamoRouterConfig::from_value(&config).is_err());
    }
    let (base, config) = fixture(Backend::Vllm, false, ReplayAffinityMode::Session);
    for (placement, scaling, placement_config, scaling_config) in [
        (
            "unknown",
            "none",
            base.adapters.placement.config.clone(),
            Value::Null,
        ),
        (
            NATIVE_DYNAMO_PROVIDER,
            "unknown",
            base.adapters.placement.config.clone(),
            Value::Null,
        ),
        (NATIVE_DYNAMO_PROVIDER, "none", Value::Null, Value::Null),
        (
            NATIVE_DYNAMO_PROVIDER,
            "none",
            base.adapters.placement.config.clone(),
            json!({"unused": true}),
        ),
    ] {
        let mut spec = base.clone();
        spec.adapters.placement.provider = placement.into();
        spec.adapters.placement.config = placement_config;
        spec.adapters.scaling.provider = scaling.into();
        spec.adapters.scaling.config = scaling_config;
        assert!(
            NativeDynamoReplayComposition::from_config(config.clone(), None)
                .validate_spec(&spec)
                .is_err()
        );
    }
}

struct GrowOnce;

impl ReplayScalingPolicy for GrowOnce {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(1.0)
    }

    fn on_tick(&mut self, _: ReplayScalingSnapshot) -> anyhow::Result<ReplayScalingDecision> {
        Ok(ReplayScalingDecision {
            target_prefill: Some(3),
            target_decode: Some(3),
            next_tick_ms: None,
        })
    }
}

struct CaptureSamples(Arc<Mutex<Vec<ReplayTelemetrySnapshot>>>);

impl ReplayTelemetryObserver for CaptureSamples {
    fn on_sample(&mut self, snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(snapshot);
        Ok(())
    }
}

#[test]
fn ordinary_native_routing_preserves_scaling_capture_and_optional_telemetry() {
    for disaggregated in [false, true] {
        let (mut spec, mut config) =
            fixture(Backend::Vllm, disaggregated, ReplayAffinityMode::Session);
        config.affinity = None;
        spec.adapters.placement.config = serde_json::to_value(&config).unwrap();
        spec.adapters.scaling.provider = "dynamo_planner".into();
        let composition =
            NativeDynamoReplayComposition::from_config(config, Some(Box::new(GrowOnce)));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let report = run_replay_with_composition(
            spec,
            ReplayEngineFactory::new(),
            Some(ReplayRuntimeInput::Requests(VecDeque::from([
                request(1, "ordinary", 0.0),
                request(2, "ordinary", 50.0),
            ]))),
            composition,
            ReplayCaptureOptions {
                capture_per_request: true,
                capture_lifecycle_evidence: true,
                ..Default::default()
            },
            Some((3.0, Box::new(CaptureSamples(samples.clone())))),
        )
        .unwrap();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.per_request.len(), 2);
        assert!(!report.runtime_evidence.lifecycle_operations.is_empty());
        assert!(
            report
                .runtime_evidence
                .lifecycle_operations
                .iter()
                .all(|event| event.at_ms == 1.0 && event.state_after_batch.active.len() == 3)
        );
        let samples = samples.lock().unwrap();
        assert_eq!(samples[0].kind, ReplayTelemetrySampleKind::Baseline);
        assert!(
            samples
                .iter()
                .any(|sample| sample.kind == ReplayTelemetrySampleKind::Periodic)
        );
        assert!(
            samples
                .iter()
                .any(|sample| sample.active_decode_ids.len() == 3)
        );
        if disaggregated {
            assert!(
                samples
                    .iter()
                    .any(|sample| sample.active_prefill_ids.len() == 3)
            );
        }
    }
}
