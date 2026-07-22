// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engine::rl_worker_metadata;

#[test]
fn from_args_discovers_aggregated_role() {
    let handle = spawn_fake_engine(FakeConfig {
        reasoning_parser: "deepseek_r1".to_string(),
        tool_call_parser: "hermes".to_string(),
        ..FakeConfig::default()
    });
    let (_engine, config) = VllmSidecarEngine::from_args(Some(vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        handle.endpoint.clone(),
    ]))
    .expect("from_args");

    assert_eq!(config.component, "backend");
    assert_eq!(config.disaggregation_mode, DisaggregationMode::Aggregated);
    assert_eq!(config.model_name, "fake-model");
    assert_eq!(config.served_model_name.as_deref(), Some("fake-served"));
    assert_eq!(config.reasoning_parser.as_deref(), Some("deepseek_r1"));
    assert_eq!(config.tool_call_parser.as_deref(), Some("hermes"));
}

#[test]
fn from_args_discovers_prefill_role_and_component() {
    let handle = spawn_fake_engine(FakeConfig::default());
    let (_engine, config) = VllmSidecarEngine::from_args(Some(vec![
        "dynamo-vllm-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        handle.endpoint.clone(),
        "--disaggregation-mode".to_string(),
        "prefill".to_string(),
    ]))
    .expect("from_args");

    assert_eq!(config.component, "prefill");
    assert_eq!(config.disaggregation_mode, DisaggregationMode::Prefill);
}

#[test]
fn rl_discovery_model_defaults_to_engine_identity_and_accepts_an_override() {
    let thunderagent_model = pb::ModelInfo {
        model_id: "Qwen/Qwen3-0.6B".to_string(),
        served_model_name: "private-routing-alias".to_string(),
        ..Default::default()
    };
    let snapshot_model = pb::ModelInfo {
        model_id: "/models/hub/snapshots/internal-path".to_string(),
        served_model_name: "zai-org/GLM-5.2-FP8".to_string(),
        served_model_aliases: vec!["glm-5.2-fp8".to_string()],
        ..Default::default()
    };

    let default_metadata = rl_worker_metadata(
        "http://worker:8120".to_string(),
        8,
        &thunderagent_model,
        None,
    )
    .expect("default");
    let overridden_metadata = rl_worker_metadata(
        "http://worker:8120".to_string(),
        8,
        &snapshot_model,
        Some("zai-org/GLM-5.2-FP8"),
    )
    .expect("served-name override");

    assert_eq!(default_metadata.model, thunderagent_model.model_id);
    assert_eq!(overridden_metadata.model, "zai-org/GLM-5.2-FP8");
    assert_eq!(
        rl_worker_metadata(
            "http://worker:8120".to_string(),
            8,
            &snapshot_model,
            Some("glm-5.2-fp8"),
        )
        .expect("advertised alias override")
        .model,
        "glm-5.2-fp8"
    );
    assert!(
        rl_worker_metadata(
            "http://worker:8120".to_string(),
            8,
            &snapshot_model,
            Some("unrelated/model"),
        )
        .is_err()
    );
}
