// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
