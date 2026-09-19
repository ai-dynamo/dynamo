// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_fixtures::*;
use serde_json::json;

#[test]
fn engine_config_advertises_supported_capabilities() {
    let model = DiscoveredModel::from_proto(model_info(), server_info()).expect("valid discovery");
    assert!(
        !model
            .engine_config()
            .runtime_data
            .contains_key("vllm_inference_v1_generate")
    );
    assert_eq!(
        model
            .engine_config()
            .runtime_data
            .get(dynamo_llm::lora::LORA_REQUIRES_REGISTRATION),
        Some(&json!(true))
    );
}

#[test]
fn rl_worker_metadata_identifies_zero_parallelism_dimensions() {
    for (dimension, expected) in [
        ("tensor", "tensor-parallel size of zero"),
        ("pipeline", "pipeline-parallel size of zero"),
    ] {
        let mut server = server_info();
        let parallelism = server.parallelism.as_mut().expect("parallelism metadata");
        match dimension {
            "tensor" => parallelism.tensor_parallel_size = 0,
            "pipeline" => parallelism.pipeline_parallel_size = 0,
            _ => unreachable!(),
        }
        let model = DiscoveredModel::from_proto(model_info(), server).expect("valid discovery");
        let error = model.rl_worker_metadata(None, None).unwrap_err();
        assert!(error.to_string().contains(expected));
    }
}

#[test]
fn discovery_rejects_zero_data_parallelism() {
    let mut server = server_info();
    server
        .parallelism
        .as_mut()
        .expect("parallelism metadata")
        .data_parallel_size = 0;

    let error = DiscoveredModel::from_proto(model_info(), server)
        .expect_err("zero data parallelism must fail discovery");

    assert!(error.to_string().contains("data-parallel size of zero"));
}

#[test]
fn startup_compatibility_rejects_parallelism_change() {
    let bootstrap = DiscoveredModel::from_proto(model_info(), server_info())
        .expect("valid bootstrap discovery");

    for dimension in ["tensor", "pipeline", "local_dp"] {
        let mut changed_server = server_info();
        let parallelism = changed_server
            .parallelism
            .as_mut()
            .expect("parallelism metadata");
        match dimension {
            "tensor" => parallelism.tensor_parallel_size += 1,
            "pipeline" => parallelism.pipeline_parallel_size += 1,
            "local_dp" => parallelism.data_parallel_size_local = 1,
            _ => unreachable!(),
        }
        let observed = DiscoveredModel::from_proto(model_info(), changed_server)
            .expect("valid startup discovery");

        assert!(
            bootstrap.ensure_startup_compatible(&observed).is_err(),
            "{dimension} parallelism change should be rejected"
        );
    }
}

#[test]
fn discovery_rejects_incompatible_model_metadata() {
    let mut unsupported_api = server_info();
    unsupported_api.api_version = "unsupported".to_string();

    let mut missing_model_id = model_info();
    missing_model_id.model_id.clear();

    let mut missing_served_name = model_info();
    missing_served_name.served_model_name.clear();

    let mut unsupported_input = model_info();
    unsupported_input.supports_token_ids_input = false;

    for (case, model, server) in [
        ("unsupported API", model_info(), unsupported_api),
        ("missing model ID", missing_model_id, server_info()),
        ("missing served name", missing_served_name, server_info()),
        ("unsupported input", unsupported_input, server_info()),
    ] {
        assert!(
            DiscoveredModel::from_proto(model, server).is_err(),
            "{case} metadata should be rejected"
        );
    }
}

#[test]
fn discovery_rejects_nonzero_dp_start_without_local_size() {
    let mut server = server_info();
    let parallelism = server.parallelism.as_mut().unwrap();
    parallelism.data_parallel_size = 8;
    parallelism.data_parallel_rank = 4;
    assert!(DiscoveredModel::from_proto(model_info(), server).is_err());
}

#[test]
fn engine_config_normalizes_total_kv_blocks_per_dp_rank() {
    let mut server = server_info();
    server
        .parallelism
        .as_mut()
        .expect("parallelism metadata")
        .data_parallel_size = 2;
    server.total_kv_blocks = 4096;

    let model =
        DiscoveredModel::from_proto(model_info(), server).expect("valid discovery metadata");
    let registration = model.engine_config().llm.expect("LLM registration");

    assert_eq!(registration.total_kv_blocks, Some(2048));
}

#[test]
fn engine_config_handles_zero_and_inexact_aggregate_kv_capacity() {
    for (aggregate_blocks, expected_per_rank_blocks) in [(0, None), (4097, Some(2048))] {
        let mut server = server_info();
        server.total_kv_blocks = aggregate_blocks;

        let model =
            DiscoveredModel::from_proto(model_info(), server).expect("valid discovery metadata");
        let registration = model.engine_config().llm.expect("LLM registration");

        assert_eq!(
            registration.total_kv_blocks, expected_per_rank_blocks,
            "aggregate blocks {aggregate_blocks}"
        );
    }
}

#[test]
fn discovered_identity_optional_metadata_and_lora_capabilities_are_preserved() {
    let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
    let config = model.engine_config();
    assert_eq!(config.model, "model-source");
    assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
    assert_eq!(config.model_aliases, vec!["model-alias"]);
    for name in ["model-source", "served-model", "model-alias"] {
        assert!(model.is_base_model_name(name));
    }
    assert!(!model.is_base_model_name("other-adapter"));
    assert!(model.supports_lora());
    let llm = config.llm.unwrap();
    assert_eq!(llm.context_length, Some(8192));
    assert_eq!(llm.kv_cache_block_size, Some(16));
    assert_eq!(llm.max_num_seqs, Some(128));
    assert_eq!(llm.max_num_batched_tokens, Some(2048));
    assert_eq!(llm.max_gpu_lora_count, Some(4));
    let mut info = model_info();
    info.supports_lora = false;
    let server = pb::ServerInfo {
        api_version: "vllm".into(),
        ..Default::default()
    };
    let model = DiscoveredModel::from_proto(info, server).unwrap();
    let llm = model.engine_config().llm.unwrap();
    assert_eq!(
        (
            llm.context_length,
            llm.kv_cache_block_size,
            llm.max_num_seqs,
            llm.max_num_batched_tokens,
            llm.max_gpu_lora_count
        ),
        (None, None, None, None, None)
    );
    assert_eq!(
        (llm.data_parallel_size, llm.data_parallel_start_rank),
        (None, None)
    );
    assert!(!model.supports_lora());
}

#[test]
fn incompatible_bootstrap_identity_and_capabilities_fail_closed() {
    let baseline = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
    let model_changes: &[fn(&mut pb::ModelInfo)] = &[
        |m| m.model_id = "changed".into(),
        |m| m.served_model_name = "changed".into(),
        |m| m.served_model_aliases.push("changed".into()),
        |m| m.reasoning_parser = "changed".into(),
        |m| m.tool_call_parser = "changed".into(),
        |m| m.supports_lora = false,
    ];
    for change in model_changes {
        let mut info = model_info();
        change(&mut info);
        let observed = DiscoveredModel::from_proto(info, server_info()).unwrap();
        assert!(
            baseline
                .ensure_startup_compatible(&observed)
                .unwrap_err()
                .to_string()
                .contains("identity changed")
        );
    }
    let mut server = server_info();
    server.rl_capabilities = None;
    let observed = DiscoveredModel::from_proto(model_info(), server).unwrap();
    assert!(
        baseline
            .ensure_startup_compatible(&observed)
            .unwrap_err()
            .to_string()
            .contains("RL capabilities changed")
    );
}

#[test]
fn engine_effective_attention_block_size_wins_with_legacy_fallback() {
    for (effective, physical, expected) in [
        (Some(64), 16, Some(64)),
        (None, 16, Some(16)),
        (Some(0), 16, Some(16)),
        (None, 0, None),
    ] {
        let mut server = server_info();
        server.effective_attention_block_size = effective;
        server.kv_block_size = physical;
        let model = DiscoveredModel::from_proto(model_info(), server).unwrap();
        assert_eq!(
            model.engine_config().llm.unwrap().kv_cache_block_size,
            expected
        );
    }
}

#[test]
fn unrepresentable_effective_block_size_is_rejected_without_truncation() {
    let mut server = server_info();
    server.effective_attention_block_size = Some(u64::from(u32::MAX) + 1);
    let error = DiscoveredModel::from_proto(model_info(), server).unwrap_err();
    assert_eq!(
        error.error_type(),
        dynamo_backend_common::ErrorType::Backend(dynamo_backend_common::BackendError::Unknown)
    );
    assert!(error.to_string().contains("effective attention block size"));
}
