// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_vllm_fixtures::{model_info, server_info};
use serde_json::json;

mod adapter {
    use super::*;

    pub(super) fn engine_config() -> EngineConfig {
        DiscoveredModel::from_proto(model_info(), server_info())
            .unwrap()
            .engine_config()
    }

    pub(super) fn without_optional_limits() -> EngineConfig {
        DiscoveredModel::from_proto(
            model_info(),
            pb::ServerInfo {
                api_version: "vllm".into(),
                ..Default::default()
            },
        )
        .unwrap()
        .engine_config()
    }

    pub(super) fn logical_block_size_and_capacity() -> EngineConfig {
        let mut server = server_info();
        server.effective_attention_block_size = Some(64);
        DiscoveredModel::from_proto(model_info(), server)
            .unwrap()
            .engine_config()
    }

    pub(super) fn without_model_identity() -> Result<EngineConfig, DynamoError> {
        let mut model = model_info();
        model.model_id.clear();
        DiscoveredModel::from_proto(model, server_info()).map(|model| model.engine_config())
    }
}

#[path = "model.rs"]
mod shared;

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn discovery_rejects_incompatible_model_metadata() {
        let mut unsupported_api = server_info();
        unsupported_api.api_version = "unsupported".to_string();

        let mut missing_served_name = model_info();
        missing_served_name.served_model_name.clear();

        let mut unsupported_input = model_info();
        unsupported_input.supports_token_ids_input = false;

        for (case, model, server) in [
            ("unsupported API", model_info(), unsupported_api),
            ("missing served name", missing_served_name, server_info()),
            ("unsupported input", unsupported_input, server_info()),
        ] {
            assert!(
                DiscoveredModel::from_proto(model, server).is_err(),
                "{case} metadata should be rejected"
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn discovery_rejects_nonzero_dp_start_without_local_size() {
        let mut server = server_info();
        let parallelism = server.parallelism.as_mut().unwrap();
        parallelism.data_parallel_size = 8;
        parallelism.data_parallel_rank = 4;
        assert!(DiscoveredModel::from_proto(model_info(), server).is_err());
    }
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn discovered_aliases_lora_and_legacy_parallelism_are_preserved() {
        let model = DiscoveredModel::from_proto(model_info(), server_info()).unwrap();
        let config = model.engine_config();
        assert_eq!(config.model_aliases, vec!["model-alias"]);
        for name in ["model-source", "served-model", "model-alias"] {
            assert!(model.is_base_model_name(name));
        }
        assert!(!model.is_base_model_name("other-adapter"));
        assert!(model.supports_lora());
        let llm = config.llm.unwrap();
        assert_eq!(llm.max_gpu_lora_count, Some(4));
        let mut info = model_info();
        info.supports_lora = false;
        let server = pb::ServerInfo {
            api_version: "vllm".into(),
            ..Default::default()
        };
        let model = DiscoveredModel::from_proto(info, server).unwrap();
        let llm = model.engine_config().llm.unwrap();
        assert_eq!(llm.max_gpu_lora_count, None);
        assert_eq!(
            (llm.data_parallel_size, llm.data_parallel_start_rank),
            (None, None)
        );
        assert!(!model.supports_lora());
    }
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn absent_effective_attention_block_size_uses_legacy_fallback() {
        for (effective, physical, expected) in [
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
}

sidecar_test! {
    lane: pre_merge;
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
}
