// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

macro_rules! sidecar_vllm_support {
    (model) => {
        use super::super::*;

        use crate::unit_vllm_fixtures::{model_info, server_info};

        pub(crate) fn engine_config() -> EngineConfig {
            DiscoveredModel::from_proto(model_info(), server_info())
                .unwrap()
                .engine_config()
        }

        pub(crate) fn without_optional_limits() -> EngineConfig {
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

        pub(crate) fn logical_block_size_and_capacity() -> EngineConfig {
            let mut server = server_info();
            server.effective_attention_block_size = Some(64);
            DiscoveredModel::from_proto(model_info(), server)
                .unwrap()
                .engine_config()
        }
    };
    (worker) => {
        use super::super::*;
        use crate::unit_vllm_fixtures::{model_info, server_info};
        use clap::Parser;
        pub(crate) fn args(mode: &str) -> Args {
            Args::try_parse_from([
                "sidecar",
                "--grpc-endpoint",
                "127.0.0.1:12345",
                "--namespace",
                "test-namespace",
                "--component",
                "configured-component",
                "--endpoint",
                "tokens",
                "--disaggregation-mode",
                mode,
                "--custom-jinja-template",
                "local-template.jinja",
            ])
            .unwrap()
        }

        pub(crate) fn worker(mode: &str) -> (VllmSidecarEngine, WorkerConfig) {
            let mut info = model_info();
            info.supports_multimodal = true;
            let model = DiscoveredModel::from_proto(info, server_info()).unwrap();
            VllmSidecarEngine::from_discovered(args(mode), model).unwrap()
        }

        pub(crate) fn is_cancelled(engine: &VllmSidecarEngine) -> bool {
            engine.cancel.is_cancelled()
        }
    };
    (requests) => {
        use super::super::*;
        pub(crate) use crate::convert::data_parallel_rank as routed_rank;

        pub(crate) fn lower_request(request: PreprocessedRequest) -> Result<(), DynamoError> {
            build_generate_request(
                request,
                "shared-request".into(),
                DisaggregationMode::Aggregated,
            )
            .map(|_| ())
        }

        pub(crate) fn selected_lora(
            request: PreprocessedRequest,
        ) -> Result<Option<String>, DynamoError> {
            build_generate_request(
                request,
                "shared-lora".into(),
                DisaggregationMode::Aggregated,
            )
            .map(|wire| Some(wire.lora_name))
        }
    };
    (responses) => {
        use super::super::*;

        pub(crate) fn convert_terminal(
            request: &PreprocessedRequest,
            reason: dynamo_backend_common::FinishReason,
        ) -> Result<LLMEngineOutput, DynamoError> {
            ResponseState::new(request, DisaggregationMode::Aggregated)
                .convert(crate::unit_vllm_fixtures::terminal_response(reason))
                .map(|output| output.expect("terminal output"))
        }

        pub(crate) fn convert_prompt_logprobs(
            request: &PreprocessedRequest,
            selected: &[f32],
            candidates: &[Vec<(u32, f32)>],
        ) -> Result<Option<serde_json::Value>, DynamoError> {
            let response = crate::unit_vllm_fixtures::prompt_logprob_response(
                &request.token_ids,
                selected,
                candidates,
            );
            ResponseState::new(request, DisaggregationMode::Aggregated)
                .convert(response)
                .map(|output| output.expect("terminal output").engine_data)
        }
    };
}
