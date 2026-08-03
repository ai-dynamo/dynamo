// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use dynamo_backend_common::{DisaggregationMode, DynamoError, EngineConfig, LlmRegistration};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};

use crate::client::{self, Discovery, VllmClient};
use crate::proto as pb;

const EXPECTED_API_VERSION: &str = "vllm";
const REQUIRED_CAPABILITIES: [&str; 3] = [
    "generate.sampling.v2",
    "generate.preprocessed_mm.v1",
    "generate.routed_experts.v1",
];

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BootstrapIdentity {
    instance_id: String,
    model_id: String,
    served_model_name: String,
    served_model_aliases: Vec<String>,
    reasoning_parser: String,
    tool_call_parser: String,
    parallelism: Option<pb::ParallelismInfo>,
}

impl BootstrapIdentity {
    pub(crate) fn from_discovery(discovery: &Discovery) -> Self {
        Self {
            instance_id: discovery.server.instance_id.clone(),
            model_id: discovery.model.model_id.clone(),
            served_model_name: discovery.model.served_model_name.clone(),
            served_model_aliases: discovery.model.served_model_aliases.clone(),
            reasoning_parser: discovery.model.reasoning_parser.clone(),
            tool_call_parser: discovery.model.tool_call_parser.clone(),
            parallelism: discovery.server.parallelism,
        }
    }

    pub(crate) fn validate(&self, discovery: &Discovery) -> Result<(), DynamoError> {
        let live = Self::from_discovery(discovery);
        if self != &live {
            return Err(client::invalid_argument(format!(
                "vLLM identity or topology changed after bootstrap: expected {self:?}, got {live:?}"
            )));
        }
        Ok(())
    }
}

pub(crate) fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: GrpcTransportConfig,
) -> Result<Discovery, DynamoError> {
    let endpoint = endpoint.clone();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?;
        runtime.block_on(async {
            let transport = GrpcTransportConfig {
                connections: NonZeroUsize::new(1).expect("one is non-zero"),
                ..transport
            };
            let (_, discovery) = VllmClient::connect_and_discover(&endpoint, transport).await?;
            Ok(discovery)
        })
    })
    .join()
    .map_err(|_| client::engine_shutdown("vLLM bootstrap thread panicked"))?
}

pub(crate) fn validate_discovery(discovery: &Discovery) -> Result<(), DynamoError> {
    if discovery.server.api_version != EXPECTED_API_VERSION {
        return Err(client::invalid_argument(format!(
            "vLLM API version `{}` is incompatible with expected `{EXPECTED_API_VERSION}`",
            discovery.server.api_version
        )));
    }
    for capability in REQUIRED_CAPABILITIES {
        if !discovery
            .server
            .capabilities
            .iter()
            .any(|advertised| advertised == capability)
        {
            return Err(client::invalid_argument(format!(
                "vLLM is missing required capability `{capability}`"
            )));
        }
    }
    if discovery.model.model_id.trim().is_empty() {
        return Err(client::invalid_argument(
            "vLLM GetModelInfo returned an empty model_id",
        ));
    }
    if !discovery.model.supports_token_ids_input {
        return Err(client::invalid_argument(
            "vLLM must support token-ID input for the Dynamo sidecar",
        ));
    }
    let parallelism =
        discovery.server.parallelism.as_ref().ok_or_else(|| {
            client::invalid_argument("vLLM GetServerInfo did not return parallelism")
        })?;
    if [
        parallelism.tensor_parallel_size,
        parallelism.pipeline_parallel_size,
        parallelism.data_parallel_size,
        parallelism.managed_data_parallel_size,
    ]
    .contains(&0)
    {
        return Err(client::invalid_argument(
            "vLLM tensor, pipeline, global data, and managed data parallel sizes must be positive",
        ));
    }
    Ok(())
}

pub(crate) fn inference_world_size(parallelism: &pb::ParallelismInfo) -> Result<u32, DynamoError> {
    [
        parallelism.tensor_parallel_size,
        parallelism.pipeline_parallel_size,
        parallelism.managed_data_parallel_size,
    ]
    .into_iter()
    .try_fold(1_u32, |world_size, size| {
        world_size
            .checked_mul(size)
            .ok_or_else(|| client::invalid_argument("vLLM inference world size overflow"))
    })
}

pub(crate) fn build_engine_config(
    configured_model_source: &str,
    discovery: &Discovery,
    mode: DisaggregationMode,
) -> EngineConfig {
    let server = &discovery.server;
    let model = &discovery.model;
    let parallelism = server.parallelism.as_ref().expect("validated parallelism");
    let served_model_name = nonempty(&model.served_model_name)
        .or_else(|| nonempty(&model.model_id))
        .or_else(|| Some(configured_model_source.to_string()));
    let runtime_data = if mode.is_prefill() {
        Default::default()
    } else {
        [(
            "vllm_inference_v1_generate".to_string(),
            serde_json::Value::Bool(true),
        )]
        .into_iter()
        .collect()
    };
    EngineConfig {
        model: configured_model_source.to_string(),
        served_model_name,
        runtime_data,
        llm: Some(LlmRegistration {
            context_length: (server.max_model_len != 0).then_some(server.max_model_len),
            kv_cache_block_size: (server.kv_block_size != 0).then_some(server.kv_block_size),
            total_kv_blocks: (server.total_kv_blocks != 0).then_some(server.total_kv_blocks),
            max_num_seqs: (server.max_running_requests != 0).then_some(server.max_running_requests),
            max_num_batched_tokens: (server.max_batched_tokens != 0)
                .then_some(server.max_batched_tokens),
            data_parallel_size: Some(parallelism.managed_data_parallel_size),
            data_parallel_start_rank: Some(parallelism.data_parallel_start_rank),
            bootstrap_host: None,
            bootstrap_port: None,
        }),
    }
}

pub(crate) fn nonempty(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovery() -> Discovery {
        Discovery {
            server: pb::ServerInfo {
                parallelism: Some(pb::ParallelismInfo {
                    tensor_parallel_size: 2,
                    pipeline_parallel_size: 3,
                    data_parallel_size: 8,
                    managed_data_parallel_size: 4,
                    data_parallel_start_rank: 4,
                    ..Default::default()
                }),
                max_model_len: 4096,
                kv_block_size: 16,
                total_kv_blocks: 512,
                max_running_requests: 64,
                max_batched_tokens: 8192,
                api_version: "vllm".to_string(),
                capabilities: vec![
                    "generate.sampling.v2".to_string(),
                    "generate.preprocessed_mm.v1".to_string(),
                    "generate.routed_experts.v1".to_string(),
                ],
                ..Default::default()
            },
            model: pb::ModelInfo {
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                served_model_name: "qwen".to_string(),
                supports_token_ids_input: true,
                ..Default::default()
            },
        }
    }

    #[test]
    fn world_size_uses_only_the_managed_dp_span() {
        let parallelism = discovery().server.parallelism.unwrap();
        assert_eq!(inference_world_size(&parallelism).unwrap(), 24);
    }

    #[test]
    fn discovery_rejects_missing_or_zero_topology() {
        let mut missing = discovery();
        missing.server.parallelism = None;
        assert!(validate_discovery(&missing).is_err());

        let mut zero = discovery();
        zero.server
            .parallelism
            .as_mut()
            .unwrap()
            .managed_data_parallel_size = 0;
        assert!(validate_discovery(&zero).is_err());
    }

    #[test]
    fn discovery_rejects_incompatible_wire_contracts() {
        let mut wrong_version = discovery();
        wrong_version.server.api_version = "legacy".to_string();
        assert!(validate_discovery(&wrong_version).is_err());

        for capability in [
            "generate.sampling.v2",
            "generate.preprocessed_mm.v1",
            "generate.routed_experts.v1",
        ] {
            let mut missing = discovery();
            missing
                .server
                .capabilities
                .retain(|candidate| candidate != capability);
            assert!(validate_discovery(&missing).is_err());
        }
    }

    #[test]
    fn engine_registration_preserves_local_source_and_reports_runtime_capacity() {
        let discovery = discovery();
        validate_discovery(&discovery).unwrap();
        let config = build_engine_config("/models/qwen", &discovery, DisaggregationMode::Decode);
        assert_eq!(config.model, "/models/qwen");
        assert_eq!(config.served_model_name.as_deref(), Some("qwen"));
        let llm = config.llm.unwrap();
        assert_eq!(llm.context_length, Some(4096));
        assert_eq!(llm.data_parallel_size, Some(4));
        assert_eq!(llm.data_parallel_start_rank, Some(4));
    }
}
