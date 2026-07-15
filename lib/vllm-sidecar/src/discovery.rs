// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, EngineConfig, LlmRegistration, LoraAdapter,
};
use tokio::time::Instant;

use crate::args::TransportConfig;
use crate::client::{self, Discovery};
use crate::proto as pb;

#[derive(Clone)]
pub(crate) struct BootstrapIdentity {
    instance_id: String,
    model_id: String,
    served_model_name: String,
    reasoning_parser: String,
    tool_call_parser: String,
}

impl BootstrapIdentity {
    pub(crate) fn from_discovery(discovery: &Discovery) -> Self {
        Self {
            instance_id: discovery.server.instance_id.clone(),
            model_id: discovery.model.model_id.clone(),
            served_model_name: discovery.model.served_model_name.clone(),
            reasoning_parser: discovery.model.reasoning_parser.clone(),
            tool_call_parser: discovery.model.tool_call_parser.clone(),
        }
    }

    pub(crate) fn validate(&self, discovery: &Discovery) -> Result<(), DynamoError> {
        let live = Self::from_discovery(discovery);
        if self.instance_id != live.instance_id
            || self.model_id != live.model_id
            || self.served_model_name != live.served_model_name
            || self.reasoning_parser != live.reasoning_parser
            || self.tool_call_parser != live.tool_call_parser
        {
            return Err(client::invalid_arg(format!(
                "engine identity changed since bootstrap: model `{}` -> `{}`, served name `{}` -> `{}`",
                self.model_id, live.model_id, self.served_model_name, live.served_model_name
            )));
        }
        Ok(())
    }
}

pub(crate) fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
) -> Result<Discovery, DynamoError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?;
    rt.block_on(async {
        tokio::time::timeout(transport.deadline, async {
            let deadline = Instant::now() + transport.deadline;
            let channel = client::connect(endpoint, transport).await?;
            let mut client = pb::control_client::ControlClient::new(channel);
            client::discover(
                &mut client,
                deadline.saturating_duration_since(Instant::now()),
            )
            .await
        })
        .await
        .map_err(|_| client::cannot_connect("vLLM gRPC bootstrap timed out"))?
    })
}

pub(crate) fn nonempty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

pub(crate) fn validate_discovery(discovery: &Discovery) -> Result<(), DynamoError> {
    if discovery.server.api_version != "vllm" {
        return Err(client::invalid_arg(format!(
            "vLLM gRPC API version `{}` is not supported",
            discovery.server.api_version
        )));
    }
    if discovery.model.model_id.trim().is_empty()
        || discovery.model.served_model_name.trim().is_empty()
    {
        return Err(client::invalid_arg(
            "engine model_id and served_model_name must be non-empty",
        ));
    }
    if !discovery.model.supports_token_ids_input {
        return Err(client::invalid_arg(
            "engine must support token-ID input for Dynamo routing",
        ));
    }
    if discovery.model.supports_lora && discovery.server.max_loras == 0 {
        return Err(client::invalid_arg(
            "LoRA-capable engine reported zero adapter capacity",
        ));
    }
    Ok(())
}

pub(crate) fn component_for_mode(mode: DisaggregationMode) -> &'static str {
    match mode {
        DisaggregationMode::Prefill => "prefill",
        _ => "backend",
    }
}

pub(crate) fn build_engine_config(discovery: &Discovery) -> EngineConfig {
    let model = &discovery.model;
    let server = &discovery.server;
    let parallelism = server.parallelism.clone().unwrap_or_default();
    let served_model_name =
        (!model.served_model_name.is_empty()).then(|| model.served_model_name.clone());

    EngineConfig {
        model: model.model_id.clone(),
        served_model_name,
        runtime_data: Default::default(),
        llm: Some(LlmRegistration {
            context_length: (server.max_model_len != 0).then_some(server.max_model_len),
            kv_cache_block_size: (server.kv_block_size != 0).then_some(server.kv_block_size),
            total_kv_blocks: (server.total_kv_blocks != 0).then_some(server.total_kv_blocks),
            max_num_seqs: (server.max_running_requests != 0).then_some(server.max_running_requests),
            max_num_batched_tokens: (server.max_batched_tokens != 0)
                .then_some(server.max_batched_tokens),
            data_parallel_size: (parallelism.data_parallel_size != 0)
                .then_some(parallelism.data_parallel_size),
            data_parallel_start_rank: (parallelism.data_parallel_start_rank != 0)
                .then_some(parallelism.data_parallel_start_rank),
            supports_lora: model.supports_lora,
            max_loras: model.supports_lora.then_some(server.max_loras),
            bootstrap_host: None,
            bootstrap_port: None,
        }),
    }
}

pub(crate) fn lora_from_proto(adapter: pb::LoraAdapter) -> Result<LoraAdapter, DynamoError> {
    if adapter.lora_id <= 0
        || adapter.lora_name.trim().is_empty()
        || !std::path::Path::new(&adapter.source_path).is_absolute()
    {
        return Err(client::engine_shutdown(
            "engine returned an invalid LoRA adapter identity",
        ));
    }
    Ok(LoraAdapter {
        id: adapter.lora_id,
        name: adapter.lora_name,
        path: adapter.source_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_discovery() -> Discovery {
        Discovery {
            server: pb::ServerInfo {
                api_version: "vllm".to_string(),
                ..Default::default()
            },
            model: pb::ModelInfo {
                model_id: "model".to_string(),
                served_model_name: "served".to_string(),
                supports_token_ids_input: true,
                ..Default::default()
            },
        }
    }

    #[test]
    fn discovery_validation_fails_closed() {
        assert!(validate_discovery(&valid_discovery()).is_ok());

        let mut discovery = valid_discovery();
        discovery.server.api_version = "other".to_string();
        assert!(validate_discovery(&discovery).is_err());

        let mut discovery = valid_discovery();
        discovery.model.supports_token_ids_input = false;
        assert!(validate_discovery(&discovery).is_err());
    }

    #[test]
    fn lora_identity_requires_positive_id_name_and_absolute_path() {
        let valid = pb::LoraAdapter {
            lora_id: 1,
            lora_name: "adapter".to_string(),
            source_path: "/models/adapter".to_string(),
        };
        assert!(lora_from_proto(valid.clone()).is_ok());
        assert!(
            lora_from_proto(pb::LoraAdapter {
                lora_id: 0,
                ..valid.clone()
            })
            .is_err()
        );
        assert!(
            lora_from_proto(pb::LoraAdapter {
                source_path: "relative".to_string(),
                ..valid
            })
            .is_err()
        );
    }
}
