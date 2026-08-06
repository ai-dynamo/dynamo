// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{DynamoError, EngineConfig, LlmRegistration};

use crate::client;
use crate::proto as pb;

const SUPPORTED_API_VERSION: &str = "vllm";

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelIdentity {
    source: String,
    served_name: String,
    aliases: Vec<String>,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveredModel {
    pub source: String,
    pub served_name: String,
    identity: ModelIdentity,
    server: pb::ServerInfo,
}

impl DiscoveredModel {
    pub(crate) fn from_proto(
        model: pb::ModelInfo,
        server: pb::ServerInfo,
    ) -> Result<Self, DynamoError> {
        if server.api_version != SUPPORTED_API_VERSION {
            return Err(client::protocol_error(format!(
                "unsupported Control API version `{}`; expected `{SUPPORTED_API_VERSION}`",
                server.api_version
            )));
        }
        let source = required("model_id", model.model_id)?;
        let served_name = required("served_model_name", model.served_model_name)?;
        if !model.supports_token_ids_input {
            return Err(client::protocol_error(
                "the discovered model does not support token-ID input",
            ));
        }
        let reasoning_parser = nonempty(model.reasoning_parser);
        let tool_call_parser = nonempty(model.tool_call_parser);
        let identity = ModelIdentity {
            source: source.clone(),
            served_name: served_name.clone(),
            aliases: model.served_model_aliases,
            reasoning_parser: reasoning_parser.clone(),
            tool_call_parser: tool_call_parser.clone(),
        };
        Ok(Self {
            source,
            served_name,
            identity,
            server,
        })
    }

    pub(crate) fn ensure_same_identity(&self, observed: &Self) -> Result<(), DynamoError> {
        if self.identity != observed.identity {
            return Err(client::protocol_error(format!(
                "model identity changed between bootstrap and startup: expected {:?}, observed {:?}",
                self.identity, observed.identity
            )));
        }
        Ok(())
    }

    pub(crate) fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            model: self.source.clone(),
            served_model_name: Some(self.served_name.clone()),
            model_aliases: self.identity.aliases.clone(),
            runtime_data: Default::default(),
            llm: Some(LlmRegistration {
                context_length: nonzero(self.server.max_model_len),
                kv_cache_block_size: nonzero(self.server.kv_block_size),
                total_kv_blocks: nonzero(self.server.total_kv_blocks),
                max_num_seqs: nonzero(self.server.max_running_requests),
                max_num_batched_tokens: nonzero(self.server.max_batched_tokens),
                ..Default::default()
            }),
        }
    }
}

fn required(field: &str, value: String) -> Result<String, DynamoError> {
    if value.trim().is_empty() {
        return Err(client::protocol_error(format!(
            "Control returned an empty {field}"
        )));
    }
    Ok(value)
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn nonzero<T>(value: T) -> Option<T>
where
    T: Default + PartialEq,
{
    (value != T::default()).then_some(value)
}
