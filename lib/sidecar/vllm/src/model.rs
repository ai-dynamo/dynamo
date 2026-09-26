// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use dynamo_backend_common::{
    DynamoError, EngineConfig, LlmRegistration, RlAdminBaseUrl, RlWorkerMetadata,
};
use dynamo_llm::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;

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
    supports_lora: bool,
    max_loras: u32,
    supports_multimodal: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveredModel {
    pub source: String,
    pub served_name: String,
    pub supports_multimodal: bool,
    identity: ModelIdentity,
    server: pb::ServerInfo,
    data_parallel_range: Range<u32>,
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
        let data_parallel_range = if let Some(parallelism) = server.parallelism.as_ref() {
            local_data_parallel_range(
                parallelism.data_parallel_size,
                parallelism.data_parallel_rank,
                parallelism.data_parallel_size_local,
            )?
        } else {
            0..1
        };
        let source = required("model_id", model.model_id)?;
        let served_name = required("served_model_name", model.served_model_name)?;
        if !model.supports_token_ids_input {
            return Err(client::protocol_error(
                "the discovered model does not support token-ID input",
            ));
        }
        let reasoning_parser = nonempty(model.reasoning_parser);
        let tool_call_parser = nonempty(model.tool_call_parser);
        let supports_lora = model.supports_lora;
        let max_loras = server.max_loras;
        let identity = ModelIdentity {
            source: source.clone(),
            served_name: served_name.clone(),
            aliases: model.served_model_aliases,
            reasoning_parser: reasoning_parser.clone(),
            tool_call_parser: tool_call_parser.clone(),
            supports_lora,
            max_loras,
            supports_multimodal: model.supports_multimodal,
        };
        Ok(Self {
            source,
            served_name,
            supports_multimodal: model.supports_multimodal,
            identity,
            server,
            data_parallel_range,
        })
    }

    pub(crate) fn ensure_startup_compatible(&self, observed: &Self) -> Result<(), DynamoError> {
        if self.identity != observed.identity {
            return Err(client::protocol_error(format!(
                "model identity changed between bootstrap and startup: expected {:?}, observed {:?}",
                self.identity, observed.identity
            )));
        }
        if self.server.parallelism != observed.server.parallelism {
            return Err(client::protocol_error(format!(
                "parallelism changed between bootstrap and startup: expected {:?}, observed {:?}",
                self.server.parallelism, observed.server.parallelism
            )));
        }
        if self.server.rl_capabilities != observed.server.rl_capabilities {
            return Err(client::protocol_error(format!(
                "RL capabilities changed between bootstrap and startup: expected {:?}, observed {:?}",
                self.server.rl_capabilities, observed.server.rl_capabilities
            )));
        }
        Ok(())
    }

    pub(crate) fn rl_capabilities(&self) -> Option<&pb::RlCapabilities> {
        self.server.rl_capabilities.as_ref()
    }

    pub(crate) fn rl_worker_metadata(
        &self,
        admin_base_url: Option<RlAdminBaseUrl>,
        configured_world_size: Option<u32>,
    ) -> Result<RlWorkerMetadata, DynamoError> {
        let parallelism = self.server.parallelism.as_ref().ok_or_else(|| {
            client::protocol_error("RL discovery requires vLLM parallelism metadata")
        })?;
        let tensor_parallel_size = nonzero(parallelism.tensor_parallel_size)
            .ok_or_else(|| client::protocol_error("vLLM reports a tensor-parallel size of zero"))?;
        let pipeline_parallel_size =
            nonzero(parallelism.pipeline_parallel_size).ok_or_else(|| {
                client::protocol_error("vLLM reports a pipeline-parallel size of zero")
            })?;
        let data_parallel_size = nonzero(parallelism.data_parallel_size)
            .ok_or_else(|| client::protocol_error("vLLM reports a data-parallel size of zero"))?;
        let expected_minimum_world_size = tensor_parallel_size
            .checked_mul(pipeline_parallel_size)
            .ok_or_else(|| client::protocol_error("vLLM reports an invalid RL world size"))?;
        let world_size = match u32::try_from(parallelism.world_size).ok().and_then(nonzero) {
            Some(engine_world_size) => {
                if engine_world_size % expected_minimum_world_size != 0 {
                    return Err(client::protocol_error(
                        "vLLM reports an engine world size that is not divisible by TP * PP",
                    ));
                }
                engine_world_size
                    .checked_mul(data_parallel_size)
                    .ok_or_else(|| {
                        client::protocol_error("vLLM reports an invalid RL world size")
                    })?
            }
            None if parallelism.world_size == 0 => {
                let world_size = configured_world_size.ok_or_else(|| {
                    client::invalid_argument(
                        "--vllm-rl-world-size is required when vLLM omits engine world size from gRPC metadata",
                    )
                })?;
                let expected_total_world_size = expected_minimum_world_size
                    .checked_mul(data_parallel_size)
                    .ok_or_else(|| {
                        client::protocol_error("vLLM reports an invalid RL world size")
                    })?;
                if world_size % expected_total_world_size != 0 {
                    return Err(client::invalid_argument(
                        "--vllm-rl-world-size must be divisible by TP * PP * DP",
                    ));
                }
                world_size
            }
            None => {
                return Err(client::protocol_error(
                    "vLLM reports an invalid engine world size",
                ));
            }
        };
        RlWorkerMetadata::new(world_size, admin_base_url)
            .map_err(|error| client::protocol_error(error.to_string()))
    }

    pub(crate) fn engine_config(
        &self,
        enable_kv_routing: bool,
    ) -> Result<EngineConfig, DynamoError> {
        let parallelism = self.server.parallelism.as_ref();
        let kv_cache_block_size = if enable_kv_routing {
            self.kv_cache_block_size()?
        } else {
            None
        };
        Ok(EngineConfig {
            model: self.source.clone(),
            served_model_name: Some(self.served_name.clone()),
            model_aliases: self.identity.aliases.clone(),
            runtime_data: [
                (
                    dynamo_llm::lora::LORA_REQUIRES_REGISTRATION.to_string(),
                    serde_json::Value::Bool(true),
                ),
                (
                    VLLM_INFERENCE_V1_GENERATE_CAPABILITY.to_string(),
                    serde_json::Value::Bool(true),
                ),
            ]
            .into_iter()
            .collect(),
            llm: Some(LlmRegistration {
                context_length: nonzero(self.server.max_model_len),
                kv_cache_block_size,
                total_kv_blocks: enable_kv_routing
                    .then(|| self.total_kv_blocks_per_rank())
                    .flatten(),
                max_num_seqs: nonzero(self.server.max_running_requests),
                max_num_batched_tokens: nonzero(self.server.max_batched_tokens),
                max_gpu_lora_count: self.supports_lora().then_some(self.max_loras()),
                data_parallel_size: parallelism.map(|_| self.data_parallel_size_local()),
                data_parallel_start_rank: parallelism.map(|_| self.data_parallel_range.start),
                ..Default::default()
            }),
        })
    }

    fn kv_cache_block_size(&self) -> Result<Option<u32>, DynamoError> {
        let Some(block_size) = self.server.effective_attention_block_size else {
            return Ok(nonzero(self.server.kv_block_size));
        };
        let block_size = u32::try_from(block_size)
            .ok()
            .and_then(nonzero)
            .ok_or_else(|| {
                client::protocol_error(format!(
                    "invalid effective_attention_block_size {block_size}; KV routing requires a nonzero size that fits u32"
                ))
            })?;
        Ok(Some(block_size))
    }

    pub(crate) fn data_parallel_range(&self) -> &Range<u32> {
        &self.data_parallel_range
    }

    fn data_parallel_size_local(&self) -> u32 {
        self.data_parallel_range.end - self.data_parallel_range.start
    }

    pub(crate) fn supports_lora(&self) -> bool {
        self.identity.supports_lora && self.identity.max_loras > 0
    }

    pub(crate) fn max_loras(&self) -> u32 {
        self.identity.max_loras
    }

    pub(crate) fn is_base_model_name(&self, name: &str) -> bool {
        name == self.identity.source
            || name == self.identity.served_name
            || self.identity.aliases.iter().any(|alias| alias == name)
    }

    fn total_kv_blocks_per_rank(&self) -> Option<u64> {
        let total_kv_blocks = nonzero(self.server.total_kv_blocks)?;
        let data_parallel_size = u64::from(self.data_parallel_size_local());
        // Control reports total KV blocks across the frontend's local DP engines.
        let per_rank = total_kv_blocks / data_parallel_size;

        if per_rank == 0 {
            tracing::warn!(
                total_kv_blocks,
                data_parallel_size,
                "vLLM reported fewer total KV blocks than DP ranks; publishing one block per rank"
            );
            return Some(1);
        }

        if total_kv_blocks % data_parallel_size != 0 {
            tracing::warn!(
                total_kv_blocks,
                data_parallel_size,
                per_rank,
                "vLLM aggregate KV blocks are not divisible by DP ranks; publishing floor per-rank capacity"
            );
        }

        Some(per_rank)
    }
}

// A zero local size means unknown, including metadata from older Control servers.
fn local_data_parallel_range(
    global_size: u32,
    start: u32,
    local_size: u32,
) -> Result<Range<u32>, DynamoError> {
    if global_size == 0 {
        return Err(client::protocol_error(
            "vLLM reports a data-parallel size of zero",
        ));
    }
    let local_size = match local_size {
        0 if start == 0 => {
            if global_size > 1 {
                tracing::warn!(
                    global_size,
                    "vLLM omits data_parallel_size_local; assuming this frontend hosts the entire DP group. Hybrid deployments require a vLLM build that reports local DP size to avoid registering unhosted ranks and underestimating per-rank KV capacity"
                );
            }
            global_size
        }
        0 => {
            return Err(client::protocol_error(format!(
                "vLLM reports data_parallel_rank {start} without data_parallel_size_local; hybrid rank ownership requires the local-size Control field"
            )));
        }
        size => size,
    };
    let end = start
        .checked_add(local_size)
        .filter(|&end| end <= global_size)
        .ok_or_else(|| {
            client::protocol_error(format!(
                "vLLM reports an invalid local data-parallel range: start {start}, local size {local_size}, global size {global_size}"
            ))
        })?;
    Ok(start..end)
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

#[cfg(test)]
mod tests {
    use super::local_data_parallel_range;

    // Regression: ambiguous, out-of-bounds, or overflowing ownership can advertise
    // unreachable DP ranks; validate the metadata before worker registration.
    #[test]
    fn local_dp_ownership_requires_a_valid_unambiguous_range() {
        for (global, start, local, expected) in [(2, 0, 0, 0..2), (8, 0, 4, 0..4), (8, 4, 4, 4..8)]
        {
            assert_eq!(
                local_data_parallel_range(global, start, local).unwrap(),
                expected
            );
        }
        for (global, start, local) in [
            (0, 0, 0),
            (8, 4, 0),
            (8, 0, 9),
            (8, 4, 5),
            (u32::MAX, u32::MAX - 1, 4),
        ] {
            assert!(
                local_data_parallel_range(global, start, local).is_err(),
                "invalid range: {start} + {local} of {global}"
            );
        }
    }
}
