// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use dynamo_backend_common::{
    DynamoError, EngineConfig, LlmRegistration, RlAdminBaseUrl, RlWorkerMetadata,
};

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
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveredModel {
    pub source: String,
    pub served_name: String,
    pub supports_multimodal: bool,
    identity: ModelIdentity,
    server: pb::ServerInfo,
    data_parallel_range: Range<u32>,
    kv_cache_block_size: Option<u32>,
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
        let kv_cache_block_size = server
            .effective_attention_block_size
            .filter(|&size| size != 0)
            .map(u32::try_from)
            .transpose()
            .map_err(|_| client::protocol_error("effective attention block size exceeds u32"))?
            .or_else(|| nonzero(server.kv_block_size));
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
        };
        Ok(Self {
            source,
            served_name,
            supports_multimodal: model.supports_multimodal,
            identity,
            server,
            data_parallel_range,
            kv_cache_block_size,
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

    pub(crate) fn engine_config(&self) -> EngineConfig {
        let parallelism = self.server.parallelism.as_ref();
        EngineConfig {
            model: self.source.clone(),
            served_model_name: Some(self.served_name.clone()),
            model_aliases: self.identity.aliases.clone(),
            runtime_data: [(
                dynamo_llm::lora::LORA_REQUIRES_REGISTRATION.to_string(),
                serde_json::Value::Bool(true),
            )]
            .into_iter()
            .collect(),
            llm: Some(LlmRegistration {
                context_length: nonzero(self.server.max_model_len),
                kv_cache_block_size: self.kv_cache_block_size,
                total_kv_blocks: self.total_kv_blocks_per_rank(),
                max_num_seqs: nonzero(self.server.max_running_requests),
                max_num_batched_tokens: nonzero(self.server.max_batched_tokens),
                max_gpu_lora_count: self.supports_lora().then_some(self.max_loras()),
                data_parallel_size: parallelism.map(|_| self.data_parallel_size_local()),
                data_parallel_start_rank: parallelism.map(|_| self.data_parallel_range.start),
                ..Default::default()
            }),
        }
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
mod unit_model {
    use super::*;
    use crate::unit_vllm_fixtures::{model_info, server_info};
    use serde_json::json;

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

    sidecar_test! {
        lane: pre_merge;
        #[test]
        fn missing_model_identity_is_rejected() {
            assert!(DiscoveredModel::from_proto({ let mut info = model_info(); info.model_id.clear(); info }, server_info()).is_err());
        }
    }

    sidecar_test! {
        lane: pre_merge;
        #[test]
        fn local_dp_ownership_requires_a_valid_unambiguous_range() {
            for (global, start, local, expected) in [(2, 0, 0, 0..2), (8, 0, 4, 0..4), (8, 4, 4, 4..8)] {
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

    sidecar_test! {
        lane: pre_merge;
        #[test]
        fn model_identity_and_limits_are_preserved() {
            let config = DiscoveredModel::from_proto(model_info(), server_info())
                .unwrap()
                .engine_config();
            assert_eq!(config.model, "model-source");
            assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
            let llm = config.llm.expect("LLM registration");
            assert_eq!(llm.context_length, Some(8192));
            assert_eq!(llm.kv_cache_block_size, Some(16));
            assert_eq!(llm.max_num_seqs, Some(128));
            assert_eq!(llm.max_num_batched_tokens, Some(2048));
        }
    }

    sidecar_test! {
        lane: pre_merge;
        #[test]
        fn missing_optional_limits_are_not_invented() {
            let llm = DiscoveredModel::from_proto(
                model_info(),
                pb::ServerInfo {
                    api_version: "vllm".into(),
                    ..Default::default()
                },
            )
            .unwrap()
            .engine_config()
            .llm
            .expect("LLM registration");
            assert_eq!(
                (
                    llm.context_length,
                    llm.kv_cache_block_size,
                    llm.max_num_seqs,
                    llm.max_num_batched_tokens,
                ),
                (None, None, None, None)
            );
        }
    }

    sidecar_test! {
        lane: pre_merge;
        #[test]
        fn logical_block_size_and_per_rank_capacity_are_registered() {
            let mut server = server_info();
            server.effective_attention_block_size = Some(64);
            let llm = DiscoveredModel::from_proto(model_info(), server)
                .unwrap()
                .engine_config()
                .llm
                .expect("LLM registration");
            assert_eq!(llm.kv_cache_block_size, Some(64));
            assert_eq!(llm.total_kv_blocks, Some(2048));
            assert_eq!(llm.data_parallel_size, Some(2));
            assert_eq!(llm.data_parallel_start_rank, Some(0));
        }
    }
}
