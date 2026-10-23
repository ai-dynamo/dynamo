// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed discovery from SGLang's metadata RPCs, and the engine's view of it.
//!
//! The sidecar serves an SGLang deployment with one data-parallel rank, and
//! so one Rust-server listener. Discovery rejects a deployment with more.

use dynamo_backend_common::{DisaggregationMode, DynamoError};

use crate::client::{self, LocalKvEventSource, NodeMetadata};
use crate::proto as pb;

/// Metadata exposed by the discovery RPCs.
#[derive(Clone, Debug)]
pub(crate) struct Discovery {
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: Option<String>,
    pub max_model_len: Option<u32>,
    pub model_info: pb::GetModelInfoResponse,
    /// GetServerInfo without `deployment_info`, which moves to its own field.
    pub server_info: pb::GetServerInfoResponse,
    pub deployment_info: pb::DeploymentInfo,
}

pub(crate) fn parse_discovery(
    model: pb::GetModelInfoResponse,
    mut server_info: pb::GetServerInfoResponse,
    models: Vec<pb::ModelCard>,
) -> Result<Discovery, DynamoError> {
    let deployment_info = server_info
        .deployment_info
        .take()
        .ok_or_else(|| client::protocol_error("GetServerInfo missing deployment_info"))?;
    // Generate responses are forwarded as token deltas. Accepting cumulative
    // output here would duplicate tokens and inflate completion usage.
    if !server_info.incremental_streaming_output {
        return Err(client::invalid_arg(
            "SGLang sidecar requires incremental streaming output; restart the SGLang server \
             with --incremental-streaming-output to prevent duplicated tokens and inflated \
             completion-token counts",
        ));
    }
    let model_path = [&model.model_path, &server_info.model_path]
        .into_iter()
        .find(|path| !path.trim().is_empty())
        .cloned()
        .ok_or_else(|| {
            client::protocol_error("SGLang GetModelInfo returned an empty model_path")
        })?;
    let tokenizer_path = Some(&model.tokenizer_path)
        .filter(|path| !path.trim().is_empty())
        .unwrap_or(&model_path)
        .clone();

    let primary = models
        .iter()
        .find(|candidate| candidate.root == model_path || candidate.id == model_path)
        .or_else(|| models.first());
    let served_model_name = Some(&server_info.served_model_name)
        .filter(|name| !name.is_empty())
        .cloned()
        .or_else(|| {
            primary
                .map(|card| card.id.as_str())
                .filter(|name| !name.is_empty() && *name != model_path)
                .map(str::to_string)
        });
    // Zero is proto3's unset value; SGLang always reports a context length.
    let max_model_len = primary
        .map(|card| card.max_model_len)
        .filter(|len| *len > 0)
        .and_then(|len| u32::try_from(len).ok())
        .or(Some(server_info.max_context_length).filter(|len| *len > 0));

    Ok(Discovery {
        model_path,
        tokenizer_path,
        served_model_name,
        max_model_len,
        model_info: model,
        server_info,
        deployment_info,
    })
}

/// Discovery metadata as the engine consumes it.
///
/// A `Result` field holds a reported value that is validated only when the
/// engine resolves it, so startup fails at that point rather than during
/// discovery.
#[derive(Clone, Debug)]
pub(crate) struct EngineInfo {
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: Option<String>,
    /// Request context limit, preferring the served model card.
    pub context_length: Option<u32>,
    pub disaggregation_mode: DisaggregationMode,
    /// Reported prefill bootstrap port, validated only for prefill workers.
    pub disaggregation_bootstrap_port: Option<u64>,
    /// Server host, the preferred prefill bootstrap host when routable.
    pub host: Option<String>,
    /// Rendezvous address, whose host is the next bootstrap host candidate.
    pub dist_init_addr: Option<String>,
    /// Engine-side parser names, omitted when empty.
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    pub skip_tokenizer_init: bool,
    pub incremental_streaming_output: bool,
    /// Native HTTP port, when the server reports one.
    pub http_port: Option<Result<u16, DynamoError>>,
    pub page_size: Option<u32>,
    /// Decode context-parallel size, at least one.
    pub dcp_size: u32,
    pub max_total_num_tokens: Option<u64>,
    pub max_prefill_tokens: Option<u64>,
    pub max_num_seqs: Option<u64>,
    pub speculative_algorithm: Option<String>,
    /// Output slots each request reserves for speculative draft tokens.
    /// `None` when the reserve is unknown: an EAGLE server that reports zero
    /// predates the field, since EAGLE reserves at least one slot.
    pub num_reserved_tokens: Option<u32>,
    /// Whether SGLang truncates over-long requests instead of rejecting them.
    pub allow_auto_truncate: bool,
    /// Host-cache tokens published as native offloading capacity.
    pub native_offloading_capacity: Option<u64>,
    /// The engine's KV-event publishers, authoritative even when empty.
    pub kv_events: Result<NodeMetadata, DynamoError>,
}

pub(crate) fn disaggregation_mode(mode: &str) -> Result<DisaggregationMode, DynamoError> {
    match mode {
        "null" | "agg" | "aggregated" => Ok(DisaggregationMode::Aggregated),
        "prefill" => Ok(DisaggregationMode::Prefill),
        "decode" => Ok(DisaggregationMode::Decode),
        mode => Err(client::protocol_error(format!(
            "unsupported SGLang disaggregation_mode `{mode}`"
        ))),
    }
}

/// Same predicate as SGLang's `SpeculativeAlgorithm.is_eagle()`, which is
/// what switches its radix cache (and so its KV events) to bigram keys.
/// `NEXTN` covers older builds that don't normalize it to `EAGLE`.
pub(crate) fn sglang_eagle_enabled(speculative_algorithm: Option<&str>) -> bool {
    speculative_algorithm
        .map(str::trim)
        .is_some_and(|algorithm| {
            matches!(
                algorithm.to_ascii_uppercase().as_str(),
                "EAGLE" | "EAGLE3" | "FROZEN_KV_MTP" | "NEXTN"
            )
        })
}

pub(crate) fn is_deepseek_v4_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "DeepseekV4ForCausalLM" | "DeepseekV4ForCausalLMNextN" | "DeepseekV4ForCausalLMDSpark"
    )
}

/// Host-cache tokens SGLang allocates from `hicache_ratio` when it does not
/// report the realized capacity.
pub(crate) fn estimated_hicache_host_tokens(
    device_tokens: u64,
    page_size: u64,
    ratio: f64,
    deepseek_v4: bool,
) -> Option<u64> {
    if deepseek_v4 {
        // Compatibility with SGLang's current DSV4 host allocator. Prefer the
        // reported capacity once SGLang exposes it.
        let device_pages = device_tokens
            .checked_add(page_size.checked_sub(1)?)?
            .checked_div(page_size)?;
        let host_pages = device_pages as f64 * ratio;
        if !host_pages.is_finite() || host_pages >= u64::MAX as f64 {
            return None;
        }
        (host_pages as u64).checked_mul(page_size)
    } else {
        let unaligned_tokens = device_tokens as f64 * ratio;
        if !unaligned_tokens.is_finite() || unaligned_tokens >= u64::MAX as f64 {
            return None;
        }
        (unaligned_tokens as u64)
            .checked_div(page_size)?
            .checked_add(1)?
            .checked_mul(page_size)
    }
}

/// Capacity a hierarchical cache adds beyond device KV under its write policy.
pub(crate) fn hicache_offloading_capacity(
    device_tokens: u64,
    write_policy: &str,
    host_tokens: u64,
) -> Option<u64> {
    if host_tokens == 0 {
        return None;
    }
    match write_policy {
        "write_back" => Some(host_tokens),
        "write_through" => host_tokens
            .checked_sub(device_tokens)
            .filter(|tokens| *tokens > 0),
        _ => None,
    }
}

/// Map typed discovery to the engine's view. Fails for a deployment with more
/// than one data-parallel rank.
pub(crate) fn engine_info(discovery: Discovery) -> Result<EngineInfo, DynamoError> {
    let Discovery {
        model_path,
        tokenizer_path,
        served_model_name,
        max_model_len,
        model_info,
        server_info,
        deployment_info: deployment,
    } = discovery;
    // Attention DP and replica DP both report every rank in dp_size. Each
    // rank has its own Rust-server listener, which this sidecar cannot yet
    // serve together.
    if deployment.dp_size > 1 {
        return Err(client::invalid_arg(format!(
            "DP>1 is not supported yet: SGLang reports dp_size {} (enable_dp_attention={}), \
             but the Dynamo SGLang sidecar serves a single data-parallel rank; launch SGLang \
             with --dp-size 1 and --attn-dp-size 1 (without --enable-dp-attention); DP>1 \
             needs a later Dynamo release",
            deployment.dp_size, deployment.enable_dp_attention
        )));
    }
    // An empty string is proto3's unset value.
    let disaggregation_mode =
        disaggregation_mode(match server_info.disaggregation_mode.as_str() {
            "" => "null",
            mode => mode,
        })?;
    let non_empty = |value: &Option<String>| value.clone().filter(|entry| !entry.is_empty());

    Ok(EngineInfo {
        model_path,
        tokenizer_path,
        served_model_name,
        context_length: max_model_len,
        disaggregation_mode,
        disaggregation_bootstrap_port: deployment.disaggregation_bootstrap_port.map(u64::from),
        host: Some(server_info.host.clone()).filter(|host| !host.is_empty()),
        dist_init_addr: deployment.dist_init_addr.clone(),
        reasoning_parser: non_empty(&model_info.reasoning_parser),
        tool_call_parser: non_empty(&model_info.tool_call_parser),
        skip_tokenizer_init: server_info.skip_tokenizer_init,
        incremental_streaming_output: server_info.incremental_streaming_output,
        http_port: http_port(server_info.port),
        page_size: deployment.page_size,
        dcp_size: deployment.dcp_size.max(1),
        max_total_num_tokens: server_info.max_total_num_tokens,
        max_prefill_tokens: Some(deployment.max_prefill_tokens).filter(|tokens| *tokens > 0),
        max_num_seqs: deployment.max_running_requests,
        speculative_algorithm: deployment.speculative_algorithm.clone(),
        num_reserved_tokens: draft_reserve(
            server_info.num_reserved_tokens,
            deployment.speculative_algorithm.as_deref(),
        ),
        allow_auto_truncate: server_info.allow_auto_truncate,
        native_offloading_capacity: hicache_native_offloading_capacity(
            &server_info,
            &deployment,
            &model_info.architectures,
        ),
        kv_events: kv_events(&deployment),
    })
}

/// The reported draft-token reserve. Zero is proto3's unset value too, and
/// EAGLE reserves at least one slot, so an EAGLE server reporting zero
/// predates the field and its reserve is unknown.
fn draft_reserve(reported: u64, speculative_algorithm: Option<&str>) -> Option<u32> {
    if reported == 0 && sglang_eagle_enabled(speculative_algorithm) {
        return None;
    }
    u32::try_from(reported).ok()
}

/// Zero is proto3's unset value: the server serves no native HTTP API.
fn http_port(port: u32) -> Option<Result<u16, DynamoError>> {
    (port != 0).then(|| {
        u16::try_from(port).map_err(|_| {
            client::protocol_error(format!(
                "SGLang GetServerInfo.port must be in 1..=65535, got {port}"
            ))
        })
    })
}

/// The realized host-cache capacity when SGLang reports it, else the
/// estimate from `hicache_ratio`.
fn hicache_native_offloading_capacity(
    server_info: &pb::GetServerInfoResponse,
    deployment: &pb::DeploymentInfo,
    architectures: &[String],
) -> Option<u64> {
    let device_tokens = server_info
        .max_total_num_tokens
        .filter(|tokens| *tokens > 0)?;
    let policy = deployment.hicache_write_policy.as_deref()?;
    if !matches!(policy, "write_back" | "write_through") {
        return None;
    }
    let host_tokens = match deployment.hicache_host_total_tokens {
        Some(tokens) => tokens,
        None => {
            if !deployment.enable_hierarchical_cache
                || deployment.hicache_size != 0
                || deployment.dcp_size.max(1) > 1
            {
                return None;
            }
            let page_size = u64::from(deployment.page_size.filter(|size| *size > 0)?);
            let ratio = deployment
                .hicache_ratio
                .filter(|ratio| ratio.is_finite() && *ratio > 0.0)?;
            estimated_hicache_host_tokens(
                device_tokens,
                page_size,
                ratio,
                architectures
                    .iter()
                    .any(|architecture| is_deepseek_v4_architecture(architecture)),
            )?
        }
    };
    hicache_offloading_capacity(device_tokens, policy, host_tokens)
}

/// The engine's KV-event publishers, validated now and resolved against the
/// registration at start.
fn kv_events(deployment: &pb::DeploymentInfo) -> Result<NodeMetadata, DynamoError> {
    let metadata = NodeMetadata::new(
        deployment.node_rank,
        deployment.nnodes,
        deployment.dp_size,
        deployment.dist_init_addr.clone(),
        deployment
            .kv_event_sources
            .iter()
            .map(|source| LocalKvEventSource {
                dp_rank: source.dp_rank,
                endpoint: source.endpoint.clone(),
                topic: source.topic.clone(),
                block_size: source.block_size,
            })
            .collect(),
    )
    .map_err(|error| client::protocol_error(error.to_string()))?;
    let dp_rank = deployment.dp_rank.unwrap_or(0);
    if dp_rank >= deployment.dp_size {
        return Err(client::protocol_error(format!(
            "GetServerInfo dp_rank {dp_rank} is outside deployment DP size {}",
            deployment.dp_size
        )));
    }
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, DisaggregationMode, ErrorType};

    use super::*;

    fn server_info() -> pb::GetServerInfoResponse {
        pb::GetServerInfoResponse {
            incremental_streaming_output: true,
            deployment_info: Some(pb::DeploymentInfo {
                nnodes: 1,
                dp_size: 1,
                dcp_size: 1,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn model_info() -> pb::GetModelInfoResponse {
        pb::GetModelInfoResponse {
            model_path: "model".into(),
            ..Default::default()
        }
    }

    fn info(
        update: impl FnOnce(&mut pb::GetServerInfoResponse),
    ) -> Result<EngineInfo, DynamoError> {
        let mut server = server_info();
        update(&mut server);
        engine_info(parse_discovery(model_info(), server, Vec::new())?)
    }

    fn assert_invalid_argument(error: DynamoError, message: &str) {
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn discovery_requires_incremental_streaming() {
        let error = info(|server| server.incremental_streaming_output = false).unwrap_err();
        assert_invalid_argument(error, "--incremental-streaming-output");
    }

    #[test]
    fn dp_above_one_fails_discovery() {
        let error =
            info(|server| server.deployment_info.as_mut().unwrap().dp_size = 2).unwrap_err();
        assert_invalid_argument(error, "DP>1 is not supported yet");
    }

    #[test]
    fn typed_fields_map_to_the_engine_view() {
        let model = pb::GetModelInfoResponse {
            model_path: "model".into(),
            tokenizer_path: "tokenizer".into(),
            ..Default::default()
        };
        let mut server = server_info();
        server.disaggregation_mode = "prefill".into();
        server
            .deployment_info
            .as_mut()
            .unwrap()
            .max_running_requests = Some(256);
        let card = pb::ModelCard {
            id: "alias".into(),
            root: "model".into(),
            max_model_len: 4096,
            ..Default::default()
        };
        let info = engine_info(parse_discovery(model, server, vec![card]).unwrap()).unwrap();
        assert_eq!(info.tokenizer_path, "tokenizer");
        assert_eq!(info.served_model_name.as_deref(), Some("alias"));
        assert_eq!(info.context_length, Some(4096));
        assert_eq!(info.disaggregation_mode, DisaggregationMode::Prefill);
        assert_eq!(info.max_num_seqs, Some(256));
    }

    #[test]
    fn hicache_prefers_the_realized_host_capacity() {
        let info = info(|server| {
            server.max_total_num_tokens = Some(8192);
            let deployment = server.deployment_info.as_mut().unwrap();
            deployment.enable_hierarchical_cache = true;
            deployment.hicache_write_policy = Some("write_back".into());
            deployment.hicache_host_total_tokens = Some(20_000);
        })
        .unwrap();
        assert_eq!(info.native_offloading_capacity, Some(20_000));
    }
}
