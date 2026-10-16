// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM's token-in/token-out `POST /inference/v1/generate`, as a generate format.
//!
//! The request stays opaque in `extra_args.vllm_tito` apart from the routing
//! controls Dynamo projects, including multimodal KV routing hashes. The
//! unary response is assembled from the worker's typed outputs. Streaming and
//! migration are not supported yet.

use std::sync::Arc;

use axum::http::StatusCode;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use dynamo_runtime::error::DynamoError;
use serde::Serialize;
use serde_json::{Value, json};

use super::generate::{
    GenerateAggregator, GenerateRequest, GenerateResponseOptions, SamplingParams, StreamOptions,
};
use crate::generate_format::{
    ClientError, ClientRequest, GenerateExchange, GenerateFormat, Inspected, Opened, Phase, Worker,
};
use crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::preprocessed_mm_identifier;
use crate::protocols::common::preprocessor::MmRoutingInfo;
use crate::protocols::common::timing::RequestTracker;

pub const NAME: &str = "vllm";
const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

/// Convert vLLM's lower-is-higher priority to Dynamo's higher-is-higher scale.
fn dynamo_routing_priority(vllm_priority: i32) -> i32 {
    vllm_priority.saturating_neg()
}

/// Borrowed worker envelope for vLLM-specific request fields.
///
/// `token_ids` are intentionally absent: `PreprocessedRequest.token_ids` is
/// the canonical routing and wire representation, and the worker reconstructs
/// the vLLM request from that field.
#[derive(Serialize)]
struct VllmTitoEnvelope<'a> {
    request_id: &'a str,
    sampling_params: &'a SamplingParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<&'a StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_salt: Option<&'a str>,
    priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_transfer_params: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[serde(flatten)]
    passthrough: &'a serde_json::Map<String, serde_json::Value>,
}

impl<'a> VllmTitoEnvelope<'a> {
    fn new(request: &'a GenerateRequest, request_id: &'a str) -> Self {
        let GenerateRequest {
            request_id: _,
            token_ids: _,
            sampling_params,
            model,
            stream,
            stream_options,
            cache_salt,
            priority,
            kv_transfer_params,
            passthrough,
        } = request;
        Self {
            request_id,
            sampling_params,
            model: model.as_deref(),
            stream: *stream,
            stream_options: stream_options.as_ref(),
            cache_salt: cache_salt.as_deref(),
            priority: *priority,
            kv_transfer_params: kv_transfer_params.as_ref(),
            passthrough,
        }
    }
}

type MmPlaceholderRange = (usize, usize, u64, Option<Vec<bool>>);

#[derive(Debug)]
struct GenerateMmRoutingProjection {
    info: MmRoutingInfo,
    /// Frontend-approved hashes in the marker form understood by the KV-event
    /// decoder. The worker applies these only to the vLLM prompt it builds;
    /// the caller's opaque `features` payload remains unchanged.
    marked_image_hashes: Vec<String>,
}

struct GenerateRoutingMetadata {
    kv_cache_block_size: u32,
    tower_connector_lora_enabled: bool,
    lora_name: Option<String>,
}

#[inline]
fn intersecting_mm_ranges<'a>(
    ranges: &'a [MmPlaceholderRange],
    block_start: usize,
    block_end: usize,
    first_intersecting: &mut usize,
) -> &'a [MmPlaceholderRange] {
    while *first_intersecting < ranges.len() {
        if ranges[*first_intersecting].1 > block_start {
            break;
        }
        *first_intersecting += 1;
    }

    let start = *first_intersecting;
    let mut end = start;
    while end < ranges.len() {
        if ranges[end].0 >= block_end {
            break;
        }
        end += 1;
    }

    &ranges[start..end]
}

/// Build the routing-only token sequence used by vLLM KV events for multimodal
/// prompts. The caller-provided `features` object remains opaque to execution;
/// this projection derives content-bound identities from the inline kwargs and
/// combines them with placeholder ranges so request-side KV hashes match
/// worker-side event hashes.
fn generate_mm_routing_info(
    request: &GenerateRequest,
    kv_cache_block_size: u32,
) -> Result<Option<GenerateMmRoutingProjection>, &'static str> {
    let Some(features) = request.passthrough.get("features") else {
        return Ok(None);
    };
    if features.is_null() {
        return Ok(None);
    }

    let features = features
        .as_object()
        .ok_or("features must be a JSON object")?;
    let Some(mm_hashes) = features.get("mm_hashes") else {
        return Ok(None);
    };
    let mm_hashes = mm_hashes
        .as_object()
        .ok_or("features.mm_hashes must be a JSON object")?;
    let mm_placeholders = features
        .get("mm_placeholders")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.mm_placeholders must be a JSON object")?;
    let kwargs_data = features
        .get("kwargs_data")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.kwargs_data must be a JSON object")?;

    if mm_hashes
        .keys()
        .chain(mm_placeholders.keys())
        .chain(kwargs_data.keys())
        .any(|modality| modality != "image")
    {
        return Err("exact /generate MM routing currently supports image placeholders only");
    }
    if kv_cache_block_size == 0 {
        return Err("KV cache block size must be non-zero");
    }

    let (hashes, placeholders, kwargs) = match (
        mm_hashes.get("image"),
        mm_placeholders.get("image"),
        kwargs_data.get("image"),
    ) {
        (None, None, None) => return Ok(None),
        (Some(hashes), Some(placeholders), Some(kwargs)) => (
            hashes
                .as_array()
                .ok_or("features.mm_hashes.image must be an array")?,
            placeholders
                .as_array()
                .ok_or("features.mm_placeholders.image must be an array")?,
            kwargs
                .as_array()
                .ok_or("features.kwargs_data.image must be an array")?,
        ),
        _ => return Err("image hashes, placeholders, and kwargs_data must all be present"),
    };
    if hashes.len() != placeholders.len() || hashes.len() != kwargs.len() {
        return Err("image hashes, placeholders, and kwargs_data must have equal lengths");
    }

    let mut ranges: Vec<MmPlaceholderRange> = Vec::with_capacity(hashes.len());
    for ((producer_hash, placeholder), encoded_kwargs) in
        hashes.iter().zip(placeholders).zip(kwargs)
    {
        producer_hash
            .as_str()
            .filter(|hash| !hash.is_empty())
            .ok_or("multimodal hashes must be non-empty strings")?;
        let raw_kwargs = BASE64_STANDARD
            .decode(
                encoded_kwargs
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .ok_or("multimodal kwargs_data must contain non-empty base64 strings")?,
            )
            .map_err(|_| "multimodal kwargs_data must contain valid base64")?;
        let identifier = preprocessed_mm_identifier("image", &raw_kwargs);
        let hash = dynamo_kv_router::protocols::hash_mm_identifier(&identifier)
            .ok_or("content-derived multimodal identifier must not be empty")?;
        let placeholder = placeholder
            .as_object()
            .ok_or("multimodal placeholders must be JSON objects")?;
        let offset = placeholder
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("multimodal placeholder offsets must be non-negative integers")?;
        let length = placeholder
            .get("length")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or("multimodal placeholder lengths must be positive integers")?;
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= request.token_ids.len())
            .ok_or("multimodal placeholder range exceeds token_ids")?;
        let is_embed = match placeholder.get("is_embed") {
            None | Some(serde_json::Value::Null) => {
                // vLLM 0.24 render responses omit sparse masks. A uniform
                // placeholder span is safely dense; a mixed span is ambiguous
                // and must retain token-only routing rather than over-substitute.
                if request.token_ids[offset..end]
                    .windows(2)
                    .any(|pair| pair[0] != pair[1])
                {
                    return Err("mixed multimodal placeholder spans require is_embed");
                }
                None
            }
            Some(value) => {
                let mask = value
                    .as_array()
                    .ok_or("multimodal placeholder is_embed must be an array")?;
                if mask.len() != length {
                    return Err(
                        "multimodal placeholder is_embed length must match placeholder length",
                    );
                }
                let mut parsed = Vec::with_capacity(mask.len());
                for entry in mask {
                    parsed.push(
                        entry
                            .as_bool()
                            .ok_or("multimodal placeholder is_embed entries must be booleans")?,
                    );
                }
                Some(parsed)
            }
        };
        ranges.push((offset, end, hash, is_embed));
    }

    if ranges.is_empty() {
        return Ok(None);
    }

    if ranges.windows(2).any(|pair| pair[0].0 > pair[1].0) {
        return Err("multimodal placeholders must be ordered by offset");
    }
    for pair in ranges.windows(2) {
        let (_, previous_end, previous_hash, _) = &pair[0];
        let (next_offset, _, next_hash, _) = &pair[1];
        if previous_end > next_offset {
            return Err("multimodal placeholder ranges must not overlap");
        }
        if previous_end == next_offset && previous_hash != next_hash {
            return Err("adjacent multimodal placeholders must share an identifier");
        }
    }

    // The worker discovers multimodal runs by scanning for its resolved image
    // token. Infer that token from the declared embed positions and require the
    // declarations to cover every occurrence. Otherwise an undeclared image
    // token can shift the worker's run-to-object alignment away from the
    // request-side projection.
    let mut image_token_id = None;
    let mut declared_image_tokens = 0;
    for (offset, end, _, is_embed) in &ranges {
        for position in *offset..*end {
            let should_embed = is_embed
                .as_ref()
                .is_none_or(|mask| mask[position - *offset]);
            if !should_embed {
                continue;
            }

            let token_id = request.token_ids[position];
            if image_token_id.is_some_and(|expected| expected != token_id) {
                return Err("multimodal embed positions must share an image token");
            }
            image_token_id = Some(token_id);
            declared_image_tokens += 1;
        }
    }
    let image_token_id = image_token_id.ok_or("multimodal placeholders contain no embed tokens")?;
    if request
        .token_ids
        .iter()
        .filter(|token_id| **token_id == image_token_id)
        .count()
        != declared_image_tokens
    {
        return Err("image tokens must be covered by multimodal placeholder ranges");
    }

    // Apply the same per-block run-order normalization as vLLM KV events, then
    // compare it with the renderer-declared positions. A sparse mask can split
    // one object into multiple runs, so exact routing is enabled only when the
    // shared worker normalization produces the same token projection.
    let block_size = kv_cache_block_size as usize;
    let mut first_intersecting = 0;
    let mut worker_objects = Vec::new();
    let mut routing_token_ids = Vec::with_capacity(request.token_ids.len());
    for block_start in (0..request.token_ids.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(request.token_ids.len());
        let block_tokens = &request.token_ids[block_start..block_end];
        let block_ranges =
            intersecting_mm_ranges(&ranges, block_start, block_end, &mut first_intersecting);
        if block_ranges.is_empty() {
            routing_token_ids.extend_from_slice(block_tokens);
            continue;
        }

        worker_objects.clear();
        let mut expected_tokens = block_tokens.to_vec();
        for (offset, end, hash, is_embed) in block_ranges {
            let intersection_start = (*offset).max(block_start);
            let intersection_end = (*end).min(block_end);
            worker_objects.push(*hash);
            for global_position in intersection_start..intersection_end {
                let should_embed = is_embed
                    .as_ref()
                    .is_none_or(|mask| mask[global_position - *offset]);
                if should_embed {
                    expected_tokens[global_position - block_start] =
                        dynamo_kv_router::protocols::pad_value_for_mm_hash(*hash);
                }
            }
        }

        let normalized_tokens = dynamo_kv_router::zmq_wire::normalize_mm_token_runs(
            block_tokens,
            image_token_id,
            &worker_objects,
        )
        .map(|(tokens, _)| tokens)
        .ok_or("multimodal block must contain a routing hash")?;
        if normalized_tokens != expected_tokens {
            return Err("sparse multimodal layout cannot be normalized exactly by worker events");
        }
        routing_token_ids.extend(normalized_tokens);
    }

    let padded_len = routing_token_ids
        .len()
        .div_ceil(block_size)
        .checked_mul(block_size)
        .ok_or("multimodal routing token length overflow")?;
    routing_token_ids.resize(padded_len, 0);

    Ok(Some(GenerateMmRoutingProjection {
        info: MmRoutingInfo {
            routing_token_ids,
            // vLLM events are normalized to the same pad-value token scheme, so
            // MM identity is already present in the alternate routing tokens.
            block_mm_infos: Vec::new(),
            expanded_prompt_len: request.token_ids.len(),
        },
        marked_image_hashes: ranges
            .iter()
            .map(|(_, _, hash, _)| dynamo_kv_router::zmq_wire::mark_mm_hash_for_extra_key(*hash))
            .collect(),
    }))
}

fn preprocessed_from_generate_with_tracker(
    request: GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    request_id: &str,
    routing_metadata: GenerateRoutingMetadata,
    tracker: Arc<RequestTracker>,
) -> anyhow::Result<PreprocessedRequest> {
    let GenerateRoutingMetadata {
        kv_cache_block_size,
        tower_connector_lora_enabled,
        lora_name,
    } = routing_metadata;
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let stop_conditions = sampling.project_stop_conditions();
    let sampling_options = sampling
        .project_sampling_options()
        .map_err(anyhow::Error::msg)?;
    let output_options = sampling
        .project_output_options()
        .map_err(anyhow::Error::msg)?;
    let routing_priority = dynamo_routing_priority(request.priority);
    // The sidecar protocol does not expose whether tower/connector LoRA is
    // active, so conservatively treat every adapter's MM identity as scoped.
    let mm_routing = if lora_name.is_some() {
        tracing::debug!(
            target: "mm_routing",
            tower_connector_lora_enabled,
            "LoRA request uses token-only multimodal routing"
        );
        None
    } else {
        match generate_mm_routing_info(&request, kv_cache_block_size) {
            Ok(info) => info,
            Err(reason) => {
                tracing::debug!(
                    target: "mm_routing",
                    reason,
                    "invalid /generate multimodal routing metadata; using token-only routing"
                );
                None
            }
        }
    };
    let vllm_tito = serde_json::to_value(VllmTitoEnvelope::new(&request, request_id))?;
    let mut extra_args = serde_json::Map::new();
    extra_args.insert("vllm_tito".to_string(), vllm_tito);
    if let Some(kv_transfer_params) = request.kv_transfer_params.as_ref() {
        extra_args.insert(
            "kv_transfer_params".to_string(),
            serde_json::Value::Object(kv_transfer_params.clone()),
        );
    }
    if let Some(skip_reading_prefix_cache) = sampling.skip_reading_prefix_cache() {
        extra_args.insert(
            "skip_reading_prefix_cache".to_string(),
            serde_json::Value::Bool(skip_reading_prefix_cache),
        );
    }
    if let Some(projection) = &mm_routing {
        extra_args.insert(
            "dynamo_mm_routing_hashes".to_string(),
            serde_json::to_value(&projection.marked_image_hashes)?,
        );
    }
    let mm_routing_info = mm_routing.map(|projection| projection.info);
    let GenerateRequest {
        token_ids,
        cache_salt,
        ..
    } = request;

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(token_ids)
        .stop_conditions(stop_conditions)
        .sampling_options(sampling_options)
        .output_options(output_options)
        .mm_routing_info(mm_routing_info)
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            expected_output_tokens: max_tokens,
            lora_name,
            cache_namespace: cache_salt,
            // `priority_jump` is a boost-only scheduler input. Preserve penalties
            // in signed `priority`, matching the standard preprocessor projection.
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        // Do not copy token_ids into this envelope. The worker must rebuild
        // that field from PreprocessedRequest.token_ids after routing.
        .extra_args(Some(serde_json::Value::Object(extra_args)))
        .tracker(Some(tracker))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

/// vLLM's `/inference/v1/generate`.
pub struct VllmFormat;

fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        404 => "not_found",
        429 => "deadline_exceeded",
        499 => "request_cancelled",
        501 => "not_implemented",
        503 => "service_unavailable",
        _ => "internal_error",
    }
}

impl GenerateFormat for VllmFormat {
    fn name(&self) -> &str {
        NAME
    }

    fn capability(&self) -> &'static str {
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY
    }

    fn open(&self, client: ClientRequest<'_>) -> Result<Opened, ClientError> {
        let fail = |status: StatusCode, message: String| ClientError {
            body: Some(self.error_body(status, &message)),
            status,
            message,
        };
        let request: GenerateRequest = serde_json::from_slice(client.body).map_err(|error| {
            let message = format!("Failed to deserialize the JSON body: {error}");
            fail(StatusCode::BAD_REQUEST, message)
        })?;
        request
            .validate()
            .map_err(|message| fail(StatusCode::BAD_REQUEST, message))?;
        if request.stream {
            let message =
                "streaming (stream=true) is not implemented for /inference/v1/generate yet";
            return Err(fail(StatusCode::NOT_IMPLEMENTED, message.to_string()));
        }
        // vLLM keeps request metadata in headers first, then the body.
        let header = |name| client.headers.get(name).and_then(|v| v.to_str().ok());
        let request_id = header(X_REQUEST_ID_HEADER)
            .map(ToOwned::to_owned)
            .or_else(|| request.request_id.clone())
            .unwrap_or_else(|| client.default_request_id.to_string());
        let data_parallel_rank =
            header(X_DATA_PARALLEL_RANK_HEADER).and_then(|v| v.trim().parse().ok());
        let placeholder = PreprocessedRequest::builder()
            .model(String::new())
            .token_ids(Vec::new())
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .map_err(|error| ClientError::new(error.to_string()))?;
        Ok(Opened {
            model: request.model.clone(),
            request: placeholder,
            request_id: request_id.clone(),
            stream: false,
            exchange: Box::new(VllmExchange {
                aggregator: GenerateAggregator::new(request_id.clone()),
                options: request.response_options(),
                request: Some(request),
                request_id,
                data_parallel_rank,
            }),
        })
    }

    fn inspect(&self, _response: Value, _phase: Phase) -> Result<Inspected, DynamoError> {
        Err(DynamoError::msg("vLLM workers produce typed outputs"))
    }

    fn error_body(&self, status: StatusCode, message: &str) -> Value {
        json!({"error": {"message": message, "type": error_type(status), "code": status.as_u16()}})
    }
}

struct VllmExchange {
    /// The client request, until it is bound to a worker set.
    request: Option<GenerateRequest>,
    request_id: String,
    data_parallel_rank: Option<u32>,
    aggregator: GenerateAggregator,
    options: GenerateResponseOptions,
}

impl GenerateExchange for VllmExchange {
    fn bind(
        &mut self,
        request: &mut PreprocessedRequest,
        worker: &Worker<'_>,
    ) -> Result<Result<(), String>, ClientError> {
        if let Some(generate) = self.request.take() {
            let metadata = GenerateRoutingMetadata {
                kv_cache_block_size: worker.kv_cache_block_size,
                tower_connector_lora_enabled: worker.tower_connector_lora_enabled,
                lora_name: worker.lora_name.map(ToOwned::to_owned),
            };
            let tracker = Arc::new(RequestTracker::new());
            *request = preprocessed_from_generate_with_tracker(
                generate,
                &request.model,
                self.data_parallel_rank,
                &self.request_id,
                metadata,
                tracker,
            )
            .map_err(|error| ClientError::new(error.to_string()))?;
        }
        Ok(Err("vLLM generate requests do not migrate yet".to_string()))
    }

    fn chunk(
        &mut self,
        output: LLMEngineOutput,
        _prefill: bool,
        _emit: &mut dyn FnMut(&Value),
    ) -> Result<(), DynamoError> {
        self.aggregator
            .apply_output(output, self.options)
            .map_err(|error| DynamoError::msg(error.to_string()))
    }

    fn migrate(
        &mut self,
        _delivered_tokens: usize,
        _request: &mut PreprocessedRequest,
    ) -> Result<Option<LLMEngineOutput>, String> {
        Err("vLLM generate requests do not migrate yet".to_string())
    }

    fn finish(&mut self) -> Result<Option<Value>, DynamoError> {
        let aggregator =
            std::mem::replace(&mut self.aggregator, GenerateAggregator::new(String::new()));
        let response = aggregator
            .finish(self.options)
            .map_err(|error| DynamoError::msg(error.to_string()))?;
        if !response.is_complete_unary() {
            return Err(DynamoError::msg(
                "generate stream ended without a complete choice",
            ));
        }
        serde_json::to_value(response)
            .map(Some)
            .map_err(|error| DynamoError::msg(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    fn preprocessed_from_generate(
        request: GenerateRequest,
        model: &str,
        data_parallel_rank: Option<u32>,
        request_id: &str,
        routing_metadata: GenerateRoutingMetadata,
    ) -> anyhow::Result<PreprocessedRequest> {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(request.token_ids.len(), None);
        preprocessed_from_generate_with_tracker(
            request,
            model,
            data_parallel_rank,
            request_id,
            routing_metadata,
            tracker,
        )
    }

    fn routing_metadata(
        kv_cache_block_size: u32,
        tower_connector_lora_enabled: bool,
        lora_name: Option<&str>,
    ) -> GenerateRoutingMetadata {
        GenerateRoutingMetadata {
            kv_cache_block_size,
            tower_connector_lora_enabled,
            lora_name: lora_name.map(str::to_string),
        }
    }

    #[test]
    fn engine_fields_reach_envelope_with_resolved_id_and_cache_namespace() {
        let raw = serde_json::json!({
            "request_id": "req-forward",
            "token_ids": [1, 2],
            "sampling_params": {
                "max_tokens": 8,
                "future_sampling_field": {"nested": true}
            },
            "model": "test-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_salt": "tenant-a",
            "features": {"future_feature": [1, 2, 3]},
            "priority": 7,
            "kv_transfer_params": {"remote": "worker-a"},
            "future_top_level_field": {"anything": "works"}
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(16, false, None),
        )
        .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            Some(8)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority),
            Some(-7),
            "vLLM lower-is-higher priority must be inverted for Dynamo routing"
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority_jump),
            Some(0.0)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.cache_namespace.as_deref()),
            Some("tenant-a")
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");

        let mut expected_envelope = raw;
        expected_envelope["request_id"] = serde_json::json!("resolved-request");
        let expected_token_ids = expected_envelope
            .as_object_mut()
            .and_then(|object| object.remove("token_ids"))
            .expect("token_ids in client request");
        assert_eq!(preprocessed.token_ids.as_slice(), &[1, 2]);
        assert_eq!(
            preprocessed
                .tracker
                .as_ref()
                .and_then(|tracker| tracker.isl_tokens()),
            Some(2)
        );
        assert_eq!(expected_token_ids, serde_json::json!([1, 2]));
        assert_eq!(envelope, &expected_envelope);
        assert!(envelope.get("token_ids").is_none());
    }

    #[test]
    fn multimodal_routing_matches_worker_events_and_preserves_execution_payload() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let raw = serde_json::json!({
            "token_ids": [10, 11, 12, 12, 12, 15, 16, 12, 12, 19],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [hash_a, hash_b]},
                "mm_placeholders": {"image": [
                    {"offset": 2, "length": 3},
                    {"offset": 7, "length": 2}
                ]},
                "kwargs_data": {"image": ["b3BhcXVlLWE=", "b3BhcXVlLWI="]}
            }
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(4, false, None),
        )
        .expect("build request");

        let hash_a = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
            "image",
            b"opaque-a",
        ))
        .expect("content-derived hash");
        let hash_b = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
            "image",
            b"opaque-b",
        ))
        .expect("content-derived hash");
        let pad_a = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash_a);
        let pad_b = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash_b);
        let mm = preprocessed
            .mm_routing_info
            .as_ref()
            .expect("multimodal routing projection");
        assert_eq!(
            mm.routing_token_ids,
            vec![10, 11, pad_a, pad_a, pad_a, 15, 16, pad_b, pad_b, 19, 0, 0]
        );
        assert!(mm.block_mm_infos.is_empty());
        assert_eq!(mm.expanded_prompt_len, 10);

        assert_eq!(
            preprocessed.token_ids.as_slice(),
            &[10, 11, 12, 12, 12, 15, 16, 12, 12, 19]
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
        assert_eq!(
            preprocessed
                .extra_args
                .as_ref()
                .and_then(|extra| extra.get("dynamo_mm_routing_hashes")),
            Some(&serde_json::json!([
                crate::protocols::common::preprocessed_mm_routing_hash("image", b"opaque-a"),
                crate::protocols::common::preprocessed_mm_routing_hash("image", b"opaque-b")
            ]))
        );

        // A frontend-approved, marker-form hash must produce the same KV hash
        // on the request and event paths.
        let mm_identifier = "1234567890abcdef".repeat(4);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier.clone()]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["cm91dGluZy1h"]}
            }
        }))
        .expect("deserialize request");
        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(4, false, None),
        )
        .expect("build request");
        let routing = preprocessed
            .mm_routing_info
            .as_ref()
            .expect("base request keeps exact MM routing");
        let request_hashes = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &routing.routing_token_ids,
            4,
            dynamo_kv_router::protocols::BlockHashOptions::default(),
        );
        let marked_identifier = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("dynamo_mm_routing_hashes"))
            .and_then(serde_json::Value::as_array)
            .and_then(|hashes| hashes.first())
            .and_then(serde_json::Value::as_str)
            .expect("frontend-approved marked MM hash")
            .to_string();
        let event_mm_info =
            dynamo_kv_router::zmq_wire::extra_keys_to_block_mm_infos(Some(vec![Some(vec![
                dynamo_kv_router::zmq_wire::ExtraKeyItem::HashWithUnsignedOffset((
                    marked_identifier,
                    1,
                )),
            ])]))
            .expect("parse canonical offset-bearing MM extra key")
            .into_iter()
            .next()
            .flatten()
            .expect("block MM metadata");
        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            4,
            7,
            &[10, 99, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                mm_extra_info: Some(event_mm_info),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hashes[0], event_block.tokens_hash);
    }

    #[test]
    fn sparse_and_ambiguous_multimodal_layouts_are_handled() {
        let mm_identifier = "opaque-renderer-image-0";
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier]},
                "mm_placeholders": {"image": [{
                    "offset": 1,
                    "length": 3,
                    "is_embed": [true, false, true]
                }]},
                "kwargs_data": {"image": ["c3BhcnNl"]}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 5)
            .expect("valid sparse MM routing metadata")
            .expect("MM routing projection");
        let mm_hash = dynamo_kv_router::protocols::hash_mm_identifier(&preprocessed_mm_identifier(
            "image", b"sparse",
        ))
        .expect("content-derived identifier");
        let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(mm_hash);
        assert_eq!(routing.info.routing_token_ids, vec![10, pad, 42, pad, 20]);

        for (name, token_ids, features, expected_error) in [
            (
                "missing sparse mask",
                serde_json::json!([10, 99, 42, 99, 20]),
                serde_json::json!({
                    "mm_hashes": {"image": ["image-0"]},
                    "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
                    "kwargs_data": {"image": ["c3BhcnNl"]}
                }),
                "mixed multimodal placeholder spans require is_embed",
            ),
            (
                "undeclared image token",
                serde_json::json!([99, 10, 99, 99, 20]),
                serde_json::json!({
                    "mm_hashes": {"image": ["image-0"]},
                    "mm_placeholders": {"image": [{"offset": 2, "length": 2}]},
                    "kwargs_data": {"image": ["c3BhcnNl"]}
                }),
                "image tokens must be covered by multimodal placeholder ranges",
            ),
            (
                "ambiguous sparse object order",
                serde_json::json!([10, 99, 42, 99, 20, 99, 30]),
                serde_json::json!({
                    "mm_hashes": {"image": ["image-a", "image-b"]},
                    "mm_placeholders": {"image": [
                        {"offset": 1, "length": 3, "is_embed": [true, false, true]},
                        {"offset": 5, "length": 1}
                    ]},
                    "kwargs_data": {"image": ["c3BhcnNlLWE=", "c3BhcnNlLWI="]}
                }),
                "sparse multimodal layout cannot be normalized exactly by worker events",
            ),
        ] {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "token_ids": token_ids,
                "sampling_params": {},
                "features": features,
            }))
            .expect("deserialize request");

            assert_eq!(
                generate_mm_routing_info(&request, 7)
                    .expect_err("ambiguous layout must disable exact routing"),
                expected_error,
                "{name}"
            );
        }
    }

    #[test]
    fn invalid_or_unsupported_multimodal_metadata_falls_back_safely() {
        for (name, token_ids, features) in [
            (
                "unrelated features",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({"future_feature": [1, 2, 3]}),
            ),
            (
                "malformed hashes",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({"mm_hashes": ["not-an-object"]}),
            ),
            (
                "empty identifier",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({
                    "mm_hashes": {"image": [""]},
                    "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                    "kwargs_data": {"image": ["opaque"]}
                }),
            ),
            (
                "overlapping placeholders",
                serde_json::json!([9, 9, 9, 4]),
                serde_json::json!({
                    "mm_hashes": {"image": ["a".repeat(64), "b".repeat(64)]},
                    "mm_placeholders": {"image": [
                        {"offset": 0, "length": 2},
                        {"offset": 1, "length": 2}
                    ]}
                }),
            ),
            (
                "unsupported modality",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({
                    "mm_hashes": {"audio": ["audio-0"]},
                    "mm_placeholders": {"audio": [{"offset": 1, "length": 2}]}
                }),
            ),
            (
                "unpaired image metadata",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({
                    "mm_hashes": {"image": ["image-0"]},
                    "mm_placeholders": {}
                }),
            ),
            (
                "unequal item counts",
                serde_json::json!([1, 2, 3, 4]),
                serde_json::json!({
                    "mm_hashes": {"image": ["image-0", "image-1"]},
                    "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
                }),
            ),
            (
                "out-of-order placeholders",
                serde_json::json!([99, 99, 3, 99]),
                serde_json::json!({
                    "mm_hashes": {"image": ["late", "early"]},
                    "mm_placeholders": {"image": [
                        {"offset": 3, "length": 1},
                        {"offset": 0, "length": 2}
                    ]}
                }),
            ),
        ] {
            let raw = serde_json::json!({
                "token_ids": token_ids,
                "sampling_params": {},
                "features": features,
            });
            let request: GenerateRequest =
                serde_json::from_value(raw.clone()).expect("deserialize request");
            let expected_token_ids = request.token_ids.clone();

            let preprocessed = preprocessed_from_generate(
                request,
                "test-model",
                None,
                "resolved-request",
                routing_metadata(4, false, None),
            )
            .expect("invalid routing metadata must not reject execution");

            assert!(preprocessed.mm_routing_info.is_none(), "{name}");
            assert_eq!(
                preprocessed.token_ids.as_slice(),
                expected_token_ids,
                "{name}"
            );
            let envelope = preprocessed
                .extra_args
                .as_ref()
                .and_then(|extra| extra.get("vllm_tito"))
                .expect("vllm_tito envelope");
            assert_eq!(envelope["features"], raw["features"], "{name}");
            assert!(
                preprocessed
                    .extra_args
                    .as_ref()
                    .and_then(|extra| extra.get("dynamo_mm_routing_hashes"))
                    .is_none(),
                "{name}"
            );
        }

        let raw = serde_json::json!({
            "token_ids": [10, 99, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["YmFzZS1pbWFnZQ=="]}
            }
        });
        let base_request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize base request");
        let base = preprocessed_from_generate(
            base_request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(4, true, None),
        )
        .expect("build base request");
        assert!(
            base.mm_routing_info.is_some(),
            "the worker setting alone does not activate adapter-scoped MM identity"
        );

        for tower_connector_lora_enabled in [false, true] {
            let adapter_request: GenerateRequest =
                serde_json::from_value(raw.clone()).expect("deserialize adapter request");
            let adapter = preprocessed_from_generate(
                adapter_request,
                "adapter-a",
                None,
                "resolved-request",
                routing_metadata(4, tower_connector_lora_enabled, Some("adapter-a")),
            )
            .expect("build adapter request");
            assert!(adapter.mm_routing_info.is_none());
            assert_eq!(
                adapter
                    .routing
                    .as_ref()
                    .and_then(|routing| routing.lora_name.as_deref()),
                Some("adapter-a")
            );
            assert!(
                adapter
                    .extra_args
                    .as_ref()
                    .and_then(|extra| extra.get("dynamo_mm_routing_hashes"))
                    .is_none()
            );
            let envelope = adapter
                .extra_args
                .as_ref()
                .and_then(|extra| extra.get("vllm_tito"))
                .expect("vllm_tito envelope");
            assert_eq!(envelope["features"], raw["features"]);
        }
    }

    #[test]
    fn omitted_max_tokens_stays_omitted_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(16, false, None),
        )
        .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, None);
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            None
        );
    }

    #[test]
    fn explicit_zero_min_tokens_stays_explicit_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {"min_tokens": 0},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(16, false, None),
        )
        .expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_projects_training_text_controls() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {
                "temperature": 0.25,
                "top_p": 0.9,
                "top_k": -1,
                "min_p": 0.05,
                "seed": 23,
                "max_tokens": 8,
                "min_tokens": 2,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "repetition_penalty": 1.1,
                "stop_token_ids": [7, 8],
                "ignore_eos": true,
                "logprobs": 1,
                "prompt_logprobs": 1,
                "skip_reading_prefix_cache": false,
                "skip_special_tokens": false
            },
            "kv_transfer_params": {
                "connector_data": {"block_ids": [1, 2]}
            },
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            None,
            "resolved-request",
            routing_metadata(16, false, None),
        )
        .expect("build request");

        assert_eq!(preprocessed.sampling_options.temperature, Some(0.25));
        assert_eq!(preprocessed.sampling_options.top_p, Some(0.9));
        assert_eq!(preprocessed.sampling_options.top_k, Some(-1));
        assert_eq!(preprocessed.sampling_options.min_p, Some(0.05));
        assert_eq!(preprocessed.sampling_options.seed, Some(23));
        assert_eq!(preprocessed.sampling_options.presence_penalty, Some(0.1));
        assert_eq!(preprocessed.sampling_options.frequency_penalty, Some(0.2));
        assert_eq!(preprocessed.sampling_options.repetition_penalty, Some(1.1));
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(2));
        assert_eq!(
            preprocessed.stop_conditions.stop_token_ids_hidden,
            Some(vec![7, 8])
        );
        assert_eq!(preprocessed.stop_conditions.ignore_eos, Some(true));
        assert_eq!(preprocessed.output_options.logprobs, Some(1));
        assert_eq!(preprocessed.output_options.prompt_logprobs, Some(1));
        assert_eq!(
            preprocessed
                .extra_args
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .and_then(|extra| extra.get("skip_reading_prefix_cache")),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(
            preprocessed
                .extra_args
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .and_then(|extra| extra.get("kv_transfer_params")),
            Some(&serde_json::json!({"connector_data": {"block_ids": [1, 2]}}))
        );
        assert_eq!(
            preprocessed
                .extra_args
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .and_then(|extra| extra.get("vllm_tito"))
                .and_then(|tito| tito.get("sampling_params"))
                .and_then(|sampling| sampling.get("top_k")),
            Some(&serde_json::json!(-1))
        );
        assert_eq!(preprocessed.output_options.skip_special_tokens, Some(false));
    }

    #[test]
    fn generate_control_shadow_carries_dp_rank_and_inverted_priority() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "priority": -7
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            Some(3),
            "resolved-request",
            routing_metadata(16, false, None),
        )
        .expect("build request");
        let routing = preprocessed.routing.as_ref().expect("routing hints");

        assert_eq!(routing.dp_rank, Some(3));
        assert_eq!(routing.priority, Some(7));
        assert_eq!(routing.priority_jump, Some(7.0));
    }

    #[test]
    fn priority_inversion_saturates_at_i32_min() {
        assert_eq!(dynamo_routing_priority(i32::MIN), i32::MAX);
    }
}
