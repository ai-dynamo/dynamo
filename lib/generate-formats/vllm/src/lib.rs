// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM's token-in/token-out `POST /inference/v1/generate`, as a generate format.
//!
//! The request stays opaque in `extra_args.vllm_tito` apart from the routing
//! controls Dynamo projects, including multimodal KV routing hashes. The
//! unary response is assembled from the worker's typed outputs. Streaming and
//! migration are not supported yet.

use axum::http::StatusCode;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use dynamo_runtime::error::DynamoError;
use serde_json::{Value, json};

use crate::generate::{GenerateAggregator, GenerateRequest, GenerateResponseOptions};
use dynamo_llm::generate_format::{
    Bound, ClientError, ClientRequest, GenerateExchange, GenerateFormat, Opened, Worker,
};
use dynamo_llm::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;
use dynamo_llm::protocols::common::extensions::HEADER_REQUEST_PRIORITY;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::preprocessed_mm_identifier;
use dynamo_llm::protocols::common::preprocessor::MmRoutingInfo;

mod errors;

pub const NAME: &str = "vllm";
const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

/// Convert vLLM's lower-is-higher priority to Dynamo's higher-is-higher scale.
fn dynamo_routing_priority(vllm_priority: i32) -> i32 {
    vllm_priority.saturating_neg()
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

fn preprocessed_from_generate_with_priority(
    mut request: GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    priority_override: Option<i32>,
    request_id: &str,
    worker: &Worker<'_>,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let stop_conditions = sampling.project_stop_conditions();
    let sampling_options = sampling.project_sampling_options();
    let output_options = sampling
        .project_output_options()
        .map_err(anyhow::Error::msg)?;
    let routing_priority =
        priority_override.unwrap_or_else(|| dynamo_routing_priority(request.priority));
    // The sidecar protocol does not expose whether tower/connector LoRA is
    // active, so conservatively treat every adapter's MM identity as scoped.
    let mm_routing = if worker.lora_name.is_some() {
        tracing::debug!(
            target: "mm_routing",
            tower_connector_lora_enabled = worker.tower_connector_lora_enabled,
            "LoRA request uses token-only multimodal routing"
        );
        None
    } else {
        match generate_mm_routing_info(&request, worker.kv_cache_block_size) {
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
    let skip_reading_prefix_cache = sampling.skip_reading_prefix_cache();
    // Keep the token buffer only in the routed request; workers reconstruct it.
    let token_ids = std::mem::take(&mut request.token_ids);
    request.request_id = Some(request_id.to_string());
    if request.model.is_some() {
        request.model = Some(model.to_string());
    }
    let mut vllm_tito = serde_json::to_value(&request)?;
    if let Some(body) = vllm_tito.as_object_mut() {
        body.remove("token_ids");
    }
    let mut extra_args = serde_json::Map::new();
    extra_args.insert("vllm_tito".to_string(), vllm_tito);
    if let Some(kv_transfer_params) = request.kv_transfer_params.as_ref() {
        extra_args.insert(
            "kv_transfer_params".to_string(),
            serde_json::Value::Object(kv_transfer_params.clone()),
        );
    }
    if let Some(skip_reading_prefix_cache) = skip_reading_prefix_cache {
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
    let has_opaque_multimodal_input = request
        .passthrough
        .get("features")
        .is_some_and(|features| !features.is_null());
    let cache_salt = request.cache_salt;

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(token_ids)
        .stop_conditions(stop_conditions)
        .sampling_options(sampling_options)
        .output_options(output_options)
        .mm_routing_info(mm_routing_info)
        .has_opaque_multimodal_input(has_opaque_multimodal_input)
        .routing(Some(
            dynamo_llm::protocols::common::preprocessor::RoutingHints {
                dp_rank: data_parallel_rank,
                expected_output_tokens: max_tokens,
                lora_name: worker.lora_name.map(str::to_owned),
                cache_namespace: cache_salt,
                // `priority_jump` is a boost-only scheduler input. Preserve penalties
                // in signed `priority`, matching the standard preprocessor projection.
                priority_jump: Some(routing_priority.max(0) as f64),
                priority: Some(routing_priority),
                ..Default::default()
            },
        ))
        .extra_args(Some(serde_json::Value::Object(extra_args)))
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
    fn worker_error(
        &self,
        error: &(dyn std::error::Error + 'static),
        phase: dynamo_llm::generate_format::ErrorPhase,
    ) -> dynamo_llm::generate_format::WorkerError {
        errors::classify(error, phase)
    }

    fn body_error(
        &self,
        error: axum::extract::rejection::BytesRejection,
        headers: &axum::http::HeaderMap,
    ) -> ClientError {
        dynamo_llm::generate_format::json_body_error(headers, error).into()
    }

    fn frontend_error(&self, error: ClientError) -> ClientError {
        error
    }

    fn supports_put(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        NAME
    }

    fn capability(&self) -> &'static str {
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY
    }

    fn open(&self, client: ClientRequest<'_>) -> Result<Opened, ClientError> {
        let request: GenerateRequest = client.json()?;
        let validation = request.validate().map_err(ClientError::new).and_then(|()| {
            if request.stream {
                Err(ClientError {
                    status: StatusCode::NOT_IMPLEMENTED,
                    message:
                        "streaming (stream=true) is not implemented for /inference/v1/generate yet"
                            .to_string(),
                    response: None,
                })
            } else {
                Ok(())
            }
        });
        // vLLM keeps request metadata in headers first, then the body.
        let header = |name| client.headers.get(name).and_then(|v| v.to_str().ok());
        // The Dynamo header uses higher-is-higher priority. vLLM's body field
        // uses lower-is-higher, so translate the winning header before storing
        // it in the engine-native body. Keep the canonical value separately for
        // Dynamo routing, including i32::MIN where saturating negation is lossy.
        let priority_override =
            header(HEADER_REQUEST_PRIORITY).and_then(|value| value.trim().parse::<i32>().ok());
        let mut request = request;
        if let Some(priority) = priority_override {
            request.priority = dynamo_routing_priority(priority);
        }
        let request_id = header(X_REQUEST_ID_HEADER)
            .map(ToOwned::to_owned)
            .or_else(|| request.request_id.clone())
            .unwrap_or_else(|| client.default_request_id.to_string());
        let data_parallel_rank =
            header(X_DATA_PARALLEL_RANK_HEADER).and_then(|v| v.trim().parse().ok());
        Ok(Opened {
            model: request.model.clone(),
            request_id: request_id.clone(),
            stream: false,
            input_tokens: request.token_ids.len(),
            validation,
            bind: Box::new(move |worker| {
                let options = request.response_options();
                let request = preprocessed_from_generate_with_priority(
                    request,
                    worker.model,
                    data_parallel_rank,
                    priority_override,
                    &request_id,
                    worker,
                )
                .map_err(|error| ClientError::new(error.to_string()))?;
                Ok(Bound {
                    request,
                    migration: Err("vLLM generate requests do not migrate yet".to_string()),
                    exchange: Box::new(VllmExchange {
                        aggregator: GenerateAggregator::new(request_id),
                        options,
                    }),
                })
            }),
        })
    }

    fn error_body(&self, status: StatusCode, message: &str) -> Value {
        json!({"error": {"message": message, "type": error_type(status), "code": status.as_u16()}})
    }
}

struct VllmExchange {
    aggregator: GenerateAggregator,
    options: GenerateResponseOptions,
}

impl GenerateExchange for VllmExchange {
    fn chunk(
        &mut self,
        output: LLMEngineOutput,
        _prefill: bool,
        _emit: &mut dyn FnMut(&Value),
    ) -> Result<(), DynamoError> {
        self.aggregator
            .apply_output(output, self.options)
            .map_err(|error| DynamoError::from(error.as_ref()))
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
            .map_err(|error| DynamoError::from(error.as_ref()))?;
        serde_json::to_value(response)
            .map(Some)
            .map_err(|error| DynamoError::msg(error.to_string()))
    }
}

#[cfg(test)]
mod tests;

pub mod generate;
