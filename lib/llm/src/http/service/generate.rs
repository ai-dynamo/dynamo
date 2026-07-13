// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental engine-native endpoint, **disabled by default**;
//! opt in via the `enable_engine_apis` builder flag or the
//! `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE` env var. When enabled it registers
//! a frontend-native handler that preserves the complete request in an opaque
//! backend envelope. Streaming (`stream=true`) remains unimplemented.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Context};
use futures::StreamExt;
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, ErrorType, HttpQueueGuard, ResponseMetricCollector};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::{MmRoutingInfo, PreprocessedRequest};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{
    GenerateRequest, GenerateResponse, GenerateResponseOptions, SamplingParams, StreamOptions,
};
use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

#[derive(Debug)]
struct GenerateRequestContext {
    request_id: String,
    data_parallel_rank: Option<u32>,
}

/// vLLM-style nested error body: `{"error": {"message", "type", "code"}}`.
#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

/// Create an Axum [`Router`] for the token-in/token-out `Generate` endpoint.
/// If no path is provided, the default path is `/inference/v1/generate`.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/inference/v1/generate".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Build a vLLM-style nested-`error` response.
fn generate_error_response(code: StatusCode, error_type: &str, message: String) -> Response {
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody {
                message,
                error_type: error_type.to_string(),
                code: code.as_u16(),
            },
        }),
    )
        .into_response()
}

/// Resolve the request metadata that vLLM keeps outside the public JSON body.
fn resolve_generate_request_context(
    headers: &HeaderMap,
    body_request_id: Option<&str>,
) -> GenerateRequestContext {
    let request_id = headers
        .get(X_REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .or_else(|| body_request_id.map(ToOwned::to_owned))
        .unwrap_or_else(|| get_or_create_request_id(headers));
    let data_parallel_rank = headers
        .get(X_DATA_PARALLEL_RANK_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());

    GenerateRequestContext {
        request_id,
        data_parallel_rank,
    }
}

/// Convert vLLM's lower-is-higher priority to Dynamo's higher-is-higher scale.
fn dynamo_routing_priority(vllm_priority: i32) -> i32 {
    vllm_priority.saturating_neg()
}

fn generate_dispatch_span(request_id: &str) -> tracing::Span {
    tracing::info_span!(target: "request_span", "generate", request_id = %request_id)
}

async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Preserve an ownership-bearing result if it completes concurrently;
        // callers re-check the context before using it.
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

fn generate_cancelled_response() -> Response {
    generate_error_response(
        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
        "request_cancelled",
        "request was cancelled".to_string(),
    )
}

fn generate_internal_error_response() -> Response {
    generate_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "internal server error".to_string(),
    )
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

/// Build the routing-only token sequence used by vLLM KV events for multimodal
/// prompts. The caller-provided `features` object remains opaque to execution;
/// this projection reads only the hashes and placeholder ranges required to
/// make request-side KV hashes match worker-side event hashes.
fn generate_mm_routing_info(
    request: &GenerateRequest,
    kv_cache_block_size: u32,
) -> Result<Option<MmRoutingInfo>, &'static str> {
    let Some(features) = request.passthrough.get("features") else {
        return Ok(None);
    };
    if features.is_null() {
        return Ok(None);
    }

    let features = features
        .as_object()
        .ok_or("features must be a JSON object")?;
    let mm_hashes = features
        .get("mm_hashes")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.mm_hashes must be a JSON object")?;
    let mm_placeholders = features
        .get("mm_placeholders")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.mm_placeholders must be a JSON object")?;

    if mm_hashes
        .keys()
        .chain(mm_placeholders.keys())
        .any(|modality| modality != "image")
    {
        return Err("exact /generate MM routing currently supports image placeholders only");
    }
    if kv_cache_block_size == 0 {
        return Err("KV cache block size must be non-zero");
    }

    let (hashes, placeholders) = match (mm_hashes.get("image"), mm_placeholders.get("image")) {
        (None, None) => return Ok(None),
        (Some(hashes), Some(placeholders)) => (
            hashes
                .as_array()
                .ok_or("features.mm_hashes.image must be an array")?,
            placeholders
                .as_array()
                .ok_or("features.mm_placeholders.image must be an array")?,
        ),
        _ => return Err("image hashes and placeholders must both be present"),
    };
    if hashes.len() != placeholders.len() {
        return Err("image hashes and placeholders must have equal lengths");
    }

    let mut ranges = Vec::with_capacity(hashes.len());
    for (hash, placeholder) in hashes.iter().zip(placeholders) {
        let hash = hash
            .as_str()
            .and_then(dynamo_kv_router::zmq_wire::hash_mm_identifier)
            .ok_or("multimodal hashes must be non-empty strings")?;
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

    ranges.sort_unstable_by_key(|(offset, _, _, _)| *offset);
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

    // vLLM's current event normalizer associates MM objects with contiguous
    // image-token runs by order, clamping excess runs to the last object in a
    // block. A sparse mask can split one object into multiple runs, so verify
    // that this run-order mapping still produces the request-side identity.
    // If it does not, retain correctness by using ordinary token routing.
    let block_size = kv_cache_block_size as usize;
    for block_start in (0..request.token_ids.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(request.token_ids.len());
        let mut worker_objects = Vec::new();
        let mut expected_by_position = vec![None; block_end - block_start];

        for (offset, end, hash, is_embed) in &ranges {
            let intersection_start = (*offset).max(block_start);
            let intersection_end = (*end).min(block_end);
            if intersection_start >= intersection_end {
                continue;
            }
            worker_objects.push(*hash);
            for global_position in intersection_start..intersection_end {
                let should_embed = is_embed
                    .as_ref()
                    .is_none_or(|mask| mask[global_position - *offset]);
                if should_embed {
                    expected_by_position[global_position - block_start] = Some(*hash);
                }
            }
        }

        let mut expected_runs = Vec::new();
        let mut current_run = None;
        for expected_hash in expected_by_position {
            match (current_run, expected_hash) {
                (None, Some(hash)) => {
                    current_run = Some(hash);
                    expected_runs.push(hash);
                }
                (Some(current), Some(hash)) if current != hash => {
                    return Err("adjacent multimodal embed positions must share an identifier");
                }
                (Some(_), None) => current_run = None,
                _ => {}
            }
        }

        for (run_index, expected_hash) in expected_runs.into_iter().enumerate() {
            let worker_hash = worker_objects
                .get(run_index)
                .or_else(|| worker_objects.last())
                .copied();
            if worker_hash != Some(expected_hash) {
                return Err(
                    "sparse multimodal layout cannot be normalized exactly by worker events",
                );
            }
        }
    }

    let mut routing_token_ids = request.token_ids.clone();
    for (offset, end, hash, is_embed) in ranges {
        let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash);
        if let Some(mask) = is_embed {
            for (token, should_embed) in routing_token_ids[offset..end].iter_mut().zip(mask) {
                if should_embed {
                    *token = pad;
                }
            }
        } else {
            routing_token_ids[offset..end].fill(pad);
        }
    }

    let padded_len = routing_token_ids
        .len()
        .div_ceil(block_size)
        .checked_mul(block_size)
        .ok_or("multimodal routing token length overflow")?;
    routing_token_ids.resize(padded_len, 0);

    Ok(Some(MmRoutingInfo {
        routing_token_ids,
        // vLLM events are normalized to the same pad-value token scheme, so
        // MM identity is already present in the alternate routing tokens.
        block_mm_infos: Vec::new(),
        expanded_prompt_len: request.token_ids.len(),
    }))
}

/// Project routing controls while retaining all engine-owned fields in
/// `extra_args.vllm_tito`. The backend remains the authority for interpreting
/// every vLLM-specific field.
fn preprocessed_from_generate(
    request: GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    request_id: &str,
    kv_cache_block_size: u32,
    lora_name: Option<String>,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let min_tokens = sampling.min_tokens();
    let ignore_eos = sampling.ignore_eos();
    let routing_priority = dynamo_routing_priority(request.priority);
    let input_tokens = request.token_ids.len();
    // With vLLM's default `enable_tower_connector_lora=false`, the vision tower
    // and connector stay on base weights, so MM identifiers are adapter-invariant.
    // `lora_name` separately salts the LM KV hashes below, allowing exact MM+LoRA
    // routing. If tower/connector LoRA is enabled, vLLM scopes MM identifiers by
    // adapter; that worker capability is not advertised here, so this projection
    // can miss the correct MM cache owner (suboptimal routing, not unsafe reuse).
    let mm_routing_info = match generate_mm_routing_info(&request, kv_cache_block_size) {
        Ok(info) => info,
        Err(reason) => {
            tracing::debug!(
                target: "mm_routing",
                reason,
                "invalid /generate multimodal routing metadata; using token-only routing"
            );
            None
        }
    };
    let vllm_tito = serde_json::to_value(VllmTitoEnvelope::new(&request, request_id))?;
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(input_tokens, None);
    let GenerateRequest {
        token_ids,
        cache_salt,
        ..
    } = request;

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(token_ids)
        .stop_conditions(StopConditions {
            max_tokens,
            min_tokens,
            ignore_eos: Some(ignore_eos),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(Default::default())
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
        .extra_args(Some(serde_json::json!({
            // Do not copy token_ids into this envelope. The worker must rebuild
            // that field from PreprocessedRequest.token_ids after routing.
            "vllm_tito": vllm_tito,
        })))
        .tracker(Some(tracker))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

/// Metrics adapter for the raw engine stream used by `/inference/v1/generate`.
///
/// Unlike the OpenAI text endpoints, Generate deliberately bypasses the
/// tokenizer/postprocessor pipeline that emits `LLMMetricAnnotation`. Its
/// token IDs are already rendered, so observe the same response metrics from
/// the raw token deltas while leaving tokenizer and media metrics untouched.
struct GenerateMetricCollector {
    response: ResponseMetricCollector,
    http_queue: Option<HttpQueueGuard>,
    tracker: Arc<RequestTracker>,
    input_tokens: usize,
    output_tokens: usize,
}

impl GenerateMetricCollector {
    fn new(
        metrics: Arc<super::metrics::Metrics>,
        model: &str,
        tracker: Arc<RequestTracker>,
        input_tokens: usize,
    ) -> Self {
        Self {
            response: metrics.clone().create_response_collector(model),
            http_queue: Some(metrics.create_http_queue_guard(model)),
            tracker,
            input_tokens,
            output_tokens: 0,
        }
    }

    fn observe(&mut self, annotated: &Annotated<LLMEngineOutput>) {
        let Some(output) = annotated.data.as_ref() else {
            return;
        };

        if let Some(worker) = self.tracker.get_worker_info() {
            self.response.set_worker_info(
                worker.prefill_worker_id,
                worker.prefill_dp_rank,
                self.tracker.prefill_worker_type().map(String::from),
                worker.decode_worker_id,
                worker.decode_dp_rank,
                self.tracker.decode_worker_type().map(String::from),
            );
        }

        let cached_tokens = output
            .completion_usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        self.response.observe_cached_tokens(cached_tokens);

        let chunk_tokens = output.token_ids.len();
        self.output_tokens += chunk_tokens;
        self.response.observe_current_osl(self.output_tokens);
        if self.response.is_first_token()
            && chunk_tokens > 0
            && let Some(guard) = self.http_queue.take()
        {
            drop(guard);
        }
        self.response
            .observe_response(self.input_tokens, chunk_tokens);
    }
}

/// Resolve, route, and dispatch a frontend-native token-in/token-out request.
async fn handler_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    if let Err(response) = check_ready(&state) {
        return response.into_response();
    }

    if let Err(message) = request.validate() {
        return generate_error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }

    if request.stream {
        return generate_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for /inference/v1/generate yet".to_string(),
        );
    }
    let response_options = request.response_options();

    let model = match &request.model {
        Some(model) => model.clone(),
        None => {
            let models = state.manager().list_generate_models();
            match models.len() {
                1 => models.into_iter().next().unwrap(),
                0 => {
                    return generate_error_response(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "no generate-capable model is registered".to_string(),
                    );
                }
                _ => {
                    return generate_error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "multiple models are registered; specify `model` in the request"
                            .to_string(),
                    );
                }
            }
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return response.into_response();
    }

    let (engine, kv_cache_block_size, lora_name) = match state
        .manager()
        .get_generate_engine_with_routing_metadata(&model)
    {
        Ok(engine_and_block_size) => engine_and_block_size,
        Err(error) => {
            let (status, error_type) = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return generate_error_response(status, error_type, error.to_string());
        }
    };

    let request_context = resolve_generate_request_context(&headers, request.request_id.as_deref());
    let preprocessed = match preprocessed_from_generate(
        request,
        &model,
        request_context.data_parallel_rank,
        &request_context.request_id,
        kv_cache_block_size,
        lora_name,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(error) => {
            return generate_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                error.to_string(),
            );
        }
    };

    let request_id = request_context.request_id;
    let context: Context<PreprocessedRequest> =
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return response.into_response(),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = generate_dispatch_span(&request_id);
    // Unary work must outlive the Axum handler so dropping the handler can signal
    // the armed connection monitor. The detached dispatch observes that kill at
    // each backend await point and then exits promptly.
    let response = match tokio::spawn(
        generate_dispatch(
            engine,
            context,
            request_id,
            model,
            state.clone(),
            response_options,
        )
        .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "generate dispatch task panicked");
            generate_internal_error_response()
        }
    };

    connection_handle.disarm();
    response
}

async fn generate_dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
    response_options: GenerateResponseOptions,
) -> Response {
    let metric_model = state.manager().metric_model_for(&model).to_string();
    let input_tokens = context.content().token_ids.len();
    let tracker = context.content().tracker.clone().unwrap_or_else(|| {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(input_tokens, None);
        tracker
    });
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        super::metrics::Endpoint::Generate,
        false,
        &request_id,
    );
    let mut metric_collector =
        GenerateMetricCollector::new(state.metrics_clone(), &metric_model, tracker, input_tokens);
    let request_context = context.context();
    let generate_result =
        match run_until_killed(request_context.as_ref(), engine.generate(context)).await {
            Some(result) => result,
            None => {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
        };
    if request_context.is_killed() {
        inflight_guard.mark_error(ErrorType::Cancelled);
        return generate_cancelled_response();
    }
    let stream = match generate_result {
        Ok(stream) => stream,
        Err(error) => {
            let was_cancelled = request_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref());
            let was_rejected = super::metrics::request_was_rejected(error.as_ref());
            inflight_guard.mark_error(if was_cancelled {
                ErrorType::Cancelled
            } else if was_rejected {
                ErrorType::Unavailable
            } else {
                ErrorType::Internal
            });
            if was_cancelled {
                return generate_cancelled_response();
            }
            if was_rejected {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected generate request");
                state
                    .metrics_clone()
                    .inc_rejection(&metric_model, super::metrics::Endpoint::Generate);
                return generate_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    "engine rejected the request".to_string(),
                );
            }
            tracing::error!(%request_id, error = %format!("{error:#}"), "engine generate call failed");
            return generate_internal_error_response();
        }
    };

    let engine_context = stream.context();
    let stream = stream.inspect(move |annotated| metric_collector.observe(annotated));
    let response_result = match run_until_killed(
        request_context.as_ref(),
        GenerateResponse::from_annotated_stream_with_options(
            stream,
            request_id.clone(),
            response_options,
        ),
    )
    .await
    {
        Some(result) => result,
        None => {
            inflight_guard.mark_error(ErrorType::Cancelled);
            return generate_cancelled_response();
        }
    };
    match response_result {
        Ok(response) => {
            if request_context.is_killed() || engine_context.is_killed() {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
            if !response.is_complete_unary() {
                inflight_guard.mark_error(ErrorType::Internal);
                tracing::error!(%request_id, "generate stream ended without a complete choice");
                return generate_internal_error_response();
            }
            inflight_guard.mark_ok();
            Json(response).into_response()
        }
        Err(error) => {
            if request_context.is_killed()
                || engine_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref())
            {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response();
            }
            inflight_guard.mark_error(ErrorType::Internal);
            tracing::error!(%request_id, %error, "failed to fold generate stream");
            generate_internal_error_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context as TaskContext, Poll},
    };

    use super::service_v2::{HttpService, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV};
    use super::*;
    use crate::http::service::metrics::{Endpoint, RequestType, Status};
    use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};
    use dynamo_runtime::{
        engine::{AsyncEngine, ResponseStream},
        pipeline::{Error, ManyOut, SingleIn},
    };
    use futures::Stream;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;
    use tracing::field::{Field, Visit};
    use tracing::{Subscriber, span};
    use tracing_subscriber::Layer;
    use tracing_subscriber::prelude::*;

    #[derive(Clone, Copy)]
    enum PendingPhase {
        Generate,
        Stream,
    }

    struct PendingOperation {
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
        polled: bool,
    }

    impl PendingOperation {
        fn new(started: Arc<Notify>, dropped: Arc<AtomicBool>) -> Self {
            Self {
                started,
                dropped,
                polled: false,
            }
        }

        fn mark_started(&mut self) {
            if !self.polled {
                self.polled = true;
                self.started.notify_one();
            }
        }
    }

    impl Future for PendingOperation {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Stream for PendingOperation {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Drop for PendingOperation {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct PendingEngine {
        phase: PendingPhase,
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
    }

    struct TerminalEngine(crate::protocols::common::FinishReason);

    struct CancelledEngine;

    struct MetricEngine;

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CancelledEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Cancelled)
                .message("backend cancelled before opening a stream")
                .build()
                .into())
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for TerminalEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
                index: Some(0),
                finish_reason: Some(self.0.clone()),
                ..Default::default()
            })]);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for MetricEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![10],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![11],
                    index: Some(0),
                    finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                    completion_usage: Some(dynamo_protocols::types::CompletionUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                        prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                            audio_tokens: None,
                            cached_tokens: Some(2),
                        }),
                        completion_tokens_details: None,
                    }),
                    ..Default::default()
                }),
            ]);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for PendingEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let operation = PendingOperation::new(self.started.clone(), self.dropped.clone());
            match self.phase {
                PendingPhase::Generate => {
                    operation.await;
                    unreachable!("pending generate operation completed")
                }
                PendingPhase::Stream => {
                    Ok(ResponseStream::new(Box::pin(operation), request.context()))
                }
            }
        }
    }

    #[derive(Clone)]
    struct RequestIdCaptureLayer(Arc<Mutex<Option<String>>>);

    impl<S: Subscriber> Layer<S> for RequestIdCaptureLayer {
        fn on_new_span(
            &self,
            attrs: &span::Attributes<'_>,
            _id: &span::Id,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = RequestIdVisitor::default();
            attrs.record(&mut visitor);
            if visitor.request_id.is_some() {
                *self.0.lock().unwrap() = visitor.request_id;
            }
        }
    }

    #[derive(Default)]
    struct RequestIdVisitor {
        request_id: Option<String>,
    }

    impl Visit for RequestIdVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "request_id" {
                self.request_id = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "request_id" {
                self.request_id = Some(format!("{value:?}"));
            }
        }
    }

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: Option<bool>) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let builder = HttpService::builder().port(port);
        let builder = match enable_generate {
            Some(enabled) => builder.enable_engine_apis(enabled),
            None => builder,
        };
        let service = builder.build().unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_structured_404() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{"top_k":-1}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{},"stream":true}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_implemented");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_rejects_empty_token_ids() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");

        assert_eq!(resp.status().as_u16(), StatusCode::BAD_REQUEST.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().is_some_and(
                |message| message.contains("token_ids must contain at least one token")
            )
        );
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_enforces_vllm_rust_request_rules() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let invalid = [
            r#"{"token_ids":[1],"sampling_params":{},"stream_options":{"include_usage":true}}"#,
            r#"{"token_ids":[1],"sampling_params":{"max_tokens":0}}"#,
            r#"{"token_ids":[1],"sampling_params":{"prompt_logprobs":-2}}"#,
            r#"{"token_ids":[1],"sampling_params":{"min_tokens":3,"max_tokens":2}}"#,
        ];

        for body in invalid {
            let resp = client
                .post(format!("http://localhost:{port}/inference/v1/generate"))
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .expect("generate request failed");
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body: serde_json::Value = resp.json().await.expect("json body");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_404_by_default() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>)],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_is_registered_when_enabled_by_env() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1"))],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                let body: serde_json::Value = resp.json().await.expect("json body");
                assert_eq!(body["error"]["type"], "not_found");
                handle.abort();
            },
        )
        .await;
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

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
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
        assert_eq!(preprocessed.token_ids, vec![1, 2]);
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
    fn multimodal_features_build_exact_routing_tokens_without_changing_execution_payload() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let raw = serde_json::json!({
            "token_ids": [10, 11, 12, 12, 12, 15, 16, 17, 17, 19],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [hash_a, hash_b]},
                "mm_placeholders": {"image": [
                    {"offset": 2, "length": 3},
                    {"offset": 7, "length": 2}
                ]},
                "kwargs_data": {"image": ["opaque-a", "opaque-b"]}
            }
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 4, None)
                .expect("build request");

        let pad_a = dynamo_kv_router::protocols::pad_value_for_mm_hash(0xaaaaaaaaaaaaaaaa);
        let pad_b = dynamo_kv_router::protocols::pad_value_for_mm_hash(0xbbbbbbbbbbbbbbbb);
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
            preprocessed.token_ids,
            vec![10, 11, 12, 12, 12, 15, 16, 17, 17, 19]
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
    }

    #[test]
    fn generate_mm_routing_hash_matches_normalized_vllm_worker_event() {
        let mm_identifier = "1234567890abcdef".repeat(2);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier.clone()]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 4)
            .expect("valid MM routing metadata")
            .expect("MM routing projection");
        let request_hashes = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &routing.routing_token_ids,
            4,
            dynamo_kv_router::protocols::BlockHashOptions::default(),
        );

        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            4,
            7,
            &[10, 99, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                mm_extra_info: Some(dynamo_kv_router::protocols::BlockExtraInfo {
                    mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
                        mm_hash: dynamo_kv_router::zmq_wire::hash_mm_identifier(&mm_identifier)
                            .expect("non-empty identifier"),
                        offsets: vec![(1, 3)],
                    }],
                }),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hashes[0], event_block.tokens_hash);
    }

    #[test]
    fn mixed_placeholder_span_without_embed_mask_disables_exact_routing() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 3}]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 5)
                .expect_err("an omitted sparse mask must not over-substitute tokens"),
            "mixed multimodal placeholder spans require is_embed"
        );
    }

    #[test]
    fn sparse_mm_embed_mask_matches_normalized_vllm_worker_event() {
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
                }]}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 5)
            .expect("valid sparse MM routing metadata")
            .expect("MM routing projection");
        let mm_hash = dynamo_kv_router::zmq_wire::hash_mm_identifier(mm_identifier)
            .expect("non-empty identifier");
        let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(mm_hash);
        assert_eq!(routing.routing_token_ids, vec![10, pad, 42, pad, 20]);

        let request_hash = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &routing.routing_token_ids,
            5,
            dynamo_kv_router::protocols::BlockHashOptions::default(),
        )[0];
        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            5,
            7,
            &[10, 99, 42, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                mm_extra_info: Some(dynamo_kv_router::protocols::BlockExtraInfo {
                    mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
                        mm_hash,
                        offsets: vec![],
                    }],
                }),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hash, event_block.tokens_hash);
    }

    #[test]
    fn sparse_multi_object_block_disables_inexact_projection() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20, 99, 30],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-a", "image-b"]},
                "mm_placeholders": {"image": [
                    {
                        "offset": 1,
                        "length": 3,
                        "is_embed": [true, false, true]
                    },
                    {"offset": 5, "length": 1}
                ]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 7)
                .expect_err("worker run-order mapping would assign image B to image A"),
            "sparse multimodal layout cannot be normalized exactly by worker events"
        );
    }

    #[test]
    fn invalid_multimodal_routing_metadata_falls_back_without_dropping_features() {
        let raw = serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [""]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["opaque"]}
            }
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 4, None)
                .expect("malformed routing metadata must not reject execution");

        assert!(preprocessed.mm_routing_info.is_none());
        assert_eq!(preprocessed.token_ids, vec![1, 2, 3, 4]);
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
    }

    #[test]
    fn default_lora_mode_composes_mm_identity_with_lora_hashing() {
        let mm_identifier = "opaque-renderer-image-0";
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
            }
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "adapter-a",
            None,
            "resolved-request",
            4,
            Some("adapter-a".to_string()),
        )
        .expect("build request");

        let routing_tokens = &preprocessed
            .mm_routing_info
            .as_ref()
            .expect("default language-only LoRA keeps MM identifiers stable")
            .routing_token_ids;
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.as_deref()),
            Some("adapter-a")
        );

        let request_hash = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            routing_tokens,
            4,
            dynamo_kv_router::protocols::BlockHashOptions {
                lora_name: Some("adapter-a"),
                ..Default::default()
            },
        )[0];
        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            4,
            7,
            &[10, 99, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                lora_name: Some("adapter-a"),
                mm_extra_info: Some(dynamo_kv_router::protocols::BlockExtraInfo {
                    mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
                        mm_hash: dynamo_kv_router::zmq_wire::hash_mm_identifier(mm_identifier)
                            .expect("non-empty identifier"),
                        offsets: vec![(1, 3)],
                    }],
                }),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hash, event_block.tokens_hash);
    }

    #[test]
    fn overlapping_multimodal_placeholders_disable_mm_routing() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [9, 9, 9, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["a".repeat(64), "b".repeat(64)]},
                "mm_placeholders": {"image": [
                    {"offset": 0, "length": 2},
                    {"offset": 1, "length": 2}
                ]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4).expect_err("overlap must disable MM routing"),
            "multimodal placeholder ranges must not overlap"
        );
    }

    #[test]
    fn non_image_modality_disables_exact_mm_projection() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"audio": ["audio-0"]},
                "mm_placeholders": {"audio": [{"offset": 1, "length": 2}]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4)
                .expect_err("non-image modality must disable exact projection"),
            "exact /generate MM routing currently supports image placeholders only"
        );
    }

    #[test]
    fn image_hashes_and_placeholders_must_be_paired() {
        for features in [
            serde_json::json!({
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {}
            }),
            serde_json::json!({
                "mm_hashes": {},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
            }),
        ] {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "token_ids": [1, 2, 3, 4],
                "sampling_params": {},
                "features": features
            }))
            .expect("deserialize request");

            assert_eq!(
                generate_mm_routing_info(&request, 4)
                    .expect_err("one-sided image metadata must disable exact routing"),
                "image hashes and placeholders must both be present"
            );
        }
    }

    #[test]
    fn image_hashes_and_placeholders_must_have_equal_lengths() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-0", "image-1"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4)
                .expect_err("item count mismatch must disable exact routing"),
            "image hashes and placeholders must have equal lengths"
        );
    }

    #[test]
    fn omitted_max_tokens_stays_omitted_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
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

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "3".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "header-request");
        assert_eq!(context.data_parallel_rank, Some(3));
    }

    #[test]
    fn generate_request_context_falls_back_and_ignores_invalid_dp_rank() {
        let mut headers = HeaderMap::new();
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "invalid".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "body-request");
        assert_eq!(context.data_parallel_rank, None);
    }

    #[test]
    fn generate_dispatch_span_uses_resolved_request_id() {
        let captured_request_id = Arc::new(Mutex::new(None));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(RequestIdCaptureLayer(captured_request_id.clone())),
        );

        let _dispatch_span = generate_dispatch_span("header-request");

        assert_eq!(
            captured_request_id.lock().unwrap().as_deref(),
            Some("header-request")
        );
    }

    fn dispatch_test_context() -> Context<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .expect("build dispatch test request"),
        )
    }

    fn metric_value<'a>(
        families: &'a [prometheus::proto::MetricFamily],
        name: &str,
        labels: &[(&str, &str)],
    ) -> &'a prometheus::proto::Metric {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"));
        family
            .get_metric()
            .iter()
            .find(|metric| {
                labels.iter().all(|(expected_name, expected_value)| {
                    metric.get_label().iter().any(|label| {
                        label.name() == *expected_name && label.value() == *expected_value
                    })
                })
            })
            .unwrap_or_else(|| panic!("missing {name} series with labels {labels:?}"))
    }

    fn assert_cancelled_dispatch_metrics(state: &service_v2::State) {
        let metric_model = state.manager().metric_model_for("test-model");
        let metrics = state.metrics_clone();
        assert_eq!(metrics.get_inflight_count(metric_model), 0);
        assert_eq!(
            metrics.get_request_counter(
                metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );
    }

    async fn await_cancelled_dispatch(
        task: tokio::task::JoinHandle<Response>,
        dropped: &AtomicBool,
        state: &service_v2::State,
    ) {
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("dispatch did not stop promptly after request kill")
            .expect("dispatch task panicked");
        assert_eq!(response.status().as_u16(), 499);
        assert!(dropped.load(Ordering::SeqCst));
        assert_cancelled_dispatch_metrics(state);
    }

    async fn assert_request_kill_interrupts_pending(phase: PendingPhase) {
        let started = Arc::new(Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(PendingEngine {
                phase,
                started: started.clone(),
                dropped: dropped.clone(),
            });
        let context = dispatch_test_context();
        let request_context = context.context();
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let task = tokio::spawn(generate_dispatch(
            engine,
            context,
            "req-pending-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        ));

        started.notified().await;
        assert_eq!(
            state
                .metrics_clone()
                .get_inflight_count(state.manager().metric_model_for("test-model")),
            1
        );
        request_context.kill();

        await_cancelled_dispatch(task, dropped.as_ref(), state.as_ref()).await;
    }

    async fn dispatch_terminal_finish_reason(
        finish_reason: crate::protocols::common::FinishReason,
    ) -> (Response, Arc<service_v2::State>) {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(TerminalEngine(finish_reason));
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-terminal-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;
        (response, state)
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_engine_generate() {
        assert_request_kill_interrupts_pending(PendingPhase::Generate).await;
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_response_stream() {
        assert_request_kill_interrupts_pending(PendingPhase::Stream).await;
    }

    #[tokio::test]
    async fn backend_error_finish_returns_sanitized_500() {
        let secret = "sensitive backend failure";
        let (response, _state) = dispatch_terminal_finish_reason(
            crate::protocols::common::FinishReason::Error(secret.to_string()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("parse error response");
        assert_eq!(body["error"]["message"], "internal server error");
        assert!(!body.to_string().contains(secret));
    }

    #[tokio::test]
    async fn backend_cancelled_finish_returns_499() {
        let (response, state) =
            dispatch_terminal_finish_reason(crate::protocols::common::FinishReason::Cancelled)
                .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn immediate_engine_cancellation_returns_499() {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(CancelledEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();

        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-immediate-cancel".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn successful_generate_populates_frontend_metrics() {
        const MODEL: &str = "generate-metric-test-model";
        const WORKER_ID: &str = "987654321";
        const DP_RANK: &str = "3";

        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(3, None);
        tracker.record_worker(
            WORKER_ID.parse().unwrap(),
            Some(DP_RANK.parse().unwrap()),
            crate::discovery::WORKER_TYPE_DECODE,
        );
        let context = Context::new(
            PreprocessedRequest::builder()
                .model(MODEL.to_string())
                .token_ids(vec![1, 2, 3])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .tracker(Some(tracker))
                .build()
                .expect("build metric test request"),
        );
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(MetricEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let metric_model = state.manager().metric_model_for(MODEL).to_string();
        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();

        let response = generate_dispatch(
            engine,
            context,
            "req-generate-metrics".to_string(),
            MODEL.to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.metrics_clone().get_inflight_count(&metric_model), 0);
        assert_eq!(
            state.metrics_clone().get_request_counter(
                &metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Success,
                &ErrorType::None,
            ),
            1
        );

        let families = registry.gather();
        let model_labels = [("model", metric_model.as_str())];
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_requests_started_total",
                &[("model", metric_model.as_str()), ("endpoint", "generate")],
            )
            .get_counter()
            .value(),
            1.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_active_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_queued_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_request_duration_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_input_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            3.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_tokens_total",
                &model_labels,
            )
            .get_counter()
            .value(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_time_to_first_token_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_inter_token_latency_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_cached_tokens", &model_labels,)
                .get_histogram()
                .get_sample_sum(),
            2.0
        );

        let worker_labels = [WORKER_ID, DP_RANK, crate::discovery::WORKER_TYPE_DECODE];
        assert_eq!(
            crate::http::service::metrics::WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
                .with_label_values(&worker_labels)
                .get(),
            3
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
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
            16,
            None,
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
