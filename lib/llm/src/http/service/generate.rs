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

mod mm_routing;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, ErrorType, HttpQueueGuard, ResponseMetricCollector};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{
    GenerateRequest, GenerateResponse, GenerateResponseOptions, SamplingParams, StreamOptions,
};
use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};
use mm_routing::{generate_mm_routing_info, validate_generate_mm_features};

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
    if lora_name.as_deref().is_some_and(|name| !name.is_empty())
        && request
            .passthrough
            .get("features")
            .is_some_and(|features| !features.is_null())
    {
        anyhow::bail!(
            "native gRPC does not yet advertise tower-LoRA multimodal cache semantics; multimodal requests with LoRA are unsupported"
        );
    }
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let min_tokens = sampling.min_tokens();
    let ignore_eos = sampling.ignore_eos();
    let routing_priority = dynamo_routing_priority(request.priority);
    let input_tokens = request.token_ids.len();
    // These renderer-provided hashes are routing hints only. The sidecar and
    // vLLM derive the receiver-cache key from the verified payload bytes. Native
    // gRPC rejects multimodal+LoRA until tower-LoRA cache semantics are advertised.
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
    if let Err(message) = validate_generate_mm_features(&request) {
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

include!("generate/tests.rs");
