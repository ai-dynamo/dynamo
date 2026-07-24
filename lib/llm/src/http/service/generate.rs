// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handlers for the token-in/token-out `Generate` APIs:
//! `POST /inference/v1/generate` for vLLM and
//! `POST` or `PUT /generate` for SGLang.
//!
//! These experimental engine-native endpoints are **disabled by default**;
//! opt in via the `enable_engine_apis` builder flag or the
//! backend-specific `DYN_*_ENABLE_*` env vars. When enabled,
//! the frontend preserves the complete request in an opaque
//! backend envelope. SGLang requests support native incremental SSE streaming.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    middleware,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::post,
};
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Context};
use futures::StreamExt;
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects};
use super::metrics::{CancellationLabels, ErrorType};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, find_invalid_argument_in_chain,
    get_body_limit, get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::local_model::runtime_config::{
    SGLANG_GENERATE_CAPABILITY, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{
    GenerateRequest, GenerateResponse, GenerateResponseOptions, SamplingParams, StreamOptions,
};
use crate::protocols::sglang::generate::{
    SglangGenerateRequest, SglangGenerateResponse, SglangResponseOptions,
};
use crate::protocols::sglang::stream::SglangGenerateStream;

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";
pub(super) const VLLM_GENERATE_DEFAULT_PATH: &str = "/inference/v1/generate";
pub(super) const SGLANG_GENERATE_DEFAULT_PATH: &str = "/generate";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerateBackend {
    Vllm,
    Sglang,
}

impl GenerateBackend {
    fn capability(self) -> &'static str {
        match self {
            Self::Vllm => VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
            Self::Sglang => SGLANG_GENERATE_CAPABILITY,
        }
    }
}

#[derive(Clone, Copy)]
enum GenerateResponseFormat {
    Vllm(GenerateResponseOptions),
    Sglang(SglangResponseOptions),
}

enum AggregatedGenerateResponse {
    Vllm(GenerateResponse),
    Sglang(serde_json::Value),
}

impl AggregatedGenerateResponse {
    fn is_complete_unary(&self) -> bool {
        match self {
            Self::Vllm(response) => response.is_complete_unary(),
            Self::Sglang(_) => true,
        }
    }

    fn into_response(self) -> Response {
        match self {
            Self::Vllm(response) => Json(response).into_response(),
            Self::Sglang(response) => Json(response).into_response(),
        }
    }
}

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

/// SGLang unary error body: `{"error": {"message": ...}}`.
#[derive(Serialize, Debug)]
struct SglangGenerateError {
    error: SglangGenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct SglangGenerateErrorBody {
    message: String,
}

/// SGLang schema-validation error body (its legacy `ErrorResponse`).
#[derive(Serialize, Debug)]
struct SglangValidationError {
    object: &'static str,
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<&'static str>,
    code: u16,
}

/// Create the vLLM-compatible token-in/token-out `Generate` route.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| VLLM_GENERATE_DEFAULT_PATH.to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_vllm_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Create the native SGLang token-in/token-out `Generate` route.
pub fn sglang_generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| SGLANG_GENERATE_DEFAULT_PATH.to_string());
    let docs = vec![
        RouteDoc::new(axum::http::Method::POST, &path),
        RouteDoc::new(axum::http::Method::PUT, &path),
    ];
    let router = Router::new()
        .route(
            &path,
            post(handler_sglang_generate).put(handler_sglang_generate),
        )
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (docs, router)
}

impl GenerateBackend {
    fn error_response(self, code: StatusCode, error_type: &str, message: String) -> Response {
        match self {
            Self::Vllm => (
                code,
                Json(GenerateError {
                    error: GenerateErrorBody {
                        message,
                        error_type: error_type.to_string(),
                        code: code.as_u16(),
                    },
                }),
            )
                .into_response(),
            Self::Sglang => (
                code,
                Json(SglangGenerateError {
                    error: SglangGenerateErrorBody { message },
                }),
            )
                .into_response(),
        }
    }
}

fn resolve_generate_request_context(
    headers: &HeaderMap,
    body_request_id: Option<&str>,
    backend: GenerateBackend,
) -> GenerateRequestContext {
    let header_request_id = headers
        .get(X_REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let body_request_id = body_request_id.map(ToOwned::to_owned);
    let request_id = match backend {
        GenerateBackend::Vllm => header_request_id.or(body_request_id),
        GenerateBackend::Sglang => body_request_id.or(header_request_id),
    }
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

fn generate_cancelled_response(backend: GenerateBackend) -> Response {
    backend.error_response(
        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
        "request_cancelled",
        "request was cancelled".to_string(),
    )
}

fn generate_internal_error_response(backend: GenerateBackend) -> Response {
    backend.error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "internal server error".to_string(),
    )
}

/// Complete vLLM-owned request envelope forwarded opaquely to its worker.
///
/// `token_ids` are intentionally absent: `PreprocessedRequest.token_ids` is
/// the canonical routing and wire representation, and the worker reconstructs
/// the engine request from that field.
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

fn preprocessed_from_vllm_generate(
    request: GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    request_id: &str,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let min_tokens = sampling.min_tokens();
    let ignore_eos = sampling.ignore_eos();
    let routing_priority = dynamo_routing_priority(request.priority);
    let mut extra_args = serde_json::Map::new();
    extra_args.insert(
        "vllm_tito".to_string(),
        serde_json::to_value(VllmTitoEnvelope::new(&request, request_id))?,
    );
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
        .output_options(OutputOptions::default())
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            expected_output_tokens: max_tokens,
            cache_namespace: cache_salt,
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::Value::Object(extra_args)))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

fn preprocessed_from_sglang_generate(
    request: SglangGenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
) -> anyhow::Result<PreprocessedRequest> {
    let max_tokens = request.max_new_tokens();
    let min_tokens = request.min_new_tokens();
    let ignore_eos = request.ignore_eos();
    let routing_priority = request.priority.unwrap_or_default();
    let return_logprob = request.return_logprob.unwrap_or(false);
    let top_logprobs_num = request.top_logprobs_num.unwrap_or(0);
    let output_options = OutputOptions {
        logprobs: return_logprob.then_some(top_logprobs_num),
        prompt_logprobs: (return_logprob && request.logprob_start_len.unwrap_or(-1) >= 0)
            .then_some(top_logprobs_num),
        return_tokens_as_token_ids: Some(true),
        ..Default::default()
    };
    let mut extra_args = serde_json::Map::new();
    extra_args.insert("sglang_tito".to_string(), request.worker_envelope());

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(request.input_ids)
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            min_tokens: Some(min_tokens),
            ignore_eos: Some(ignore_eos),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(output_options)
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            expected_output_tokens: Some(max_tokens),
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::Value::Object(extra_args)))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

enum IncomingGenerateRequest {
    Vllm(GenerateRequest),
    Sglang(SglangGenerateRequest),
}

impl IncomingGenerateRequest {
    fn backend(&self) -> GenerateBackend {
        match self {
            Self::Vllm(_) => GenerateBackend::Vllm,
            Self::Sglang(_) => GenerateBackend::Sglang,
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Vllm(request) => request.validate(),
            Self::Sglang(request) => request.validate(),
        }
    }

    fn stream(&self) -> bool {
        match self {
            Self::Vllm(request) => request.stream,
            Self::Sglang(request) => request.stream,
        }
    }

    fn body_request_id(&self) -> Option<&str> {
        match self {
            Self::Vllm(request) => request.request_id.as_deref(),
            Self::Sglang(request) => request.rid.as_deref(),
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            Self::Vllm(request) => request.model.as_deref(),
            Self::Sglang(_) => None,
        }
    }

    fn response_format(&self) -> GenerateResponseFormat {
        match self {
            Self::Vllm(request) => GenerateResponseFormat::Vllm(request.response_options()),
            Self::Sglang(request) => GenerateResponseFormat::Sglang(request.response_options()),
        }
    }

    fn into_preprocessed(
        self,
        model: &str,
        data_parallel_rank: Option<u32>,
        request_id: &str,
    ) -> anyhow::Result<PreprocessedRequest> {
        match self {
            Self::Vllm(request) => {
                preprocessed_from_vllm_generate(request, model, data_parallel_rank, request_id)
            }
            Self::Sglang(request) => {
                preprocessed_from_sglang_generate(request, model, data_parallel_rank)
            }
        }
    }
}

fn adapt_openai_error(
    backend: GenerateBackend,
    response: super::openai::ErrorResponse,
) -> Response {
    if backend == GenerateBackend::Vllm {
        return response.into_response();
    }
    let (status, Json(error)) = response;
    backend.error_response(status, "request_error", error.message().to_string())
}

async fn handler_vllm_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    handle_generate(state, headers, IncomingGenerateRequest::Vllm(request)).await
}

async fn handler_sglang_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    request: Result<Json<SglangGenerateRequest>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(SglangValidationError {
                    object: "error",
                    message: error.body_text(),
                    error_type: "Bad Request",
                    param: None,
                    code: StatusCode::BAD_REQUEST.as_u16(),
                }),
            )
                .into_response();
        }
    };
    handle_generate(state, headers, IncomingGenerateRequest::Sglang(request)).await
}

/// Resolve, route, and dispatch a frontend-native token-in/token-out request.
async fn handle_generate(
    state: Arc<service_v2::State>,
    headers: HeaderMap,
    request: IncomingGenerateRequest,
) -> Response {
    let backend = request.backend();
    if let Err(response) = check_ready(&state) {
        return adapt_openai_error(backend, response);
    }
    if let Err(message) = request.validate() {
        return backend.error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }
    let streaming = request.stream();
    if streaming && backend == GenerateBackend::Vllm {
        return backend.error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for the vLLM Generate API yet".to_string(),
        );
    }
    let response_format = request.response_format();

    let model = match request.model() {
        Some(model) => model.to_string(),
        None => {
            let models = state
                .manager()
                .list_generate_models_for_capability(backend.capability());
            match models.len() {
                1 => models.into_iter().next().unwrap(),
                0 => {
                    return backend.error_response(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "no generate-capable model is registered".to_string(),
                    );
                }
                _ => {
                    let message = match backend {
                        GenerateBackend::Vllm => {
                            "multiple models are registered; specify `model` in the request"
                        }
                        GenerateBackend::Sglang => {
                            "multiple SGLang models are registered; configure a model-specific generate endpoint"
                        }
                    };
                    return backend.error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        message.to_string(),
                    );
                }
            }
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return adapt_openai_error(backend, response);
    }
    let engine = match state
        .manager()
        .get_generate_engine_for_capability(&model, backend.capability())
    {
        Ok(engine) => engine,
        Err(error) => {
            let (status, error_type) = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return backend.error_response(status, error_type, error.to_string());
        }
    };

    let request_context =
        resolve_generate_request_context(&headers, request.body_request_id(), backend);
    let preprocessed = match request.into_preprocessed(
        &model,
        request_context.data_parallel_rank,
        &request_context.request_id,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(error) => {
            return backend.error_response(
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
            Err(response) => return adapt_openai_error(backend, response),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: if streaming { "streaming" } else { "unary" }.to_string(),
    };
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = generate_dispatch_span(&request_id);
    let response = match tokio::spawn(
        generate_dispatch(
            engine,
            context,
            request_id,
            model,
            state.clone(),
            response_format,
            backend,
            streaming,
            stream_handle,
        )
        .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "generate dispatch task panicked");
            generate_internal_error_response(backend)
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
    response_format: GenerateResponseFormat,
    backend: GenerateBackend,
    streaming: bool,
    stream_handle: ConnectionHandle,
) -> Response {
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        state.manager().metric_model_for(&model),
        super::metrics::Endpoint::Generate,
        streaming,
        &request_id,
    );
    let request_context = context.context();
    let generate_result =
        match run_until_killed(request_context.as_ref(), engine.generate(context)).await {
            Some(result) => result,
            None => {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response(backend);
            }
        };
    if request_context.is_killed() {
        inflight_guard.mark_error(ErrorType::Cancelled);
        return generate_cancelled_response(backend);
    }
    let stream = match generate_result {
        Ok(stream) => stream,
        Err(error) => {
            let was_cancelled = request_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref());
            let was_rejected = super::metrics::request_was_rejected(error.as_ref());
            let invalid_argument = find_invalid_argument_in_chain(error.as_ref());
            inflight_guard.mark_error(if was_cancelled {
                ErrorType::Cancelled
            } else if was_rejected {
                ErrorType::Unavailable
            } else if invalid_argument.is_some() {
                ErrorType::Validation
            } else {
                ErrorType::Internal
            });
            if was_cancelled {
                return generate_cancelled_response(backend);
            }
            if was_rejected {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected generate request");
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
                return backend.error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    "engine rejected the request".to_string(),
                );
            }
            if let Some(invalid_argument) = invalid_argument {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected invalid generate request");
                return backend.error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    invalid_argument.message().to_string(),
                );
            }
            tracing::error!(%request_id, error = %format!("{error:#}"), "engine generate call failed");
            return generate_internal_error_response(backend);
        }
    };

    let engine_context = stream.context();
    if streaming {
        let GenerateResponseFormat::Sglang(options) = response_format else {
            unreachable!("vLLM streaming requests are rejected before dispatch")
        };
        let stream =
            SglangGenerateStream::from_annotated_stream(stream, request_id.clone(), options).map(
                |result| {
                    result
                        .map(|value| Event::default().data(value.to_string()))
                        .map_err(axum::Error::new)
                },
            );
        let stream = monitor_for_disconnects(stream, engine_context, inflight_guard, stream_handle);
        let mut response = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            response = response.keep_alive(KeepAlive::default().interval(keep_alive));
        }
        return response.into_response();
    }

    let aggregate = async {
        match response_format {
            GenerateResponseFormat::Vllm(options) => {
                GenerateResponse::from_annotated_stream_with_options(
                    stream,
                    request_id.clone(),
                    options,
                )
                .await
                .map(AggregatedGenerateResponse::Vllm)
            }
            GenerateResponseFormat::Sglang(options) => {
                SglangGenerateResponse::from_annotated_stream(stream, request_id.clone(), options)
                    .await
                    .map(AggregatedGenerateResponse::Sglang)
            }
        }
    };
    let response_result = match run_until_killed(request_context.as_ref(), aggregate).await {
        Some(result) => result,
        None => {
            inflight_guard.mark_error(ErrorType::Cancelled);
            return generate_cancelled_response(backend);
        }
    };
    match response_result {
        Ok(response) => {
            if request_context.is_killed() || engine_context.is_killed() {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response(backend);
            }
            if !response.is_complete_unary() {
                inflight_guard.mark_error(ErrorType::Internal);
                tracing::error!(%request_id, "generate stream ended without a complete choice");
                return generate_internal_error_response(backend);
            }
            inflight_guard.mark_ok();
            response.into_response()
        }
        Err(error) => {
            if request_context.is_killed()
                || engine_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref())
            {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_cancelled_response(backend);
            }
            if let Some(invalid_argument) = find_invalid_argument_in_chain(error.as_ref()) {
                inflight_guard.mark_error(ErrorType::Validation);
                tracing::warn!(%request_id, %error, "invalid generate stream response");
                return backend.error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    invalid_argument.message().to_string(),
                );
            }
            inflight_guard.mark_error(ErrorType::Internal);
            tracing::error!(%request_id, %error, "failed to fold generate stream");
            generate_internal_error_response(backend)
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

    use super::service_v2::{
        HTTP_SVC_SGLANG_GENERATE_PATH_ENV, HTTP_SVC_VLLM_GENERATE_PATH_ENV, HttpService,
        SGLANG_ENABLE_GENERATE_ENV, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV,
    };
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

    struct SglangStreamEngine;

    struct CancelledEngine;

    struct InvalidArgumentEngine;

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
        for InvalidArgumentEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Backend(
                    dynamo_runtime::error::BackendError::InvalidArgument,
                ))
                .message("invalid SGLang sampling parameter")
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
        for SglangStreamEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![101],
                    text: Some("a".to_string()),
                    index: Some(0),
                    engine_data: Some(serde_json::json!({
                        "sglang_meta_info": {"finish_reason": null, "prompt_tokens": 1}
                    })),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![102],
                    text: Some("b".to_string()),
                    index: Some(0),
                    finish_reason: Some(crate::protocols::common::FinishReason::Length),
                    engine_data: Some(serde_json::json!({
                        "sglang_meta_info": {
                            "finish_reason": {"type": "length", "length": 2},
                            "prompt_tokens": 1,
                            "completion_tokens": 2
                        }
                    })),
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
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn sglang_generate_post_and_put_use_native_error_shape() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/generate", port);
        for request in [
            client
                .post(&url)
                .json(&serde_json::json!({"input_ids": [1, 2, 3]})),
            client
                .put(&url)
                .json(&serde_json::json!({"input_ids": [1, 2, 3]})),
        ] {
            let resp = request
                .send()
                .await
                .expect("SGLang generate request failed");
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);
            let body: serde_json::Value = resp.json().await.expect("json body");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("no generate-capable model"))
            );
            assert!(body["error"].get("type").is_none());
            assert!(body["error"].get("code").is_none());
        }
        let resp = client
            .post(&url)
            .json(&serde_json::json!({"input_ids": "bad"}))
            .send()
            .await
            .expect("SGLang generate request failed");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["object"], "error");
        assert_eq!(body["type"], "Bad Request");
        assert_eq!(body["code"], 400);

        let resp = client
            .post(&url)
            .json(&serde_json::json!({
                "input_ids": [1, 2, 3],
                "sampling_params": {"n": 2}
            }))
            .send()
            .await
            .expect("SGLang generate request failed");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("must be 1")
        );
        handle.abort();
    }

    #[tokio::test]
    async fn sglang_generate_rejects_unsupported_top_level_fields() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let private_fields = [
            (
                "bootstrap_info",
                serde_json::json!({"bootstrap_host": "client"}),
            ),
            ("bootstrap_host", serde_json::json!("client")),
            ("bootstrap_port", serde_json::json!(1234)),
            ("bootstrap_room", serde_json::json!(7)),
            (
                "disaggregated_params",
                serde_json::json!({"bootstrap_room": 7}),
            ),
            (
                "kv_transfer_params",
                serde_json::json!({"remote": "client"}),
            ),
            ("cache_salt", serde_json::json!("tenant-a")),
            ("future_vllm_field", serde_json::json!(true)),
        ];

        for (field, value) in private_fields {
            let mut body = serde_json::json!({
                "input_ids": [1, 2, 3]
            });
            body[field] = value;
            let resp = client
                .post(format!("http://localhost:{}/generate", port))
                .json(&body)
                .send()
                .await
                .expect("SGLang generate request failed");
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "field={field}");
            let error: serde_json::Value = resp.json().await.expect("json body");
            assert!(error["error"]["message"].is_string());
            assert!(error["error"].get("type").is_none());
            assert!(error["error"].get("code").is_none());
        }

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
    async fn sglang_streaming_dispatch_emits_incremental_sse_and_done() {
        let service = HttpService::builder().build().unwrap();
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1],
            "stream": true
        }))
        .unwrap();
        let response = generate_dispatch(
            Arc::new(SglangStreamEngine),
            dispatch_test_context(),
            "req-sglang-stream".to_string(),
            "test-model".to_string(),
            service.state_clone(),
            GenerateResponseFormat::Sglang(request.response_options()),
            GenerateBackend::Sglang,
            true,
            disabled_stream_handle(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        let data: Vec<_> = body
            .lines()
            .filter_map(|line| {
                line.strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
            })
            .collect();
        assert_eq!(data.len(), 3);
        let first: serde_json::Value = serde_json::from_str(data[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(data[1]).unwrap();
        assert_eq!(first["text"], "a");
        assert_eq!(first["output_ids"], serde_json::json!([101]));
        assert_eq!(first["meta_info"]["finish_reason"], serde_json::Value::Null);
        assert_eq!(second["text"], "b");
        assert_eq!(second["output_ids"], serde_json::json!([102]));
        assert_eq!(second["meta_info"]["finish_reason"]["type"], "length");
        assert_eq!(data[2], "[DONE]");
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
            [
                (VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>),
                (SGLANG_ENABLE_GENERATE_ENV, None::<&str>),
                (HTTP_SVC_VLLM_GENERATE_PATH_ENV, None::<&str>),
                (HTTP_SVC_SGLANG_GENERATE_PATH_ENV, None::<&str>),
            ],
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
            [
                (VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1")),
                (SGLANG_ENABLE_GENERATE_ENV, None),
                (HTTP_SVC_VLLM_GENERATE_PATH_ENV, None),
                (HTTP_SVC_SGLANG_GENERATE_PATH_ENV, None),
            ],
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
            preprocessed_from_vllm_generate(request, "test-model", None, "resolved-request")
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
        assert_eq!(expected_token_ids, serde_json::json!([1, 2]));
        assert_eq!(envelope, &expected_envelope);
        assert!(envelope.get("token_ids").is_none());
    }

    #[test]
    fn sglang_projection_uses_native_envelope_and_controls() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [11, 12],
            "sampling_params": {
                "max_new_tokens": 8,
                "min_new_tokens": 2,
                "temperature": 0.3,
                "sampling_seed": 4
            },
            "return_logprob": true,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "return_routed_experts": true,
            "routed_experts_start_len": 4,
            "priority": 7
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_sglang_generate(request, "test-model", Some(3))
            .expect("build SGLang request");

        assert_eq!(preprocessed.token_ids, vec![11, 12]);
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(2));
        assert_eq!(preprocessed.sampling_options.n, Some(1));
        assert_eq!(preprocessed.output_options.logprobs, Some(2));
        assert_eq!(preprocessed.output_options.prompt_logprobs, Some(2));
        assert_eq!(
            preprocessed.output_options.return_tokens_as_token_ids,
            Some(true)
        );
        let routing = preprocessed.routing.as_ref().expect("routing hints");
        assert_eq!(routing.dp_rank, Some(3));
        assert_eq!(routing.priority, Some(7));

        let extra_args = preprocessed.extra_args.as_ref().expect("extra args");
        assert!(extra_args.get("vllm_tito").is_none());
        let envelope = extra_args.get("sglang_tito").expect("sglang_tito envelope");
        assert_eq!(envelope["sampling_params"]["sampling_seed"], 4);
        assert_eq!(envelope["return_logprob"], true);
        assert_eq!(envelope["logprob_start_len"], 0);
        assert_eq!(envelope["top_logprobs_num"], 2);
        assert_eq!(envelope["return_routed_experts"], true);
        assert_eq!(envelope["routed_experts_start_len"], 4);
        assert!(envelope.get("rid").is_none());
        assert!(envelope.get("input_ids").is_none());
        assert!(envelope.get("priority").is_none());
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
            preprocessed_from_vllm_generate(request, "test-model", None, "resolved-request")
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
            preprocessed_from_vllm_generate(request, "test-model", None, "resolved-request")
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "3".parse().unwrap());

        let context =
            resolve_generate_request_context(&headers, Some("body-request"), GenerateBackend::Vllm);

        assert_eq!(context.request_id, "header-request");
        assert_eq!(context.data_parallel_rank, Some(3));
    }

    #[test]
    fn generate_request_context_matches_sglang_body_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());

        let context = resolve_generate_request_context(
            &headers,
            Some("body-request"),
            GenerateBackend::Sglang,
        );

        assert_eq!(context.request_id, "body-request");
    }

    #[test]
    fn generate_request_context_falls_back_and_ignores_invalid_dp_rank() {
        let mut headers = HeaderMap::new();
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "invalid".parse().unwrap());

        let context =
            resolve_generate_request_context(&headers, Some("body-request"), GenerateBackend::Vllm);

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

    fn disabled_stream_handle() -> ConnectionHandle {
        let (sender, _receiver) = tokio::sync::oneshot::channel();
        ConnectionHandle::create_disabled(sender)
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
            GenerateResponseFormat::Vllm(GenerateResponseOptions::default()),
            GenerateBackend::Vllm,
            false,
            disabled_stream_handle(),
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
            GenerateResponseFormat::Vllm(GenerateResponseOptions::default()),
            GenerateBackend::Vllm,
            false,
            disabled_stream_handle(),
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
            GenerateResponseFormat::Vllm(GenerateResponseOptions::default()),
            GenerateBackend::Vllm,
            false,
            disabled_stream_handle(),
        )
        .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn worker_invalid_argument_returns_400() {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(InvalidArgumentEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();

        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-invalid-argument".to_string(),
            "test-model".to_string(),
            state,
            GenerateResponseFormat::Vllm(GenerateResponseOptions::default()),
            GenerateBackend::Vllm,
            false,
            disabled_stream_handle(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read validation response");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("parse response");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid SGLang")
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

        let preprocessed =
            preprocessed_from_vllm_generate(request, "test-model", Some(3), "resolved-request")
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
