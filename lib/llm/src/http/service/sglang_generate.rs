// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `POST`/`PUT /generate` HTTP adapter.
//!
//! The public body is parsed only far enough to project Dynamo routing controls;
//! engine-owned fields stay opaque and are forwarded to the SGLang worker.
//! Workers stream SGLang's incremental output; `stream=false` joins it.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
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

use super::disconnect::{
    ConnectionHandle, create_connection_monitor, monitor_for_disconnects_with_error,
};
use super::engine_output_metrics::{GenerateMetricCollector, generate_dispatch_span};
use super::error::SanitizedError;
use super::metrics::{CancellationLabels, ErrorType};
use super::openai::{
    check_model_serving_ready, check_ready, find_invalid_argument_in_chain, get_body_limit,
    get_or_create_request_id, preprocessed_context_from_headers,
};
use super::{RouteDoc, service_v2};
use crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use crate::protocols::response_merge::{
    PREFILL_OUTPUT_ANNOTATION, PREFILL_OUTPUT_FIRST_ANNOTATION, PrefillNeed,
};
use crate::protocols::sglang::generate::SglangGenerateRequest;
use crate::protocols::sglang::merge::merger;
use crate::protocols::sglang::stream::SglangGenerateStream;
use crate::request_trace::{
    RequestEndTraceState, build_request_end_trace_state, wrap_engine_output_request_end_stream,
};

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";
pub(super) const DEFAULT_PATH: &str = "/generate";

fn canonical_generate_models(
    manager: &crate::discovery::ModelManager,
    models: Vec<String>,
) -> Vec<String> {
    models
        .into_iter()
        .map(|model| manager.resolve_canonical_name(&model))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[derive(Debug)]
struct RequestContext {
    request_id: String,
    data_parallel_rank: Option<u32>,
}

#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
}
fn generate_error(message: String) -> GenerateError {
    GenerateError {
        error: GenerateErrorBody { message },
    }
}

#[derive(Serialize, Debug)]
struct ValidationError {
    object: &'static str,
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<&'static str>,
    code: u16,
}

pub fn router(state: Arc<service_v2::State>, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    let docs = vec![
        RouteDoc::new(axum::http::Method::POST, &path),
        RouteDoc::new(axum::http::Method::PUT, &path),
    ];
    let router = Router::new()
        .route(&path, post(handler).put(handler))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (docs, router)
}

fn error_response(code: StatusCode, message: String) -> Response {
    (code, Json(generate_error(message))).into_response()
}
fn stream_error(error: &(dyn std::error::Error + 'static)) -> (ErrorType, String) {
    // Deadline is checked before overload so a chain carrying both markers keeps
    // the deadline outcome, matching the OpenAI surface.
    let (error_type, message) = if super::metrics::request_deadline_exceeded(error) {
        (
            ErrorType::Cancelled,
            super::metrics::REQUEST_DEADLINE_EXCEEDED_MESSAGE.to_string(),
        )
    } else if super::metrics::request_was_rejected(error) {
        (ErrorType::Overload, SanitizedError::Overloaded.to_string())
    } else if super::metrics::request_was_unavailable(error) {
        (
            ErrorType::Unavailable,
            SanitizedError::Unavailable.to_string(),
        )
    } else if let Some(error) = find_invalid_argument_in_chain(error) {
        (ErrorType::Validation, error.message().to_string())
    } else if super::metrics::request_was_cancelled(error) {
        (ErrorType::Cancelled, SanitizedError::Cancelled.to_string())
    } else {
        (ErrorType::Internal, SanitizedError::Internal.to_string())
    };
    let body = serde_json::to_string(&generate_error(message)).expect("serializable error");
    (error_type, body)
}

fn adapt_openai_error(response: super::openai::ErrorResponse) -> Response {
    let (status, Json(error)) = response;
    error_response(status, error.message().to_string())
}

fn resolve_request_context(headers: &HeaderMap, body_request_id: Option<&str>) -> RequestContext {
    let request_id = body_request_id
        .map(ToOwned::to_owned)
        .or_else(|| {
            headers
                .get(X_REQUEST_ID_HEADER)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| get_or_create_request_id(headers));
    let data_parallel_rank = headers
        .get(X_DATA_PARALLEL_RANK_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());

    RequestContext {
        request_id,
        data_parallel_rank,
    }
}

async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

fn cancelled_response() -> Response {
    error_response(
        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
        "request was cancelled".to_string(),
    )
}

fn internal_error_response() -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal server error".to_string(),
    )
}

/// Caller-supplied deadline elapsed before the engine produced a stream: HTTP
/// 429 with a `Cancelled` metric label, the same contract as the OpenAI
/// surface's `ErrorMessage::from_anyhow` deadline branch.
fn deadline_exceeded_response() -> Response {
    error_response(
        StatusCode::TOO_MANY_REQUESTS,
        super::metrics::REQUEST_DEADLINE_EXCEEDED_MESSAGE.to_string(),
    )
}

fn preprocessed_request(
    request: SglangGenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
    request_id: &str,
) -> anyhow::Result<PreprocessedRequest> {
    let max_tokens = request.max_new_tokens().map_err(anyhow::Error::msg)?;
    let min_tokens = request.min_new_tokens().map_err(anyhow::Error::msg)?;
    let ignore_eos = request.ignore_eos().map_err(anyhow::Error::msg)?;
    let routing_priority = request.priority.unwrap_or_default();
    let cache_namespace = request.cache_namespace().map(str::to_owned);
    let streaming = request.stream;
    let (input_ids, worker_envelope) = request.into_worker_envelope(request_id);
    // The router records TTFT and ITL into the request's tracker.
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(input_ids.len(), None);
    // Hold decode output for prefill only when every streamed chunk needs it.
    let annotations = match merger().prefill_need(&worker_envelope) {
        PrefillNeed::None => vec![],
        PrefillNeed::BeforeFirst if streaming => vec![
            PREFILL_OUTPUT_ANNOTATION.to_string(),
            PREFILL_OUTPUT_FIRST_ANNOTATION.to_string(),
        ],
        _ => vec![PREFILL_OUTPUT_ANNOTATION.to_string()],
    };
    let mut extra_args = serde_json::Map::new();
    extra_args.insert("sglang_tito".to_string(), worker_envelope);

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(input_ids)
        .stop_conditions(StopConditions {
            max_tokens,
            min_tokens,
            ignore_eos,
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(OutputOptions {
            return_tokens_as_token_ids: Some(true),
            ..Default::default()
        })
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            cache_namespace,
            ..Default::default()
        }))
        .extra_args(Some(serde_json::Value::Object(extra_args)))
        .annotations(annotations)
        .tracker(Some(tracker))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

async fn handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    request: Result<Json<SglangGenerateRequest>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ValidationError {
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

    if let Err(response) = check_ready(&state) {
        return adapt_openai_error(response);
    }
    if let Err(message) = request.validate() {
        return error_response(StatusCode::BAD_REQUEST, message);
    }
    let streaming = request.stream;

    let models = canonical_generate_models(
        state.manager(),
        state
            .manager()
            .list_generate_models_for_capability(SGLANG_GENERATE_CAPABILITY),
    );
    let model = match models.len() {
        1 => models.into_iter().next().unwrap(),
        0 => {
            return error_response(
                StatusCode::NOT_FOUND,
                "no generate-capable model is registered".to_string(),
            );
        }
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "multiple SGLang models are registered; configure a model-specific generate endpoint"
                    .to_string(),
            );
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return adapt_openai_error(response);
    }
    let selection = match state
        .manager()
        .get_generate_engine_for_capability_with_routing(&model, SGLANG_GENERATE_CAPABILITY)
    {
        Ok(selection) => selection,
        Err(error) => {
            let status = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    StatusCode::SERVICE_UNAVAILABLE
                }
                _ => StatusCode::NOT_FOUND,
            };
            return error_response(status, error.to_string());
        }
    };

    let request_context = resolve_request_context(&headers, request.rid.as_deref());
    let preprocessed = match preprocessed_request(
        request,
        &model,
        request_context.data_parallel_rank,
        &request_context.request_id,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.to_string()),
    };

    let request_id = request_context.request_id;
    let context: Context<PreprocessedRequest> =
        match preprocessed_context_from_headers(preprocessed, request_id, &headers) {
            Ok(context) => context,
            Err(response) => return adapt_openai_error(response),
        };
    let trace_state = build_request_end_trace_state(
        &context,
        &context.tracker,
        &context,
        selection.kv_cache_block_size as usize,
    );
    let metric_model = state.manager().metric_model_for(&model).to_string();
    let cancellation_labels = CancellationLabels {
        model: metric_model.clone(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: if streaming { "streaming" } else { "unary" }.to_string(),
    };
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.context(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = generate_dispatch_span(context.id(), &metric_model);
    let response = match tokio::spawn(
        dispatch(
            selection.engine,
            context,
            metric_model,
            state.clone(),
            stream_handle,
            streaming,
            trace_state,
        )
        .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "SGLang generate dispatch task panicked");
            internal_error_response()
        }
    };

    connection_handle.disarm();
    response
}

async fn dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    metric_model: String,
    state: Arc<service_v2::State>,
    stream_handle: ConnectionHandle,
    streaming: bool,
    trace_state: Option<RequestEndTraceState>,
) -> Response {
    let request_context = context.context();
    let request_id = request_context.id();
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        super::metrics::Endpoint::Generate,
        streaming,
        request_id,
    );
    let mut metric_collector = GenerateMetricCollector::new(
        state.metrics_clone(),
        &metric_model,
        context.tracker.clone().unwrap_or_default(),
        context.token_ids.len(),
    );
    let generate_result =
        match run_until_killed(request_context.as_ref(), engine.generate(context)).await {
            Some(result) => result,
            None => {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return cancelled_response();
            }
        };
    if request_context.is_killed() {
        inflight_guard.mark_error(ErrorType::Cancelled);
        return cancelled_response();
    }
    let stream = match generate_result {
        Ok(stream) => stream,
        Err(error) => {
            let deadline_exceeded = super::metrics::request_deadline_exceeded(error.as_ref());
            let was_cancelled = request_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref());
            let was_rejected = super::metrics::request_was_rejected(error.as_ref());
            let was_unavailable = super::metrics::request_was_unavailable(error.as_ref());
            let invalid_argument = find_invalid_argument_in_chain(error.as_ref());
            inflight_guard.mark_error(if deadline_exceeded || was_cancelled {
                ErrorType::Cancelled
            } else if was_rejected || was_unavailable {
                ErrorType::Unavailable
            } else if invalid_argument.is_some() {
                ErrorType::Validation
            } else {
                ErrorType::Internal
            });
            if deadline_exceeded {
                return deadline_exceeded_response();
            }
            if was_cancelled {
                return cancelled_response();
            }
            if was_rejected {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected SGLang generate request");
                state
                    .metrics_clone()
                    .inc_rejection(&metric_model, super::metrics::Endpoint::Generate);
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "engine rejected the request".to_string(),
                );
            }
            if was_unavailable {
                tracing::warn!(
                    %request_id,
                    error = %format!("{error:#}"),
                    "no worker available for SGLang generate request"
                );
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    SanitizedError::Unavailable.to_string(),
                );
            }
            if let Some(invalid_argument) = invalid_argument {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected invalid SGLang generate request");
                return error_response(
                    StatusCode::BAD_REQUEST,
                    invalid_argument.message().to_string(),
                );
            }
            tracing::error!(%request_id, error = %format!("{error:#}"), "SGLang engine generate call failed");
            return internal_error_response();
        }
    };

    let engine_context = stream.context();
    let stream =
        wrap_engine_output_request_end_stream(stream, trace_state).map(move |mut output| {
            metric_collector.observe(&mut output);
            output
        });
    if !streaming {
        let fold = SglangGenerateStream::fold(stream, merger());
        return match run_until_killed(request_context.as_ref(), fold).await {
            Some(Ok(response)) if !request_context.is_killed() && !engine_context.is_killed() => {
                inflight_guard.mark_ok();
                Json(response).into_response()
            }
            Some(Err(error)) => {
                let (error_type, body) = stream_error(&error);
                if error_type == ErrorType::Internal {
                    tracing::error!(%request_id, %error, "SGLang generate stream failed");
                }
                let status = match error_type {
                    ErrorType::Validation => StatusCode::BAD_REQUEST,
                    ErrorType::Overload | ErrorType::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                    ErrorType::Cancelled => {
                        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST)
                    }
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                inflight_guard.mark_error(error_type);
                let json = [(axum::http::header::CONTENT_TYPE, "application/json")];
                (status, json, body).into_response()
            }
            _ => {
                inflight_guard.mark_error(ErrorType::Cancelled);
                cancelled_response()
            }
        };
    }
    // One buffer reused across chunks.
    let mut buffer = Vec::new();
    let render = move |response: &serde_json::Value| {
        buffer.clear();
        serde_json::to_writer(&mut buffer, response).map_err(axum::Error::new)?;
        let json = std::str::from_utf8(&buffer).map_err(axum::Error::new)?;
        Ok(Event::default().data(json))
    };
    let stream = SglangGenerateStream::from_annotated_stream(stream, merger(), render)
        .map(|result| result.map_err(axum::Error::new).and_then(|event| event));
    let stream = monitor_for_disconnects_with_error(
        stream,
        engine_context,
        inflight_guard,
        stream_handle,
        stream_error,
    );
    let mut response = Sse::new(stream);
    if let Some(keep_alive) = state.sse_keep_alive() {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    response.into_response()
}

#[cfg(test)]
#[path = "sglang_generate_tests.rs"]
mod tests;
