// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `POST`/`PUT /generate` HTTP adapter.
//!
//! The public body is parsed only far enough to project Dynamo routing controls;
//! engine-owned fields stay opaque and are forwarded to the SGLang worker.
//! Only SGLang's incremental SSE mode is currently exposed.

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

use super::disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects};
use super::metrics::{CancellationLabels, ErrorType};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, find_invalid_argument_in_chain,
    get_body_limit, get_or_create_request_id,
};
use super::{RouteDoc, service_v2};
use crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use crate::protocols::sglang::generate::SglangGenerateRequest;
use crate::protocols::sglang::stream::SglangGenerateStream;

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";
pub(super) const DEFAULT_PATH: &str = "/generate";

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
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody { message },
        }),
    )
        .into_response()
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
    let mut extra_args = serde_json::Map::new();
    extra_args.insert(
        "sglang_tito".to_string(),
        request.worker_envelope(request_id)?,
    );

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(request.input_ids)
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
            expected_output_tokens: max_tokens,
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::Value::Object(extra_args)))
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
    if !request.stream {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "non-streaming SGLang generate requests are not implemented; set stream=true"
                .to_string(),
        );
    }

    let models = state
        .manager()
        .list_generate_models_for_capability(SGLANG_GENERATE_CAPABILITY);
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
    let engine = match state
        .manager()
        .get_generate_engine_for_capability(&model, SGLANG_GENERATE_CAPABILITY)
    {
        Ok(engine) => engine,
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
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return adapt_openai_error(response),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "streaming".to_string(),
    };
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = tracing::info_span!(
        target: "request_span",
        "generate",
        request_id = %request_id
    );
    let response = match tokio::spawn(
        dispatch(
            engine,
            context,
            request_id,
            model,
            state.clone(),
            stream_handle,
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
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
    stream_handle: ConnectionHandle,
) -> Response {
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        state.manager().metric_model_for(&model),
        super::metrics::Endpoint::Generate,
        true,
        &request_id,
    );
    let request_context = context.context();
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
                return cancelled_response();
            }
            if was_rejected {
                tracing::warn!(%request_id, error = %format!("{error:#}"), "engine rejected SGLang generate request");
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "engine rejected the request".to_string(),
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
    let stream = SglangGenerateStream::from_annotated_stream(stream, request_id).map(|result| {
        result
            .map(|value| Event::default().data(value.to_string()))
            .map_err(axum::Error::new)
    });
    let stream = monitor_for_disconnects(stream, engine_context, inflight_guard, stream_handle);
    let mut response = Sse::new(stream);
    if let Some(keep_alive) = state.sse_keep_alive() {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    response.into_response()
}
