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
use super::error::SanitizedError;
use super::generate::GenerateMetricCollector;
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
use crate::request_trace::{RequestEndTraceState, wrap_engine_output_request_end_stream};

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
        match preprocessed_context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return adapt_openai_error(response),
        };
    let (request, unit_context) = context.into_parts();
    let trace_state = crate::request_trace::build_request_end_trace_state(
        &request,
        &request.tracker,
        &unit_context,
        selection.kv_cache_block_size as usize,
    );
    let context = Context::rejoin(request, unit_context);
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

    let dispatch_span = super::generate::generate_dispatch_span(
        &request_id,
        state.manager().metric_model_for(&model),
    );
    let response = match tokio::spawn(
        dispatch(
            selection.engine,
            context,
            request_id,
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
    request_id: String,
    state: Arc<service_v2::State>,
    stream_handle: ConnectionHandle,
    streaming: bool,
    trace_state: Option<RequestEndTraceState>,
) -> Response {
    let model = context.model.clone();
    let metric_model = state.manager().metric_model_for(&model);
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        metric_model,
        super::metrics::Endpoint::Generate,
        streaming,
        &request_id,
    );
    let mut metric_collector = GenerateMetricCollector::new(
        state.metrics_clone(),
        metric_model,
        context.tracker.clone().unwrap_or_default(),
        context.token_ids.len(),
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
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
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
mod tests {
    use super::*;

    use crate::http::service::generate::tests::{WorkerUnavailableEngine, dispatch_test_context};
    use crate::http::service::metrics::{Endpoint, RequestType, Status};
    use crate::protocols::common::llm_backend::LLMEngineOutput;
    use dynamo_runtime::protocols::maybe_error::MaybeError;

    #[test]
    fn cache_salt_matches_routing_and_native_payload() {
        for salt in [None, Some(""), Some("tenant-a")] {
            let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
                "input_ids": [1, 2],
                "cache_salt": salt,
            }))
            .unwrap();
            let request = preprocessed_request(request, "test-model", None, "request-id").unwrap();
            assert_eq!(request.tracker.unwrap().isl_tokens(), Some(2));
            let expected = salt.filter(|salt| !salt.is_empty());
            assert_eq!(
                request.routing.unwrap().cache_namespace.as_deref(),
                expected
            );
            assert_eq!(
                request.extra_args.unwrap()["sglang_tito"]["cache_salt"].as_str(),
                expected,
            );
        }
    }

    #[tokio::test]
    async fn worker_unavailable_dispatch_returns_503() {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            std::sync::Arc::new(WorkerUnavailableEngine);
        let state = crate::http::service::service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();

        let response = dispatch(
            engine,
            dispatch_test_context(),
            "req-sglang-worker-unavailable".to_string(),
            state.clone(),
            ConnectionHandle::create_disabled(tx),
            true,
            None,
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("parse error response");
        assert_eq!(body["error"]["message"], "Service temporarily unavailable");
        assert!(!body.to_string().contains("unknown endpoint"));

        let metric_model = state.manager().metric_model_for("test-model");
        assert_eq!(
            state.metrics_clone().get_request_counter(
                metric_model,
                &Endpoint::Generate,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Unavailable,
            ),
            1
        );
    }

    #[test]
    fn model_aliases_are_deduplicated_before_implicit_selection() {
        let manager = crate::discovery::ModelManager::new();
        assert!(manager.register_alias("alias", "primary"));

        let models = canonical_generate_models(
            &manager,
            vec![
                "primary".to_string(),
                "alias".to_string(),
                "other".to_string(),
            ],
        );

        assert_eq!(models, vec!["other".to_string(), "primary".to_string()]);
    }

    #[test]
    fn streaming_errors_use_sglang_shape() {
        let error = anyhow::anyhow!("backend failure");
        let (error_type, body) = stream_error(error.as_ref());
        let error: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error_type, ErrorType::Internal);
        assert_eq!(
            error,
            serde_json::json!({"error": {"message": "Internal server error"}})
        );
    }

    #[test]
    fn streaming_errors_preserve_typed_classification() {
        use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

        for (dynamo_type, expected_type, expected_message) in [
            (
                DynamoErrorType::InvalidArgument,
                ErrorType::Validation,
                "invalid sampling parameters",
            ),
            (
                DynamoErrorType::Unavailable,
                ErrorType::Unavailable,
                "Service temporarily unavailable",
            ),
            (
                DynamoErrorType::WorkerUnavailable,
                ErrorType::Unavailable,
                "Service temporarily unavailable",
            ),
            (
                DynamoErrorType::Cancelled,
                ErrorType::Cancelled,
                "Request cancelled",
            ),
        ] {
            let error = DynamoError::builder()
                .error_type(dynamo_type)
                .message("invalid sampling parameters")
                .build();
            let error = axum::Error::new(error);
            let (error_type, body) = stream_error(&error);
            let body: serde_json::Value = serde_json::from_str(&body).unwrap();

            assert_eq!(error_type, expected_type);
            assert_eq!(body["error"]["message"], expected_message);
        }
    }

    #[test]
    fn streaming_deadline_wins_over_overload_classification() {
        use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

        let error = DynamoError::builder()
            .error_type(DynamoErrorType::WorkerOverloaded)
            .message("worker overloaded")
            .cause(
                DynamoError::builder()
                    .error_type(DynamoErrorType::DeadlineExceeded)
                    .reason(
                        dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded")
                            .unwrap(),
                    )
                    .message("internal deadline detail")
                    .build(),
            )
            .build();
        let (error_type, body) = stream_error(&error);
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();

        assert_eq!(error_type, ErrorType::Cancelled);
        assert_eq!(body["error"]["message"], "request deadline exceeded");
    }

    struct DeadlineEngine;

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<
                crate::protocols::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
            >,
            dynamo_runtime::pipeline::Error,
        > for DeadlineEngine
    {
        async fn generate(
            &self,
            _request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<
                crate::protocols::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
            >,
            dynamo_runtime::pipeline::Error,
        > {
            Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::DeadlineExceeded)
                .reason(
                    dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded")
                        .unwrap(),
                )
                .message("router deadline exceeded before stream start")
                .build()
                .into())
        }
    }

    #[tokio::test]
    async fn pre_stream_deadline_maps_to_http_429_with_cancelled_metric() {
        use crate::http::service::metrics::{Endpoint, RequestType, Status};

        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(DeadlineEngine);
        let service = service_v2::HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let context = Context::new(
            PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .expect("build dispatch test request"),
        );
        let (mut connection_handle, stream_handle) = create_connection_monitor(
            context.context(),
            None,
            CancellationLabels {
                model: "test-model".to_string(),
                endpoint: Endpoint::Generate.to_string(),
                request_type: "streaming".to_string(),
            },
        )
        .await;

        let response = dispatch(
            engine,
            context,
            "req-deadline".to_string(),
            state.clone(),
            stream_handle,
            true,
            None,
        )
        .await;
        connection_handle.disarm();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["message"], "request deadline exceeded");
        assert_eq!(
            state.metrics_clone().get_request_counter(
                state.manager().metric_model_for("test-model"),
                &Endpoint::Generate,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );
    }

    struct ReplayEngine(Vec<crate::protocols::Annotated<LLMEngineOutput>>);

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for ReplayEngine
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
            // Pace chunks so TTFT and ITL observe distinct arrival times.
            let stream = Box::pin(futures::stream::iter(self.0.clone()).then(|output| async {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                output
            }));
            Ok(dynamo_runtime::engine::ResponseStream::new(
                stream,
                request.context(),
            ))
        }
    }

    struct FirstChunkThenPendingEngine;

    #[async_trait::async_trait]
    impl
        dynamo_runtime::engine::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > for FirstChunkThenPendingEngine
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<
            dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
            dynamo_runtime::pipeline::Error,
        > {
            let stream =
                futures::stream::iter([sglang_chunk(vec![5])]).chain(futures::stream::pending());
            Ok(dynamo_runtime::engine::ResponseStream::new(
                Box::pin(stream),
                request.context(),
            ))
        }
    }

    fn sglang_chunk(token_ids: Vec<u32>) -> crate::protocols::Annotated<LLMEngineOutput> {
        crate::protocols::Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": token_ids}})),
            token_ids,
            index: Some(0),
            ..Default::default()
        })
    }

    /// A P/D stream: prefill's output item, then two decode chunks with
    /// terminal finish reason and usage.
    fn observed_outputs() -> Vec<crate::protocols::Annotated<LLMEngineOutput>> {
        let prefill = crate::protocols::Annotated {
            event: Some(PREFILL_OUTPUT_ANNOTATION.to_string()),
            ..crate::protocols::Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({"sglang_response": {"meta_info": {}}})),
                ..Default::default()
            })
        };
        let mut last = sglang_chunk(vec![6]);
        let output = last.data.as_mut().unwrap();
        output.finish_reason = Some(crate::protocols::common::FinishReason::Stop);
        output.completion_usage = Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
            prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                cached_tokens: Some(2),
                ..Default::default()
            }),
            completion_tokens_details: None,
        });
        vec![prefill, sglang_chunk(vec![5]), last]
    }

    fn observed_context(request_id: &str, streaming: bool) -> Context<PreprocessedRequest> {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1, 2, 3],
            "stream": streaming,
        }))
        .unwrap();
        let request = preprocessed_request(request, "test-model", None, request_id).unwrap();
        Context::with_id_and_metadata(request, request_id.to_string(), Default::default())
    }

    /// Attach agent context and x-request-id as the handler's header parsing
    /// does, then build trace state through the policy seam.
    fn traced_context(
        mut context: Context<PreprocessedRequest>,
        trace_enabled: bool,
    ) -> (Context<PreprocessedRequest>, Option<RequestEndTraceState>) {
        context.agent_context = Some(crate::protocols::common::extensions::AgentContext {
            session_id: "sglang-session".to_string(),
            parent_session_id: None,
            session_final: None,
            agent_headers: Default::default(),
            input_trigger: None,
        });
        context.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "client-x-request-id".to_string(),
        );
        let (request, unit_context) = context.into_parts();
        let trace_state = crate::request_trace::build_request_end_trace_state_for_policy(
            &request,
            &request.tracker,
            &unit_context,
            2,
            trace_enabled,
        );
        (Context::rejoin(request, unit_context), trace_state)
    }

    async fn request_end_records(
        receiver: &mut tokio::sync::broadcast::Receiver<crate::request_trace::RequestTraceRecord>,
        request_id: &str,
    ) -> Vec<crate::request_trace::RequestTraceRecord> {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
        let mut records = Vec::new();
        while let Ok(result) = tokio::time::timeout_at(deadline, receiver.recv()).await {
            match result {
                Ok(record)
                    if record
                        .request
                        .as_ref()
                        .is_some_and(|request| request.request_id == request_id) =>
                {
                    records.push(record);
                }
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
        records
    }

    #[tokio::test]
    async fn dispatch_records_frontend_metrics() {
        use crate::http::service::generate::tests::metric_value;

        for (streaming, worker_id) in [(false, 7_654_321_001_u64), (true, 7_654_321_002)] {
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let registry = prometheus::Registry::new();
            state.metrics_clone().register(&registry).unwrap();
            let context = observed_context("req-sglang-metrics", streaming);
            context.tracker.as_ref().unwrap().record_worker(
                worker_id,
                Some(0),
                crate::discovery::WORKER_TYPE_DECODE,
            );
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let response = dispatch(
                Arc::new(ReplayEngine(observed_outputs())),
                context,
                "req-sglang-metrics".to_string(),
                state.clone(),
                ConnectionHandle::create_disabled(tx),
                streaming,
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();

            let metric_model = state.manager().metric_model_for("test-model");
            let request_type = if streaming {
                RequestType::Stream
            } else {
                RequestType::Unary
            };
            assert_eq!(state.metrics_clone().get_inflight_count(metric_model), 0);
            assert_eq!(
                state.metrics_clone().get_request_counter(
                    metric_model,
                    &Endpoint::Generate,
                    &request_type,
                    &Status::Success,
                    &ErrorType::None,
                ),
                1
            );
            let families = registry.gather();
            let labels = [("model", metric_model)];
            let histogram = |name| metric_value(&families, name, &labels).get_histogram();
            assert_eq!(
                histogram("dynamo_frontend_input_sequence_tokens").get_sample_sum(),
                3.0
            );
            assert_eq!(
                histogram("dynamo_frontend_output_sequence_tokens").get_sample_sum(),
                2.0
            );
            assert_eq!(
                histogram("dynamo_frontend_cached_tokens").get_sample_sum(),
                2.0
            );
            assert_eq!(
                histogram("dynamo_frontend_time_to_first_token_seconds").get_sample_count(),
                1
            );
            assert_eq!(
                histogram("dynamo_frontend_inter_token_latency_seconds").get_sample_count(),
                1
            );
            assert_eq!(
                metric_value(&families, "dynamo_frontend_output_tokens_total", &labels)
                    .get_counter()
                    .value(),
                2.0
            );
            let worker_id = worker_id.to_string();
            let worker_labels = [
                worker_id.as_str(),
                "0",
                crate::discovery::WORKER_TYPE_DECODE,
            ];
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
        }
    }

    #[tokio::test]
    async fn dispatch_emits_request_end_trace_once() {
        crate::request_trace::init_bus_for_test(64);
        let mut receiver = crate::request_trace::subscribe();
        for (streaming, trace_enabled) in [(false, true), (true, true), (false, false)] {
            let request_id = format!("req-sglang-trace-{streaming}-{trace_enabled}");
            let (context, trace_state) =
                traced_context(observed_context(&request_id, streaming), trace_enabled);
            assert_eq!(trace_state.is_some(), trace_enabled);
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let response = dispatch(
                Arc::new(ReplayEngine(observed_outputs())),
                context,
                request_id.clone(),
                state,
                ConnectionHandle::create_disabled(tx),
                streaming,
                trace_state,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();

            let records = request_end_records(&mut receiver, &request_id).await;
            if !trace_enabled {
                assert!(records.is_empty());
                continue;
            }
            assert_eq!(records.len(), 1, "streaming={streaming}");
            let record = &records[0];
            assert_eq!(
                record.agent_context.as_ref().unwrap().session_id,
                "sglang-session"
            );
            let request = record.request.as_ref().unwrap();
            assert_eq!(request.x_request_id.as_deref(), Some("client-x-request-id"));
            assert_eq!(request.model.as_deref(), Some("test-model"));
            assert_eq!(request.input_tokens, Some(3));
            assert_eq!(request.output_tokens, Some(2));
            assert_eq!(request.cached_tokens, Some(2));
            assert_eq!(request.replay.as_ref().unwrap().input_length, 3);
            let finish = request.finish_reason_metadata.as_ref().unwrap();
            assert_eq!(finish.backend_finish_reason.as_deref(), Some("stop"));
        }
    }

    #[tokio::test]
    async fn streaming_cancellation_emits_request_end_trace_once() {
        crate::request_trace::init_bus_for_test(64);
        let mut receiver = crate::request_trace::subscribe();
        let request_id = "req-sglang-trace-cancel";
        let (context, trace_state) = traced_context(observed_context(request_id, true), true);
        let request_context = context.context();
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = dispatch(
            Arc::new(FirstChunkThenPendingEngine),
            context,
            request_id.to_string(),
            state.clone(),
            ConnectionHandle::create_disabled(tx),
            true,
            trace_state,
        )
        .await;
        let mut body = response.into_body().into_data_stream();
        assert!(body.next().await.unwrap().is_ok());
        request_context.kill();
        while body.next().await.is_some() {}

        let records = request_end_records(&mut receiver, request_id).await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].request.as_ref().unwrap().output_tokens, Some(1));
        assert_eq!(
            state.metrics_clone().get_request_counter(
                state.manager().metric_model_for("test-model"),
                &Endpoint::Generate,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );
    }

    #[tokio::test]
    async fn unary_dispatch_joins_chunks_and_maps_stream_errors() {
        let chunk = |ids: serde_json::Value| {
            crate::protocols::Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": ids}})),
                ..Default::default()
            })
        };
        let invalid = dynamo_runtime::error::DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::InvalidArgument)
            .message("bad sampling params")
            .build();
        let failed = crate::protocols::Annotated::from_err(invalid);
        for (outputs, status, body) in [
            (
                vec![chunk(serde_json::json!([5])), chunk(serde_json::json!([6]))],
                StatusCode::OK,
                serde_json::json!({"output_ids": [5, 6]}),
            ),
            (
                vec![chunk(serde_json::json!([5])), failed],
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": {"message": "bad sampling params"}}),
            ),
        ] {
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let response = dispatch(
                Arc::new(ReplayEngine(outputs)),
                dispatch_test_context(),
                "req-unary".to_string(),
                state,
                ConnectionHandle::create_disabled(tx),
                false,
                None,
            )
            .await;
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                body
            );
        }
    }
}
