// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::Request,
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use base64::Engine as _;
use bytes::Bytes;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, Context},
    protocols::annotated::AnnotationsProvider,
};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use super::{
    RouteDoc,
    disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects},
    error::HttpError,
    metrics::{
        CancellationLabels, Endpoint, ErrorType, EventConverter,
        process_response_and_observe_metrics,
        process_response_using_event_converter_and_observe_metrics,
    },
    service_v2,
};
use crate::engines::ValidateRequest;
use crate::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
use crate::protocols::openai::nvext::apply_header_routing_overrides;
use crate::protocols::openai::{
    audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest},
    chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
        NvCreateChatCompletionStreamResponse,
    },
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    images::{NvCreateImageRequest, NvImagesResponse},
    responses::{NvCreateResponse, NvResponse, ResponseParams, chat_completion_to_response},
    videos::{NvCreateVideoRequest, NvVideosResponse},
};
use crate::protocols::unified::UnifiedRequest;
use crate::request_template::RequestTemplate;
use crate::types::Annotated;
use dynamo_runtime::logging::get_distributed_tracing_context;
use tracing::Instrument;

pub const DYNAMO_REQUEST_ID_HEADER: &str = "x-dynamo-request-id";

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

const VALIDATION_PREFIX: &str = "Validation: ";

// Default axum max body limit without configuring is 2MB: https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html
/// Default body limit in bytes (45MB) to support 500k+ token payloads.
/// Can be configured at runtime using the DYN_HTTP_BODY_LIMIT_MB environment variable.
pub(super) fn get_body_limit() -> usize {
    std::env::var(env_llm::DYN_HTTP_BODY_LIMIT_MB)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(45 * 1024 * 1024)
}

pub type ErrorResponse = (StatusCode, Json<ErrorMessage>);

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct ErrorMessage {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

fn map_error_code_to_error_type(code: StatusCode) -> String {
    match code.canonical_reason() {
        Some(reason) => reason.to_string(),
        None => "UnknownError".to_string(),
    }
}

/// Classify error for metrics based on status code and message
fn classify_error_for_metrics(code: StatusCode, message: &str) -> ErrorType {
    match code {
        StatusCode::BAD_REQUEST => {
            // 400
            if message.starts_with("Validation:") {
                ErrorType::Validation
            } else {
                ErrorType::Internal
            }
        }
        StatusCode::NOT_FOUND => ErrorType::NotFound, // 404
        StatusCode::NOT_IMPLEMENTED => ErrorType::NotImplemented, // 501
        StatusCode::TOO_MANY_REQUESTS => ErrorType::Overload, // 429
        StatusCode::SERVICE_UNAVAILABLE => ErrorType::Overload, // 503
        StatusCode::INTERNAL_SERVER_ERROR => ErrorType::Internal, // 500
        _ if code.is_client_error() => ErrorType::Validation, // other 4xx
        _ => ErrorType::Internal,                     // everything else
    }
}

/// Extract ErrorType from ErrorResponse for metrics
fn extract_error_type_from_response(response: &ErrorResponse) -> ErrorType {
    classify_error_for_metrics(response.0, &response.1.message)
}

impl ErrorMessage {
    /// Not Found Error
    pub fn model_not_found() -> ErrorResponse {
        let code = StatusCode::NOT_FOUND;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: "Model not found".to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Service Unavailable
    /// This is returned when the service is live, but not ready.
    pub fn _service_unavailable() -> ErrorResponse {
        let code = StatusCode::SERVICE_UNAVAILABLE;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: "Service is not ready".to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Internal Service Error
    /// Return this error when the service encounters an internal error.
    /// We should return a generic message to the client instead of the real error.
    /// Internal Services errors are the result of misconfiguration or bugs in the service.
    pub fn internal_server_error(msg: &str) -> ErrorResponse {
        tracing::error!("Internal server error: {msg}");
        let code = StatusCode::INTERNAL_SERVER_ERROR;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: msg.to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// Not Implemented Error
    /// Return this error when the client requests a feature that is not yet implemented.
    /// This should be used for features that are planned but not available.
    pub fn not_implemented_error<T: Display>(msg: T) -> ErrorResponse {
        tracing::error!("Not Implemented error: {msg}");
        let code = StatusCode::NOT_IMPLEMENTED;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: msg.to_string(),
                error_type,
                code: code.as_u16(),
            }),
        )
    }

    /// The OAI endpoints call an [`dynamo.runtime::engine::AsyncEngine`] which are specialized to return
    /// an [`anyhow::Error`]. This method will convert the [`anyhow::Error`] into an [`HttpError`].
    /// If successful, it will return the [`HttpError`] as an [`ErrorMessage::internal_server_error`]
    /// with the details of the error.
    pub fn from_anyhow(err: anyhow::Error, alt_msg: &str) -> ErrorResponse {
        // First check for PipelineError::ServiceOverloaded
        if let Some(pipeline_err) =
            err.downcast_ref::<dynamo_runtime::pipeline::error::PipelineError>()
            && matches!(
                pipeline_err,
                dynamo_runtime::pipeline::error::PipelineError::ServiceOverloaded(_)
            )
        {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorMessage {
                    message: pipeline_err.to_string(),
                    error_type: map_error_code_to_error_type(StatusCode::SERVICE_UNAVAILABLE),
                    code: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
                }),
            );
        }

        // Check for DynamoError with InvalidArgument → HTTP 400
        if let Some(dynamo_err) = err.downcast_ref::<dynamo_runtime::error::DynamoError>()
            && dynamo_err.error_type() == dynamo_runtime::error::ErrorType::InvalidArgument
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorMessage {
                    message: dynamo_err.message().to_string(),
                    error_type: map_error_code_to_error_type(StatusCode::BAD_REQUEST),
                    code: StatusCode::BAD_REQUEST.as_u16(),
                }),
            );
        }

        // Then check for HttpError
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorMessage::from_http_error(http_error),
            Err(err) => ErrorMessage::internal_server_error(&format!("{alt_msg}: {err:#}")),
        }
    }

    /// Implementers should only be able to throw 400-499 errors.
    pub fn from_http_error(err: HttpError) -> ErrorResponse {
        if err.code < 400 || err.code >= 500 {
            return ErrorMessage::internal_server_error(&err.message);
        }
        match StatusCode::from_u16(err.code) {
            Ok(code) => (
                code,
                Json(ErrorMessage {
                    message: err.message,
                    error_type: map_error_code_to_error_type(code),
                    code: code.as_u16(),
                }),
            ),
            Err(_) => ErrorMessage::internal_server_error(&err.message),
        }
    }
}

impl From<HttpError> for ErrorMessage {
    fn from(err: HttpError) -> Self {
        ErrorMessage {
            message: err.message,
            error_type: map_error_code_to_error_type(
                StatusCode::from_u16(err.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            ),
            code: err.code,
        }
    }
}

// Problem: Currently we are using JSON from axum as the request validator. Whenever there is an invalid JSON, it will return a 422.
// But all the downstream apps that relies on openai based APIs, expects to get 400 for all these cases otherwise they fail badly
// Solution: Intercept the response from handlers and convert ANY 422 status codes to 400 with the actual error message.
pub async fn smart_json_error_middleware(request: Request<Body>, next: Next) -> Response {
    let response = next.run(request).await;

    if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
        let (_parts, body) = response.into_parts();
        let body_bytes = axum::body::to_bytes(body, get_body_limit())
            .await
            .unwrap_or_default();
        let error_message = String::from_utf8_lossy(&body_bytes).to_string();
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorMessage {
                message: error_message,
                error_type: map_error_code_to_error_type(StatusCode::BAD_REQUEST),
                code: StatusCode::BAD_REQUEST.as_u16(),
            }),
        )
            .into_response()
    } else {
        // Pass through if it is not a 422
        response
    }
}

/// Return the request ID for the current request.
///
/// The canonical request ID is set by `make_inference_request_span()` and stored
/// in the `DistributedTraceContext` via `DistributedTraceIdLayer`. This function
/// retrieves it, falling back to a validated `x-dynamo-request-id` header value
/// (deprecated, DEP #7812) or a new UUID.
///
/// **Deprecation (DEP #7812):** The `x-dynamo-request-id` header is deprecated.
/// Clients should rely on server-generated request IDs instead of supplying their own.
pub(super) fn get_or_create_request_id(headers: &HeaderMap) -> String {
    // Validate x-dynamo-request-id header if present, warn on invalid values.
    // DEP #7812: x-dynamo-request-id is deprecated — clients should rely on
    // server-generated request IDs instead of supplying their own.
    let validated_header = if let Some(raw) = headers.get(DYNAMO_REQUEST_ID_HEADER) {
        tracing::warn!(
            "{} header is deprecated (DEP #7812); server-generated request IDs should be used instead",
            DYNAMO_REQUEST_ID_HEADER
        );
        match raw.to_str() {
            Err(_) => {
                tracing::warn!(
                    "{} header must be a valid UTF-8 string",
                    DYNAMO_REQUEST_ID_HEADER
                );
                None
            }
            Ok(s) if uuid::Uuid::parse_str(s).is_err() => {
                tracing::warn!(
                    "{} header must be a valid UUID, got: {}",
                    DYNAMO_REQUEST_ID_HEADER,
                    s
                );
                None
            }
            Ok(s) => Some(s.to_string()),
        }
    } else {
        None
    };

    // Prefer trace context (set by make_inference_request_span via DistributedTraceIdLayer)
    if let Some(trace_context) = get_distributed_tracing_context()
        && let Some(request_id) = trace_context.request_id
    {
        return request_id;
    }

    // Fallback: use validated header for backwards compat, or generate new UUID
    validated_header.unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

/// OpenAI Completions Request Handler
///
/// This method will handle the incoming request for the `/v1/completions endpoint`. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
async fn handler_completions(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(mut request): Json<NvCreateCompletionRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone(),
        endpoint: Endpoint::Completions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    // possibly long running task
    // if this returns a streaming response, the stream handle will be armed and captured by the response stream
    let response = tokio::spawn(completions(state, request, stream_handle).in_current_span())
        .await
        .map_err(|e| {
            ErrorMessage::internal_server_error(&format!(
                "Failed to await chat completions task: {:?}",
                e,
            ))
        })?;

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    response
}

#[tracing::instrument(skip_all)]
async fn completions(
    state: Arc<service_v2::State>,
    request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    use crate::protocols::openai::completions::get_prompt_batch_size;

    // return a 503 if the service is not ready
    check_ready(&state)?;

    // Validate stream_options is only used when streaming (NVBug 5662680)
    validate_completion_stream_options(&request)?;

    validate_completion_fields_generic(&request)?;

    // Detect batch prompts
    let batch_size = get_prompt_batch_size(&request.inner.prompt);
    let n = request.inner.n.unwrap_or(1);

    // If single prompt or single-element batch, use original flow
    if batch_size == 1 {
        return completions_single(state, request, stream_handle).await;
    }

    // Batch processing: handle multiple prompts
    completions_batch(state, request, stream_handle, batch_size, n).await
}

/// Handle single prompt completions (original logic)
#[tracing::instrument(skip_all)]
async fn completions_single(
    state: Arc<service_v2::State>,
    request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    let request_id = request.id().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = request.inner.model.clone();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Completions,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // todo - error handling should be more robust
    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateCompletionResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    if streaming {
        // For streaming, we'll drop the http_queue_guard on the first token
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream
            .map(move |response| {
                // Calls observe_response() on each token
                process_response_using_event_converter_and_observe_metrics(
                    EventConverter::from(response),
                    &mut response_collector,
                    &mut http_queue_guard,
                )
            })
            .filter_map(|result| {
                use futures::future;
                // Transpose Result<Option<T>> -> Option<Result<T>>
                future::ready(result.transpose())
            });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Tap the stream to collect metrics for non-streaming requests without altering items
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response = NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id,
                    e
                );
                let err_response = ErrorMessage::internal_server_error(&format!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id, e
                ));
                inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                err_response
            })?;

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response).into_response())
    }
}

/// Handle batch prompt completions (multiple prompts with n choices each)
#[tracing::instrument(skip_all)]
async fn completions_batch(
    state: Arc<service_v2::State>,
    request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
    batch_size: usize,
    n: u8,
) -> Result<Response, ErrorResponse> {
    use crate::protocols::openai::completions::extract_single_prompt;
    use futures::stream::{self, StreamExt};

    let request_id = request.id().to_string();
    let streaming = request.inner.stream.unwrap_or(false);
    let model = request.inner.model.clone();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Completions,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // Generate streams for each prompt in the batch
    let mut all_streams = Vec::new();
    let mut first_ctx = None;

    for prompt_idx in 0..batch_size {
        // Extract single prompt at this index
        let single_prompt = extract_single_prompt(&request.inner.prompt, prompt_idx);

        // Create a new request with this single prompt
        let mut single_request = request.content().clone();
        single_request.inner.prompt = single_prompt;

        // Generate unique request_id for each prompt: original_id-{prompt_idx}
        let unique_request_id = format!("{}-{}", request.id(), prompt_idx);
        let single_request_context = Context::with_id(single_request, unique_request_id);

        // Generate stream for this prompt
        let stream = engine.generate(single_request_context).await.map_err(|e| {
            let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

        // Capture context from first stream
        if first_ctx.is_none() {
            first_ctx = Some(stream.context());
        }

        // Remap choice indices: choice.index += prompt_idx * n
        let prompt_idx_u32 = prompt_idx as u32;
        let n_u32 = n as u32;
        let remapped_stream = stream.map(move |mut response| {
            if let Some(ref mut data) = response.data {
                for choice in &mut data.inner.choices {
                    choice.index += prompt_idx_u32 * n_u32;
                }
            }
            response
        });

        all_streams.push(remapped_stream);
    }

    // Merge all streams
    let merged_stream = stream::select_all(all_streams);

    // capture the context to cancel the stream if the client disconnects
    let ctx = first_ctx.expect("At least one stream should be generated");

    let annotations_vec = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateCompletionResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let merged_stream = stream::iter(annotations_vec).chain(merged_stream);

    if streaming {
        // For streaming, we'll drop the http_queue_guard on the first token
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = merged_stream
            .map(move |response| {
                // Calls observe_response() on each token
                process_response_using_event_converter_and_observe_metrics(
                    EventConverter::from(response),
                    &mut response_collector,
                    &mut http_queue_guard,
                )
            })
            .filter_map(|result| {
                use futures::future;
                // Transpose Result<Option<T>> -> Option<Result<T>>
                future::ready(result.transpose())
            });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Tap the stream to collect metrics for non-streaming requests without altering items
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = merged_stream.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response = NvCreateCompletionResponse::from_annotated_stream(stream, parsing_options)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id,
                    e
                );
                let err_response = ErrorMessage::internal_server_error(&format!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id, e
                ));
                inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                err_response
            })?;

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response).into_response())
    }
}

#[tracing::instrument(skip_all)]
async fn embeddings(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateEmbeddingRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = Context::with_id(request, request_id);
    let request_id = request.id().to_string();

    // Embeddings are typically not streamed, so we default to non-streaming
    let streaming = false;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight = state.metrics_clone().create_inflight_guard(
        model,
        Endpoint::Embeddings,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(model);

    // todo - error handling should be more robust
    let engine = state.manager().get_embeddings_engine(model).map_err(|_| {
        let err_response = ErrorMessage::model_not_found();
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate embeddings");
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // Process stream to collect metrics and drop http_queue_guard on first token
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each token - drops http_queue_guard on first token
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Embeddings are typically returned as a single response (non-streaming)
    // so we fold the stream into a single response
    let response = NvCreateEmbeddingResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!(
                "Failed to fold embeddings stream for {}: {:?}",
                request_id,
                e
            );
            let err_response =
                ErrorMessage::internal_server_error("Failed to fold embeddings stream");
            inflight.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

async fn handler_chat_completions(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    Json(mut request): Json<NvCreateChatCompletionRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone(),
        endpoint: Endpoint::ChatCompletions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let response =
        tokio::spawn(chat_completions(state, template, request, stream_handle).in_current_span())
            .await
            .map_err(|e| {
                ErrorMessage::internal_server_error(&format!(
                    "Failed to await chat completions task: {:?}",
                    e,
                ))
            })?;

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    response
}

/// Checks if an Annotated event represents a backend error and extracts error information.
/// Returns Some((message, status_code)) if it's an error, None otherwise.
fn extract_backend_error_if_present<T: serde::Serialize>(
    event: &Annotated<T>,
) -> Option<(String, StatusCode)> {
    #[derive(serde::Deserialize)]
    struct ErrorPayload {
        message: Option<String>,
        code: Option<u16>,
    }

    // Check if event type is "error" (from postprocessor when FinishReason::Error is encountered)
    if let Some(event_type) = &event.event
        && event_type == "error"
    {
        // Extract error string: prefer DynamoError field, fallback to legacy comment.
        // Use message() instead of to_string() for DynamoError to avoid prefixing
        // the ErrorType (e.g., "Unknown: {...}"), which would break JSON parsing.
        let error_str = if let Some(ref dynamo_err) = event.error {
            let mut parts = Vec::new();
            let mut current: Option<&dyn std::error::Error> = Some(dynamo_err);
            while let Some(e) = current {
                if let Some(de) = e.downcast_ref::<dynamo_runtime::error::DynamoError>() {
                    parts.push(de.message().to_string());
                } else {
                    parts.push(e.to_string());
                }
                current = e.source();
            }
            parts.join(", ")
        } else {
            event
                .comment
                .as_ref()
                .map(|c| c.join(", "))
                .unwrap_or_else(|| "Unknown error".to_string())
        };

        // Try to parse as error JSON to extract status code
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&error_str) {
            let code = error_payload
                .code
                .and_then(|c| StatusCode::from_u16(c).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let message = error_payload.message.unwrap_or(error_str);
            return Some((message, code));
        }

        return Some((error_str, StatusCode::INTERNAL_SERVER_ERROR));
    }

    // Check if the data payload itself contains an error structure with code >= 400
    if let Some(data) = &event.data
        && let Ok(json_value) = serde_json::to_value(data)
        && let Ok(error_payload) = serde_json::from_value::<ErrorPayload>(json_value.clone())
        && let Some(code_num) = error_payload.code
        && code_num >= 400
    {
        let code = StatusCode::from_u16(code_num).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let message = error_payload
            .message
            .unwrap_or_else(|| json_value.to_string());
        return Some((message, code));
    }

    // Check if comment contains error information (without event: error)
    if let Some(comments) = &event.comment
        && !comments.is_empty()
    {
        let comment_str = comments.join(", ");

        // Try to parse comment as error JSON with code >= 400
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&comment_str)
            && let Some(code_num) = error_payload.code
            && code_num >= 400
        {
            let code = StatusCode::from_u16(code_num).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let message = error_payload.message.unwrap_or(comment_str);
            return Some((message, code));
        }

        // Comments present with no data AND no event type indicates error
        // (events with event types like "request_id" or "event.dynamo.test.sentinel" are annotations)
        if event.data.is_none() && event.event.is_none() {
            return Some((comment_str, StatusCode::INTERNAL_SERVER_ERROR));
        }
    }

    None
}

/// Checks if the first event in the stream is a backend error.
/// Returns Err(ErrorResponse) if error detected, Ok(stream) otherwise.
pub(super) async fn check_for_backend_error(
    mut stream: impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>
    + Send
    + Unpin
    + 'static,
) -> Result<
    impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send,
    ErrorResponse,
> {
    use futures::stream::StreamExt;

    // Peek at the first event
    if let Some(first_event) = stream.next().await {
        // Check if it's an error event
        if let Some((error_msg, status_code)) = extract_backend_error_if_present(&first_event) {
            return Err((
                status_code,
                Json(ErrorMessage {
                    message: error_msg,
                    error_type: map_error_code_to_error_type(status_code),
                    code: status_code.as_u16(),
                }),
            ));
        }

        // Not an error - reconstruct stream with first event
        let reconstructed_stream = futures::stream::iter(vec![first_event]).chain(stream);
        Ok(reconstructed_stream)
    } else {
        // Empty stream - this shouldn't happen but handle gracefully
        Ok(futures::stream::iter(vec![]).chain(stream))
    }
}

/// Serialize `payload` and wrap it as an SSE event with the given name.
fn make_dispatch_event(
    event_name: &str,
    payload: &impl serde::Serialize,
) -> Option<Result<Event, axum::Error>> {
    match serde_json::to_string(payload) {
        Ok(json) => Some(Ok(Event::default().event(event_name).data(json))),
        Err(e) => {
            tracing::warn!("streaming_{event_name}: failed to serialize: {e}");
            None
        }
    }
}

/// Emits early `event: tool_call_dispatch` SSE events for any complete tool calls found in a
/// streaming response chunk, when `DYN_ENABLE_STREAMING_TOOL_DISPATCH` is enabled.
///
/// Dynamo backends emit each tool call as a single complete chunk (id + name + arguments
/// all present), so we can dispatch immediately upon seeing the chunk rather than waiting
/// for `finish_reason="tool_calls"` to arrive. Each event payload includes `choice_index`
/// for correct disambiguation when `n > 1`.
fn streaming_tool_dispatch_events(
    response: &crate::types::Annotated<NvCreateChatCompletionStreamResponse>,
    dispatched_ids: &mut HashSet<String>,
) -> Vec<Result<Event, axum::Error>> {
    let Some(data) = &response.data else {
        return vec![];
    };

    let mut events = vec![];
    for choice in &data.inner.choices {
        let Some(tool_calls) = &choice.delta.tool_calls else {
            continue;
        };
        for chunk in tool_calls {
            // Only dispatch when the tool call is fully formed (id + name + arguments)
            let has_name_and_args = chunk
                .function
                .as_ref()
                .is_some_and(|f| f.name.is_some() && f.arguments.is_some());

            if let (true, Some(id)) = (has_name_and_args, &chunk.id) {
                // Skip already-dispatched tool calls (dedup guard, matches
                // the stopped/done flags in Anthropic/Responses converters).
                if !dispatched_ids.insert(id.clone()) {
                    continue;
                }
                let payload = serde_json::json!({
                    "choice_index": choice.index,
                    "tool_call": chunk,
                });
                events.extend(make_dispatch_event("tool_call_dispatch", &payload));
            }
        }
    }
    events
}

/// Accumulates reasoning tokens and emits a single `event: reasoning_dispatch` SSE event
/// when the complete reasoning block has been decoded (i.e. when `reasoning_content`
/// transitions from `Some(token)` to `None`), matching the UX of `tool_call_dispatch`.
///
/// The buffer is maintained across chunks by the caller (captured in the flat_map closure).
/// Flushing also occurs when `finish_reason` is set, to handle max_tokens during reasoning.
fn accumulate_reasoning_dispatch(
    response: &crate::types::Annotated<NvCreateChatCompletionStreamResponse>,
    buffers: &mut HashMap<u32, String>,
) -> Vec<Result<Event, axum::Error>> {
    let Some(data) = &response.data else {
        return vec![];
    };

    let mut events = vec![];
    for choice in &data.inner.choices {
        let buffer = buffers.entry(choice.index).or_default();
        let has_reasoning = choice
            .delta
            .reasoning_content
            .as_ref()
            .is_some_and(|r| !r.is_empty());

        if has_reasoning {
            buffer.push_str(choice.delta.reasoning_content.as_ref().unwrap());
        }

        // Emit when reasoning transitions to None OR when the stream ends (finish_reason).
        if !buffer.is_empty() && (!has_reasoning || choice.finish_reason.is_some()) {
            let payload = serde_json::json!({
                "index": choice.index,
                "reasoning_content": buffer.as_str(),
            });
            events.extend(make_dispatch_event("reasoning_dispatch", &payload));
            buffer.clear();
        }
    }
    events
}

/// OpenAI Chat Completions Request Handler
///
/// This method will handle the incoming request for the /v1/chat/completions endpoint. The endpoint is a "source"
/// for an [`super::OpenAIChatCompletionsStreamingEngine`] and will return a stream of responses which will be
/// forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
async fn chat_completions(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<NvCreateChatCompletionRequest>,
    mut stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = request.id().to_string();

    // Determine streaming mode early
    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // Apply template values first to resolve the model before creating metrics guards
    if let Some(template) = template {
        if request.inner.model.is_empty() {
            request.inner.model = template.model.clone();
        }
        if request.inner.temperature.unwrap_or(0.0) == 0.0 {
            request.inner.temperature = Some(template.temperature);
        }
        if request.inner.max_completion_tokens.unwrap_or(0) == 0 {
            request.inner.max_completion_tokens = Some(template.max_completion_tokens);
        }
    }

    // Capture the resolved model after template application for metrics and engine lookup
    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    // todo - determine the proper error code for when a request model is not present
    let model = request.inner.model.clone();

    tracing::trace!("Received chat completions request: {:?}", request.content());

    // Create inflight_guard early to ensure all errors (including validation) are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::ChatCompletions,
        streaming,
        &request_id,
    );

    // Handle unsupported fields - if Some(resp) is returned by
    // validate_chat_completion_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code. Otherwise, proceeed.
    if let Err(err_response) = validate_chat_completion_unsupported_fields(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Handle required fields like messages shouldn't be empty.
    if let Err(err_response) = validate_chat_completion_required_fields(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Validate stream_options is only used when streaming (NVBug 5662680)
    if let Err(err_response) = validate_chat_completion_stream_options(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Handle Rest of Validation Errors
    if let Err(err_response) = validate_chat_completion_fields_generic(&request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Create HTTP queue guard after template resolution so labels are correct
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // prepare any requested annotations
    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::from_annotation(ANNOTATION_REQUEST_ID, &request_id).ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic

    if streaming {
        // For streaming responses, we return HTTP 200 immediately without checking for errors.
        // Once HTTP 200 OK is sent, we cannot change the status code, so any backend errors
        // must be delivered as SSE events with `event: error` in the stream (handled by
        // EventConverter and monitor_for_disconnects). This is standard SSE behavior.
        stream_handle.arm(); // allows the system to detect client disconnects and cancel the LLM generation

        let mut http_queue_guard = Some(http_queue_guard);
        let tool_dispatch_enabled = state.streaming_tool_dispatch_enabled();
        let reasoning_dispatch_enabled = state.streaming_reasoning_dispatch_enabled();
        let mut reasoning_buffer: HashMap<u32, String> = HashMap::new();
        let mut dispatched_tool_ids: HashSet<String> = HashSet::new();

        // flat_map lets us optionally prepend extra SSE events before each regular chunk:
        //   - `event: tool_call_dispatch`  — complete tool call detected early (tool dispatch)
        //   - `event: reasoning_dispatch`  — complete reasoning block (emitted once)
        // When both flags are off the flat_map is equivalent to the original map + filter_map.
        let stream = stream.flat_map(move |response| {
            // Extract side-channel events before the response is consumed by EventConverter.
            let mut events: Vec<Result<Event, axum::Error>> = vec![];
            if tool_dispatch_enabled {
                events.extend(streaming_tool_dispatch_events(
                    &response,
                    &mut dispatched_tool_ids,
                ));
            }
            if reasoning_dispatch_enabled {
                events.extend(accumulate_reasoning_dispatch(
                    &response,
                    &mut reasoning_buffer,
                ));
            }

            // Convert to SSE event (this consumes the response).
            // EventConverter will detect `event: "error"` and convert to SSE error events.
            let sse_result = process_response_using_event_converter_and_observe_metrics(
                EventConverter::from(response),
                &mut response_collector,
                &mut http_queue_guard,
            );

            // Side-channel events come first, then the regular data event.
            match sse_result {
                Ok(Some(ev)) => events.push(Ok(ev)),
                Ok(None) => {}
                Err(e) => events.push(Err(e)),
            }
            stream::iter(events)
        });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Check first event for backend errors before aggregating (non-streaming only)
        let stream_with_check =
            check_for_backend_error(stream)
                .await
                .map_err(|error_response| {
                    tracing::error!(request_id, "Backend error detected: {:?}", error_response);
                    inflight_guard.mark_error(extract_error_type_from_response(&error_response));
                    error_response
                })?;

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream_with_check.inspect(move |response| {
            // Calls observe_response() on each token - drops http_queue_guard on first token
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response =
            NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
                .await
                .map_err(|e| {
                    tracing::error!(
                        request_id,
                        "Failed to parse chat completion response: {:?}",
                        e
                    );
                    let err_response = ErrorMessage::internal_server_error(&format!(
                        "Failed to parse chat completion response: {}",
                        e
                    ));
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }
        Ok(Json(response).into_response())
    }
}

/// Checks for unsupported fields in the request.
/// Returns Some(response) if unsupported fields are present.
#[allow(deprecated)]
pub fn validate_chat_completion_unsupported_fields(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;

    if inner.function_call.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string()
                + "`function_call` is deprecated. Please migrate to use `tool_choice` instead.",
        ));
    }

    if inner.functions.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string()
                + "`functions` is deprecated. Please migrate to use `tools` instead.",
        ));
    }

    Ok(())
}

/// Validates that required fields are present and valid in the chat completion request
pub fn validate_chat_completion_required_fields(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;

    if inner.messages.is_empty() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'messages' field cannot be empty. At least one message is required.",
        }));
    }

    Ok(())
}

/// Validates that stream_options is only used when stream=true for chat completions (NVBug 5662680)
pub fn validate_chat_completion_stream_options(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;
    let streaming = inner.stream.unwrap_or(false);
    if !streaming && inner.stream_options.is_some() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'stream_options' field is only allowed when 'stream' is set to true.",
        }));
    }
    Ok(())
}

/// Validates a chat completion request and returns an error response if validation fails.
///
/// This function calls the `validate` method implemented for `NvCreateChatCompletionRequest`.
/// If validation fails, it maps the error into an OpenAI-compatible error response.
pub fn validate_chat_completion_fields_generic(
    request: &NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    request.validate().map_err(|e| {
        ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &e.to_string(),
        })
    })
}

/// Validates that stream_options is only used when stream=true for completions (NVBug 5662680)
pub fn validate_completion_stream_options(
    request: &NvCreateCompletionRequest,
) -> Result<(), ErrorResponse> {
    let inner = &request.inner;
    let streaming = inner.stream.unwrap_or(false);
    if !streaming && inner.stream_options.is_some() {
        return Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string()
                + "The 'stream_options' field is only allowed when 'stream' is set to true.",
        }));
    }
    Ok(())
}

/// Validates a completion request and returns an error response if validation fails.
///
/// This function calls the `validate` method implemented for `NvCreateCompletionRequest`.
/// If validation fails, it maps the error into an OpenAI-compatible error response.
pub fn validate_completion_fields_generic(
    request: &NvCreateCompletionRequest,
) -> Result<(), ErrorResponse> {
    request.validate().map_err(|e| {
        ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &e.to_string(),
        })
    })
}

/// OpenAI Responses Request Handler
///
/// This method will handle the incoming request for the /v1/responses endpoint.
async fn handler_responses(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    Json(mut request): Json<NvCreateResponse>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    request.nvext = apply_header_routing_overrides(request.nvext.take(), &headers);

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let cancellation_labels = CancellationLabels {
        model: request.inner.model.clone().unwrap_or_default(),
        endpoint: Endpoint::Responses.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let response =
        tokio::spawn(responses(state, template, request, stream_handle).in_current_span())
            .await
            .map_err(|e| {
                ErrorMessage::internal_server_error(&format!(
                    "Failed to await responses task: {:?}",
                    e,
                ))
            })?;

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    response
}

#[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.id()))]
async fn responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<NvCreateResponse>,
    mut stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // Apply template values if present, with sensible defaults for the Responses API.
    // Unlike chat completions where backends may have their own defaults, the Responses API
    // should provide a generous default to avoid truncated responses (especially with
    // reasoning models that emit <think> tokens).
    const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

    if let Some(template) = template {
        if request.inner.model.as_deref().unwrap_or("").is_empty() {
            request.inner.model = Some(template.model.clone());
        }
        if request.inner.temperature.is_none() {
            request.inner.temperature = Some(template.temperature);
        }
        if request.inner.max_output_tokens.is_none() {
            request.inner.max_output_tokens = Some(template.max_completion_tokens);
        }
    } else if request.inner.max_output_tokens.is_none() {
        request.inner.max_output_tokens = Some(DEFAULT_MAX_OUTPUT_TOKENS);
    }
    tracing::trace!("Received responses request: {:?}", request.inner);

    let model = request.inner.model.clone().unwrap_or_default();
    let streaming = request.inner.stream.unwrap_or(false);

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Responses,
        streaming,
        request.id(),
    );

    // Handle unsupported fields - if Some(resp) is returned by validate_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code.
    if let Some(resp) = validate_response_unsupported_fields(&request) {
        inflight_guard.mark_error(ErrorType::NotImplemented);
        return Ok(resp.into_response());
    }

    // Extract request parameters before into_parts() consumes the request.
    // These are echoed back in the Response object per the OpenAI spec.
    let response_params = ResponseParams {
        model: request.inner.model.clone(),
        temperature: request.inner.temperature,
        top_p: request.inner.top_p,
        max_output_tokens: request.inner.max_output_tokens,
        store: request.inner.store,
        tools: request.inner.tools.clone(),
        tool_choice: request.inner.tool_choice.clone(),
        instructions: request.inner.instructions.clone(),
        reasoning: request.inner.reasoning.clone(),
        text: request.inner.text.clone(),
        service_tier: request.inner.service_tier,
        include: request.inner.include.clone(),
        truncation: request.inner.truncation,
    };
    let request_id = request.id().to_string();
    let (orig_request, context) = request.into_parts();

    let unified_request: UnifiedRequest = orig_request.try_into().map_err(|e: anyhow::Error| {
        tracing::error!(
            request_id,
            error = %e,
            "Failed to convert NvCreateResponse to UnifiedRequest",
        );
        let err_response = ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string()
                + "Failed to convert responses request: "
                + &e.to_string(),
        );
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;
    // Extract the API context before consuming the UnifiedRequest — this
    // carries Responses-specific fields (previous_response_id, store, etc.)
    // that the stream converter needs for faithful response reconstruction.
    let responses_ctx = unified_request.responses_context().cloned();
    let mut chat_request = unified_request.into_inner();

    // Always use internal streaming for aggregation.
    // Set stream_options.include_usage so the backend sends token counts in the final chunk.
    chat_request.inner.stream = Some(true);
    chat_request.inner.stream_options =
        Some(dynamo_protocols::types::ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        });

    let request = context.map(|mut _req| chat_request);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|_| {
            let err_response = ErrorMessage::model_not_found();
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    tracing::trace!("Issuing generate call for responses");

    // issue the generate call on the engine
    let engine_stream = engine.generate(request).await.map_err(|e| {
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate completions");
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    // Capture the context to cancel the stream if the client disconnects
    let ctx = engine_stream.context();

    if streaming {
        // For streaming responses, we return HTTP 200 immediately without checking for errors.
        // Once HTTP 200 OK is sent, we cannot change the status code, so any backend errors
        // must be delivered as SSE events in the stream. This is standard SSE behavior.
        stream_handle.arm(); // allows the system to detect client disconnects and cancel the LLM generation

        // Streaming path: convert chat completion stream chunks to Responses API SSE events.
        // The engine yields Annotated<NvCreateChatCompletionStreamResponse>. We extract the
        // inner stream response data and convert it to Responses API events.
        use crate::protocols::openai::responses::stream_converter::ResponseStreamConverter;
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut converter = match responses_ctx {
            Some(ctx) => ResponseStreamConverter::with_context(model.clone(), response_params, ctx),
            None => ResponseStreamConverter::new(model.clone(), response_params),
        };
        let start_events = converter.emit_start_events();

        // Use std::sync::Mutex (not tokio) since process_chunk/emit_end_events are
        // synchronous -- no .await while lock is held. Avoids async lock overhead per token.
        let converter = std::sync::Arc::new(std::sync::Mutex::new(converter));
        let converter_end = converter.clone();

        // Track whether the backend sent an error event during the stream.
        // Shared between event_stream (writer) and done_stream (reader).
        let saw_error = std::sync::Arc::new(AtomicBool::new(false));
        let saw_error_end = saw_error.clone();

        let mut http_queue_guard = Some(http_queue_guard);

        // Process each annotated chunk: extract the stream response data, convert to events
        let event_stream = engine_stream
            .inspect(move |response| {
                process_response_and_observe_metrics(
                    response,
                    &mut response_collector,
                    &mut http_queue_guard,
                );
            })
            .filter_map(move |annotated_chunk| {
                let converter = converter.clone();
                let saw_error = saw_error.clone();
                async move {
                    // Check for backend error before extracting data.
                    // Error events have data: None and event: Some("error").
                    if annotated_chunk.data.is_none() {
                        if annotated_chunk.event.as_deref() == Some("error") {
                            saw_error.store(true, Ordering::Release);
                        }
                        return None;
                    }
                    let stream_resp = annotated_chunk.data?;
                    let mut conv = converter.lock().expect("converter lock poisoned");
                    let events = conv.process_chunk(&stream_resp);
                    Some(stream::iter(events))
                }
            })
            .flatten();

        // Chain: start_events -> chunk_events -> end_events
        let start_stream = stream::iter(start_events);

        let done_stream = stream::once(async move {
            let mut conv = converter_end.lock().expect("converter lock poisoned");
            let end_events = if saw_error_end.load(Ordering::Acquire) {
                conv.emit_error_events()
            } else {
                conv.emit_end_events()
            };
            stream::iter(end_events)
        })
        .flatten();

        let full_stream = start_stream.chain(event_stream).chain(done_stream);

        let full_stream = full_stream.map(|result| result.map_err(axum::Error::new));

        // Wrap with disconnect monitoring: detects client disconnects, cancels generation,
        // and defers inflight_guard.mark_ok() until the stream completes.
        let stream = monitor_for_disconnects(full_stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Non-streaming path: aggregate stream into single response

        // Check first event for backend errors before aggregating (non-streaming only)
        let stream_with_check =
            check_for_backend_error(engine_stream)
                .await
                .map_err(|error_response| {
                    tracing::error!(request_id, "Backend error detected: {:?}", error_response);
                    inflight_guard.mark_error(extract_error_type_from_response(&error_response));
                    error_response
                })?;

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream_with_check.inspect(move |response| {
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response =
            NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
                .await
                .map_err(|e| {
                    tracing::error!(request_id, "Failed to fold responses stream: {:?}", e);
                    let err_response = ErrorMessage::internal_server_error(&format!(
                        "Failed to fold responses stream: {}",
                        e
                    ));
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;

        // Convert NvCreateChatCompletionResponse --> NvResponse
        let response: NvResponse =
            chat_completion_to_response(response, &response_params, responses_ctx.as_ref())
                .map_err(|e| {
                    tracing::error!(
                        request_id,
                        "Failed to convert NvCreateChatCompletionResponse to NvResponse: {:?}",
                        e
                    );
                    let err_response =
                        ErrorMessage::internal_server_error("Failed to convert internal response");
                    inflight_guard.mark_error(extract_error_type_from_response(&err_response));
                    err_response
                })?;

        inflight_guard.mark_ok();
        // If the engine context was killed (client disconnect), the response was
        // assembled but never delivered. Override to cancelled.
        if ctx.is_killed() {
            inflight_guard.mark_error(ErrorType::Cancelled);
        }

        Ok(Json(response).into_response())
    }
}

/// Checks for unsupported fields in the request.
/// Returns Some(response) if unsupported fields are present.
pub fn validate_response_unsupported_fields(
    request: &NvCreateResponse,
) -> Option<impl IntoResponse> {
    let inner = &request.inner;

    if inner.background == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`background: true` is not supported.",
        ));
    }
    if inner.previous_response_id.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`previous_response_id` is not supported.",
        ));
    }
    if inner.prompt.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`prompt` is not supported.",
        ));
    }
    if inner.store == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`store: true` is not supported.",
        ));
    }
    None
}

// todo - abstract this to the top level lib.rs to be reused
// todo - move the service_observer to its own state/arc
fn check_ready(_state: &Arc<service_v2::State>) -> Result<(), ErrorResponse> {
    // if state.service_observer.stage() != ServiceStage::Ready {
    //     return Err(ErrorMessage::service_unavailable());
    // }
    Ok(())
}

/// openai compatible format
/// Example:
/// {
///  "object": "list",
///  "data": [
///    {
///      "id": "model-id-0",
///      "object": "model",
///      "created": 1686935002,
///      "owned_by": "organization-owner"
///    },
///    ]
/// }
async fn list_models_openai(
    State(state): State<Arc<service_v2::State>>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut data = Vec::new();

    let models: HashSet<String> = state.manager().model_display_names();
    for model_name in models {
        data.push(ModelListing {
            id: model_name.clone(),
            object: "model", // Per OpenAI spec, this should be "model"
            created,
            owned_by: "nvidia".to_string(),
        });
    }

    let out = ListModelOpenAI {
        object: "list",
        data,
    };
    Ok(Json(out).into_response())
}

#[derive(Serialize)]
struct ListModelOpenAI {
    object: &'static str, // always "list"
    data: Vec<ModelListing>,
}

#[derive(Serialize)]
struct ModelListing {
    id: String,
    object: &'static str, // always "model" per OpenAI spec
    created: u64,         // Seconds since epoch
    owned_by: String,
}

/// Create an Axum [`Router`] for the OpenAI API Completions endpoint
/// If not path is provided, the default path is `/v1/completions`
pub fn completions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_completions))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Create an Axum [`Router`] for the OpenAI API Chat Completions endpoint
/// If not path is provided, the default path is `/v1/chat/completions`
pub fn chat_completions_router(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/chat/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_chat_completions))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state((state, template));
    (vec![doc], router)
}

/// Create an Axum [`Router`] for the OpenAI API Embeddings endpoint
/// If not path is provided, the default path is `/v1/embeddings`
pub fn embeddings_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/embeddings".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(embeddings))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// List Models
pub fn list_models_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    // Standard OpenAI compatible list models endpoint
    let openai_path = path.unwrap_or("/v1/models".to_string());
    let doc_for_openai = RouteDoc::new(axum::http::Method::GET, &openai_path);

    let router = Router::new()
        .route(&openai_path, get(list_models_openai))
        .with_state(state);

    (vec![doc_for_openai], router)
}

/// Create an Axum [`Router`] for the OpenAI API Responses endpoint
/// If not path is provided, the default path is `/v1/responses`
pub fn responses_router(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/responses".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_responses))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state((state, template));
    (vec![doc], router)
}

async fn images(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateImageRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = Context::with_id(request, request_id);
    let request_id = request.id().to_string();

    // Images are typically not streamed, so we default to non-streaming
    let streaming = false;

    // Get the model name from the request (diffusion model)
    let model = request
        .inner
        .model
        .as_ref()
        .map(|m| match m {
            dynamo_protocols::types::ImageModel::DallE2 => "dall-e-2".to_string(),
            dynamo_protocols::types::ImageModel::DallE3 => "dall-e-3".to_string(),
            dynamo_protocols::types::ImageModel::Other(s) => s.clone(),
        })
        .unwrap_or_else(|| "diffusion".to_string());

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // Get the image generation engine
    let engine = state
        .manager()
        .get_images_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Images,
        streaming,
        &request_id,
    );

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // Issue the generate call on the engine
    // Note: This uses ServerStreamingEngine for internal routing/distribution,
    // NOT for client-facing SSE streaming. The stream is immediately folded into
    // a single response below.
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate images"))?;

    // Process stream to collect metrics and drop http_queue_guard on first response
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each item - drops http_queue_guard on first item
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Images are returned as a single response (non-streaming to client)
    // Fold the internal stream into a single response
    let response = NvImagesResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!("Failed to fold images stream for {}: {:?}", request_id, e);
            ErrorMessage::internal_server_error("Failed to fold images stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// Create an Axum [`Router`] for the OpenAI API Images endpoint
/// If not path is provided, the default path is `/v1/images/generations`
pub fn images_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/images/generations".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(images))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

async fn videos(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = Context::with_id(request, request_id);
    let request_id = request.id().to_string();

    // Videos are typically not streamed, so we default to non-streaming
    let streaming = false;

    // Get the model name from the request (video generation model)
    let model = request.model.clone();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // Get the video generation engine
    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Videos,
        streaming,
        &request_id,
    );

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate videos"))?;

    // Process stream to collect metrics and drop http_queue_guard on first token
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        // Calls observe_response() on each token - drops http_queue_guard on first token
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Videos are typically returned as a single response (non-streaming)
    // so we fold the stream into a single response
    let response = NvVideosResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!("Failed to fold videos stream for {}: {:?}", request_id, e);
            ErrorMessage::internal_server_error("Failed to fold videos stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// [EXPERIMENTAL] MJPEG streaming handler for `/v1/videos/stream`.
///
/// The backend is expected to yield one [`NvVideosResponse`] per frame, carrying a
/// JPEG-encoded frame as `data[0].b64_json`. This handler decodes each frame and
/// writes it as an MJPEG multipart boundary so the client receives a live
/// `multipart/x-mixed-replace` stream viewable directly in a browser `<img>` tag
/// or via `ffplay http://.../v1/videos/stream`.
async fn video_stream(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = Context::with_id(request, request_id);
    let model = request.model.clone();

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Videos, true, request.id());

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to start video stream"))?;

    // Capture the context to cancel the stream if the client disconnects.
    let ctx = stream.context();

    // Create connection monitor. The connection_handle is disarmed immediately because
    // video_stream returns the streaming body directly (graceful handler exit).
    // The stream_handle is armed below and lives inside the monitored stream so that
    // a client disconnect (body drop) signals the engine context to cancel.
    let (mut connection_handle, mut stream_handle) = create_connection_monitor(
        ctx.clone(),
        Some(state.metrics_clone()),
        CancellationLabels {
            model: model.clone(),
            endpoint: Endpoint::Videos.to_string(),
            request_type: "stream".to_string(),
        },
    )
    .await;
    connection_handle.disarm();

    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    // Map each annotated NvVideosResponse to an MJPEG boundary chunk.
    // The backend yields one response per frame with the JPEG in data[0].b64_json.
    let mjpeg_stream = stream.filter_map(|annotated| async move {
        let ann = match annotated.ok() {
            Ok(a) => a,
            Err(e) => {
                tracing::error!("Video stream error: {e}");
                return None;
            }
        };
        let response = ann.data?;
        let frame = response.data.into_iter().next()?;
        let b64 = frame.b64_json?;
        let jpeg_bytes = match base64::prelude::BASE64_STANDARD.decode(&b64) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("Failed to decode frame base64: {e}");
                return None;
            }
        };
        let header = format!(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg_bytes.len()
        );
        let mut chunk = Vec::with_capacity(header.len() + jpeg_bytes.len() + 2);
        chunk.extend_from_slice(header.as_bytes());
        chunk.extend_from_slice(&jpeg_bytes);
        chunk.extend_from_slice(b"\r\n");
        Some(Ok::<Bytes, std::convert::Infallible>(Bytes::from(chunk)))
    });

    // Arm the stream handle and monitor for client disconnects or context cancellation.
    // inflight.mark_ok() is deferred until the stream ends naturally. If the stream is
    // dropped early (client disconnect), the armed stream_handle signals the connection
    // monitor, which cancels the engine context.
    stream_handle.arm();
    let monitored_stream = async_stream::stream! {
        tokio::pin!(mjpeg_stream);
        loop {
            tokio::select! {
                frame = mjpeg_stream.next() => {
                    match frame {
                        Some(item) => yield item,
                        None => {
                            // Stream ended naturally: mark inflight OK and disarm the handle.
                            inflight.mark_ok();
                            stream_handle.disarm();
                            break;
                        }
                    }
                }
                _ = ctx.stopped() => {
                    tracing::trace!("Context stopped; breaking MJPEG stream");
                    break;
                }
            }
        }
    };

    axum::http::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(
            axum::http::header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .body(Body::from_stream(monitored_stream))
        .map(|r| r.into_response())
        .map_err(|e| {
            ErrorMessage::internal_server_error(&format!("Failed to build MJPEG response: {e}"))
        })
}

/// Create an Axum [`Router`] for the OpenAI API Videos endpoint
/// If no path is provided, the default path is `/v1/videos`
///
/// Two routes are registered:
/// - `POST /v1/videos`        — non-streaming, returns a single JSON response
/// - `POST /v1/videos/stream` — MJPEG streaming via `multipart/x-mixed-replace`
pub fn videos_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/videos".to_string());
    let stream_path = format!("{}/stream", path);
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let stream_doc = RouteDoc::new(axum::http::Method::POST, &stream_path);
    let router = Router::new()
        .route(&path, post(videos))
        .route(&stream_path, post(video_stream))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc, stream_doc], router)
}

async fn audio_speech(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateAudioSpeechRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    let response_format = request.response_format.clone();
    let request_id = get_or_create_request_id(&headers);
    let request = Context::with_id(request, request_id);
    let request_id = request.id().to_string();

    let streaming = false;

    // model is optional in the request; fall back to the first registered model
    let model = request.model.clone().unwrap_or_else(|| {
        state
            .manager()
            .model_display_names()
            .into_iter()
            .next()
            .unwrap_or_default()
    });

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    let engine = state
        .manager()
        .get_audios_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Audios,
        streaming,
        &request_id,
    );

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate audio"))?;

    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });

    let response = NvAudioSpeechResponse::from_annotated_stream(stream)
        .await
        .map_err(|e| {
            tracing::error!("Failed to fold audio stream for {}: {:?}", request_id, e);
            ErrorMessage::internal_server_error("Failed to fold audio stream")
        })?;

    // Check for failure before marking success
    if response.status == "failed" {
        return Ok((axum::http::StatusCode::BAD_REQUEST, Json(response)).into_response());
    }

    inflight.mark_ok();

    // If response contains b64_json audio data, decode and return as binary
    // (matching OpenAI/vLLM-Omni behavior: curl --output file.wav)
    if let Some(first) = response.data.first()
        && let Some(b64) = &first.b64_json
        && let Ok(audio_bytes) = base64::engine::general_purpose::STANDARD.decode(b64)
    {
        let content_type = match response_format.as_deref().unwrap_or("wav") {
            "mp3" => "audio/mpeg",
            "flac" => "audio/flac",
            "pcm" => "audio/pcm",
            "aac" => "audio/aac",
            "opus" => "audio/ogg; codecs=opus",
            _ => "audio/wav",
        };
        return Ok(Response::builder()
            .header("content-type", content_type)
            .body(axum::body::Body::from(audio_bytes))
            .unwrap());
    }

    // Fallback: return JSON (url format responses)
    Ok(Json(response).into_response())
}

/// Create an Axum [`Router`] for the Audio Speech endpoint
/// Default path is `/v1/audio/speech`
pub fn audios_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/audio/speech".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(audio_speech))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::discovery::ModelManagerError;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::common_ext::CommonExt;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use dynamo_protocols::types::responses::{CreateResponse, Input, PromptConfig};
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        CreateCompletionRequest,
    };

    const BACKUP_ERROR_MESSAGE: &str = "Failed to generate completions";

    fn http_error_from_engine(code: u16) -> Result<(), anyhow::Error> {
        Err(HttpError {
            code,
            message: "custom error message".to_string(),
        })?
    }

    fn other_error_from_engine() -> Result<(), anyhow::Error> {
        Err(ModelManagerError::ModelNotFound("foo".to_string()))?
    }

    fn make_base_request() -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: Input::Text("hello".into()),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
        }
    }

    #[test]
    fn test_http_error_response_from_anyhow() {
        let err = http_error_from_engine(400).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::BAD_REQUEST);
        assert_eq!(response.1.message, "custom error message");
    }

    #[test]
    fn test_error_response_from_anyhow_out_of_range() {
        let err = http_error_from_engine(399).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");

        let err = http_error_from_engine(500).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");

        let err = http_error_from_engine(501).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, "custom error message");
    }

    #[test]
    fn test_other_error_response_from_anyhow() {
        let err = other_error_from_engine().unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.1.message,
            format!(
                "{}: {}",
                BACKUP_ERROR_MESSAGE,
                other_error_from_engine().unwrap_err()
            )
        );
    }

    #[test]
    fn test_service_overloaded_error_response_from_anyhow() {
        use dynamo_runtime::pipeline::error::PipelineError;

        let err: anyhow::Error = PipelineError::ServiceOverloaded(
            "All workers are busy, please retry later".to_string(),
        )
        .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.1.message,
            "Service temporarily unavailable: All workers are busy, please retry later"
        );
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_clean_request() {
        let request = make_base_request();
        let result = validate_response_unsupported_fields(&request);
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_parallel_tool_calls() {
        let mut request = make_base_request();
        request.inner.parallel_tool_calls = Some(true);
        let result = validate_response_unsupported_fields(&request);
        assert!(result.is_none(), "parallel_tool_calls should be supported");
    }

    #[test]
    fn test_validate_unsupported_fields_detects_flags() {
        #[allow(clippy::type_complexity)]
        let unsupported_cases: Vec<(&str, Box<dyn FnOnce(&mut CreateResponse)>)> = vec![
            ("background", Box::new(|r| r.background = Some(true))),
            (
                "previous_response_id",
                Box::new(|r| r.previous_response_id = Some("prev-id".into())),
            ),
            (
                "prompt",
                Box::new(|r| {
                    r.prompt = Some(PromptConfig {
                        id: "template-id".into(),
                        version: None,
                        variables: None,
                    })
                }),
            ),
            ("store", Box::new(|r| r.store = Some(true))),
        ];

        for (field, set_field) in unsupported_cases {
            let mut req = make_base_request();
            (set_field)(&mut req.inner);
            let result = validate_response_unsupported_fields(&req);
            assert!(result.is_some(), "Expected rejection for `{field}`");
        }
    }

    #[test]
    fn test_validate_chat_completion_required_fields_empty_messages() {
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![],
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!(
                    "{VALIDATION_PREFIX}The 'messages' field cannot be empty. At least one message is required."
                )
            );
        }
    }

    #[test]
    fn test_validate_chat_completion_required_fields_with_messages() {
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_ok());
    }

    #[test]
    // Test for all Bad Requests Example for Chat Completion
    // 1. Echo:  Should be a boolean : Not Done
    // 2. Frequency Penalty: Should be a float between -2.0 and 2.0 : Done
    // 3. logprobs: Done
    // 4. Model Format: Should be a string : Not Done
    // 5. Prompt or Messages Validation
    // 6. Max Tokens: Should be a positive integer
    // 7. Presence Penalty: Should be a float between -2.0 and 2.0 : Done
    // 8. Stop : Should be a string or an array of strings : Not Done
    // 9. Invalid or Out of range temperature: Done
    // 10.Invalid or out of range top_p: Done
    // 11. Repetition Penalty: Should be a float between 0.0 and 2.0 : Done
    // 12. Logprobs: Should be a positive integer between 0 and 5 : Done
    // invalid or non existing user : Only empty string is not allowed validation is there. How can we check non-extisting user ?
    // Unknown fields : Done (rejected via extra_fields catch-all)
    // guided_whitespace_pattern null or invalid : Not Done
    // "response_format": { "type": "invalid_format" } : Not Done
    // "logit_bias": { "invalid_token": "not_a_number" }, : Partial Validation is already there
    fn test_bad_base_request_for_completion() {
        // Frequency Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                frequency_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Frequency penalty must be between -2 and 2, got -3")
            );
        }

        // Presence Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                presence_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Presence penalty must be between -2 and 2, got -3")
            );
        }

        // Temperature: Should be a float between 0.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                temperature: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Temperature must be between 0 and 2, got -3")
            );
        }

        // Top P: Should be a float between 0.0 and 1.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                top_p: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_p must be between 0 and 1, got -3")
            );
        }

        // Repetition Penalty: Should be a float between 0.0 and 2.0
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                ..Default::default()
            },
            common: CommonExt::builder()
                .repetition_penalty(-3.0)
                .build()
                .unwrap(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Repetition penalty must be between 0 and 2, got -3")
            );
        }

        // Logprobs: Should be a positive integer between 0 and 5
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                logprobs: Some(6),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Logprobs must be between 0 and 5, got 6")
            );
        }
    }

    #[test]
    fn test_metadata_field_nested() {
        use serde_json::json;

        // Test metadata field with nested object
        let request = NvCreateCompletionRequest {
            inner: CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: json!({
                "user": {"id": 1, "name": "user-1"},
                "session": {"id": "session-1", "timestamp": 1640995200}
            })
            .into(),
            unsupported_fields: Default::default(),
        };

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_ok());

        // Verify metadata is accessible
        assert!(request.metadata.is_some());
        assert_eq!(request.metadata.as_ref().unwrap()["user"]["id"], 1);
    }

    #[test]
    fn test_bad_base_request_for_chatcompletion() {
        // Frequency Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                frequency_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };

        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Frequency penalty must be between -2 and 2, got -3")
            );
        }

        // Presence Penalty: Should be a float between -2.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                presence_penalty: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Presence penalty must be between -2 and 2, got -3")
            );
        }

        // Temperature: Should be a float between 0.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                temperature: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Temperature must be between 0 and 2, got -3")
            );
        }

        // Top P: Should be a float between 0.0 and 1.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                top_p: Some(-3.0),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_p must be between 0 and 1, got -3")
            );
        }

        // Repetition Penalty: Should be a float between 0.0 and 2.0
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                ..Default::default()
            },
            common: CommonExt::builder()
                .repetition_penalty(-3.0)
                .build()
                .unwrap(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Repetition penalty must be between 0 and 2, got -3")
            );
        }

        // Top Logprobs: Should be a positive integer between 0 and 20
        let request = NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                        name: None,
                    },
                )],
                top_logprobs: Some(25),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!("{VALIDATION_PREFIX}Top_logprobs must be between 0 and 20, got 25")
            );
        }
    }

    #[test]
    fn test_chat_completions_unknown_fields_rejected() {
        // Test that known unsupported fields are rejected and all shown in error message
        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "add_special_tokens": true,
            "documents": ["doc1"],
            "chat_template": "custom"
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json).unwrap();

        // Verify all unsupported fields were captured
        assert!(
            request
                .unsupported_fields
                .contains_key("add_special_tokens")
        );
        assert!(request.unsupported_fields.contains_key("documents"));
        assert!(request.unsupported_fields.contains_key("chat_template"));

        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            let msg = &error_response.1.message;
            assert!(msg.contains("Unsupported parameter"));
            // Verify all fields appear in the error message
            assert!(msg.contains("add_special_tokens"));
            assert!(msg.contains("documents"));
            assert!(msg.contains("chat_template"));
        }
    }

    #[test]
    fn test_completions_unsupported_fields_rejected() {
        // Test that known unsupported fields are rejected and all shown in error message
        let json = r#"{
            "model": "test-model",
            "prompt": "Hello",
            "add_special_tokens": true,
            "response_format": {"type": "json_object"}
        }"#;

        let request: NvCreateCompletionRequest = serde_json::from_str(json).unwrap();

        // Verify both unsupported fields were captured
        assert!(
            request
                .unsupported_fields
                .contains_key("add_special_tokens")
        );
        assert!(request.unsupported_fields.contains_key("response_format"));

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            let msg = &error_response.1.message;
            assert!(msg.contains("Unsupported parameter"));
            // Verify both fields appear in error message
            assert!(msg.contains("add_special_tokens"));
            assert!(msg.contains("response_format"));
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_error_event() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an error event
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec!["Backend service unavailable".to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "Backend service unavailable");
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_json_error_and_code() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an error event with JSON payload containing error code in comment
        let error_json =
            r#"{"message":"prompt > max_seq_len","type":"Internal Server Error","code":500}"#;
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error_json.to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error with correct status code extracted from JSON
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "prompt > max_seq_len");
            assert_eq!(error_response.1.code, 500);
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_normal_event() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use dynamo_protocols::types::CreateChatCompletionStreamResponse;
        use futures::stream::{self, StreamExt};

        // Create a normal data event
        let normal_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "test-id".to_string(),
                    choices: vec![],
                    created: 0,
                    model: "test-model".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    service_tier: None,
                    usage: None,
                },
                nvext: None,
            }),
            id: Some("msg-1".to_string()),
            event: None,
            comment: None,
            error: None,
        };

        let test_stream = stream::iter(vec![normal_event.clone()]);
        let result = check_for_backend_error(test_stream).await;

        // Should return Ok with the stream
        assert!(result.is_ok());
        let mut returned_stream = result.unwrap();

        // Verify we can read the event back from the stream
        let first = returned_stream.next().await;
        assert!(first.is_some());
        let first_event = first.unwrap();
        assert_eq!(first_event.id, Some("msg-1".to_string()));
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_empty_stream() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream::{self, StreamExt};

        // Create an empty stream
        let test_stream =
            stream::iter::<Vec<Annotated<NvCreateChatCompletionStreamResponse>>>(vec![]);
        let result = check_for_backend_error(test_stream).await;

        // Should return Ok with an empty stream
        assert!(result.is_ok());
        let mut returned_stream = result.unwrap();

        // Verify stream is empty
        let first = returned_stream.next().await;
        assert!(first.is_none());
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_comment_but_no_event_type() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Create an event with comment but no event type and no data (error indicator)
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: None,
            comment: Some(vec!["Connection timeout".to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        // Should return an error based on is_backend_error_event logic
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.message, "Connection timeout");
        }
    }

    #[test]
    fn test_classify_error_for_metrics_validation() {
        // 400 with "Validation:" prefix to validation
        let error_type =
            classify_error_for_metrics(StatusCode::BAD_REQUEST, "Validation: Invalid parameter");
        assert_eq!(error_type, ErrorType::Validation);

        // 400 WITHOUT "Validation:" to internal (fallback)
        let error_type = classify_error_for_metrics(StatusCode::BAD_REQUEST, "Some other error");
        assert_eq!(error_type, ErrorType::Internal);
    }

    #[test]
    fn test_classify_error_for_metrics_status_codes() {
        assert_eq!(
            classify_error_for_metrics(StatusCode::NOT_FOUND, "Model not found"),
            ErrorType::NotFound
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::NOT_IMPLEMENTED, "Feature not supported"),
            ErrorType::NotImplemented
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded"),
            ErrorType::Overload
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::SERVICE_UNAVAILABLE, "Overloaded"),
            ErrorType::Overload
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::INTERNAL_SERVER_ERROR, "Panic"),
            ErrorType::Internal
        );
    }

    #[test]
    fn test_classify_error_for_metrics_client_errors() {
        // Other 4xx errors should be classified as validation
        assert_eq!(
            classify_error_for_metrics(StatusCode::UNAUTHORIZED, "Unauthorized"),
            ErrorType::Validation
        );
        assert_eq!(
            classify_error_for_metrics(StatusCode::FORBIDDEN, "Forbidden"),
            ErrorType::Validation
        );
    }

    #[test]
    fn test_extract_error_type_from_response_validation() {
        let response = ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: "Validation: bad input".to_string(),
        });
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::Validation
        );
    }

    #[test]
    fn test_extract_error_type_from_response_not_found() {
        let response = ErrorMessage::model_not_found();
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::NotFound
        );
    }

    #[test]
    fn test_extract_error_type_from_response_internal() {
        let response = ErrorMessage::internal_server_error("Something went wrong");
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::Internal
        );
    }

    #[test]
    fn test_extract_error_type_from_response_not_implemented() {
        let response = ErrorMessage::not_implemented_error("Feature not available");
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::NotImplemented
        );
    }

    // ── streaming dispatch tests ──────────────────────────────────────

    use std::collections::{HashMap, HashSet};

    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
        ChatCompletionToolType, CreateChatCompletionStreamResponse, FinishReason,
        FunctionCallStream,
    };
    use dynamo_runtime::protocols::annotated::Annotated;

    /// Extract the JSON data payload from an SSE Event's Debug output.
    ///
    /// `axum::response::sse::Event` doesn't expose its fields publicly and doesn't
    /// implement `Display` (the wire format is only produced during response
    /// serialization). The `Debug` representation includes the event name and data
    /// string, so we parse it here.
    ///
    /// WARNING: Coupled to axum's internal Debug format for `Event`. If an axum
    /// upgrade changes the Debug output, these tests will break. Preferred over
    /// spinning up an actual SSE stream for unit test simplicity.
    fn extract_sse_data_json(event: &axum::response::sse::Event) -> serde_json::Value {
        // The Event Debug format is:
        //   Event { buffer: b"event: <name>\ndata: <json>\n", flags: ... }
        // We extract the JSON after "data: " and unescape the byte-string encoding.
        let debug = format!("{:?}", event);

        let data_marker = "data: ";
        let after_data = debug
            .find(data_marker)
            .map(|p| p + data_marker.len())
            .expect("no 'data: ' in Event debug output");

        let rest = &debug[after_data..];
        let json_start = rest.find('{').expect("no JSON object after data:");

        let mut depth = 0i32;
        let mut json_end = 0;
        for (i, b) in rest[json_start..].bytes().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        json_end = json_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        let raw = &rest[json_start..json_end];

        // Unescape byte-string Debug format:
        // \\\\\" -> PLACEHOLDER (nested escaped quotes in JSON string values)
        // \\\"   -> "           (structural quotes)
        // Then restore: PLACEHOLDER -> \"
        let s = raw
            .replace("\\\\\\\"", "\x00NESTED\x00")
            .replace("\\\"", "\"")
            .replace("\x00NESTED\x00", "\\\"");

        // Handle \\xHH byte sequences (non-ASCII in Debug byte-string format)
        let mut result = Vec::new();
        let sbytes = s.as_bytes();
        let mut idx = 0;
        while idx < sbytes.len() {
            if idx + 3 < sbytes.len()
                && sbytes[idx] == b'\\'
                && sbytes[idx + 1] == b'x'
                && let Ok(v) = u8::from_str_radix(
                    std::str::from_utf8(&sbytes[idx + 2..idx + 4]).unwrap_or(""),
                    16,
                )
            {
                result.push(v);
                idx += 4;
                continue;
            }
            result.push(sbytes[idx]);
            idx += 1;
        }

        let final_str = String::from_utf8_lossy(&result);
        serde_json::from_str(&final_str).unwrap_or_else(|e| {
            panic!(
                "failed to parse JSON from Event: {e}\nraw: {raw}\nunescaped: {s}\nfinal: {final_str}"
            )
        })
    }

    /// Assert that an SSE Event has the expected event type name.
    /// Uses "event: <name>\n" pattern to avoid substring false-matches.
    fn assert_event_type(event: &axum::response::sse::Event, expected: &str) {
        let debug = format!("{:?}", event);
        let pattern = format!("event: {expected}\\n");
        assert!(
            debug.contains(&pattern),
            "expected event type '{expected}' not found in: {debug}"
        );
    }

    /// Build a minimal Annotated<Response> with the given choices.
    fn make_stream_response(
        choices: Vec<ChatChoiceStream>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let response = NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices,
                created: 0,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
        };
        Annotated {
            id: Some("test-id".to_string()),
            data: Some(response),
            event: None,
            comment: None,
            error: None,
        }
    }

    fn make_choice_with_reasoning(
        index: u32,
        reasoning: Option<&str>,
        finish: Option<FinishReason>,
    ) -> ChatChoiceStream {
        #[allow(deprecated)]
        ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: reasoning.map(|s| s.to_string()),
            },
            finish_reason: finish,
            stop_reason: None,
            logprobs: None,
        }
    }

    fn make_choice_with_tool_call(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ChatChoiceStream {
        let tool_call = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: id.map(|s| s.to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: name.map(|s| s.to_string()),
                arguments: arguments.map(|s| s.to_string()),
            }),
        };
        #[allow(deprecated)]
        ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: Some(vec![tool_call]),
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        }
    }

    // ── streaming_tool_dispatch_events tests ──

    #[test]
    fn test_tool_dispatch_emits_event_for_complete_tool_call() {
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            Some("call_123"),
            Some("get_weather"),
            Some(r#"{"city":"Paris"}"#),
        )]);

        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert_eq!(events.len(), 1);

        let event = events[0].as_ref().unwrap();
        assert_event_type(event, "tool_call_dispatch");
        let json = extract_sse_data_json(event);
        assert_eq!(json["choice_index"], 0);
        assert_eq!(json["tool_call"]["id"], "call_123");
        assert_eq!(json["tool_call"]["function"]["name"], "get_weather");
        assert_eq!(
            json["tool_call"]["function"]["arguments"],
            r#"{"city":"Paris"}"#
        );
    }

    #[test]
    fn test_tool_dispatch_skips_incomplete_tool_call_no_id() {
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            None, // no id
            Some("get_weather"),
            Some(r#"{"city":"Paris"}"#),
        )]);

        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty(), "should not dispatch without id");
    }

    #[test]
    fn test_tool_dispatch_skips_incomplete_tool_call_no_name() {
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            Some("call_123"),
            None, // no name
            Some(r#"{"city":"Paris"}"#),
        )]);

        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty(), "should not dispatch without name");
    }

    #[test]
    fn test_tool_dispatch_skips_incomplete_tool_call_no_arguments() {
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            Some("call_123"),
            Some("get_weather"),
            None, // no arguments
        )]);

        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty(), "should not dispatch without arguments");
    }

    #[test]
    fn test_tool_dispatch_multiple_tool_calls() {
        let tc1 = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_1".to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Paris"}"#.to_string()),
            }),
        };
        let tc2 = ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call_2".to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_time".to_string()),
                arguments: Some(r#"{"tz":"UTC"}"#.to_string()),
            }),
        };
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: Some(vec![tc1, tc2]),
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert_eq!(events.len(), 2, "should dispatch both tool calls");

        // Verify each dispatched event has the correct tool call data
        let json0 = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json0["tool_call"]["id"], "call_1");
        assert_eq!(json0["tool_call"]["function"]["name"], "get_weather");

        let json1 = extract_sse_data_json(events[1].as_ref().unwrap());
        assert_eq!(json1["tool_call"]["id"], "call_2");
        assert_eq!(json1["tool_call"]["function"]["name"], "get_time");
    }

    #[test]
    fn test_tool_dispatch_no_data() {
        let response: Annotated<NvCreateChatCompletionStreamResponse> = Annotated {
            id: Some("test".to_string()),
            data: None,
            event: None,
            comment: None,
            error: None,
        };
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn test_tool_dispatch_empty_choices() {
        let response = make_stream_response(vec![]);
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn test_tool_dispatch_mixed_complete_and_incomplete() {
        // One complete tool call and one incomplete (missing arguments = streaming delta).
        // Only the complete one should dispatch.
        let complete = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_complete".to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Paris"}"#.to_string()),
            }),
        };
        let incomplete = ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call_partial".to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: Some("search".to_string()),
                arguments: None, // still streaming
            }),
        };
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: Some(vec![complete, incomplete]),
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert_eq!(
            events.len(),
            1,
            "only the complete tool call should dispatch"
        );

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["tool_call"]["id"], "call_complete");
    }

    #[test]
    fn test_tool_dispatch_function_none() {
        // Tool call chunk with function: None — should not dispatch and should not panic.
        let tool_call = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_999".to_string()),
            r#type: Some(ChatCompletionToolType::Function),
            function: None,
        };
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                function_call: None,
                tool_calls: Some(vec![tool_call]),
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty(), "function: None should not dispatch");
    }

    #[test]
    fn test_tool_dispatch_empty_arguments_still_dispatches() {
        // arguments: Some("") is considered complete — intentional.
        // Some backends emit empty-string arguments for parameterless tools.
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            Some("call_empty"),
            Some("no_params_tool"),
            Some(""),
        )]);

        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert_eq!(events.len(), 1, "empty arguments should still dispatch");

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["tool_call"]["id"], "call_empty");
        assert_eq!(json["tool_call"]["function"]["name"], "no_params_tool");
        assert_eq!(json["tool_call"]["function"]["arguments"], "");
    }

    #[test]
    fn test_tool_dispatch_n_greater_than_1_includes_choice_index() {
        // Regression test: with n > 1, each choice should carry its own choice_index
        // so clients can disambiguate which choice the tool call belongs to.
        let choice_0 = make_choice_with_tool_call(
            0,
            Some("call_a"),
            Some("get_weather"),
            Some(r#"{"city":"Paris"}"#),
        );
        let choice_1 = make_choice_with_tool_call(
            1,
            Some("call_b"),
            Some("get_time"),
            Some(r#"{"tz":"UTC"}"#),
        );

        let response = make_stream_response(vec![choice_0, choice_1]);
        let events = streaming_tool_dispatch_events(&response, &mut HashSet::new());
        assert_eq!(events.len(), 2, "should dispatch from both choices");

        let json0 = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json0["choice_index"], 0);
        assert_eq!(json0["tool_call"]["id"], "call_a");

        let json1 = extract_sse_data_json(events[1].as_ref().unwrap());
        assert_eq!(json1["choice_index"], 1);
        assert_eq!(json1["tool_call"]["id"], "call_b");
    }

    #[test]
    fn test_tool_dispatch_dedup_skips_already_dispatched_id() {
        // Simulate a backend that sends the same complete tool call in two consecutive chunks.
        // The HashSet should prevent the second dispatch.
        let response = make_stream_response(vec![make_choice_with_tool_call(
            0,
            Some("call_dup"),
            Some("get_weather"),
            Some(r#"{"city":"Paris"}"#),
        )]);

        let mut dispatched = HashSet::new();

        // First call — should dispatch
        let events = streaming_tool_dispatch_events(&response, &mut dispatched);
        assert_eq!(events.len(), 1);

        // Second call with same response — should be deduped
        let events = streaming_tool_dispatch_events(&response, &mut dispatched);
        assert!(events.is_empty(), "duplicate id should not dispatch twice");
    }

    // ── accumulate_reasoning_dispatch tests ──

    #[test]
    fn test_reasoning_dispatch_accumulates_and_emits_once() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk 1: reasoning token "Let me"
        let r1 = make_stream_response(vec![make_choice_with_reasoning(0, Some("Let me"), None)]);
        let events = accumulate_reasoning_dispatch(&r1, &mut buffers);
        assert!(
            events.is_empty(),
            "should not emit yet — still accumulating"
        );
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Let me"));

        // Chunk 2: reasoning token " think"
        let r2 = make_stream_response(vec![make_choice_with_reasoning(0, Some(" think"), None)]);
        let events = accumulate_reasoning_dispatch(&r2, &mut buffers);
        assert!(
            events.is_empty(),
            "should not emit yet — still accumulating"
        );
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Let me think"));

        // Chunk 3: reasoning ends (None), meaning normal content follows
        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = accumulate_reasoning_dispatch(&r3, &mut buffers);
        assert_eq!(events.len(), 1, "should emit single reasoning_dispatch");

        let event = events[0].as_ref().unwrap();
        assert_event_type(event, "reasoning_dispatch");
        let json = extract_sse_data_json(event);
        assert_eq!(json["reasoning_content"], "Let me think");
        assert_eq!(json["index"], 0);

        // Buffer for choice 0 should be cleared (removed or empty)
        assert!(
            buffers.get(&0).is_none_or(|s| s.is_empty()),
            "buffer should be cleared after emit"
        );
    }

    #[test]
    fn test_reasoning_dispatch_flushes_on_finish_reason() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk 1: reasoning token
        let r1 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some("Thinking..."),
            None,
        )]);
        accumulate_reasoning_dispatch(&r1, &mut buffers);

        // Chunk 2: finish_reason=length while still in reasoning (max_tokens hit)
        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some(" more"),
            Some(FinishReason::Length),
        )]);
        let events = accumulate_reasoning_dispatch(&r2, &mut buffers);
        assert_eq!(events.len(), 1, "should flush on finish_reason");

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Thinking... more");
    }

    #[test]
    fn test_reasoning_dispatch_flushes_on_stop() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk 1: reasoning token
        let r1 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some("Analysis complete"),
            None,
        )]);
        accumulate_reasoning_dispatch(&r1, &mut buffers);

        // Chunk 2: finish_reason=stop while still in reasoning
        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some("."),
            Some(FinishReason::Stop),
        )]);
        let events = accumulate_reasoning_dispatch(&r2, &mut buffers);
        assert_eq!(events.len(), 1, "should flush on FinishReason::Stop");

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Analysis complete.");
    }

    #[test]
    fn test_reasoning_dispatch_no_reasoning_no_event() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk with no reasoning content at all
        let r = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = accumulate_reasoning_dispatch(&r, &mut buffers);
        assert!(events.is_empty(), "no reasoning content = no event");
    }

    #[test]
    fn test_reasoning_dispatch_empty_string_not_accumulated() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk with empty string reasoning (treated as no-reasoning)
        let r = make_stream_response(vec![make_choice_with_reasoning(0, Some(""), None)]);
        let events = accumulate_reasoning_dispatch(&r, &mut buffers);
        assert!(events.is_empty());
        assert!(
            buffers.get(&0).is_none_or(|s| s.is_empty()),
            "empty string should not accumulate"
        );
    }

    #[test]
    fn test_reasoning_dispatch_no_data() {
        let mut buffers: HashMap<u32, String> = HashMap::new();
        let response: Annotated<NvCreateChatCompletionStreamResponse> = Annotated {
            id: Some("test".to_string()),
            data: None,
            event: None,
            comment: None,
            error: None,
        };
        let events = accumulate_reasoning_dispatch(&response, &mut buffers);
        assert!(events.is_empty());
    }

    #[test]
    fn test_reasoning_dispatch_empty_choices() {
        let mut buffers: HashMap<u32, String> = HashMap::new();
        let response = make_stream_response(vec![]);
        let events = accumulate_reasoning_dispatch(&response, &mut buffers);
        assert!(events.is_empty());
    }

    #[test]
    fn test_reasoning_dispatch_multi_choice_independent_buffers() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Both choices emit reasoning in same chunk
        let r1 = make_stream_response(vec![
            make_choice_with_reasoning(0, Some("Thinking A"), None),
            make_choice_with_reasoning(1, Some("Thinking B"), None),
        ]);
        let events = accumulate_reasoning_dispatch(&r1, &mut buffers);
        assert!(events.is_empty(), "both still accumulating");
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Thinking A"));
        assert_eq!(buffers.get(&1).map(|s| s.as_str()), Some("Thinking B"));

        // Choice 0 stops reasoning, choice 1 continues
        let r2 = make_stream_response(vec![
            make_choice_with_reasoning(0, None, None),
            make_choice_with_reasoning(1, Some(" more"), None),
        ]);
        let events = accumulate_reasoning_dispatch(&r2, &mut buffers);
        assert_eq!(events.len(), 1, "only choice 0 should emit");
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Thinking A");
        assert_eq!(json["index"], 0);

        // Choice 1 stops reasoning
        let r3 = make_stream_response(vec![make_choice_with_reasoning(1, None, None)]);
        let events = accumulate_reasoning_dispatch(&r3, &mut buffers);
        assert_eq!(events.len(), 1, "choice 1 should emit");
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Thinking B more");
        assert_eq!(json["index"], 1);
    }

    #[test]
    fn test_reasoning_dispatch_multiple_blocks() {
        // Reasoning -> emit -> more reasoning -> emit again.
        // Verifies that after the buffer is cleared, a new reasoning block
        // accumulates independently.
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // First reasoning block
        let r1 = make_stream_response(vec![make_choice_with_reasoning(0, Some("First"), None)]);
        accumulate_reasoning_dispatch(&r1, &mut buffers);

        let r2 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = accumulate_reasoning_dispatch(&r2, &mut buffers);
        assert_eq!(events.len(), 1);
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "First");

        // Second reasoning block — buffer was cleared, should accumulate fresh
        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, Some("Second"), None)]);
        accumulate_reasoning_dispatch(&r3, &mut buffers);

        let r4 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = accumulate_reasoning_dispatch(&r4, &mut buffers);
        assert_eq!(events.len(), 1);
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(
            json["reasoning_content"], "Second",
            "second emit should only contain second block's content"
        );
    }

    #[test]
    fn test_reasoning_dispatch_unicode() {
        // Verify that CJK characters and emoji survive the JSON roundtrip.
        let mut buffers: HashMap<u32, String> = HashMap::new();

        let r1 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some("让我想想 🤔"),
            None,
        )]);
        accumulate_reasoning_dispatch(&r1, &mut buffers);

        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some(" 分析完成 ✅"),
            None,
        )]);
        accumulate_reasoning_dispatch(&r2, &mut buffers);

        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = accumulate_reasoning_dispatch(&r3, &mut buffers);
        assert_eq!(events.len(), 1);

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "让我想想 🤔 分析完成 ✅");
    }
}
