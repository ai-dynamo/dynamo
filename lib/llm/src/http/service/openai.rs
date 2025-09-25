// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
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
        sse::{KeepAlive, Sse},
    },
    routing::{get, post},
};
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
        Endpoint, EventConverter, process_response_and_observe_metrics,
        process_response_using_event_converter_and_observe_metrics,
    },
    service_v2,
};
use crate::engines::ValidateRequest;
use crate::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
use crate::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse},
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    responses::{NvCreateResponse, NvResponse},
};
use crate::request_template::RequestTemplate;
use crate::types::Annotated;
use dynamo_runtime::logging::get_distributed_tracing_context;
use tracing::Instrument;

pub const DYNAMO_REQUEST_ID_HEADER: &str = "x-dynamo-request-id";

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

// Default axum max body limit without configuring is 2MB: https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html
/// Default body limit in bytes (45MB) to support 500k+ token payloads.
/// Can be configured at compile time using the DYN_FRONTEND_BODY_LIMIT_MB environment variable
fn get_body_limit() -> usize {
    std::env::var("DYN_HTTP_BODY_LIMIT_MB")
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
    pub fn not_implemented_error(msg: &str) -> ErrorResponse {
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

        // Then check for HttpError
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorMessage::from_http_error(http_error),
            Err(err) => ErrorMessage::internal_server_error(&format!("{alt_msg}: {err}")),
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
        let body_bytes = axum::body::to_bytes(body, usize::MAX)
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

/// Get the request ID from a primary source, or next from the headers, or lastly create a new one if not present
// TODO: Similar function exists in lib/llm/src/grpc/service/openai.rs but with different signature and simpler logic
fn get_or_create_request_id(primary: Option<&str>, headers: &HeaderMap) -> String {
    // Try to get request id from trace context
    if let Some(trace_context) = get_distributed_tracing_context()
        && let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id
    {
        return x_dynamo_request_id;
    }

    // Try to get the request ID from the primary source
    if let Some(primary) = primary
        && let Ok(uuid) = uuid::Uuid::parse_str(primary)
    {
        return uuid.to_string();
    }

    // Try to get the request ID header as a string slice
    let request_id_opt = headers
        .get(DYNAMO_REQUEST_ID_HEADER)
        .and_then(|h| h.to_str().ok());

    // Try to parse the request ID as a UUID, or generate a new one if missing/invalid
    let uuid = match request_id_opt {
        Some(request_id) => {
            uuid::Uuid::parse_str(request_id).unwrap_or_else(|_| uuid::Uuid::new_v4())
        }
        None => uuid::Uuid::new_v4(),
    };

    uuid.to_string()
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
    Json(request): Json<NvCreateCompletionRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // create the context for the request
    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

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
    // return a 503 if the service is not ready
    check_ready(&state)?;

    validate_completion_fields_generic(&request)?;

    let request_id = request.id().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = request.inner.model.clone();
    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // update the request to always stream
    let request = request.map(|mut req| {
        req.inner.stream = Some(true);
        req
    });

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_completions_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let parsing_options = state.manager().get_parsing_options(&model);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // Create inflight_guard before calling engine to ensure errors are counted
    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Completions, streaming);

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate completions"))?;

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
        let stream = stream.map(move |response| {
            // Calls observe_response() on each token
            process_response_using_event_converter_and_observe_metrics(
                EventConverter::from(response),
                &mut response_collector,
                &mut http_queue_guard,
            )
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
                ErrorMessage::internal_server_error("Failed to fold completions stream")
            })?;

        inflight_guard.mark_ok();
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

    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id);
    let request_id = request.id().to_string();

    // Embeddings are typically not streamed, so we default to non-streaming
    let streaming = false;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(model);

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_embeddings_engine(model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Embeddings, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate embeddings"))?;

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
            ErrorMessage::internal_server_error("Failed to fold embeddings stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

async fn handler_chat_completions(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    Json(request): Json<NvCreateChatCompletionRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // create the context for the request
    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

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

    // Handle unsupported fields - if Some(resp) is returned by
    // validate_chat_completion_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code. Otherwise, proceeed.
    validate_chat_completion_unsupported_fields(&request)?;

    // Handle required fields like messages shouldn't be empty.
    validate_chat_completion_required_fields(&request)?;

    // Handle Rest of Validation Errors
    validate_chat_completion_fields_generic(&request)?;

    // Apply template values if present
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
    tracing::trace!("Received chat completions request: {:?}", request.content());

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // update the request to always stream
    let request = request.map(|mut req| {
        req.inner.stream = Some(true);
        req
    });

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    // todo - determine the proper error code for when a request model is not present
    let model = request.inner.model.clone();

    // Create HTTP queue guard after template resolution so labels are correct
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let engine = state
        .manager()
        .get_chat_completions_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let parsing_options = state.manager().get_parsing_options(&model);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let annotations = request.annotations();

    // Create inflight_guard before calling engine to ensure errors are counted
    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::ChatCompletions, streaming);

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate completions"))?;

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
        stream_handle.arm(); // allows the system to detect client disconnects and cancel the LLM generation

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream.map(move |response| {
            // Calls observe_response() on each token
            process_response_using_event_converter_and_observe_metrics(
                EventConverter::from(response),
                &mut response_collector,
                &mut http_queue_guard,
            )
        });
        let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream.inspect(move |response| {
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
                        "Failed to fold chat completions stream for: {:?}",
                        e
                    );
                    ErrorMessage::internal_server_error(&format!(
                        "Failed to fold chat completions stream: {}",
                        e
                    ))
                })?;

        inflight_guard.mark_ok();
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

    if inner.parallel_tool_calls == Some(true) {
        return Err(ErrorMessage::not_implemented_error(
            "`parallel_tool_calls: true` is not supported.",
        ));
    }

    if inner.function_call.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            "`function_call` is deprecated. Please migrate to use `tool_choice` instead.",
        ));
    }

    if inner.functions.is_some() {
        return Err(ErrorMessage::not_implemented_error(
            "`functions` is deprecated. Please migrate to use `tools` instead.",
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
            message: "The 'messages' field cannot be empty. At least one message is required."
                .to_string(),
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
            message: e.to_string(),
        })
    })
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
            message: e.to_string(),
        })
    })
}

/// OpenAI Responses Request Handler
///
/// This method will handle the incoming request for the /v1/responses endpoint.
async fn handler_responses(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    Json(request): Json<NvCreateResponse>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // create the context for the request
    let request_id = get_or_create_request_id(request.inner.user.as_deref(), &headers);
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // create the connection handles
    let (mut connection_handle, _stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

    let response = tokio::spawn(responses(state, template, request).in_current_span())
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

#[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.id()))]
async fn responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<NvCreateResponse>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // Create http_queue_guard early - tracks time waiting to be processed
    let model = request.inner.model.clone();
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    // Handle unsupported fields - if Some(resp) is returned by validate_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code. Otherwise, proceeed.
    if let Some(resp) = validate_response_unsupported_fields(&request) {
        return Ok(resp.into_response());
    }

    // Handle non-text (image, audio, file) inputs - if Some(resp) is returned by
    // validate_input_is_text_only, then we are handling something other than Input::Text(_).
    // We will log an error message and early return a 501 NOT_IMPLEMENTED status code.
    // Otherwise, proceeed.
    if let Some(resp) = validate_response_input_is_text_only(&request) {
        return Ok(resp.into_response());
    }

    // Apply template values if present
    if let Some(template) = template {
        if request.inner.model.is_empty() {
            request.inner.model = template.model.clone();
        }
        if request.inner.temperature.unwrap_or(0.0) == 0.0 {
            request.inner.temperature = Some(template.temperature);
        }
        if request.inner.max_output_tokens.unwrap_or(0) == 0 {
            request.inner.max_output_tokens = Some(template.max_completion_tokens);
        }
    }
    tracing::trace!("Received chat completions request: {:?}", request.inner);

    let request_id = request.id().to_string();
    let (request, context) = request.into_parts();

    let mut request: NvCreateChatCompletionRequest = request.try_into().map_err(|e| {
        tracing::error!(
            request_id,
            "Failed to convert NvCreateResponse to NvCreateChatCompletionRequest: {:?}",
            e
        );
        ErrorMessage::not_implemented_error(&format!(
            "Only Input::Text(_) is currently supported: {}",
            e
        ))
    })?;

    let request = context.map(|mut _req| {
        request.inner.stream = Some(false);
        request
    });

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let engine = state
        .manager()
        .get_chat_completions_engine(&model)
        .map_err(|_| ErrorMessage::model_not_found())?;

    let parsing_options = state.manager().get_parsing_options(&model);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    tracing::trace!("Issuing generate call for chat completions");

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate completions"))?;

    // Create inflight_guard now that actual processing has begun
    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Responses, false);

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

    // TODO: handle streaming, currently just unary
    let response =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
            .await
            .map_err(|e| {
                tracing::error!(
                    request_id,
                    "Failed to fold chat completions stream for: {:?}",
                    e
                );
                ErrorMessage::internal_server_error(&format!(
                    "Failed to fold chat completions stream: {}",
                    e
                ))
            })?;

    // Convert NvCreateChatCompletionResponse --> NvResponse
    let response: NvResponse = response.try_into().map_err(|e| {
        tracing::error!(
            request_id,
            "Failed to convert NvCreateChatCompletionResponse to NvResponse: {:?}",
            e
        );
        ErrorMessage::internal_server_error("Failed to convert internal response")
    })?;

    inflight_guard.mark_ok();

    Ok(Json(response).into_response())
}

pub fn validate_response_input_is_text_only(
    request: &NvCreateResponse,
) -> Option<impl IntoResponse> {
    match &request.inner.input {
        dynamo_async_openai::types::responses::Input::Text(_) => None,
        _ => Some(ErrorMessage::not_implemented_error(
            "Only `Input::Text` is supported. Structured, multimedia, or custom input types are not yet implemented.",
        )),
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
            "`background: true` is not supported.",
        ));
    }
    if inner.include.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`include` is not supported.",
        ));
    }
    if inner.instructions.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`instructions` is not supported.",
        ));
    }
    if inner.max_tool_calls.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`max_tool_calls` is not supported.",
        ));
    }
    if inner.metadata.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`metadata` is not supported.",
        ));
    }
    if inner.parallel_tool_calls == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            "`parallel_tool_calls: true` is not supported.",
        ));
    }
    if inner.previous_response_id.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`previous_response_id` is not supported.",
        ));
    }
    if inner.prompt.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`prompt` is not supported.",
        ));
    }
    if inner.reasoning.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`reasoning` is not supported.",
        ));
    }
    if inner.service_tier.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`service_tier` is not supported.",
        ));
    }
    if inner.store == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            "`store: true` is not supported.",
        ));
    }
    if inner.stream == Some(true) {
        return Some(ErrorMessage::not_implemented_error(
            "`stream: true` is not supported.",
        ));
    }
    if inner.text.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`text` is not supported.",
        ));
    }
    if inner.tool_choice.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`tool_choice` is not supported.",
        ));
    }
    if inner.tools.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`tools` is not supported.",
        ));
    }
    if inner.truncation.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`truncation` is not supported.",
        ));
    }
    if inner.user.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            "`user` is not supported.",
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
            object: "object",
            created,                        // Where would this come from? The GGUF?
            owned_by: "nvidia".to_string(), // Get organization from GGUF
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
    object: &'static str, // always "object"
    created: u64,         //  Seconds since epoch
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
        .with_state((state, template));
    (vec![doc], router)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::discovery::ModelManagerError;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::common_ext::CommonExt;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use dynamo_async_openai::types::responses::{
        CreateResponse, Input, InputContent, InputItem, InputMessage, PromptConfig,
        Role as ResponseRole, ServiceTier, TextConfig, TextResponseFormat, ToolChoice,
        ToolChoiceMode, Truncation,
    };
    use dynamo_async_openai::types::{
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
                model: "test-model".into(),
                background: None,
                include: None,
                instructions: None,
                max_output_tokens: None,
                max_tool_calls: None,
                metadata: None,
                parallel_tool_calls: None,
                previous_response_id: None,
                prompt: None,
                reasoning: None,
                service_tier: None,
                store: None,
                stream: None,
                text: None,
                tool_choice: None,
                tools: None,
                truncation: None,
                user: None,
                temperature: None,
                top_logprobs: None,
                top_p: None,
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
    fn test_validate_input_is_text_only_accepts_text() {
        let request = make_base_request();
        let result = validate_response_input_is_text_only(&request);
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_input_is_text_only_rejects_items() {
        let mut request = make_base_request();
        request.inner.input = Input::Items(vec![InputItem::Message(InputMessage {
            kind: Default::default(),
            role: ResponseRole::User,
            content: InputContent::TextInput("structured".into()),
        })]);
        let result = validate_response_input_is_text_only(&request);
        assert!(result.is_some());
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_clean_request() {
        let request = make_base_request();
        let result = validate_response_unsupported_fields(&request);
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_unsupported_fields_detects_flags() {
        #[allow(clippy::type_complexity)]
        let unsupported_cases: Vec<(&str, Box<dyn FnOnce(&mut CreateResponse)>)> = vec![
            ("background", Box::new(|r| r.background = Some(true))),
            (
                "include",
                Box::new(|r| r.include = Some(vec!["file_search_call.results".into()])),
            ),
            (
                "instructions",
                Box::new(|r| r.instructions = Some("System prompt".into())),
            ),
            ("max_tool_calls", Box::new(|r| r.max_tool_calls = Some(3))),
            ("metadata", Box::new(|r| r.metadata = Some(HashMap::new()))),
            (
                "parallel_tool_calls",
                Box::new(|r| r.parallel_tool_calls = Some(true)),
            ),
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
            (
                "reasoning",
                Box::new(|r| r.reasoning = Some(Default::default())),
            ),
            (
                "service_tier",
                Box::new(|r| r.service_tier = Some(ServiceTier::Auto)),
            ),
            ("store", Box::new(|r| r.store = Some(true))),
            ("stream", Box::new(|r| r.stream = Some(true))),
            (
                "text",
                Box::new(|r| {
                    r.text = Some(TextConfig {
                        format: TextResponseFormat::Text,
                    })
                }),
            ),
            (
                "tool_choice",
                Box::new(|r| r.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Required))),
            ),
            ("tools", Box::new(|r| r.tools = Some(vec![]))),
            (
                "truncation",
                Box::new(|r| r.truncation = Some(Truncation::Auto)),
            ),
            ("user", Box::new(|r| r.user = Some("user-id".into()))),
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
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "The 'messages' field cannot be empty. At least one message is required."
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
    // add_special_tokens null or invalid : Not Done
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
        };

        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Frequency penalty must be between -2 and 2, got -3"
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
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Presence penalty must be between -2 and 2, got -3"
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
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Temperature must be between 0 and 2, got -3"
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
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Top_p must be between 0 and 1, got -3"
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
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Repetition penalty must be between 0 and 2, got -3"
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
        };
        let result = validate_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Logprobs must be between 0 and 5, got 6"
            );
        }
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
        };

        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Frequency penalty must be between -2 and 2, got -3"
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
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Presence penalty must be between -2 and 2, got -3"
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
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Temperature must be between 0 and 2, got -3"
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
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Top_p must be between 0 and 1, got -3"
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
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Repetition penalty must be between 0 and 2, got -3"
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
        };
        let result = validate_chat_completion_fields_generic(&request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                "Top_logprobs must be between 0 and 20, got 25"
            );
        }
    }
}
